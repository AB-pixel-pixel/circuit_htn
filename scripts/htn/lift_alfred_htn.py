# -*- coding: utf-8 -*-
import os
import sys
import json
import pickle
import glob
from collections import defaultdict, Counter

# Add path to import existing modules
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

try:
    from circuit_htn_node import CircuitHTNNode
    import config
except ImportError:
    # Fallback if config is not found or path issues
    pass

# --- 1. Lifted HTN Data Structures ---

class LiftedHTNNode:
    CHOICE = 0
    SEQUENCE = 1
    PRIMITIVE = 2

    def __init__(self, name, node_type, parameters=None):
        self.name = name
        self.node_type = node_type
        self.parameters = parameters or []  # List of var names, e.g. ["?obj", "?loc"]
        
        # Logical predicates (sets of tuples)
        # e.g. ("At", "?obj", "?loc")
        self.preconditions = set()
        self.effects = set()
        
        self.children = []
        self.probabilities = []  # For CHOICE nodes
        
        # Natural language descriptions associated with this method
        self.nl_templates = [] 
        
        # Bookkeeping for visualization
        self.id = None

    def add_child(self, node, prob=None):
        self.children.append(node)
        if prob is not None:
            self.probabilities.append(prob)

    def to_dict(self):
        """Serialize for JSON/Visualization"""
        return {
            "name": self.name,
            "type": "CHOICE" if self.node_type == self.CHOICE else "SEQUENCE" if self.node_type == self.SEQUENCE else "PRIMITIVE",
            "parameters": self.parameters,
            "preconditions": [list(p) for p in self.preconditions],
            "effects": [list(e) for e in self.effects],
            "children": [c.to_dict() for c in self.children],
            "nl_templates": self.nl_templates
        }

    def __repr__(self):
        return f"{self.name}({','.join(self.parameters)})"

# --- 2. Action & State Abstraction ---

def parse_action_string(action_str):
    """
    Parses 'PickupObject_apple' into ('PickupObject', ['apple'])
    """
    parts = action_str.split('_')
    template = parts[0]
    args = parts[1:]
    return template, args

def get_action_signature(action_str):
    """
    Returns the template name, e.g., 'PickupObject'
    """
    return action_str.split('_')[0]

# --- 3. ALFRED Data Loading ---

def load_alfred_metadata(alfred_root):
    """
    Scans ALFRED train directory to build a map:
    task_id (from directory name) -> {scene_info, annotations}
    """
    metadata = {}
    train_dir = os.path.join(alfred_root, "train")
    
    print(f"Scanning ALFRED metadata in {train_dir}...")
    # Limiting scan for performance in this interactive session, 
    # but in prod this should scan all.
    # We'll use a glob to find traj_data.json files.
    
    traj_files = glob.glob(os.path.join(train_dir, "**", "traj_data.json"), recursive=True)
    
    # Process all files
    for fpath in traj_files:
        dir_path = os.path.dirname(fpath)
        task_id = os.path.basename(dir_path)
        
        # Load scene info
        try:
            with open(fpath, 'r') as f:
                traj_data = json.load(f)
        except:
            continue
            
        # Load annotations (natural language)
        ann_path = os.path.join(dir_path, "turk_annotations.json")
        anns = []
        if os.path.exists(ann_path):
            with open(ann_path, 'r') as f:
                ann_data = json.load(f)
                # ann_data['turk_annotations']['anns'] is a list of dicts
                for item in ann_data.get('turk_annotations', {}).get('anns', []):
                    anns.append(item.get('task_desc', ''))
        
        metadata[task_id] = {
            'scene': traj_data.get('scene', {}),
            'high_pddl': traj_data.get('plan', {}).get('high_pddl', []),
            'anns': anns
        }
        
    print(f"Loaded metadata for {len(metadata)} tasks.")
    return metadata

# --- 4. Lifting Algorithm ---

def lift_htn(grounded_root, alfred_metadata):
    """
    Lifts a grounded HTN into a generalized HTN.
    
    Assumption: The grounded HTN root is a SEQUENCE, which has a child CHOICE (C0).
    The children of C0 are the individual task demonstrations (SEQUENCES).
    """
    
    print("Starting HTN Lifting Process...")
    
    # 1. Identify the Task Choice Node
    # Usually Root -> Sequence -> Choice -> [Tasks...]
    # Or Root -> Choice -> [Tasks...]
    
    task_choice_node = None
    if grounded_root.node_type == CircuitHTNNode.CHOICE:
        task_choice_node = grounded_root
    elif grounded_root.node_type == CircuitHTNNode.SEQUENCE:
        # DFS to find the main choice node
        queue = [grounded_root]
        while queue:
            curr = queue.pop(0)
            if curr.node_type == CircuitHTNNode.CHOICE:
                task_choice_node = curr
                break
            if hasattr(curr, 'children'):
                queue.extend(curr.children)
    
    if not task_choice_node:
        print("Could not find a Choice node to lift.")
        return None

    print(f"Found Task Choice Node with {len(task_choice_node.children)} children.")

    # 2. Group children by Action Structure (Signature)
    # Key: Tuple of action templates (e.g., ("GotoLocation", "PickupObject", ...))
    # Value: List of (child_node, extracted_args_list, task_id_guess)
    groups = defaultdict(list)
    
    for child in task_choice_node.children:
        if child.node_type != CircuitHTNNode.SEQUENCE:
            continue
            
        # Extract action sequence
        # The children of this sequence are primitives (or sequences of primitives)
        # We assume flat structure for simplicity here as per construct_alfred_htn
        
        # Flatten the sequence to get primitives
        primitives = []
        def get_primitives(n):
            if n.node_type == CircuitHTNNode.PRIMITIVE:
                primitives.append(n)
            elif hasattr(n, 'children'):
                for c in n.children:
                    get_primitives(c)
        get_primitives(child)
        
        # Build signature
        signature = []
        args_sequence = []
        
        for p in primitives:
            if not p.action: continue
            template, args = parse_action_string(p.action)
            # Skip NoOp or Terminate if desired, but keep for completeness
            if template in ['NoOp', 'init_action', 'term_action']:
                continue
            signature.append(template)
            args_sequence.append(args)
            
        # Include arg counts in signature to ensure consistent parameterization
        sig_with_counts = []
        for i, templ in enumerate(signature):
            count = len(args_sequence[i])
            sig_with_counts.append((templ, count))
            
        sig_tuple = tuple(sig_with_counts)
        
        # Guess Task ID from node name? 
        # In construct_alfred_htn, node names are like 'S12'. 
        # We don't have direct link back to Task ID unless we stored it.
        # But we can try to match metadata by action sequence length/content if strictly needed.
        # For now, we store the args_sequence which contains the specific objects.
        groups[sig_tuple].append({
            'node': child,
            'args': args_sequence
        })

    print(f"Grouped tasks into {len(groups)} distinct methods (structures).")

    # 3. Create Lifted Nodes
    lifted_root = LiftedHTNNode("Root", LiftedHTNNode.SEQUENCE)
    lifted_choice = LiftedHTNNode("Method_Choice", LiftedHTNNode.CHOICE)
    lifted_root.add_child(lifted_choice)
    
    for sig_complex, instances in groups.items():
        if not sig_complex: continue
        
        # Extract simple signature for logic
        sig = [s[0] for s in sig_complex]
        
        # --- Method Naming & Parameter Extraction ---
        # Strategy: 
        # 1. Look at args across instances. 
        # 2. If args at pos i vary, it's a variable ?var_i.
        # 3. If args at pos i are constant, it's a constant.
        
        # We need to unify arguments.
        # args_sequence is List[List[str]], e.g. [['apple'], ['shelf'], ...]
        
        # Transpose to get list of args for each step across all instances
        # instances[0]['args'] is [ ['apple'], ['shelf'] ] (for one demo)
        # num_steps = len(sig)
        
        # We will collect all values for each argument position
        # Flattening args: (step_idx, arg_idx) -> set of values
        arg_values_map = defaultdict(set)
        
        for inst in instances:
            seq_args = inst['args']
            for step_idx, step_args in enumerate(seq_args):
                for arg_idx, val in enumerate(step_args):
                    arg_values_map[(step_idx, arg_idx)].add(val)
        
        # Define parameters
        parameters = []
        step_arg_to_param = {} # Map (step_idx, arg_idx) -> param_name or const value
        
        for (step_idx, arg_idx), values in arg_values_map.items():
            # Always treat as variable to avoid hardcoding specific instances
            # if len(values) > 1:
            if True:
                # It's a variable
                param_name = f"?obj_{step_idx}_{arg_idx}" # Simple naming
                parameters.append(param_name)
                step_arg_to_param[(step_idx, arg_idx)] = param_name
            # else:
            #     # It's a constant
            #     step_arg_to_param[(step_idx, arg_idx)] = list(values)[0]
                
        # Generate Method Name
        # e.g. "Method_GotoLocation_PickupObject_..."
        # Simplify: "Method_" + first action + "_" + last action
        if sig:
             # Use full signature for precise distinction to avoid merging semantically different tasks (e.g. Heat vs Cool)
             base_name = f"Method_{'_'.join(sig)}"
             # Append structure info (arg counts) to distinguish variants
             counts = "_".join([str(s[1]) for s in sig_complex])
             method_name = f"{base_name}_{counts}"
        else:
             method_name = "Method_Empty"
        
        # Create Lifted Sequence (The Method)
        method_node = LiftedHTNNode(method_name, LiftedHTNNode.SEQUENCE, parameters=parameters)
        
        # Add steps (Primitives) to the Method
        for i, action_template in enumerate(sig):
            # Reconstruct args for this step
            # We don't know how many args this template usually takes, 
            # but we can infer from what we saw in arg_values_map
            step_params = []
            arg_idx = 0
            while (i, arg_idx) in step_arg_to_param:
                step_params.append(step_arg_to_param[(i, arg_idx)])
                arg_idx += 1
            
            step_node = LiftedHTNNode(action_template, LiftedHTNNode.PRIMITIVE, parameters=step_params)
            method_node.add_child(step_node)
            
        # --- Precondition / Effect Learning (Simplified) ---
        # We look at the first action's object for Precondition (e.g. Pickup ?obj -> At ?obj ?loc)
        # We look at the last action for Effect (e.g. Put ?obj ?loc -> At ?obj ?loc)
        
        # This is a heuristic. A real solver would need full state traces.
        # Here we fulfill the user's request by extracting "Initial State" and "End State" concepts.
        
        # Extract variables from the method parameters
        vars_in_method = [p for p in parameters if p.startswith('?')]
        
        if "PickupObject" in sig:
            # Find which param corresponds to PickupObject
            idx = sig.index("PickupObject")
            # Assuming first arg is the object
            obj_param = step_arg_to_param.get((idx, 0))
            if obj_param:
                method_node.preconditions.add(("Visible", "Agent", obj_param))
                method_node.effects.add(("Holding", "Agent", obj_param))

        if "PutObject" in sig:
            idx = sig.index("PutObject")
            obj_param = step_arg_to_param.get((idx, 0))
            loc_param = step_arg_to_param.get((idx, 1))
            if obj_param and loc_param:
                method_node.effects.add(("At", obj_param, loc_param))
                method_node.effects.add(("~Holding", "Agent", obj_param))
        
        # --- Natural Language Mapping ---
        # Heuristic: Match metadata tasks by signature
        # If a task in metadata has the same action template sequence, we assume it's relevant.
        matched_anns = []
        for task_id, meta in alfred_metadata.items():
            meta_pddl = meta.get('high_pddl', [])
            meta_sig = []
            for step in meta_pddl:
                da = step.get('discrete_action', {})
                act = da.get('action', '')
                if act:
                    meta_sig.append(act)
            
            # Simple check: if meta_sig matches our sig (ignoring NoOp/Init/Term which we might have filtered)
            # This is a loose match. For strict match we need to align carefully.
            # Here we just check if the main action sequence is a subsequence or identical.
            
            # For MVP, let's just check length and content roughly
            if len(meta_sig) == len(sig):
                if tuple(meta_sig) == sig:
                    matched_anns.extend(meta.get('anns', []))
        
        # Deduplicate and limit
        unique_anns = list(set(matched_anns))
        method_node.nl_templates = unique_anns[:5] # Keep top 5 examples
        if not method_node.nl_templates:
             method_node.nl_templates.append(f"Execute sequence: {' -> '.join(sig)}")
        
        # Add to choice
        lifted_choice.add_child(method_node, prob=len(instances))

    return lifted_root

# --- 5. Visualization ---

def generate_lifted_html(root, output_file):
    # Convert to VisJS format
    nodes = []
    edges = []
    
    counter = 0
    queue = [(root, -1)] # (node, parent_id)
    
    while queue:
        curr, parent_id = queue.pop(0)
        my_id = counter
        counter += 1
        
        # Label generation
        label = curr.name
        if curr.parameters:
            label += f"\n({', '.join(curr.parameters)})"
        
        # Pre/Eff in tooltip
        title = ""
        if curr.nl_templates:
            title += "NL: " + "\\n".join(curr.nl_templates[:3]) + "\\n\\n"
        if curr.preconditions:
            title += "Pre: " + ", ".join([str(p) for p in curr.preconditions]) + "\\n"
        if curr.effects:
            title += "Eff: " + ", ".join([str(e) for e in curr.effects])
            
        shape = 'box'
        color = '#97C2FC'
        if curr.node_type == LiftedHTNNode.CHOICE:
            shape = 'triangle'
            color = '#FFA807'
            label = "Choice"
        elif curr.node_type == LiftedHTNNode.SEQUENCE:
            if curr.name == "Root":
                shape = 'diamond'
                color = '#FB7E81'
            else:
                shape = 'ellipse' # Method
                color = '#97C2FC'
        elif curr.node_type == LiftedHTNNode.PRIMITIVE:
            shape = 'box'
            color = '#7BE141'
            
        nodes.append({
            'id': my_id,
            'label': label,
            'shape': shape,
            'color': color,
            'title': title
        })
        
        if parent_id != -1:
            edges.append({'from': parent_id, 'to': my_id, 'arrows': 'to'})
            
        for child in curr.children:
            queue.append((child, my_id))
            
    # HTML Template
    html_content = """
<!DOCTYPE html>
<html>
<head>
    <title>Lifted HTN Visualization</title>
    <script type="text/javascript" src="https://unpkg.com/vis-network/standalone/umd/vis-network.min.js"></script>
    <style type="text/css">
        body { margin: 0; padding: 0; overflow: hidden; }
        #mynetwork { width: 100vw; height: 100vh; border: 1px solid lightgray; }
    </style>
</head>
<body>
<div id="mynetwork"></div>
<script type="text/javascript">
    var nodes = new vis.DataSet(__NODES__);
    var edges = new vis.DataSet(__EDGES__);
    var container = document.getElementById('mynetwork');
    var data = { nodes: nodes, edges: edges };
    var options = {
        layout: { hierarchical: { direction: "UD", sortMethod: "directed" } },
        physics: { enabled: false }
    };
    var network = new vis.Network(container, data, options);
</script>
</body>
</html>
"""
    html_content = html_content.replace('__NODES__', json.dumps(nodes))
    html_content = html_content.replace('__EDGES__', json.dumps(edges))
    
    with open(output_file, 'w') as f:
        f.write(html_content)
    print(f"Lifted HTN visualization saved to {output_file}")


def main():
    pkl_file = None
    if len(sys.argv) > 1:
        pkl_file = sys.argv[1]
    
    # Auto-detect latest if not provided
    if not pkl_file:
        output_dir = "domain_knowledge"
        base_name = "alfred_htn"
        existing_files = glob.glob(os.path.join(output_dir, f"{base_name}_*.pkl"))
        # Filter only numeric versions
        valid_files = []
        for f in existing_files:
             fname = os.path.basename(f)
             if fname.replace(f"{base_name}_", "").replace(".pkl", "").isdigit():
                 valid_files.append(f)
        
        if valid_files:
            # Sort by version number
            valid_files.sort(key=lambda x: int(os.path.basename(x).replace(f"{base_name}_", "").replace(".pkl", "")))
            pkl_file = valid_files[-1]
            print(f"Auto-detected latest input file: {pkl_file}")
        else:
             # Fallback
             pkl_file = "domain_knowledge/alfred_htn_带变量.pkl"

    if not os.path.exists(pkl_file):
        print(f"File {pkl_file} not found.")
        return

    # Load Grounded HTN
    print(f"Loading Grounded HTN from {pkl_file}...")
    with open(pkl_file, 'rb') as f:
        grounded_root = pickle.load(f)
        
    # Load Metadata
    try:
        alfred_path = config.ALFRED_DATA_PATH
        if os.path.exists(alfred_path):
            metadata = load_alfred_metadata(alfred_path)
        else:
            print("ALFRED_DATA_PATH not found, skipping metadata loading.")
            metadata = {}
    except Exception as e:
        print(f"Error loading metadata: {e}")
        metadata = {}
    
    # Lift
    lifted_root = lift_htn(grounded_root, metadata)
    
    if lifted_root:
        # Determine output version based on input filename
        # If input is alfred_htn_005.pkl, output should be alfred_htn_lifted_005.pkl
        input_basename = os.path.basename(pkl_file)
        # Check if input follows version pattern
        if input_basename.startswith("alfred_htn_") and input_basename.replace("alfred_htn_", "").replace(".pkl", "").isdigit():
            version = input_basename.replace("alfred_htn_", "").replace(".pkl", "")
            out_pkl = f"domain_knowledge/alfred_htn_lifted_{version}.pkl"
            out_html = f"domain_knowledge/alfred_htn_lifted_{version}.html"
        else:
            # Fallback versioning
            out_base = "alfred_htn_lifted"
            existing_outs = glob.glob(f"domain_knowledge/{out_base}_*.pkl")
            max_v = 0
            for f in existing_outs:
                try:
                     v = int(os.path.basename(f).replace(f"{out_base}_", "").replace(".pkl", ""))
                     if v > max_v: max_v = v
                except: pass
            out_pkl = f"domain_knowledge/{out_base}_{max_v+1:03d}.pkl"
            out_html = f"domain_knowledge/{out_base}_{max_v+1:03d}.html"

        with open(out_pkl, 'wb') as f:
            pickle.dump(lifted_root, f)
        print(f"Saved Lifted HTN to {out_pkl}")
        
        # Visualize
        generate_lifted_html(lifted_root, out_html)
        
    else:
        print("Lifting failed.")

if __name__ == "__main__":
    main()
