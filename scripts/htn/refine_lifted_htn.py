# -*- coding: utf-8 -*-
import os
import sys
import json
import pickle
import glob
from collections import defaultdict

# Add path to import existing modules
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

try:
    from lift_alfred_htn import LiftedHTNNode, generate_lifted_html
except ImportError:
    pass

def refine_htn(lifted_root):
    """
    Refines the Lifted HTN by grouping methods with the same name.
    
    Structure Transformation:
    Before:
      Root -> Choice -> [Method_A_1, Method_A_2, Method_B_1]
    
    After:
      Root -> Choice -> [Abstract_Task_A, Abstract_Task_B]
      Abstract_Task_A -> Choice -> [Method_A_1, Method_A_2]
      Abstract_Task_B -> Choice -> [Method_B_1]
      
    This allows "branching" for the same high-level task name based on different implementations (variables/structure).
    """
    
    print("Starting HTN Refinement Process...")
    
    # 1. Find the main Method Choice Node
    method_choice_node = None
    if lifted_root.node_type == LiftedHTNNode.SEQUENCE:
        for child in lifted_root.children:
            if child.node_type == LiftedHTNNode.CHOICE:
                method_choice_node = child
                break
    elif lifted_root.node_type == LiftedHTNNode.CHOICE:
        method_choice_node = lifted_root
        
    if not method_choice_node:
        print("Could not find Method Choice node.")
        return None
        
    # 2. Group children by Name
    # Key: Method Name (e.g., "Method_GotoLocation_to_PutObject")
    # Value: List of Method Nodes
    groups = defaultdict(list)
    
    for child in method_choice_node.children:
        groups[child.name].append(child)
        
    print(f"Found {len(groups)} unique abstract tasks from {len(method_choice_node.children)} methods.")
    
    # 3. Create new structure
    new_choice_node = LiftedHTNNode("Refined_Method_Choice", LiftedHTNNode.CHOICE)
    
    # We need to rebuild the root sequence to point to this new choice
    # Or just replace children of existing choice if we want to keep root intact
    # Let's replace children of method_choice_node to be minimally invasive?
    # No, we want intermediate Abstract Nodes.
    
    new_children = []
    
    for task_name, methods in groups.items():
        # Clean up task name? remove "Method_" prefix to make it look like a Task?
        # e.g. "Task_GotoLocation_to_PutObject"
        abstract_name = task_name.replace("Method_", "Task_")
        
        # Merge Parameters: Union of all parameters from all methods
        # This is tricky. For now, we collect all unique param names.
        # In a real system, we'd need signature alignment.
        all_params = set()
        for m in methods:
            for p in m.parameters:
                all_params.add(p)
        sorted_params = sorted(list(all_params))
        
        # Create Abstract Task Node (Choice)
        # This node represents the high-level task. 
        # Its children are the different implementations (methods).
        abstract_node = LiftedHTNNode(abstract_name, LiftedHTNNode.CHOICE, parameters=sorted_params)
        
        # Add Preconditions/Effects to Abstract Node?
        # Ideally, the abstract task has the Intersection of Pre/Eff of its methods.
        # (Must be valid for all implementations)
        # For MVP, let's skip rigorous P/E intersection.
        
        # Add methods as children
        for m in methods:
            # We might want to rename the method to distinguish variants?
            # e.g. "Impl_1", "Impl_2" or keep original name?
            # Keeping original name is fine, but visually confusing if they are identical.
            # Let's append an ID or Variant info.
            m.name = m.name.replace("Method_", "Impl_")
            abstract_node.add_child(m)
            
        new_children.append(abstract_node)
        
    # Update the original choice node
    method_choice_node.children = new_children
    method_choice_node.name = "Root_Abstract_Tasks"
    
    return lifted_root

def main():
    pkl_file = None
    if len(sys.argv) > 1:
        pkl_file = sys.argv[1]
    
    # Auto-detect latest lifted file
    if not pkl_file:
        output_dir = "domain_knowledge"
        base_name = "alfred_htn_lifted"
        existing_files = glob.glob(os.path.join(output_dir, f"{base_name}_*.pkl"))
        valid_files = []
        for f in existing_files:
             fname = os.path.basename(f)
             # alfred_htn_lifted_001.pkl
             if fname.replace(f"{base_name}_", "").replace(".pkl", "").isdigit():
                 valid_files.append(f)
        
        if valid_files:
            valid_files.sort(key=lambda x: int(os.path.basename(x).replace(f"{base_name}_", "").replace(".pkl", "")))
            pkl_file = valid_files[-1]
            print(f"Auto-detected latest lifted file: {pkl_file}")

    if not pkl_file or not os.path.exists(pkl_file):
        print(f"File {pkl_file} not found.")
        return

    # Load Lifted HTN
    print(f"Loading Lifted HTN from {pkl_file}...")
    with open(pkl_file, 'rb') as f:
        lifted_root = pickle.load(f)
        
    # Refine
    refined_root = refine_htn(lifted_root)
    
    if refined_root:
        # Output filename: alfred_htn_refined_XXX.pkl
        # Base version on input file
        input_basename = os.path.basename(pkl_file)
        version = "000"
        if input_basename.startswith("alfred_htn_lifted_"):
            version = input_basename.replace("alfred_htn_lifted_", "").replace(".pkl", "")
            
        out_pkl = f"domain_knowledge/alfred_htn_refined_{version}.pkl"
        out_html = f"domain_knowledge/alfred_htn_refined_{version}.html"
        
        with open(out_pkl, 'wb') as f:
            pickle.dump(refined_root, f)
        print(f"Saved Refined HTN to {out_pkl}")
        
        # Visualize
        # We need to make sure generate_lifted_html handles the deeper nesting
        # The existing visualizer relies on recursion, so it should handle it fine.
        generate_lifted_html(refined_root, out_html)
        
    else:
        print("Refinement failed.")

if __name__ == "__main__":
    main()
