# -*- coding: utf-8 -*-
import os
import sys
import pickle
import glob
from collections import defaultdict

# Add path to import existing modules
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

try:
    from lift_alfred_htn import LiftedHTNNode
except ImportError:
    pass

OUTPUT_DIR = "GTPyhop/Examples/alfred_htn"

def generate_actions_py(output_path):
    content = """\"\"\"
ALFRED Primitives (Actions) for GTPyhop.
\"\"\"

import gtpyhop

# --- Helper Functions ---
def find_obj_by_type(state, obj_type):
    # Return first object of type that is reachable/visible if needed
    # For abstract planning, we just return the type if we treat types as instances
    # Or we assume state.objects[obj_type] exists.
    return obj_type

# --- Actions ---

def goto_location(state, agent, location):
    # Preconditions: None (simplified)
    state.loc[agent] = location
    return state

def pickup_object(state, agent, obj):
    # Pre: At location of object, Hand empty
    # For now, heuristic check
    if state.holding[agent] is None:
        state.holding[agent] = obj
        state.pos[obj] = agent
        return state
    return False

def put_object(state, agent, obj, location):
    # Pre: Holding object, At location
    if state.holding[agent] == obj:
        state.holding[agent] = None
        state.pos[obj] = location
        return state
    return False

def clean_object(state, agent, obj):
    if state.holding[agent] == obj:
        state.is_clean[obj] = True
        return state
    return False

def heat_object(state, agent, obj):
    if state.holding[agent] == obj:
        state.is_hot[obj] = True
        return state
    return False

def cool_object(state, agent, obj):
    if state.holding[agent] == obj:
        state.is_cold[obj] = True
        return state
    return False

def toggle_object_on(state, agent, obj):
    state.is_on[obj] = True
    return state

def toggle_object_off(state, agent, obj):
    state.is_on[obj] = False
    return state

def slice_object(state, agent, obj):
    state.is_sliced[obj] = True
    return state
    
def noop(state, agent):
    return state

# Map from HTN action names to functions
# Note: HTN actions might be "PickupObject", function is "pickup_object"
gtpyhop.declare_actions(
    goto_location,
    pickup_object,
    put_object,
    clean_object,
    heat_object,
    cool_object,
    toggle_object_on,
    toggle_object_off,
    slice_object,
    noop
)
"""
    with open(output_path, 'w') as f:
        f.write(content)
    print(f"Generated {output_path}")

def clean_name(name):
    return name.replace("-", "_").replace(" ", "_").replace("?", "")

def generate_methods_py(root_node, output_path):
    lines = []
    lines.append('"""')
    lines.append('Methods generated from Lifted HTN.')
    lines.append('"""')
    lines.append('import gtpyhop')
    lines.append('')
    
    # 1. Identify Tasks and Methods
    # Root -> Choice (Root_Abstract_Tasks) -> [Task_A, Task_B] -> [Impl_1, Impl_2]
    
    tasks = [] # List of (task_name, [method_names])
    methods = [] # List of (method_name, parameters, subtasks)
    
    # Find Root Choice
    root_choice = None
    if root_node.node_type == LiftedHTNNode.SEQUENCE:
        for c in root_node.children:
            if c.node_type == LiftedHTNNode.CHOICE:
                root_choice = c
                break
    elif root_node.node_type == LiftedHTNNode.CHOICE:
        root_choice = root_node
        
    if not root_choice:
        print("Error: Invalid HTN structure for conversion.")
        return

    # Helper to map HTN action name to python function name
    def map_action(name):
        mapping = {
            "GotoLocation": "goto_location",
            "PickupObject": "pickup_object",
            "PutObject": "put_object",
            "CleanObject": "clean_object",
            "HeatObject": "heat_object",
            "CoolObject": "cool_object",
            "ToggleObject": "toggle_object_on", # Map generic Toggle to On
            "ToggleObjectOn": "toggle_object_on",
            "ToggleObjectOff": "toggle_object_off",
            "SliceObject": "slice_object",
            "NoOp": "noop"
        }
        return mapping.get(name, name.lower())

    for task_node in root_choice.children:
        task_name = clean_name(task_node.name)
        
        # Task parameters: intersection/union of method params? 
        # In refine script we stored them in task_node.parameters
        task_params = [clean_name(p) for p in task_node.parameters]
        
        method_names = []
        
        for method_node in task_node.children:
            method_name = clean_name(method_node.name)
            method_names.append(method_name)
            
            # Method parameters
            method_params = [clean_name(p) for p in method_node.parameters]
            
            # Subtasks
            subtasks = []
            for child in method_node.children:
                # Child is PRIMITIVE
                if child.node_type == LiftedHTNNode.PRIMITIVE:
                    act_name = map_action(child.name)
                    # Args: agent + child.parameters
                    # Note: child.parameters are strings like "?obj_0_0" or "apple"
                    # We need to map them to method_params if they are variables
                    args = ['agent'] + [clean_name(p) for p in child.parameters]
                    subtasks.append((act_name, args))
            
            methods.append({
                'name': method_name,
                'params': ['state', 'agent'] + method_params, # Add state, agent
                'subtasks': subtasks
            })
            
        tasks.append({
            'name': task_name,
            'methods': method_names
        })

    # 2. Write Methods
    for m in methods:
        lines.append(f"def {m['name']}({', '.join(m['params'])}):")
        # lines.append(f"    # Subtasks: {m['subtasks']}")
        
        # Precondition check (simplified)
        # We assume if parameters match, it's applicable.
        # Ideally we check state.
        
        task_list_str = []
        for st_name, st_args in m['subtasks']:
            # st_args are variable names (strings). 
            # In the function body, these refer to local variables.
            # We need to output them as code, not strings.
            # e.g. ('goto_location', agent, loc_0_1)
            args_code = ", ".join(st_args)
            task_list_str.append(f"('{st_name}', {args_code})")
            
        return_stmt = f"    return [{', '.join(task_list_str)}]"
        lines.append(return_stmt)
        lines.append("")

    # 3. Declare Tasks
    lines.append("# Declare tasks")
    for t in tasks:
        m_names = ", ".join(t['methods'])
        lines.append(f"gtpyhop.declare_task_methods('{t['name']}', {m_names})")
        
    # 4. Declare Root Task
    # We create a 'perform_task' that can branch into any of the specific tasks
    # This is effectively what the Root Choice does.
    # But in Pyhop, we typically invoke a specific task.
    # To mimic the Choice, we can define 'perform_task' with methods [Task_A_Method_1, Task_B_Method_1, ...]
    # But parameters are different.
    # Easier: Just expose the tasks. The user/evaluator will pick the task based on Goal.
    
    with open(output_path, 'w') as f:
        f.write("\n".join(lines))
    print(f"Generated {output_path}")

def generate_init_py(output_path):
    content = """
import gtpyhop

# Initialize domain
the_domain = gtpyhop.Domain(__package__)

from .actions import *
from .methods import *
"""
    with open(output_path, 'w') as f:
        f.write(content)
    print(f"Generated {output_path}")

def main():
    pkl_file = None
    if len(sys.argv) > 1:
        pkl_file = sys.argv[1]
    
    # Auto-detect latest refined
    if not pkl_file:
        output_dir = "domain_knowledge"
        base_name = "alfred_htn_refined"
        existing_files = glob.glob(os.path.join(output_dir, f"{base_name}_*.pkl"))
        valid_files = []
        for f in existing_files:
             fname = os.path.basename(f)
             if fname.replace(f"{base_name}_", "").replace(".pkl", "").isdigit():
                 valid_files.append(f)
        
        if valid_files:
            valid_files.sort(key=lambda x: int(os.path.basename(x).replace(f"{base_name}_", "").replace(".pkl", "")))
            pkl_file = valid_files[-1]
            print(f"Auto-detected latest refined file: {pkl_file}")

    if not pkl_file or not os.path.exists(pkl_file):
        print(f"File {pkl_file} not found.")
        return

    print(f"Loading Refined HTN from {pkl_file}...")
    with open(pkl_file, 'rb') as f:
        root = pickle.load(f)
        
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        
    generate_actions_py(os.path.join(OUTPUT_DIR, "actions.py"))
    generate_methods_py(root, os.path.join(OUTPUT_DIR, "methods.py"))
    generate_init_py(os.path.join(OUTPUT_DIR, "__init__.py"))

if __name__ == "__main__":
    main()
