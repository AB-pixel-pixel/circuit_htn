import os
import sys
import json
import pickle
import glob
import random

# Import generated domain
current_dir = os.path.dirname(os.path.abspath(__file__)) # scripts/htn
root_dir = os.path.dirname(os.path.dirname(current_dir)) # circuit_htn
sys.path.append(root_dir)
sys.path.append(os.path.join(root_dir, 'GTPyhop'))
sys.path.append(os.path.join(root_dir, 'GTPyhop', 'Examples'))

import gtpyhop
import alfred_htn
from env.thor_env import ThorEnv
from scripts.htn.htn_thor_env import HTNThorEnv
from scripts.htn import config

def setup_state(env, agent_name='agent'):
    """
    Construct GTPyhop state from ThorEnv
    """
    state = gtpyhop.State('current_state')
    state.loc = {agent_name: 'start'}
    state.pos = {}
    state.holding = {agent_name: None}
    state.is_clean = {}
    state.is_hot = {}
    state.is_cold = {}
    state.is_on = {}
    state.is_sliced = {}
    
    # Populate objects
    # We use objectId as the key
    for obj in env.last_event.metadata['objects']:
        oid = obj['objectId']
        # Simplified position: just 'in_scene' or assume parent receptacle if we can find it
        # THOR gives 'parentReceptacles'.
        if obj['parentReceptacles']:
            state.pos[oid] = obj['parentReceptacles'][0]
        else:
            state.pos[oid] = 'table' # Fallback default
            
        state.is_clean[oid] = obj['isDirty'] == False
        state.is_hot[oid] = False # properties not always available directly
        state.is_cold[oid] = False
        state.is_on[oid] = obj['isToggled']
        state.is_sliced[oid] = obj['isSliced']
        
    return state

def heuristic_entity_extraction(instruction, scene_objects):
    """
    Extract object IDs from instruction based on simple string matching.
    Returns a list of candidate object IDs.
    """
    candidates = []
    words = instruction.lower().split()
    
    for obj in scene_objects:
        oid = obj['objectId']
        otype = obj['objectType'].lower()
        # If object type name is in instruction
        if otype in instruction.lower():
            candidates.append(oid)
            
    # Remove duplicates and sort
    return list(set(candidates))

def evaluate():
    # Load dataset
    train_dir = os.path.join(config.ALFRED_DATA_PATH, "train")
    traj_files = glob.glob(os.path.join(train_dir, "**", "traj_data.json"), recursive=True)
    
    # Limit for quick eval
    eval_files = traj_files[:5] 
    
    print(f"Evaluating on {len(eval_files)} tasks...")
    
    env = HTNThorEnv()
    
    success_count = 0
    plan_found_count = 0
    
    for i, fpath in enumerate(eval_files):
        print(f"\n--- Task {i}: {os.path.basename(os.path.dirname(fpath))} ---")
        
        with open(fpath, 'r') as f:
            traj = json.load(f)
            
        # Reset Env
        env.reset(traj['scene']['scene_num'])
        env.restore_scene(traj['scene']['object_poses'], traj['scene']['object_toggles'], traj['scene']['dirty_and_empty'])
        
        # Setup State
        state = setup_state(env)
        
        # Get Instruction
        # Load annotation
        ann_path = os.path.join(os.path.dirname(fpath), "turk_annotations.json")
        instruction = "Do task"
        if os.path.exists(ann_path):
            with open(ann_path) as af:
                anns = json.load(af)
                instruction = anns['turk_annotations']['anns'][0]['task_desc']
        print(f"Instruction: {instruction}")
        
        # Extract Entities
        relevant_objects = heuristic_entity_extraction(instruction, env.last_event.metadata['objects'])
        print(f"Extracted Candidates: {len(relevant_objects)}")
        
        # Try to find a plan
        # We don't know which task to call or argument order.
        # Brute force: Try all declared tasks with permutations of candidates?
        # That's expensive.
        # 
        # Strategy: Use the Ground Truth PDDL to identify the intended Task Name (from our PKL mapping)
        # This verifies "Partial Success" (if we knew what to do, could we do it?)
        
        # 1. Get GT action sequence template
        gt_plan = traj['plan']['high_pddl']
        gt_sig = []
        for step in gt_plan:
            act = step['discrete_action']['action']
            if act not in ['NoOp', 'Initialize', 'Term']:
                gt_sig.append(act)
        
        # 2. Find matching task in methods.py (via PKL logic or name matching)
        # Our convert script generated task names like "Method_Goto_Pickup..."
        # Let's construct the expected name
        if not gt_sig:
            print("Empty GT plan, skipping.")
            continue
            
        expected_name = f"Task_Method_{gt_sig[0]}_to_{gt_sig[-1]}"
        print(f"Looking for task: {expected_name}")
        
        # Check if this task exists in gtpyhop
        # gtpyhop stores tasks in current_domain.
        # We can just try calling it.
        
        # Arguments: We need to bind the variables.
        # The params are sorted(union(params)).
        # We need to pick relevant objects from GT args.
        gt_args = set()
        for step in gt_plan:
            for arg in step['discrete_action']['args']:
                # Clean arg (remove |id)
                if isinstance(arg, str):
                    clean = arg.split('|')[0]
                    # Find matching object in scene
                    # This is tricky: we need the specific ID that corresponds to 'Apple' in this scene.
                    # We pick the first one matching type.
                    for obj in env.last_event.metadata['objects']:
                        if obj['objectType'] == clean:
                            gt_args.add(obj['objectId'])
                            break
        
        gt_args_list = sorted(list(gt_args))
        # Add agent
        args = ['agent'] + gt_args_list
        
        # Try planning
        # We need to match the arity of the function.
        # Since we don't know the exact arity expected by the generated function easily here,
        # we might fail.
        # BUT, Pyhop is flexible.
        
        print(f"Trying to plan for {expected_name} with args {args}")
        
        try:
            # We assume the generated module has the function
            # But tasks are strings in declare_task_methods.
            # To plan: gtpyhop.find_plan(state, [(task_name, arg1, arg2...)])
            
            # We need to guess the exact arguments count.
            # Let's try passing the objects we found.
            # If length mismatch, pyhop will fail.
            
            # Simplified: Just try to plan!
            plan = gtpyhop.find_plan(state, [(expected_name, *args[1:])]) # remove agent from plan call usually? 
            # In generate_methods, we defined `def task(state, agent, ...)`
            # So we pass `agent` + others.
            
            if plan:
                print(f"Plan found: {plan}")
                plan_found_count += 1
                
                # Execute
                all_success = True
                for step in plan:
                    # step is (action_name, arg1, ...)
                    ok, msg = env.step(step)
                    if not ok:
                        print(f"Execution failed at {step}: {msg}")
                        all_success = False
                        break
                
                if all_success:
                    print("Task execution successful!")
                    success_count += 1
            else:
                print("No plan found.")
                
        except Exception as e:
            print(f"Planning error: {e}")

    print(f"Summary: Found Plan {plan_found_count}/{len(eval_files)}, Exec Success {success_count}/{len(eval_files)}")

if __name__ == "__main__":
    evaluate()
