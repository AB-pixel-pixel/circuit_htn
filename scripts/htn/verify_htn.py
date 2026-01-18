import os
import sys
import json
import glob

# Setup paths
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(os.path.dirname(current_dir))
sys.path.append(root_dir)
sys.path.append(os.path.join(root_dir, 'gen'))
sys.path.append(os.path.join(root_dir, 'GTPyhop'))
sys.path.append(os.path.join(root_dir, 'GTPyhop', 'Examples'))

import gtpyhop

# Import generated domain
try:
    import alfred_htn
except ImportError:
    print("Error: Could not import alfred_htn. Make sure it is generated in GTPyhop/Examples/alfred_htn")
    sys.exit(1)

from env.thor_env import ThorEnv
from scripts.htn.htn_thor_env import HTNThorEnv
from scripts.htn import config

def setup_state(env, agent_name='agent'):
    state = gtpyhop.State('current_state')
    state.loc = {agent_name: 'start'}
    state.pos = {}
    state.holding = {agent_name: None}
    state.is_clean = {}
    state.is_hot = {}
    state.is_cold = {}
    state.is_on = {}
    state.is_sliced = {}
    
    for obj in env.last_event.metadata['objects']:
        oid = obj['objectId']
        # Use objectId as the identifier
        if obj['parentReceptacles']:
            state.pos[oid] = obj['parentReceptacles'][0]
        else:
            state.pos[oid] = 'table' 
            
        state.is_clean[oid] = obj['isDirty'] == False
        state.is_hot[oid] = False 
        state.is_cold[oid] = False
        state.is_on[oid] = obj['isToggled']
        state.is_sliced[oid] = obj['isSliced']
        
    return state

def clean_arg(arg):
    # Arguments in ALFRED might contain '|' which we might have split or kept
    # In lift_alfred_htn, we split by '_'.
    # If the action string was constructed as "Action_Arg1_Arg2", we need to match that.
    # But here we are extracting from GT JSON.
    # Let's just use the raw string, assuming the planner expects what was in the graph.
    return arg

def verify():
    # Load dataset
    train_dir = os.path.join(config.ALFRED_DATA_PATH, "train")
    # Get a few random files or specific ones
    traj_files = glob.glob(os.path.join(train_dir, "**", "traj_data.json"), recursive=True)
    eval_files = traj_files[:10] # Test on 10 samples
    
    print(f"Verifying on {len(eval_files)} tasks...")
    
    env = HTNThorEnv()
    
    results = {
        'identical': 0,
        'correct': 0,
        'failed_plan': 0,
        'failed_exec': 0,
        'total': 0
    }
    
    for i, fpath in enumerate(eval_files):
        print(f"\n--- Task {i}: {os.path.basename(os.path.dirname(fpath))} ---")
        
        with open(fpath, 'r') as f:
            traj = json.load(f)
            
        # Reset Env
        env.reset(traj['scene']['scene_num'])
        env.restore_scene(traj['scene']['object_poses'], traj['scene']['object_toggles'], traj['scene']['dirty_and_empty'])
        
        state = setup_state(env)
        
        # 1. Get GT Plan
        gt_plan_pddl = traj['plan']['high_pddl']
        gt_actions = []
        gt_flat_args = []
        
        for step in gt_plan_pddl:
            act = step['discrete_action']['action']
            args = step['discrete_action']['args']
            if act in ['NoOp', 'Initialize', 'Term']:
                continue
            gt_actions.append(act)
            gt_flat_args.extend(args)
            
        if not gt_actions:
            print("Empty GT plan.")
            continue
            
        # 2. Identify Task
        # The task name in our HTN includes suffix for arg counts
        counts = []
        for step in gt_plan_pddl:
            act = step['discrete_action']['action']
            if act in ['NoOp', 'Initialize', 'Term']:
                continue
            args = step['discrete_action']['args']
            counts.append(str(len(args)))
            
        counts_str = "_".join(counts)
        base_name = f"Task_{'_'.join(gt_actions)}"
        task_name = f"{base_name}_{counts_str}"
        print(f"Target Task: {task_name}")
        
        # 3. Plan
        # We pass the flattened GT arguments as parameters.
        # Note: Our lifted methods expect parameters corresponding to the action sequence.
        # If we lifted correctly, the parameters should match the GT args sequence.
        
        # In ALFRED JSON, args are sometimes just Object ID, sometimes Type.
        # We need to be careful. The graph construction might have used Type_ID.
        # But let's try passing exactly what is in the JSON args.
        
        try:
            # Clean args (sometimes they are lists?)
            clean_args = []
            for a in gt_flat_args:
                if isinstance(a, list): a = a[0] # Should not happen in high_pddl usually
                clean_args.append(str(a))
                
            print(f"Planning with args: {clean_args}")
            
            # gtpyhop.find_plan expects a list of tasks: [(name, arg1, arg2...)]
            # agent is the first arg usually
            plan = gtpyhop.find_plan(state, [(task_name, 'agent', *clean_args)])
            
            if not plan:
                print("No plan found.")
                results['failed_plan'] += 1
                results['total'] += 1
                continue
                
            # 4. Compare
            # Plan is a list of (action_name, arg1, ...)
            # We need to compare this sequence to GT.
            
            print(f"Plan found: {len(plan)} steps")
            
            # Check for identity
            is_identical = True
            if len(plan) != len(gt_actions):
                is_identical = False
                print(f"Length mismatch: Plan {len(plan)} vs GT {len(gt_actions)}")
            else:
                for idx, (p_act, *p_args) in enumerate(plan):
                    gt_act = gt_actions[idx]
                    # Normalize names: pickup_object -> pickupobject, PickupObject -> pickupobject
                    p_norm = p_act.replace('_', '').lower()
                    g_norm = gt_act.replace('_', '').lower()
                    if p_norm != g_norm:
                        is_identical = False
                        print(f"Mismatch at step {idx}: {p_norm} vs {g_norm}")
                        break
            
            if is_identical:
                print("Plan is IDENTICAL to Ground Truth.")
                results['identical'] += 1
            else:
                print("Plan is DIFFERENT.")
                    
            # Check execution (Correctness)
            all_success = True
            for step in plan:
                ok, msg = env.step(step)
                if not ok:
                    print(f"Execution failed at {step}: {msg}")
                    all_success = False
                    break
            
            if all_success:
                print("Execution Successful (Correct)")
                results['correct'] += 1
            else:
                results['failed_exec'] += 1
                
        except Exception as e:
            print(f"Error: {e}")
            results['failed_plan'] += 1
            
        results['total'] += 1
        
    print("\n--- Verification Summary ---")
    print(f"Total: {results['total']}")
    print(f"Identical/Correct: {results['correct']}")
    print(f"Failed Plan: {results['failed_plan']}")
    print(f"Failed Exec: {results['failed_exec']}")

if __name__ == "__main__":
    verify()
