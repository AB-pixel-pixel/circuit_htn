import os
import sys
import json
import glob
import re
import random
from collections import defaultdict
import openai
import inspect
import difflib
import time

# Setup paths
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(os.path.dirname(current_dir))
sys.path.append(root_dir)
sys.path.append(os.path.join(root_dir, 'GTPyhop'))
sys.path.append(os.path.join(root_dir, 'GTPyhop', 'Examples'))

try:
    import gtpyhop
    import alfred_htn
    from scripts.htn import config
except ImportError:
    print("Error: Could not import required modules.")
    sys.exit(1)

# --- Configuration ---
USE_LLM = config.ENABLE_LLM
if USE_LLM:
    client = openai.OpenAI(
        api_key=config.OPENAI_API_KEY,
        base_url=config.OPENAI_API_BASE
    )

def setup_state(traj):
    agent_name = 'agent'
    state = gtpyhop.State('current_state')
    state.loc = {agent_name: 'start'}
    state.pos = {}
    state.holding = {agent_name: None}
    state.is_clean = {}
    state.is_hot = {}
    state.is_cold = {}
    state.is_on = {}
    state.is_sliced = {}
    
    # Load objects from scene
    for obj in traj['scene']['object_poses']:
        oid = obj['objectName']
        state.pos[oid] = 'unknown_location' 
        
        state.is_clean[oid] = False
        state.is_hot[oid] = False
        state.is_cold[oid] = False
        state.is_on[oid] = False
        state.is_sliced[oid] = False
        
    return state

def get_available_tasks_info():
    tasks = {}
    for task_name, methods in gtpyhop.current_domain._task_method_dict.items():
        if not methods: continue
        method = methods[0]
        sig = inspect.signature(method)
        params = list(sig.parameters.keys())
        arg_count = len(params) - 2 # minus state, agent
        tasks[task_name] = arg_count
    return tasks

def get_closest_task(task_name, available_tasks):
    if task_name in available_tasks:
        return task_name
    matches = difflib.get_close_matches(task_name, available_tasks.keys(), n=1, cutoff=0.6)
    if matches:
        return matches[0]
    return None

class Planner:
    def plan(self, instruction, state, available_tasks, objects):
        raise NotImplementedError

class HeuristicPlanner(Planner):
    def plan(self, instruction, state, available_tasks, objects):
        print("Heuristic planner cannot handle complex argument mapping without NLP.")
        return None, None

class LLMPlanner(Planner):
    def plan(self, instruction, state, available_tasks, objects):
        obj_list_str = ", ".join(objects[:50]) 
        
        tasks_prompt = ""
        for t, c in list(available_tasks.items())[:40]: 
            tasks_prompt += f"- {t}: expects {c} arguments\n"
        tasks_prompt += "(...and more similar tasks)"

        prompt = f"""
You are an expert planner for the ALFRED household robotics domain.
Your goal is to map a natural language instruction to a specific HTN Task and its arguments.

Domain Information:
- Available Objects: {obj_list_str}
- Available HTN Tasks (with expected argument count):
{tasks_prompt}

Instruction: "{instruction}"

Requirements:
1. Select the most appropriate HTN Task EXACTLY from the provided list. Do not invent new names.
   - The task name describes the sequence of actions.
2. Extract the arguments for the task from the objects list.
   - You MUST provide exactly the number of arguments expected by the task.
   - Arguments should be specific object IDs from the list.
   - If a specific object is not mentioned, pick the most plausible one.
   - Do NOT include 'agent' in the args list.
   - Arguments typically follow the sequence of actions in the task name.

Output Format (JSON only):
{{
  "task": "Task_Name",
  "args": ["arg1", "arg2", ...]
}}
"""
        try:
            response = client.chat.completions.create(
                model=config.OPENAI_MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1
            )
            content = response.choices[0].message.content
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0]
            elif "```" in content:
                content = content.split("```")[1].split("```")[0]
                
            data = json.loads(content.strip())
            
            args = data['args']
            if args and args[0] == 'agent':
                args = args[1:]
                
            return data['task'], args
        except Exception as e:
            print(f"LLM Error: {e}")
            return None, None

def evaluate():
    print(f"=== Evaluating TESTS_SEEN (Mode: {'LLM' if USE_LLM else 'Heuristic'}) ===")
    
    test_dir = os.path.join(config.ALFRED_DATA_PATH, "tests_seen")
    traj_files = glob.glob(os.path.join(test_dir, "**", "traj_data.json"), recursive=True)
    
    # Use fixed seed to select random 30
    random.seed(42)
    # Filter files first to ensure consistent sorting before shuffle
    traj_files.sort()
    
    # Select 30 random files if possible, else all
    sample_size = 30
    if len(traj_files) > sample_size:
        traj_files = random.sample(traj_files, sample_size)
    else:
        print(f"Warning: Only found {len(traj_files)} files, using all.")
    
    print(f"Selected {len(traj_files)} trajectories for evaluation.")
    
    available_tasks = get_available_tasks_info()
    planner = LLMPlanner() if USE_LLM else HeuristicPlanner()
    
    stats = {'total': 0, 'planned': 0, 'failed': 0}
    timing_stats = []
    
    start_total_time = time.time()
    
    for i, fpath in enumerate(traj_files):
        print(f"\n--- Task {i}: {os.path.basename(os.path.dirname(fpath))} ---")
        stats['total'] += 1
        
        with open(fpath, 'r') as f:
            traj = json.load(f)
            
        state = setup_state(traj)
        objects = list(state.pos.keys())
        
        instruction = traj['turk_annotations']['anns'][0]['task_desc']
        print(f"Instruction: {instruction}")
        
        # Timing Plan + HTN
        t0 = time.time()
        
        task_name, args = planner.plan(instruction, state, available_tasks, objects)
        
        t_plan_end = time.time()
        
        plan_result = "FAIL"
        
        if not task_name:
            print("Planner failed to produce a task.")
            stats['failed'] += 1
        else:
            # Fuzzy match task name
            corrected_name = get_closest_task(task_name, available_tasks)
            if corrected_name and corrected_name != task_name:
                print(f"Corrected task name '{task_name}' to '{corrected_name}'")
                task_name = corrected_name
                
            if not task_name or task_name not in available_tasks:
                print(f"Planner failed: Invalid task '{task_name}'")
                stats['failed'] += 1
            else:
                print(f"Planner proposed: {task_name}")
                # print(f"Args ({len(args)}): {args}")
                
                expected_count = available_tasks.get(task_name, 0)
                if len(args) != expected_count:
                    # print(f"Warning: Expected {expected_count} args, got {len(args)}. Padding/Truncating...")
                    if len(args) < expected_count:
                        args += ['None'] * (expected_count - len(args))
                    else:
                        args = args[:expected_count]
                
                try:
                    full_args = ['agent'] + args
                    gtpyhop.verbose = 0
                    plan = gtpyhop.find_plan(state, [(task_name, *full_args)])
                    
                    if plan:
                        print(f"HTN Plan Found: {len(plan)} steps")
                        stats['planned'] += 1
                        plan_result = "SUCCESS"
                    else:
                        print("HTN failed to find a plan (Preconditions or method failure).")
                        stats['failed'] += 1
                        
                except Exception as e:
                    print(f"Execution Error: {e}")
                    stats['failed'] += 1
        
        t_end = time.time()
        duration = t_end - t0
        timing_stats.append(duration)
        print(f"Task Duration: {duration:.4f}s ({plan_result})")

    end_total_time = time.time()
    total_duration = end_total_time - start_total_time
    avg_time = sum(timing_stats) / len(timing_stats) if timing_stats else 0
    
    print("\n=== Summary ===")
    print(f"Total Tasks: {stats['total']}")
    print(f"Success: {stats['planned']}")
    print(f"Failed: {stats['failed']}")
    print(f"Success Rate: {stats['planned']/stats['total']*100:.1f}%")
    print("-" * 20)
    print(f"Total Time: {total_duration:.2f}s")
    print(f"Avg Time per Task: {avg_time:.4f}s")

if __name__ == "__main__":
    evaluate()
