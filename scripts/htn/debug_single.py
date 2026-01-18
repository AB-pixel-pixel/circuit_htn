import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(os.path.dirname(current_dir))
sys.path.append(root_dir)
sys.path.append(os.path.join(root_dir, 'GTPyhop'))
sys.path.append(os.path.join(root_dir, 'GTPyhop', 'Examples'))

import gtpyhop
import alfred_htn

def setup_minimal_state(agent_name='agent'):
    state = gtpyhop.State('current_state')
    state.loc = {agent_name: 'start'}
    state.pos = {}
    state.holding = {agent_name: None}
    state.is_clean = {}
    state.is_hot = {}
    state.is_cold = {}
    state.is_on = {}
    state.is_sliced = {}
    return state

state = setup_minimal_state()
task_name = "Task_GotoLocation_PickupObject_GotoLocation_PutObject_PickupObject_GotoLocation_PutObject_1_1_1_2_1_1_2"
args = ['countertop', 'butterknife', 'pan', 'butterknife', 'pan', 'pan', 'countertop', 'pan', 'countertop']

print(f"Planning for {task_name} with args {args}")
plan = gtpyhop.find_plan(state, [(task_name, 'agent', *args)], verbose=3)
print(f"Plan: {plan}")
