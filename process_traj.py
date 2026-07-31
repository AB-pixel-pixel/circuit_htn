import sys
import os
import json
import copy
import numpy as np
import traceback
import multiprocessing
import glob
import argparse
from tqdm import tqdm

# Add paths
sys.path.append(os.getcwd())
sys.path.append(os.path.join(os.getcwd(), 'gen'))

# Override HOME to ensure ai2thor uses local directory
os.environ['HOME'] = os.getcwd()

# Set AI2THOR_HOME to avoid permission issues
os.environ['AI2THOR_HOME'] = os.path.join(os.getcwd(), 'ai2thor_cache')
if not os.path.exists(os.environ['AI2THOR_HOME']):
    os.makedirs(os.environ['AI2THOR_HOME'])

import gen.constants
sys.modules['constants'] = gen.constants
import constants

# Set LOG_FILE
constants.LOG_FILE = 'logs_gen_process'
constants.RECORD_VIDEO_IMAGES = True # Enable video recording
if not os.path.exists(constants.LOG_FILE):
    os.makedirs(constants.LOG_FILE)
    
planner_problems_dir = os.path.join(constants.LOG_FILE, 'planner', 'generated_problems')
if not os.path.exists(planner_problems_dir):
    os.makedirs(planner_problems_dir)

from gen.game_states.task_game_state_full_knowledge import TaskGameStateFullKnowledge
from env.thor_env import ThorEnv
from gen.utils import game_util

def find_closest_object_id(old_id, metadata):
    if not old_id:
        return old_id
    parts = old_id.split('|')
    if len(parts) < 4:
        return old_id
    
    obj_type = parts[0]
    try:
        x = float(parts[1])
        y = float(parts[2])
        z = float(parts[3])
    except ValueError:
        return old_id
    
    candidates = [o for o in metadata['objects'] if o['objectType'] == obj_type]
    if not candidates:
        # print(f"Warning: No candidates for {obj_type}")
        return old_id
        
    best_dist = float('inf')
    best_id = None
    
    for cand in candidates:
        pos = cand['position']
        dist = (pos['x'] - x)**2 + (pos['y'] - y)**2 + (pos['z'] - z)**2
        if dist < best_dist:
            best_dist = dist
            best_id = cand['objectId']
            
    return best_id

def process_trajectory(traj_path, output_dir):
    # print(f"Processing trajectory: {traj_path}")
    with open(traj_path, 'r') as f:
        traj_data = json.load(f)

    # Setup output paths
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # We need to mimic the directory structure for pddl states
    constants.save_path = os.path.join(output_dir, 'raw_images')
    if not os.path.exists(constants.save_path):
        os.makedirs(constants.save_path)
    
    # Initialize Env
    # 添加 try-finally 块，确保在发生异常时正确关闭环境，释放内存资源
    # Add try-finally block to ensure the environment is properly closed and memory resources are released when an exception occurs
    env = ThorEnv()
    try:
        game_state = TaskGameStateFullKnowledge(env)

        # Setup scene
        scene_num = traj_data['scene']['scene_num']
        object_poses = traj_data['scene']['object_poses']
        
        # Adjust object poses for AI2-THOR 5.0 (lift objects slightly to avoid collision/falling)
        for obj_pose in object_poses:
            obj_pose['position']['y'] += 0.05
            
        object_toggles = traj_data['scene']['object_toggles']
        dirty_and_empty = traj_data['scene']['dirty_and_empty']
        random_seed = traj_data['scene']['random_seed']
        
        # Setup constants.data_dict
        constants.data_dict = copy.deepcopy(traj_data)
        # print("constants.data_dict keys:", constants.data_dict.keys())
        constants.data_dict['pddl_state'] = [] 

        # Reset env
        # print(f"Resetting to scene {scene_num}")
        env.reset(f"FloorPlan{scene_num}")
        
        # restore_scene in original code doesn't take seed
        env.restore_scene(object_poses, object_toggles, dirty_and_empty)
        
        # Init action
        init_action = traj_data['scene']['init_action']
        if 'rotateOnTeleport' in init_action:
            del init_action['rotateOnTeleport']
        if isinstance(init_action.get('rotation'), (int, float)):
             init_action['rotation'] = dict(x=0, y=init_action['rotation'], z=0)
        init_action['standing'] = True
        init_action['forceAction'] = True
        env.step(init_action)

        # Manually update game_state
        game_state.scene_num = scene_num
        game_state.scene_name = f"FloorPlan{scene_num}"
        game_state.event = env.last_event
        game_state.process_frame() 
        
        # Load openable points
        points_source = f"gen/layouts/{game_state.scene_name}-openable.json"
        if os.path.exists(points_source):
            with open(points_source, 'r') as f:
                game_state.openable_object_to_point = json.load(f)
        else:
            # print(f"Warning: {points_source} not found")
            game_state.openable_object_to_point = {}

        # Set pddl_params
        pddl_params = traj_data['pddl_params']
        
        # Map target names to indices
        def get_idx(name):
            return constants.OBJECTS.index(name) if name and name in constants.OBJECTS else None

        game_state.object_target = get_idx(pddl_params['object_target'])
        game_state.parent_target = get_idx(pddl_params['parent_target'])
        game_state.toggle_target = get_idx(pddl_params['toggle_target'])
        game_state.mrecep_target = get_idx(pddl_params['mrecep_target'])
        game_state.task_target = (game_state.object_target, game_state.parent_target,
                                  game_state.toggle_target, game_state.mrecep_target)

        # Set goals
        constants.pddl_goal_type = traj_data['task_type']

        # Initialize graph
        from gen.graph import graph_obj
        game_state.gt_graph = graph_obj.Graph(use_gt=True, construct_graph=True, scene_id=scene_num)
        
        # 视频保存需要 VideoSaver
        from gen.utils import video_util
        video_saver = video_util.VideoSaver(frame_rate=10) # default frame rate

        game_state.bounds = np.array([game_state.gt_graph.xMin, game_state.gt_graph.yMin,
                                    game_state.gt_graph.xMax - game_state.gt_graph.xMin + 1,
                                    game_state.gt_graph.yMax - game_state.gt_graph.yMin + 1])
        game_state.agent_height = env.last_event.metadata['agent']['position']['y']
        game_state.camera_height = game_state.agent_height + constants.CAMERA_HEIGHT_OFFSET
        game_state.pose = game_util.get_pose(game_state.event)

        # print("Initializing receptacle points...")
        game_state.update_receptacle_nearest_points()
        
        # Reset pose to init_action after rotations
        env.step(init_action)
        game_state.process_frame()

        high_pddl = traj_data['plan']['high_pddl']
        new_high_pddl = []

        # print(f"Processing {len(high_pddl)} actions...")
        
        for i, action_info in enumerate(high_pddl):
            game_state.problem_id = f"{traj_data['task_id']}_{i}"
            
            pddl_str = game_state.state_to_pddl()
            pddl_file = constants.data_dict['pddl_state'][-1]
            
            new_item = copy.deepcopy(action_info)
            new_item['pddl_state_file'] = pddl_file
            new_high_pddl.append(new_item)
            
            planner_action = action_info['planner_action']
            
            if planner_action['action'] == 'End':
                break
                
            # print(f"Step {i}: {planner_action['action']}")
            
            if planner_action['action'] == 'GotoLocation':
                action_to_run = game_state.get_teleport_action(planner_action)
            else:
                action_to_run = copy.deepcopy(planner_action)
            
            # Cleanup for AI2-THOR 5.0
            keys_to_remove = ['coordinateObjectId', 'coordinateReceptacleObjectId', 'forceVisible', 'rotateOnTeleport']
            for k in list(action_to_run.keys()):
                if k in keys_to_remove:
                    del action_to_run[k]
            
            # Map object IDs
            if 'objectId' in action_to_run:
                action_to_run['objectId'] = find_closest_object_id(action_to_run['objectId'], game_state.event.metadata)
            if 'receptacleObjectId' in action_to_run:
                action_to_run['receptacleObjectId'] = find_closest_object_id(action_to_run['receptacleObjectId'], game_state.event.metadata)

            # Handle ToggleObject
            if action_to_run['action'] == 'ToggleObject':
                obj_id = action_to_run['objectId']
                obj = game_util.get_object(obj_id, game_state.event.metadata)
                if obj and obj['isToggled']:
                     action_to_run['action'] = 'ToggleObjectOff'
                else:
                     action_to_run['action'] = 'ToggleObjectOn'
                action_to_run['forceAction'] = True
                
            # Handle Pickup/Put
            if action_to_run['action'] in ['PickupObject', 'PutObject', 'OpenObject', 'CloseObject', 'SliceObject']:
                action_to_run['forceAction'] = True
                    
            try:
                game_state.step(action_to_run)
            except Exception as e:
                # print(f"Error executing step {i}: {e}")
                # traceback.print_exc()
                pass

        constants.data_dict['plan']['high_pddl'] = new_high_pddl
        
        new_traj_path = os.path.join(output_dir, 'traj_data_with_pddl.json')
        with open(new_traj_path, 'w') as f:
            json.dump(constants.data_dict, f, indent=4)
            
        # print(f"Saved processed trajectory to {new_traj_path}")
        
        # Save video
        images_path = os.path.join(constants.save_path, '*.png')
        video_path = os.path.join(output_dir, 'video.mp4')
        video_saver.save(images_path, video_path)
        
    finally:
        # 无论发生什么情况（包括异常），都必须调用 stop 来清理环境
        # Must call stop to clean up the environment regardless of what happens (including exceptions)
        env.stop() # Stop unity

def worker(args):
    traj_path, output_dir = args
    try:
        process_trajectory(traj_path, output_dir)
    except Exception as e:
        print(f"Failed to process {traj_path}: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', default='/Users/ab/Code/construct_domain_knowledge/circuit_htn/alfred_data/train')
    parser.add_argument('--output_dir', default='/Users/ab/Code/construct_domain_knowledge/circuit_htn/alfred_data/train_full')
    parser.add_argument('--processes', type=int, default=6)
    args = parser.parse_args()

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    print(f"Searching for trajectories in {args.input_dir}...")
    traj_files = glob.glob(os.path.join(args.input_dir, "**", "traj_data.json"), recursive=True)
    print(f"Found {len(traj_files)} trajectories.")

    # Prepare tasks
    tasks = []
    for traj_path in traj_files:
        # Rel path: e.g. look_at_obj.../trial.../traj_data.json
        rel_path = os.path.relpath(os.path.dirname(traj_path), args.input_dir)
        out_path = os.path.join(args.output_dir, rel_path)
        tasks.append((traj_path, out_path))

    # Run
    # Use spawn for safety with ai2thor
    multiprocessing.set_start_method('spawn', force=True)
    
    with multiprocessing.Pool(processes=args.processes) as pool:
        list(tqdm(pool.imap_unordered(worker, tasks), total=len(tasks)))
