# -*- coding: utf-8 -*-
"""
Extended CircuitHTN
"""
import time
import pickle
import networkx as nx
import numpy as np
from task_graph_to_htn import *
from lfd_trace_to_task_graph import *
from htn import *
from demonstrations_to_graph_v2 import *
from subprocess import check_call
import pydot
import os, sys
import shutil
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
#from simulator.simulator import *
#from simulator.chair_assembly_simulator.chair_simulator import *
from simulator.table_setting_simulator.table_setting_simulator import *
from simulator.drill_assembly_simulator.drill_assembly_simulator import *
from os import listdir
from os.path import isfile, join
from copy import deepcopy
from restructure_graph import *

SALAD_DEMONSTRATION_DIR = "../simulator/50_salad_dataset/ann-ts/activityAnnotationsFilteredOrdered/"

def generate_action_graphs_from_demonstrations(paths):
    task_plans_collection = construct_task_plans(paths)
    lfd_trace = construct_transition_with_probabilities(task_plans_collection)
    root_state = paths[0][0]
    root_action = paths[0][1]
    actionGraphBuilder = StateVectorTaskGraphBuilder()
    action_graph = actionGraphBuilder.convertLFDTraceToTaskGraph(lfd_trace, root_state, root_action)
    return action_graph

def salad_demonstrations_to_htn(demos):
    paths = []
    for demo in demos:
        demo_file_name = '../simulator/50_salad_dataset/ann-ts/activityAnnotationsFilteredOrdered/'
        if demo < 10:
            demo_file_name += '0'
        demo_file_name += str(demo) + '-1-activityAnnotation.txt'

        demo_actions = get_actions_from_demo(demo_file_name)

        state_action_pair, final_state = run_actions(demo_actions)

        # state_action_pair = get_state_action_pair_from_demonstration_file(filename)
        state_action_pair.insert(0, "init_action")
        state_action_pair.insert(0, "init_state")
        state_action_pair.pop()
        state_action_pair.append("term_state")
        state_action_pair.append("term_action")
        paths.append(state_action_pair)
    return generate_action_graphs_from_demonstrations(paths)

def chair_demonstrations_to_htn(demos):
    paths = []
    for demo in demos:
        demo_file_name = ".././simulator/chair_assembly_dataset/" + demo
        actions = get_actions_from_demo(demo_file_name)
#        print(actions)
        state_action_pairs, final_state = run_actions(actions)
        state_action_pairs.insert(0, "init_action")
        state_action_pairs.insert(0, "init_state")
        state_action_pairs.pop()
        state_action_pairs.append("term_state")
        state_action_pairs.append("term_action")
#        print(state_action_pairs)
        paths.append(state_action_pairs)    
    return generate_action_graphs_from_demonstrations(paths)

def drill_demonstrations_to_htn(demos):
    paths = []
#    for i in range(10):
    for demo in demos:
        state_action_pairs, final_state = run_actions_drill(demo)
        state_action_pairs.insert(0, "init_action")
        state_action_pairs.insert(0, "init_state")
        state_action_pairs.pop()
        state_action_pairs.append("term_state")
        state_action_pairs.append("term_action")
        paths.append(state_action_pairs)
    return generate_action_graphs_from_demonstrations(paths)


from tqdm import tqdm

def action_graph_to_htn(action_graph):
    htn_graph = create_init_htn_graph(action_graph)
    node_start = None
    node_terminate = None
    
    for action in htn_graph.nodes:
        action_name = action.get_name()
        if action_name.find('init_action') != -1:
            node_start = action
        elif action.name.find('term_action') != -1:
            node_terminate = action

    start_time = time.time()
    
    initial_nodes = len(list(htn_graph.nodes))
    print(f"开始 HTN 规约，初始节点数: {initial_nodes}")
    pbar = tqdm(total=initial_nodes, desc="Reducing HTN", unit="nodes")
    
    # Track previous count to update pbar correctly
    last_node_count = initial_nodes

    while len(list(htn_graph.nodes)) > 1:
        prev_node_count = len(list(htn_graph.nodes))
        
        # Update progress bar
        reduced = last_node_count - prev_node_count
        if reduced > 0:
            pbar.update(reduced)
        last_node_count = prev_node_count
        
        combined_htns_in_parallel, htn1, htn2 = check_and_combine_htns_in_parallel(htn_graph)
        combined_htns_in_series, htn3, htn4 = check_and_combine_htns_in_series(htn_graph)
        
        if not(combined_htns_in_series or combined_htns_in_parallel):
            restructure_htn_graph(htn_graph, node_start, node_terminate)

        # Removed timeout check for large datasets
        # if time.time() - start_time > 30:
        #    return -1, False
        
        # New Logic: Stop if we are stuck (no reduction happened this round)
        curr_node_count = len(list(htn_graph.nodes))
        if curr_node_count == prev_node_count:
            # Check if restructure actually changed anything?
            # If not, we are stuck.
            # To be safe, if node count didn't decrease and no combination happened, we break.
            pbar.write(f"Reduction stalled at {curr_node_count} nodes. Treating remaining nodes as parallel choices.")
            break
            
    # Final update
    final_reduced = last_node_count - len(list(htn_graph.nodes))
    if final_reduced > 0:
        pbar.update(final_reduced)
    pbar.close()

    nodes = list(htn_graph.nodes)
    if len(nodes) > 1:
        # Create a root ChoiceNode that combines all remaining nodes
        # Assuming init_state and term_state can be generic or derived from first node
        root = ChoiceNode("Root_Choice_Forest", "init_state", "term_state")
        # Add all remaining nodes as children
        # Note: We treat them as equally probable for now, or could weight by their own subtree size
        # Since add_children expects a list of nodes, we pass them all.
        # However, check_and_combine logic might have left some edges? 
        # If we just group them, we might lose top-level edges.
        # But if reduction stalled, it usually means we have disjoint paths or complex cycles.
        # Treating as Choice is the requested behavior.
        
        # We need to assign probabilities. Default to uniform.
        root.add_children(nodes)
        return root, True

    return list(htn_graph.nodes)[0], True

def action_graph_to_htn_naive(action_graph):
    htn_graph = create_init_htn_graph(action_graph)
    for action in htn_graph.nodes:
        action_name = action.get_name()
        if action_name.find('init_action') != -1:
            node_start = action
        elif action.name.find('term_action') != -1:
            node_terminate = action

    while len(list(htn_graph.nodes)) > 1:
        combined_htns_in_parallel, htn1, htn2 = check_and_combine_htns_in_parallel(htn_graph)
        combined_htns_in_series, htn3, htn4 = check_and_combine_htns_in_series(htn_graph)

        if not(combined_htns_in_series or combined_htns_in_parallel):
            return -1, False

    # if len(list(htn_graph.nodes)) == 1:
        # print("HTN Created")

    return list(htn_graph.nodes)[0], True
    
def visualize_with_graphviz_dot(digraph, file_name):
    if shutil.which('dot') is None:
        print("graphviz 'dot' command not found, skipping visualization.")
        return

    digraph.graph['node'] = {'ordering': 'out'}
    G = nx.nx_pydot.to_pydot(digraph)

    nx.drawing.nx_pydot.write_dot(digraph, file_name + ".dot")
    try:
        check_call(['dot', '-Tpng', file_name + '.dot', '-o', file_name + '.png'])
    except Exception as e:
        print(f"graphviz visualization failed: {e}")
    
def hardcoded_example():
    demos = [['pickup_plate', 'place_plate', 'pickup_banana', 'place_banana', 'pickup_orange', 'place_orange', 'pickup_cup', 'place_cup', 'pickup_knife', 'place_knife', 'pickup_spoon', 'place_spoon', 'pickup_bottle', 'pour_water', 'place_bottle', 'pickup_cube', 'place_cube'],
             ['pickup_plate', 'place_plate', 'pickup_knife', 'place_knife', 'pickup_spoon', 'place_spoon', 'pickup_orange',	'place_orange',	'pickup_banana', 'place_banana', 'pickup_cup', 'place_cup', 'pickup_cube', 'place_cube', 'pickup_bottle', 'pour_water', 'place_bottle'],
             ['pickup_plate', 'place_plate', 'pickup_cup', 'place_cup', 'pickup_banana', 'place_banana', 'pickup_orange', 'place_orange', 'pickup_knife', 'place_knife', 'pickup_bottle', 'pour_water', 'place_bottle', 'pickup_cube', 'place_cube', 'pickup_spoon', 'place_spoon'],
             ['pickup_knife', 'place_knife', 'pickup_plate', 'place_plate', 'pickup_spoon', 'place_spoon', 'pickup_banana', 'place_banana', 'pickup_orange', 'place_orange', 'pickup_cup', 'place_cup', 'pickup_bottle', 'pour_water', 'place_bottle', 'pickup_cube', 'place_cube'],
             ['pickup_plate', 'place_plate', 'pickup_cup', 'place_cup', 'pickup_orange', 'place_orange', 'pickup_banana', 'place_banana', 'pickup_knife', 'place_knife', 'pickup_spoon', 'place_spoon', 'pickup_bottle', 'pour_water', 'place_bottle', 'pickup_cube', 'place_cube'],
             ['pickup_plate', 'place_plate', 'pickup_cup', 'place_cup', 'pickup_orange', 'place_orange', 'pickup_banana', 'place_banana', 'pickup_cube', 'place_cube', 'pickup_spoon', 'place_spoon', 'pickup_knife', 'place_knife', 'pickup_bottle', 'pour_water', 'place_bottle'],
             ['pickup_cup', 'place_cup', 'pickup_bottle', 'pour_water', 'place_bottle', 'pickup_cube', 'place_cube', 'pickup_plate', 'place_plate', 'pickup_spoon', 'place_spoon', 'pickup_banana', 'place_banana', 'pickup_orange', 'place_orange', 'pickup_knife', 'place_knife'],
             ['pickup_cup', 'place_cup', 'pickup_cube', 'place_cube', 'pickup_plate', 'place_plate', 'pickup_orange', 'place_orange', 'pickup_banana', 'place_banana', 'pickup_spoon', 'place_spoon', 'pickup_knife', 'place_knife', 'pickup_bottle', 'pour_water', 'place_bottle'],
             ['pickup_cup', 'place_cup', 'pickup_cube', 'place_cube', 'pickup_plate', 'place_plate', 'pickup_orange', 'place_orange', 'pickup_banana', 'place_banana', 'pickup_spoon', 'place_spoon', 'pickup_knife', 'place_knife', 'pickup_bottle', 'pour_water', 'place_bottle'],
             ['pickup_plate', 'place_plate', 'pickup_cup', 'place_cup', 'pickup_cube', 'place_cube', 'pickup_bottle', 'pour_water', 'place_bottle', 'pickup_knife', 'place_knife', 'pickup_spoon', 'place_spoon', 'pickup_orange', 'place_orange', 'pickup_banana', 'place_banana']]    
    
    action_graph = table_demonstrations_to_htn(demos)
    visualize_with_graphviz_dot(action_graph, "action_graph")
    built_htn, result = action_graph_to_htn(action_graph)
    htn_digraph = convertToDiGraph(built_htn)
    visualize_with_graphviz_dot(htn_digraph, 'htn')

def hardcoded_drill_example():
    '''Learn an HTN from the hard-coded example and save the model to the current directory'''
    demos = [
        ['grab_tools1', 'attach_shell1', 'screw1', 'hold_for_robot1', 'grab_tools2', 'attach_shell2', 'screw2', 'hold_for_robot2'],
        ['grab_tools1', 'grab_tools2', 'attach_shell1', 'attach_shell2', 'screw1', 'screw2', 'hold_for_robot1', 'hold_for_robot2']
    ]
    action_graph = drill_demonstrations_to_htn(demos)
    visualize_with_graphviz_dot(action_graph, "action_graph")
    built_htn, result = action_graph_to_htn(action_graph)

    # print('HTN graph:')
    # printHTN(built_htn)
    # print()
    htn_digraph = convertToDiGraph(built_htn)
    visualize_with_graphviz_dot(htn_digraph, 'htn')
    # nx.write_gpickle(htn_digraph, 'htn.pkl', 2)
    circuitHTN = convertToCircuitHTN(built_htn)
    print(circuitHTN.text_output())
    pickle.dump(circuitHTN, open('htn.pkl', 'wb'), 2)


if __name__=="__main__":

    '''
    # Simply uncomment this line and comment out the rest of the main function to run a hardcoded demo example
    hardcoded_drill_example()
    '''

    in_file = open('experiment_groups_table_setting.pkl', 'rb')
    old_demo_list = pickle.load(in_file)
    demo_list = []
    for element1 in old_demo_list:
        for element2 in element1:
            demo_list.append(element2)

    for demo_no in range(len(demo_list)):
        demos = demo_list[demo_no]
        print(demos)
        action_graph = table_demonstrations_to_htn(demos)
        try:
            built_htn, result = action_graph_to_htn_naive(action_graph)
        except:
            print("failure:",demo_list[demo_no])
            continue
        
        if result == False:
            print('reduction failed for demo set:', demo_no)
        else:
            # save model
            circuitHTN = convertToCircuitHTN(built_htn)
            file_name = ''
            for i in range(len(demos)):
                file_name += str(demos[i])
                if i < len(demos) - 1:
                    file_name += '-'
            pickle.dump(circuitHTN, open('htn-' + file_name + '.pkl', 'wb'), 2)

            # sample 100 plans from the model
            action_lists = []
            for i in range(100):
                random_walk = built_htn.random_walk()
                actions = []
                for element in random_walk:
                    action = element.get_name()
                    if 'init_action' in action or 'term_action' in action:
                        continue
                    action = action.split(' ')[-1]
                    action = action.split('-')[0]
                    actions.append(action)
                action_lists.append(actions)
    
            out_file = open(file_name + '.pkl', 'wb')
            pickle.dump(action_lists, out_file)

    '''
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('n', type=int)
    args = parser.parse_args()
    
    in_file = open('experiment_groups.pkl', 'rb')
    old_demo_list = pickle.load(in_file)
    demo_list = []
    for element1 in old_demo_list:
        for element2 in element1:
            demo_list.append(element2)
    
    demos = demo_list[0]
    print(demos)
    action_graph = chair_demonstrations_to_htn(demos)
    print(type(action_graph))

    
    
    in_file = open('demo_list.pkl', 'rb')
    demo_list = pickle.load(in_file)
    
    demos = demo_list[args.n]
    action_graph = salad_demonstrations_to_htn(demos)
    '''
    
    '''
    # visualize_with_graphviz_dot(action_graph, 'action_graph')
    built_htn, result = action_graph_to_htn_naive(action_graph)

    if result == False:
        print('reduction failed for demo set:', args.n)
    else:
        action_lists = []
        for i in range(100):
            random_walk = built_htn.random_walk()
            actions = []
            for element in random_walk:
                action = element.get_name()
                if 'init_action' in action or 'term_action' in action:
                    continue
                action = action.split(' ')[-1]
                action = action.split('-')[0]
                actions.append(action)
            action_lists.append(actions)

        file_name = ''
        for i in range(len(demos)):
            file_name += str(demos[i])
            if i < len(demos) - 1:
                file_name += '-'
        out_file = open(file_name + '.pkl', 'wb')
        pickle.dump(action_lists, out_file)

        print(built_htn.count_choices(), built_htn.count_sequences(), built_htn.count_edges())

        # htn_digraph = convertToDiGraph(built_htn)
        # visualize_with_graphviz_dot(htn_digraph, 'htn')
        '''
    
    
    
