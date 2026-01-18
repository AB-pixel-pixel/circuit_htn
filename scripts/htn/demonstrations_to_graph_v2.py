import numpy as np
import copy 
import networkx as nx
import matplotlib.pyplot as plt
from lfd_trace_to_task_graph import *
import collections
import json

def split_task_plan(path):
    """
    将扁平的任务计划路径分割为 (Pre-state, Action, Post-state, Next-Action) 的元组列表。
    输入 path 格式: ['', 'a0', 's0', 'a1', 's1', ...]
    """
    return [(path[i], path[i + 1], path[i + 2], path[i+3])
        for i in range(0, len(path) - 3, 2)]

def construct_task_plans(task_plans):
    """
    处理多个演示任务计划。
    """
    task_plans_collection = []
    for individual_task_plan in task_plans:
        task_plans_collection.append(copy.deepcopy(split_task_plan(individual_task_plan)))
    return task_plans_collection

def construct_transition_with_probabilities(task_plans_collection):
    """
    统计转换频率并计算概率。
    transition_probabilities[action] 返回一个字典，键为 (predecessor_state, successor_state, successor_action)。
    """
    # transition_probabilities[action] returns a dict whose keys are transitions in the form of a 3-tuple:
    # (predecessor_state, successor_state, successor_action)
    # transition_probabilities[action][transition] = probability of transition
    transition_probabilities = {}
    for plan in task_plans_collection:
        for action_state_tuple in plan:
            # action_state_tuple: (state_before, action, state_after, next_action)
            # key 是当前动作 action (索引 1)
            if action_state_tuple[1] not in transition_probabilities:
                transition_probabilities[action_state_tuple[1]] = {}
            # transition: (state_before, state_after, next_action)
            transition = (action_state_tuple[0], action_state_tuple[2], action_state_tuple[3])
            if transition in transition_probabilities[action_state_tuple[1]]:
                transition_probabilities[action_state_tuple[1]][transition] += 1
            else:
                transition_probabilities[action_state_tuple[1]][transition] = 1
    
    # 计算概率
    for action in transition_probabilities:
        total_freq = 0.0
        for transition in transition_probabilities[action]:
            total_freq += transition_probabilities[action][transition]
        for transition in transition_probabilities[action]:
            transition_probabilities[action][transition] /= total_freq
    # Hacky solution to get a key for the terminating action:
    # 为终止动作添加一个空条目，防止后续查找出错
    transition_probabilities[task_plans_collection[0][-1][-1]] = {}
    return transition_probabilities

def all_transitions(transitions):
    """生成器：遍历所有转换"""
    for a0 in transitions:
        for(s0, s1, a1) in transitions[a0]:
            yield(s0, a0, s1, a1)


if __name__=="__main__":
    # 示例数据加载（已注释）
    # with open('chair_state_1.json', 'r') as fp:
    #   chair_state = json.load(fp)

    # with open('chair_action_1.json', 'r') as fq:
    #   chair_action = json.load(fq)

    # print(chair_state)
    # print(chair_action)

    ## linear graph plan (线性图计划示例)
    path1 = ['', 'a0', 's0', 'a1', 's1', 'a2', 's2', 'a3', 's3', 'terminate_action']

    ## single split plan (单分支计划示例)
    # path1 = ['', 'a0', 's0', 'a1', 's1', 'a2', 's2', 'a3', 's3', 'a5', 's5', 'terminate_action']
    # path2 = ['', 'a0', 's0', 'a1', 's1', 'a4', 's3', 'a5', 's5', 'terminate_action']

    ## double split plan (双分支计划示例)
    # path1 = ['', 'a0', 's0', 'a1', 's1', 'a2', 's2', 'a4', 's5', 'a6', 's6', 'terminate_action']
    # path2 = ['', 'a0', 's0', 'a1', 's1', 'a3', 's2', 'a4', 's5', 'a6', 's6', 'terminate_action']
    # path3 = ['', 'a0', 's0', 'a5', 's5', 'a6', 's6', 'terminate_action']

    ## triple split plan (三分支计划示例)
    # path1 = ['', 'a0', 's0', 'a1', 's4', 'a4', 's5', 'terminate_action']
    # path2 = ['', 'a0', 's0', 'a2', 's4', 'a4', 's5', 'terminate_action']
    # path3 = ['', 'a0', 's0', 'a3', 's4', 'a4', 's5','terminate_action']

    ## chair plan (椅子组装计划示例)
    # path1 = ['', 'a0', 's0', 'a1', 's1', 'a2', 's2', 'a3', 's3', 'a4', 's4', 'a5', 's9', 'a6', 's10', 'terminate_action']
    # path2 = ['', 'a0', 's0', 'a3', 's5', 'a4', 's6', 'a1', 's7', 'a2', 's4', 'a5', 's9', 'a6', 's10', 'terminate_action']

    ## random complicated plan (随机复杂计划示例)
    # path1 = ['', 'a0', 's0', 'a1', 's1', 'a2', 's3', 'a3', 's6', 'a4', 's9', 'a5', 's10', 'a6']
    # path2 = ['', 'a0', 's0', 'a1', 's1', 'a2', 's3', 'a4', 's7', 'a3', 's9', 'a5', 's10', 'a6']
    # path3 = ['', 'a0', 's0', 'a2', 's2', 'a3', 's4', 'a4', 's8', 'a1', 's9', 'a5', 's10', 'a6']
    # path4 = ['', 'a0', 's0', 'a2', 's2', 'a4', 's5', 'a3', 's8', 'a1', 's9', 'a5', 's10', 'a6']

    task_plans = []
    task_plans.append(path1)
    # task_plans.append(path2)
    # task_plans.append(path3)
    # task_plans.append(path4)
    
    # 1. 构建计划集合
    task_plans_collection = construct_task_plans(task_plans)
    # 2. 计算转换概率
    transitions = construct_transition_with_probabilities(task_plans_collection)
    # print(transitions)

    # 3. 准备 LfD 轨迹数据
    lfd_trace = {}
    for element in transitions:
        lfd_trace[element] = list(transitions[element])
    #lfd_trace['terminate_action'] = []

    # print(lfd_trace)
    
    # 4. 构建任务图
    # 注意：这里需要根据实际的起始状态和动作进行调整，这里假设从空状态和 'a0' 开始
    # StateVectorTaskGraphBuilder 在 lfd_trace_to_task_graph.py 中定义
    taskGraphBuilder = StateVectorTaskGraphBuilder()
    
    # 这里的 convertLFDTraceToTaskGraph 调用似乎缺少参数，原代码可能与类定义不匹配
    # 原类定义: convertLFDTraceToTaskGraph(self, lfd_trace, root_state, root_action)
    # 这里修正调用方式，或者假设 taskGraphBuilder 已经被修改以适应无参数调用（但这不太可能）
    # 根据 path1 = ['', 'a0', ...], root_state='', root_action='a0'
    
    # graph = taskGraphBuilder.convertLFDTraceToTaskGraph() # 原代码这行会报错，因为缺少参数
    graph = taskGraphBuilder.convertLFDTraceToTaskGraph(lfd_trace, '', 'a0') 
    
    print(graph)
    ## Plot the Graph with Edge Attributes (绘制带边属性的图)
    nx.draw_networkx(graph, with_labels=True)
    plt.show()
    print('compiled graph')
	
	
					
