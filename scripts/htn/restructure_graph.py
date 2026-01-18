# -*- coding: utf-8 -*-
"""
Resolves the "bad graph" scenario
解决“坏图”场景（即无法直接通过简单的串行/并行合并规则归约的图结构）。
这通常涉及到图的重构和展开。
"""
import numpy as np
from copy import deepcopy
import htn
import networkx as nx
from htn import *
#from test_code import *

def bfs_search(G, node_start):
    """
    广度优先搜索 (BFS)，计算节点的层级（level）和父节点（parent）。
    用于确定节点在图中的深度和拓扑关系。
    """
    level = {node_start:0}
    parent = {node_start:[None]}
    i=1
    frontier = [node_start]
    
    while frontier:
        nextt = []
        for u in frontier:
            for v in G.neighbors(u):
                if (v not in level) or (parent[v]!=u):
                    level[v] = i
                    try:
                        parent[v].extend([u])
                    except KeyError:
                        parent[v] = [u]
                    nextt.append(v)
        frontier = nextt
        i += 1
    return level, parent

def reach_terminate_node(G, node_start, node_terminate):
    """
    检查从 node_start 是否可以到达 node_terminate。
    用于验证图的连通性。
    """
    flag = 0
    level = {node_start:0}
    parent = {node_start:None}
    i = 1
    frontier = [node_start]
    while frontier:
        nextt = []
        for u in frontier:
            for v in G.neighbors(u):
                if v not in level:
                    level[v] = i
                    parent[v] = u
                    nextt.append(v)
                    if v == node_terminate:
                        flag = 1
        frontier = nextt
        i += 1
    
    if flag == 1:
        return True
    else:
        return False

def least_common_successor(G, node1, node2, node_start, node_terminate):
    """
    寻找 node1 和 node2 的“最小公共后继”（Least Common Successor, LCS）。
    LCS 是两个分支路径汇合的地方。
    算法通过随机游走寻找候选，并验证在移除该节点后是否会阻断到达终止节点的路径（即该节点是否是必经之路）。
    """
    node1_set = set()
    node2_set = set()
    
    # 从 node1 随机游走收集后继
    node = node1
    while(sum(1 for _ in G.successors(node))!=0):
        successors_node_1 = [s for s in G.successors(node)]
        node_id = np.random.randint(0,len(successors_node_1))
        node = successors_node_1[node_id]
        node1_set.add(node)
    
    # 从 node2 随机游走收集后继
    node = node2
    while(sum(1 for _ in G.successors(node))!=0):
        successors_node_2 = [s for s in G.successors(node)]
        node_id = np.random.randint(0,len(successors_node_2))
        node = successors_node_2[node_id]
        node2_set.add(node)
    
    # 找交集
    intersection_set_total = node1_set.intersection(node2_set)
    intersection_set_constrained = set()
    
    # 验证交集中的节点是否是必经之路（即移除后是否阻断路径）
    for element in intersection_set_total:
        G_temp = deepcopy(G)
        try:
            G_temp.remove_node(element)
        except KeyError:
            pass
    
        flag_1 = reach_terminate_node(G_temp, node1, node_terminate)
        flag_2 = reach_terminate_node(G_temp, node2, node_terminate)
        
        # 如果移除 element 后，node1 和 node2 都无法到达终点，则该 element 是关键节点
        if(flag_1 == False and flag_2 == False):
            intersection_set_constrained.add(element)
    
    # 在所有关键节点中，找到层级最低（最浅）的那个
    level, parent = bfs_search(G, node_start)
    lowest_level = float('inf')
    reconv_node = None # 初始化
    for reconv_node_temp in intersection_set_constrained:
        if level[reconv_node_temp]<lowest_level:
            lowest_level = level[reconv_node_temp]
            reconv_node = reconv_node_temp
    return reconv_node

def find_choices(G, node, node_start, node_terminate, choices, parent):
    """
    递归查找图中的分支结构，识别潜在的 Choice 节点及其汇合点 (LCS)。
    """
    for child_node in G.neighbors(node):
        if child_node not in parent:
            for second_child_node in G.neighbors(node):
                if(child_node != second_child_node):
                    lcs = least_common_successor(G, child_node, second_child_node, node_start, node_terminate)
                    choices.append((node, child_node, second_child_node, lcs))
            parent[child_node] = node
            find_choices(G, child_node, node_start, node_terminate, choices, parent)

def group_choices(choices, level):
    """
    对找到的选择结构进行分组和排序。
    """
    ## order the choices
    final_choices = set()
    for t in choices:
        new_t = deepcopy(t)
        lst_size = len(t)
        for i in range(0, lst_size):
            for j in range(0, lst_size-i-1):
                if (level[new_t[j]] > level[new_t[j + 1]]):
                    new_t = list(new_t)
                    temp = new_t[j]
                    new_t[j]= new_t[j + 1]
                    new_t[j + 1]= temp
                    new_t = tuple(new_t)
        final_choices.add(deepcopy(new_t))
    
    ## output choice
    output_choice = {}
    for element in final_choices:
        try:
            output_choice[(element[0], element[-1])].extend(element[1:-1])
        except KeyError:
            output_choice[(element[0], element[-1])] = list(element[1:-1])
    
    for element in output_choice:
        output_choice[element] = list(set(output_choice[element]))
    
    ## modify the representation of choice
    modified_final_choices = {}
    for element in output_choice:
        try:
            key = [tuple(output_choice[element]), element[1]]
            modified_final_choices[element[0]].append(key)
        except KeyError:
            modified_final_choices[element[0]] = []
            key = [tuple(output_choice[element]), element[1]]
            modified_final_choices[element[0]].append(key)
    
    return modified_final_choices

def return_choices(G, node_start, node_terminate):
    """
    主入口函数：返回图中所有的选择结构。
    """
    ## bfs levels
    level, _ = bfs_search(G, node_start)
    
    ## choices
    choices = []
    parent = {node_start:None}
    find_choices(G, node_start, node_start, node_terminate, choices, parent)
    
    ## modified choices
    modified_choices = group_choices(choices, level)
    
    ## modify it to a form that can be used by checkSubgraph & expandSubgraph
    modified_choices_final = []
    for element in modified_choices:
        for element_2 in modified_choices[element]:
            individual_choice = [element]
            for element_3 in element_2:
                if (isinstance(element_3, PrimitiveNode) or isinstance(element_3, SequentialNode) or isinstance(element_3, ChoiceNode)):
                    individual_choice.append(element_3)
                else:
                    individual_choice.append(list(element_3))
            modified_choices_final.append(individual_choice)
    return modified_choices_final

def checkSubgraph(G, start_node, end_node, start_successors):
    '''
    Check if a set of LCS nodes and paths comprise a reducible subgraph
    检查一组 LCS 节点和路径是否构成一个可归约的子图。

    params:
        G : the full task graph
        start_node : name of the initial (source) node in G
        end_node : name of the final (sync) node in G
        start_successors : list of the names of start_node's successors that should be traversed

    return (bool, int) : (True if the subgraph is reducible, size of the subgraph)
    '''
    if len(start_successors) < 2:
        return False, -1

    valid_neighbors = [start_node, end_node]
    subgraph_nodes = []
    subgraph_nodes.extend(start_successors)

    frontier = []
    explored = []
    frontier.extend(start_successors)
    while len(frontier) > 0:
        n = frontier[0]
        frontier.remove(n)
        if n in explored or n == end_node:
            continue

        successors = list(G.successors(n))
        if len(successors) == 0:
            return False, -1

        frontier = successors + frontier
        frontier.extend(successors)
        explored.append(n)
        if n not in subgraph_nodes:
            subgraph_nodes.append(n)

    if len(valid_neighbors) > 2:
        return False, -1

    valid_neighbors.extend(subgraph_nodes)

    for n in subgraph_nodes:
        for neighbor in G.predecessors(n):
            if neighbor not in valid_neighbors:
                return False, -1

    return True, len(subgraph_nodes)

def expandGraph(G, start_node, end_node, start_successors):
    '''
    Replace a subgraph with an expanded subgraph that can be resolved further with sequence/parallel rules
    用一个扩展的子图替换原子图，使其可以通过串行/并行规则进一步解析。
    这通常涉及复制节点以消除非结构化的依赖。

    params:
        G : the full task graph (this will be modified with the result)
        start_node : name of the initial (source) node in G
        end_node : name of the final (sync) node in G
        start_successors : list of the names of start_node's successors that should be traversed
    '''
    G2 = nx.DiGraph()
    G2.add_nodes_from([start_node, end_node])

    middle_nodes = []
    i = 1
    # 为每个分支创建独立的路径（复制中间节点）
    for n in start_successors:
        node_name = deepcopy(n)
        node_name.change_name(node_name.get_name() + '-' + str(i))
        G2.add_node(node_name)
        G2.add_edge(start_node, node_name, prob=G.edges[(start_node, n)]['prob'])

        if n == end_node:
            continue

        explored_nodes = [end_node]
        nodes = [n]
        while (len(nodes) > 0):
            current_node = nodes[0]
            nodes.remove(current_node)
            if current_node not in explored_nodes:
                explored_nodes.append(current_node)
            
            node_name = deepcopy(current_node)
            node_name.change_name(current_node.get_name() + '-' + str(i))

            if current_node not in middle_nodes:
                middle_nodes.append(current_node)

            successors = list(G.successors(current_node))
            for s in successors:
                if s == end_node:
                    s_name = end_node
                else:
                    s_name = deepcopy(s)
                    s_name.change_name(s.get_name() + '-' + str(i))
                G2.add_node(s_name)
                G2.add_edge(node_name, s_name, prob=G.edges[(current_node, s)]['prob'])
                if s not in explored_nodes:
                    nodes.insert(0, s)

        i += 1

    # nx.draw(G2, with_labels=True)
    # plt.show()
    
    # print("edges")
#    for edge in G2.edges:
#        print(type(edge[0]))
#        print(type(edge[1]))
#        print(G2.get_edge_data(edge[0], edge[1])['prob'])

    # 替换原图中的节点和边
    G.remove_nodes_from(middle_nodes)
    G.add_nodes_from(list(G2.nodes))
    for edge in G2.edges:
        G.add_edge(edge[0], edge[1], prob=G2.edges[edge]['prob'])
        
    return G

def sort_subgraphs(subgraphs):
    l = len(subgraphs)
    for i in range(0,l):
        for j in range(0,l-i-1):
            if (subgraphs[j][1] > subgraphs[j + 1][1]):
                temp = subgraphs[j]
                subgraphs[j] = subgraphs[j+1]
                subgraphs[j+1] = temp
    return subgraphs
    

def restructure_htn_graph(htn_graph, node_start, node_terminate):
    """
    重构 HTN 图的入口函数。
    寻找可归约的子图并进行扩展。
    """
    subgraphs = []
    for candidate_subgraph in return_choices(htn_graph, node_start, node_terminate):
        each_subgraph = []
        flag, length = checkSubgraph(htn_graph, candidate_subgraph[0], candidate_subgraph[2], candidate_subgraph[1])
        if flag == True:
            each_subgraph.append(candidate_subgraph)
            each_subgraph.append(length)
            subgraphs.append(each_subgraph)
    
    subgraphs = sort_subgraphs(deepcopy(subgraphs))
    if len(subgraphs) > 0:
        smallest_candidate_subgraph = subgraphs[0][0]
        new_G = expandGraph(htn_graph, smallest_candidate_subgraph[0], smallest_candidate_subgraph[2], smallest_candidate_subgraph[1])        
        return new_G
    return htn_graph # 如果没有找到可归约的子图，返回原图
            
            
if __name__=="__main__":
    # htn = salad_demonstrations_to_htn() # This function is not defined in this file context, commenting out
    # Placeholder for main execution logic if needed
    pass
