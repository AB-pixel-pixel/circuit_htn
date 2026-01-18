import networkx as nx
import htn
from subprocess import check_call
import pydot
import llm_utils # 引入 LLM 工具模块

choiceid = 0
seqid = 0

def create_init_htn_graph(task_graph):
    """
    初始化 HTN 图。
    将任务图（Task Graph）中的每个动作节点转换为 HTN 的 PrimitiveNode。
    """
    htn_graph = nx.DiGraph()
    temp_dict = {}
    # 遍历任务图中的所有节点，转换为 PrimitiveNode
    for action_node in task_graph.nodes:
        prestate = ""
        # 获取入边的状态作为 prestate
        for in_edge in task_graph.in_edges(action_node):
            prestate = task_graph.edges[in_edge[0], in_edge[1]]['state']
            break
        poststate = ""
        # 获取出边的状态作为 poststate
        for out_edge in task_graph.out_edges(action_node):
            poststate = task_graph.edges[out_edge[0], out_edge[1]]['state']
            break
        primitive_node = htn.PrimitiveNode(action_node, prestate, poststate)
        temp_dict[action_node] = primitive_node
        htn_graph.add_node(primitive_node)

    # 复制边到 HTN 图中
    for edge in task_graph.edges:
        htn_graph.add_edge(temp_dict[edge[0]], temp_dict[edge[1]], prob=task_graph.edges[edge]['prob'])

    return htn_graph

def merge_semantically_identical_nodes(htn_graph, htn1, htn2):
    """
    将两个语义上相同的节点合并为一个。
    保留 htn1，移除 htn2，并将 htn2 的流量（概率）加到 htn1 上。
    """
    htn1_predecessors = list(htn_graph.predecessors(htn1))
    
    # 假设只有一个前驱（基于并行合并的前提条件）
    pred = htn1_predecessors[0]
    
    htn1_prob = htn_graph.get_edge_data(pred, htn1)['prob']
    htn2_prob = htn_graph.get_edge_data(pred, htn2)['prob']
    
    # 更新 htn1 的概率
    # 注意：这里我们假设 htn1 和 htn2 是并行分支，所以概率直接相加
    nx.set_edge_attributes(htn_graph, {(pred, htn1): {'prob': htn1_prob + htn2_prob}})
    
    # 移除 htn2
    htn_graph.remove_node(htn2)
    # print(f"Semantically merged {htn2.name} into {htn1.name}")

def check_and_combine_semantically_identical_nodes(htn_graph):
    """
    检查并合并语义上相同的并行节点。
    条件：
    1. 共享相同的前驱和后继（并行结构）。
    2. 节点名称在语义上相似（通过 LLM 判断）。
    """
    combined = False
    nodes = list(htn_graph.nodes)
    for i in range(len(nodes)):
        for j in range(i + 1, len(nodes)):
            htn1 = nodes[i]
            htn2 = nodes[j]
            
            # 必须先存在于图中（可能在之前的迭代中被移除）
            if not htn_graph.has_node(htn1) or not htn_graph.has_node(htn2):
                continue
                
            htn1_predecessors = list(htn_graph.predecessors(htn1))
            htn2_predecessors = list(htn_graph.predecessors(htn2))
            htn1_successors = list(htn_graph.successors(htn1))
            htn2_successors = list(htn_graph.successors(htn2))
            
            # 检查结构条件：相同的单个前驱和单个后继
            if len(htn1_predecessors) == 1 and len(htn2_predecessors) == 1 \
                    and len(htn1_successors) == 1 and len(htn2_successors) == 1 \
                    and htn1_predecessors[0] == htn2_predecessors[0] \
                    and htn1_successors[0] == htn2_successors[0]:
                
                # 检查语义条件
                if llm_utils.are_nodes_semantically_similar(htn1.name, htn2.name):
                    merge_semantically_identical_nodes(htn_graph, htn1, htn2)
                    combined = True
                    return combined, htn1, htn2 # 每次只合并一对，保持循环简单
                    
    return combined, None, None

def combine_htns_in_parallel(htn_graph, htn1, htn2):
    """
    将两个并行的 HTN 节点合并为一个 ChoiceNode。
    假设 htn1 和 htn2 共享相同的前驱和后继。
    """
    global choiceid
    htn1_predecessors = list(htn_graph.predecessors(htn1))
    htn1_successors = list(htn_graph.successors(htn1))

    # 获取进入概率
    htn1_prob = htn_graph.get_edge_data(htn1_predecessors[0], htn1)['prob']
    htn2_prob = htn_graph.get_edge_data(htn1_predecessors[0], htn2)['prob']

    # 从图中移除原始节点
    htn_graph.remove_node(htn1)
    htn_graph.remove_node(htn2)

    # 创建新的 ChoiceNode 或合并到现有的 ChoiceNode
    if htn1.__class__.__name__ == 'ChoiceNode' and htn2.__class__.__name__ == 'ChoiceNode':
        # 如果两个都是 ChoiceNode，合并它们的子节点
        htn1.add_children_with_freq(htn2.getChildren(), htn2.get_children_freq())
        choice_node = htn1
    elif htn2.__class__.__name__ == 'ChoiceNode':
        # 如果 htn2 是 ChoiceNode，将 htn1 加入
        htn2.add_child_with_freq(htn1, htn1_prob)
        choice_node = htn2
    elif htn1.__class__.__name__ == 'ChoiceNode':
        # 如果 htn1 是 ChoiceNode，将 htn2 加入
        htn1.add_child_with_freq(htn2, htn2_prob)
        choice_node = htn1
    else:
        # 创建新的 ChoiceNode
        choice_node = htn.ChoiceNode('C' + str(choiceid), htn1.prestate, htn1.poststate)
        choiceid += 1
        choice_node.add_child_with_freq(htn1, htn1_prob)
        choice_node.add_child_with_freq(htn2, htn2_prob)


    # 将新节点加入图并连接
    htn_graph.add_node(choice_node)
    htn_graph.add_edge(htn1_predecessors[0], choice_node, prob=htn1_prob + htn2_prob)
    htn_graph.add_edge(choice_node, htn1_successors[0], prob=1.0)
    
    return choice_node

def check_and_combine_htns_in_parallel(htn_graph):
    """
    检查图中是否存在可以并行合并的节点对，如果存在则进行合并。
    条件：相同的单个前驱和相同的单个后继。
    优化：批量合并所有可合并的节点。
    """
    combined_any = False
    last_htn1, last_htn2 = None, None
    
    # 1. Group nodes by (pred, succ)
    groups = {} # (pred, succ) -> [node1, node2, ...]
    
    # Snapshot of nodes
    nodes = list(htn_graph.nodes)
    for node in nodes:
        if not htn_graph.has_node(node): continue
            
        try:
            preds = list(htn_graph.predecessors(node))
            succs = list(htn_graph.successors(node))
            
            if len(preds) == 1 and len(succs) == 1:
                key = (preds[0], succs[0])
                if key not in groups:
                    groups[key] = []
                groups[key].append(node)
        except:
            pass
            
    # 2. Process groups
    for key, group in groups.items():
        if len(group) > 1:
            # Iteratively merge
            current_node = group[0]
            for i in range(1, len(group)):
                next_node = group[i]
                if htn_graph.has_node(current_node) and htn_graph.has_node(next_node):
                    current_node = combine_htns_in_parallel(htn_graph, current_node, next_node)
                    combined_any = True
                    last_htn1, last_htn2 = group[0], group[1] # Just for return values
                    
    return combined_any, last_htn1, last_htn2

# htn1 connects to htn2
def combine_htns_in_series(htn_graph, htn1, htn2):
    """
    将两个串行连接的 HTN 节点合并为一个 SequentialNode。
    htn1 -> htn2
    """
    global seqid
    # 保存前驱和后继的连接信息
    htn1_predecessors_with_prob = []
    for predecessor in htn_graph.predecessors(htn1):
        htn1_pred_prob = htn_graph.get_edge_data(predecessor, htn1)['prob']
        htn1_predecessors_with_prob.append((predecessor, htn1_pred_prob))
    htn2_successors_with_prob = []
    for successor in htn_graph.successors(htn2):
        htn2_succ_prob = htn_graph.get_edge_data(htn2, successor)['prob']
        htn2_successors_with_prob.append((successor, htn2_succ_prob))

    # 移除旧节点
    htn_graph.remove_node(htn1)
    htn_graph.remove_node(htn2)

    # 创建或扩展 SequentialNode
    if htn1.__class__.__name__ == 'SequentialNode' and htn2.__class__.__name__ == 'SequentialNode':
        # 合并两个序列
        htn1.add_children(htn2.get_children())
        htn1.poststate = htn2.poststate
        sequence_node = htn1
    elif htn2.__class__.__name__ == 'SequentialNode':
        # 将 htn1 加到 htn2 前面
        htn2.add_child_to_front(htn1)
        htn2.prestate = htn1.prestate
        sequence_node = htn2
    elif htn1.__class__.__name__ == 'SequentialNode':
        # 将 htn2 加到 htn1 后面
        htn1.add_child(htn2)
        htn1.poststate = htn2.poststate
        sequence_node = htn1
    else:
        # 创建新序列
        sequence_node = htn.SequentialNode('S' + str(seqid), htn1.prestate, htn2.poststate)
        seqid += 1
        sequence_node.add_child(htn1)
        sequence_node.add_child(htn2)

    # 添加新节点并恢复连接
    htn_graph.add_node(sequence_node)
    for htn1_pred, htn1_pred_prob in htn1_predecessors_with_prob:
        htn_graph.add_edge(htn1_pred, sequence_node, prob=htn1_pred_prob)
    for htn2_succ, htn2_succ_prob in htn2_successors_with_prob:
        htn_graph.add_edge(sequence_node, htn2_succ, prob=htn2_succ_prob)

    return sequence_node

def check_and_combine_htns_in_series(htn_graph):
    """
    检查图中是否存在可以串行合并的节点对。
    条件：htn1 的唯一后继是 htn2，且 htn2 的唯一前驱是 htn1。
    优化：批量合并不重叠的串行对。
    """
    combined_any = False
    last_htn1, last_htn2 = None, None
    
    nodes = list(htn_graph.nodes)
    available = set(nodes)
    
    for u in nodes:
        if u not in available: continue
        if not htn_graph.has_node(u): continue
        
        try:
            succs = list(htn_graph.successors(u))
            if len(succs) == 1:
                v = succs[0]
                if v in available and htn_graph.has_node(v):
                    preds_v = list(htn_graph.predecessors(v))
                    if len(preds_v) == 1 and preds_v[0] == u:
                        # Found match (u, v)
                        # Remove from available to avoid overlap in this pass
                        available.discard(u)
                        available.discard(v)
                        
                        # Merge
                        combine_htns_in_series(htn_graph, u, v)
                        combined_any = True
                        last_htn1, last_htn2 = u, v
        except:
            pass
                
    return combined_any, last_htn1, last_htn2

def visualize_with_graphviz_dot(digraph, file_name):
    """使用 Graphviz 可视化当前图状态"""
    nx.drawing.nx_pydot.write_dot(digraph, "./debug_graphs/"+file_name + ".dot")
    check_call(['dot', '-Tpng', "./debug_graphs/"+file_name + '.dot', '-o', "./debug_graphs/"+file_name + '.png'])

def task_graph_to_htn(task_graph):
    """
    主函数：将任务图转换为 HTN。
    通过反复应用串行和并行合并规则，直到图被归约为单个根节点或无法继续归约。
    """
    htn_graph = create_init_htn_graph(task_graph)
    # print(nx.to_dict_of_lists(htn_graph))

    no = 0
    # 只要图中节点数大于1，就尝试合并
    while len(list(htn_graph.nodes)) > 1:
        # 1. 尝试并行语义合并 (新增)
        combined_semantically, s1, s2 = check_and_combine_semantically_identical_nodes(htn_graph)
        if combined_semantically:
            # 如果发生了语义合并，重新开始循环（图结构已变）
            continue

        # 2. 尝试结构并行合并
        combined_htns_in_parallel, htn1, htn2 = check_and_combine_htns_in_parallel(htn_graph)
        
        # 3. 尝试结构串行合并
        combined_htns_in_series, htn3, htn4 = check_and_combine_htns_in_series(htn_graph)
        
        # print("start of iteration")
        # print(htn1)
        # print(htn2)
        # print(htn3)
        # print(htn4)

        ## plot the reduced graph at each iteration to check for bugs
        # visualize_with_graphviz_dot(htn_graph, str(no))
        no += 1

        # 如果既没有串行合并也没有并行合并，说明遇到了无法归约的结构（需要 restructure）或已经完成
        if not (combined_htns_in_series or combined_htns_in_parallel):
            # print("graph:", nx.to_dict_of_lists(htn_graph))
            # print("type of graph:", type(htn_graph))
            # print("Unable to create HTN graph")
            return htn_graph
            break

    # if len(list(htn_graph.nodes)) == 1:
    #     print("Created HTN graph")


    # print(nx.to_dict_of_lists(htn_graph))
    # print(list(htn_graph.nodes)[0])

    return list(htn_graph.nodes)[0]
