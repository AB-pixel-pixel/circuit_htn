import networkx as nx
import collections

####Legacy Code#####

counter = 0
# a0 = 'idle_action'
# a6 = 'terminate_action'
def getLFDTrace():
    dict = {}
    dict['a0'] = [(('s0',), 'a1'), (('s0',), 'a2')]
    dict['a1'] = [(('s0', 's1'), 'a2'), (('s0', 's1', 's2', 's3', 's4'), 'a5')]
    dict['a2'] = [(('s0', 's1', 's2'), 'a3'), (('s0', 's1', 's2'), 'a4'), (('s0', 's2'), 'a3'), (('s0', 's2'), 'a4')]
    dict['a3'] = [(('s0', 's1', 's2', 's3'), 'a4'), (('s0', 's2', 's3'), 'a4'), (('s0', 's2', 's3', 's4'), 'a1'), (('s0', 's1', 's2', 's3', 's4'), 'a5')]
    dict['a4'] = [(('s0', 's1', 's2', 's4'), 'a3'), (('s0', 's2', 's4'), 'a3'), (('s0', 's2', 's3', 's4'), 'a1'), (('s0', 's1', 's2', 's3', 's4'), 'a5')]
    dict['a5'] = [(('s0', 's1', 's2', 's3', 's4', 's5'), 'a6')]
    dict['a6'] = []

    # dict = {}
    # dict['a0'] = [(('s0',), 'a1'), (('s0',), 'a2')]
    # dict['a1'] = [(('s0', 's1'), 'a2'), (('s0', 's1', 's2'), 'a3')]
    # dict['a2'] = [(('s0', 's1', 's2'), 'a3'), (('s0', 's2'), 'a1')]
    # dict['a3'] = []
    return dict

def convertLFDTraceToTaskGraph(lfd_trace):
    graph = nx.DiGraph()
    current_node = ((), 'a0')
    visitedSet = {}
    convertLFDTraceToTaskGraphHelper(lfd_trace, graph, current_node, visitedSet)
    return graph

def convertLFDTraceToTaskGraphHelper(lfd_trace, graph, current_node, visitedSet):
    global counter
    # TODO: Use matplotlib to plot node and edge attributes. Once this occurs, replace below line with:
    # node_id = str(self.counter)
    node_id = str(counter) + ": " + current_node[1]
    counter += 1
    visitedSet[current_node] = node_id
    graph.add_node(node_id, action=current_node[1])
    for neighbor in lfd_trace[current_node[1]]:
        current_node_state_comp = list(current_node[0])
        current_node_state_comp.append('s' + current_node[1][len(current_node) - 1])
        next_node_state_comp = tuple(current_node_state_comp)
        if collections.Counter(next_node_state_comp) == collections.Counter(neighbor[0]):
            if neighbor not in visitedSet:
                convertLFDTraceToTaskGraphHelper(lfd_trace, graph, neighbor, visitedSet)
            graph.add_edge(node_id, visitedSet[neighbor], state_composition=neighbor[0])


####End of Legacy Code#####

# Each state with id (e.g. 's6') represents a unique state vector.
class StateVectorTaskGraphBuilder:
    """
    负责将 LfD (Learning from Demonstration) 轨迹转换为任务图的构建器。
    使用状态向量作为节点/边的标识。
    """
    def __init__(self):
        self.counter = 0

    # a0 = 'idle_action'
    # a6 = 'terminate_action'
    def getLFDTrace(self):
        """
        获取示例 LfD 轨迹数据。
        数据格式: adjList[action] = ('state before action', 'state after action', 'successor action node')
        注意：这里的 key 是动作，value 是一个列表，包含可能的转换。
        这实际上是一个邻接表表示。
        """
        # Nodes are in the format of:
        # adjList[action] = ('state before action', 'state after action', 'successor action node')
        adjList = {}
        adjList['a0'] = [('', 's0', 'a1'), ('', 's0', 'a2')]
        adjList['a1'] = [('s0', 's1', 'a2'), ('s8', 's9', 'a5')]
        adjList['a2'] = [('s1', 's3', 'a3'), ('s1', 's3', 'a4'), ('s0', 's2', 'a3'), ('s0', 's2', 'a4')]
        adjList['a3'] = [('s3', 's6', 'a4'), ('s7', 's9', 'a5'), ('s2', 's4', 'a4'), ('s5', 's8', 'a1')]
        adjList['a4'] = [('s6', 's9', 'a5'), ('s3', 's7', 'a3'), ('s4', 's8', 'a1'), ('s2', 's5', 'a3')]
        adjList['a5'] = [('s9', 's10', 'a6')]
        adjList['a6'] = []
        return adjList

    def getLFDTrace2(self):
        """获取另一个示例 LfD 轨迹数据"""
        adjList = {}
        adjList['a0'] = [('', 's0', 'a1'), ('', 's0', 'a2'), ('', 's0', 'a3')]
        adjList['a1'] = [('s0', 's1', 'a4')]
        adjList['a2'] = [('s0', 's1', 'a4')]
        adjList['a3'] = [('s0', 's2', 'a5')]
        adjList['a4'] = [('s1', 's2', 'a5')]
        adjList['a5'] = []
        return adjList

    def convertLFDTraceToTaskGraph(self, lfd_trace, root_state, root_action):
        """
        将 LfD 轨迹转换为 NetworkX 的有向图 (Task Graph)。
        
        Args:
            lfd_trace: 轨迹数据字典
            root_state: 初始状态
            root_action: 初始动作
        """
        graph = nx.DiGraph()
        current_node = (root_state, root_action) # 节点由 (前驱状态, 动作) 唯一标识
        visitedSet = {} # 用于记录已访问的节点，避免重复创建
        self.convertLFDTraceToTaskGraphHelper(lfd_trace, graph, current_node, visitedSet)
        return graph

    # Current_node is a tuple containing previous state (edge) and current action (node)
    def convertLFDTraceToTaskGraphHelper(self, lfd_trace, graph, current_node, visitedSet):
        """
        递归辅助函数，用于深度优先遍历轨迹并构建图。
        """
        #TODO: Use matplotlib to plot node and edge attributes. Once this occurs, replace below line with:
        #node_id = str(self.counter)
        node_id = str(self.counter) + " - " + current_node[1] # 创建唯一的节点 ID
        self.counter += 1
        visitedSet[current_node] = node_id
        graph.add_node(node_id, action=current_node[1]) # 在图中添加节点
        
        # 遍历当前动作的所有后续转换
        for neighbor in lfd_trace[current_node[1]]:
            # neighbor 格式: (state_before, state_after, successor_action)
            # 检查转换的前置状态是否匹配当前状态
            if current_node[0] == neighbor[0]:
                neighbor_node = (neighbor[1], neighbor[2]) # 下一个节点标识: (state_after, successor_action)
                if neighbor_node not in visitedSet:
                    self.convertLFDTraceToTaskGraphHelper(lfd_trace, graph, neighbor_node, visitedSet)
                # 添加边，边上记录转换后的状态和概率
                graph.add_edge(node_id, visitedSet[neighbor_node], state=neighbor[1], prob=lfd_trace[current_node[1]][neighbor])



class StateCompositionTaskGraphBuilder:

    def __init__(self):
        self.counter = 0

    # a0 = 'idle_action'
    # a6 = 'terminate_action'
    def getLFDTrace(self):
        adjList = {}
        adjList['a0'] = [(('s0',), 'a1'), (('s0',), 'a2')]
        adjList['a1'] = [(('s0', 's1'), 'a2'), (('s0', 's1', 's2', 's3', 's4'), 'a5')]
        adjList['a2'] = [(('s0', 's1', 's2'), 'a3'), (('s0', 's1', 's2'), 'a4'), (('s0', 's2'), 'a3'),
                      (('s0', 's2'), 'a4')]
        adjList['a3'] = [(('s0', 's1', 's2', 's3'), 'a4'), (('s0', 's2', 's3'), 'a4'), (('s0', 's2', 's3', 's4'), 'a1'),
                      (('s0', 's1', 's2', 's3', 's4'), 'a5')]
        adjList['a4'] = [(('s0', 's1', 's2', 's4'), 'a3'), (('s0', 's2', 's4'), 'a3'), (('s0', 's2', 's3', 's4'), 'a1'),
                      (('s0', 's1', 's2', 's3', 's4'), 'a5')]
        adjList['a5'] = [(('s0', 's1', 's2', 's3', 's4', 's5'), 'a6')]
        adjList['a6'] = []
        return adjList


    def convertLFDTraceToTaskGraph(self, lfd_trace):
        graph = nx.DiGraph()
        current_node = ((), 'a0')
        visitedSet = {}
        self.convertLFDTraceToTaskGraphHelper(lfd_trace, graph, current_node, visitedSet)
        return graph


    def convertLFDTraceToTaskGraphHelper(self, lfd_trace, graph, current_node, visitedSet):
        # TODO: Use matplotlib to plot node and edge attributes. Once this occurs, replace below line with:
        # node_id = str(self.counter)
        node_id = str(self.counter) + ": " + current_node[1]
        self.counter += 1
        visitedSet[current_node] = node_id
        graph.add_node(node_id, action=current_node[1])
        for neighbor in lfd_trace[current_node[1]]:
            current_node_state_comp = list(current_node[0])
            current_node_state_comp.append('s' + current_node[1][len(current_node) - 1])
            next_node_state_comp = tuple(current_node_state_comp)
            if collections.Counter(next_node_state_comp) == collections.Counter(neighbor[0]):
                if neighbor not in visitedSet:
                    self.convertLFDTraceToTaskGraphHelper(lfd_trace, graph, neighbor, visitedSet)
                graph.add_edge(node_id, visitedSet[neighbor], state_composition=neighbor[0])


def main():
    taskGraphBuilder = StateVectorTaskGraphBuilder()
    lfdTrace = taskGraphBuilder.getLFDTrace()
    graph = taskGraphBuilder.convertLFDTraceToTaskGraph(lfdTrace)
    # TODO: Use matplotlib to draw the graph.
    print('compiled graph')


if __name__ == '__main__':
    main()
