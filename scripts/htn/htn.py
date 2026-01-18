import copy
import random
import networkx as nx

from circuit_htn_node import CircuitHTNNode

class Node:
    """HTN 节点的基类"""
    def __init__(self, name, prestate, poststate):
        self.children = []
        self.name = name  # 节点名称
        self.prestate = prestate  # 执行前的状态（预期）
        self.poststate = poststate # 执行后的状态（预期）

    def get_name(self):
        return self.name

    def add_child(self, node):
        return

    def add_children(self, nodes):
        return

    def get_children(self):
        return self.children

    def random_walk(self):
        """随机游走，生成一个可能的动作序列"""
        return
    
    def change_name(self, node_name):
        self.name = node_name

    def __repr__(self):
        return self.prestate + ', ' + self.name

    def __str__(self):
        return self.name

class PrimitiveNode(Node):
    """原始动作节点（叶子节点），对应具体的 Action"""
    def add_child(self, node):
        print("Primitive nodes cannot have children")

    def add_children(self, node):
        print("Primitive nodes cannot have children")

    def random_walk(self):
        """原始节点的随机游走就是它自己"""
        return [self]

    def count_edges(self):
        return 0

    def count_choices(self):
        return 0

    def count_sequences(self):
        return 0

    def __hash__(self):
        return hash(self.name)
    
    def __eq__(self, other):
        if isinstance(other, PrimitiveNode):
            return self.name == other.name
        return False

class ChoiceNode(Node):
    """选择节点，表示子节点之间是 OR 关系，带有概率"""
    def __init__(self, name, prestate, poststate):
        super().__init__(name, prestate, poststate)
        self.children_freq = [] # 记录子节点对应的频率/概率

    def add_child(self, node):
        self.add_child_with_freq(node, 1.0)

    def add_child_with_freq(self, node, freq):
        self.children.append(node)
        self.children_freq.append(freq)

    def add_children(self, nodes):
        self.add_children_with_freq(nodes, [1.0] * len(nodes))

    def add_children_with_freq(self, nodes, node_frequencies):
        self.children.extend(nodes)
        self.children_freq.extend(node_frequencies)

    def get_children_freq(self):
        return self.children_freq

    def random_walk(self):
        """根据频率概率选择一个子节点进行游走"""
        weights = copy.deepcopy(self.children_freq)
        total_weight = 0
        for i in range(len(weights)):
            total_weight += weights[i]
        for i in range(len(weights)):
            weights[i] /= float(total_weight)
        r = random.random()
        n = 0
        choice = len(weights) - 1
        for i in range(len(weights)):
            n += weights[i]
            if n > r:
                choice = i
                break
        return self.children[choice].random_walk()

        # my ubuntu distro only has python 3.5, so I implemented this myself above :(
        # return random.choices(population=self.children, weights=self.children_freq)[0].random_walk()

    def count_edges(self):
        count = len(self.children)
        for child in self.children:
            count += child.count_edges()
        return count

    def count_choices(self):
        count = 1
        for child in self.children:
            count += child.count_choices()
        return count

    def count_sequences(self):
        count = 0
        for child in self.children:
            count += child.count_sequences()
        return count

    def __hash__(self):
        return hash(self.name)
    
    def __eq__(self,other):
        if isinstance(other, ChoiceNode):
            return self.name == other.name
        return False
    

class SequentialNode(Node):
    """序列节点，表示子节点之间是 AND 关系，按顺序执行"""
    def add_child_to_front(self, node):
        self.children.insert(0, node)

    def add_child(self, node):
        self.children.append(node)

    def add_children(self, nodes):
        self.children.extend(nodes)

    def random_walk(self):
        """序列节点的随机游走是所有子节点游走结果的串联"""
        walk = []
        for child in self.children:
            walk.extend(child.random_walk())
        return walk

    def count_edges(self):
        count = len(self.children)
        for child in self.children:
            count += child.count_edges()
        return count

    def count_choices(self):
        count = 0
        for child in self.children:
            count += child.count_choices()
        return count

    def count_sequences(self):
        count = 1
        for child in self.children:
            count += child.count_sequences()
        return count

    def __hash__(self):
        return hash(self.name)
    
    def __eq__(self,other):
        if isinstance(other, SequentialNode):
            return self.name == other.name
        return False


def printHTN(root_htn_node):
    return printHTNHelper(root_htn_node)

def printHTNHelper(node, level=0):
    """递归打印 HTN 结构"""
    spacing = ''
    for i in range(level):
        spacing += '  '

    name = node.name
    if node.__class__.__name__ == 'ChoiceNode':
        name += ' [child_probs: ' + str(node.get_children_freq()) + ']'
    print(spacing + name)

    children = node.get_children()
    for i in range(len(children)):
        child = children[i]
        printHTNHelper(child, level=level+1)

def convertToDiGraph(root_htn_node):
    """将 HTN 转换为 NetworkX 的有向图，用于可视化"""
    digraph = nx.DiGraph()
    print(root_htn_node, ' is the root of the tree.')
    return convertToDiGraphHelper(root_htn_node, digraph)

def convertToDiGraphHelper(root_htn_node, digraph):
    node_type = "primitive"
    if root_htn_node.__class__.__name__ == 'ChoiceNode':
        node_type = "choice"
    elif root_htn_node.__class__.__name__ == 'SequentialNode':
        node_type = "sequence"
    digraph.add_node(root_htn_node.name, htn_node_type=node_type)
    children = root_htn_node.get_children()
    for i in range(len(children)):
        child = children[i]
        convertToDiGraphHelper(child, digraph)
        if root_htn_node.__class__.__name__ == 'ChoiceNode':
            digraph.add_edge(root_htn_node.name, child.name, prob=root_htn_node.get_children_freq()[i])
        else:
            digraph.add_edge(root_htn_node.name, child.name)
    return digraph

def convertToCircuitHTN(root_htn_node):
    """将内部的 HTN 节点转换为 CircuitHTNNode 类型，可能用于后续导出"""
    if root_htn_node.__class__.__name__ == 'ChoiceNode':
        circuitHTN = CircuitHTNNode(name=str(root_htn_node.name), node_type=CircuitHTNNode.CHOICE,
                                    probabilities=root_htn_node.get_children_freq())
    elif root_htn_node.__class__.__name__ == 'SequentialNode':
        circuitHTN = CircuitHTNNode(name=str(root_htn_node.name), node_type=CircuitHTNNode.SEQUENCE)
    else:
        action = root_htn_node.name
        action = action[action.find('-') + 2:]
        if action != 'init_action' and action != 'term_action':
            circuitHTN = CircuitHTNNode(name=str(root_htn_node.name), node_type=CircuitHTNNode.PRIMITIVE, action=action)
        
        # 修正：如果是原始节点，这里可能需要返回，或者处理逻辑需要调整，根据原代码逻辑保持一致
        # 原代码逻辑似乎假设根节点如果是 Primitive，这里可能没有正确处理 children 递归（因为 primitive 没有 children）
        # 但如果是 Choice/Sequence，下面会递归

    if root_htn_node.__class__.__name__ == 'ChoiceNode' or root_htn_node.__class__.__name__ == 'SequentialNode':
        for c in root_htn_node.get_children():
            convertToCircuitHTNHelper(c, circuitHTN)

    return circuitHTN

def convertToCircuitHTNHelper(node, circuitHTNNode):
    if node.__class__.__name__ == 'ChoiceNode':
        child = CircuitHTNNode(name=str(node.name), node_type=CircuitHTNNode.CHOICE, parent=circuitHTNNode,
                                    probabilities=node.get_children_freq())
        circuitHTNNode.add_child(child)

        for c in node.get_children():
            convertToCircuitHTNHelper(c, child)

    elif node.__class__.__name__ == 'SequentialNode':
        child = CircuitHTNNode(name=str(node.name), node_type=CircuitHTNNode.SEQUENCE, parent=circuitHTNNode)
        circuitHTNNode.add_child(child)

        for c in node.get_children():
            convertToCircuitHTNHelper(c, child)

    else:  # primitive
        action = node.name
        action = action[action.find('-') + 2:]
        if action != 'init_action' and action != 'term_action':
            child = CircuitHTNNode(name=str(node.name), node_type=CircuitHTNNode.PRIMITIVE, action=action, parent=circuitHTNNode)
            circuitHTNNode.add_child(child)
