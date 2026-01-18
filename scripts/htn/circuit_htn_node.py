from __future__ import division
import pickle

class CircuitHTNNode(object):
    # 定义节点类型常量
    CHOICE = 0      # 选择节点（表示可以在多个子任务中选择一个执行）
    SEQUENCE = 1    # 序列节点（表示子任务需要按顺序执行）
    PRIMITIVE = 2   # 原始节点（表示具体的动作，是叶子节点）

    def __init__(self, name='', node_type=PRIMITIVE, parent=None, action=None, probabilities=[]):
        self.name = name  # 节点的名称
        self.node_type = node_type  # 节点类型，决定了子节点的执行顺序（顺序、选择或无子节点）
        self.action = action  # 具体的原始动作（仅对 PRIMITIVE 类型节点有效）

        self.parent = parent  # HTN 节点的父节点，如果存在；用于在网络中向上遍历
        self.children = []  # HTN 子节点的列表，如果存在；用于向下遍历以到达原始动作

        self.probabilities = probabilities  # 执行每个子节点的概率（用于 CHOICE 类型的决策节点）

    def set_children(self, nodes):
        """设置子节点列表"""
        self.children = nodes

    def add_child(self, node):
        """添加一个子节点"""
        self.children.append(node)

    def add_children(self, node_list):
        """批量添加子节点"""
        self.children.extend(node_list)

    def remove_child(self, node):
        """移除一个子节点，如果是选择节点，同时移除对应的概率"""
        if self.node_type == CircuitHTNNode.CHOICE:
            self.probabilities.pop(self.children.index(node))
        self.children.remove(node)

    def replace_child(self, old_child, new_child):
        """用新节点替换旧节点，保持位置不变"""
        i = self.children.index(old_child)
        self.remove_child(old_child)
        self.children.insert(i, new_child)

    def normalize_probabilities(self):
        """归一化概率，使所有子节点的概率之和为 1"""
        total = sum(self.probabilities)
        for i in range(len(self.probabilities)):
            self.probabilities[i] /= total

    def text_output(self, level=0, parent_type=None):
        """生成树状结构的文本表示，用于打印调试"""
        htn_str = ''
        for i in range(level):
            htn_str += '  '
        if parent_type is not None:
            if parent_type == CircuitHTNNode.SEQUENCE:
                htn_str += '=> ' # 序列的子节点前缀
            elif parent_type == CircuitHTNNode.CHOICE:
                htn_str += '<: ' # 选择的子节点前缀
        htn_str += str(self)

        for c in self.children:
            htn_str += '\n' + c.text_output(level + 1, self.node_type)
        return htn_str

    @staticmethod
    def type_to_string(node_type):
        """将节点类型常量转换为字符串描述"""
        if node_type == CircuitHTNNode.PRIMITIVE:
            return 'primitive'
        elif node_type == CircuitHTNNode.SEQUENCE:
            return 'sequence'
        elif node_type == CircuitHTNNode.CHOICE:
            return 'choice'

    def __str__(self):
        """对象的字符串表示"""
        if self.node_type == CircuitHTNNode.PRIMITIVE:
            return self.name + ' [' + self.action + ']'
        else:
            return self.name + ' (' + CircuitHTNNode.type_to_string(self.node_type) + ')'


    def __repr__(self):
        return str(self)