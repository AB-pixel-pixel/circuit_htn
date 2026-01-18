# -*- coding: utf-8 -*-
import os
import sys
import pickle
import config
from construct_alfred_htn import get_alfred_demonstrations

# 添加路径以导入 modules
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)
scripts_dir = os.path.dirname(current_dir)
if scripts_dir not in sys.path:
    sys.path.append(scripts_dir)

try:
    import circuitHTN
    import htn
    from circuit_htn_node import CircuitHTNNode
except ImportError:
    pass # 路径设置正确的话应该能导入

def extract_actions_from_path(path):
    """从原始路径列表提取动作序列"""
    actions = []
    # path: [init_state, init_action, s0, a0, ..., term_state, term_action]
    # 动作在奇数索引: 1, 3, 5...
    for i in range(1, len(path), 2):
        action = path[i]
        # 过滤掉起始和终止动作
        if "init_action" in action or "term_action" in action:
            continue
        actions.append(action)
    return tuple(actions)

def extract_actions_from_htn_walk(node_list):
    """从 HTN random_walk 返回的节点列表提取动作序列"""
    actions = []
    for node in node_list:
        name = node.name
        # 过滤掉起始和终止动作
        if "init_action" in name or "term_action" in name:
            continue
        # 移除可能的 ID 后缀 (例如 "PickApple-1" -> "PickApple")
        # 注意：在 expandGraph 中，节点可能会被重命名为 name-i
        # 但原始动作名可能本身就包含 '-' (例如 Alfred 中的动作通常是 "PickObject")
        # 这里假设 expandGraph 增加的是 "-数字" 后缀
        
        # 简单的去后缀逻辑：如果是以 "-数字" 结尾，则去掉
        # 但更安全的是保留原始动作名。在 construct_alfred_htn 中，动作名直接来自 json。
        # 我们假设 restructure_graph.py 可能会添加后缀。
        
        # 尝试还原：如果最后一部分是数字，则去掉
        parts = name.rsplit('-', 1)
        if len(parts) > 1 and parts[1].isdigit():
            action_name = parts[0]
        else:
            action_name = name
            
        actions.append(action_name)
    return tuple(actions)

def evaluate_htn(htn_file_path, valid_data_dir, num_samples=1000, num_valid_demos=50):
    print(f"Loading HTN model from {htn_file_path}...")
    try:
        with open(htn_file_path, 'rb') as f:
            # 注意：这里加载的是 CircuitHTNNode 对象（最终保存的格式）
            # 但我们需要的是 htn.py 中的 Node 对象来进行 random_walk
            # 因为 CircuitHTNNode 没有 random_walk 方法（或者逻辑不同）
            # 查看 construct_alfred_htn.py，它保存的是 convertToCircuitHTN 的结果
            
            # 问题：CircuitHTNNode (circuit_htn_node.py) 没有 random_walk 方法。
            # htn.py 中的 ChoiceNode/SequentialNode 有 random_walk。
            # 
            # 解决方案：
            # 1. 修改 construct_alfred_htn.py 同时保存原始 HTN 对象。
            # 2. 或者在 CircuitHTNNode 中实现 random_walk。
            # 3. 或者在这个脚本里重新训练（不推荐，慢）。
            
            # 让我们检查 circuit_htn_node.py。它确实没有 random_walk。
            # 但是它有结构信息 (children, probabilities)。我们可以轻松实现一个 random_walk。
            circuit_htn_root = pickle.load(f)
    except FileNotFoundError:
        print(f"Error: HTN file {htn_file_path} not found. Please run construct_alfred_htn.py first.")
        return

    print(f"Loading validation data from {valid_data_dir}...")
    valid_paths_raw = get_alfred_demonstrations(valid_data_dir, num_demos=num_valid_demos)
    if not valid_paths_raw:
        print("No validation data found.")
        return

    valid_trajectories = set()
    for path in valid_paths_raw:
        valid_trajectories.add(extract_actions_from_path(path))
    
    print(f"Loaded {len(valid_trajectories)} unique validation trajectories.")

    print(f"Sampling {num_samples} plans from HTN...")
    sampled_trajectories = set()
    
    for _ in range(num_samples):
        # 执行 random walk
        traj = random_walk_circuit_htn(circuit_htn_root)
        sampled_trajectories.add(traj)
        
    print(f"Sampled {len(sampled_trajectories)} unique plans.")
    
    # 计算覆盖率
    # 检查有多少验证集轨迹 被 采样轨迹包含
    matches = 0
    for valid_traj in valid_trajectories:
        if valid_traj in sampled_trajectories:
            matches += 1
            
    print("-" * 30)
    print("Evaluation Results")
    print("-" * 30)
    print(f"Validation Set Size: {len(valid_trajectories)}")
    print(f"Exact Matches: {matches}")
    print(f"Coverage: {matches / len(valid_trajectories):.2%}")
    print("-" * 30)

def random_walk_circuit_htn(node):
    """
    针对 CircuitHTNNode 的随机游走实现
    """
    # 节点类型常量
    CHOICE = 0
    SEQUENCE = 1
    PRIMITIVE = 2
    
    if node.node_type == PRIMITIVE:
        # 过滤 init/term
        if "init_action" in node.name or "term_action" in node.name:
            return []
            
        # 清理名称 (同上)
        name = node.name
        parts = name.rsplit('-', 1)
        if len(parts) > 1 and parts[1].isdigit():
            action_name = parts[0]
        else:
            action_name = name
            
        # 注意：CircuitHTNNode 的 name 属性可能就是动作名，或者 action 属性是动作名
        # 查看 convertToCircuitHTN:
        # circuitHTN = CircuitHTNNode(name=str(root_htn_node.name), ..., action=action)
        # 所以优先使用 action 属性
        if node.action:
            return [node.action]
        return [action_name]

    elif node.node_type == SEQUENCE:
        traj = []
        for child in node.children:
            traj.extend(random_walk_circuit_htn(child))
        return tuple(traj) # 递归中间返回 list, 最后转 tuple? 不，这里需要在循环中 extend

    elif node.node_type == CHOICE:
        import random
        if not node.probabilities or len(node.probabilities) != len(node.children):
            # Fallback if probs are missing
            choice_idx = random.randrange(len(node.children))
        else:
            # 根据概率选择
            r = random.random()
            n = 0
            choice_idx = len(node.children) - 1
            for i, prob in enumerate(node.probabilities):
                n += prob
                if n > r:
                    choice_idx = i
                    break
        
        return random_walk_circuit_htn(node.children[choice_idx])

    return []

# 辅助函数：修正 SEQUENCE 的返回值处理
def random_walk_circuit_htn(node):
    CHOICE = 0
    SEQUENCE = 1
    PRIMITIVE = 2
    
    if node.node_type == PRIMITIVE:
        if "init_action" in node.name or "term_action" in node.name:
            return []
        if node.action:
            return [node.action]
        # Fallback
        name = node.name
        parts = name.rsplit('-', 1)
        if len(parts) > 1 and parts[1].isdigit():
            return [parts[0]]
        return [name]
        
    elif node.node_type == SEQUENCE:
        full_traj = []
        for child in node.children:
            full_traj.extend(random_walk_circuit_htn(child))
        return full_traj
        
    elif node.node_type == CHOICE:
        import random
        if not node.children:
            return []
            
        if not node.probabilities or len(node.probabilities) != len(node.children):
            choice_idx = random.randrange(len(node.children))
        else:
            r = random.random()
            n = 0
            choice_idx = len(node.children) - 1
            for i, prob in enumerate(node.probabilities):
                n += prob
                if n > r:
                    choice_idx = i
                    break
        return random_walk_circuit_htn(node.children[choice_idx])
    
    return []

def main():
    htn_path = "alfred_htn.pkl"
    # 使用 valid_seen 作为验证集
    valid_dir = os.path.join(config.ALFRED_DATA_PATH, "valid_seen")
    
    # 检查路径
    if not os.path.exists(valid_dir):
        print(f"Warning: {valid_dir} does not exist. Trying train dir for demo purposes.")
        valid_dir = os.path.join(config.ALFRED_DATA_PATH, "train")
        
    evaluate_htn(htn_path, valid_dir, num_samples=500, num_valid_demos=20)

if __name__ == "__main__":
    main()
