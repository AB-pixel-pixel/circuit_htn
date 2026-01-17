
# -*- coding: utf-8 -*-
import os
import sys
import json
import glob
import pickle
import networkx as nx

# 将当前脚本所在的目录添加到 sys.path，以便导入同目录下的模块
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

# 尝试添加上一级目录到 sys.path，以便 circuitHTN.py 能找到 simulator 模块
# circuitHTN.py 位于 circuit_htn/scripts/htn/
# simulator 位于 circuit_htn/scripts/simulator/
# 所以我们需要 circuit_htn/scripts/ 在 sys.path 中
scripts_dir = os.path.dirname(current_dir)
if scripts_dir not in sys.path:
    sys.path.append(scripts_dir)

# 导入 circuitHTN 中的相关函数
# generate_action_graphs_from_demonstrations: 从演示路径生成动作图
# action_graph_to_htn: 将动作图转换为 HTN
# convertToCircuitHTN: 将 HTN 转换为 CircuitHTN 对象以便保存
# visualize_with_graphviz_dot: 使用 Graphviz 可视化图形
try:
    from circuitHTN import generate_action_graphs_from_demonstrations, action_graph_to_htn, convertToCircuitHTN, visualize_with_graphviz_dot
    from htn import convertToDiGraph
except ImportError as e:
    print(f"导入模块失败: {e}")
    print(f"sys.path: {sys.path}")
    sys.exit(1)

def get_alfred_demonstrations(train_dir, num_demos=50):
    """
    读取 ALFRED 训练数据集中的轨迹数据，并将其转换为 circuitHTN 需要的格式。
    
    Args:
        train_dir: ALFRED 训练集根目录
        num_demos: 要读取的演示数量，默认为 50
        
    Returns:
        paths: 包含演示路径的列表，每个路径是一个列表 [init_state, init_action, s0, a0, ..., term_state, term_action]
    """
    demos = []
    # 查找所有的 traj_data.json 文件
    print(f"正在 {train_dir} 中搜索 traj_data.json 文件...")
    files = []
    # 使用 os.walk 递归遍历目录
    for root, dirs, filenames in os.walk(train_dir):
        for filename in filenames:
            if filename == 'traj_data.json':
                files.append(os.path.join(root, filename))
                if len(files) >= num_demos:
                    break
        if len(files) >= num_demos:
            break
            
    print(f"找到了 {len(files)} 条轨迹数据。")
    
    paths = []
    for file_path in files:
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
            
            # 获取高层 PDDL 计划
            high_pddl = data.get('plan', {}).get('high_pddl', [])
            if not high_pddl:
                print(f"文件 {file_path} 中没有 high_pddl，跳过。")
                continue
                
            # 构建 (state, action) 序列
            # circuitHTN 需要的格式是 [state, action, state, action, ...]
            # 为了避免将所有状态合并为一个（导致死循环或极其复杂的图），
            # 我们为每个步骤生成唯一的虚拟状态。
            # 这样每个演示将是一条独立的路径。
            
            path = []
            
            # 按照 circuitHTN 的习惯，我们需要添加 init_state 和 init_action
            path.append("init_state")
            path.append("init_action")
            
            # 使用文件名或索引作为演示的唯一标识
            demo_id = os.path.basename(os.path.dirname(file_path))
            
            for i, step in enumerate(high_pddl):
                discrete_action = step.get('discrete_action', {})
                action_name = discrete_action.get('action')
                args = discrete_action.get('args', [])
                
                # 格式化动作字符串
                full_action = action_name
                if args:
                    str_args = [str(arg) for arg in args]
                    full_action += "_" + "_".join(str_args)
                
                # 生成唯一的虚拟状态
                # 格式: state_{demo_id}_{step_index}
                unique_state = f"state_{demo_id}_{i}"
                
                path.append(unique_state)
                path.append(full_action)
            
            # 添加终止状态和动作
            path.append("term_state")
            path.append("term_action")
            
            paths.append(path)
            # print(f"成功处理轨迹: {file_path}") # 打印日志（可选）
            
        except Exception as e:
            print(f"处理文件 {file_path} 时出错: {e}")
            continue

    return paths

def main():
    # ALFRED 数据集路径
    alfred_train_dir = "/Users/ab/Code/construct_domain_knowledge/alfred/data/json_2.1.0/train"
    
    # 获取命令行参数中的演示数量
    num_demos = 50
    if len(sys.argv) > 1:
        try:
            num_demos = int(sys.argv[1])
        except ValueError:
            print("警告: 无效的参数，使用默认值 50")
            
    print(f"开始读取 ALFRED 演示数据 (目标数量: {num_demos})...")
    paths = get_alfred_demonstrations(alfred_train_dir, num_demos=num_demos)
    
    if not paths:
        print("未找到任何演示数据，程序结束。")
        return

    print(f"正在从 {len(paths)} 条演示数据构建动作图 (Action Graph)...")
    # 生成动作图
    action_graph = generate_action_graphs_from_demonstrations(paths)
    
    print("正在可视化动作图 (保存为 alfred_action_graph.png)...")
    # 可视化动作图
    try:
        visualize_with_graphviz_dot(action_graph, "alfred_action_graph")
    except Exception as e:
        print(f"可视化动作图失败 (可能是 graphviz 未安装): {e}")
    
    print("正在构建 HTN...")
    # 将动作图转换为 HTN
    # action_graph_to_htn 会尝试合并并行和串行的结构
    built_htn, result = action_graph_to_htn(action_graph)
    
    if result:
        print("HTN 构建成功！")
        
        print("正在可视化 HTN (保存为 alfred_htn.png)...")
        try:
            htn_digraph = convertToDiGraph(built_htn)
            visualize_with_graphviz_dot(htn_digraph, 'alfred_htn')
        except Exception as e:
            print(f"可视化 HTN 失败: {e}")
        
        print("正在将 HTN 保存到 alfred_htn.pkl...")
        # 保存为 pickle 文件
        circuitHTN = convertToCircuitHTN(built_htn)
        with open('alfred_htn.pkl', 'wb') as f:
            pickle.dump(circuitHTN, f, 2) # 使用协议 2 以兼容旧版本 Python（如果需要）
            
        print("完成。")
    else:
        print("HTN 构建失败。可能是无法进一步规约图形。")

if __name__ == "__main__":
    main()
