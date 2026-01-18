
import pickle
import json
import os
import sys

# 添加路径以导入 CircuitHTNNode
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

try:
    from circuit_htn_node import CircuitHTNNode
except ImportError:
    print("Could not import CircuitHTNNode from circuit_htn_node.py")
    # 如果导入失败，我们尝试定义一个兼容的类，但这对于 pickle 可能不起作用
    # 因为 pickle 需要类的全名匹配。
    # 通常 pickle 文件中记录的是 'circuit_htn_node.CircuitHTNNode'
    sys.exit(1)

def load_htn(pkl_path):
    with open(pkl_path, 'rb') as f:
        return pickle.load(f)

def htn_to_visjs(root_node):
    nodes = []
    edges = []
    
    # 使用栈进行遍历，同时维护父节点ID
    # (node, parent_id, edge_label)
    stack = [(root_node, None, None)]
    
    # 用于生成唯一ID
    node_id_counter = 0
    # 映射对象ID到visjs ID，防止重复处理
    visited = {} 
    
    while stack:
        curr, parent_id, edge_label = stack.pop(0) # BFS
        
        # 获取当前节点的唯一ID
        curr_obj_id = id(curr)
        if curr_obj_id in visited:
            my_id = visited[curr_obj_id]
        else:
            my_id = node_id_counter
            node_id_counter += 1
            visited[curr_obj_id] = my_id
            
            # 确定节点样式
            # 确保节点属性存在
            node_name = getattr(curr, 'name', 'Unknown')
            node_type = getattr(curr, 'node_type', CircuitHTNNode.PRIMITIVE)
            action = getattr(curr, 'action', None)
            
            label = str(node_name)
            color = '#97C2FC' # 默认蓝
            shape = 'ellipse'
            
            if node_type == CircuitHTNNode.PRIMITIVE:
                shape = 'box'
                color = '#7BE141' # 绿
                if action:
                    label += f"\n[{action}]"
            elif node_type == CircuitHTNNode.SEQUENCE:
                shape = 'diamond'
                color = '#FB7E81' # 红
                label = f"SEQ\n{node_name}"
            elif node_type == CircuitHTNNode.CHOICE:
                shape = 'triangle'
                color = '#FFA807' # 橙
                label = f"CHOICE\n{node_name}"
            
            nodes.append({
                'id': my_id,
                'label': label,
                'shape': shape,
                'color': color,
                'title': str(curr) # tooltip
            })
            
            # 处理子节点
            children = getattr(curr, 'children', [])
            probs = getattr(curr, 'probabilities', [])
            
            for i, child in enumerate(children):
                child_label = ""
                if node_type == CircuitHTNNode.SEQUENCE:
                    child_label = str(i + 1)
                elif node_type == CircuitHTNNode.CHOICE:
                    if i < len(probs):
                        child_label = f"{probs[i]:.2f}"
                
                stack.append((child, my_id, child_label))
        
        # 添加边
        if parent_id is not None:
            edge = {
                'from': parent_id,
                'to': my_id,
                'arrows': 'to'
            }
            if edge_label:
                edge['label'] = edge_label
                edge['font'] = {'align': 'top'}
            edges.append(edge)
            
    return nodes, edges

def generate_html(nodes, edges, output_file):
    html_content = """
<!DOCTYPE html>
<html>
<head>
    <title>HTN Visualization</title>
    <script type="text/javascript" src="https://unpkg.com/vis-network/standalone/umd/vis-network.min.js"></script>
    <style type="text/css">
        body { margin: 0; padding: 0; overflow: hidden; }
        #mynetwork {
            width: 100vw;
            height: 100vh;
            border: 1px solid lightgray;
        }
        #controls {
            position: absolute;
            top: 10px;
            left: 10px;
            z-index: 100;
            background: white;
            padding: 10px;
            border: 1px solid #ccc;
            border-radius: 5px;
        }
    </style>
</head>
<body>
<div id="controls">
    <h3>HTN Visualization</h3>
    <p>Scroll to zoom, drag to move.</p>
    <button onclick="network.fit()">Fit All</button>
</div>
<div id="mynetwork"></div>
<script type="text/javascript">
    var nodes = new vis.DataSet(__NODES__);
    var edges = new vis.DataSet(__EDGES__);

    var container = document.getElementById('mynetwork');
    var data = {
        nodes: nodes,
        edges: edges
    };
    var options = {
        layout: {
            hierarchical: {
                direction: "UD",
                sortMethod: "directed",
                nodeSpacing: 150,
                levelSeparation: 150,
                blockShifting: true,
                edgeMinimization: true,
                parentCentralization: true,
                shakeTowards: 'roots'
            }
        },
        physics: {
            enabled: false
        },
        edges: {
            smooth: {
                type: 'cubicBezier',
                forceDirection: 'vertical',
                roundness: 0.4
            }
        },
        interaction: {
            dragNodes: false,
            dragView: true,
            zoomView: true
        }
    };
    var network = new vis.Network(container, data, options);
</script>
</body>
</html>
"""
    html_content = html_content.replace('__NODES__', json.dumps(nodes))
    html_content = html_content.replace('__EDGES__', json.dumps(edges))
    
    with open(output_file, 'w') as f:
        f.write(html_content)
    print(f"Visualization saved to {output_file}")

def visualize_htn_from_pkl(pkl_file):
    if not os.path.exists(pkl_file):
        print(f"File {pkl_file} not found.")
        return
        
    print(f"Loading {pkl_file}...")
    try:
        root = load_htn(pkl_file)
    except Exception as e:
        print(f"Error loading pickle: {e}")
        return
    
    print("Converting to network format...")
    nodes, edges = htn_to_visjs(root)
    print(f"Generated {len(nodes)} nodes and {len(edges)} edges.")
    
    output_html = os.path.splitext(pkl_file)[0] + '.html'
    generate_html(nodes, edges, output_html)


if __name__ == "__main__":
    pkl_file = "domain_knowledge/alfred_htn_带变量.pkl" # 'alfred_htn.pkl'
    if len(sys.argv) > 1:
        pkl_file = sys.argv[1]
        
    visualize_htn_from_pkl(pkl_file)
