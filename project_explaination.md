# 项目解释与代码导读 (Project Explanation and Code Walkthrough)

本文档旨在帮助理解该项目的代码结构、各个文件的功能以及它们如何对应于论文 *Chen et al. - 2021 - Learning Hierarchical Task Networks with Preferences from Unannotated Demonstrations* 中的算法流程。

## 1. 项目概览

该项目实现了从无标注的演示数据（Unannotated Demonstrations）中学习分层任务网络（Hierarchical Task Networks, HTN）的算法。核心流程是将一系列的动作序列（Demonstrations）转化为任务图（Task Graph），然后通过一系列的图归约操作（合并串行和并行结构）将其转化为HTN结构。

### 核心 Pipeline

1.  **输入处理**: 读取演示数据（Action sequences + State changes）。
2.  **构建任务图 (Task Graph Construction)**: 将演示轨迹转化为一个有向图，其中节点是动作，边表示状态转换。
3.  **HTN 归约 (HTN Reduction)**:
    *   **串行合并 (Series Combination)**: 将线性连接的节点合并为序列节点 (SequentialNode)。
    *   **并行合并 (Parallel Combination)**: 将共享相同前驱和后继的节点合并为选择节点 (ChoiceNode)。
    *   **图重构 (Graph Restructuring)**: 处理无法直接通过串行/并行合并的复杂子图结构（"Bad Graph" scenarios）。
4.  **输出**: 生成最终的 HTN 结构。

## 2. 文件详细说明

以下是 `scripts/htn/` 目录下主要代码文件的功能说明：

### 2.1 数据结构定义

*   **`htn.py`**:
    *   **功能**: 定义了 HTN 的中间节点类型。
    *   **核心类**:
        *   `Node`: 基类。
        *   `PrimitiveNode`: 原始动作节点（叶子节点）。
        *   `ChoiceNode`: 选择节点（OR结构），包含子节点的选择概率。
        *   `SequentialNode`: 序列节点（AND结构），表示一系列按顺序执行的动作。
    *   **对应论文**: 对应 HTN 的树状结构定义。

*   **`circuit_htn_node.py`**:
    *   **功能**: 定义了最终输出或用于电路设计的 HTN 节点结构 `CircuitHTNNode`。
    *   **特点**: 更加轻量级，包含 `node_type` (CHOICE, SEQUENCE, PRIMITIVE) 和 `probabilities` 等属性，方便后续导出或使用。

### 2.2 算法实现流程

*   **`demonstrations_to_graph_v2.py`**:
    *   **功能**: 整个流程的入口脚本（之一）。
    *   **主要逻辑**:
        1.  定义或读取演示路径 (`task_plans`)。
        2.  计算状态转换概率 (`construct_transition_with_probabilities`)。
        3.  调用 `StateVectorTaskGraphBuilder` 构建任务图。
        4.  使用 NetworkX 和 Matplotlib 可视化初始任务图。

*   **`lfd_trace_to_task_graph.py`**:
    *   **功能**: 负责将 LfD (Learning from Demonstration) 的轨迹数据转换为 NetworkX 的有向图。
    *   **核心类**: `StateVectorTaskGraphBuilder`。
    *   **逻辑**: 遍历轨迹中的 (状态, 动作) 对，创建图的节点和边，并记录状态转换。

*   **`task_graph_to_htn.py`**:
    *   **功能**: 核心算法实现，将扁平的任务图转化为分层的 HTN。
    *   **主要函数**:
        *   `create_init_htn_graph`: 初始化，将任务图的每个节点包装为 `PrimitiveNode`。
        *   `check_and_combine_htns_in_series`: 检测并合并串行结构（A -> B 变为 Sequence(A, B)）。
        *   `check_and_combine_htns_in_parallel`: 检测并合并并行结构（A -> B, A -> C, B -> D, C -> D 变为 A -> Choice(B, C) -> D）。
        *   `task_graph_to_htn`: 主循环，反复执行串行和并行合并，直到图收敛为一个根节点。

*   **`restructure_graph.py`**:
    *   **功能**: 处理复杂的图结构，当简单的串行/并行合并无法继续时使用。
    *   **逻辑**: 寻找“最小公共后继”(Least Common Successor) 和“可归约子图”，通过复制节点和展开图结构来消除非结构化的连接，使其能够被标准的 HTN 归约算法处理。这对应论文中处理 "Loop" 或复杂分支合并的部分。

## 3. 代码与论文的对应关系

| 代码文件 | 论文概念 | 说明 |
| :--- | :--- | :--- |
| `demonstrations_to_graph_v2.py` | Demonstrations $D$ | 输入的演示轨迹集合 |
| `lfd_trace_to_task_graph.py` | Task Graph Construction | 从轨迹构建初始的有向无环图 (DAG) 或带环图 |
| `task_graph_to_htn.py` | Algorithm 1: HTN Learning | 实现了自底向上的归约过程 (Algorithm 1) |
| `combine_htns_in_series` | Sequence Reduction | 识别 $n_1 \to n_2$ 结构并合并 |
| `combine_htns_in_parallel` | Choice Reduction | 识别 $n_1 \to \{n_2, n_3\} \to n_4$ 结构并合并 |
| `restructure_graph.py` | Handling Irreducible Graphs | 论文中并未详细展开所有工程细节，但对应于处理复杂依赖和偏好学习中的图展开逻辑 |

## 4. 可视化 Pipeline

```mermaid
graph TD
    A[演示数据 (Demonstrations)] -->|输入| B(demonstrations_to_graph_v2.py);
    B -->|构建| C{任务图 (Task Graph)};
    C -->|转换| D(lfd_trace_to_task_graph.py);
    D -->|生成 NetworkX 图| E[初始扁平图];
    E -->|输入| F(task_graph_to_htn.py);
    F -->|归约循环| G{检查结构};
    G -- 发现串行 --> H[合并串行节点];
    G -- 发现并行 --> I[合并选择节点];
    H --> G;
    I --> G;
    G -- 无法归约 --> J(restructure_graph.py);
    J -->|重构图| G;
    G -- 归约完成 --> K[最终 HTN 根节点];
    K -->|输出| L[htn.pkl / htn.dot];
```

## 5. 如何运行

通常的运行流程可能是：
1. 准备演示数据（在 `demonstrations_to_graph_v2.py` 中定义或加载文件）。
2. 运行 `demonstrations_to_graph_v2.py` 查看生成的任务图。
3. 使用 `task_graph_to_htn.py` 中的逻辑（通常被上层脚本调用）生成 HTN。
