# exp1 实验计划：基于 RL 的网络拓扑优化

## 需求概述

基于 ATOP 和 neuroplan 的参考代码，实现一个使用 PPO 算法优化网络拓扑的系统。

**实验参数：**
- GPU 数量：16
- 拓扑深度：2
- 拓扑宽度：3

**核心需求：**
1. 使用 ATOP 生成初始网络拓扑
2. 将 ATOP 拓扑转换为 NP 格式
3. 使用 PPO 进行拓扑优化（动作：删除边）
4. 奖励函数：基于 cost、latency、fault tolerance 三个指标（等权重）
5. 训练容错：每 100 步保存检查点，支持断点续训
6. 高性能：使用现代 RL 和 GNN 库

## 代码探索结果

### ATOP 关键实现
- **拓扑生成：** `ATOP/generator/network.py:construct_topology()`
- **数据结构：** `ATOP/NSGAII/solution.py:NetTopology`
  - `topology`: Topology 对象（nodes, layers）
  - `connection_blocks`: 层间连接参数
  - `blueprint`: 层内连接蓝图
- **三个指标计算：**
  - Latency: `ATOP/simulator/forestcoll_scorer.py:forestcoll_score()`
  - Cost: `ATOP/simulator/networkcost_scorer.py:network_cost()`
  - Fault Tolerance: `ATOP/simulator/faulttolerance_scorer.py:fault_tolerance_score()`

### neuroplan 关键实现
- **拓扑表示：** `neuroplan/source/topology/topology.py:Topology`
  - `ip`: Network 对象（links, routers）
  - `optic`: OpticNetwork 对象
- **RL 环境：** `neuroplan/source/rl/plan_env.py:PlanEnv`
  - 动作空间：Discrete(n_links × max_delta_bw) - 选择链路+增加带宽
  - 状态：Line Graph + GCN (edge2node_adj + edge_feature)
  - 奖励：-cost × norm_param，违反约束 -400/-1
- **RL 算法：** VPG (Spinning Up)，不是 PPO
- **状态转换：**
  - `get_edge2node_adj()`: 构建 Line Graph 邻接矩阵
  - `get_edge_feature()`: 提取边特征（带宽）
  - GCN 处理：`ac.py:GCN.forward()`

## 实现方案

### 1. 架构设计

创建新目录 `rl_topo_opt/`，保持与 ATOP 和 neuroplan 的独立性：

```
rl_topo_opt/
├── converter.py         # ATOP -> 简化图表示
├── env/
│   ├── topo_env.py      # RL 环境（删边动作）
│   └── reward.py        # 奖励函数（三指标整合）
├── models/
│   ├── gnn.py           # GNN 状态编码器
│   └── ppo_agent.py     # PPO 智能体
├── training/
│   ├── trainer.py       # 训练主循环
│   └── checkpoint.py    # 检查点管理
└── main.py              # 训练入口
```

### 2. ATOP 到 RL 环境的转换

**数据结构映射：**
- ATOP NetTopology → SimplifiedTopology
  - `nodes`: 节点 ID 列表
  - `edges`: 边列表（可删除的边）
  - `node_types`: {node_id: 'GPU'/'SWITCH'}
  - `edge_bandwidths`: 边带宽
  - `original_net_topo`: 保留原始数据用于指标计算

**转换逻辑（converter.py）：**
1. 从 GraphNode.siblings 提取所有边
2. 从 connection_blocks 和 blueprint 提取边带宽
3. 保留原始 NetTopology 引用（用于三指标计算）

### 3. RL 环境设计

**动作空间：**
- `Discrete(n_edges)` - 选择一条边删除
- 与 neuroplan 相反（neuroplan 是增加带宽）

**状态表示：**
- 节点特征矩阵 F: (n_nodes, feature_dim)
  - 节点类型（GPU=1, SWITCH=0）
  - 节点度数（归一化）
  - 节点层级
- 邻接矩阵 A: (n_nodes, n_nodes)

**Episode 终止条件：**
- 图不连通（惩罚 -1000）
- 边数过少
- 达到最大步数

### 4. 奖励函数设计

**三指标整合（等权重）：**
```python
reward = -(cost_norm + latency_norm + ft_norm) / 3
```

**指标计算复用 ATOP：**
- Cost: `ATOP/simulator/networkcost_scorer.py:network_cost()`
- Latency: `ATOP/simulator/forestcoll_scorer.py:forestcoll_score()`
- Fault Tolerance: `ATOP/simulator/faulttolerance_scorer.py:fault_tolerance_score()`

**关键：** 删边后需同步更新 NetTopology 的 siblings 集合

### 5. PPO 实现选择

**推荐方案：Stable-Baselines3 + PyTorch Geometric**

**理由：**
- SB3: 成熟的 PPO 实现，支持自定义策略网络
- PyG: 高性能 GNN 库，支持 GPU 加速
- 易于集成：SB3 允许自定义 ActorCriticPolicy

**GNN 策略网络（gnn.py）：**
- 使用 GCNConv 提取图特征
- 3 层 GCN：3 → 64 → 128 → 128
- 图级别池化：global_mean_pool

**动作掩码：**
- 避免删除关键边（导致图不连通）
- 修改 logits，将不可行动作的概率设为 -inf

### 6. 训练容错机制

**检查点内容（每 100 步保存）：**
- 模型参数（PPO 的 actor-critic 网络）
- 优化器状态
- 当前拓扑结构（SimplifiedTopology）
- 训练步数和指标
- 时间戳

**断点续训：**
- 加载检查点恢复模型、优化器、拓扑、训练步数
- 只保留最近 5 个检查点

### 7. 性能优化

**GNN 库：PyTorch Geometric**
- GPU 加速，批处理高效
- 稀疏图优化（COO 格式）

**并行化策略：**
- 使用 SB3 的 SubprocVecEnv 并行采样
- 创建 8 个并行环境（利用 8 卡 V100）

**指标计算优化：**
- 缓存机制：相同拓扑结构的指标结果缓存
- 异步计算：使用多进程池并行计算

## 关键文件清单

### 需要创建的新文件

1. **`rl_topo_opt/converter.py`**
   - ATOP NetTopology → SimplifiedTopology 转换

2. **`rl_topo_opt/env/topo_env.py`**
   - Gym 环境实现（删边动作空间）

3. **`rl_topo_opt/env/reward.py`**
   - 三指标整合奖励函数

4. **`rl_topo_opt/models/gnn.py`**
   - PyTorch Geometric GNN 特征提取器

5. **`rl_topo_opt/models/ppo_agent.py`**
   - 基于 SB3 的 PPO 智能体（支持动作掩码）

6. **`rl_topo_opt/training/trainer.py`**
   - 训练主循环（集成检查点保存）

7. **`rl_topo_opt/training/checkpoint.py`**
   - 检查点管理器

8. **`rl_topo_opt/main.py`**
   - 训练入口脚本

### 无需修改的现有文件（通过导入复用）

- `ATOP/generator/network.py` - 初始拓扑生成
- `ATOP/simulator/*.py` - 三指标计算

## 实现步骤

### Phase 1: 基础设施
1. 创建 `rl_topo_opt/` 目录结构
2. 实现 `converter.py`（ATOP → SimplifiedTopology）
3. 编写单元测试验证转换正确性

### Phase 2: RL 环境
1. 实现 `TopoEnv`（动作空间、状态表示、step 逻辑）
2. 实现 `reward.py`（整合三指标）
3. 测试环境的 Gym 接口兼容性

### Phase 3: GNN 模型
1. 实现 `GNNFeatureExtractor`（PyTorch Geometric）
2. 集成到 SB3 的 `ActorCriticPolicy`
3. 测试前向传播和梯度反向传播

### Phase 4: PPO 训练
1. 实现 `MaskedPPO`（支持动作掩码）
2. 实现 `CheckpointManager`（每 100 步保存）
3. 编写 `trainer.py` 主循环

### Phase 5: 优化与调试
1. 添加指标缓存和并行计算
2. 调整超参数（学习率、batch size、GNN 层数）
3. 在小规模数据（GPU=16）上验证收敛性

### Phase 6: 大规模训练
1. 部署到 Linux 服务器（8 卡 V100）
2. 使用 `SubprocVecEnv` 并行采样
3. 监控训练曲线和检查点

## 验证方法

### 单元测试
1. **转换器测试：** 验证 ATOP NetTopology → SimplifiedTopology 的正确性
   - 检查节点数、边数是否一致
   - 检查边带宽是否正确提取

2. **环境测试：** 验证 TopoEnv 的 Gym 接口
   - 测试 reset()、step()、render()
   - 检查动作空间和状态空间的维度

3. **奖励函数测试：** 验证三指标计算
   - 对比 ATOP 的原始计算结果
   - 检查归一化是否正确

### 集成测试
1. **端到端训练：** 在小规模数据上运行完整训练流程
   - GPU=16, 深度=2, 宽度=3
   - 训练 1000 步，检查是否收敛

2. **检查点测试：** 验证断点续训
   - 训练 500 步，保存检查点
   - 加载检查点，继续训练 500 步
   - 检查训练曲线是否连续

3. **性能测试：** 验证 GPU 利用率和训练速度
   - 监控 GPU 内存使用
   - 测量每步训练时间

## 用户确认的技术选型

- **PPO 框架：** Stable-Baselines3（成熟、易于集成 GNN）
- **GNN 库：** PyTorch Geometric（高性能、GPU 加速）
- **实现策略：** 完整实现所有功能（一次性实现所有模块）

## 依赖库

```bash
# 核心依赖
pip install torch==2.0.0
pip install torch-geometric==2.3.0
pip install stable-baselines3==2.0.0
pip install gym==0.26.0
pip install networkx==3.1
pip install numpy scipy

# ATOP 依赖
pip install sympy pandas

# 可选（监控）
pip install tensorboard wandb
```

## 潜在挑战与解决方案

### 挑战 1: 指标计算耗时
**解决方案：**
- 使用缓存避免重复计算
- 考虑用近似算法替代精确计算

### 挑战 2: 动作空间随拓扑变化
**解决方案：**
- 使用固定大小的动作空间（初始边数），删除的边标记为不可用

### 挑战 3: 奖励稀疏
**解决方案：**
- 添加中间奖励（如每删一条边给予小的负奖励）
- 使用 Reward Shaping（如基于图的连通性）

### 挑战 4: GNN 过拟合
**解决方案：**
- 使用 Dropout（在 GCN 层之间）
- 数据增强（随机初始化多个拓扑）
