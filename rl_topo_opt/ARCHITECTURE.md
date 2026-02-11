# RL Topology Optimization - 架构设计

## 项目结构

```
rl_topo_opt/
├── converter.py              # ATOP → SimplifiedTopology 转换器
├── main.py                   # 训练入口脚本
│
├── env/                      # RL 环境模块
│   ├── __init__.py
│   ├── topo_env.py          # Gym 环境（删边动作）
│   └── reward.py            # 奖励函数（三指标整合）
│
├── models/                   # 模型模块
│   ├── __init__.py
│   ├── gnn.py               # GNN 特征提取器（GCN 3层）
│   └── ppo_agent.py         # PPO 智能体（动作掩码）
│
├── training/                 # 训练模块
│   ├── __init__.py
│   ├── trainer.py           # 训练主循环
│   └── checkpoint.py        # 检查点管理
│
├── requirements.txt          # 依赖列表
├── install.sh               # 安装脚本
└── test_basic.py            # 基本功能测试
```

**核心文件说明：**
- `converter.py`: 负责 ATOP 拓扑的转换和简化
- `env/topo_env.py`: 实现 Gym 环境接口，动作为删除边
- `env/reward.py`: 整合 cost、latency、fault tolerance 三个指标
- `models/gnn.py`: 使用图卷积网络编码拓扑状态
- `models/ppo_agent.py`: PPO 算法实现，支持动作掩码
- `training/trainer.py`: 训练循环和评估
- `training/checkpoint.py`: 定期保存训练检查点

## 模块说明

### 转换器（converter.py）

将 ATOP 生成的 NetTopology 转换为 SimplifiedTopology：
- 提取节点、边、节点类型、边带宽的映射关系
- 实现 `is_connected()` 检查拓扑的连通性（BFS 算法）
- 支持边的增删操作

### 环境模块（env/）

**topo_env.py - Gym 环境**
- 状态：节点特征 + 邻接矩阵 + 动作掩码
- 动作：选择一条边删除（离散动作空间，大小为边数）
- 终止条件：拓扑断连、边数过少、达到最大步数
- 奖励：-1000（断连）或三指标综合奖励

**reward.py - 奖励函数**
- 整合三个指标：成本（cost）、延迟（latency）、容错性（fault_tolerance）
- 三个指标权重相等（各 1/3）
- 缓存机制避免重复计算

### 模型模块（models/）

**gnn.py - GNN 特征提取器**
- 三层图卷积网络（GCN）
- 层大小：3 → 64 → 128 → 128
- 包含批归一化和 dropout（0.2）
- 全局平均池化输出图级特征

**ppo_agent.py - PPO 智能体**
- 基于 Stable-Baselines3 的 PPO 实现
- 自定义 policy 支持 GNN 特征提取和动作掩码
- Policy network 和 value network 共享 GNN 编码器

### 训练模块（training/）

**trainer.py - 训练主循环**
- 数据收集、模型更新、评估功能
- 集成 checkpoint 保存回调
- 支持渐进式训练和断点续训

**checkpoint.py - 检查点管理**
- 每 N 步自动保存模型、训练状态、指标
- 自动清理旧检查点，最多保留 5 个
- 支持从检查点恢复训练

## 工作流程

### 1. 拓扑生成与转换

```
ATOP 生成器
    ↓
NetTopology（ATOP 格式）
    ↓
converter.py 转换
    ↓
SimplifiedTopology（简化格式）
```

### 2. RL 训练循环

```
初始化拓扑
    ↓
Gym 环境 reset → 状态观测
    ↓
GNN 编码拓扑状态
    ↓
PPO 选择删除边的动作
    ↓
环境 step → 新状态 + 奖励
    ↓
计算三指标奖励
    ↓
PPO 更新网络
    ↓
每 100 步保存检查点
    ↓
重复直到训练完成
```

### 3. 数据流

- **输入**：ATOP 拓扑（节点数、边数、连接关系）
- **处理**：GNN 编码 → PPO 决策 → 边删除 → 奖励计算
- **输出**：优化后的拓扑、训练日志、检查点

## 关键设计说明

### 1. 为什么使用 GNN

- 拓扑是图结构，直接用向量编码会损失结构信息
- GNN（特别是 GCN）能保留图的邻接关系和节点特征
- 通过多层卷积，能学习到拓扑的全局结构信息

### 2. 奖励函数设计

三指标均衡优化：
- **成本（cost）**：降低硬件成本
- **延迟（latency）**：减少通信延迟
- **容错性（fault_tolerance）**：提升容错能力

三个指标权重相等，避免单指标优化导致的不平衡。

### 3. 动作空间与动作掩码

- **动作**：离散选择，从当前拓扑的所有边中选一条删除
- **掩码**：在删边前检查，删除该边是否导致断连
  - 若是，将该动作标记为无效（mask = 0）
  - PPO 会避免选择无效动作

### 4. 训练容错机制

- 每 100 步保存一次检查点（模型、优化器状态、训练步数）
- 自动保留最近 5 个检查点，删除更早的版本
- 支持从任意检查点恢复训练，无需从头开始

### 5. 为什么边删除是优化目标

- 拓扑从初始状态出发，通过逐步删除低效的边来优化
- 删除操作具有明确的物理意义：减少硬件成本
- 动作掩码保证拓扑始终连通，满足基本可用性要求
