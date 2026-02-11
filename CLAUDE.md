# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

TopoDesign 是一个网络拓扑优化系统，包含两个核心组件：

1. **ATOP** (`ATOP/`) - 基于 NSGA-II 进化算法的网络拓扑生成与优化框架
2. **rl_topo_opt** (`rl_topo_opt/`) - 基于强化学习（PPO + GNN）的拓扑优化系统

优化目标：最小化成本（cost）、延迟（latency），同时最大化容错性（fault tolerance）。

## Development Commands

```bash
# 安装依赖
cd rl_topo_opt
pip install -r requirements.txt

# 快速训练（测试用）
python main.py --n_gpus 8 --total_timesteps 1000

# 完整训练
python main.py --n_gpus 16 --depth 2 --width 3 --total_timesteps 100000

# 断点续训
python main.py --resume ./checkpoints/checkpoint_step_1000_*.pkl

# 仅评估
python main.py --eval_only --resume ./checkpoints/checkpoint_step_100000_*.pkl

# TensorBoard 监控
tensorboard --logdir ./logs

# 运行测试
pytest
```

## Architecture

### ATOP 模块 (`ATOP/`)

- **generator/network.py** - 拓扑生成器，使用 block-wise 结构生成 GPU-Switch 层级拓扑
- **NSGAII/** - NSGA-II 进化算法实现（selection, mutation, fitness）
- **simulator/** - 评估函数：networkcost_scorer, forestcoll_scorer, faulttolerance_scorer
- **framework/optimizer.py** - 进化算法框架

拓扑数据结构：
```python
NetTopology:
  - topology.nodes: {node_id -> GraphNode}
  - topology.layers: [layer0, layer1, ...]  # layer0=GPU, others=SWITCH
  - connection_blocks: {(i,j): {'i': blocks_i, 'j': blocks_j, 'e_ij': links, 'b_ij': bandwidth}}
  - blueprint: {layer_idx: {'Di': dim, 'Sik': [...], 'Bii': bandwidth}}
```

### rl_topo_opt 模块 (`rl_topo_opt/`)

- **main.py** - 训练入口，调用 ATOP 生成初始拓扑
- **converter.py** - ATOP 拓扑 → 简化拓扑格式转换
- **env/topo_env.py** - Gym 环境，动作为删除边
- **env/reward.py** - 多指标奖励函数（cost/latency/fault_tolerance 等权重）
- **models/gnn.py** - 3层 GCN 特征提取器
- **models/ppo_agent.py** - Stable-Baselines3 PPO + 动作掩码
- **training/trainer.py** - 训练循环 + 检查点管理

数据流：
```
ATOP generator → NetTopology → converter.py → SimplifiedTopology
                                          ↓
                          Gym env reset/step → GNN encode → PPO action
                                          ↓
                          reward computation (3 metrics)
```

## Key Constraints

- PPO 动作空间：离散选择删除某条边，动作掩码防止拓扑断连
- 奖励函数三个指标权重相等（各 1/3）
- 检查点每 100 步自动保存，最多保留 5 个
- 拓扑始终保持连通（BFS 检查）
