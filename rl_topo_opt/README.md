# RL-based Topology Optimization

基于强化学习（PPO）的网络拓扑优化系统，使用 GNN 状态编码和多指标奖励函数实现自动化拓扑优化。

## 功能特性

- 使用 ATOP 生成初始网络拓扑
- 基于 PPO 算法优化拓扑结构（动作：删除边）
- GNN（Graph Convolutional Network）状态编码
- 多指标奖励函数：cost、latency、fault tolerance
- 训练容错：每 100 步保存检查点，支持断点续训
- 动作掩码：避免无效动作

## 快速开始

详见 [QUICKSTART.md](./QUICKSTART.md)
