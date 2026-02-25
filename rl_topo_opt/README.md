# RL-based Topology Optimization

基于强化学习（PPO）的网络拓扑优化系统，使用 GNN 状态编码和多指标奖励函数实现自动化拓扑优化。

## 功能特性

- 使用 ATOP 生成初始网络拓扑
- 可选初始拓扑重构：保留节点并重建为全连接拓扑（full mesh）
- 基于 PPO 算法优化拓扑结构（动作：删除边）
- GNN（Graph Convolutional Network）状态编码
- 多指标奖励函数：cost、latency、fault tolerance
- 训练容错：每 100 步保存检查点，支持断点续训
- 动作掩码：避免无效动作

## 拓扑重构参数

- `--topology_rewrite_mode {none,full_mesh}`
  - `none`（默认）：保持 ATOP 原始拓扑
  - `full_mesh`：删除原有连接并将所有节点重建为完全图
- `--full_mesh_bandwidth FLOAT`
  - 仅在 `full_mesh` 模式生效，默认 `1.0`

## 依赖项目

本项目依赖以下两个外部项目，需要在 `rl_topo_opt` 目录的**同级目录**下克隆：

```bash
# 在 TopoDesign 目录下执行
cd /path/to/TopoDesign

# 克隆 ATOP（拓扑生成器）
git clone https://github.com/shadowmydx/ATOP.git

# 克隆 NeuroPlan（网络规划工具）
git clone https://github.com/netx-repo/neuroplan.git
```

克隆后的目录结构应为：
```
TopoDesign/
├── rl_topo_opt/       # 本项目
├── ATOP/              # ATOP 项目
└── neuroplan/         # NeuroPlan 项目
```

## 快速开始

详见 [QUICKSTART.md](./QUICKSTART.md)

## 环境安装（uv）

本项目使用 `uv` 管理虚拟环境与依赖，并通过 `requirements.lock` 复刻已验证环境。

```bash
cd rl_topo_opt
./install.sh
```

安装完成后可用以下命令验证：

```bash
.venv/bin/python -V
uv pip freeze --python .venv/bin/python
```
