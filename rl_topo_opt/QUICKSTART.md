# 快速开始指南

## 安装依赖

### 前置条件

```bash
uv --version
```

若未安装 `uv`，请先参考官方文档安装：<https://docs.astral.sh/uv/getting-started/installation/>

### 方法 1：使用安装脚本（推荐）

```bash
cd rl_topo_opt
./install.sh
```

该脚本会：
- 使用 Python 3.9 创建/复用 `.venv`
- 按 `requirements.lock` 精确同步依赖

### 方法 2：手动使用 uv（等价）

```bash
cd rl_topo_opt
uv venv --python 3.9 --allow-existing .venv
uv pip sync --python .venv/bin/python requirements.lock
```

### 安装后验证

```bash
.venv/bin/python -V
uv pip freeze --python .venv/bin/python
.venv/bin/python test_basic.py
```

### 维护锁文件（仅在需要升级依赖时）

```bash
cd rl_topo_opt
uv pip freeze --python .venv/bin/python > requirements.lock
```

## 训练命令

### 快速训练（测试用）

```bash
.venv/bin/python main.py --n_gpus 8 --total_timesteps 1000
```

### 完整训练

```bash
.venv/bin/python main.py \
    --n_gpus 16 \
    --depth 2 \
    --width 3 \
    --total_timesteps 100000 \
    --checkpoint_interval 100 \
    --learning_rate 3e-4 \
    --tensorboard_log ./logs
```

### 使用全连接初始拓扑训练

```bash
.venv/bin/python main.py \
    --n_gpus 16 \
    --depth 2 \
    --width 3 \
    --topology_rewrite_mode full_mesh \
    --full_mesh_bandwidth 1.0 \
    --total_timesteps 100000
```

说明：
- 默认 `--topology_rewrite_mode none`，保持原始 ATOP 拓扑
- 启用 `full_mesh` 后，会保留节点并把初始边重建为全连接（完全图）

### 断点续训

```bash
.venv/bin/python main.py --resume ./checkpoints/checkpoint_step_1000_*.pkl
```

### 仅评估

```bash
.venv/bin/python main.py \
    --eval_only \
    --resume ./checkpoints/checkpoint_step_100000_*.pkl \
    --n_eval_episodes 20
```

## 监控与调试

### TensorBoard 实时监控

```bash
tensorboard --logdir ./logs
```

然后打开浏览器访问：`http://localhost:6006`

可以查看：
- 训练曲线（奖励、损失）
- 学习率变化
- PPO 算法的详细指标

### 检查点管理

检查点自动保存在 `./checkpoints` 目录下，每 100 步保存一次。最多保留 5 个最近的检查点。

查看检查点：
```bash
ls -lh ./checkpoints
```
