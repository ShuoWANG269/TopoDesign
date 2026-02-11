# 快速开始指南

## 安装依赖

### 方法 1：使用安装脚本（推荐）

```bash
cd rl_topo_opt
chmod +x install.sh
./install.sh
```

### 方法 2：手动安装

```bash
cd rl_topo_opt
pip3 install -r requirements.txt
```

## 训练命令

### 快速训练（测试用）

```bash
python main.py --n_gpus 8 --total_timesteps 1000
```

### 完整训练

```bash
python main.py \
    --n_gpus 16 \
    --depth 2 \
    --width 3 \
    --total_timesteps 100000 \
    --checkpoint_interval 100 \
    --learning_rate 3e-4 \
    --tensorboard_log ./logs
```

### 断点续训

```bash
python main.py --resume ./checkpoints/checkpoint_step_1000_*.pkl
```

### 仅评估

```bash
python main.py \
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
