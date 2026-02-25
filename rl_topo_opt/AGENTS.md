# 仓库贡献指南

## 项目结构与模块组织
`rl_topo_opt` 是强化学习拓扑优化主模块。
- `main.py`：训练与评估入口。
- `env/`：环境、奖励与状态相关逻辑。
- `models/`：GNN 与 PPO agent 模型定义。
- `training/`：训练器与 checkpoint 管理。
- `add01_full_mesh_topology/`：初始拓扑重写策略（如 `full_mesh`）。
- `tests/unit/`：按 P0-P3 分层的测试集合。

注意：该模块默认依赖同级目录的 `ATOP/` 与 `neuroplan/`，请保持目录关系不变。

## 构建、测试与开发命令
建议先在 `rl_topo_opt/` 下创建并激活 Python 3.9+ 虚拟环境。
- `./install.sh`：安装核心依赖与 `pytest`。
- `python main.py --n_gpus 8 --total_timesteps 1000`：快速训练冒烟测试。
- `python main.py --eval_only --resume ./checkpoints/<ckpt>.pkl`：仅评估已训练模型。
- `python test_basic.py`：基础连通性/环境流程检查。
- `pytest tests/unit -q`：运行主要单元测试。

## 编码风格与命名规范
- 使用 4 空格缩进，遵循 PEP 8。
- 函数与变量使用 `snake_case`，类名使用 `PascalCase`。
- 新增公共函数建议补充类型标注与简短 docstring。
- 修改代码时保持小步提交，避免无关重构。

## 测试指南
- 测试框架统一为 `pytest`。
- 涉及 `converter`、`env`、`reward`、`models`、`training` 的改动必须补测试。
- 优先复用 `tests/conftest.py` 里的 fixture，避免重复构造拓扑与环境。
- 默认测试应可重复且耗时可控，避免把长时间训练放入常规测试路径。

## 提交与 Pull Request 规范
- 提交信息建议使用 Conventional Commit：`feat:`、`fix:`、`docs:`、`chore:`。
- 每次提交聚焦单一主题，例如“仅修改 `env/reward.py` 与对应测试”。
- PR 描述需包含：变更目标、影响范围、执行过的测试命令、已知风险。
- 涉及训练行为变化时，附关键日志（如 reward 曲线、checkpoint 信息）。
