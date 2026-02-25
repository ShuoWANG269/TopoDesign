# 仓库贡献指南

## 项目结构与模块组织
本仓库由三个相互关联的 Python 项目组成：
- `rl_topo_opt/`：强化学习拓扑优化主流程（入口 `main.py`，核心模块在 `env/`、`models/`、`training/`、`add01_full_mesh_topology/`，测试在 `tests/unit/`）。
- `ATOP/`：拓扑生成与 NSGA-II 相关实现（`generator/`、`NSGAII/`、`framework/`，测试在 `test/`）。
- `neuroplan/`：NeuroPlan 研究代码（`source/` 与 `spinningup/`）。

保持跨项目导入关系稳定：`rl_topo_opt` 默认要求 `ATOP` 和 `neuroplan` 与其处于同级目录。

## 构建、测试与开发命令
建议使用 Python 3.9+ 虚拟环境。
- `cd rl_topo_opt && ./install.sh`：安装运行与测试依赖。
- `cd rl_topo_opt && python main.py --n_gpus 8 --total_timesteps 1000`：快速训练冒烟验证。
- `python rl_topo_opt/test_basic.py`：基础转换与环境可用性检查。
- `pytest rl_topo_opt/tests/unit -q`：运行 RL 相关单元/集成倾向测试。
- `pytest ATOP/test -q`：运行 ATOP 工作流与评分器测试。

NeuroPlan 的专用环境准备请参考 `neuroplan/README.md`（含 Gurobi/C++ 编译步骤）。

## 编码风格与命名规范
- Python 代码采用 4 空格缩进，遵循 PEP 8；函数/变量用 `snake_case`，类名用 `PascalCase`。
- 新增公共 API 优先使用显式且带类型标注的函数签名（与 `rl_topo_opt` 现有风格一致）。
- 测试文件沿用既有命名模式，如 `rl_topo_opt/tests/unit/` 下的 `P0_test_*.py`、`P1_test_*.py`。
- 仓库未强制配置格式化工具；请保持改动最小，并与周边代码风格一致。

## 测试指南
- 测试框架统一使用 `pytest`（覆盖 `rl_topo_opt` 与 `ATOP`）。
- 对 converter、env、reward、model、training 的行为变更，必须同步新增或更新测试。
- 优先复用 `rl_topo_opt/tests/conftest.py` 中的 fixtures 构造拓扑与环境，避免重复搭建逻辑。
- 默认测试路径中避免引入超长训练测试，保持测试快速且可重复。

## 提交与 Pull Request 规范
- 提交信息建议遵循已出现的 Conventional Commit 风格：`docs: ...`、`chore: ...`、`feat: ...`、`fix: ...`。
- 每次提交尽量按模块聚焦（例如 `rl_topo_opt/env`、`ATOP/simulator`）。
- PR 需包含：变更目的、关键修改点、影响路径、已执行测试命令、已知限制。
- 若改动影响训练或评估行为，请附相关 issue/说明，并提供日志或截图作为证据。
