[← 返回 README](../README_zh.md) ｜ [English](development.md)

# 开发

请使用 Python 3.11 和仓库中已有的 `.venv`。开发工具列在根项目的 `dev` extra 中。如果是全新检出的仓库，请按照[安装指南](installation.zh.md)操作。对于已有环境，只有在确认安装计划会保留所有已安装的版本和文件之后，才添加缺失的依赖。不要运行 `uv run`，不要重建环境，也不要使用同步 / 重新安装类命令。

## 检查

在仓库根目录下运行检查：

```bash
# 相关的测试层级；按改动影响的层级选择。
make test-unit
make test-integration
make test-static

# 全部 Python 测试，JUnit 输出位于 temp/test-results/ 下。
make test

# 只读的 Ruff 检查；CI 中运行的是同样的检查。
make lint

# 类型检查使用仓库 pyproject.toml 中的设置。
.venv/bin/pyright
```

Makefile 默认使用 `.venv/bin/python`。它会屏蔽 GPU 可见性、选用 CPU 版 JAX、将数值计算库的线程数限制为两个、使用 Matplotlib 的 `Agg` 后端，并禁用 pytest 插件自动加载。这些设置使检查不受特定机器上的 ROS pytest 插件以及共享 GPU 负载的影响。`PYTHON=/path/to/python make test-unit` 可以指定另一个已有的解释器。

对于小改动，可以只运行相关的测试，例如：

```bash
make test-unit PYTEST='.venv/bin/python -m pytest tests/unit/test_control_channel.py'
```

每个 Python 测试都声明一个层级标记：`unit`、`integration` 或 `static`。集成测试覆盖本地 HTTP/socket 服务、录制格式、模拟硬件、数据采集、评测和回放，但不能证明真实机器人、相机、仿真器、远程策略或发布行为的正确性。

## 源码地图

| 模块 | 位置 |
|------|----------|
| CLI 入口 | `src/main.py` |
| 运行时、状态机、命令分发 | `src/core/app/` |
| 设备目录、工作站覆盖配置、独立服务 | `src/core/devices/` |
| Python 预设的加载与校验 | `src/core/config.py`, `src/core/cfg/` |
| 录制、SIM 导出、元数据、质检 | `src/core/recorder/` |
| 浏览器 UI 与 HTTP API | `src/core/app/console/` |
| 机器人模型与运动学 | `src/robots/` |
| 通信层 / 策略 / 推理策略 | `src/transport/`, `src/policy_client/`, `src/strategy/` |
| 数据集浏览、审核、传输 | `tools/datasets/` |
| LeRobot v3.0、MCAP、HDF5 转换 | `tools/conversion/` |
| 硬件与遥操作默认配置 | `examples/hardware/*/config.yaml`, `examples/input_sources/*/config.yaml` |

各硬件 SDK 项目有各自独立的 Python 环境。根项目的 Ruff 配置排除了内置的 ARX SDK 目录；请避免格式化第三方 SDK。如果你使用 pre-commit，请注意其配置中包含会修改文件的 hook；它与 `make lint` 相互独立，且 `pre-commit` 不在根项目的 `dev` extra 中。

对于仅涉及文档的改动，请对照源码检查所引用的路径、配置名、链接和命令约定。只有在改动的行为确实需要时，才需要运行运行时和硬件测试。
