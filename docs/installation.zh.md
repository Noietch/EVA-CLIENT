[← 返回 README](../README_zh.md) ｜ [English](installation.md)

# 安装

EVA 应用需要 **Python 3.11**（`>=3.11,<3.12`）。它的运行环境与 `examples/hardware/` 下各硬件 SDK 的环境相互独立。项目的 `uv` 平台配置以 Linux x86_64 为目标平台。

## 全新检出后的首次安装

先安装 [uv](https://github.com/astral-sh/uv)，然后运行：

```bash
git clone https://github.com/Noietch/EVA-CLIENT.git
cd EVA-CLIENT
uv venv --python 3.11 .venv
uv pip install --python .venv/bin/python -e ".[dev]"
```

以上命令假定 `.venv` 尚不存在。对于已有的检出，请复用其现有环境，不要重建，也不要运行同步、升级或重新安装类命令。在添加缺失的依赖之前，先检查安装计划，只有在现有包及其版本保持不变时才继续。请直接通过 `.venv/bin/` 运行 EVA 和各项检查。

目前 `pyproject.toml` 中声明的根项目 extra 只有：

| Extra | 内容 |
|-------|----------|
| `dev` | pytest、pytest-cov、Ruff、Pyright |
| `viz` | `fast-simplification`，用于网格简化 |

硬件 SDK、CAN 驱动、ROS 包和相机 SDK 都属于各自的硬件环境。在配置启动器之前，请先阅读[硬件指南](../examples/hardware/SUPPOTED_HARDWARE.md)以及对应机器人的 README。根项目没有 `franka`、`arx`、`ur5e` 或 `camera` 这些 extra。

## 通信层依赖

| 通信层 | 依赖 |
|-----------|--------------|
| `zmq` | 一个使用 EVA 传输协议的机器人、模拟硬件或仿真器节点；无需 ROS |
| `dataset` | 一个兼容的本地数据集；无需 ROS 或硬件节点 |
| `ros1` | 兼容的 ROS 1 安装，并且能够访问其 Python 模块 |
| `ros2` | 兼容的 ROS 2 安装，并且能够访问 `rclpy` 和消息包 |

即使机器人 SDK 运行在另一个进程中，ROS 依赖仍取决于所选的通信层。不要默认系统自带的 ROS Python 环境与 Python 3.11 兼容；请参照对应机器人的配置说明。

## 打开应用

在仓库根目录下运行：

```bash
.venv/bin/eva --language en
```

不指定 `--config` 时，EVA 会加载 `configs/00_base/defaults.py` 并打开 DEVICE 页。默认的浏览器地址为 `http://127.0.0.1:8080`。这只是启动应用本身；硬件就绪和策略推理需要各自另行配置。

如需离线数据集工作流，可使用自带的示例预设：

```bash
.venv/bin/eva --config configs/00_openloop/dual_agilex_piper_openloop.py
```

该预设指向 `examples/agilex_dataset`。可在 REPLAY 页查看录制的数据；DEBUG 页中的策略推理使用所配置的策略后端。接下来请继续阅读[快速开始](quick-start.zh.md)，了解设备、业务预设和无界面控制。
