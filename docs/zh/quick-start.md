[← 返回 README](../../README_zh.md) ｜ [English](../en/quick-start.md)

# 快速开始

请在仓库根目录下、使用已有的应用环境运行以下命令。
`eva` 入口的实现位于 [`src/main.py`](../../src/main.py)。

## 从 DEVICE 页开始

```bash
.venv/bin/eva --language en
# 中文控制台：
# .venv/bin/eva --language zh
```

`--config` 是可选参数。不传时，EVA 会加载共享默认配置并打开 **DEVICE** 页。
选择机器人、遥操作输入源、相机组合，以及真机或 Fake 硬件模式；先应用所选配置，
再启动所需的设备组件。确认各组件就绪后，再启用运动。

设备选择和参数覆盖会持久化到 `~/.local/eva-client/hardware_config.yaml`；
可通过 `EVA_WORKSTATION_PATH` 指定其他文件。加载业务预设时，已保存的选择同样会生效，
因此当当前机器人与预设名称不一致时，请检查 DEVICE 页。dataset 通信层的预设会保留其自身配置的机器人和数据集设置。

同样的设备操作流程也可以通过 CLI 完成：

```bash
.venv/bin/eva device list
.venv/bin/eva device config
.venv/bin/eva device status
.venv/bin/eva device logs
```

想体验软件模拟机器人，可在 DEVICE 页选择 ARX R5、不选相机，选择 Fake 模式，
应用后启动机器人组件。对应的命令如下：

```bash
.venv/bin/eva device select --robot arx_r5 --camera none
.venv/bin/eva device set robot.mode fake
.venv/bin/eva device start robot
```

Fake 模式使用共享的 ZMQ fake 节点，不需要实体机器人的 SDK 环境。它只提供合成观测；
真实的策略服务器是另一个独立组件。如需在本地演练推理流程，可在本地工作流预设中设置
`policy.type="mock"`。
用完后停止由 EVA 托管的 fake 进程：

```bash
.venv/bin/eva device stop robot
```

设备进程由一个独立运行的后台服务管理。关闭浏览器或退出 EVA 都不会停止它们；
请通过 DEVICE 页或 `eva device stop` 显式停止。

## 选择工作流预设

```bash
# 关节空间部署：DEBUG
.venv/bin/eva --config configs/01_deploy/dual_agilex_piper/openpi_qpos.py

# 末端空间部署，经 IK 转换为关节指令
.venv/bin/eva --config configs/01_deploy/ur5e/openpi_eef.py

# 遥操作数据采集：COLLECT
.venv/bin/eva --config configs/02_collection/dual_yam.py

# 离线 dataset 通信层，使用自带示例数据
.venv/bin/eva --config configs/00_openloop/dual_agilex_piper_openloop.py

# 多 checkpoint 评测：EVAL
.venv/bin/eva --config configs/03_evaluation/dual_agilex_piper_eval.py
```

使用前请检查预设中的路径、服务端点、任务集挂载以及模型/checkpoint 设置。
预设不会自动启动策略服务器或仿真器。
SIM 录制（EVA-SIM，即将发布）和场景规划分别见[录制](recording.md)和
[采集场景规划](collection_scene_plan.md)。

## 手动启动硬件

机器人执行节点与 EVA 运行在不同进程中。如果手动启动节点，请先启动节点再连接 EVA，
并避免再通过 DEVICE 页重复启动同一组件。请使用各机器人独立的 SDK 环境和启动脚本：

| 机器人 | 默认通信层 | 启动脚本 / 说明 |
|-------|-------------------|-------------------------|
| AgileX Piper | `ros1` | [`run_agilex_ros.sh`](../../examples/hardware/agilex_piper/run_agilex_ros.sh)，[说明](../../examples/hardware/agilex_piper/README.md) |
| Galaxea R1 Lite | `ros2` | [`run_hardware.sh`](../../examples/hardware/r1_lite/run_hardware.sh)，[说明](../../examples/hardware/r1_lite/README.md) |
| ARX R5 | `zmq` | [`run_hardware.sh`](../../examples/hardware/arx_r5/run_hardware.sh)，[说明](../../examples/hardware/arx_r5/README.md) |
| ARX X5 | `zmq` | [`run_hardware.sh`](../../examples/hardware/arx_x5/run_hardware.sh)，[说明](../../examples/hardware/arx_x5/README.md) |
| Dual Franka | `zmq` | [`run_hardware.sh`](../../examples/hardware/franka/run_hardware.sh)，[说明](../../examples/hardware/franka/README.md) |
| UR5e | `zmq` | [`run_hardware.sh`](../../examples/hardware/ur5e/run_hardware.sh)，[说明](../../examples/hardware/ur5e/README.md) |
| AgiBot G2 | `zmq` | [`node.py`](../../examples/hardware/agibot_g2/node.py)，[说明](../../examples/hardware/agibot_g2/README.md) |
| Dual YAM | `zmq` | [`run_hardware.sh`](../../examples/hardware/yam/run_hardware.sh)，[说明](../../examples/hardware/yam/README.md) |

Fake 模式下所有已注册机器人都使用 `zmq`。其相机和动力学都是合成的，不能说明实体硬件已经就绪。

## 浏览器与远程访问

启动日志会显示 Web 地址；默认为 `http://127.0.0.1:8080`，服务绑定在 `0.0.0.0`。
如需通过本地的另一个端口访问远程 EVA 控制台：

```bash
ssh -L 18080:127.0.0.1:8080 '<user>@<eva-host>'
# 在你的工作站上打开 http://127.0.0.1:18080。
```

策略流量使用其单独配置的端口。例如，EVA 在本地运行，而策略服务器在另一台主机上监听 9000 端口：

```bash
ssh -N -L 9000:127.0.0.1:9000 '<user>@<policy-host>'
```

对于该隧道，请设置 `policy.host="127.0.0.1"` 和 `policy.port=9000`。评测也可以通过
`eval_cfg` 托管 SSH 端口转发。

## 无界面模式与数据集工具

```bash
.venv/bin/eva --config configs/01_deploy/arx_r5/openpi_qpos.py --headless
```

无界面模式会关闭 HTTP 控制台和浏览器可视化。在 TTY 终端中，推理 CLI 提供
`task`、`setup`、`run`、`stop`、`reset` 和 `status` 命令。
无界面模式会自动启用 [ZMQ 控制通道](control-channel.md)；其配置的主机和端口也可供外部自动化程序使用。
启用浏览器时，如需开放该通道，请设置 `control_channel.enabled=True`。

独立的数据集浏览器不需要运行中的机器人或 EVA 控制台：

```bash
bash tools/datasets/start.sh --read-only \
  --plans-root <task_sets_dir> --assets-root <assets_dir> --collection-root <datasets_dir>
```

它监听 8412 端口，且需要显式指定数据根目录：可以通过命令行参数传入，也可以在通过
`--config` 指定的 Python 文件中写一个
`dataset_browser = dict(plans_root=..., assets_root=..., collection_root=...)` 配置段。
只读模式会禁用编辑和传输操作，但服务仍可能创建目录索引和预览缓存。数据布局和质检（QC）/导出流程见
[录制](recording.md)。

| 参数 | 说明 | 默认值 |
|----------|-------------|---------|
| `--config` | 业务 `.py` 预设 | 共享默认配置，进入 DEVICE 页 |
| `--web-port` | HTTP 控制台端口 | `8080` |
| `--headless` | 关闭 Web 控制台 | 关闭 |
| `--language`, `--locale` | 控制台语言，`en` 或 `zh` | `en` |

设备子命令有自己的 `--workspace` 和 `--url` 选项，需放在操作名之前：
`eva device --url http://127.0.0.1:8081 status`。
