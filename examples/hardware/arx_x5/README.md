# ARX X5 双臂真机启动

该目录用于启动双臂 X5、电机 CAN、三台 Intel RealSense D405 和 EVA ZMQ
硬件节点。X5 与 R5 使用相同的状态、动作和 VR 控制流程；差异仅为 X5 SDK
和 X5A URDF。

## 固定配置

启动脚本已经按当前机器写死以下配置，不需要设置环境变量：

```text
左臂:       /dev/arxcan1 -> can1
右臂:       /dev/arxcan3 -> can3
X5 类型:    2（2025 双轨夹爪）
状态端口:   tcp://127.0.0.1:5555
动作端口:   tcp://127.0.0.1:5556
相机:       640x480 BGR8, 30 FPS
网页端口:   8080
```

D405 使用固定序列号，不依赖 SDK 枚举顺序：

```text
cam_high:        409122271504
cam_left_wrist:  352122272510
cam_right_wrist: 352122271326
```

三台 D405 每次启动都会自动加载 `d405_profile.yaml`。启动脚本默认使用 `day`
profile：关闭自动曝光，high/left/right 分别固定曝光 `6000/8000/7000 us`；
`--realsense-profile night` 可切回夜间的 `12000/14000/14000 us`。两套 profile
都使用画板白色标定得到的白平衡 high `4390 K`、left `4450 K`、right `4410 K`，
并加载各相机的小幅 BGR 校正增益。

D405 的自动白平衡可以工作，但当前白桌面/画板占据画面较大时，自动曝光会把
约 68%～86% 的像素推到饱和区，因此 X5 默认不使用自动曝光，避免白板和桌面
过曝。采集环境改变时应在 `d405_profile.yaml` 中重新标定对应的昼/夜值。

## 首次安装

X5 使用独立的 Python 3.12 项目和虚拟环境，不修改仓库根环境或 R5。执行：

```bash
git submodule update --init --recursive
bash examples/hardware/arx_x5/setup_env.sh
```

`run_hardware.sh` 和 `run_fake.sh` 都只使用
`examples/hardware/arx_x5/.venv/bin/python`，不会激活或读取仓库根目录的
`.venv`。如果 `python3.12` 不在 `PATH` 中，可在安装时显式指定解释器：

```bash
ARX_X5_PYTHON=/path/to/python3.12 bash examples/hardware/arx_x5/setup_env.sh
```

脚本会自动完成以下操作：

1. 创建 `examples/hardware/arx_x5/.venv`。
2. 安装 `pyrealsense2` 和 X5 硬件节点依赖。
3. 使用仓库内置的官方
   [ARARX_X5_beta SDK-V2](https://github.com/ARXroboticsX/ARARX_X5_beta)，当前同步到
   `1aa8a7d2ddefced2229f953feffc248be1c0b44d`，路径为
   `examples/hardware/arx_x5/SDK/X5`。
4. 校验官方 CPython 3.12 二进制和 `SingleArm` 接口，并安装 `pin`、
   `python-can`、`ruckig` 等 vendor import 所需依赖。安装过程不执行 FK/IK
   计算，也不连接机器人。

EVA Client 中新增的 X5 集成代码遵循仓库根目录的 Apache-2.0 许可证。
`ARARX_X5_beta` 是独立的第三方 submodule，不在 EVA Client 的 Apache-2.0
授权范围内；固定版本目前未包含许可证文件，公开使用或分发前需向上游确认授权。

SDK-V2 直接使用 SocketCAN 和自己的 200 Hz 控制线程，不再依赖旧版 ROS/KDL
pybind 扩展。X5 FK、IK 和 VR 重定向仍使用仓库自己的
`src/robots/kinematics/pyroki.py` 和 X5A URDF。

## 启动 X5 hardware

首次安装完成后只需一个命令：

```bash
bash examples/hardware/arx_x5/run_hardware.sh
```

脚本只启动双臂电机、三台 D405 和硬件 ZMQ 节点。按 `Ctrl-C` 时机械臂进入
保护模式并释放相机和 ZMQ 端口。WebXR 与 EVA 需要在另外两个终端单独启动：

每次启动会先使用 SDK-V2 的 CAN 总线，只向 X5-2025 夹爪电机 ID 8 发送厂商
`clear error` 帧并校验故障码为 0；不会改夹爪零点，也不会在这个步骤向六轴电机
发送位置命令。若夹爪故障仍存在或没有反馈，hardware node 不会继续启动。

```bash
python examples/input_sources/vr_webxr/node.py \
  --host 127.0.0.1 \
  --port 43876 \
  --endpoint tcp://127.0.0.1:8765 \
  --ack-endpoint tcp://127.0.0.1:8766
```

```bash
eva --config configs/02_collection/arx_x5_vr.py
```

浏览器打开 `http://127.0.0.1:8080`。头显通过本机或 ADB 转发访问：

```text
http://127.0.0.1:43876/?token=<TOKEN_FROM_NODE_LOG>&mode=ar
```

## 无硬件 Fake X5

Fake X5 使用与真机相同的 EVA ZMQ action/state 协议，不需要 X5 SDK、CAN 或
RealSense。它默认将收到的 action qpos 直接设置为 state，并为三路相机发送周期
颜色图像。

在仓库根目录启动 fake node：

```bash
bash examples/hardware/arx_x5/run_fake.sh
```

需要模拟较平滑的机械响应时，可以切换回每个 qpos 独立的二阶系统：

```bash
bash examples/hardware/arx_x5/run_fake.sh --dynamics-mode second-order
```

终端输入 `r` 后回车可将 qpos、速度和 action target 重置到 X5 初始值；输入 `q`
后回车退出。Fake plant 的迭代生命周期不受 EVA 的 ARM、Collect 或录制状态控制。

另一个终端按上文方式启动 WebXR node，然后启动 EVA：

```bash
eva --config configs/02_collection/arx_x5_vr.py
```

该配置保留全部三路 camera，采集结果写入
`work_dirs/collection/arx_x5_vr`。

## 启动行为

X5 `_connect` 连接后先读取当前 6 轴位置，再原样回写当前值进入位置控制。随后
硬件节点通过 SDK-V2 的轨迹接口，用 `5 s` 让两臂同步到达初始位。实时动作以约 `33 ms`
duration 交给官方 Ruckig/200 Hz 控制线程插值；EVA 仍以 30 Hz 发布并记录 qpos
动作目标。J2/J3 的下限使用官方物理范围 `0°`，不会再把 IK
负裕量持续压在机械零位。默认初始位是官方 X5 home 位；需要全 0 初始位时，使用下方的
`--start-at-zero` 参数。

停止时按 `Ctrl-C`，节点会进入保护模式并释放相机。需要临时跳过某只机械臂时
可使用：

```bash
bash examples/hardware/arx_x5/run_hardware.sh --disabled-arm left_arm
```

若启动时有单关节不上电，执行故障恢复启动：

```bash
bash examples/hardware/arx_x5/reset_hardware.sh
```

该脚本会先停止残留的 hardware node，重建左右臂 `slcand`/CAN 接口，并分别执行
SDK 使能检查和完整 `disable`/`close`。右臂还会用 `0.01 rad` 微动验证 J2
确实响应。默认最多重试 3 次；只有两臂的 `offline_joints` 为空、`fault` 为
`None`，且右 J2 微动通过，才会继续启动和回零。可用 `ARX_X5_RESET_ATTEMPTS`
调整次数：

```bash
ARX_X5_RESET_ATTEMPTS=5 bash examples/hardware/arx_x5/reset_hardware.sh
```

如果需要让双臂在节点启动时移动到六轴全 0 的位置，可使用：

```bash
bash examples/hardware/arx_x5/run_hardware.sh --start-at-zero
```

默认启动仍移动到官方 X5 home 位。

查看参数不会连接电机或初始化 CAN：

```bash
bash examples/hardware/arx_x5/run_hardware.sh --help
```
