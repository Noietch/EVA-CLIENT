[← 返回 README](../README_zh.md) ｜ [English](concepts.md)

# 核心概念

## 设备进程与通信层

EVA-CLIENT 负责策略推理、操作员控制、录制和控制台。硬件 SDK 与相机采集运行在
[`examples/hardware/`](../examples/hardware/) 和
[`examples/input_sources/`](../examples/input_sources/) 下的独立进程中。各设备的
`config.yaml` 提供硬件默认值；Python 预设负责选择模型和工作流。控制台的 DEVICE
页负责设备选择、参数设置、启动和健康状态管理。

通信层读取观测并发布机器人指令。策略输出的末端目标会先转换为关节指令，再送往执行层。

| `transport.type` | 用途 | 要求 |
| --- | --- | --- |
| `ros1` | ROS 1 机器人话题 | ROS 1 环境 |
| `ros2` | ROS 2 机器人话题，支持压缩相机流 | ROS 2 环境 |
| `zmq` | 通过 EVA 的 ZeroMQ 观测/动作协议连接执行节点 | 匹配的执行节点 |
| `dataset` | 离线回放 LeRobot v2.1 episode | 本地数据集，且 state、action 和视频列已完成映射 |

使用 ZMQ 时，`transport.image_mode="stream"` 会消费流式图像。
`"on_demand"` 则在策略推理前发送图像请求，并等待新的图像版本，等待时长受
`image_request_timeout_s` 限制（默认 2 秒）。执行节点必须实现图像请求；超时后不会返回策略帧。

<a id="policy-backends"></a>
## 策略后端

EVA 观测包含 `state`（扁平的 float32 向量）、`images`（相机名到 uint8 数组的映射）和
`prompt`（字符串）。图像尺寸、RGB 转换、填充以及 HWC/CHW 布局都在调用策略之前配置，
且必须与模型的输入约定一致。策略返回形状为 `[T, D]` 的 `actions`：每行对应一个动作步，
`D` 由所配置的动作空间和机器人决定。

| `policy.type` | 连接方式 | 协议约定 |
| --- | --- | --- |
| `openpi` | WebSocket + OpenPI msgpack-numpy | 直接发送观测；当元数据声明 `supports_reset` 时使用远程重置 |
| `openpi_rtc` | WebSocket + OpenPI msgpack-numpy | 回传上一次响应的 `origin_actions` 或 `actions`，按 `backend_options.latency_k` 平移 |
| `starvla` | WebSocket + 带类型的 msgpack 封装 | 按观测顺序发送所有相机图像、语言指令和状态；可选 `unnorm_key` |
| `xpolicylab` | WebSocket + 带类型的 msgpack-numpy 帧 | `hello`、`infer` 和 `reset` 交互；显式声明关节/末端机械臂及夹爪维度 |
| `gr00t` | ZeroMQ REQ/REP + msgpack-numpy | 与本体相关的 `video_keys`、`state_key`、`language_key` 和 `action_keys` |
| `mock` | 本地 | 生成合成动作块，用于联调 |
| `replay` | 本地 | 使用数据集中录制的动作；需要 dataset 通信层 |

XPolicyLab 会发送指令、按名称组织的相机彩色图像数组以及按名称组织的状态字段。
对于两条关节空间机械臂，传输中的状态/动作顺序为：左臂关节、左夹爪、右臂关节、右夹爪。
末端空间机械臂每条使用七个值（`xyz + wxyz`），再加上所配置的夹爪开度。重置会创建新的
trial 标识，并重新开始推理步计数。预设说明见[配置](configuration.zh.md)。

## 推理策略

策略后端决定传输协议；推理策略决定动作块如何调度。DEBUG 页可以在已配置的推理策略之间切换。

| 键 | 行为 |
| --- | --- |
| `sync` | 阻塞式推理，然后执行截取后的动作块；默认执行长度为 5 步 |
| `async` | 后台推理，对重叠的动作块做线性平滑 |
| `naive` | 后台推理；每个新动作块直接替换缓冲中的轨迹 |
| `act` | 后台推理，对重叠预测做时间集成（temporal ensembling） |
| `rtc` | 后台推理，带延迟补偿和线性重叠；与 `openpi_rtc` 搭配可实现 RTC 服务端条件化 |

`execute_horizon` 用于截取动作块。`inference_cfg.inference_rate` 控制后台请求频率
（默认 3 Hz），`publish_rate` 控制动作发布频率（默认 30 Hz）。`latency_k` 控制后台推理策略的
前端裁剪量；`exp_weight_m` 控制 ACT 集成的衰减。`openpi_rtc` 后端的
`backend_options.latency_k` 与推理策略的 `args.latency_k` 是两个独立参数。

## 机器人与动作空间

机器人注册名包括 `agilex_piper`、`arx_r5`、`arx_x5`、`dual_franka`、
`dual_yam`、`r1lite`、`ur5e` 和 `agibot_g2`。硬件预设定义了关节组、夹爪、相机和末端连杆。
关节向量的顺序遵循机器人的 `state_composition`。

`inference_cfg.obs_space` 和 `action_space` 可分别独立选择 `JointState` 或 `EEFPose`。
末端空间的布局为每条机械臂 `xyz + rotation [+ gripper]`，并按 `n_arms` 依次拼接。
支持的旋转表示有 `quat`、`euler`、`rotvec` 和 `rot6d`；四元数默认为 `wxyz`，欧拉角默认为
`xyz`、单位为弧度。默认的四元数布局在带夹爪时每条机械臂 8 个值。改用其他表示或设置
`include_gripper=False` 会改变这一宽度。

末端空间的策略输出会被归一化为标准四元数位姿，再经 PyRoki IK 转换。`robot.eef_links`
指定目标连杆，因此即使向量宽度相同，TCP 预设和法兰预设也可能对应不同的物理目标。
请使用与模型训练约定相匹配的预设。

## 评测标识

评测配置会增加 `eval_cfg`，其中包含 checkpoint 选项、任务、尝试次数和存储设置。
EVAL 页也可以从任务集目录加载场景规划。感知场景的尝试会把 `task_set`、`task_id` 和
`scene_id` 绑定在一起，并随 trial 保存一份场景映射快照。这样即使源场景规划之后发生变化，
RESULT 页和 DASHBOARD 视图仍然与当次尝试的布局对应。详见
[配置](configuration.zh.md#evaluation-and-scene-plans)和
[场景规划格式](collection_scene_plan.zh.md)。
