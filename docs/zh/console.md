[← 返回 README](../../README_zh.md) ｜ [English](../en/console.md)

# 控制台

EVA 提供一个统一的浏览器控制台，用于设备配置、数据采集、策略推理、回放和评测。启动时需指定一个工作流配置：

```bash
.venv/bin/eva --config configs/02_collection/dual_yam.py
```

控制台共有八个页面。`console.initial_tab` 决定启动后的落地页；设为 `auto` 时，若存在评测配置则打开 EVAL 页，否则打开 DEBUG 页。采集配置通常会显式指定 COLLECT 页。可用的控件和工作流取决于所加载的机器人、策略、遥操作和评测配置。

| 页面 | 用途 |
| --- | --- |
| DASHBOARD | 采集 / 评测产出、场景筛选、质检计数、数据集传输以及与云端的对比 |
| DEBUG | 以 Full access 或 Ask for approval 模式推理；实时机器人、相机、模型选择以及本地执行历史 |
| DEVICE | 设备发现 / 启动、机器人与相机设置、遥操作控制以及手动关节控制 |
| COLLECT | 选择数据集和规划好的采集槽位，启用运动，录制遥操作 episode，回放、质检审核以及数据格式转换 |
| REPLAY | 加载已录制的数据集，并通过 dataset 通信层回放观测 / 动作 |
| RL | 策略 rollout，可选 critic、人工干预、保存控制以及 episode 回放 |
| EVAL | 选择模型、任务、场景和 trial；准备机器人、运行、停止并为里程碑打分 |
| RESULT | 浏览已保存的评测结果，同步回放视频、机器人运动和状态曲线 |

## Debug 会话与录制

DEBUG 页会在左侧边栏保留已执行的会话。默认本地根目录为 `~/.eva-client/`：会话元数据保存在 `sessions/<session-id>/session.json`，录制的 rollout 数据集保存在 `rollouts/<rollout-id>/`。点击 Stop 会保存当前执行；再次发送会继续同一次运行，并将新的录制追加进去。如需开启一次独立的运行，请使用侧边栏中的 **New run**。它会清空提示词，但保留模型、权限、策略和录制偏好。空草稿不会被保存。重启 EVA 后历史记录依然可用。

输入框中的圆形 Record 图标控制下一次执行是否录制。关闭 Record 时，历史记录会保存提示词、模型、权限模式、策略、时间戳、结果和动作数量。开启 Record 时，还会额外采集相机图像和动作，并在执行停止时自动保存。策略视频的采集帧率至少为录制 FPS，与动作发布频率相互独立。ZMQ 策略录制会采样最新一帧，而不是累积 FIFO 积压；完整的遥操作数据采集则保留 FIFO 采集方式。图像在编码前保持惰性加载，因此较长的录制也不会把所有解码后的像素都留在内存中。在 Ask for approval 模式下，每次批准都会执行待执行的动作块，并自动请求下一段预览。每个新的动作块都需要等待批准；Stop 会保存当前片段。继续录制的内容会作为同一数据集中的新 episode 出现，可通过 Replay 的 episode 控件选择。继续之前请等待保存完成。在执行过程中或存在待批准内容时，录制选项会被锁定。

Debug 权限可以在执行或预览期间更改。在 Full access 与 Ask for approval 之间切换时，会在下一个动作边界处暂停，保留任务、运行、准备状态以及已执行的步数，并丢弃之前的预览。按 Send 即可在新权限下继续；仅切换权限本身绝不会触发机器人运动，也不会让机械臂回到原位。

侧边栏是一个按时间排序的运行列表。点击一个已录制的运行，即可直接在现有的 REPLAY 工作区中打开。回放包括现有的时间轴、相机、3D 视图和曲线图。没有录制的执行会保留元数据，并标记为未录制。将鼠标悬停在某个运行上会显示红色的垃圾桶图标。删除运行会将其从列表中移除；**Undo** 可以恢复。被删除的录制文件仍可在 `~/.eva-client/trash/` 下找回。在另一个运行执行期间，已完成的运行也可以删除或恢复。当前活动的运行以及仍在保存中的录制会保持锁定。

手动夹爪锁定默认关闭。开关打开后，手动设置的张开 / 闭合状态会在模型执行期间保持不变；关闭开关会释放已有的锁定。在准备或 Reset 期间，输入框的主操作按钮会变为 Stop/Resume。`inference_cfg.default_strategy` 决定启动时的推理策略；异步模型请求会在后台以网络能达到的最高速率运行。Async/RTC 的线性重叠会对机械臂关节进行混合，而夹爪则直接使用经过延迟对齐的最新指令。如需同时混合夹爪，请在该策略的 `args` 中设置 `blend_gripper=True`；默认值为 `False`。夹爪所在的列根据当前机器人推导得出，而不是依赖固定的动作向量偏移。

```python
console = dict(
    session_history=True,
    history_dir="~/.eva-client",
    record_rollouts=False,  # 当前 EVA 进程的默认选择
)
```

`record_rollouts` 提供 Record 的初始选择；之后的执行可以在界面中更改。DEBUG 历史录制与数据采集、RL 和评测的录制设置相互独立。修改控制台配置后需要重启 EVA。

## 采集流程

先在 COLLECT 页选择一个数据集，再在场景 / 任务 / 轮次网格中选择一个采集槽位。挂载的[场景规划](./collection_scene_plan.md)为每个任务提供参考照片和目标数量。场景和任务筛选只改变显示的网格，采集槽位 ID 保持不变。内联的 `collection.tasks` 配置同样使用该选择器，此时只有一个 DEFAULT 场景，每个请求的 episode 对应一个采集槽位。目标数为 `-1` 的槽位始终可用，可重复采集。

启用 MOTION 以激活配置好的遥操作源。START RECORD 开始录制；STOP 结束录制，遥操作保持激活并停留在当前位姿。CANCEL 会丢弃当前这一条录制。执行 HOME 之前请先关闭 MOTION；遥操作处于待命或激活状态时，HOME 会被拒绝。保存队列已满时，无法开始新的录制。在将一条录制视为可用之前，请先确认其已保存 / 失败状态。

网格区分 TO COLLECT、UNREVIEWED、PASSED 和 FAILED 四种状态。采集完成数与质检通过数是两个不同的计数。选中一个已保存的 episode 进行回放，然后使用 PASS、FAIL、UNREVIEWED 以及备注控件。重新录制某个已规划的采集槽位，会以相同的 episode 索引替换该槽位之前的 episode，而不会额外增加一个已完成的目标。存储、采集检查和质检优先级详见[录制](./recording.md)。

EVA-SIM（即将推出）配置会在录制前绑定所选的仿真场景 / 任务。仿真器写出执行过程的 MCAP；EVA 会对其进行校验，并在后台生成原生数据集。相机渲染可能稍后才进行，因此刚保存的 SIM 录制一开始可能没有视频。

<a id="dataset-operations"></a>
## 数据集操作

DASHBOARD 页展示原始的采集和评测活动，支持按日期和场景筛选。其数据集管理器会列出当前采集配置所挂载的数据集、本地采集 / 质检计数、远端计数，以及数据 / 任务 / 质检的对比结果。选择数据集后，可分别上传或下载数据、任务定义或质检结果。Refresh cloud and verify 会执行远端检查和对比，但不会上传本地改动。

这些操作使用 `collection.storage.huggingface`，以后台任务的形式运行，提供进度、预计剩余时间（ETA）、逐数据集结果和停止控制。下载数据时，会把质检记录与本地记录合并，而不是直接覆盖台账。上传数据时不包含质检结果，因此需要通过单独的操作发布质检结果。对正在录制或保存中的数据集执行的操作会被拒绝，直到采集完成。独立的[数据集服务](../../tools/datasets/app.py)也使用同一套传输引擎。

COLLECT 页还提供 DOWNLOAD ASSETS、任务集同步、UPLOAD DATA 和格式转换。任务集同步涵盖所有已配置的任务集。原生格式为 LeRobot v2.1；转换会在同级目录下生成 LeRobot v3、HDF5 或 MCAP。转换会携带质检元数据，并包含所有源 episode；由下游使用方自行应用验收过滤。可选的 SFTP/S3 交付和 Hugging Face 同步在 `collection.storage` 下配置。

## 场景评测

EVAL 页使用 `eval_cfg.task_set_dir` 中配置的任务和任务集。对于场景任务，需要依次选择任务集、任务、场景和 trial。只有该任务引用的场景才有效。启动请求会将 `task_set`、`task_id` 和 `scene_id` 与一个不可变的 `clip_id` 绑定在一起；保存的 trial 会保留一份场景映射快照。若提供了仿真器元数据，则必须与所选的任务和场景一致。

点击 RUN 之前先准备机器人。STOP 并保存完成后，标记里程碑并保存得分 / 备注。打分与录制的片段、模型、场景和 trial 绑定；正在运行或尚未保存的尝试无法打分。重新进入 EVAL 页时会从尚未打分的工作继续。RESULT 页在对 trial 分组时会包含场景标识，并可打开任务参考照片、相机画面、机器人回放和曲线图。

<a id="rl-and-human-intervention"></a>
## RL 与人工干预

选择任务和策略，然后依次点击 SETUP 和 RUN。Critic 为可选项，选中后会提供实时价值遥测和回放评分。启用人工干预后，会暂停策略控制，由配置好的遥操作源接管。被接受的片段仍属于该 rollout 的一部分；放弃（abandon）则会沿配置的回滚路径返回。SAVE 会持久化该 rollout 及其干预元数据。

<p align="center">
  <video src="https://github.com/user-attachments/assets/62883016-1d5c-4688-9c39-740810b36b86" controls muted></video>
</p>

<a id="headless--low-power-mode"></a>
## 无界面 / 低功耗模式

```bash
.venv/bin/eva --config configs/01_deploy/dual_yam/openpi_qpos.py --headless
```

无界面模式会跳过 HTTP 服务、浏览器场景、MJPEG 流、额外的可视化读取器以及 RESULT 预览。通信层、策略、录制和状态机会继续运行。ZMQ [控制通道](./control-channel.md)会自动启用，使用 `control_channel.host` 和 `port`（默认 `127.0.0.1:5757`）。在此模式下 `--web-port` 不起作用。

当 stdin 是终端时，进程还会提供一个简单的 REAL 模式推理命令行：

```text
eva> info
eva> tasks
eva> task 0
eva> setup
eva> run
eva> status
eva> stop
eva> reset
```

`quit` 会退出该命令行，但 EVA 继续运行。通过管道或在后台运行的进程会跳过该命令行，可以通过 ZMQ 进行控制。状态转换和周期性的 `[HB]` 日志行会在日志中反映进度和错误。CLI 的 REAL 模式与“将真实通信层连接到 EVA-SIM（即将推出）进程”是两回事；DEBUG SIM 指的是在客户端场景中进行策略预览。

机器专属的 `*.local.py` 配置覆盖文件会被 Git 忽略。


## 显式内容来源

任务集和历史数据集只会从已配置的来源加载：

- COLLECT 场景规划：`collection.task_set_dir`；为空表示没有场景规划。
- EVAL 场景规划：`eval_cfg.task_set_dir`；为空表示没有任务集。
- Dashboard：`dashboard.raw_roots`；为空表示没有数据集。`work_dir` 和当前运行目录不会被自动加入。`raw_roots_only` 已不再需要。
- RESULT：已配置的 `eval_cfg.checkpoints`（对于已配置的评测，则为当前模型），以及在 `eval_cfg.result_models` 中显式列出的历史名称。其他模型目录不会被加载。
- 独立数据集浏览器：所选 `--config` 中的 `dataset_browser.plans_root`、`assets_root` 和 `collection_root`；三者均为必填项。

一个已配置的根目录可以包含多个数据集或任务集；只会在该根目录中搜索。
