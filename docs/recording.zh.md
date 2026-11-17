[← 返回 README](../README_zh.md) ｜ [English](recording.md)

# 录制

EVA 会把数据采集录制的 take 和策略 rollout 都保存为 LeRobot v2.1 数据集。
录制需要显式启用，各工作流分别决定自己的输出根目录。

| 工作流 | 数据集位置 |
| --- | --- |
| COLLECT | `<data_root>/<collection_dir>`；`collection_dir` 由任务集的 `info.yaml` 提供 |
| COLLECT（未设置 `collection_dir`） | `<data_root>/datasets/real_robot/<robot_type>/<dataset>` |
| EVAL | `<eval.output_dir>/<model_name>/episodes/raw` |
| RL / rollout | 该工作流配置的存储/日志目录 |

`collection.storage.data_root` 可覆盖采集根目录。未设置时，EVA 会根据外层的
`task_sets/` 目录推断，否则默认为 `datasets/data_collection`。`collection.storage.log_dir`
存放运行日志和界面状态，并不是采集数据集的根目录。任务集的发现规则和路径见
[场景规划](./collection_scene_plan.zh.md)。

一个典型的原生数据集包含：

```text
<dataset>/
├── data/chunk-000/episode_000000.parquet
├── videos/chunk-000/<video_key>/episode_000000.mp4
├── raw/<uuid>.mcap                         # SIM 执行记录（启用时）
└── meta/
    ├── info.json
    ├── episodes.jsonl
    ├── episodes_stats.jsonl                # 采集特征统计
    ├── tasks.jsonl
    ├── stats.json
    ├── qc.jsonl                           # 人工审核与自动质检台账
    └── render_reports/episode_000000.json  # 可选的 SIM 回放报告
```

只有当相应的采集/渲染路径实际产生了视频和 SIM 文件时，它们才会出现。请通过
`meta/info.json` 和每个 episode 的 `video_keys` 来确定实际包含的特征和媒体文件，
不要假定每个 episode 都包含所有相机。

## 采集与保存流程

真机数据采集会记录状态、下发的动作和相机采样，然后按配置的输出 FPS 对齐。
基础采集 schema 会写入 `observations.state.qpos`、`observations.state.eef`、`action.qpos`
和 `action.eef`；具体需要哪些列和相机由机器人/配置覆盖项决定。每一行还带有时间戳、
采集时间、帧/episode/任务索引以及全局索引。

对于由通信层驱动的真机数据采集，采集过程独立于主控制循环运行。保存使用一个有容量上限的后台队列，
队列满时会拒绝开始新的 take。真机数据采集的 STOP 会等待保存结果；SIM 的 STOP 则把校验通过的执行记录
放入后台队列进行准备，因此在认定其已提交之前，请先检查任务状态。STOP 会让机器人停留在当前位姿，
遥操作保持激活。CANCEL 会丢弃当前 take。

场景录制会在 episode 上附加 `task_id`、`scene_id`、从零开始计数的 `scene_round` 以及 `slot_id`。
对同一采集槽位重录时，新录制会以相同的 episode 索引替换之前的录制，并在保存事务中清除过期的审核/渲染产物。
未绑定槽位的录制会分配新的 episode 索引。评测尝试会携带其 clip/trial/模型标识以及保存的得分。
Rollout 录制还会保留人工干预标签和分段元数据。

<a id="eva-sim-execution-recording"></a>
## EVA-SIM 执行录制

> EVA-SIM 即将发布。下文所述的客户端录制路径已经包含在 EVA-Client 中；
> 仿真器及其采集配置将随 EVA-SIM 一同发布。

将 `collection.sim_recording_url` 设置为仿真器的 HTTP 端点（例如
`http://127.0.0.1:8093`）。SIM 采集配置会关闭实时相机录制，并将 `render_service.url`
留空，以便离线渲染。

START 时，EVA 要求已绑定仿真器场景元数据，读取 `/api/status`，锁定 `scene_revision`，
在数据集中预留 `raw/<uuid>.mcap`，然后调用 `/api/recording/start`。STOP 调用
`/api/recording/stop`；CANCEL 调用 `/api/recording/cancel`。仿真器必须能写入所提供的绝对路径，
客户端也必须能读取该路径。

EVA 会校验 MCAP 文件头、步序列、时序、维度、episode ID 以及完整的文件尾。原生行遵循“先观测后动作”语义：
第 `k` 行包含执行动作 `k` 之前的状态，第零行包含初始状态。最后一个动作执行后的状态保留在执行记录中。
FPS 由 `1 / control_dt` 推导；此路径不会像真实相机对齐那样对执行记录重采样。元数据会把原生 episode
关联到其 `sim_execution_path`、schema 版本以及 `sim_metadata`。

配置了渲染服务 URL 时，采集保存成功后会在后台发送通知。回放校验会读取 episode 显式指定的
`render_report` 路径或 `meta/render_reports/episode_<index>.json`。报告缺失视为待处理；
报告格式错误或任务评估未通过则判定为回放失败。成功采集到 MCAP 并不代表回放成功或任务完成。

## 采集质量与审核质检

采集过程会在 `episodes.jsonl` 中写入 `quality=green|red` 以及结构化的 `quality_issues`。
红色 take 会保留以供检查。检查项包括录制过短、缺少必需的相机流、图像重复、向量无效以及时序/对齐问题。
采集保存后，后台自动质检会运行，检查时长异常值、相机画面静止/卡顿模式，以及指令或观测到的机械臂关节限位；
结果存入 `meta/qc.jsonl`。

控制台和数据集服务共用以下状态：

| 状态 | 含义 |
| --- | --- |
| Pending / TO COLLECT | 该槽位还没有已保存的 take |
| Unreviewed | 已保存的 take 尚无通过/拒绝结论，或仍需审核 |
| Passed | 人工明确 PASS，接受该 take |
| Failed | 被人工 FAIL、采集红色标记、自动拒绝或必需的 SIM 回放失败所拒绝 |

人工 PASS 会覆盖采集、自动质检和 SIM 回放的失败结果。人工 FAIL 会拒绝该 episode。
没有 PASS 时，即使人工状态为 UNREVIEWED，自动质检失败和必需的回放失败仍然有效。
覆盖之后，回放诊断信息依然可见。仅凭采集质量为绿色并不代表质检通过。
请在 COLLECT 页或数据集审核服务中使用 PASS/FAIL/UNREVIEWED 和备注；这些操作写入质检台账，而不会修改源录制。

## 相机时序与冻结帧

YAM 观测携带的是每个相机最新缓存的帧，因此连续的观测可能复用同一次曝光。相机采样使用观测组装时刻
进行软件对齐。较小的对齐偏差并不能说明相机之间已经实现了曝光同步。断开相机会清空其缓存和预览。

设备界面中 YAM D405 + 两台 Gemini 305 的组合以 640×480、60 FPS 采集，相机转发和机器人观测均为 60 Hz。
原生录制仍使用配置的输出 FPS（通常为 30）。包含 Gemini 355 的组合使用 30 FPS。
更高的上游帧率可以减少采样间隙，但最新帧缓存并不会保留每一次曝光。

采集过程会用 `missing_camera_stream` 标记缺失的已配置相机流。对于完全重复的外部相机流，或机械臂发生运动
而画面重复的腕部相机流，会标记为 `frozen_camera`。`CameraSpec.attached_to` 指明相机所属的机械臂；
`None` 表示外部相机。该检查会忽略夹爪以及小于 0.02 rad 的机械臂运动。静止的腕部相机画面保持不变是正常的。
像素差异可能来自传感器噪声，因此单凭像素差异并不能说明场景存在有效运动。这些采集检查只作用于新保存的 take，
不会自动重新标记历史录制。

## 格式转换与交付

原生格式本身就是 LeRobot v2.1。COLLECT 页可将数据集转换为 `lerobot_v3`、`hdf5` 或 `mcap`，
输出到同级的 `<dataset_name>_<format>` 目录。重复转换会替换之前生成的输出。所有源 episode 都会被转换，
质检元数据也会保留；由下游使用方决定接受哪些 episode。转换并不是只导出 PASS 数据的导出操作。

Hugging Face 数据集、任务和质检数据的传输是数据集管理器中各自独立的操作。下载数据时会合并本地与云端的质检结果；
上传数据时不包含质检数据。可选的 SFTP 或 S3 交付在 `collection.storage` 中配置。
在认定交付完成之前，请检查后台任务结果和上传回执。
