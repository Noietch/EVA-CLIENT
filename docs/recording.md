[← Back to README](../README.md)

# 🎞️ Recording

Every run — teleop capture from **COLLECT** or model rollout from **DEBUG** /
**EVAL** — is written to the same LeRobot v2.1 dataset. The on-disk layout is:

```
<log_dir>/<task name>/
├── data/chunk-000/episode_000000.parquet     # 1 row per step, SFT-ready
├── videos/chunk-000/<camera>/episode_000000.mp4
└── meta/
    ├── info.json
    ├── episodes.jsonl                         # per-episode length, task, quality flag
    ├── episodes_stats.jsonl
    ├── tasks.jsonl
    └── stats.json                             # per-feature min/max/mean/std
```

Inference runs additionally carry a debug long-table sidecar (multiple raw
predictions per step for overlapping async / RTC chunks), joined to the main
table on `absolute_step`. Cameras are encoded as streaming mp4. Per-frame QC
marks each episode `green` or `red` with a reason (`episode_too_short`,
`non_monotonic_timestamp`, `missing_camera`, …); red episodes are still saved,
issues recorded in `episodes.jsonl`. Eval trials save on STOP; milestone scores
live in dataset metadata. Re-running the same eval prompt/trial overwrites that
episode in place — teleop always appends. A sample dataset lives at
`../examples/agilex_dataset/`.

### 相机断连与重复画面检查

YAM 每个观测周期携带各相机的最新缓存图像；相机尚未更新时可沿用上一帧。
图像使用观测组装时间参与软件对齐，因此对齐质量标记不代表多相机曝光同步。
相机连接断开时清空缓存及预览画面。

设备界面的 YAM 默认组合（D405 + 两台 Gemini 305）以 640×480、60 FPS 采集，
相机转发及机器人观测发布为 60 Hz，采集文件仍按 30 FPS 时间轴选帧保存。
提高上游频率用于减小取样间隔；当前缓存仍只保留最新图像，不能保证保留每次曝光，
也不能用观测时间偏差替代实际曝光时差。包含 Gemini 355 的组合仍使用 30 FPS。

保存采集数据时，下列问题会写入 `quality=red` 和 `quality_issues`：

- `missing_camera_stream`：配置的相机整段没有图像样本。
- `frozen_camera`：外部相机，或挂载在运动机械臂上的相机，整段图像 diff 全为零。

`robots.base.CameraSpec.attached_to` 指定相机挂载的执行器组，`None` 表示外部相机。
各机器人定义中的左右腕相机分别挂载到 `left_arm` / `right_arm`，UR5e 腕相机挂载到 `arm`。
按实际关节反馈判断运动，忽略夹爪维度及小于 0.02 rad 的关节变化，避免反馈噪声触发检查。
静止机械臂的腕相机允许图像不变。
像素 diff 用于检查整段重复图像；传感器噪声也可能产生非零 diff。
检查在新采集数据保存时执行，不会自动修改历史数据的质量标签。
