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

实时预览按每路相机的真实收帧标识判断新鲜度，不使用 MJPEG 重发或画面像素变化作为心跳。
连续 1 秒没有新帧时，对应相机保留位置并显示红框和告警文字；恢复收帧后自动恢复。
界面以约 200 ms 周期轮询，因此告警在超过 1 秒后的下一次轮询显示。
YAM 的 RealSense / Orbbec 缓存只在实际采集到帧时更新标识，重复读取缓存不会计为新帧。

保存采集数据时，下列问题会写入 `quality=red` 和 `quality_issues`：

- `missing_camera_stream`：配置的相机整段没有图像样本。
- `camera_frame_timeout`：某路相机连续至少 1 秒没有新帧，包括采集开头、末尾和断线后恢复的间隔。
- `frozen_camera`：外部相机，或挂载在运动机械臂上的相机，整段图像 diff 全为零。

`robots.base.CameraSpec.attached_to` 指定相机挂载的执行器组，`None` 表示外部相机。
各机器人定义中的左右腕相机分别挂载到 `left_arm` / `right_arm`，UR5e 腕相机挂载到 `arm`。
按实际关节反馈判断运动，忽略夹爪维度及小于 0.02 rad 的关节变化，避免反馈噪声触发检查。
静止机械臂的腕相机允许图像不变，但仍必须持续收到新帧。
像素 diff 是整段重复图像的补充检查：传感器噪声也可能产生非零 diff，因此断连主要由真实帧标识检测。
检查在新采集数据保存时执行，不会自动修改历史数据的质量标签。
