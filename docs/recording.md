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
    ├── state.json                            # robot + per-episode simulator rebuild state
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

Simulator-backed recordings additionally put `robot_name`, `task_name`, layout
`split`, `scene_index`, and a concrete `scene` in `meta/state.json`. The scene includes
each object's `asset_path`, initial `position`, `quaternion_wxyz`, and physical role,
plus resolved robot and camera poses. Rendering no longer depends on replaying a
placement seed against mutable task-layout code. This is state format v2; a recorded
`seed` is retained only as a compatibility fallback for older datasets and simulators.

EVA Sim `--skip-render` episodes are recorded as state-only datasets: Parquet and
metadata are written normally, `videos/` is omitted, and the automatic quality result
is green. The live console keeps updating robot state without opening camera streams.
A dataset exposes only camera features present in every episode; extra
per-episode videos remain available to the console through `episodes.jsonl`. Mixing
state-only and rendered observations inside one episode is a QC failure.
