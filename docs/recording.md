[← Back to README](../README.md)

# Recording

EVA records collection takes and policy rollouts as LeRobot v2.1 datasets.
Recording must be enabled, and each workflow determines its own output root.

| Workflow | Dataset location |
| --- | --- |
| COLLECT | `<data_root>/<collection_dir>`; a task set's `info.yaml` supplies `collection_dir` |
| COLLECT without `collection_dir` | `<data_root>/datasets/real_robot/<robot_type>/<dataset>` |
| EVAL | `<eval.output_dir>/<model_name>/episodes/raw` |
| RL / rollout | The workflow's configured storage/log directory |

`collection.storage.data_root` overrides the collection root. Otherwise EVA
infers it from the enclosing `task_sets/` directory or defaults to
`datasets/data_collection`. `collection.storage.log_dir` holds runtime logs and
UI state; it is not the collection dataset root. See
[Scene plans](./collection_scene_plan.md) for task-set discovery and paths.

A typical native dataset contains:

```text
<dataset>/
├── data/chunk-000/episode_000000.parquet
├── videos/chunk-000/<video_key>/episode_000000.mp4
├── raw/<uuid>.mcap                         # SIM execution, when enabled
└── meta/
    ├── info.json
    ├── episodes.jsonl
    ├── episodes_stats.jsonl                # collection feature statistics
    ├── tasks.jsonl
    ├── stats.json
    ├── qc.jsonl                           # review and automatic QC ledger
    └── render_reports/episode_000000.json  # optional SIM replay report
```

Videos and SIM files are present only when that capture/render path produces
them. Use `meta/info.json` and each episode's `video_keys` to discover actual
features and media instead of assuming every episode has every camera.

## Capture and save lifecycle

Real collection records state, issued actions, and camera samples, then aligns
them to the configured output FPS. The base collection schema writes
`observations.state.qpos`, `observations.state.eef`, `action.qpos`, and
`action.eef`; robot/config overrides decide which columns and cameras are
required. Each row also carries timestamp, capture time, frame/episode/task
indices, and a global index.

Capture runs independently of the main control loop for transport-driven real
collection. Saving uses a bounded background queue. A new take is refused when
the queue is full. Real collection STOP waits for its save result; SIM STOP queues
the validated execution for background preparation, so inspect the job state
before assuming it is committed. STOP leaves the robot at its current pose and
teleop active. CANCEL discards the active take.

Scene recordings attach `task_id`, `scene_id`, zero-based `scene_round`, and
`slot_id` to the episode. A retake of the same slot replaces its previous
recording at the same episode index and clears stale review/render artifacts as
part of the save transaction. Unbound recordings allocate a new episode index.
Evaluation attempts carry their clip/trial/model identity and saved scores.
Rollout recordings also preserve human-intervention labels and segment metadata.

## EVA-SIM execution recording

Set `collection.sim_recording_url` to the simulator HTTP endpoint. The provided
EVA-SIM collection configs use `http://127.0.0.1:8093`, disable live camera
recording, and leave `render_service.url` empty for offline rendering.

At START, EVA requires bound simulator scene metadata, reads `/api/status`, pins
`scene_revision`, reserves `raw/<uuid>.mcap` inside the dataset, and calls
`/api/recording/start`. STOP calls `/api/recording/stop`; CANCEL calls
`/api/recording/cancel`. The simulator must be able to write the supplied absolute
path and the client must be able to read it.

EVA validates the MCAP header, step sequence, timing, dimensions, episode ID, and
complete footer. Native rows follow observe-before-action semantics: row `k`
contains the state before action `k`, including the initial state for row zero.
The final post-action state is retained in the execution record. FPS is derived
from `1 / control_dt`; this path does not resample the execution with real-camera
alignment. Metadata links the native episode to its `sim_execution_path`, schema
version, and `sim_metadata`.

When a render-service URL is configured, a successful collection save sends a
background notification. Replay validation reads the episode's explicit
`render_report` path or `meta/render_reports/episode_<index>.json`. A missing
report is pending; malformed reports or unsuccessful task evaluation produce a
replay failure. Capturing an MCAP does not prove successful replay or task
completion.

## Capture quality and review QC

Capture writes `quality=green|red` and structured `quality_issues` in
`episodes.jsonl`. A red take is retained for inspection. Checks include short
recordings, missing required camera streams, repeated images, invalid vectors,
and timing/alignment issues. Background automatic QC runs after a collection
save and checks duration outliers, static/stalled camera patterns, and commanded
or observed arm-joint limits; its findings are stored in `meta/qc.jsonl`.

The console and dataset service share these states:

| State | Meaning |
| --- | --- |
| Pending / TO COLLECT | No saved take is available for the slot |
| Unreviewed | A saved take has no accepting/rejecting verdict or still requires review |
| Passed | An explicit human PASS accepts the take |
| Failed | A human FAIL, capture red flag, automatic rejection, or required failed SIM replay rejects it |

Human PASS overrides capture, automatic, and SIM replay failures. Human FAIL
rejects the episode. Without PASS, automatic failure and required replay failure
remain effective even if the human state is UNREVIEWED. Replay diagnostics remain
visible after an override. Green capture quality alone does not mean QC passed.
Use PASS/FAIL/UNREVIEWED and notes in COLLECT or the dataset review service; these
write the QC ledger instead of changing the source recording.

## Camera timing and frozen frames

YAM observations carry each camera's latest cached frame, so consecutive
observations can reuse an exposure. Camera samples use observation-assembly time
for software alignment. A small alignment skew does not establish exposure
synchronization between cameras. Disconnecting a camera clears its cache and
preview.

The device UI's YAM D405 + two Gemini 305 combination captures at 640×480 and
60 FPS, with 60 Hz camera forwarding and robot observations. Native recordings
still use their configured output FPS (normally 30). Combinations including a
Gemini 355 use 30 FPS. Higher upstream rates reduce sample gaps but the latest
frame cache does not retain every exposure.

Capture marks a missing configured stream with `missing_camera_stream`. It marks
an entirely repeated external-camera stream, or a repeated wrist-camera stream
whose arm moved, with `frozen_camera`. `CameraSpec.attached_to` identifies the
arm; `None` identifies an external camera. Capture ignores grippers and arm
motion below 0.02 rad for this check. A stationary wrist camera may legitimately
remain unchanged. Pixel differences can include sensor noise, so they do not by
themselves establish useful scene motion. These capture checks apply to newly
saved takes; they do not relabel historical recordings automatically.

## Conversion and delivery

The native format is already LeRobot v2.1. COLLECT converts a dataset to
`lerobot_v3`, `hdf5`, or `mcap` in the sibling
`<dataset_name>_<format>` directory. Repeating conversion replaces that generated
output. All source episodes are converted and QC metadata is preserved; the
consumer decides which episodes to accept. Conversion is not a PASS-only export.

Hugging Face dataset, task, and QC transfers are separate dataset-manager
actions. Downloading data merges local/cloud QC; uploading data excludes QC.
Optional SFTP or S3 delivery is configured in `collection.storage`. Inspect
background-job results and upload receipts before treating delivery as complete.
