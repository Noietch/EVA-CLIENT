[← Back to README](../README.md)

# Console

EVA provides one browser console for device setup, collection, policy inference,
replay, and evaluation. Start it with a workflow config:

```bash
.venv/bin/eva --config configs/02_collection/dual_yam.py
```

The console has eight tabs. `console.initial_tab` selects the landing page;
`auto` opens EVAL when an evaluation config is present and DEBUG otherwise.
Collection configs usually select COLLECT explicitly. Controls and available
workflows depend on the loaded robot, policy, teleop, and evaluation config.

| Tab | Purpose |
| --- | --- |
| DASHBOARD | Collection/evaluation throughput, scene filtering, QC counts, dataset transfers, and cloud comparisons |
| DEBUG | Closed-loop inference in REAL, SIM preview, or STEP mode; live robot, cameras, state/action charts, and inference tuning |
| DEVICE | Device discovery/startup, robot and camera settings, teleop controls, and manual joint control |
| COLLECT | Select a dataset and planned slot, enable motion, record teleop episodes, replay, review QC, and convert data |
| REPLAY | Load a recorded dataset and play observations/actions through the dataset transport |
| RL | Policy rollout with an optional critic, human intervention, save controls, and episode replay |
| EVAL | Select model, task, scene, and trial; prepare the robot, run, stop, and score milestones |
| RESULT | Browse saved evaluation results and replay synchronized video, robot motion, and state charts |

## Collection workflow

Choose a dataset in COLLECT, then a slot in the scene/task/round grid. A mounted
[scene plan](./collection_scene_plan.md) supplies object placement and target
counts. Scene and task filters change the displayed grid; the slot ID remains
stable. Inline `collection.tasks` configs also use the picker, with a DEFAULT
scene and one slot per requested episode. A target of `-1` stays available for
repeated captures.

Enable MOTION to activate the configured teleop source. START RECORD begins a
recording; STOP finishes it and leaves teleop active at the current pose. CANCEL
discards the active take. Disable MOTION before HOME; HOME is rejected while
teleop is armed or active. A full save queue prevents a new recording from
starting. Check the saved/failed state before treating a take as available.

The grid distinguishes TO COLLECT, UNREVIEWED, PASSED, and FAILED. Capture
completion and QC acceptance are different counts. Select a saved episode for
replay, then use PASS, FAIL, UNREVIEWED, and the note controls. Re-recording a
planned slot replaces that slot's previous episode at the same episode index;
it does not add another completed target. See [Recording](./recording.md) for
storage, capture checks, and QC precedence.

EVA-SIM configs bind the selected simulator scene/task before recording. The
simulator writes execution MCAP; EVA validates it and prepares the native dataset
in the background. Camera rendering can happen later, so a saved SIM take may
initially have no video. `dual_yam_eva_sim_v1_2_train.py` and
`dual_yam_eva_sim_v1_2_eval.py` select the corresponding v1.2 task sets; these are
collection configs for their respective layouts.

## Dataset operations

DASHBOARD reports raw collection and evaluation activity, with date and scene
filters. Its dataset manager lists the datasets mounted by the active collection
config, local collection/QC counts, remote counts, and data/task/QC comparisons.
Select datasets to upload or download data, task definitions, or QC separately.
Refresh cloud and verify performs a remote inspection and comparison; it does
not upload a local change.

These operations use `collection.storage.huggingface` and run as background jobs
with progress, ETA, per-dataset results, and stop controls. Data downloads merge
QC records with local records instead of simply overwriting the ledger. Data
uploads exclude QC, so publish QC with its own action. Operations on a dataset
that is recording or saving are rejected until capture finishes. The same
transfer engine is used by the standalone [dataset service](../tools/datasets/app.py).

COLLECT also offers DOWNLOAD ASSETS, task-set synchronization, UPLOAD DATA, and
format conversion. Task-set synchronization covers all configured task sets.
The native format is LeRobot v2.1; conversions produce LeRobot v3, HDF5, or MCAP
in a sibling directory. Conversion carries QC metadata and includes every source
episode; consumers apply their own acceptance filter. Optional SFTP/S3 delivery
and Hugging Face sync are configured under `collection.storage`.

## Scene evaluation

EVAL uses configured tasks and discovered evaluation task sets. For a scene task,
select a task set, task, scene, and trial. Only scenes referenced by that task are
valid. The start request binds `task_set`, `task_id`, and `scene_id` together with
an immutable `clip_id`; the saved trial keeps a scene-map snapshot. Simulator
metadata, when supplied, must match the chosen task and scene.

Prepare the robot before RUN. After STOP and save completion, mark milestones
and save the score/note. Scoring is tied to the recorded clip, model, scene, and
trial; a running or unsaved attempt cannot be scored. Re-entering EVAL resumes
from unscored work. RESULT includes scene identity when grouping trials and can
open recorded scene layouts, cameras, robot replay, and charts.

## RL and human intervention

Select a task and policy, then SETUP and RUN. Critic selection is optional and
adds live value telemetry and replay scoring. When enabled, human intervention
pauses policy control and lets the configured teleop source take over. Accepted
segments remain part of the rollout; abandon returns through the configured
rollback path. SAVE persists the rollout and its intervention metadata.

<p align="center">
  <video src="https://github.com/user-attachments/assets/62883016-1d5c-4688-9c39-740810b36b86" controls muted></video>
</p>

## Headless / low-power mode

```bash
.venv/bin/eva --config configs/01_deploy/dual_yam/openpi_qpos.py --headless
```

Headless skips the HTTP server, browser scene, MJPEG streams, extra
visualization reader, and RESULT preview. The transport, policy, recording, and
state machine continue to run. The ZMQ [control channel](./control-channel.md)
is enabled automatically, using `control_channel.host` and `port` (default
`127.0.0.1:5757`). `--web-port` has no effect in this mode.

When stdin is a terminal, the process also provides a small REAL-mode inference
prompt:

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

`quit` leaves the prompt while EVA keeps running. Piped/background processes skip
the prompt and can be controlled over ZMQ. State transitions and periodic
`[HB]` lines expose progress and errors in logs. The CLI's REAL mode is separate
from connecting a real transport to an EVA-SIM process; DEBUG SIM means policy
preview in the client scene.

Machine-specific `*.local.py` config overrides are ignored by Git.
