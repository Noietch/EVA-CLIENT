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
| DEBUG | Full access or Ask for approval inference; live robot, cameras, model selection, and local execution history |
| DEVICE | Device discovery/startup, robot and camera settings, teleop controls, and manual joint control |
| COLLECT | Select a dataset and planned slot, enable motion, record teleop episodes, replay, review QC, and convert data |
| REPLAY | Load a recorded dataset and play observations/actions through the dataset transport |
| RL | Policy rollout with an optional critic, human intervention, save controls, and episode replay |
| EVAL | Select model, task, scene, and trial; prepare the robot, run, stop, and score milestones |
| RESULT | Browse saved evaluation results and replay synchronized video, robot motion, and state charts |

## Debug sessions and recordings

DEBUG keeps executed sessions in the left sidebar. The default local root is
`~/.eva-client/`: session metadata lives in `sessions/<session-id>/session.json`
and recorded rollout datasets live in `rollouts/<rollout-id>/`. Stop saves the
current execution; sending again continues the same run and appends its recording.
Use **New run** in the sidebar to start a separate run. It clears the prompt while
keeping model, permission, strategy, and recording preferences. Empty drafts are
not saved. History remains available after restarting EVA.

The composer’s circular Record icon controls recording for the next execution.
With Record off, history stores the prompt, model, permission mode, strategy,
timestamps, result, and action count. With Record on, it also captures camera
images and actions and saves them automatically when execution stops. Policy video
capture runs at least at the recording FPS, independently of the action publish
rate. ZMQ policy recordings sample the latest frame instead of accumulating a
FIFO backlog; full teleop collection keeps its FIFO capture. Images remain lazy
until encoding so longer recordings do not hold all decoded pixels in memory.
In Ask for approval mode, each
approval executes the pending chunk and automatically requests the next preview.
Each new chunk waits for approval; Stop saves the current segment. Continued
recordings appear as additional episodes in the same dataset and can be selected
using Replay’s episode controls. Wait for saving to finish before continuing.
Recording selection is locked during an execution or pending approval.

Debug permissions can be changed during execution or preview. Switching between
Full access and Ask for approval pauses at the next action boundary, preserves
the task, run, setup, and executed step count, and discards the previous preview.
Press Send to continue under the new permission; switching alone never starts
robot motion or moves the arm home.

The sidebar is a single list of runs ordered by time. Click a recorded run to
open it directly in the existing REPLAY workspace. Playback includes the existing
timeline, cameras, 3D view, and charts. Executions without recordings retain their
metadata and are marked as not recorded. Hover a run to reveal its red trash
icon. Deleting a run removes it from the list; **Undo** restores it. Deleted
recording files remain recoverable under `~/.eva-client/trash/`. Finished runs can
be deleted or restored while another run is executing. The active run and a
recording that is still saving remain locked.

Manual gripper locking defaults to off. With the switch on, manual open/close
stays fixed during model execution; switching it off releases existing locks.
During preparation or Reset, the composer’s main action becomes Stop/Resume.
The Dream-1 deployment defaults to `async` with 48-step chunks and 10 Hz action
publishing. `inference_cfg.default_strategy` selects the startup strategy; model
requests run in the background at the achievable network rate.
Async/RTC linear overlap blends arm joints while grippers use the latest
latency-aligned command directly. Set `blend_gripper=True` in that strategy’s
`args` to blend grippers too; the default is `False`. Gripper columns are derived
from the active robot rather than fixed action-vector offsets.

```python
console = dict(
    session_history=True,
    history_dir="~/.eva-client",
    record_rollouts=False,  # default selection for this EVA process
)
```

`record_rollouts` supplies the initial Record selection; the UI can change it for
later executions. DEBUG history recording is separate from collection, RL, and
evaluation recording settings. Restart EVA after changing console configuration.

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

EVAL uses configured tasks and task sets from `eval_cfg.task_set_dir`. For a scene task,
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


## Explicit content sources

Task sets and historical datasets are loaded only from configured sources:

- COLLECT scene plans: `collection.task_set_dir`; empty means no scene plan.
- EVAL scene plans: `eval_cfg.task_set_dir`; empty means no task sets.
- Dashboard: `dashboard.raw_roots`; empty means no datasets. `work_dir` and the
  current run directory are not automatically added. `raw_roots_only` is no longer needed.
- RESULT: configured `eval_cfg.checkpoints` (or the current model for a configured
  evaluation), plus historical names explicitly listed in `eval_cfg.result_models`.
  Other model directories are not loaded.
- Standalone dataset browser: `dataset_browser.plans_root`, `assets_root`, and
  `collection_root` in the selected `--config`; all three are required.

A configured root may contain multiple datasets or task sets; only that root is searched.
