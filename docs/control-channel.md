[← Back to README](../README.md)

# Control channel (ZMQ)

The ZMQ control channel lets scripts, simulators, and shortcut clients send
operator commands to EVA's state machine. It shares the command queue and live
session with the browser. It exposes the allow-listed command catalog and three
read-only queries; dataset transfer, QC writes, scoring, device configuration,
and other HTTP-only APIs are separate interfaces.

## Enable the channel

```python
control_channel = dict(
    enabled=True,
    host="127.0.0.1",
    port=5757,
)
```

It is disabled by default and enabled automatically by `eva --headless`.
`127.0.0.1` binds locally; use an SSH tunnel for another host, or configure the
bind address deliberately. The channel can issue robot motion commands.

The control socket is independent of the robot transport (`zmq`, `ros1`, `ros2`,
or `dataset`). Observations/actions continue over that transport; sparse operator
commands do not add a request/reply round trip to each inference step.

## Wire protocol and completion

Send JSON objects over ZMQ REQ/REP. Every reply contains `ok`. Ordinary command
replies acknowledge queueing, not completion or successful motion. State gates
are applied later by the main loop; poll status and inspect `last_error`.

| Request | Reply shape |
| --- | --- |
| `{"cmd":"web:run"}` | `{"ok":true,"cmd":"web:run"}` |
| `{"cmd":"web:tab_switch:collect"}` | `{"ok":true,"tab":"collect"}` |
| `{"query":"status"}` | `{"ok":true,"data":{...}}` |
| Invalid request or disallowed verb | `{"ok":false,"error":"..."}` |

Only `web:*` verbs in
[`command_catalog.py`](../src/core/app/command_catalog.py) are allowed.
`tab_switch` updates the active console tab and queues its transition. Haptics are sent directly to
the VR client. `web:start` with `clip_id` validates/binds an evaluation attempt
before queueing it. Other supported commands enter the shared queue.

## Commands

`<...>` denotes a value to substitute. Mode, policy, task, and strategy selections
must match the loaded workflow and current state.

| Workflow | Commands |
| --- | --- |
| DEBUG / session | `web:select_mode:<real\|sim\|step\|manual>`, `web:switch_task:<task>`, `web:select_strategy:<key>`, `web:connect`, `web:disconnect`, `web:run`, `web:halt`, `web:console_reset` |
| STEP | `web:step_infer`, `web:step_commit`, `web:step_cancel` |
| Manual | `web:manual_qpos:<comma-separated-qpos>`, `web:manual_dispatch:<dispatch>`, `web:manual_send`, `web:manual_home` |
| Collection | `web:tab_switch:collect`, `web:collect_arm:<0\|1>`, `web:collect_home`, `web:collect_start`, `web:collect_stop`, `web:collect_cancel` |
| Evaluation | `web:tab_switch:eval`, `web:switch_ckpt:<slot>`, `web:switch_task:<task>`, `web:warmup`, `web:start`, `web:stop`, `web:eval_cancel`, `web:reset` |
| RL | `web:tab_switch:rl`, `web:rl_select_task:<task>`, `web:rl_select_policy:<slot>`, `web:rl_select_critic:<slot>`, `web:rl_setup`, `web:run`, `web:halt`, `web:console_reset` |
| Rollout / HIL | `web:rollout_stop`, `web:rollout_save`, `web:rollout_intervention_enabled:<0\|1>`, `web:rollout_intervention_abandon` |
| Replay | `web:load_replay_dataset:<payload>`, `web:clear_replay`, `web:select_episode:<index>`, `web:set_replay_fps:<fps>`, `web:replay_seek:<frame>` |

`web:select_task` is not a catalog verb: use `web:switch_task:<task>`. The current
catalog also omits `web:setup` and `web:select_collect_task`. DEBUG setup must be
completed through the browser or in-process headless CLI before ZMQ starts a run;
collection dataset/task selection uses the HTTP console API. Gripper
commands support `web:gripper:<side>:<open|close>:<0|1>` (the final value controls
locking), a numeric value, or a side-only toggle. The catalog also includes
bootstrap, initialization, operator-action, and inference-parameter commands;
use its templates for those specialized flows.

Browser mode publishes the complete catalog at `GET /api/config` under
`control_channel.commands`, including canonical command templates and related
DOM controls. The `config` query exposes the same metadata in headless mode.
An allow-listed verb is not a guarantee that every payload or state is valid.

## Collection and SIM metadata

Select a dataset/task and slot through the browser or HTTP collection APIs,
then enter COLLECT and enable teleop with an explicit command:

```json
{"cmd":"web:tab_switch:collect"}
{"cmd":"web:collect_arm:1"}
{"cmd":"web:collect_start"}
{"cmd":"web:collect_stop"}
```

Send each line as a separate request and poll status between dependent steps.
An `armed` key on `tab_switch` does not arm motion. STOP finishes the take and
leaves teleop active; use `web:collect_arm:0` to disable it before HOME.

A mounted scene plan requires a selected collection slot. Slot selection and
skip/defer are provided by the HTTP collection APIs. A dataset/task selection
alone does not replace slot binding. Poll `status.collect` for capture/save state.

For `collection.sim_recording_url` configs, pass a `sim_metadata` object on
`web:collect_start` containing the selected simulator scene/task metadata.
The recorder validates it against the simulator status and pins
`scene_revision`. A non-object metadata value is rejected. The snapshot is
consumed by that recording; bind the next scene again for the next take. See
[Recording](./recording.md#eva-sim-execution-recording).

## Evaluation attempts

`web:start` supports the same attempt-binding path as the HTTP start API when
`clip_id` is provided:

```json
{
  "cmd": "web:start",
  "clip_id": "unique-attempt-id",
  "prompt": "pick cup",
  "trial": 1,
  "model_name": "configured-model-name",
  "task_set": "configured-task-set",
  "task_id": "TASK-001",
  "scene_id": "SC-001"
}
```

EVA must be in the `ready` phase. Trial numbers are positive integers.
`model_name` is optional but, when supplied, must match the active model.
`task_set`, `task_id`, and `scene_id` are either supplied together or omitted
for a prompt-only attempt. The prompt and scene must belong to the selected
plan. Optional `sim_metadata` is checked for matching scene/task identity.
A successful reply reserves the attempt and moves its phase to `starting`.

Use `web:stop` to finish an evaluation attempt and wait for its saved record.
Milestone scoring remains an HTTP/browser operation (`/api/score`), tied to the
recorded clip and model. The control channel does not expose a score command.
DEBUG's `web:run`/`web:halt` and evaluation's `web:start`/`web:stop` serve their
respective state-machine flows.

## VR haptics

```json
{"cmd":"web:haptic","hand":"right","intensity":0.6,"duration_ms":80}
```

`hand` accepts `left`, `right`, `both`, or alias `all` (default `both`). Intensity
is clamped to `0..1`; duration is clamped to `1..1000` ms. Defaults are `0.6` and
`80` ms. Numeric values must be finite. The positional form is also accepted:

```text
web:haptic:right:0.6:80
```

The response reports the hands actually sent to the native EVA-VR client's
outbound queue. A disconnected headset returns `ok:false`. The native app applies
the feedback through OpenXR; the command does not change robot session state.

## Queries and CLI client

| Query | Contents |
| --- | --- |
| `status` | Live session/phase, policy/transport health, collection and rollout state, history, and last error |
| `config` | Robot/transport, tasks, modes, strategies, evaluation/RL configuration, and command metadata |
| `frame` | Latest visualization frame, qpos, and camera keys |

[`eva_ctl.py`](../examples/control_channel/eva_ctl.py) provides a REQ client:

```bash
.venv/bin/python examples/control_channel/eva_ctl.py query config
.venv/bin/python examples/control_channel/eva_ctl.py cmd web:switch_task:pick_cup
.venv/bin/python examples/control_channel/eva_ctl.py cmd web:run
.venv/bin/python examples/control_channel/eva_ctl.py query status
.venv/bin/python examples/control_channel/eva_ctl.py wait-idle
.venv/bin/python examples/control_channel/eva_ctl.py haptic right --intensity 0.6 --duration-ms 80
```

Use a configured task string in place of `pick_cup`. `wait-idle` polls until
`session_status` is no longer `running`; because commands are queued, it is not
a start acknowledgement. Observe the expected running/phase transition first
when automating a run. Headless also offers a TTY-only REAL inference prompt;
see [Console](./console.md#headless--low-power-mode).

## Extending the catalog

Implement the state-machine branch in
[`run.py`](../src/core/app/run.py) and add its entry to the shared command catalog.
The catalog supplies both the ZMQ allow-list and console shortcut metadata.
Commands needing HTTP-style context mutation must also implement that binding in
[`control_channel.py`](../src/core/app/control_channel.py); queueing a string
alone does not reproduce an HTTP handler's metadata writes.
