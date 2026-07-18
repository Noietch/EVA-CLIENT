[← Back to README](../README.md)

# 🎮 Control channel (ZMQ)

The control channel exposes **every console button** to an external process — a
simulator, a script, another service — over a ZMQ **REP** socket, so evaluation
can run headless without a browser.

## Why one channel = all buttons

Every console button collapses, on the backend, to one action: putting a `web:*`
string onto the internal command queue, dispatched by
[`core.app.run.handle_command`](../src/core/app/run.py). The HTTP console
(`core/app/console/server.py`) is just an "HTTP endpoint → command queue"
translator. The control channel forwards any allow-listed `web:*` command onto
that same queue — so a single socket exposes the full button surface. It shares
the console's `ConsoleContext`, so state driven over ZMQ and state driven from the
browser stay in lockstep.

## Enable it

Off by default. In your config:

```python
control_channel = dict(
    enabled=True,
    host="127.0.0.1",   # localhost only; set "0.0.0.0" to accept remote callers
    port=5757,
)
```

> ⚠️ **Security:** this channel can drive a real robot. `0.0.0.0` exposes that to
> the network. Keep it on `127.0.0.1` unless the simulator runs on another host
> (and then prefer an SSH tunnel).

## Wire protocol

JSON request → JSON reply (REQ/REP). Every reply carries `ok: true|false`.

| Request | Reply |
|---|---|
| `{"cmd": "web:run"}` | `{"ok": true, "cmd": "web:run"}` |
| `{"cmd": "web:tab_switch:collect", "armed": true}` | `{"ok": true, "tab": "collect", "armed": true}` |
| `{"cmd": "web:select_collect_task", "task": "pick apple"}` | `{"ok": true, "selected_collect_task": "pick apple"}` |
| `{"query": "status"}` | `{"ok": true, "data": { …status… }}` |
| unknown / disallowed | `{"ok": false, "error": "…"}` |

## Commands (`web:*`)

Each command is gated by the current `web_phase` / `session_status` — a command
invalid in the current state is ignored (check `last_error` / logs). Preconditions
below come from [`_handle_web_command`](../src/core/app/run.py).

### Session / run
| Command | Arg | Web button | Notes |
|---|---|---|---|
| `web:select_mode:<mode>` | `real`/`sim`/`step`/`manual` | mode picker | pick before setup |
| `web:select_task:<task>` | task string | DEBUG task picker | blocked while running |
| `web:select_strategy:<key>` | strategy key | strategy picker | must exist in config |
| `web:setup` | — | SETUP | needs task + mode selected |
| `web:run` | — | RUN / START | REAL/SIM start; STEP previews a chunk |
| `web:halt` | — | HALT | stop continuous publishing |
| `web:console_reset` | — | RESET | reset arm to home |
| `web:connect` / `web:disconnect` | — | connect toggle | policy connection |

### Single-step (STEP mode)
| Command | Web button |
|---|---|
| `web:step_infer` | infer one chunk on sim |
| `web:step_commit` | replay pending chunk on real |
| `web:step_cancel` | drop pending chunk |

### Gripper
| Command | Meaning |
|---|---|
| `web:gripper:<side>:<open\|close>:<0\|1>` | drive side (`l`/`r`) to state; `1` locks it during RUN |
| `web:gripper:<side>:<value>` | lock to a numeric value |
| `web:gripper:<side>` | toggle |

### Collection (COLLECT) — full teleop capture
| Command | Extra keys | Notes |
|---|---|---|
| `web:select_collect_task` | `{"task": "..."}` | sets the collect task (mirrors HTTP; no queue) |
| `web:tab_switch:collect` | `{"armed": true}` | enter COLLECT and arm teleop (required before start) |
| `web:collect_start` | — | begin recording an episode (needs armed COLLECT) |
| `web:collect_stop` | — | finish + save the episode |
| `web:collect_cancel` | — | discard the in-flight episode |

### Manual
`web:manual_qpos:<c,s,v>` · `web:manual_send` · `web:manual_home` · `web:manual_dispatch`

### Eval / checkpoint
`web:switch_ckpt:<slot>` (0→Model A…) · `web:warmup` · `web:tab_switch:eval`

> **Scoring stays manual** — `eval_start` (bind trial identity) and `score` (write
> to the dataset) are intentionally **not** exposed here; do those in the browser.

### Rollout / intervention / replay
`web:rollout_stop` · `web:rollout_save` · `web:rollout_intervention_mode:<absolute|relative>`
· `web:rollout_intervention_abandon` · `web:load_replay_dataset` · `web:clear_replay`
· `web:set_replay_fps:<n>` · `web:replay_seek:<frame>` · `web:select_episode:<n>`

## Queries (read-only)

| Query | Returns |
|---|---|
| `status` | live snapshot — `session_status`, `cli_mode`, `step_index`, `chunk_index`, `policy_connected`, `transport_connected`, `collect`, `rollout`, `last_error`, … (full shape: `_serialize_status`) |
| `config` | static config — robot/transport type, tasks, strategies, modes (`_serialize_config`) |
| `frame` | latest qpos + camera keys (`_serialize_frame`) |

## CLI client

[`examples/control_channel/eva_ctl.py`](../examples/control_channel/eva_ctl.py) is a thin REQ client:

```bash
python examples/control_channel/eva_ctl.py cmd web:select_mode:sim
python examples/control_channel/eva_ctl.py cmd web:run
python examples/control_channel/eva_ctl.py query status
python examples/control_channel/eva_ctl.py wait-idle          # block until the run finishes
python examples/control_channel/eva_ctl.py cmd web:tab_switch:collect --json '{"armed": true}'
```

## Automated evaluation loop

A simulator can import `send` directly and run a closed loop (add
`examples/control_channel/` to your path, or copy `eva_ctl.py` next to your script):

```python
import time
from eva_ctl import send, wait_idle

send(cmd="web:select_mode:sim")
send(cmd="web:setup")
for task in tasks:
    send(cmd=f"web:select_task:{task}")
    send(cmd="web:run")
    final = wait_idle()          # polls status until session_status != "running"
    send(cmd="web:console_reset")
    # inspect `final` (step_index, last_error, …) to record the trial outcome
```

## Headless / low-power mode

Running `eva --headless` starts this channel automatically (no web server) and adds
an in-process interactive CLI. See [Console → Headless / low-power mode](./console.md#-headless--low-power-mode).


## Adding a new command

The allow-list in [`control_channel._ALLOWED_VERBS`](../src/core/app/control_channel.py)
mirrors the verb branches in `run.py::_handle_web_command`. When you add a verb
there, add it to the allow-list too — otherwise the channel rejects it.
