[← Back to README](../README.md)

# 🧭 Console

A single console app with seven tabs, organized into four flows. Tab
availability is config-driven — a deploy config lands on **DEBUG**, an eval
config (`eval_cfg.checkpoints[]`) lands on **EVAL**; unrelated tabs are greyed
out or read-only.

**RL flow**

| Tab       | Purpose                                                                                                             |
|-----------|---------------------------------------------------------------------------------------------------------------------|
| **RL**    | policy rollout with optional critic telemetry, HIL intervention, rollout saving/QC, and synchronized episode replay |

**Collection flow**

| Tab         | Purpose                                                                                                            |
|-------------|--------------------------------------------------------------------------------------------------------------------|
| **MANUAL**  | hand-drive a real robot with per-joint qpos sliders (no policy in loop); stage target pose then `SEND` or `HOME` — for hardware bring-up |
| **COLLECT** | teleop recording into a LeRobot v2.1 episode; background saver + in-tab QC PASS/FAIL replay                        |
| **REPLAY**  | open-loop playback of a recorded episode via the `dataset` transport, same live view as DEBUG                      |

**Deployment flow**

| Tab       | Purpose                                                                                                            |
|-----------|--------------------------------------------------------------------------------------------------------------------|
| **DEBUG** | default tab — prompt → config → setup → control; runs one closed-loop policy run in `REAL` / `SIM` / `STEP` (single-step breakpoint) modes with live 3D arm + camera + action/state charts |

**Evaluation flow**

| Tab        | Purpose                                                                                                            |
|------------|--------------------------------------------------------------------------------------------------------------------|
| **EVAL**   | multi-checkpoint sweep — each checkpoint shown as Model A/B/… + its real name, per-prompt trials with milestone scoring, optional shuffling and SSH port-forward to a remote policy server |
| **RESULT** | read-only browser over eval recordings: `model → task → trial` tree with score/success gauges, synced camera + 3D URDF + per-dimension state chart replay |

---

## 🔋 Headless / low-power mode

`eva --config <cfg> --headless` runs the console **without any web server** — for
low-power or simulator-driven inference. It skips the HTTP console, the 3D URDF
scene (mesh loading + forward kinematics), the camera MJPEG streams, the extra
visualization observation reader, and the RESULT-tab episode preview. Only the
transport ↔ policy ↔ state-machine main loop runs.

Headless has **two trigger entries feeding the same command queue** — use either,
or both at once:

1. **Interactive CLI** — when stdin is a terminal, `eva --headless` drops into an
   in-process prompt for the real-robot inference flow (no network hop). The mode is
   fixed to REAL (SIM only previews to the disabled 3D canvas):
   ```
   eva> info                  # endpoints / ports / robot, from the config
   eva> tasks                 # list configured debug tasks with indices
   eva> task 0                # select by index (or: task pick the apple)
   eva> setup
   eva> run
   eva> status                # state + policy/transport health
   eva> stop
   eva> reset
   ```
   `quit` leaves the prompt; the run keeps going. Backgrounded / piped runs (stdin
   not a TTY) skip the CLI and stay driven by ZMQ alone.

2. **ZMQ control channel** — a simulator / script drives the same session remotely.
   The channel is forced on in headless (host/port from `control_channel` config,
   default `127.0.0.1:5757`). See [Control channel](./control-channel.md).

**Status is surfaced via logs**, not a web page:

- **Event logs** — every state transition logs: `[STATUS] ready -> running`,
  `[PHASE] …`, `[CMD] run …`, `[STAGE] …`.
- **Heartbeat** — while running, a one-line status prints every ~5s (every ~30s when
  idle), so a long run visibly progresses even with no transitions:
  ```
  [HB] mode=real status=running phase=running step=142 chunk=9 infer=23.4ms policy=ok error=-
  ```

Ports in play (all from `--config`, not the CLI): **policy** endpoint (`policy.host:port`,
default `:9000`), the **transport** to the robot node, and the **ZMQ control channel**
(`:5757`). `--web-port` is unused in headless.

The browser console is normally started with:

```bash
eva --config configs/04_rl/r1lite_rl.local.py
```

Local `.local.py` files are machine-specific overrides and are ignored by git.
