[← Back to README](../README.md)

# 🧭 Web console

A single console app with six tabs, organized into three flows. Tab
availability is config-driven — a deploy config lands on **DEBUG**, an eval
config (`eval_cfg.checkpoints[]`) lands on **EVAL**; unrelated tabs are greyed
out or read-only.

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
