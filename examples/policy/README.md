# Policy examples

EVA-CLIENT consumes observations shaped as `{"state", "images", "prompt"}` and
executes policy action chunks shaped `[T, D]`. The selected backend adapts that
contract to the server protocol. Robot state order, image channels/layout, action
space, and publish rate must match the deployed model.

| Integration | EVA backend | Start / preset |
| --- | --- | --- |
| [Fake policy](fake_policy/fake_policy_server.py) | `openpi` or `starvla` over WebSocket | `bash examples/policy/fake_policy/run_fake_policy.sh --robot dual_yam` |
| XPolicyLab | `xpolicylab` over WebSocket | External XPolicyLab server; [`dual_agilex_piper/xpolicylab.py`](../../configs/01_deploy/dual_agilex_piper/xpolicylab.py) |

## Fake policy

The fake server runs with the repository's existing `.venv/bin/python`, defaults
to port 9000, and supports OpenPI and StarVLA request envelopes. It derives action
width from the incoming state, with `--robot` or `--action-dim` providing the
fallback. `--action-mode qpos` is the default; use `--action-mode eef` with a
matching EEF client configuration. `--chunk-size` defaults to 50.

```bash
bash examples/policy/fake_policy/run_fake_policy.sh --robot dual_yam
.venv/bin/eva --config configs/01_deploy/dual_yam/openpi_qpos.py
```

These commands start separate long-running processes; use separate terminals.
Synthetic actions exercise transport and scheduling, and do not measure model
quality. The local `mock` policy requires no server; `replay` uses recorded
trajectory actions with a dataset transport.

## XPolicyLab

XPolicyLab is a native client backend in
[`src/policy_client/xpolicylab.py`](../../src/policy_client/xpolicylab.py), rather
than an API bridge in this directory. It uses typed `hello`, `infer`, and `reset`
messages with request IDs, evaluation/trial identity, and inference-step numbers.
It converts EVA cameras to `vision.<camera>.color`, the prompt to `instruction`,
and the flat state to named arm/gripper fields. Responses become finite float32
`[T, D]` chunks.

Start an official [XPolicyLab](https://github.com/XPolicyLab/XPolicyLab) policy
server first (`python setup_policy_server.py --config_path policy/<POLICY>/deploy.yml
--port 19000` from its root), then run the matching preset:

```bash
.venv/bin/eva --config configs/01_deploy/dual_agilex_piper/xpolicylab.py
```

The dual Piper preset selects port 19000, joint actions with arm dimensions
`[6, 6]` and gripper dimensions `[1, 1]`, and ZMQ on-demand images. The execution
node must support image requests. The
[dual YAM simulator evaluation preset](../../configs/03_evaluation/dual_yam_eva_sim_eval.py)
uses `action_type="ee"`, `[7, 7]` pose dimensions (`xyz + wxyz`), `[1, 1]`
grippers, and the robot's TCP links.

For offline inspection, customize dataset, checkpoint, server command, and output
paths in the
[XPolicyLab open-loop example](../../configs/00_openloop/dual_agilex_piper_xpolicylab.py),
then run:

```bash
.venv/bin/python tools/openloop.py --config configs/00_openloop/dual_agilex_piper_xpolicylab.py
```

That example expects a sibling checkout of the official
[XPolicyLab](https://github.com/XPolicyLab/XPolicyLab) with the policy installed
and its `deploy.yml` pointing at your checkpoint. The tool writes `trajectory.jsonl`, `summary.json`, and `open_loop.png`
into a new output directory and stops its configured server when finished.
See [configuration](../../docs/configuration.md) for image, space, and timeout
settings.
