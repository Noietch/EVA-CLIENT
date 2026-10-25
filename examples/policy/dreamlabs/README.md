# DreamLabs API example

This bridge connects EVA-CLIENT to Dream-1:
`EVA-CLIENT → OpenPI WebSocket → server.py → DreamLabs HTTP API`.
The official SDK manages the API session, camera history, and Dream IK.
EVA-CLIENT uses `policy.type="openpi"`; the server advertises `supports_reset`
so episode resets reach the API session.

## Environment and credentials

The launcher always invokes `examples/policy/dreamlabs/.venv/bin/python`.
Dependencies are declared in [pyproject.toml](pyproject.toml), including Python
3.11, `dreamlabs-client==1.0.9`, a pinned Dream IK revision, and NumPy 1.26.4.
Keep this environment separate from EVA-CLIENT's root `.venv`.

If the example environment is absent, create it once from the repository root:

```bash
uv venv --python 3.11 examples/policy/dreamlabs/.venv
```

Reuse an existing environment as-is. Before adding dependencies, inspect the
installation plan:

```bash
uv pip install --dry-run --python examples/policy/dreamlabs/.venv/bin/python \
  -r examples/policy/dreamlabs/pyproject.toml
```

Proceed with installation only when the plan adds missing packages and preserves
all installed package versions and files. If it proposes a replacement,
reinstallation, upgrade, or downgrade, stop and resolve the conflict before
installation. For an additions-only plan, apply the same request without
`--dry-run`:

```bash
uv pip install --python examples/policy/dreamlabs/.venv/bin/python \
  -r examples/policy/dreamlabs/pyproject.toml
```

Do not recreate an existing environment. Invoke Python directly;
`uv run` is prohibited in this repository.

The server reads its API key through the SDK from `DREAM_API_KEY` or
`~/.dreamlabs/credentials`. Keep the credential in the server's existing secret
channel or credential store, outside EVA configuration files and command arguments.

## Start the bridge and client

From the repository root, use separate terminals:

```bash
bash examples/policy/dreamlabs/serve.sh --robot yam
```

```bash
.venv/bin/eva --config configs/01_deploy/dual_yam/dream.py
```

The included EVA preset selects port 5555, joint observations/actions, 320 × 176
HWC images, a 20 Hz publish rate, and one warmup chunk. Its sync execution horizon
is 48 steps; async uses a 30-step horizon with `latency_k=0`.

The listener starts before model lookup and IK compilation. Metadata and reset
are available before the first inference; the first inference initializes the
model and needs working API credentials.

| Launcher option | Default / behavior |
| --- | --- |
| `--host` | `POLICY_HOST`, otherwise `127.0.0.1` |
| `--port` | `POLICY_PORT`, otherwise `5555` |
| `--checkpoint` | `DREAM_CHECKPOINT`, otherwise `dreamlabs/dream-1-beta` |
| `--robot` | `DREAM_ROBOT`, otherwise `yam`; choices come from the pinned SDK's `ARMS` |
| `--head` | `joint` for YAM, `eef` for other robot selectors |

Explicit arguments override launcher environment defaults. For a remote bridge,
set `policy.host` in the EVA preset and bind the bridge to an interface reachable
from that client.

## Observation and action contract

The bridge requires all three cameras: `cam_high`, `cam_left_wrist`, and
`cam_right_wrist`. Each must be an HWC RGB uint8 array. They map to Dream's
`video.overhead`, `video.wrist_l`, and `video.wrist_r`, and are resized to 320 × 176
and encoded as JPEG at quality 85 for the API request.

State order is left-arm joints, left gripper, right-arm joints, right gripper.
The selected robot's IK specification determines the arm widths, excluding held
joints. YAM uses 14 values. Joint values are radians; each gripper uses one value,
0 closed and 1 open. The bridge supplies YAM grippers as `state.left_gripper` and
`state.right_gripper`; other selectors use `state.left_grip` and `state.right_grip`.

The API's `q_l_chunk` and `q_r_chunk` are concatenated in that same arm/gripper
order and returned as nonempty finite float32 `[T, D]` joint targets. Even when
`--head eef` selects the model head, the bridge returns joint targets after SDK
kinematics. Keep the EVA action space set to `JointState`.

## Episode and session behavior

Use one server process per EVA-CLIENT. Reset before each episode; reset ends the
previous API episode and clears its history when the model has been initialized.
Disconnecting the WebSocket also resets the episode.

Observations update once per inference request. SDK history therefore depends on
client request timing; the bridge has no separate camera streaming endpoint. Its
API configuration requests a 9-frame observation window at 5 FPS, actions at
20 Hz, and a 30-second timeout.

If the SDK reports `DreamSessionExpired`, it adopts the replacement session and
clears its smoother; the bridge requests an action once more from the current
observation. This keeps the WebSocket episode open. A second failure reaches the
client. Other API errors, including warmup and rate limits, are returned without
automatic retries.

## Other robot selectors

Use `--robot` with a selector accepted by the pinned SDK; inspect the installed
choices with:

```bash
examples/policy/dreamlabs/.venv/bin/python examples/policy/dreamlabs/server.py --help
```

`--head eef` is available for YAM and required for other selectors. The launcher
uses the selected SDK specification to derive both arm widths and one gripper
value per arm. An SDK selector describes kinematics; it does not establish model
compatibility or a matching EVA hardware configuration.

For another robot, supply its EVA robot/hardware preset, matching joint order,
gripper units, all three HWC RGB cameras, and a 20 Hz publish rate. The included
client preset is for dual YAM only. Custom arm JSON files are not exposed by this
launcher; extending that interface requires a code change and measured tool
geometry.

Source of the bridge contract: [server.py](server.py), [serve.sh](serve.sh), and
[the EVA preset](../../../configs/01_deploy/dual_yam/dream.py).
