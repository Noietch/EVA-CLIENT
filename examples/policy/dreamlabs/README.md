# DreamLabs API example

This is a lightweight Dream-1 API bridge:
`EVA-CLIENT -> WebSocket -> this example -> DreamLabs HTTP API`.
The official SDK manages sessions, camera history, and robot kinematics.
EVA-CLIENT reuses the existing `openpi` WebSocket backend, including episode reset
when the server advertises `supports_reset`.

## Isolated environment

Dependencies are configured in [pyproject.toml](pyproject.toml). This example uses
its own Python 3.11 environment, including JAX/JAXlib 0.7.1 for Dream IK and
NumPy 1.26.4. EVA-CLIENT's environment stays separate.

From the repository root, create the environment once; reuse it if it exists:

```bash
uv venv --python 3.11 examples/policy/dreamlabs/.venv
uv pip install --python examples/policy/dreamlabs/.venv/bin/python \
  -r examples/policy/dreamlabs/pyproject.toml
```

The server reads its API key from `DREAM_API_KEY` or `~/.dreamlabs/credentials`.
Keep credentials on the server, outside EVA configuration files.

Start directly with Bash; the script selects the isolated interpreter:

```bash
cd examples/policy/dreamlabs
bash serve.sh --robot yam
```

YAM is the default, so `bash serve.sh` also works. Select another built-in arm with
`--robot metal`, `--robot ai_worker_faas`, or any selector in the table below.
The head defaults to `joint` for YAM and `eef` for other robots. Use `--head eef`
to run YAM's EEF head. Dream-1 on another robot requires the EEF head.

The script also accepts `--host`, `--port`, and `--checkpoint`. Environment defaults
are `POLICY_HOST=127.0.0.1`, `POLICY_PORT=5555`, `DREAM_ROBOT=yam`, and
`DREAM_CHECKPOINT=dreamlabs/dream-1-beta`. Command-line arguments override them.

The WebSocket listener starts before model lookup or IK compilation. Those happen
on the first inference request, which requires a valid API key. Metadata and reset
are available without credentials.

From the repository root, in EVA-CLIENT's environment:

```bash
.venv/bin/eva --config configs/01_deploy/dual_yam/dream.py
```

For a remote server, update `policy.host` in the client configuration and bind the
server to the corresponding network interface.

## Observation and action contract

The input contains three HWC RGB cameras (`cam_high`, `cam_left_wrist`, and
`cam_right_wrist`) and a flat state vector: left-arm joints, left gripper,
right-arm joints, right gripper. The selected robot determines each arm's joint
count; held joints are excluded. YAM has a 14-value state vector. Joints use radians;
grippers use 0 for closed and 1 for open. Images are encoded as 320 x 176 JPEGs at
quality 85. Actions execute at 20 Hz.

The `reset` message ends the previous episode and clears its history. Reset before
each episode. Use one server process per EVA-CLIENT.

This example updates observations once per inference request, so camera history
sampling depends on the client request rate. It has no separate camera streaming
endpoint. If DreamLabs replaces an expired inference session, the SDK adopts the
new session and clears its action smoother. The bridge requests a fresh action
once using the current observation; the WebSocket and robot episode stay open.
If that request also fails, the error reaches the client. Model warmup, rate
limits, and other API errors are returned without automatic retries.

## Dream IK robot support

The pinned `dreamlabs-client==1.0.9` exposes these built-in `ik.arm` selectors:

| Selector | Robot |
| --- | --- |
| `yam` | YAM |
| `metal` | MakerMods Metal |
| `makers` | Makers Arm |
| `ai_worker_faas` | AI Worker |
| `trossen_wb_faas` | Trossen WidowX AI |
| `mantis_faas` | Mantis |
| `robo_t_faas` | RoboT |
| `openarm2_faas` | OpenArm2 |
| `bracketbot_faas` | BracketBot |
| `so101_faas` | SO-101 |
| `galaxea_faas` | Galaxea |
| `seeed_faas` | Seeed reBot B601-DM |
| `unitree_r1_faas` | Unitree R1 |

The SDK also accepts a custom arm JSON file describing its URDF, base and flange
links, TCP, approach and jaw axes, and rest pose. The tool geometry must be measured
and confirmed on the real gripper. An arm file describes kinematics; it does not
establish compatibility with every model.

For Dream-1 on YAM, this example uses `ik={"arm": "yam", "head": "joint"}`:
the SDK computes the hand poses and runs the model's joint targets. Dream-1 on
another embodiment requires the EEF head and its own arm configuration. The
lower-level `dream_ik` package also contains simulation robot specifications;
those are distinct from the API client's accepted built-in selectors.

The launcher selects among these built-in arms and derives their state/action
widths from the official IK specifications. The included EVA deployment preset
is for **dual YAM**. For another robot, use its own EVA robot configuration with
`policy.type="openpi"`, matching joint/gripper units, HWC RGB cameras, and a 20 Hz
action rate. The SDK's custom JSON arm-file interface is not exposed by this launcher.

References: [DreamLabs documentation](https://platform.dreamlabs.ai/docs/),
[pinned SDK arm selectors](https://github.com/dreamlabs-co/dreamlabs/blob/2d2f5239eb737c9271e15054637fe952758d5823/dream_library/dream_client/src/dream/client_ik.py),
and [arm file format](https://github.com/dreamlabs-co/dreamlabs/blob/2d2f5239eb737c9271e15054637fe952758d5823/dream_library/dream_client/src/dream/arm_file.py).
