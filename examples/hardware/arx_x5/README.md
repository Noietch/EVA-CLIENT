# ARX X5 — In-Sim Teleoperation Collection

Teleoperate a **dual ARX X5 that is simulated by eva_sim** and record the
episodes with EVA. This mirrors the real ARX flow in `examples/hardware/arx`,
but the ARX follower is **eva_sim** instead of CAN hardware — so there is no ARX
follower SDK, CAN, or Orbbec code here. The only real device is the pair of
**Alicia-D leader arms**, read through the real ARX example's leader SDK.

```
Alicia-D leaders --(node.py)--> eva_sim (follower + renderer) --(obs)--> EVA (records)
```

## Files

- `node.py`: the sim-teleop node. Reuses `examples.hardware.arx`'s `ArxTeleopSource`
  (Alicia-D leader reader + Alicia→ARX joint/gripper mapping) verbatim, reads the
  follower QPos from eva_sim's observation stream to calibrate, and publishes
  absolute 14-D QPos to eva_sim's action endpoint.
- `run_node.sh`: launcher. Starts eva_sim (in the eva_sim venv) as the ARX X5
  follower, then starts `node.py` (in the eva-client venv). Prints the EVA record
  command to run in a third terminal.

## Roles and ports

eva_sim's `scripts/run.py` is the execution layer: it **binds** the ZMQ
endpoints (like the real `examples/hardware/arx/node.py` does), and when EVA arms
a collection episode eva_sim bundles our teleop action back into its observation.
EVA subscribes and records that `action_qpos` — the same path it uses for the
real ARX node.

```text
tcp://127.0.0.1:5555   eva_sim PUB obs   <- node.py SUB, EVA SUB
tcp://127.0.0.1:5556   eva_sim SUB action <- node.py PUB
```

## Run

Point `EVA_SIM_DIR` at your eva_sim checkout, then launch eva_sim + the teleop
node together:

```bash
EVA_SIM_DIR=/path/to/eva_sim \
  bash examples/hardware/arx_x5/run_node.sh
# or: bash examples/hardware/arx_x5/run_node.sh /path/to/eva_sim
```

Once it reports eva_sim is up, start EVA in a **third terminal** to record:

```bash
source .venv/bin/activate
eva --config configs/02_collection/arx_x5.py --web-port 8080
```

Open the EVA web console at `http://127.0.0.1:8080`, switch to the collect tab to
arm teleop, and start/stop episodes there. Recorded episodes land under EVA's
`work_dirs/collection/arx_x5/` (see `configs/02_collection/arx_x5.py`).

### Useful overrides

```bash
TASK=push \                       # eva_sim task (pick_and_place, lift, push, pull, rotate)
PUBLISH_RATE=30 \                 # teleop publish rate (Hz)
ALICIA_READER=duo \               # duo = pure serial (no ARX SDK); sdk = alicia_d_sdk
EVA_SIM_WARMUP_SECONDS=25 \       # how long to wait for eva_sim to bind before teleop starts
  EVA_SIM_DIR=/path/to/eva_sim bash examples/hardware/arx_x5/run_node.sh \
  --left-alicia-port /dev/serial/by-id/<left> \
  --right-alicia-port /dev/serial/by-id/<right>
```

Extra args after the script name are forwarded to `node.py` (e.g. Alicia ports).

## Notes

- The teleop node runs in the **eva-client** venv (which has `pyserial` and the
  ARX example SDK); eva_sim runs in its **own** venv (Isaac Sim). `run_node.sh`
  activates each in its own subshell.
- Default Alicia ports match the real ARX example:
  - `left_arm`:  `/dev/serial/by-id/usb-1a86_USB_Single_Serial_5C4C192742-if00`
  - `right_arm`: `/dev/serial/by-id/usb-1a86_USB_Single_Serial_5C4C192642-if00`
- eva_sim expects gripper `0 = closed, 1 = open` and maps it onto the X5 USD
  finger stroke; `ArxTeleopSource` already normalizes the Alicia gripper to that.
