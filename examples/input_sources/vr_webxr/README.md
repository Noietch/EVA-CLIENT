# WebXR VR Input Node

`node.py` serves a WebXR page on the server and sends PICO/Quest input to EVA
Client over ZMQ. The example below uses page port `43876` and ZMQ ports
`8765/8766`.

## Start the Server

```bash
cd "$CLIENT_ROOT"
source .venv/bin/activate

python examples/input_sources/vr_webxr/node.py \
  --host 127.0.0.1 \
  --port 43876 \
  --endpoint tcp://127.0.0.1:8765 \
  --ack-endpoint tcp://127.0.0.1:8766
```

Keep the process running. The WebXR URL and PICO command printed in the logs
can be used directly. If `--token` is omitted, the node generates a random
token; use the URL from the logs.

## Input Protocol

The VR protocol version sent by the node to EVA Client is `3`. Each hand sends
the absolute WebXR pose: `position` and `orientation_xyzw`. EVA Client's
retargeter computes the relative-pose reference and accumulated delta.

The node consumes each hand's `grip` (WebXR gamepad button 1) locally to
toggle that hand's `grip_engaged` state after a long press. One long press
enables it and the next disables it. The raw grip button is not sent to EVA
Client; only the debounced state is sent with the absolute pose. For an
unauthorized hand, the client uses the inactive-arm mask to hold the current
robot joint positions. When authorization is restored, the new controller-pose
delta is added to the previously accumulated delta.

When ARM is OFF or teleop is reset, the client clears the accumulated deltas
and reference poses for both hands. After each grip long-press toggle, the node
sends a short haptic feedback signal for the corresponding controller to the
WebXR page. The actual vibration effect depends on WebXR Gamepad haptics support
in the device browser.

The existing short-press `arm_toggle` event for the right-hand B button (WebXR
gamepad button 5) is unchanged; `client/app` still owns the global ARM state.
Grip authorization and the B-button ARM state are independent data streams.

## Ubuntu Port Forwarding

Keep the SSH tunnel running on the Ubuntu host connected to the PICO:

```bash
ssh -N \
  -p 22 \
  -o ExitOnForwardFailure=yes \
  -o ServerAliveInterval=30 \
  -o ServerAliveCountMax=3 \
  -L 43876:127.0.0.1:43876 \
  <remote-user>@<remote-host>
```

## Open or Refresh the PICO Page on Ubuntu

Connect the PICO to the Ubuntu host with a USB cable and query the device
serial number:

```bash
adb devices -l
```

When only one authorized device is present, run the script to identify it,
create the ADB reverse port forwarding, and open the WebXR page:

```bash
export VR_TOKEN="<TOKEN_FROM_NODE_LOG>"
./examples/input_sources/vr_webxr/open_pico.sh
```

You can also specify the device explicitly with an argument or environment
variable:

```bash
./examples/input_sources/vr_webxr/open_pico.sh "<PICO_SERIAL>"

export PICO_SERIAL="<PICO_SERIAL>"
export VR_TOKEN="<TOKEN_FROM_NODE_LOG>"
./examples/input_sources/vr_webxr/open_pico.sh
```

Force a page refresh (`-S` first stops the browser application hosting the URL):

```bash
VR_URL="http://127.0.0.1:43876/?token=<TOKEN_FROM_NODE_LOG>&mode=ar&reload=$(date +%s)"
adb -s "$PICO_SERIAL" shell "am start -S -a android.intent.action.VIEW -d '$VR_URL'"
```

After opening the page, click `ENTER MR`. The example uses `127.0.0.1` through
SSH/ADB. Directly accessing the node through a remote IP requires HTTPS/WSS.
