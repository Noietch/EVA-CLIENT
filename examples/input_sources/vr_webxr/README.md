# WebXR VR Input Node

`node.py` serves a WebXR page on the server and sends PICO/Quest input to EVA
Client over ZMQ. The example below uses page port `43876` and ZMQ ports
`8765/8766`.

## Start the Server

In EVA's DEVICE page, select `VR` under Operation, apply the selection, and
press Operation Start. The detached device service starts this node and provides
the authenticated page through the WebXR button. Robot and Camera have separate
Start controls. Closing EVA does not stop the device service; use Operation Stop
to release the VR listener.

For a standalone bridge, run from the repository root using its existing
environment. Supply the token through standard input:

```bash
cd /path/to/EVA-CLIENT
read -rsp 'VR access token: ' VR_TOKEN
printf '\n'
export VR_TOKEN
printf '%s\n' "$VR_TOKEN" | .venv/bin/python \
  examples/input_sources/vr_webxr/node.py \
  --token-stdin \
  --host 127.0.0.1 \
  --port 43876 \
  --endpoint tcp://127.0.0.1:8765 \
  --ack-endpoint tcp://127.0.0.1:8766
```

Keep the process running. The node logs its address without the token; the
launcher below must receive the same `VR_TOKEN` from its own shell. If the token
is omitted, the node generates one but does not print an authenticated URL, so
use DEVICE or supply a known token through standard input.

## Input Protocol

The VR protocol version sent by the node to EVA Client is `3`. Each hand sends
the absolute WebXR pose: `position` and `orientation_xyzw`. EVA Client's
retargeter computes the relative-pose reference and accumulated delta.

The node consumes each hand's `grip` (WebXR gamepad button 1) locally to
toggle that hand's `grip_engaged` state after a long press. One long press
enables it and the next disables it. The raw grip button is not sent to EVA
Client; only the debounced state is sent with the absolute pose. For an
unauthorized hand, the client uses the inactive-arm mask to hold the current
robot joint positions. The first long-press unlock calibrates that controller
against the arm's live measured EEF pose. When authorization is restored after
a later lock, the new controller-pose delta is added to the previously
accumulated delta.

When ARM is OFF or teleop is reset, the client clears the accumulated deltas
and reference poses for both hands. After each grip long-press toggle, the node
sends a short haptic feedback signal for the corresponding controller to the
WebXR page. The actual vibration effect depends on WebXR Gamepad haptics support
in the device browser. The PICO launcher defaults to PICO Browser; set
`WOLVIC_PACKAGE` to an installed Wolvic package to use its OpenXR backend.

The right-hand B button (WebXR gamepad button 5) emits the `arm_toggle` event
immediately on press, once per press; the EVA app still owns the global ARM
state.
Pressing B does not calibrate the controllers. Grip authorization and the
B-button ARM state are independent data streams.

The remaining controller gestures map to operator intents:

| Gesture | Intent |
| --- | --- |
| Right A short press/release | `record_toggle` |
| Right A hold (default 1 second) | `record_cancel`, suppressing release toggle |
| Left X release | `home` |
| Left Y release | `intervention_toggle` (collection QC outside RL) |
| Left stick direction | review navigation with delayed repeat |
| Left stick press | `review_select` |

The host node's `--long-press-ms` option controls the hold duration.

## Native PICO client

统一启动和安装说明见 [`../eva-pico/README.md`](../eva-pico/README.md)。

WebXR is only one transport. The standalone [EVA-VR repository](https://github.com/Noietch/EVA-VR)
is a native OpenXR APK named **EVA-VR**. Source: [github.com/Noietch/EVA-VR](https://github.com/Noietch/EVA-VR); direct APK: [latest release](https://github.com/Noietch/EVA-VR/releases/latest). It connects to the same
WebSocket endpoint, sends the same `frame` payload, and handles `haptic`
messages through the native PICO/OpenXR action. Mac and Linux can run the
unchanged EVA server.

The native client sends messages in this shape:

```json
{
  "type": "frame",
  "version": 1,
  "seq": 42,
  "client_time_ms": 1234567890,
  "reference_space": "local-floor",
  "controllers": {
    "left": {"valid": true, "position": [0, 0, 0], "orientation_xyzw": [0, 0, 0, 1], "buttons": [], "axes": [], "profiles": ["pico-4-ultra"]},
    "right": {"valid": true, "position": [0, 0, 0], "orientation_xyzw": [0, 0, 0, 1], "buttons": [], "axes": [], "profiles": ["pico-4-ultra"]}
  }
}
```

EVA sends a haptic request only for an explicit response or haptic event;
ordinary `frame` messages do not vibrate the controller. The wire message is:


```json
{"type":"haptic","hand":"left","intensity":0.6,"duration_ms":80}
```

The native APK handles haptics through OpenXR; the WebXR page uses the browser's
Gamepad haptics API where available. The WebSocket server runs on Mac and Linux.

## Ubuntu Port Forwarding

Keep the SSH tunnel running on the Ubuntu host connected to the PICO:

```bash
ssh -N \
  -p 22 \
  -o ExitOnForwardFailure=yes \
  -o ServerAliveInterval=30 \
  -o ServerAliveCountMax=3 \
  -L 43876:127.0.0.1:43876 \
  "<remote-user>@<remote-host>"
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
read -rsp 'VR access token: ' VR_TOKEN
printf '\n'
export VR_TOKEN
./examples/input_sources/vr_webxr/open_pico.sh
```

You can also specify the device explicitly with an argument or environment
variable:

```bash
./examples/input_sources/vr_webxr/open_pico.sh "<PICO_SERIAL>"

export PICO_SERIAL="<PICO_SERIAL>"
read -rsp 'VR access token: ' VR_TOKEN
printf '\n'
export VR_TOKEN
./examples/input_sources/vr_webxr/open_pico.sh
```

To reopen or refresh the page, run the same launcher again:

```bash
./examples/input_sources/vr_webxr/open_pico.sh "$PICO_SERIAL"
```

After opening the page, click `ENTER MR`. The launcher defaults to PICO Browser
(`com.pico.browser`). To use Wolvic, select its installed package explicitly:

```bash
WOLVIC_PACKAGE=com.igalia.wolvic ./examples/input_sources/vr_webxr/open_pico.sh
# Mainland Wolvic builds may use com.cn.igalia.wolvic instead.
```

The example uses `127.0.0.1` through SSH/ADB. Directly accessing the node
through a remote IP requires HTTPS/WSS.
