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
robot joint positions. The first long-press unlock calibrates that controller
against the arm's live measured EEF pose. When authorization is restored after
a later lock, the new controller-pose delta is added to the previously
accumulated delta.

When ARM is OFF or teleop is reset, the client clears the accumulated deltas
and reference poses for both hands. After each grip long-press toggle, the node
sends a short haptic feedback signal for the corresponding controller to the
WebXR page. The actual vibration effect depends on WebXR Gamepad haptics support
in the device browser. The PICO launcher opens the page explicitly in Wolvic,
whose OpenXR backend includes WebXR haptic support.

The right-hand B button (WebXR gamepad button 5) emits the `arm_toggle` event
immediately on press, once per press; `client/app` still owns the global ARM
state.
Pressing B does not calibrate the controllers. Grip authorization and the
B-button ARM state are independent data streams.

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

Only the native PICO APK handles device-specific vibration. The WebSocket
server remains portable across Mac and Linux.

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

After opening the page, click `ENTER MR`. The launcher targets the mainland
Wolvic package `com.cn.igalia.wolvic` and activity
`com.igalia.wolvic.VRBrowserActivity`. For an overseas Wolvic build, override
the package if needed:

```bash
WOLVIC_PACKAGE=com.igalia.wolvic ./examples/input_sources/vr_webxr/open_pico.sh
```

The example uses `127.0.0.1` through SSH/ADB. Directly accessing the node
through a remote IP requires HTTPS/WSS.
