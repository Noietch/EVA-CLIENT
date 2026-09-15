# EVA-VR Native PICO Input

EVA-CLIENT keeps both PICO input paths:

| Path | Method | Use |
|---|---|---|
| [`../vr_webxr`](../vr_webxr/) | PICO Browser + WebXR + WebSocket | Browser-only input and quick protocol checks |
| [EVA-VR repository](https://github.com/Noietch/EVA-VR) | Native OpenXR APK + WebSocket | Native input, status panel, and reliable haptics |

原生客户端的独立开源仓库：

- [EVA-VR 源码](https://github.com/Noietch/EVA-VR)
- [EVA-VR 最新 APK Release](https://github.com/Noietch/EVA-VR/releases/latest)

## Host Node

Both clients connect to `vr_webxr/node.py`. Mac and Linux are supported.
The native path uses ADB reverse for a USB-connected PICO or a LAN URL.

在 `EVA-CLIENT` 根目录启动：

```bash
cd /Users/yhq/Workspace/EVA-CLIENT
source .venv/bin/activate

python examples/input_sources/vr_webxr/node.py \
  --host 127.0.0.1 \
  --port 43876 \
  --token eva \
  --endpoint tcp://127.0.0.1:8765 \
  --ack-endpoint tcp://127.0.0.1:8766
```

如果 PICO 通过局域网直接访问 Mac/Linux，不使用 ADB reverse，把主机监听地址改为：

```bash
python examples/input_sources/vr_webxr/node.py \
  --host 0.0.0.0 \
  --port 43876 \
  --token YOUR_TOKEN \
  --endpoint tcp://127.0.0.1:8765 \
  --ack-endpoint tcp://127.0.0.1:8766
```

The native client uses the fixed host token `eva`.

## One-Click Native Start

The device service uses `eva_pico` as the native teleop option. After selecting
Robot, `EVA-VR (PICO)`, and a camera in the Device panel, press Start once. The
service starts the WebSocket node, establishes ADB reverse, and launches the
installed `org.eva.pico.input` APK.

The same launcher can be used directly:

```bash
PICO_SERIAL="$(adb devices | awk 'NR==2 && $2=="device" {print $1}')" \
  examples/input_sources/eva-pico/start.sh --launch-only
```

The app connects to:

```text
ws://127.0.0.1:43876/ws?token=eva
```

## WebXR Alternative

这个方案不需要安装原生 APK，使用 PICO Browser 或已有的 WebXR 浏览器
打开页面。详细说明和已有启动脚本见 [`../vr_webxr/README.md`](../vr_webxr/README.md)。

USB/ADB reverse 启动：

```bash
export PICO_SERIAL="$(adb devices | awk 'NR==2 && $2=="device" {print $1}')"
export VR_TOKEN=YOUR_TOKEN
examples/input_sources/vr_webxr/open_pico.sh
```

WebXR 可以传姿态、按钮和摇杆数据，但手柄震动依赖浏览器是否开放
WebXR Gamepad haptics。PICO 4 Ultra 当前浏览器实测不保证震动，因此需要
震动时使用下面的原生方案。

## Native APK

### 直接安装 Release APK

1. 在电脑上下载 [最新 APK](https://github.com/Noietch/EVA-VR/releases/latest)。
2. 在 PICO 4 Ultra 开启 USB 调试并连接电脑。
3. 确认设备在线：

```bash
adb devices -l
```

4. 安装 APK：

```bash
adb install -r EVA-VR-v0.2.0.apk
```

5. 启动本地 haptic 测试：

```bash
adb shell am force-stop org.eva.pico.input
adb shell am start -n org.eva.pico.input/.MainActivity \
  --ez haptic_test true
```

此时应用名为 **EVA-VR**。正前方会显示状态面板，包含：

- Head、Left、Right 的 XYZ；
- Roll、Pitch、Yaw；
- 左右手柄 trigger、squeeze、摇杆、A/B/X/Y、menu 的按键/触摸状态；
- `HOST: CONNECTED` 或 `HOST: DISCONNECTED`；
- EVA 图标和原生 OpenXR 手柄震动。

### 通过 ADB reverse 连接 EVA

如果 `node.py` 在当前电脑的 `127.0.0.1:43876` 运行：

```bash
adb reverse tcp:43876 tcp:43876
adb shell am force-stop org.eva.pico.input
adb shell am start -n org.eva.pico.input/.MainActivity \
  --es server_url \
  'ws://127.0.0.1:43876/ws?token=YOUR_TOKEN'
```

查看连接和震动日志：

```bash
adb logcat -s "EVA-VR" "OpenXR"
```

正常连接会看到：

```text
WebSocket connected
```

正常震动会看到：

```text
EVA-VR haptic hand=1 amplitude=1.00 duration_ms=300 result=XR_SUCCESS
```

### 通过局域网连接 EVA

让 PICO 和 Mac/Linux 在同一个局域网，假设主机 IP 是
`192.168.1.20`：

```bash
adb shell am start -n org.eva.pico.input/.MainActivity \
  --es server_url \
  'ws://192.168.1.20:43876/ws?token=YOUR_TOKEN'
```

如果连接失败，检查主机防火墙是否放行 TCP `43876`，并确认 node 使用了
`--host 0.0.0.0`。

### 从源码构建

```bash
git clone https://github.com/Noietch/EVA-VR.git
cd EVA-VR
export JAVA_HOME=/opt/homebrew/opt/openjdk@17/libexec/openjdk.jdk/Contents/Home
export ANDROID_HOME="$HOME/Library/Android/sdk"
./build.sh
adb install -r app/build/outputs/apk/debug/app-debug.apk
```

源码工程已经是完整工程，不需要 `vendor/OpenXR_Demos`，也不需要 Unity。
`EVA-CLIENT` 本身不跟踪这套原生工程源码。

## 停止与重新启动

停止应用：

```bash
adb shell am force-stop org.eva.pico.input
```

重新连接时，先重新建立 reverse：

```bash
adb reverse tcp:43876 tcp:43876
```

然后重新执行原生 APK 的 `am start` 命令即可。

## 排错顺序

1. `adb devices -l` 确认 PICO 已授权。
2. 检查 `node.py` 是否监听 `43876`。
3. 检查 token 是否一致。
4. ADB reverse 模式下确认 `adb reverse tcp:43876 tcp:43876` 已执行。
5. 看原生面板是否从 `HOST: DISCONNECTED` 变为 `HOST: CONNECTED`。
6. 看 `adb logcat` 是否出现 `XR_SUCCESS`。
