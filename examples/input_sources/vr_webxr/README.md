# WebXR VR 输入节点

`node.py` 在服务器上提供 WebXR 页面，并通过 ZMQ 把 PICO/Quest 的输入发送给
EVA Client。下面示例使用页面端口 `43876`、ZMQ 端口 `8765/8766`。

## 服务器启动

```bash
cd "$CLIENT_ROOT"
source .venv/bin/activate

python examples/input_sources/vr_webxr/node.py \
  --host 127.0.0.1 \
  --port 43876 \
  --endpoint tcp://127.0.0.1:8765 \
  --ack-endpoint tcp://127.0.0.1:8766
```

保持进程运行。日志中的 WebXR URL 和 PICO 命令可直接使用。省略 `--token` 时，
节点会生成随机 token，应使用日志中的 URL。

## 输入协议

节点发送给 EVA Client 的 VR 协议版本为 `3`。每只手发送 WebXR 的绝对位姿：
`position` 和 `orientation_xyzw`。相对位姿的参考点和累计 delta 由 EVA Client 的
retargeter 计算。

每只手的 `grip`（WebXR gamepad button 1）由节点本地消费，用于长按切换该手的
`grip_engaged` 状态。长按一次开启，再次长按关闭；raw grip 按钮不会进入 EVA Client，
只有防抖后的状态会随绝对位姿发送。未授权的手由 client 通过 inactive-arm mask 保持
当前机器人关节位置；再次授权时，新的手柄位姿增量会叠加到之前累计的 delta 上。

ARM OFF 或 teleop reset 时，client 清理左右手累计的 delta 和参考位姿。
每次 grip 长按触发 toggle 后，node 会向 WebXR 页面发送对应手柄的短 haptic 反馈；
实际震动效果取决于设备浏览器是否支持 WebXR Gamepad haptics。

右手 B 键（WebXR gamepad button 5）的原有短按 `arm_toggle` 事件保持不变；它仍由
client/app 负责全局 ARM 状态。Grip 权限和 B 键 ARM 状态是两条独立的数据流。

## Ubuntu 端口转发

在连接 PICO 的 Ubuntu 主机上保持 SSH 隧道运行：

```bash
ssh -N \
  -p 22 \
  -o ExitOnForwardFailure=yes \
  -o ServerAliveInterval=30 \
  -o ServerAliveCountMax=3 \
  -L 43876:127.0.0.1:43876 \
  <remote-user>@<remote-host>
```

## Ubuntu 打开或刷新 PICO 页面

用 USB 线将 PICO 连接到 Ubuntu 主机，并查询设备序列号：

```bash
adb devices -l
```

只有一台已授权设备时，运行脚本即可自动识别设备、建立 ADB 反向端口转发并打开
WebXR 页面：

```bash
export VR_TOKEN="<TOKEN_FROM_NODE_LOG>"
./examples/input_sources/vr_webxr/open_pico.sh
```

也可以通过参数或环境变量明确指定设备：

```bash
./examples/input_sources/vr_webxr/open_pico.sh "<PICO_SERIAL>"

export PICO_SERIAL="<PICO_SERIAL>"
export VR_TOKEN="<TOKEN_FROM_NODE_LOG>"
./examples/input_sources/vr_webxr/open_pico.sh
```

强制刷新页面（`-S` 会先停止承载该 URL 的浏览器应用）：

```bash
VR_URL="http://127.0.0.1:43876/?token=<TOKEN_FROM_NODE_LOG>&mode=ar&reload=$(date +%s)"
adb -s "$PICO_SERIAL" shell "am start -S -a android.intent.action.VIEW -d '$VR_URL'"
```

进入页面后点击 `ENTER MR`。这里通过 `127.0.0.1` + SSH/ADB 访问，直接用远程 IP
访问节点时则需要 HTTPS/WSS。
