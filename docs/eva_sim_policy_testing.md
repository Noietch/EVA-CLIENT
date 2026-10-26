# EVA-SIM 评测

目标是让不同 Policy 代码库用同一套 EVA-SIM 任务评测。整体只有两步：准备 SIM，再在 Policy 仓库提供一个总体评测脚本。

流程图：[EVA-SIM 两步评测流程](eva_sim_evaluation_flow.excalidraw)。

## 评测原理

Policy 根据图像、机器人状态和指令预测动作；EVA-CLIENT 负责通信和动作转换；EVA-SIM 执行动作、判断成功并记录结果。

Client 默认透传 `[H, W, 3] uint8` 原图，不 resize、不裁剪、不归一化、不转换颜色通道。模型预处理放在 Policy 内部。同步模式收到动作 ACK 后继续，不额外轮询关节是否到位。

每次只评测一个版本，并显式选择 `eef` 或 `qpos`：

| 版本 | `eef` 的实际坐标 | `qpos` |
| --- | --- | --- |
| v1.1 | gripper 末端 | 支持 |
| v1.2 | TCP | 支持 |

两版都传 `--action-mode eef`，Client 会按版本选择正确坐标。这个映射已用实际数据的 qpos FK 抽样确认。

## Step-1 准备 SIM

EVA-SIM 的 `scripts/evaluation/` 只负责启动 SIM + Client。版本、动作模式、GPU 和端口都由参数指定：

```bash
# v1.2，EEF 实际使用 TCP，默认 GPU 0～7
bash scripts/evaluation/eval_v1.2.sh --action-mode eef

# v1.1，qpos，使用 4 张卡和自定义端口
bash scripts/evaluation/eval_v1.1.sh \
  --action-mode qpos \
  --sim-gpus 0,2,4,6 \
  --policy-port 21000 \
  --base-port 22000
```

`--sim-gpus` 支持任意 1～8 张卡，默认 8 张。第 i 路连接 `policy-port + i`；SIM/Client 使用 `base-port + i*10 + 1..4`。例如 `base-port=22000` 时，前两路分别使用 `22001～22004` 和 `22011～22014`。

脚本直接使用 EVA-CLIENT 仓库已有的 `openpi_eef.py`、`openpi_tcp.py` 或 `openpi_qpos.py`，不生成临时配置。单独执行主要用于 fake Policy 和服务调试；正式评测由下一步的 Policy 总体脚本调用。

## Step-2 准备 Policy

每个 Policy 代码库只需要提供一个总体评测脚本，例如 `eval_eva_sim.sh`。这个脚本需要完成：

1. 接收 checkpoint、版本、`eef/qpos`、GPU 和端口参数。
2. 每个并发槽位启动一个 Policy 服务，或连接已经启动的服务。
3. 生成 task manifest，均匀分配任务并预分配 episode、task 和帧索引。
4. 携带相同的 manifest 和 worker rank 调用 `eval_v1.1.sh` 或 `eval_v1.2.sh`，再启动 EVA-SIM 的 `scripts/evaluate.py` worker。
5. 监控子进程，失败时保留日志并清理本次启动的服务；完成后汇总 `summary.json`。

新 Policy 优先兼容 OpenPI 接口：接收 `images`、`state` 和 `prompt`，返回 `[T, D]` 的 `actions`。如果使用 XPolicyLab 原生协议，SIM 服务脚本传 `--policy-type xpolicylab`；图像仍然原样透传，resize 和归一化由 XPolicyLab 内部完成。

### 适配项目实例

#### EVA-VLA

EVA-VLA 已提供总体脚本：

```bash
# v1.2 EEF，默认 8 张卡
bash benchmarks/eva_sim/eval_yam.sh \
  --ckpt-path /path/to/checkpoint \
  --task eval_v1.2 \
  --action-mode eef

# v1.1 qpos，指定 4 张卡和端口
bash benchmarks/eva_sim/eval_yam.sh \
  --ckpt-path /path/to/checkpoint \
  --task eval_v1.1 \
  --action-mode qpos \
  --sim-gpus 0,2,4,6 \
  --policy-port 21000 \
  --base-port 22000
```

checkpoint 的 `inference_metadata.yaml` 必须声明匹配的 `control_mode`：`eef` 或 `qpos`。结果写入 `<checkpoint>/logs/eva_sim/<run-id>/`，先看 `summary.json`，具体失败再看 `meta/episode.jsonl`、视频和 worker 日志。

#### XPolicyLab

XPolicyLab 需要按上面的五项职责提供自己的总体脚本。原生服务连接 SIM 时使用：

```bash
bash scripts/evaluation/eval_v1.2.sh \
  --action-mode eef \
  --sim-gpus 0,2,4,6 \
  --policy-type xpolicylab \
  --policy-port 21000 \
  --base-port 22000
```

这条命令只准备 SIM + Client；XPolicyLab 的总体脚本还要负责启动 Policy、分配任务、调用 worker 和汇总结果。

当前分支：EVA-CLIENT 和 EVA-VLA 使用 `dev/eva-sim`，EVA-SIM 使用 `dev/evaluation-services`。
