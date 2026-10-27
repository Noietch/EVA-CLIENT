# EVA-SIM 评测

用同一套 EVA-SIM 任务评测不同 Policy 代码库。接入分为两步：准备 SIM、准备 Policy。当前 Pi05 的两条启动命令见下面的「XPolicyLab」示例。

## 评测原理

Policy 根据图像、机器人状态和指令预测动作；EVA-CLIENT 负责通信和动作转换；EVA-SIM 执行动作、判断成功并记录结果。

评测入口透传 `[H, W, 3] uint8` 原图，不 resize、不裁剪、不归一化、不转换颜色通道；模型预处理放在 Policy 内部。同步模式执行 Policy 返回的完整动作 chunk，收到动作 ACK 后继续，不额外轮询关节是否到位。

原图像素和 prompt 不改写；qpos 状态及动作按 float32 协议传输，eef/TCP 按机器人定义做坐标转换。预热响应不执行，任务结束或停止时会中止剩余动作。

每次只评测一个版本，并显式选择 `eef` 或 `qpos`：

| 版本 | `eef` 的实际坐标 | `qpos` |
| --- | --- | --- |
| v1.1 | gripper 末端 | 支持 |
| v1.2 | TCP | 支持 |

两版都传 `--action-mode eef`，Client 按版本选择正确坐标；也都可以传 `--action-mode qpos`。模式必须与 Policy 输出一致。

## Step-1 准备 SIM

EVA-SIM 的 `/cfs/mlp/training/docker/user/hadoop-aipnlp/baizitong/eva-ws/EVA-SIM/scripts/evaluation/` 提供 `eval.sh`、`eval_v1.1.sh`、`eval_v1.2.sh`，只负责启动 SIM + Client。两个版本脚本分别固定任务版本和 `eval` split。

正式评测由 Policy 的评测脚本调用这些入口，用户无需额外启动一次 SIM。服务直接使用 Client 已有的动作配置，不需要手工生成配置；单独调用可用于 fake Policy 和服务调试。

## Policy 适配

### Step-2 准备 Policy

新 Policy 优先兼容 OpenPI 接口：接收 `images`、`state` 和 `prompt`，返回 `[T, D]` 的 `actions`。如果使用 XPolicyLab 原生协议，SIM 服务脚本传 `--policy-type xpolicylab`；图像仍然原样透传，resize 和归一化由 XPolicyLab 内部完成。

### 适配项目实例

每个项目提供一个总体评测脚本，接收 checkpoint、评测版本、`eef/qpos`、GPU 和起始端口，并连接本项目的 Policy 服务。脚本负责分配任务、启动 SIM + Client 和评测 worker、汇总结果，并回收自己启动的进程。

#### XPolicyLab

XPolicyLab 负责模型加载、原图预处理和动作推理。以下两个脚本由 XPolicyLab 自己提供：

- `policy/Pi_05/sft/serve.sh`：加载指定 checkpoint，在选定 GPU 上启动多路 Policy 服务。
- `eval_eva_sim.sh`：连接这些服务，分配任务，启动 SIM + Client 和评测 worker，汇总结果。

当前 Pi05 checkpoint 评测 v1.2/qpos 的启动命令：

```bash
# 1. 启动 8 路 Policy，端口 18000～18007；就绪后命令返回，服务在后台运行
bash /cfs/mlp/training/docker/user/hadoop-aipnlp/baizitong/eva-ws/eva-xpolicylab/policy/Pi_05/sft/serve.sh \
  --checkpoint /cfs/mlp/training/docker/user/hadoop-aipnlp/yinxuanyu/workspace/eva-xpolicylab/policy/Pi_05/work_dirs/dual_yam_sim_v12_epoch10_pro6000d_20261024_v2/checkpoints/24510 \
  --device_id 0,1,2,3,4,5,6,7 \
  --model_num 1 \
  --base_port 18000

# 2. 开始 v1.2 全量评测；自动启动 SIM + Client、分配任务并汇总结果
bash /cfs/mlp/training/docker/user/hadoop-aipnlp/baizitong/eva-ws/eva-xpolicylab/eval_eva_sim.sh \
  --checkpoint /cfs/mlp/training/docker/user/hadoop-aipnlp/yinxuanyu/workspace/eva-xpolicylab/policy/Pi_05/work_dirs/dual_yam_sim_v12_epoch10_pro6000d_20261024_v2/checkpoints/24510 \
  --sim-root /cfs/mlp/training/docker/user/hadoop-aipnlp/baizitong/eva-ws/EVA-SIM \
  --client-root /cfs/mlp/training/docker/user/hadoop-aipnlp/baizitong/eva-ws/EVA-CLIENT \
  --task eval_v1.2 \
  --action-mode qpos \
  --sim-gpus 0,1,2,3,4,5,6,7 \
  --policy-base-port 18000 \
  --base-port 19600
```

参数怎么改：

- GPU：支持 1～8 张卡；同时修改第一条的 `--device_id` 和第二条的 `--sim-gpus`，保持卡数和顺序一致。例如三卡都填 `5,6,7`，仍使用从 18000 开始的三个 Policy 端口。
- 版本：第二条用 `--task eval_v1.1` 或 `--task eval_v1.2`，一次只选一个。
- 动作：用 `--action-mode eef` 或 `qpos`；当前 checkpoint 使用 `qpos`。
- 端口：第一条的 `--base_port` 与第二条的 `--policy-base-port` 必须相同，都是 Policy 起始端口。第二条的 `--base-port` 是 SIM/Client 起始端口，两组不能重叠。
- CPU：默认每路 SIM 分配 8 个逻辑 CPU、Client 分配 4 个，评测 worker 共享 4 个；8 卡合计 100 个。可用 `--sim-cpu-count`、`--client-cpu-count`、`--control-cpu-count` 调整。

第 i 路 Policy 使用 `policy-base-port + i`，示例为 `18000～18007`。SIM/Client 第 i 路使用 `base-port + i*10 + 1..4`，i 从 0 开始：示例前两路分别是 `19601～19604`、`19611～19614`。

结果写入上述 checkpoint 的 `logs/eva_sim/<run-id>/`；进度每 30 秒更新。评测结束会回收 SIM/Client，Policy 服务继续保留。

| 文件 | 内容 |
| --- | --- |
| `summary.json` | 已完成数、成功率、任务平均分、模型未完成数和基础设施失败数 |
| `meta/episode.jsonl` | 每条任务的成功状态、得分/满分、子任务明细、耗时和录制路径 |
| `progress.json` / `performance.jsonl` | 最新进度/历史采样：ETA、episode/min、steps/s、耗时分布、各路推理耗时和 GPU 利用率/显存 |
| `groups/group_*/workers/` | SIM、Client、worker 日志及故障诊断、重试记录 |

任务分数 = 已完成子任务数 / 子任务总数，最终成功条件固定计为最后一个子任务；成功任务为满分。任务平均分见 `summary.json`，子任务总完成率见 `progress.json` / `performance.jsonl` 的 `aggregate`。视频只保存 Policy 实际使用的观测图像，时间戳按仿真时间对齐，完整关闭后可播放。

SIM/Client 退出或任务无进展超时后，脚本会保存诊断、重启该路服务并重试一次；仍失败则记为基础设施失败，继续后续任务。`--resume <run-dir>` 只跳过已完成且录制文件齐全的记录，旧日志和未完成文件会保留；续测必须使用相同的 Client/Policy 推理设置，切换完整 chunk 等设置时应新建一轮评测。

#### EVA-VLA

分支：`dev/eva-sim`。以下均为仓库相对路径：

| 脚本 | 功能 |
| --- | --- |
| `benchmarks/eva_sim/serve_policies.sh` | 加载 checkpoint，在指定 GPU 和起始端口启动多路 EVA-VLA Policy 服务 |
| `benchmarks/eva_sim/eval_yam.sh` | 连接已有 Policy 服务，启动 SIM + Client、分配评测任务并汇总结果 |

在 EVA-VLA 仓库根目录打开两个终端。下面以 8 卡、v1.2/qpos 为例，把两条命令中的 checkpoint 路径替换成自己的：

```bash
# 终端 1：启动 Policy，端口 20000～20007；保持此终端运行
bash benchmarks/eva_sim/serve_policies.sh \
  --ckpt-path work_dirs/YOUR_RUN/checkpoints/YOUR_STEP/hf_ckpt \
  --policy-gpus 0,1,2,3,4,5,6,7 \
  --policy-base-port 20000

# 终端 2：开始评测，自动启动 SIM + Client 和 worker
bash benchmarks/eva_sim/eval_yam.sh \
  --ckpt-path work_dirs/YOUR_RUN/checkpoints/YOUR_STEP/hf_ckpt \
  --sim-root /cfs/mlp/training/docker/user/hadoop-aipnlp/baizitong/eva-ws/EVA-SIM \
  --client-root /cfs/mlp/training/docker/user/hadoop-aipnlp/baizitong/eva-ws/EVA-CLIENT \
  --task eval_v1.2 \
  --action-mode qpos \
  --sim-gpus 0,1,2,3,4,5,6,7 \
  --policy-base-port 20000 \
  --base-port 19600
```

测 v1.1 时改为 `--task eval_v1.1`；动作模式按 checkpoint 选择 `eef` 或 `qpos`。改卡数时，两条命令的 GPU 列表保持数量和顺序一致，两处 `--policy-base-port` 也保持一致。

checkpoint 的 `inference_metadata.yaml` 必须声明匹配的 `control_mode`：`eef` 或 `qpos`。EVA-VLA 只负责自己的 Policy，不负责启动 XPolicyLab。

参考分支：EVA-CLIENT 和 EVA-VLA 使用 `dev/eva-sim`，EVA-SIM 使用 `master`。EVA-VLA 的其他适配参考分支有 `codex/eva-sim-lamfeat-ckpt1078-adapter`、`codex/arx-x5-task-set-scene-integration`。
