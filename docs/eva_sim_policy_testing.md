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

## Policy 适配

### Step-2 准备 Policy

每个 Policy 代码库只需要提供一个总体评测脚本，例如 `eval_eva_sim.sh`。这个脚本需要完成：

1. 接收 checkpoint、版本、`eef/qpos`、GPU 和端口参数。
2. 每个并发槽位启动一个 Policy 服务，或连接已经启动的服务。
3. 生成 task manifest，均匀分配任务并预分配 episode、task 和帧索引。
4. 携带相同的 manifest 和 worker rank 调用 `eval_v1.1.sh` 或 `eval_v1.2.sh`，再启动 EVA-SIM 的 `scripts/evaluate.py` worker。
5. 监控子进程，失败时保留日志并清理本次启动的服务；完成后汇总 `summary.json`。

同一轮 manifest 中的 `(task_item_index, layout_index)` 必须唯一，不能把同一个 case 重复分给多个 worker，否则录制器无法唯一确定 episode。

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

XPolicyLab 需要按上面的五项职责提供自己的总体脚本。当前 Pi05/qpos 服务已用下面的命令跑通：

```bash
# XPolicyLab 根目录：默认 8 张卡，端口 18000～18007
cd /cfs/mlp/training/docker/user/hadoop-aipnlp/yinxuanyu/workspace/eva-xpolicylab
bash policy/Pi_05/sft/serve.sh \
  --checkpoint policy/Pi_05/work_dirs/dual_yam_sim_v12_epoch10_pro6000d_20261024_v2/checkpoints/24510 \
  --device_id 0,1,2,3,4,5,6,7 \
  --model_num 1 \
  --base_port 18000

# EVA-VLA 根目录：一条命令完成 manifest、SIM/Client、worker 和汇总
cd /cfs/mlp/training/docker/user/hadoop-aipnlp/baizitong/eva-ws/EVA-VLA
bash benchmarks/eva_sim/eval_yam.sh \
  --ckpt-path /cfs/mlp/training/docker/user/hadoop-aipnlp/yinxuanyu/workspace/eva-xpolicylab/policy/Pi_05/work_dirs/dual_yam_sim_v12_epoch10_pro6000d_20261024_v2/checkpoints/24510 \
  --action-mode qpos \
  --external-policy \
  --policy-type xpolicylab \
  --policy-port 18000 \
  --base-port 19600
```

EVA-SIM 单独入口未传 `--sim-cpus` 时，每个 SIM 默认使用 4 个 Kit/PXR 工作线程；上面的 EVA-VLA 总体入口默认传 `--sim-cpu-count 8`。2026-10-26 的服务层探针完成 8/8 条评测，7 条成功、1 条达到 482 步上限，基础设施失败为 0，共生成 8 个 Parquet 和 24 个视频，证据位于 EVA-SIM 的 `temp/xpolicylab_pi05_probe8_final_20261026/`。

上面的第二条命令默认运行 v1.2 全量评测。2026-10-27 基于相同配置额外传入 `--max-trials-per-task 8` 完成有界探针：8/8 条完整落盘，5 条成功、3 条模型失败、基础设施失败为 0。证据位于 checkpoint 的 `logs/eva_sim/xpolicylab_pi05_v12_qpos_probe8b_20261027/`。

用户只需要执行上面两个脚本。第二个脚本会检查 8 路 Policy 握手，自动生成 manifest，启动和回收 SIM/Client，执行 worker，并在 checkpoint 的 `logs/eva_sim/<run-id>/summary.json` 写最终汇总；不需要手工生成配置或单独调用 `scripts/evaluate.py`。

参考分支：EVA-CLIENT 和 EVA-VLA 使用 `dev/eva-sim`，EVA-SIM 使用 `master`。
