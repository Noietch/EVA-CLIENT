# EVA SIM 评测怎么跑

这套评测让不同的 policy 在同一套仿真任务上完成操作。公共流程负责连接 policy、EVA-CLIENT 和 EVA-SIM；各代码库的 checkpoint、服务启动和总体评测入口见后面的“Policy 适配”。

可编辑流程图：[EVA-SIM 多 Policy 1～8 卡评测流程](eva_sim_evaluation_flow.excalidraw)。

公共服务使用 EVA-CLIENT 的 `dev/eva-sim` 和 EVA-SIM 的 `dev/evaluation-services`。EVA-SIM 的服务脚本尚未合入远端 `master`。

## 评测原理

三个部分各做一件事：policy 根据画面和机器人状态预测动作，EVA-CLIENT 传递观察并把预测动作转换为机器人控制命令，EVA-SIM 执行动作并判断任务是否成功。

每个任务重复这个过程：SIM 给出当前画面和状态 → Client 请求 policy → policy 返回一段动作 → Client 交给 SIM 执行 → 读取下一次观察。直到任务成功，或者用完规定的步数。

**Client 默认透传原图**，不 resize、不裁剪、不归一化，也不改变图像布局和颜色通道。模型需要的图像处理由 policy 内部完成。同步模式收到 SIM 的动作执行确认后就继续，不再额外轮询关节是否到位。

EVA-SIM 的 `scripts/evaluation/` 只启动 SIM + Client，连接已有 policy。各 policy 代码库的总体评测入口负责启动模型服务、准备任务、调用评测 worker 并汇总结果。

## Step-1 选评测版本和动作坐标

每次只测一个版本，在下面两个版本入口中选一个：

| EVA-SIM 服务入口 | 固定版本 | Client 动作坐标 |
| --- | --- | --- |
| `scripts/evaluation/eval_v1.1.sh` | v1.1 | `qpos` 或 `eef`；`eef` 是旧 `gripper` 末端 |
| `scripts/evaluation/eval_v1.2.sh` | v1.2 | `qpos` 或 `eef`；`eef` 是 TCP |
| `scripts/evaluation/eval.sh` | 显式指定 `--task` | 显式指定 `--action-mode eef/qpos` |

版本脚本固定版本和 eval split，动作模式必须由用户传 `--action-mode eef` 或 `--action-mode qpos`。两个版本都能用 qpos 评测；选择 eef 时，虽然传入参数相同，v1.1 实际使用 gripper EEF，v1.2 实际使用 TCP。

本次直接读取 v1.1 split 的 train/eval，以及 v1.2 train、eval_pos、转换输出各一个 Parquet 的前 8 帧，比较左右臂的观察和动作。v1.1 对 `gripper` FK 的最大位置误差为 `1.43e-7 m`，v1.2 对 TCP 为 `2.06e-8 m`；用错 link 则相差约 `0.14593 m`、`π rad`。这是有界样本证据，不是全量校验；当前本地 `splits` 只有 v1.1，v1.2 结论来自实际 v1.2 数据目录。

核对的是 `observations.state.qpos/eef` 和 `action.qpos/eef`：qpos 为 14 维，EEF 为 16 维，四元数顺序为 WXYZ。采样数据位于 EVA-CLIENT 的 `datasets/data_collection/`：v1.1 来自 `splits/v1.1` 的 train/eval manifest；v1.2 来自 `datasets/eva_sim/dual_yam/{train_v1.2,eval_v1.2_pos}` 和 `outputs/train_v1.2`。判断依据是同帧 qpos 的 FK 与记录位姿吻合，目录名只用于定位样本。

## Step-2 准备 policy 服务

先在 policy 代码库中启动服务，具体命令见“Policy 适配”。新 policy 优先兼容 Client 已有的 OpenPI 接口；服务接收 `images`、`state` 和 `prompt`，返回形状为 `[T, D]` 的 `actions`。相机名、状态顺序、动作维度和坐标必须与模型对齐。

选择 N 张 SIM 卡时准备 N 个 policy 端点。Client 按槽位连接 `policy-port + slot`，默认从 `20000` 开始；例如 4 路从 `21000` 开始，就是 `21000～21003`。Policy 使用哪些卡，由各代码库的服务入口决定。

## Step-3 启动 SIM 和 Client

在 EVA-SIM 根目录执行，默认使用 GPU 0～7 共 8 张卡，也可指定任意 1～8 张卡，卡号不要求连续：

```bash
# v1.2 EE（实际为 TCP），默认 8 路，连接已有 20000～20007 Policy
bash scripts/evaluation/eval_v1.2.sh --action-mode eef

# v1.1 qpos，指定 4 张卡和端口
bash scripts/evaluation/eval_v1.1.sh \
  --action-mode qpos --sim-gpus 0,2,4,6 \
  --policy-host 127.0.0.1 --policy-port 21000 --base-port 22000
```

`--base-port` 默认是 `19600`。每组使用 `base-port + slot*10 + 1..4`：基础端口为 `22000` 时，第一组用 `22001～22004`，第二组用 `22011～22014`。这些端口不能与 policy 或已有服务重叠。

Client 在 CPU 上运行。服务脚本直接选择仓库已有的 `openpi_eef.py`、`openpi_tcp.py` 或 `openpi_qpos.py`，端口和评测参数通过 `src/main.py --eval-service` 传入，不生成临时配置文件。运行时使用 `transport.resize=False`、`image_layout="hwc"`、`convert_bgr_to_rgb=False`，按 `[H, W, 3] uint8` 发送原图。`resize_pad=False` 只表示不补边，不能代替关闭 resize。

加 `--dry-run` 可查看选中的 preset、卡号和命令。Ctrl-C 关闭本次启动的 SIM 和 Client，保留外部 policy。这一步只让服务就绪，不会自动开始评测；使用代码库的一键总体入口时，由它调用服务层，避免重复启动同一组端口。

## Step-4 驱动任务

总体评测入口准备 task manifest，分配任务，然后调用 EVA-SIM 的 [评测 worker](../../EVA-SIM/scripts/evaluate.py)。manifest 需要预分配 episode、task 和帧索引；SIM 与 worker 必须使用相同 manifest 和 worker rank，参数约定见 [EVA-SIM README](../../EVA-SIM/README.md#evaluatepy)。

每个任务先重置场景、设置指令，再启动推理和动作执行；结束后保存轨迹、视频和结果，进入下一个任务。只启动服务还不具备这部分任务编排，新的 policy 总体入口也需要接入它。

## Step-5 检查结果

服务配置、端点和日志保存在服务层的 `--output` 目录，默认是 `EVA-SIM/work_dirs/eval_services/<时间戳>/`。各 worker 写入 `trials.jsonl`，SIM 写入 `recordings.jsonl`；最终结果目录和汇总由 policy 的总体入口负责。

判断是否稳定，要看任务有没有漏跑、重复、卡死或执行超时，再看模型成功率。尤其检查 `action_timeout`，它可能被记成普通任务失败，不能只看完成数量。

## Policy 适配

### EVA-VLA

使用 Sankuai `robotics/eva-vla` 的 `dev/eva-sim` 分支。总体入口 `benchmarks/eva_sim/eval.sh` 负责启动或复用 policy、按机器人和物理配置分组、按任务最大步数尽量均匀分配任务、调用 SIM 服务和 worker，最后汇总结果。任务少于卡数时，只启动需要的执行链路。

#### Step-1 选好 checkpoint 和评测版本

在 EVA-VLA 根目录执行：

```bash
# 默认：只测 v1.2，使用 GPU 0～7 共 8 张卡
bash benchmarks/eva_sim/eval_yam.sh \
  --ckpt-path /path/to/hf_checkpoint --action-mode eef

# v1.2 改用 qpos，并指定 4 张卡
bash benchmarks/eva_sim/eval_yam.sh \
  --ckpt-path /path/to/hf_checkpoint --action-mode qpos --sim-gpus 0,2,4,6

# 传参数切换成 v1.1，仍然默认使用 8 张卡
bash benchmarks/eva_sim/eval_yam.sh \
  --ckpt-path /path/to/hf_checkpoint --action-mode eef --task eval_v1.1
```

这里传导出的 HF checkpoint，目录里需要有 `inference_metadata.yaml`。**每次只测一个版本**，用 `--task` 在 v1.1 和 v1.2 之间切换，不传时默认只测 v1.2。同时选择 `eval_v1.1` 和 `eval_v1.2` 会直接报错；同一版本内部仍可传多个任务分片。切换版本不需要修改脚本或重新手动部署服务。

`--action-mode` 只有 `eef` 和 `qpos` 两个选择，并且必须显式传入。`eef` checkpoint 的 `control_mode` 应为 `eef`，版本会进一步决定 Client 使用 gripper EEF 还是 TCP；`qpos` checkpoint 的 `control_mode` 应为 `qpos`，v1.1 和 v1.2 都支持。总体入口会拒绝参数与 checkpoint 元数据不一致。当前默认使用 TX 的 baizitong 环境和 GPU 0–7，也可以通过 `--sim-gpus` 选择任意 1～8 张卡，卡号不要求连续。

#### Step-2 选择卡数并启动 policy 服务

脚本在每张卡上各加载一份相同的 checkpoint。`--sim-gpus` 决定并发槽位数；不传 `--policy-gpus` 时，policy 默认使用同一组卡。如果 policy 和 SIM 要分开用卡，可以分别指定两个列表，但数量必须相同，并按列表位置一一配对。

```bash
# 1 张卡：policy、Client、SIM 都只启动一组
bash benchmarks/eva_sim/eval.sh \
  --ckpt-path /path/to/hf_checkpoint --action-mode eef --sim-gpus 3

# 4 张不连续的卡
bash benchmarks/eva_sim/eval.sh \
  --ckpt-path /path/to/hf_checkpoint --action-mode qpos --sim-gpus 0,2,4,6

# policy 和 SIM 分开用卡，两个列表按位置配对
bash benchmarks/eva_sim/eval.sh --ckpt-path /path/to/hf_checkpoint \
  --action-mode eef --sim-gpus 0,2,4,6 --policy-gpus 1,3,5,7

# 不传 --sim-gpus 时默认使用 0,1,2,3,4,5,6,7
```

policy 服务默认从端口 `20000` 开始，启动 N 个服务就使用 `20000` 到 `20000 + N - 1`。脚本确认所有服务都能连接后再开始评测，不用手动开多个终端。

需要将 Policy 服务单独保持运行时，在 EVA-VLA 中分两步执行；两条命令的端口和服务数量要对应：

```bash
# 终端一：只启动 4 个 EVA-VLA Policy 服务
bash benchmarks/eva_sim/serve_policies.sh \
  --ckpt-path /path/to/hf_checkpoint --policy-gpus 0,2,4,6 --policy-port 21000

# 终端二：总体评测复用已有服务，退出时保留这些外部 Policy
bash benchmarks/eva_sim/eval.sh \
  --ckpt-path /path/to/hf_checkpoint --task eval_v1.2 \
  --action-mode eef --sim-gpus 0,2,4,6 \
  --external-policy --policy-host 127.0.0.1 --policy-port 21000
```

想换端口，用 `--policy-port` 指定起始值。例如 4 张卡从 `21000` 开始，四个服务就会使用 `21000–21003`，客户端会自动连接对应端口：

```bash
bash benchmarks/eva_sim/eval.sh \
  --ckpt-path /path/to/hf_checkpoint \
  --task eval_v1.1 \
  --action-mode qpos \
  --sim-gpus 0,2,4,6 \
  --policy-port 21000
```

SIM 和 Client 之间的通信端口由 `--base-port` 控制，默认是 `19600`。每组占用四个端口，各组起始值相差 10：第一组用 `19601–19604`，第二组用 `19611–19614`，依此类推。需要换时，在上面的命令后加 `--base-port 22000`，第一组就改用 `22001–22004`。两套端口不要重叠，也不要占用已有服务的端口。

#### 结果和运行约定

结果保存在 `<checkpoint>/logs/eva_sim/<run-id>/`。先看 `summary.json` 中的完成情况和成功率，再按 `meta/episode.jsonl`、视频和日志排查具体任务。评测结束后关闭本次启动的服务，保留通过 `--external-policy` 连接的外部 policy。

服务退出或等待超时时会报错并保留日志；重新启动使用新的结果目录，目前没有自动续跑。默认 policy 与 SIM 共用对应 GPU，显存是否够用取决于 checkpoint 和场景。

Kit 运行目录默认是 `~/.cache/eva_sim/<hostname>/gpu_<gpu_uuid>/kit/`，GPU 独占锁覆盖 SIM 生命周期；同机同 GPU 的 worker 重启会复用缓存，首次启动仍需编译。`EVA_SIM_CACHE_ROOT` 可覆盖缓存根目录。

当前一键入口只启动 EVA-VLA 服务，尚未接入 XPolicyLab 的服务启动或 OpenPI 适配。已通过本地入口参数、任务分片、槽位绑定、故障退出和清理检查；真实 checkpoint 与八路 SIM 的完整评测和长时间稳定性仍需实测。

#### 可以参考的分支

下面这些分支都在 Sankuai 的 `robotics/eva-vla` 仓库中。它们解决的问题不同，参考实现时按需要看对应部分，不要直接把整个分支合进当前评测分支。

| 分支 | 适合参考的内容 |
| --- | --- |
| `dev/eva-sim` | 当前 EVA-SIM 评测主线。包含 1～8 路 policy、Client、SIM 编排，v1.1/v1.2 切换、端口分配、同步执行和结果汇总；新改动以这个分支为准。 |
| `dev/eva-sim-integration` | 较早的 EVA-SIM 集成和 AgileX VR SFT 配置。适合参考数据集与训练配置，评测启动方式已经旧于 `dev/eva-sim`。 |
| `master-robolab-dev` | RoboLab 的 OpenPI 初始化、`deploy/serve.py` 和 OpenPI proxy。新 policy 想兼容 OpenPI 接口时，可以参考服务适配方式。 |
| `feature/genie_sim_sft` | 另一套仿真 benchmark 的环境解析、评测入口和 SFT 配置。适合参考“训练配置加独立 benchmark 入口”的组织方式，不复用它的任务协议。 |

查看参考分支时使用 `git show <分支>:<文件>` 或单独创建 worktree，避免切换当前正在评测的 checkout。

### XPolicyLab

优先复用 OpenPI 协议。使用原生 WebSocket 服务时，Client 也已有 `xpolicylab` 接口，只转换协议字段，不负责 resize；缩放、裁剪和归一化由 XPolicyLab 内部的模型适配代码完成。

先单独启动 XPolicyLab 服务，再在 EVA-SIM 根目录执行：

```bash
bash scripts/evaluation/eval_v1.2.sh \
  --action-mode eef --sim-gpus 0,2,4,6 \
  --policy-type xpolicylab --policy-port 21000 --base-port 22000
```

选择 `--action-mode eef` 时，v1.1 和 v1.2 的 `action_type` 都是 `ee`，Client 分别继承 `openpi_eef.py` 和 `openpi_tcp.py`；选择 `qpos` 时，两版都使用 joint action。原生握手不自动协商坐标，服务端模型必须与参数和版本一致。这个命令只启动 SIM + Client；任务分配、worker 调用和汇总仍需由总体入口完成。

### 接入其他 policy 或新增步骤

新 policy 先用 fake 服务测一条任务，核对动作格式和原图尺寸、像素，再逐步扩到所需卡数，最后替换成真实模型。OpenPI 可复用 [fake server](../examples/policy/fake_policy/fake_policy_server.py)，XPolicyLab 原生协议有 `demo_policy` 可用。

新增任务时，在 EVA-SIM 准备任务配置，用共享服务入口 `eval.sh --task 新任务名 --action-mode eef/qpos`，并让总体入口生成相应 manifest。

新增处理步骤时，整次评测只执行一次的逻辑放在各 policy 的总体入口；每个任务都执行的逻辑放在 EVA-SIM 的单任务评测流程。EVA-VLA 的总体入口示例是 [eval.py](../../EVA-VLA/benchmarks/eva_sim/eval.py)。

## 现在验证到了哪一步

服务层已通过真实 Client 配置加载，以及 fake 子进程的启动、超时、故障与退出清理测试。OpenPI 和 XPolicyLab 两种协议的八路通信测试共完成 800 次推理。

原图透传也已验证：经过生产观察构造后，两种协议收到的三路图像尺寸、类型和像素均与输入一致；两侧真实 WebSocket 服务都已完成往返验证。这些证据不代表真实模型与多路 SIM 的完整评测已经通过。
