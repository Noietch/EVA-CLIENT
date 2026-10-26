# EVA SIM 评测怎么跑

这套评测是让不同的 policy 在同一套仿真任务上完成操作。先以 EVA-VLA 跑通，再按同样的方式接入 XPolicyLab 等其他代码库。

可编辑流程图：[EVA-SIM 多 Policy 1～8 卡评测流程](eva_sim_evaluation_flow.excalidraw)。

## 评测原理

三个部分各做一件事：policy 根据画面和机器人状态预测动作，EVA-CLIENT 把数据和动作传过去，EVA-SIM 执行动作并判断任务是否成功。

新 policy 对接 EVA-CLIENT 时，优先复用已有的 OpenPI 接口。**Client 默认透传原图**，不 resize、不裁剪、不归一化，也不改变图像布局和颜色通道。EVA-VLA 和 XPolicyLab 都接收原图，模型需要的处理放在各自的模型适配代码里完成。

每个任务都会重复这个过程：SIM 给出当前画面和状态 → Client 请求 policy → policy 返回一段动作 → Client 交给 SIM 执行 → 读取下一次观察。直到任务成功，或者用完规定的步数。

选择 N 张卡就同时跑 N 组这样的过程，N 可以是 1～8。每组有自己的 policy 服务、Client 和 SIM，各自处理分到的任务；脚本按任务最大步数尽量均匀分配负载。

## Step-1 选好 checkpoint 和评测版本

在 EVA-VLA 根目录执行：

```bash
# 只测 v1.1
bash benchmarks/eva_sim/eval.sh --ckpt-path /path/to/hf_checkpoint --task eval_v1.1

# 只测 v1.2
bash benchmarks/eva_sim/eval.sh --ckpt-path /path/to/hf_checkpoint --task eval_v1.2

# 不传版本时，默认只测 v1.2
bash benchmarks/eva_sim/eval.sh --ckpt-path /path/to/hf_checkpoint
```

这里传导出的 HF checkpoint，目录里需要有 `inference_metadata.yaml`。**每次只测一个版本**，用 `--task` 在 v1.1 和 v1.2 之间切换，不传时默认只测 v1.2。同时选择 `eval_v1.1` 和 `eval_v1.2` 会直接报错；同一版本内部仍可传多个任务分片。切换版本不需要修改脚本或重新手动部署服务。

脚本按 checkpoint 中的配置选择关节或末端动作空间。当前默认使用 TX 的 baizitong 环境和 GPU 0–7，也可以通过 `--sim-gpus` 选择任意 1～8 张卡，卡号不要求连续。

## Step-2 选择卡数并启动 policy 服务

脚本在每张卡上各加载一份相同的 checkpoint。`--sim-gpus` 决定并发槽位数；不传 `--policy-gpus` 时，policy 默认使用同一组卡。如果 policy 和 SIM 要分开用卡，可以分别指定两个列表，但数量必须相同，并按列表位置一一配对。

```bash
# 1 张卡：policy、Client、SIM 都只启动一组
bash benchmarks/eva_sim/eval.sh --ckpt-path /path/to/hf_checkpoint --sim-gpus 3

# 4 张不连续的卡
bash benchmarks/eva_sim/eval.sh --ckpt-path /path/to/hf_checkpoint --sim-gpus 0,2,4,6

# policy 和 SIM 分开用卡，两个列表按位置配对
bash benchmarks/eva_sim/eval.sh --ckpt-path /path/to/hf_checkpoint \
  --sim-gpus 0,2,4,6 --policy-gpus 1,3,5,7

# 不传 --sim-gpus 时默认使用 0,1,2,3,4,5,6,7
```

policy 服务默认从端口 `20000` 开始，启动 N 个服务就使用 `20000` 到 `20000 + N - 1`。脚本确认所有服务都能连接后再开始评测，不用手动开多个终端。

想换端口，用 `--policy-port` 指定起始值。例如 4 张卡从 `21000` 开始，四个服务就会使用 `21000–21003`，客户端会自动连接对应端口：

```bash
bash benchmarks/eva_sim/eval.sh \
  --ckpt-path /path/to/hf_checkpoint \
  --task eval_v1.1 \
  --sim-gpus 0,2,4,6 \
  --policy-port 21000
```

SIM 和 Client 之间的通信端口由 `--base-port` 控制，默认是 `19600`。每组占用四个端口，各组起始值相差 10：第一组用 `19601–19604`，第二组用 `19611–19614`，依此类推。需要换时，在上面的命令后加 `--base-port 22000`，第一组就改用 `22001–22004`。两套端口不要重叠，也不要占用已有服务的端口。

## Step-3 启动 Client 和 SIM 并分配任务

每个 Client 固定连接自己的 policy 服务和 SIM。脚本按机器人和物理配置分组，使用 1～8 路并行，并根据各任务的最大步数尽量均匀拆分；任务少于卡数时，只启动需要的执行链路。

Client 的默认配置是 `transport.resize=False`、`image_layout="hwc"`、`convert_bgr_to_rgb=False`，按 `[H, W, 3] uint8` 发送 SIM 原图。EVA-VLA 生成的评测配置也显式使用这些值。`resize_pad=False` 只是关闭补边，不能用它来表示“不 resize”；需要由 Client 缩放的其他场景必须显式开启 `resize=True`。

Client 在 CPU 上运行。默认 policy 和 SIM 共用对应的 GPU，因此实际能否放下，还要看 checkpoint 和场景占用的显存。

## Step-4 逐个执行任务

每个任务开始前，先重置场景、设置任务指令，再开始推理和执行。同步模式已经配好：收到 SIM 的动作执行确认后就继续，**不再额外轮询关节是否到位**。

任务结束后保存轨迹、视频和结果，再进入下一个任务。服务退出或等待超时时会报错，结束本次评测并保留日志。失败后重新启动会使用新的结果目录，目前没有自动接着上次进度跑的功能。

## Step-5 查看结果

结果保存在：

```text
<checkpoint>/logs/eva_sim/<run-id>/
```

先看 `summary.json`：里面有本次所选版本的完成情况和成功率。需要排查具体任务时，再看 `meta/episode.jsonl`、视频和日志。全部结束后，脚本会关闭本次启动的服务和客户端。

判断系统是否稳定，要看任务有没有漏跑、重复、卡死或执行超时；判断模型好不好，再看成功率。尤其要检查 `action_timeout`，它可能被记成普通的任务失败，不能只看完成数量。

## 后面怎么接新的内容

**接一个新的 policy，例如 XPolicyLab：**优先让新服务兼容 Client 已有的 OpenPI 协议，复用 `policy.type="openpi"`。Client 发出 `images`、`state` 和 `prompt`，服务返回 `actions`，形状为 `[T, D]`。相机命名、机器人状态顺序和动作空间要与模型对齐；图像按原尺寸传入，由 XPolicyLab 内部的具体模型适配代码完成 resize、裁剪和归一化。

Client 也已有原生 `xpolicylab` 接口，可直接连接它的 WebSocket 服务；这个接口负责转换字段，不负责 resize。两种协议不同，使用原生接口时需要相应配置。目前 EVA-VLA 的一键脚本只启动 EVA-VLA 服务，尚未接入 XPolicyLab 的服务启动或 OpenPI 适配。

**加一套评测任务：**在 EVA-SIM 中准备任务配置，再通过 `--task 新任务名` 指定，每次只选一套。

**加一个处理步骤：**先明确它是整次评测只做一次，还是每个任务都要做。前者放在 EVA-VLA 的 [批量评测入口](../../EVA-VLA/benchmarks/eva_sim/eval.py)，例如启动前检查或结束后统计；后者放在 EVA-SIM 的 [单任务评测流程](../../EVA-SIM/scripts/evaluate.py)，例如重置后检查场景。

新接入的 policy 先用 fake 服务测一条任务，核对服务收到的图像尺寸和像素与 SIM 输出一致，再逐步扩到需要的卡数，最后替换成真实模型跑全量。OpenPI 可以复用 [fake server](../examples/policy/fake_policy/fake_policy_server.py)，XPolicyLab 原生协议有 `demo_policy` 可用；假动作也要符合对应机器人的动作格式。

## EVA-VLA 里可以参考的分支

下面这些分支都在 Sankuai 的 `robotics/eva-vla` 仓库中。它们解决的问题不同，参考实现时按需要看对应部分，不要直接把整个分支合进当前评测分支。

| 分支 | 适合参考的内容 |
| --- | --- |
| `dev/eva-sim` | 当前 EVA-SIM 评测主线。包含 1～8 路 policy、Client、SIM 编排，v1.1/v1.2 切换、端口分配、同步执行和结果汇总；新改动以这个分支为准。 |
| `dev/eva-sim-integration` | 较早的 EVA-SIM 集成和 AgileX VR SFT 配置。适合参考数据集与训练配置，评测启动方式已经旧于 `dev/eva-sim`。 |
| `master-robolab-dev` | RoboLab 的 OpenPI 初始化、`deploy/serve.py` 和 OpenPI proxy。新 policy 想兼容 OpenPI 接口时，可以参考服务适配方式。 |
| `feature/genie_sim_sft` | 另一套仿真 benchmark 的环境解析、评测入口和 SFT 配置。适合参考“训练配置加独立 benchmark 入口”的组织方式，不复用它的任务协议。 |

查看参考分支时使用 `git show <分支>:<文件>` 或单独创建 worktree，避免切换当前正在评测的 checkout。

## 现在验证到了哪一步

EVA-VLA 的一键入口已实现。1～8 路资源配置、服务启动、端口和槽位绑定、故障退出与清理均已通过本地测试；OpenPI 和 XPolicyLab 两种协议的八路通信测试共完成 800 次推理。

版本与任务分片的 9 项检查已通过，脚本会在启动服务前拒绝同时选择两个版本。

原图透传也已验证：EVA-CLIENT 经过生产观察构造后，OpenPI 和 XPolicyLab socket 收到的三路图像尺寸、类型和像素均与输入一致；EVA-VLA 的真实 `WebsocketPolicyServer` 及 XPolicyLab 的真实 `PolicyServer` 都已完成往返验证。

这些检查还没有覆盖真实 checkpoint 与八路 SIM 一起运行，完整评测和长时间稳定性仍需实测。
