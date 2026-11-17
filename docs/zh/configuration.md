[← 返回 README](../../README_zh.md) ｜ [English](../en/configuration.md)

# 配置

## 预设与继承

业务预设是 Python 文件，采用 mmengine 风格的 `_base_` 继承。`_base_` 中的路径相对于所在文件解析。字典会递归合并；标量值和整个列表会直接替换父级的值。`_delete_=True` 会替换继承来的整个字典子树。

[`configs/00_base/defaults.py`](../../configs/00_base/defaults.py) 提供共享的模型与工作流默认值。机器人 SDK 设置、相机选择、端点和硬件默认值位于 `examples/hardware/*/config.yaml` 和 `examples/input_sources/*/config.yaml`。加载器会把所选硬件的默认值放在 Python 预设之下，预设中显式指定的字段优先生效。通过 DEVICE 工作区加载时，会应用该工作区中的选择和设置。

| 目录 | 用途 |
| --- | --- |
| `configs/00_openloop/` | 离线数据集评测，可选择同时启动策略服务 |
| `configs/01_deploy/<robot>/` | 机器人与模型部署；同级的 `_base.py` 存放共享设置 |
| `configs/02_collection/` | 遥操作、任务集、录制与上传设置 |
| `configs/03_evaluation/` | checkpoint / 任务评测以及仿真器评测（EVA-SIM，即将发布） |
| `configs/04_rl/` | RL 策略 / critic 选择与 rollout 设置 |

例如，放在 `configs/01_deploy/dual_agilex_piper/_base.py` 旁边的预设可以覆盖其中的策略，同时保留硬件和录制的默认值：

```python
_base_ = ["_base.py"]

policy = dict(
    type="openpi_rtc",
    host="127.0.0.1",
    port=9000,
    backend_options=dict(latency_k=4),
)

inference_cfg = dict(debug_tasks=["pick up the yellow spoon"])
inference_strategies = {
    "async": dict(args=dict(latency_k=4)),
    "rtc": dict(args=dict(latency_k=4)),
}
```

如果要从一个选项完全不同的后端切换过来，请在 `policy` 中使用 `_delete_=True`。`debug_tasks`、`checkpoints`、`robot.eef_links` 等列表在替换时必须完整写出。

## 策略与图像输入

`policy.type` 可选 `openpi`、`openpi_rtc`、`starvla`、`xpolicylab`、`gr00t`、`mock` 或 `replay`。`host` 和 `port` 指定服务地址；`backend_options` 存放与协议相关的字段。详见[核心概念](concepts.md#policy-backends)。

部署预设中的 `policy` 也可以写成非空的 `list[dict]`。每一项都需要唯一的 `name`，启动时默认选中第一项。DEBUG 页的输入工具栏会列出这些名称，空闲时可以在其中切换模型：

```python
policy = [
    dict(name="Model A", type="openpi", host="127.0.0.1", port=5555),
    dict(name="Model B", type="openpi", host="127.0.0.1", port=5556,
         backend_options=dict()),
]
```

每一项都可以单独覆盖后端类型、端点和 `backend_options`。机器人、图像预处理和推理策略设置由列表中的所有模型共享。切换模型时会清除旧的策略连接、预览 chunk 和 setup 状态，下一次发送时再准备所选模型。运行或准备期间禁止切换。原有的单个 `policy = dict(...)` 写法仍然可用。修改配置中的模型列表后需要重启。

| 预设 | 模型约定 |
| --- | --- |
| [`dual_agilex_piper/xpolicylab.py`](../../configs/01_deploy/dual_agilex_piper/xpolicylab.py) | XPolicyLab，端口 19000，两条 6D 关节臂和 1D 夹爪，ZMQ 按需取图 |
| [`dual_yam_eva_sim_eval.py`](../../configs/03_evaluation/dual_yam_eva_sim_eval.py) | XPolicyLab 末端 / TCP 评测，两条 7D 位姿臂和 1D 夹爪，ZMQ 按需取图，启用控制通道 |
| [`dual_agilex_piper_xpolicylab.py`](../../configs/00_openloop/dual_agilex_piper_xpolicylab.py) | 离线 XPolicyLab 示例，包含数据集列映射和官方 [XPolicyLab](https://github.com/XPolicyLab/XPolicyLab) 服务启动命令；请按本地路径自行修改 |

对于 XPolicyLab，需要设置 `action_type="joint"` 或 `"ee"`，并为单臂或双臂设置 `arm_dims` 和 `gripper_dims`。末端控制的机械臂要求 `arm_dims` 的每一项为 7（`xyz + wxyz`）。观测空间和动作空间必须与这些维度一致。可选的 `evaluation_id` 用于设置协议中的评测标识，不设置则自动生成。超时单位均为秒：`request_timeout_s` 默认 120，`connect_timeout_s` 默认 30，`handshake_timeout_s` 默认 60。预设可以覆盖这些值。

图像预处理在 `transport` 下配置：

| 字段 | 共享默认值 | 含义 |
| --- | --- | --- |
| `convert_bgr_to_rgb` | `False` | 默认保持通道顺序不变；`True` 时交换红蓝通道 |
| `image_height`, `image_width` | `224`, `224` | `resize=True` 时的目标尺寸，否则忽略 |
| `resize` | `False` | 默认保持源分辨率；`True` 时启用缩放 |
| `resize_pad` | `True` | 启用缩放时，通过补边保持宽高比；`False` 时直接缩放 |
| `image_layout` | `"hwc"` | 默认保持通道在后的布局；`"chw"` 时显式转置为通道在前 |
| `image_mode` | `"stream"` | ZMQ 策略帧使用流式图像，或使用 `"on_demand"` 按需请求图像 |
| `image_request_timeout_s` | `2.0` | 按需模式下等待新图像版本的最长时间 |

请检查完整的预设继承链，因为机器人或工作流预设可能会覆盖这些默认值。通道转换应与通信层解码后的图像通道顺序保持一致；仿真器的 XPolicyLab 评测预设会保留输入图像的通道顺序和分辨率。

数据集通信层还需要 `dataset_dir`、`episode_id` 和 `dataset_keys`。请把 `state_key`、`action_key` 以及（对于末端观测）`eef_key` 设置为数据集中实际的列名。`video_keys` 把 EVA 相机名映射到数据集的视频列。映射为空时使用 `observation.images.<camera>`；映射非空时只会选择其中映射的相机。

## 运行时与输出

`inference_cfg` 用于选择观测 / 动作空间、推理频率与发布频率、预热 chunk 数、DEBUG 提示词以及 DEBUG 的最长运行时间。`inference_strategies` 把控制台中的策略键映射到已注册的类及其 `args`；只覆盖某个策略的 args 时，会保留其继承来的类和其他设置。

`work_dir` 默认为 `work_dirs`。当配置了采集 schema 列而 `collection.storage.log_dir` 为空时，加载器会将其推导为 `<work_dir>/<preset_stem>`。部署预设可以显式设置 rollout 路径。`collection` 控制示教数据与任务集；`rollout` 控制策略 episode 的录制；`eval_cfg.storage` 控制评测录制。数据集目录结构见[录制](recording.md)。

`console.initial_tab` 默认为 `"auto"`（评测配置下打开 EVAL 页，否则打开 DEBUG 页）；预设也可以显式指定工作区。`control_channel.enabled` 默认为 `False`，本地地址为 `127.0.0.1`，端口 5757。自动化用法见[控制通道](control-channel.md)。

<a id="evaluation-and-scene-plans"></a>
## 评测与场景规划

继承一个部署预设和 [`configs/00_base/evaluation.py`](../../configs/00_base/evaluation.py)，然后填写 `eval_cfg`。每个 checkpoint 包含名称、相对于评测预设的部署 `config` 路径、`port`，以及可选的 `host`（默认 `127.0.0.1`）。加载器会独立解析每个部署配置，并用该 checkpoint 的 host 和 port 覆盖其策略端点。

```python
_base_ = ["../01_deploy/dual_yam/openpi_qpos.py", "../00_base/evaluation.py"]

eval_cfg = dict(
    task_set_dir="datasets/data_collection/task_sets/v1.2",
    trials_per_prompt=5,
    inference_strategy="async",
    checkpoints=[dict(
        name="openpi_yam",
        config="../01_deploy/dual_yam/openpi_qpos.py",
        host="127.0.0.1",
        port=9000,
    )],
)
```

`task_set_dir` 可以是一个目录，也可以是目录列表。相对路径从仓库根目录解析。目录中可以直接包含 `tasks.csv`，也可以包含若干任务集子目录或已知的版本目录。EVAL 页只会加载在 `eval_cfg.task_set_dir` 中显式配置的路径。未设置时不会加载任何评测任务集，也不会回退到数据根目录或采集任务集。带版本的规划会在显示名称中保留版本号，例如 `v1.2/<task_set>`。

控制台会把规划中的任务与 `eval_cfg.tasks` 中编写的匹配元数据（包括里程碑）合并。基于场景的尝试要求同时提供 `task_set`、`task_id` 和 `scene_id`；所选任务必须属于该场景，且提示词必须一致。每次尝试都会保存场景映射快照，供 RESULT 页复盘。不带场景标识的纯提示词任务仍然受支持。CSV / YAML 文件格式见[场景规划](collection_scene_plan.md)，操作流程见[控制台](console.md)。

启动时会合并预设继承链和硬件默认值、加载任务集、构建控制空间、推导路径、校验设置，并解析评测 / RL 的策略配置。`eval_cfg` 或 `rl_cfg` 为空时，对应的工作流会被禁用。


## 数据集浏览来源

```python
dashboard = dict(raw_roots=["work_dirs/my_eval", "datasets/my_collection"])

# 在 RESULT 页中显示的、位于评测输出目录下的历史模型目录。
eval_cfg = dict(result_models=["previous_model"])
```

当 `dashboard.raw_roots=[]` 时，Dashboard 保持为空，不会自动包含 `work_dir` 或当前运行目录。RESULT 页只读取已配置的 checkpoint 和显式列出的 `result_models`，其他无关的模型目录会被忽略。如需使用独立的数据集浏览器，请在一个 Python 文件中写入 `dataset_browser = dict(plans_root=..., assets_root=..., collection_root=...)`，然后运行 `bash tools/datasets/start.sh --config <that_file.py>` 启动。
