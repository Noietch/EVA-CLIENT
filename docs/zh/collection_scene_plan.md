[← 返回 README](../../README_zh.md) ｜ [English](../en/collection_scene_plan.md)

# 采集场景规划

场景规划是一份运行时输入，由 COLLECT、DASHBOARD、评测以及数据集审核服务共用。一份规划描述了任务、物体摆放、布局坐标，以及每个任务/场景组合所需的录制条数。控制台会把它展开为稳定的采集槽位；使用时无需任何迁移脚本，也不依赖历史任务计数快照。

## 目录与配置

默认数据根目录为 `datasets/data_collection`：

```text
<data_root>/
├── task_sets/
│   └── <version>/<task_set>/
│       ├── info.yaml
│       ├── layout.yaml
│       ├── scene.csv
│       ├── tasks.csv
│       └── objects.csv                 # or info.yaml.objects_file
├── datasets/<domain>/<robot>/<task_set>/
└── assets/
```

`collection.task_set_dir` 可以是单个叶子目录、一个父目录，或一组根目录列表。发现机制只检查直接子目录以及受支持的版本目录 `larybenchv2.v1.1` 和 `larybenchv2.v1.2`，不会递归搜索任意的历史目录树。在所有挂载的采集根目录中，数据集名称必须唯一。单个叶子目录默认以其目录名作为名称，可用 `task_set_name` 覆盖这一单任务集名称；父目录或列表形式的配置则按名称暴露每个发现到的叶子目录。

```python
collection = dict(
    task_set_dir="datasets/data_collection/task_sets",
    storage=dict(data_root="datasets/data_collection"),
)
```

[ARX X5 任务集采集](../../configs/02_collection/arx_x5_tasks_set.py)和[双 Piper 任务集采集](../../configs/02_collection/dual_agilex_piper_tasks_set.py)是持续维护的配置示例。[`examples/task_sets/dual_yam_pickplace_example`](../../examples/task_sets/dual_yam_pickplace_example) 是一份小而完整的规划（四个任务、两个场景、十个槽位）：其中三个任务带有 `reference_image` 照片，另一个没有。它既可以作为模板，也可以直接用作 `collection.task_set_dir`。对于不使用场景文件的工作流，仍可使用内联的 `collection.tasks`。如果单个任务集目录不存在，配置中编写的任务继续生效；而一旦挂载了目录，就必须包含有效的 `tasks.csv`。

`info.yaml` 可以定义：

```yaml
collection_dir: datasets/real_robot/dual_yam/example_plan
task_order: spatial
objects_file: objects.csv
```

数据集路径的 basename 必须与所选数据集名称一致。`collection_dir` 必须以 `datasets/` 开头，保持为相对路径，且不能包含 `..`。省略时，路径默认为 `datasets/real_robot/<robot_type>/<dataset>`。SIM 规划可以提供自己的 `datasets/eva_sim/...` 路径。运行时的游标与日志状态保存在 `collection.storage.log_dir`（或 `work_dir`）下，槽位文件按主机区分。

## 布局：`layout.yaml`

坐标始终来自布局文件，而不是根据 `P1`/`P2` 这类命名约定推断。位置 ID 可以表示网格、不规则台面，或货架上的位置。

```yaml
schema_version: 1
layout_id: example_layout_v1
coordinate_frame: table_top_left
unit: mm
orientation: x_right_y_down
calibration_status: unverified
bounds:
  width: 500
  height: 500
sampling_points:
  - position_id: P1
    x: 0
    y: 0
    calibration_status: unverified
  - position_id: P2
    x: 250
    y: 0
    calibration_status: unverified
```

近似坐标请保持为 `unverified`。只有当所有解析出的点都同时具备两个坐标且标记为 `verified` 时，运行时才会认为该布局已标定。点级别的 unit/frame/status 取值可以覆盖布局的默认值。注意，把布局标记为已标定，并不代表机器人或相机本身已经标定。控制台不会绘制布局或摆放；这些信息用于空间任务排序以及面向仿真器的规划。操作员实际看到的是任务的参考照片（见下文的 `reference_image`）。

## 场景：`scene.csv`

每个 `scene_id` 标识一种摆放配置。`placements` 是存放在 CSV 单元格中的 JSON 数组，请按常规 CSV 规则为该单元格加引号。

| 字段 | 含义 |
| --- | --- |
| `scene_id` | 稳定的场景引用 |
| `layout_id` | 所编写布局的引用 |
| `placements` | 物体及其占用的位置 ID |

一条摆放的结构如下：

```json
[
  {"object_id": "OBJ-RACK", "position_ids": ["P1", "P2"]},
  {"object_id": "OBJ-CUP", "position_ids": ["P3"], "fixed": false,
   "pose_xyzw": [0.2, 0.1, 0.0, 0.0, 0.0, 0.0, 1.0]}
]
```

跨越多个点的连续物体仍然只算一条摆放，只是带有多个 `position_ids`。对于面向仿真器的规划，如果编写了七个数值的 `pose_xyzw` 和 `fixed`，解析器也会保留它们。由任务引用场景；场景不会再重复维护自己的任务列表。

## 任务：`tasks.csv`

请严格使用以下列名拼写。当前 schema 的计数字段名中保留了 `epsiodes` 这一拼写。

| 字段 | 含义 |
| --- | --- |
| `task_id` | 稳定的任务标识，用于槽位和元数据 |
| `prompt_en` | 非空的运行时任务文本 |
| `prompt_zh` | 可选的翻译展示文本 |
| `total_epsiodes_count` | 整数总目标；加载采集任务时必填 |
| `scene_ids` | 以分号分隔的场景 ID，顺序与对应计数一致 |
| `scene_epsiodes_count` | 以分号分隔的计数，与 `scene_ids` 一一对应 |
| `action`、`category` | 可选的任务分类 |
| `operation_object` | 可选的物体标签，多个时用 `/` 分隔 |
| `operation_object_ids` | 可选的以分号分隔的物体 ID，用于空间排序 |
| `reference_image` | 可选的照片路径，相对于任务集目录（JPG/PNG/WebP） |

COLLECT、EVAL、DASHBOARD 和 RESULT 页会把任务的 `reference_image` 作为其场景参考展示。没有参考照片的任务只显示其 prompt；不存在位置网格视图。

例如，一个 `scene_ids=SC-001;SC-002`、`scene_epsiodes_count=3;2` 的任务会展开为五个槽位，此时应将 `total_epsiodes_count=5` 设为与之相符。固定槽位规划由各场景的计数决定；计数缺失或不为正数时不会生成槽位。旧的 `total_target` 和 `scene_targets` 字段不属于当前 schema。

加载配置时，相同的英文 prompt 会在录制器的 prompt 列表中合并，同时保留 task ID 到 prompt 的绑定关系。不同的 task ID 仍会产生不同的槽位。`collection.task_prompt_prefixes` 可以为每个数据集添加运行时前缀；任务编写的原始文本在场景界面中依然可用。

## 物体与照片

`info.yaml.objects_file` 指定物体目录文件，路径相对于规划根目录，默认为 `objects.csv`。共享的物体目录可以通过相对路径引用。解析器支持以下字段：

| 字段 | 含义 |
| --- | --- |
| `object_id` | 摆放引用；目录中也接受 `物品代码` |
| `object_name` | 英文/规范名称；也接受 `英文名` |
| `object_name_zh` | 中文展示名称；也接受 `名称` |
| `photo_dir` | 照片子目录；也可由 `照片目录` 或 `名称` 提供 |
| `color` | 可选的展示颜色 |

照片从被引用的物体目录文件旁边的 `object_photos/<photo_dir>/` 中读取，按排序后的第一张 JPG/JPEG/PNG/WebP 图片作为预览。展示名称优先使用中文名，其次是规范名称，最后是物体 ID。照片缺失时显示文字占位。运行时不会从旧版的 `photo_path` 列或假定的 `<object_id>.jpg` 文件名中查找照片。

## 槽位顺序、进度与补采

控制台和数据集工具共用 [`scene_plan.py`](../../src/core/utils/scene_plan.py) 中的同一个槽位规划器，排序为 场景 → 任务 → 轮次：

1. 场景按解析出的场景顺序排列。
2. 当 `task_order: spatial`（默认）时，任务先按首次提及的手（左、右、未知）排序，再按被操作物体在布局中的行优先位置排序。空间顺序由坐标决定；没有可用坐标的点排在有坐标的点之后，并保持编写顺序。
3. `task_order: csv` 在每个场景内保留 CSV 中的任务顺序。
4. 每个任务/场景组合按其目标数展开为从零开始编号的轮次。

稳定键为 `<task_id>:<scene_id>:<round_index>`。每个 episode 会保存这个 `slot_id`、任务/场景 ID 以及 `scene_round`。修改 ID 或计数会改变规划与已有录制之间的对应关系，因此在编辑进行中的规划时，请保持已被引用的标识不变。

槽位选择器可以按场景/任务筛选、选择已保存的某条录制进行回放/审核，或将某个槽位延后。游标和延后选择仅保存在本机。进度由已保存的记录和质检台账决定；TO COLLECT、UNREVIEWED、PASSED 和 FAILED 四种状态彼此独立。补采会替换该槽位已有的 episode 索引。质检与保存语义详见[录制](./recording.md)。

当前运行时直接根据各目录文件生成这份固定规划，不会生成带随机种子的 `run_plan.csv`，也不会随机分配位置。调整覆盖范围时，请显式编写所需的场景 ID 和计数。

## 评测场景标识

`eval_cfg.task_set_dir` 显式指定评测根目录；未设置时不会加载任何评测任务集。同样，采集场景规划也需要 `collection.task_set_dir`；运行时不会搜索 `work_dirs`，也不会使用 `EVA_SCENE_PLAN_DIR`。必要时可用带版本号的名称区分不同的评测叶子目录。

在 EVAL 页中选择一个任务及其引用的某个场景。启动请求会同时提供 `task_set`、`task_id` 和 `scene_id`；服务端会校验 prompt、场景归属关系，以及请求中附带的仿真器任务/场景元数据。保存的尝试记录包含所选场景映射的快照，因此 RESULT 页可以重建该次尝试对应的布局。在采集配置中，名称包含 `eval` 的任务集决定了要采集哪些布局；它本身并不会启动策略评测。
