# 场景化采集方案（设计稿）

这份文档定义“任务、场景、位置、物体、采集记录”之间的关系，并记录 COLLECT 页二维场景地图的第一版交互。当前布局按约 50×50 cm 的 3×3 参考点预览，但仍明确标记为未校准。

## 1. 当前任务表审计

迁移前的 `work_dirs/desktop_remaining_tasks.csv` 共包含：

| 指标 | 数量 | 说明 |
| --- | ---: | --- |
| 任务行 | 68 | 相同物体的不同动作方向/双臂分工仍是不同任务 |
| 目标采集条数 | 950 | `数量` 列总和 |
| 场景引用次数 | 190 | 每个任务按场景顺序展开后的引用次数 |
| 独立场景布局 | 131 | 按物体和 P 位组合去重；目标数不参与场景 ID |
| 独立物体/器具标签 | 47 | 包含盘子架、盘子、装有黄豆的塑料碗等状态化物体 |
| 每场景目标数 | 5 | 当前 190 个场景片段均为 5 条 |

迁移前的 `场景摆放` 列适合人读，但不适合长期维护和程序展示：它同时混合了场景顺序、位置、物体、目标数和中文标点。迁移脚本已经为每一行补上 `task_id`、`scene_ids` 和 `scene_targets`，并生成 `layout.yaml`、`scene.csv`、`tasks.csv`、`objects.csv`；原始 CSV 已移到 `work_dirs/trash/` 保存。

当前 `work_dirs/tasks_set` 已按现场物体清单做过一次裁剪：删除颜料/纸张任务及其 3 个场景，将剥皮香蕉替换为芹菜。因此当前目录包含 67 条任务、128 个场景和 46 个对象；上表保留的是迁移时的历史基线。

## 2. 当前使用近似 3×3，但不绑定布局形状

P1-P9 本身只能被视为“位置引用 ID”，不能从编号推断空间关系。当前根据 50×50 cm 的需求，在 `layout.yaml` 中给出近似的 3×3 参考坐标；这不是 P 编号的永久规则。未来仍然可以替换为不规则布局，因此位置 ID 和几何布局必须拆开：

- `position_id`：稳定的业务引用，例如 `P4`。
- `layout_id`：一次实际布置/标定的版本，例如 `desktop_layout_v1`。
- `geometry`：二维点 `(x, y)` 为主，也可以扩展为层位或多边形；没有测量前留空。
- `orientation`：相机/操作者视角下的前后左右定义；没有确认前标记 `unverified`。
- `physical_reference`：现实桌面上的胶带、孔位、货架层或其他可复核标记。

推荐的登记方式是先画一张俯视图，再决定布局类型：

```text
                操作者/相机方向（待确认）
       P?             P?
             P?
   P?                 P?
```

上图表达的是布局关系。当前参考布局使用 `(0, 0)` 到 `(500, 500)` mm 的 3×3 点阵；现场测量后可以调整为任意形状，例如：

- 不规则桌面上的离散点：P1、P2、P3……；
- 货架：`R1-L1`、`R1-L2` 这类层位；
- 门体/开关面板：一个物理物体对应多个操作区域；
- 多桌面或多相机：通过不同 `layout_id` 分开管理。

当前坐标是采集前的近似值，现场完成测量后再修正 `layout.yaml` 的几何和方位字段，并把 `calibration_status` 从 `unverified` 改为 `verified`。

### 随机位置采集

如果一个任务要在九个候选位置上各采集 5 条，推荐先生成批次计划，再开始录制：

```text
task_id + layout_id + seed
        ↓
候选 position_id 集合（例如 P1...P9）
        ↓
固定 seed 的随机排列
        ↓
run_plan.csv：position_id / scene_id / target_episodes=5 / order
```

随机规则需要满足：

- `seed`、候选位置集合和算法版本写入批次计划，失败重采时可以完全复现；
- 每个位置的目标数独立记录为 5，不用从任务总数反推；
- 有固定物体（架子、黑板等）时，只随机允许移动的物体，避免生成物理上不可能的摆放；
- 位置有碰撞、可达性或安全约束时，先过滤候选集合，再做随机排列；
- 迁移前任务表中很多任务只覆盖 P4-P6，不能直接把它们变成 P1-P9 九个位置，否则任务总目标会从 15 变成 45，必须显式修改 `scene_targets`。

建议由系统生成批次展开表 `run_plan.csv`：

| 字段 | 含义 |
| --- | --- |
| `batch_id` | 一次实际采集批次 |
| `task_id` | 任务引用 |
| `scene_id` | 该次生成的具体摆放 |
| `layout_id` | 坐标/标定版本 |
| `position_id` | 本条样本使用的候选位置 |
| `target_episodes` | 该位置需要的有效条数，通常为 5 |
| `seed` | 随机种子 |
| `order` | 操作员执行顺序 |

这样“随机”只决定计划顺序和允许的位置分配，场景照片、物体照片和坐标仍然是可追溯的固定数据。

## 3. 文件职责

正式维护的最小集合是三个必填文件和一个可选文件。采集批次的随机顺序另由系统生成，不手工改写基础定义。

### 3.1 `layout.yaml`：布局和采样点

一个文件描述一个当前使用的布局版本，以及这个布局中有多少个采样点。它是二维坐标的唯一来源，不从 `P1`、`P2` 的编号猜测网格关系。

```yaml
schema_version: 1
layout_id: desktop_layout_v1
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
    physical_reference: tape mark A
    calibration_status: unverified
  - position_id: P2
    x: 250
    y: 0
    physical_reference: tape mark B
    calibration_status: unverified
```

`sampling_points` 的条目数就是当前布局的采样点数，可以是 3、9、12 或任意其他数量。当前 9 个点先写入近似坐标；没有完成测量时，`calibration_status` 保持 `unverified`，页面会显示近似坐标和校准提示。

### 3.2 `scene.csv`：场景定义

一行表示一个完整的 scene，不反向记录任务。`placements` 使用 JSON 数组保存这个场景中的全部摆放关系；同一个连续物体占据多个采样点时，只保留一个元素，用 `position_ids` 表示它占据的点：

| 字段 | 含义 |
| --- | --- |
| `scene_id` | 独立场景 ID，例如 `SC-001` |
| `layout_id` | 引用的布局版本 |
| `placements` | `[ {"position_ids":["P1","P2"], "object_id":"OBJ-001"}, ... ]` |
| `scene_photo` | 可选的场景整体照片路径 |
| `notes` | 遮挡、朝向、复位等场景备注 |

例如盘子架占 P1-P3、盘子占 P4，就在同一行的 `placements` 中写两个对象，其中盘子架的 `position_ids` 是 `["P1","P2","P3"]`。前端解析时再把这些点展开到地图节点，但语义上仍然是一个连续物体。

### 3.3 `tasks.csv`：任务定义和场景目标

任务文件描述“要做什么”和“每个场景采多少条”：

| 字段 | 含义 |
| --- | --- |
| `task_id` | 稳定任务 ID，例如 `TASK-001` |
| `action` | 动作类别 |
| `category` | 任务类别 |
| `operation_object` | 操作对象组合 |
| `prompt_en` | 发送给采集/策略的任务文本 |
| `total_target` | 任务总目标数 |
| `scene_ids` | 按执行顺序引用的场景 ID，使用 `;` 分隔 |
| `scene_targets` | 与 `scene_ids` 一一对应的有效条数，使用 `;` 分隔 |

一个场景可以被多个任务复用；任务的动作、prompt 和每场景目标独立维护。旧的 `desktop_remaining_tasks.csv` 只作为迁移输入，不再作为程序唯一数据源。

### 3.4 `objects.csv`：物体目录（推荐）

虽然物体照片可能没有，但建议保留物体目录，让名称、状态和照片路径有固定位置：

| 字段 | 含义 |
| --- | --- |
| `object_id` | 稳定物体 ID，例如 `OBJ-002` |
| `object_name` | 规范名称，通常填英文或跨语言短名 |
| `object_name_zh` | 中文显示名称 |
| `photo_path` | 可选的物体照片路径 |
| `appearance_notes` | 颜色、尺寸、朝向、易混淆点 |

`object_name` 和 `object_name_zh` 可以只先填中文字段；页面会依次回退到中文名、规范名和 `object_id`。照片按 `object_id` 命名时可以自动发现，例如 `OBJ-002.jpg`，没有照片则显示名称占位。

### 3.5 `run_plan.csv`：批次计划（系统生成）

`run_plan.csv` 不属于基础定义，由系统依据 `task_id + layout_id + seed` 生成。它记录随机后的执行顺序、每个位置的目标 5 条、批次 ID 和实际进度，失败重采时可以用同一个 seed 复现。

登记文件由一次性迁移脚本生成，脚本已归档到 `work_dirs/trash/collection_scene_plan.py`，不属于运行时依赖。脚本按首次出现顺序分配 ID，重复运行不会改变当前数据的编号；修改任务顺序或删除历史行前，应先确认下游采集记录是否已经引用这些 ID。

迁移前的 `desktop_remaining_tasks.csv`、`scene_registry.csv`、`layout_positions.csv`、`object_catalog.csv` 已移到 `work_dirs/trash/`，只作回滚参考，不再作为正式输入。

### 3.6 采集配置接入

采集配置可以直接引用这四个文件组成的目录，不需要把任务逐条复制到 Python：

任务集版本使用 `configs/02_collection/arx_x5_vr_tasks_set.py`；原有的
`configs/02_collection/arx_x5_vr.py` 保持不变，继续提供原来的固定任务配置。

```python
collection = dict(
    task_set_dir="work_dirs/tasks_set",
    task_set_name="ArxKine_PnP_DivObj_Norm_Sngl_Base_v1_scene_1_20260828",
)
```

加载配置时，`tasks.csv` 的 `prompt_en` 和 `total_target` 会转换成采集器现有的 `(prompt, target)` 任务列表；`scene.csv`、`layout.yaml` 和 `objects.csv` 则由 COLLECT 页从同一个目录读取。`task_set_name` 决定录制数据集目录名，省略时默认使用任务集文件夹名。目录是机器本地资源时可以暂时不存在，此时配置内手写的 `collection.tasks` 作为备用；目录存在后以 `tasks.csv` 为准。

## 4. 采集文档编写方案

每个采集批次包含三层文档，不把所有内容堆进任务 CSV。

### A. 场景基线文档（一次/布局版本）

文件名：`scene_layout_<layout_id>.md`。

固定章节：

1. **版本与适用范围**：布局 ID、桌面/机器人/相机、负责人、标定日期。
2. **坐标与视角**：俯视图、操作者方向、相机方向、单位、原点；没有确认的字段写 `TBD`。
3. **位置表**：每个 `position_id` 的物理参照、尺寸、允许误差和照片。
4. **安全边界**：机械臂禁入区、易碎/尖锐物体、清洁要求。
5. **复位规则**：一条 episode 结束后如何恢复，谁负责确认，异常如何处理。
6. **变更记录**：任何位置移动或物体替换都生成新的 `layout_id`，不能覆盖旧照片。

### B. 场景卡片（每个 `scene_id`）

固定章节：

1. 场景 ID、整体照片和引用它的任务；
2. 俯视图中标出每个位置的物体，物体卡片链接到 `objects.csv`；
3. 本场景目标条数、已采集有效条数、剩余条数；
4. 初始状态、动作前提、动作结束状态；
5. 成功判定、失败判定和允许重试条件；
6. 采集完成后的复位检查清单。

### C. 任务采集单（每个 `task_id`）

固定章节：

1. 中英文任务文本和双臂分工；
2. `scene_ids` 的执行顺序；
3. 每个场景目标 5 条（或登记表中的目标值）；
4. 每条 episode 的开始/结束条件和 QC 规则；
5. 异常记录：场景 ID、episode ID、问题、是否重采；
6. 批次总结：有效、拒绝、待复核、缺失照片/标定项。

## 5. 采集页的二维地图交互（第一版已接入）

当前 COLLECT 页按以下顺序工作：

1. 选择任务，读取它的 `scene_ids`；
2. 显示当前场景照片和位置布局，格子/节点里显示对应物体照片；规则网格使用较大的方形单元格，非规则布局按 `(x, y)` 比例定位；
3. 使用任务行的 `scene_targets` 显示 `当前场景 2/3 · 有效 3/5`，而不是只显示任务总数；
4. 当前场景达到目标后，操作员点击“下一场景”，或由系统在 QC 通过后提示切换；
5. 点击预览地图中的任意位置会打开大图弹窗，显示坐标、物体和可选照片；
6. 每条录制元数据写入 `task_id`、`scene_id`、`layout_id` 的接入点已预留，这样统计能按任务、场景、位置回溯；
7. 如果照片、位置校准或复位检查未完成，页面显示提示而不是静默掩盖缺口。

这里的“格子”只是 UI 呈现组件：当 `layout_id` 不是规则网格时，前端应使用登记的二维坐标或节点连线；当前近似坐标会按比例预览，但页面显示 `UNVERIFIED`，不能把它当成现场精确位置。

建议的采集卡片结构如下，实际布局由位置坐标决定：

```text
SCENE 02 / 09                         VALID 3 / 5
layout: desktop_layout_v1 · seed: 48192 · READY

        +----------------+  +----------------+  +----------------+
        | P4             |  | P5             |  | P6             |
        | [object photo] |  | [object photo] |  | [object photo] |
        | 盘子           |  | 盘子架         |  | empty          |
        +----------------+  +----------------+  +----------------+

 [PREVIOUS SCENE]       [NEXT SCENE]       TARGET 5 / POSITION
```

规则网格时单元格建议至少 150×150 px（窄屏按可用宽度缩放但保持方形）；非规则布局用坐标比例布置卡片，避免为了“看起来像网格”而改变真实位置。照片加载失败显示物体名称和 `PHOTO MISSING`，不显示一张容易误认的假图。

## 6. 目前还需要现场补齐的内容

1. P1-P9 的真实物理对应关系、操作者/相机方向、尺寸和允许误差；
2. 是否确实只有九个位置，以及未来是否需要 `P10`、层位或多桌面 ID；
3. 46 个物体/状态标签的实物照片和易混淆说明；
4. 每个场景的复位动作、成功/失败判定和安全注意事项；
5. 运行批次 ID 和采集人员字段，以便同一场景跨日期追踪。

## 7. 验收标准

在正式开始现场采集前，应满足：

- 所有任务行都有有效 `task_id` 和至少一个 `scene_id`；
- `scene_targets` 的目标数之和等于任务的 `数量`，且与 `scene_ids` 数量相同；
- 所有 `scene_id` 都能在 `scene.csv` 中找到，且所有 `position_id` 都能在 `layout.yaml` 中找到；
- 位置表已完成现场标定，或者页面明确阻止“已校准采集”；
- 场景/物体照片缺失会在批次检查中显式列出；有照片时用于辅助摆放确认，没有照片时不阻断采集；
- 采集结果可按 `task_id → scene_id → episode_id` 查询，不依赖解析中文长句。
