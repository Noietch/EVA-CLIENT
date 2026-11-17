[← 返回 README](../README_zh.md) ｜ [English](data_collection_sop.md)

# 数据采集 SOP

本文是在 EVA-Client 中开展一场整场采集的端到端操作流程：规划任务、启动采集站、采集、监控、审核、修复、交付。每一步都可以在浏览器控制台或独立的数据集浏览器中完成，并且始终作用于同一个原生 LeRobot v2.1 数据集。

```text
 Plan ──► Bring up ──► Collect ──► Monitor ──► QC ──► Repair ──► Deliver
 task set   DEVICE      COLLECT    DASHBOARD   auto +   retake /   HF / SFTP /
 (photo or  cameras,    slots,     live flags  human    trim /     S3, format
  no photo) preflight   MOTION/REC             review   edit       conversion
                ▲                                          │
                └──────────── failed slots return ─────────┘
```

| 阶段 | 位置 | 产出 |
| --- | --- | --- |
| 1. 规划 | 任务集文件、数据集浏览器 | 一份带场景计数且通过校验的任务集 |
| 2. 启动 | DEVICE 页 | 机器人、遥操作和相机就绪 |
| 3. 采集 | COLLECT 页 | 绑定到槽位的已保存录制 |
| 4. 监控 | COLLECT 页头部、DASHBOARD | 实时采集健康状况与整场采集进度 |
| 5. 质检 | COLLECT、数据集浏览器 | 写入 `meta/qc.jsonl` 的 PASS / FAIL 结论 |
| 6. 修复 | COLLECT、数据集浏览器 | 补采或裁剪后的 episode |
| 7. 交付 | DASHBOARD 数据集管理器、COLLECT | 已上传的数据与质检结果、格式转换产物 |

## 1. 规划任务集

任务集定义了要采集什么：有哪些任务、每个任务在哪些场景中录制，以及每个任务/场景组合需要录制多少条。控制台会将其展开为固定的采集槽位（`<task_id>:<scene_id>:<round>`），因此进度始终以规划为基准来衡量。完整 schema 见[采集场景规划](./collection_scene_plan.zh.md)。

每个任务会以以下两种方式之一展示给操作员：

- **带照片。** 在 `tasks.csv` 中将 `reference_image` 设为预期布置的照片。操作员在录制前按照片摆好桌面。
- **不带照片。** 将 `reference_image` 留空。操作员只能看到任务 prompt。

可以从 [`examples/task_sets/dual_yam_pickplace_example`](../examples/task_sets/dual_yam_pickplace_example) 开始：四个任务、两个场景、十个槽位，其中三个任务带照片，一个不带。

整场采集开始前：

1. 填写 `info.yaml`（`dataset_name`、`robot_type`、`collection_dir`）、`tasks.csv` 和 `scene.csv`。
2. 在[数据集浏览器](./console.zh.md#dataset-operations)中打开该任务集（`bash tools/datasets/start.sh --config <file.py>`）并运行校验。采集前修复所有报告的错误。
3. 一旦开始采集，就冻结 `task_id`、`scene_id` 和各项计数。修改它们会改变已保存录制与槽位之间的归属关系。
4. 发布任务集（Hugging Face 任务集同步），确保每个采集站使用同一版本。

## 2. 启动采集站

1. 启动一个挂载了该任务集的采集配置，例如：

   ```python
   _base_ = ["dual_yam.py"]
   collection = dict(
       task_set_dir="examples/task_sets/dual_yam_pickplace_example",
       storage=dict(data_root="datasets/data_collection"),
   )
   ```

2. 在 DEVICE 页中启动机器人、遥操作源和相机。等待每个相机都显示实时预览；EVA 会在使用机器人之前检查相机是否就绪。
3. 运行 preflight 和 HOME。机械臂出现任何异常行为时，请使用安全停止。
4. 多个采集站可以写入同一个数据根目录；数据集写入按数据集串行执行。

## 3. 采集

1. 在 COLLECT 页中选择数据集。槽位网格会显示每个场景/任务/轮次。
2. 选择下一个 TO COLLECT 槽位，或让控制台自动前进到该槽位。可以使用场景和任务筛选器，一次只处理一个场景。
3. 按照任务照片（或 prompt）布置桌面。
4. 开启 MOTION，然后点击 START RECORD，执行任务。
5. 点击 STOP 保存本条录制，或点击 CANCEL 丢弃。STOP 之后机器人保持当前姿态，遥操作仍处于激活状态。
6. 执行 HOME 之前先关闭 MOTION。
7. 确认本条录制已显示为已保存后再继续。保存队列已满时，新的录制会被阻塞，直到保存跟上。

默认操作员控制：

| 操作 | 键盘 | VR 手柄 |
| --- | --- | --- |
| 切换 MOTION | `M` | `B` |
| 开始 / 停止录制 | `R` | `A` |
| 取消当前录制 | 按住 `Esc` 1 秒 | 按住 `A` 1 秒 |
| HOME | `H` | `X` |
| 切换左 / 右臂 | | 按住 `L GRIP` / `R GRIP` 1 秒 |

当某个槽位暂时无法录制（缺少物体、场景不对）时，使用 **defer** 跳过它；该槽位仍保留在规划中，之后可以再采集。

## 4. 监控

**录制过程中。** 控制台头部显示推理/步进/图像速率，相机预览实时更新。缺失、停滞或冻结的相机流会在已保存的录制上被标记（`missing_camera_stream`、`frozen_camera`），该录制会被标红，而不是悄无声息地混入数据集。

**整场采集层面。** DASHBOARD 读取配置中的 `dashboard.raw_roots`：

| 面板 | 关注点 |
| --- | --- |
| Dataset overview | episode 数、帧数、时长、质检通过率、已上传 / 未上传 |
| Collection by day | 每日产出；出现空档说明有采集站闲置 |
| Usable output | 通过质检的录制占比 |
| Upload sync | 本地数据尚未上传的数据集 |

可按日期范围和场景筛选，以对比不同采集站、操作员或布置。在无界面模式下，状态切换和周期性的 `[HB]` 日志行会提供同样的进度信息。

## 5. 质检

质检分为两层，两者都写入 `meta/qc.jsonl`，不会改动录制本身。

**自动质检。** 保存时，采集检查会把以下情况的录制标红：录制过短、相机流缺失、图像重复、向量无效，以及时序/对齐问题。保存后，后台质检还会检查时长异常值、静止或停滞的相机，以及超出机器人 URDF 限位的关节。

**人工审核。** 在 COLLECT 页或数据集浏览器中选中一条已保存的录制，观看同步回放，并标记为 PASS、FAIL 或 UNREVIEWED，可附带备注。

| 槽位状态 | 含义 |
| --- | --- |
| TO COLLECT | 尚无已保存的录制 |
| UNREVIEWED | 已保存，等待人工结论 |
| PASSED | 人工 PASS，接受该录制 |
| FAILED | 人工 FAIL、采集标红或自动拒绝 |

人工 PASS 会覆盖自动判定的失败；人工 FAIL 则一律拒绝。仅有绿色的采集标记并不代表该录制已通过质检。交付前请审核每一条录制。

## 6. 修复

- **补采。** 选择一个 FAILED、已延后或已完成的槽位重新录制。新录制会在同一 episode 索引处替换旧录制，清除过时的审核结论，并回到 UNREVIEWED 状态。槽位数量不会增加。
- **裁剪。** 在数据集浏览器中裁掉一条合格录制首尾的空闲片段。裁剪会原地重写该 episode，并保持状态、动作和所有相机流对齐。
- **修正元数据。** 在数据集浏览器中编辑任务描述和物体元数据，而不是手动修改文件。

修复之后，请重新审核该录制。如此循环，直到待交付场景的槽位网格中不再有 FAILED 或 TO COLLECT 槽位。

## 7. 交付

1. 在 DASHBOARD 数据集管理器中选中数据集，运行 **Refresh cloud and verify**，对比本地与远端的数据、任务和质检结果。
2. 分别上传数据和上传质检结果。两者是独立操作：上传数据时永远不会附带质检结果，下载数据时会将远端质检结果与本地质检结果合并，而不是覆盖。
3. 也可以通过 SFTP 或 S3（`collection.storage`）交付，或在 COLLECT 页中转换为 LeRobot v3、HDF5 或 MCAP。转换会包含所有 episode 及其质检元数据，由下游使用方按结论筛选。
4. 在报告交付完成之前，检查任务结果和上传回执。正在录制或保存中的数据集会拒绝传输任务。

## 每日检查清单

| 时机 | 检查项 |
| --- | --- |
| 上班开始 | 已同步最新任务集；相机实时在线；preflight 和 HOME 通过 |
| 每条录制 | 确认已保存；无红色标记；桌面已按照片复位 |
| 每小时 | DASHBOARD 上的产出量和可用产出率；无停滞的采集站 |
| 下班结束 | 所有录制已审核；FAILED 槽位已补采或延后；数据与质检结果已上传；Upload sync 无待上传项 |
