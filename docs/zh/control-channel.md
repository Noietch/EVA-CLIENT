[← 返回 README](../../README_zh.md) ｜ [English](../en/control-channel.md)

# 控制通道（ZMQ）

ZMQ 控制通道允许脚本、仿真器和快捷键客户端向 EVA 的状态机发送操作员命令。它与浏览器共享同一个命令队列和实时会话。控制通道开放了白名单内的命令目录和三个只读查询；数据集传输、质检（QC）写入、打分、设备配置以及其他仅限 HTTP 的 API 属于独立的接口。

## 启用控制通道

```python
control_channel = dict(
    enabled=True,
    host="127.0.0.1",
    port=5757,
)
```

控制通道默认关闭，使用 `eva --headless` 时会自动启用。`127.0.0.1` 仅绑定本机；如需从其他主机访问，请使用 SSH 隧道，或在充分考虑后再修改绑定地址。注意，控制通道可以下发机器人运动命令。

控制 socket 独立于机器人通信层（`zmq`、`ros1`、`ros2` 或 `dataset`）。观测与动作仍然通过该通信层传输；稀疏的操作员命令不会给每个推理步骤额外增加一次请求 / 应答往返。

## 传输协议与完成语义

通过 ZMQ REQ/REP 发送 JSON 对象。每个应答都包含 `ok` 字段。普通命令的应答只表示命令已入队，并不代表命令已完成或运动已成功。状态检查由主循环稍后执行；请轮询状态并查看 `last_error`。

| 请求 | 应答格式 |
| --- | --- |
| `{"cmd":"web:run"}` | `{"ok":true,"cmd":"web:run"}` |
| `{"cmd":"web:tab_switch:collect"}` | `{"ok":true,"tab":"collect"}` |
| `{"query":"status"}` | `{"ok":true,"data":{...}}` |
| 无效请求或不允许的命令 | `{"ok":false,"error":"..."}` |

只允许 [`command_catalog.py`](../../src/core/app/command_catalog.py) 中的 `web:*` 命令。`tab_switch` 会更新当前控制台页面并将对应的切换动作入队。触觉反馈会直接发送给 VR 客户端。带 `clip_id` 的 `web:start` 会先校验并绑定一次评测尝试，再将其入队。其他受支持的命令进入共享队列。

## 命令

`<...>` 表示需要替换的值。模式、策略、任务和推理策略的选择必须与已加载的工作流及当前状态相符。

| 工作流 | 命令 |
| --- | --- |
| DEBUG / 会话 | `web:select_mode:<real\|sim\|step\|manual>`, `web:switch_task:<task>`, `web:select_strategy:<key>`, `web:connect`, `web:disconnect`, `web:run`, `web:halt`, `web:console_reset` |
| STEP | `web:step_infer`, `web:step_commit`, `web:step_cancel` |
| 手动 | `web:manual_qpos:<comma-separated-qpos>`, `web:manual_dispatch:<dispatch>`, `web:manual_send`, `web:manual_home` |
| 采集 | `web:tab_switch:collect`, `web:collect_arm:<0\|1>`, `web:collect_home`, `web:collect_start`, `web:collect_stop`, `web:collect_cancel` |
| 评测 | `web:tab_switch:eval`, `web:switch_ckpt:<slot>`, `web:switch_task:<task>`, `web:warmup`, `web:start`, `web:stop`, `web:eval_cancel`, `web:reset` |
| RL | `web:tab_switch:rl`, `web:rl_select_task:<task>`, `web:rl_select_policy:<slot>`, `web:rl_select_critic:<slot>`, `web:rl_setup`, `web:run`, `web:halt`, `web:console_reset` |
| Rollout / HIL | `web:rollout_stop`, `web:rollout_save`, `web:rollout_intervention_enabled:<0\|1>`, `web:rollout_intervention_abandon` |
| 回放 | `web:load_replay_dataset:<payload>`, `web:clear_replay`, `web:select_episode:<index>`, `web:set_replay_fps:<fps>`, `web:replay_seek:<frame>` |

`web:select_task` 不在命令目录中，请使用 `web:switch_task:<task>`。当前的命令目录也不包含 `web:setup` 和 `web:select_collect_task`。在通过 ZMQ 启动运行之前，必须先在浏览器或进程内的无界面 CLI 中完成 DEBUG setup；采集的数据集 / 任务选择则使用 HTTP 控制台 API。夹爪命令支持 `web:gripper:<side>:<open|close>:<0|1>`（最后一个值控制是否锁定）、数值形式，或只指定一侧的切换形式。命令目录中还包括引导、初始化、操作员动作和推理参数等命令；这些专用流程请参照目录中的命令模板。

浏览器模式下，完整的命令目录通过 `GET /api/config` 的 `control_channel.commands` 字段发布，其中包括规范的命令模板和相关的 DOM 控件。在无界面模式下，`config` 查询会提供同样的元数据。命令在白名单内，并不保证任意参数或任意状态下都有效。

## 采集与 SIM 元数据

先通过浏览器或 HTTP 采集 API 选择数据集 / 任务和槽位，然后进入 COLLECT 页，并用显式命令启用遥操作：

```json
{"cmd":"web:tab_switch:collect"}
{"cmd":"web:collect_arm:1"}
{"cmd":"web:collect_start"}
{"cmd":"web:collect_stop"}
```

每一行作为一个单独的请求发送，并在存在依赖关系的步骤之间轮询状态。`tab_switch` 上的 `armed` 键并不会启用运动。STOP 会结束本次录制，但遥操作保持启用；HOME 之前请先用 `web:collect_arm:0` 关闭遥操作。

挂载了场景规划时，必须先选择一个采集槽位。槽位选择以及跳过 / 推迟由 HTTP 采集 API 提供。仅选择数据集 / 任务并不能代替槽位绑定。请轮询 `status.collect` 获取录制 / 保存状态。

对于配置了 `collection.sim_recording_url` 的场景（EVA-SIM，即将发布），请在 `web:collect_start` 中传入 `sim_metadata` 对象，其中包含所选仿真器场景 / 任务的元数据。录制器会将其与仿真器状态进行校验，并锁定 `scene_revision`。非对象类型的元数据会被拒绝。该快照只会被本次录制使用；下一次录制前需要重新绑定场景。详见[录制](./recording.md#eva-sim-execution-recording)。

## 评测尝试

提供 `clip_id` 时，`web:start` 支持与 HTTP start API 相同的尝试绑定流程：

```json
{
  "cmd": "web:start",
  "clip_id": "unique-attempt-id",
  "prompt": "pick cup",
  "trial": 1,
  "model_name": "configured-model-name",
  "task_set": "configured-task-set",
  "task_id": "TASK-001",
  "scene_id": "SC-001"
}
```

EVA 必须处于 `ready` 阶段。trial 编号为正整数。`model_name` 可选，但如果提供，必须与当前模型一致。`task_set`、`task_id` 和 `scene_id` 要么同时提供，要么全部省略（此时为纯提示词尝试）。提示词和场景必须属于所选的规划。可选的 `sim_metadata` 会检查其中的场景 / 任务标识是否匹配。应答成功表示该尝试已被预留，其阶段变为 `starting`。

使用 `web:stop` 结束一次评测尝试，并等待其记录保存完成。里程碑打分仍然是 HTTP / 浏览器操作（`/api/score`），与录制的 clip 和模型绑定。控制通道不提供打分命令。DEBUG 的 `web:run`/`web:halt` 与评测的 `web:start`/`web:stop` 分别服务于各自的状态机流程。

## VR 触觉反馈

```json
{"cmd":"web:haptic","hand":"right","intensity":0.6,"duration_ms":80}
```

`hand` 可取 `left`、`right`、`both`，或别名 `all`（默认 `both`）。强度会被限制在 `0..1` 范围内；时长会被限制在 `1..1000` ms 范围内。默认值分别为 `0.6` 和 `80` ms。数值必须是有限值。也支持位置参数形式：

```text
web:haptic:right:0.6:80
```

应答会报告实际送入原生 EVA-VR 客户端发送队列的手。头显未连接时返回 `ok:false`。原生应用通过 OpenXR 施加反馈；该命令不会改变机器人会话状态。

## 查询与 CLI 客户端

| 查询 | 内容 |
| --- | --- |
| `status` | 实时会话 / 阶段、策略 / 通信层健康状态、采集与 rollout 状态、历史记录以及最近一次错误 |
| `config` | 机器人 / 通信层、任务、模式、推理策略、评测 / RL 配置以及命令元数据 |
| `frame` | 最新的可视化帧、qpos 和相机键 |

[`eva_ctl.py`](../../examples/control_channel/eva_ctl.py) 提供了一个 REQ 客户端：

```bash
.venv/bin/python examples/control_channel/eva_ctl.py query config
.venv/bin/python examples/control_channel/eva_ctl.py cmd web:switch_task:pick_cup
.venv/bin/python examples/control_channel/eva_ctl.py cmd web:run
.venv/bin/python examples/control_channel/eva_ctl.py query status
.venv/bin/python examples/control_channel/eva_ctl.py wait-idle
.venv/bin/python examples/control_channel/eva_ctl.py haptic right --intensity 0.6 --duration-ms 80
```

请把 `pick_cup` 替换为已配置的任务字符串。`wait-idle` 会一直轮询，直到 `session_status` 不再是 `running`；由于命令是排队执行的，它并不能作为启动确认。自动化运行时，请先观察到预期的 running / 阶段切换。无界面模式还提供一个仅限 TTY 的 REAL 推理提示；见[控制台](./console.md#headless--low-power-mode)。

## 扩展命令目录

在 [`run.py`](../../src/core/app/run.py) 中实现对应的状态机分支，并在共享命令目录中添加该命令的条目。命令目录同时提供 ZMQ 白名单和控制台快捷键元数据。需要像 HTTP 那样修改上下文的命令，还必须在 [`control_channel.py`](../../src/core/app/control_channel.py) 中实现相应的绑定；仅仅把字符串放入队列，并不能复现 HTTP 处理函数对元数据的写入。
