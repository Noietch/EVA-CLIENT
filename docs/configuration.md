[← Back to README](../README.md)

# Configuration

## Presets and inheritance

Business presets are Python files with mmengine-style `_base_` inheritance.
Paths in `_base_` are relative to the containing file. Dictionaries merge
recursively; scalar values and whole lists replace their parent value.
`_delete_=True` replaces an inherited dictionary subtree.

[`configs/00_base/defaults.py`](../configs/00_base/defaults.py) supplies shared
model and workflow defaults. Robot SDK settings, camera choices, endpoints, and
hardware defaults live in `examples/hardware/*/config.yaml` and
`examples/input_sources/*/config.yaml`. The loader puts the selected hardware
defaults underneath the Python preset; a preset's explicit fields win. DEVICE
workspace selections and settings are applied when loading through that workspace.

| Directory | Purpose |
| --- | --- |
| `configs/00_openloop/` | Offline dataset evaluation, optionally launching a policy server |
| `configs/01_deploy/<robot>/` | Robot and model deployment; sibling `_base.py` holds shared settings |
| `configs/02_collection/` | Teleoperation, task sets, recording, and upload settings |
| `configs/03_evaluation/` | Checkpoint/task evaluation and simulator evaluation |
| `configs/04_rl/` | RL policy/critic choices and rollout settings |

For example, a preset beside `configs/01_deploy/dual_agilex_piper/_base.py` can
override its policy while retaining hardware and recording defaults:

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

Use `_delete_=True` in `policy` when switching from a backend with unrelated
options. Lists such as `debug_tasks`, `checkpoints`, and `robot.eef_links` must be
specified in full when replacing them.

## Policy and image inputs

`policy.type` selects `openpi`, `openpi_rtc`, `starvla`, `xpolicylab`, `gr00t`,
`mock`, or `replay`. `host` and `port` select the server; `backend_options` holds
protocol-specific fields. See [core concepts](concepts.md#policy-backends).

Deploy presets also accept `policy` as a non-empty `list[dict]`. Each entry needs
a unique `name`; the first entry is selected at startup. The DEBUG input toolbar
lists these names and lets you switch models while idle:

```python
policy = [
    dict(name="Model A", type="openpi", host="127.0.0.1", port=5555),
    dict(name="Model B", type="openpi", host="127.0.0.1", port=5556,
         backend_options=dict()),
]
```

Each entry can override its backend type, endpoint, and `backend_options`.
Robot, image preprocessing, and inference strategy settings are shared by the
listed models. Switching clears the old policy connection, preview chunks, and
setup state; the next send prepares the selected model. Switching is blocked
while running or preparing. The existing single `policy = dict(...)` format
continues to work. Restart after changing the configured model list.

| Preset | Model contract |
| --- | --- |
| [`dual_yam/dream.py`](../configs/01_deploy/dual_yam/dream.py) | DreamLabs bridge via `openpi`, port 5555, 320 × 176 HWC images, joint actions at 20 Hz |
| [`dual_agilex_piper/xpolicylab.py`](../configs/01_deploy/dual_agilex_piper/xpolicylab.py) | XPolicyLab, port 19000, two 6D joint arms and 1D grippers, ZMQ images on demand |
| [`dual_yam_eva_sim_eval.py`](../configs/03_evaluation/dual_yam_eva_sim_eval.py) | XPolicyLab EEF/TCP evaluation, two 7D pose arms and 1D grippers, ZMQ images on demand, control channel enabled |
| [`dual_agilex_piper_xpolicylab.py`](../configs/00_openloop/dual_agilex_piper_xpolicylab.py) | Offline XPolicyLab example with dataset column mappings and a sibling EVA-XPolicyLab server command; customize local paths |

For XPolicyLab, set `action_type="joint"` or `"ee"`, plus `arm_dims` and
`gripper_dims` for one or two arms. EEF arms require `arm_dims` entries of 7
(`xyz + wxyz`). The observation and action spaces must agree with these widths.
Optional `evaluation_id` sets the protocol evaluation identity; otherwise it is
generated. Timeouts are seconds: `request_timeout_s` defaults to 120,
`connect_timeout_s` to 30, and `handshake_timeout_s` to 60. Presets may override
these values.

Image preprocessing is configured under `transport`:

| Field | Shared default | Meaning |
| --- | --- | --- |
| `convert_bgr_to_rgb` | `False` | Preserve channels by default; `True` swaps red and blue |
| `image_height`, `image_width` | `224`, `224` | Target dimensions when `resize=True`; ignored otherwise |
| `resize` | `False` | Preserve source resolution; `True` enables resizing |
| `resize_pad` | `True` | When resizing is enabled, preserve aspect ratio with padding; `False` resizes directly |
| `image_layout` | `"hwc"` | Preserve channel-last layout; `"chw"` explicitly transposes to channel-first |
| `image_mode` | `"stream"` | ZMQ policy frames use streaming images or `"on_demand"` image requests |
| `image_request_timeout_s` | `2.0` | Maximum wait for a new image revision in on-demand mode |

Check the complete preset chain because a robot or workflow preset may override
these defaults. Match channel conversion to the transport's decoded image order;
the simulator XPolicyLab eval preset preserves the incoming channels and resolution.

Dataset transport also needs `dataset_dir`, `episode_id`, and `dataset_keys`.
Set `state_key`, `action_key`, and, for EEF observations, `eef_key` to the dataset's
actual columns. `video_keys` maps EVA camera names to dataset video columns.
An empty map uses `observation.images.<camera>`; a nonempty map selects only its
mapped cameras.

## Runtime and outputs

`inference_cfg` selects observation/action spaces, inference and publish rates,
warmup chunks, DEBUG prompts, and maximum DEBUG duration. `inference_strategies`
maps console strategy keys to registered classes and their `args`; overriding
one strategy's args preserves its inherited class and other settings.

`work_dir` defaults to `work_dirs`. When collection schema columns are configured
and `collection.storage.log_dir` is empty, the loader derives it as
`<work_dir>/<preset_stem>`. Deployment presets can set explicit rollout paths.
`collection` controls demonstrations and task sets; `rollout` controls
policy episode recording; `eval_cfg.storage` controls evaluation recordings.
See [recording](recording.md) for the dataset layout.

`console.initial_tab` defaults to `"auto"` (EVAL for an evaluation config,
otherwise DEBUG); presets may choose a workspace explicitly.
`control_channel.enabled` defaults to `False`, with local host `127.0.0.1` and
port 5757. See [control channel](control-channel.md) for automation.

## Evaluation and scene plans

Inherit a deploy preset and
[`configs/00_base/evaluation.py`](../configs/00_base/evaluation.py), then fill
`eval_cfg`. Each checkpoint contains a name, a deploy `config` path relative to
the evaluation preset, a `port`, and an optional `host` (default `127.0.0.1`).
The loader resolves each deploy config independently and overrides its policy
endpoint with that checkpoint's host and port.

```python
_base_ = ["../01_deploy/dual_yam/dream.py", "../00_base/evaluation.py"]

eval_cfg = dict(
    task_set_dir="datasets/data_collection/task_sets/v1.2",
    trials_per_prompt=5,
    inference_strategy="async",
    checkpoints=[dict(
        name="dream_yam",
        config="../01_deploy/dual_yam/dream.py",
        host="127.0.0.1",
        port=5555,
    )],
)
```

`task_set_dir` accepts a directory or list of directories. Relative paths resolve
from the repository root. A directory may contain `tasks.csv` directly, task-set
children, or known version folders. EVAL only loads paths explicitly configured
in `eval_cfg.task_set_dir`. Without that setting, no evaluation task sets are loaded;
the data root and collection task sets are not used as fallbacks. Versioned plans keep their
version in the displayed name, for example `v1.2/<task_set>`.

The console combines plan tasks with matching authored `eval_cfg.tasks`
metadata, including milestones. Scene-aware attempts require `task_set`,
`task_id`, and `scene_id` together; the selected task must belong to that scene
and its prompt must match. Each attempt preserves the scene-map snapshot for
RESULT review. Plain prompt tasks remain supported without scene identity.
See [scene plans](collection_scene_plan.md) for CSV/YAML files and
[console](console.md) for the operator workflow.

Startup merges the preset chain and hardware defaults, loads task sets, builds
control spaces, derives paths, validates settings, and resolves evaluation/RL
policy configs. An empty `eval_cfg` or `rl_cfg` disables that workflow.


## Dataset browsing sources

```python
dashboard = dict(raw_roots=["work_dirs/my_eval", "datasets/my_collection"])

# Historical model directories below the evaluation output directory to show in RESULT.
eval_cfg = dict(result_models=["previous_model"])
```

With `dashboard.raw_roots=[]`, Dashboard stays empty. It does not automatically
include `work_dir` or the current run directory. RESULT reads configured checkpoints
and explicitly listed `result_models`; unrelated model directories are ignored.
For the standalone dataset browser, edit `configs/dataset_browser.py` and launch
`bash tools/datasets/start.sh --config configs/dataset_browser.py`.
