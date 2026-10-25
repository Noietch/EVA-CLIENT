[← Back to README](../README.md)

# Collection scene plans

Scene plans are runtime inputs shared by COLLECT, the dashboard, evaluation, and
the dataset review service. A plan describes tasks, object placements, layout
coordinates, and the number of recordings required for each task/scene pair.
The console expands it into stable collection slots; no migration script or
historical task-count snapshot is required to use it.

## Directory and configuration

The default data root is `datasets/data_collection`:

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

`collection.task_set_dir` accepts one leaf, a parent directory, or a list of
roots. Discovery checks direct children and the supported version directories
`larybenchv2.v1.1` and `larybenchv2.v1.2`; it does not recursively search arbitrary
historical trees. Dataset names must be unique across mounted collection roots.
A single leaf defaults to its directory name; `task_set_name` can override that
single-set name. Parent/list configs expose each discovered leaf by name.

```python
collection = dict(
    task_set_dir="datasets/data_collection/task_sets",
    storage=dict(data_root="datasets/data_collection"),
)
```

[ARX X5 task-set collection](../configs/02_collection/arx_x5_tasks_set.py)
and [dual Piper task-set collection](../configs/02_collection/dual_agilex_piper_tasks_set.py)
are maintained config examples. Inline `collection.tasks` remains available for
workflows without scene files. A missing single task-set directory leaves the
authored tasks in effect; a mounted directory must have a valid `tasks.csv`.

`info.yaml` can define:

```yaml
collection_dir: datasets/real_robot/dual_yam/example_plan
task_order: spatial
objects_file: objects.csv
```

The dataset basename must match the selected dataset name. `collection_dir` must
start with `datasets/`, remain relative, and contain no `..`. When omitted, the
path defaults to `datasets/real_robot/<robot_type>/<dataset>`. SIM plans can
provide their own `datasets/eva_sim/...` path. Runtime cursor/log state is kept
under `collection.storage.log_dir` (or `work_dir`), with host-specific slot files.

## Layout: `layout.yaml`

Coordinates come from the layout file, never from a `P1`/`P2` naming convention.
Position IDs can represent a grid, an irregular surface, or shelf locations.

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

Keep approximate coordinates `unverified`. The runtime considers a layout
calibrated only when every parsed point has both coordinates and is marked
`verified`. Point-level unit/frame/status values can override layout defaults.
`display_grid` optionally describes the display arrangement. The map is a scene
reference; marking it calibrated does not itself calibrate the robot or cameras.

## Scenes: `scene.csv`

Each `scene_id` identifies a placement configuration. `placements` is a JSON
array stored in a CSV cell; quote that cell using normal CSV rules.

| Field | Meaning |
| --- | --- |
| `scene_id` | Stable scene reference |
| `layout_id` | Authored layout reference |
| `placements` | Objects and the position IDs they occupy |

A placement has this shape:

```json
[
  {"object_id": "OBJ-RACK", "position_ids": ["P1", "P2"]},
  {"object_id": "OBJ-CUP", "position_ids": ["P3"], "fixed": false,
   "pose_xyzw": [0.2, 0.1, 0.0, 0.0, 0.0, 0.0, 1.0]}
]
```

A continuous object spanning multiple points remains one placement with multiple
`position_ids`. The map expands those points while preserving their grouping.
The parser also retains a seven-value `pose_xyzw` and `fixed` when authored for
simulator-aware plans. Tasks reference scenes; scenes do not duplicate their
own list of tasks.

## Tasks: `tasks.csv`

Use these exact column spellings. The current schema retains `epsiodes` in its
count field names.

| Field | Meaning |
| --- | --- |
| `task_id` | Stable task identity used in slots and metadata |
| `prompt_en` | Nonempty runtime task text |
| `prompt_zh` | Optional translated display text |
| `total_epsiodes_count` | Integer total target; required when loading collection tasks |
| `scene_ids` | Semicolon-separated scene IDs, in corresponding-count order |
| `scene_epsiodes_count` | Semicolon-separated counts aligned with `scene_ids` |
| `action`, `category` | Optional task classification |
| `operation_object` | Optional object label(s), separated with `/` |
| `operation_object_ids` | Optional semicolon-separated object IDs for spatial ordering |

For example, a task with `scene_ids=SC-001;SC-002` and
`scene_epsiodes_count=3;2` expands into five slots. Set
`total_epsiodes_count=5` to match. Scene counts drive the fixed slot plan; no
slots are generated for a missing/nonpositive count. The old `total_target` and
`scene_targets` fields are not the current schema.

Config loading merges identical English prompts into the recorder's prompt
list while retaining task-ID-to-prompt bindings. Distinct task IDs still produce
distinct slots. `collection.task_prompt_prefixes` can add a per-dataset runtime
prefix; the task's authored text remains available to the scene UI.

## Objects and photos

`info.yaml.objects_file` selects the object catalog relative to the plan root;
its default is `objects.csv`. Shared catalogs can be referenced by relative
path. The parser supports these fields:

| Field | Meaning |
| --- | --- |
| `object_id` | Placement reference; the catalog also accepts `物品代码` |
| `object_name` | English/canonical name; `英文名` is accepted |
| `object_name_zh` | Chinese display name; `名称` is accepted |
| `photo_dir` | Photo subdirectory; `照片目录` or `名称` can supply it |
| `color` | Optional display color |

Photos are read from `object_photos/<photo_dir>/` next to the referenced object
catalog. The first sorted JPG/JPEG/PNG/WebP image is used for the preview. Display
names prefer Chinese, then canonical name, then object ID. A missing photo leaves
a text placeholder. The runtime does not discover a photo from a legacy
`photo_path` column or an assumed `<object_id>.jpg` filename.

## Slot order, progress, and retakes

The console and dataset tools use one slot planner in
[`scene_plan.py`](../src/core/utils/scene_plan.py). Order is scene → task → round:

1. Scenes follow the parsed scene order.
2. With `task_order: spatial` (the default), tasks sort by first-mentioned hand
   (left, right, unknown), then the operated object's row-major layout position.
   Coordinates determine spatial order; points without usable coordinates retain
   authored order after points with coordinates.
3. `task_order: csv` preserves CSV task order within each scene.
4. Each task/scene pair expands its target into zero-based rounds.

The stable key is `<task_id>:<scene_id>:<round_index>`. An episode stores this
`slot_id`, task/scene IDs, and `scene_round`. Changing IDs or counts changes the
plan's relationship with existing recordings, so preserve referenced identities
when editing an active plan.

The picker can filter by scene/task, select a saved take for replay/review, or
defer a slot. Cursor and deferred choices are host-local. Saved records and the
QC ledger determine progress; TO COLLECT, UNREVIEWED, PASSED, and FAILED remain
distinct. A retake replaces the slot's existing episode index. See
[Recording](./recording.md) for QC and save semantics.

The current runtime generates this fixed plan directly from the catalogs. It
does not generate a seeded `run_plan.csv` or randomize position assignments.
Author the required scene IDs and counts explicitly when changing coverage.

## Evaluation scene identity

`eval_cfg.task_set_dir` can select evaluation roots independently. With no explicit
root, evaluation discovers plans under `<data_root>/task_sets` and falls back to
the collection roots when none are found. Version-qualified names distinguish
evaluation leaves when needed.

EVAL selects a task and one of its referenced scenes. Start requests supply
`task_set`, `task_id`, and `scene_id` together; the server validates the prompt,
scene membership, and any supplied simulator task/scene metadata. The saved
attempt includes the selected scene-map snapshot, so RESULT can reconstruct the
layout associated with that attempt. A task-set name containing `eval` in a
collection config chooses which layouts to capture; it does not by itself start
a policy evaluation.
