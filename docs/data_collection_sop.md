[← Back to README](../README.md) ｜ [中文](data_collection_sop.zh.md)

# Data collection SOP

This is the end-to-end operating procedure for a collection campaign in
EVA-Client: plan the tasks, bring up the station, collect, monitor, review,
repair, and deliver. Every step runs from the browser console or the standalone
dataset browser, and every step works on the same native LeRobot v2.1 dataset.

```text
 Plan ──► Bring up ──► Collect ──► Monitor ──► QC ──► Repair ──► Deliver
 task set   DEVICE      COLLECT    DASHBOARD   auto +   retake /   HF / SFTP /
 (photo or  cameras,    slots,     live flags  human    trim /     S3, format
  no photo) preflight   MOTION/REC             review   edit       conversion
                ▲                                          │
                └──────────── failed slots return ─────────┘
```

| Stage | Where | Output |
| --- | --- | --- |
| 1. Plan | Task-set files, dataset browser | A validated task set with scene counts |
| 2. Bring up | DEVICE tab | Robot, teleop, and cameras ready |
| 3. Collect | COLLECT tab | Saved takes bound to slots |
| 4. Monitor | COLLECT header, DASHBOARD | Live capture health and campaign progress |
| 5. QC | COLLECT, dataset browser | PASS / FAIL verdicts in `meta/qc.jsonl` |
| 6. Repair | COLLECT, dataset browser | Retaken or trimmed episodes |
| 7. Deliver | DASHBOARD dataset manager, COLLECT | Uploaded data and QC, converted formats |

## 1. Plan the task set

A task set defines what to collect: tasks, the scenes each task is recorded in,
and how many takes each task/scene pair needs. The console expands it into fixed
collection slots (`<task_id>:<scene_id>:<round>`), so progress is always
measured against the plan. See [Collection scene plans](./collection_scene_plan.md)
for the full schema.

Each task is shown to the operator in one of two ways:

- **Photo.** Set `reference_image` in `tasks.csv` to a picture of the expected
  setup. The operator matches the table to the photo before recording.
- **No photo.** Leave `reference_image` empty. The operator sees the task prompt
  only.

Start from
[`examples/task_sets/dual_yam_pickplace_example`](../examples/task_sets/dual_yam_pickplace_example):
four tasks, two scenes, ten slots, three tasks with a photo and one without.

Before the campaign:

1. Fill in `info.yaml` (`dataset_name`, `robot_type`, `collection_dir`),
   `tasks.csv`, and `scene.csv`.
2. Open the set in the [dataset browser](./console.md#dataset-operations)
   (`bash tools/datasets/start.sh --config <file.py>`) and run its validation.
   Fix every reported error before collecting.
3. Freeze `task_id`, `scene_id`, and counts once collection starts. Changing them
   changes which saved takes belong to which slot.
4. Publish the task set (Hugging Face task-set sync) so every station uses the
   same version.

## 2. Bring up the station

1. Launch a collection config with the task set mounted, for example:

   ```python
   _base_ = ["dual_yam.py"]
   collection = dict(
       task_set_dir="examples/task_sets/dual_yam_pickplace_example",
       storage=dict(data_root="datasets/data_collection"),
   )
   ```

2. In DEVICE, start the robot, the teleop source, and the cameras. Wait until
   every camera shows a live preview; EVA checks camera readiness before the
   robot is used.
3. Run the preflight and HOME. Use safe stop whenever the arm behaves
   unexpectedly.
4. Several stations can write to the same data root; dataset writes are
   serialized per dataset.

## 3. Collect

1. In COLLECT, choose the dataset. The slot grid shows every scene/task/round.
2. Pick the next TO COLLECT slot, or let the console advance to it. Use the
   scene and task filters to work through one scene at a time.
3. Set up the table to match the task photo (or the prompt).
4. Enable MOTION, then START RECORD. Perform the task.
5. STOP to save the take, or CANCEL to discard it. After STOP the robot stays at
   its current pose and teleop stays active.
6. Disable MOTION before HOME.
7. Confirm the take shows as saved before moving on. A full save queue blocks a
   new recording until saving catches up.

Default operator controls:

| Action | Keyboard | VR controller |
| --- | --- | --- |
| Toggle MOTION | `M` | `B` |
| Start / stop recording | `R` | `A` |
| Cancel the active take | hold `Esc` 1 s | hold `A` 1 s |
| HOME | `H` | `X` |
| Toggle left / right arm | | hold `L GRIP` / `R GRIP` 1 s |

Use **defer** to skip a slot that cannot be recorded now (missing object, wrong
scene); it stays in the plan and can be collected later.

## 4. Monitor

**During a take.** The console header shows inference/step/image rates and the
camera previews update live. A missing, stalled, or frozen camera stream is
flagged on the saved take (`missing_camera_stream`, `frozen_camera`), and the
take is marked red instead of silently entering the dataset.

**Across the campaign.** DASHBOARD reads the configured `dashboard.raw_roots`:

| Panel | What to watch |
| --- | --- |
| Dataset overview | Episodes, frames, hours, QC pass rate, uploaded / not uploaded |
| Collection by day | Throughput per day; gaps show idle stations |
| Usable output | Share of takes that pass QC |
| Upload sync | Datasets with local data not yet uploaded |

Filter by date range and scene to compare stations, operators, or setups. In
headless mode, state transitions and periodic `[HB]` log lines carry the same
progress.

## 5. QC

QC has two layers, and both write to `meta/qc.jsonl` without changing the
recording itself.

**Automatic.** At save time, capture checks mark a take red for short
recordings, missing camera streams, repeated images, invalid vectors, and
timing/alignment problems. After saving, background QC also checks duration
outliers, static or stalled cameras, and joints beyond the robot's URDF limits.

**Human review.** Select a saved take in COLLECT or the dataset browser, watch the
synchronized replay, and mark PASS, FAIL, or UNREVIEWED with a note.

| Slot state | Meaning |
| --- | --- |
| TO COLLECT | No saved take yet |
| UNREVIEWED | Saved, waiting for a human verdict |
| PASSED | A human PASS accepts the take |
| FAILED | Human FAIL, red capture, or automatic rejection |

A human PASS overrides automatic failures; a human FAIL always rejects. A green
capture flag alone does not mean the take passed QC. Review every take before
delivery.

## 6. Repair

- **Retake.** Select a FAILED, deferred, or completed slot and record it again.
  The new take replaces the old one at the same episode index, clears its stale
  review, and returns to UNREVIEWED. The slot count does not grow.
- **Trim.** In the dataset browser, trim the idle head or tail of a good take.
  Trimming rewrites the episode in place and keeps state, actions, and every
  camera stream aligned.
- **Fix metadata.** Edit task descriptions and object metadata in the dataset
  browser instead of editing files by hand.

After a repair, review the take again. Loop until the slot grid has no FAILED or
TO COLLECT slots left for the scenes you are delivering.

## 7. Deliver

1. In the DASHBOARD dataset manager, select the datasets and run **Refresh cloud
   and verify** to compare local and remote data, tasks, and QC.
2. Upload data and upload QC. They are separate actions: data uploads never carry
   QC, and data downloads merge remote QC with local QC instead of overwriting it.
3. Optionally deliver through SFTP or S3 (`collection.storage`) or convert to
   LeRobot v3, HDF5, or MCAP from COLLECT. Conversions include every episode with
   its QC metadata; consumers filter by verdict.
4. Check the job results and upload receipts before reporting the delivery as
   done. Datasets that are recording or saving reject transfer jobs.

## Daily checklist

| When | Check |
| --- | --- |
| Start of shift | Latest task set synced; cameras live; preflight and HOME pass |
| Every take | Saved state confirmed; no red flag; table reset to the photo |
| Every hour | DASHBOARD throughput and usable-output rate; no stalled station |
| End of shift | All takes reviewed; FAILED slots retaken or deferred; data and QC uploaded; Upload sync clean |
