"""R1 Lite: shared deploy settings across openpi / eva, eef / qpos."""

_base_ = ["../../00_base/defaults.py", "../../00_base/collection.py"]

console = dict(initial_tab="debug")

robot = dict(type="r1_lite")

collection = dict(
    storage=dict(log_dir="work_dirs/collection/r1lite", fps=15),
    schema=dict(min_episode_frames=10),
)

rollout = dict(
    storage=dict(
        enabled=True,
        log_dir="work_dirs/rollout/r1lite",
        fps=15,
        image_height=360,
        image_width=640,
    ),
    intervention=dict(control_mode="relative"),
)
