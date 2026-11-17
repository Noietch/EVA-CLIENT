"""ARX R5: teleop data collection (schema + tasks)."""

_base_ = ["../00_base/defaults.py", "../00_base/collection.py"]

console = dict(initial_tab="collect")

collection = dict(
    storage=dict(log_dir="work_dirs/collection/arx_r5/"),
    tasks=dict(
        pick_up_the_apple=[("pick up the apple", 10)],
        pick_up_the_orange=[("pick up the orange", 10)],
    ),
)

robot = dict(type="arx_r5")
