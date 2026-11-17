"""ARX X5: example real-robot eval against XPolicyLab policy servers.

Each checkpoint entry is one served model: joint/qpos models use the joint
preset and eef models use the eef preset. A checkpoint runs on its deploy preset
(publish rate, strategy args) with only this eval section grafted on. Point
task_set_dir at a task set and each host/port at the server of that model.
"""

_base_ = ["../01_deploy/arx_x5/xpolicylab.py", "../00_base/evaluation.py"]

console = dict(initial_tab="eval")

eval_cfg = dict(
    task_set_dir="work_dirs/tasksets/<your_task_set>",
    # One task-set row is one episode-level eval case, so one trial covers it.
    trials_per_prompt=1,
    cli_mode="real",
    # Execute the whole returned chunk, then read the next observation. Use
    # "async" to overlap inference with execution.
    inference_strategy="sync",
    checkpoints=[
        dict(
            name="arx_x5_xpolicylab_qpos",
            config="../01_deploy/arx_x5/xpolicylab.py",
            host="127.0.0.1",
            port=8410,
        ),
        dict(
            name="arx_x5_xpolicylab_eef",
            config="../01_deploy/arx_x5/xpolicylab_eef.py",
            # Pin EEF to link6 when the training demos used link6 FK.
            robot=dict(eef_links=["link6", "link6"]),
            host="127.0.0.1",
            port=8411,
        ),
    ],
)
