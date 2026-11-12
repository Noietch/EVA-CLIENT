"""ARX X5: local real-robot eval against the served v1.4 XPolicyLab fleet.

Each checkpoint entry is one served real model; joint/qpos models use the joint
preset and eef models use the eef preset. A checkpoint runs on its deploy preset
(publish rate, strategy args) with only this eval section grafted on.

The v1.4 demonstrations were collected with the client FK at link6 (each arm's
base_link), not the URDF TCP, so eef checkpoints pin ``eef_links`` to link6.
"""

_base_ = ["../01_deploy/arx_x5/xpolicylab.py", "../00_base/evaluation.py"]

console = dict(initial_tab="eval")

eval_cfg = dict(
    task_set_dir="/home/arx/projects/EVA-CLIENT/work_dirs/tasksets/larybench2_20261102_v1.4_real",
    # One row in this task set is one episode-level eval case, so a single trial
    # per task covers it. Raise this only to repeat tasks intentionally.
    trials_per_prompt=1,
    cli_mode="real",
    # Sync, as in simulation: execute the whole returned chunk, then read the next
    # observation. Async with latency_k=0 replayed every new chunk from its stale
    # first step (re-opening a closed gripper) and blended the EEF grippers.
    inference_strategy="sync",
    checkpoints=[
        dict(
            name="arx_x5_larybench2_xpolicylab_g05",
            config="../01_deploy/arx_x5/xpolicylab.py",
            host="10.245.71.100",
            port=8410,
        ),
        dict(
            name="arx_x5_larybench2_xpolicylab_pi_05",
            config="../01_deploy/arx_x5/xpolicylab.py",
            host="10.245.6.224",
            port=8411,
        ),
        dict(
            name="arx_x5_larybench2_xpolicylab_internvla_a1_5",
            config="../01_deploy/arx_x5/xpolicylab.py",
            host="10.245.6.224",
            port=8412,
        ),
        dict(
            name="arx_x5_larybench2_xpolicylab_gr00t_n17",
            config="../01_deploy/arx_x5/xpolicylab.py",
            host="10.245.6.224",
            port=8413,
        ),
        dict(
            name="arx_x5_larybench2_xpolicylab_xiaomi_robotics_1",
            config="../01_deploy/arx_x5/xpolicylab_eef.py",
            robot=dict(eef_links=["link6", "link6"]),
            host="10.245.76.49",
            port=8414,
        ),
        dict(
            name="arx_x5_larybench2_xpolicylab_openwam",
            config="../01_deploy/arx_x5/xpolicylab_eef.py",
            robot=dict(eef_links=["link6", "link6"]),
            host="10.245.6.224",
            port=8415,
        ),
        dict(
            name="arx_x5_larybench2_xpolicylab_hy_embodied_05_vla",
            config="../01_deploy/arx_x5/xpolicylab_eef.py",
            robot=dict(eef_links=["link6", "link6"]),
            host="10.245.6.224",
            port=8416,
        ),
        dict(
            name="arx_x5_larybench2_xpolicylab_lingbot_vla",
            config="../01_deploy/arx_x5/xpolicylab.py",
            host="10.245.6.224",
            port=8417,
        ),
        dict(
            name="arx_x5_larybench2_xpolicylab_starvla_real",
            config="../01_deploy/arx_x5/xpolicylab.py",
            host="33.229.215.241",
            port=8418,
        ),
        dict(
            name="arx_x5_larybench2_xpolicylab_pi_0",
            config="../01_deploy/arx_x5/xpolicylab.py",
            host="10.245.6.224",
            port=8419,
        ),
        dict(
            name="arx_x5_larybench2_xpolicylab_vpp2",
            config="../01_deploy/arx_x5/xpolicylab_eef.py",
            robot=dict(eef_links=["link6", "link6"]),
            host="10.245.6.31",
            port=8410,
        ),
    ],
)
