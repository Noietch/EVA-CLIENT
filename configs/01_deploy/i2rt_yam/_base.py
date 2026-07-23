"""I2RT YAM single-arm shared deploy settings."""

_base_ = ["../../00_base/defaults.py"]

robot = dict(
    type="i2rt_yam",
    gripper_threshold=None,
)

transport = dict(
    type="zmq",
    resize_pad=False,
    image_layout="hwc",
    # The currently attached setup has one D405. Remove this entry in a local
    # override when a wrist camera is installed and passed to the hardware node.
    disabled_cameras=["cam_wrist"],
)

policy = dict(type="openpi")

inference_cfg = dict(
    # Manual follower checkout: limit joints to 0.15 rad/s at the default 30 Hz
    # and hold the final target long enough for the bounded tracking trim.
    manual_max_qpos_step=0.005,
    manual_settle_duration=2.0,
)

rollout = dict(
    storage=dict(
        enabled=True,
        log_dir="work_dirs/rollout/i2rt_yam",
        fps=30,
        save_queue_max=15,
        async_save=True,
    ),
    intervention=dict(control_mode="relative"),
)
