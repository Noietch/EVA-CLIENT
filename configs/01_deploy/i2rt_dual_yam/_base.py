"""I2RT dual YAM shared deploy settings."""

_base_ = ["../../00_base/defaults.py"]

robot = dict(
    type="i2rt_dual_yam",
    gripper_threshold=None,
)

transport = dict(
    type="zmq",
    resize_pad=False,
    image_layout="hwc",
    # The currently attached setup has one scene D405. Remove these entries in
    # a local override when wrist cameras are installed and mapped by serial.
    disabled_cameras=["cam_left_wrist", "cam_right_wrist"],
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
        log_dir="work_dirs/rollout/i2rt_dual_yam",
        fps=30,
        save_queue_max=15,
        async_save=True,
    ),
    intervention=dict(control_mode="relative"),
)
