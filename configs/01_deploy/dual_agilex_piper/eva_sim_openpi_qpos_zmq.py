"""Dual Piper: openpi policy over EVA Sim ZMQ transport, qpos action space."""

_base_ = ["openpi_qpos.py"]

robot = dict(
    extra_cameras=dict(third_person="cam_third_person"),
)

transport = dict(
    type="zmq",
    resize_pad=False,
    image_layout="hwc",
    convert_bgr_to_rgb=False,
    topics={},
    ros_master_teleop=dict(
        enabled=True,
        python="/home/agilex/miniconda3/envs/aloha/bin/python3",
        left_topic="/master/joint_left",
        right_topic="/master/joint_right",
    ),
    ssh=dict(host="10.1.122.20", user="yangdonglin", port=22),
    # ssh=dict(host="", user="", port=0),
)

# control_channel = dict(
#     enabled=True,
#     host="127.0.0.1",
#     port=5757,
# )
