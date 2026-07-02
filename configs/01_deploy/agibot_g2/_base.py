"""AgiBot G2: shared deploy settings for openpi (eef / qpos)."""

_base_ = ['../../00_base/defaults.py']

robot = dict(
    type='agibot_g2',
    gripper_threshold=-0.4,
    gripper_open=0.0,
    gripper_close=-0.785,
)

transport = dict(
    type='zmq',
)

policy = dict(
    type='openpi_rtc',
    backend_options=dict(latency_k=4),
)

inference_cfg = dict(
    debug_tasks=[
        'Put the pen into the pen holder.',
        'Close the laptop.',
        'Put the tissue into the trash bin.',
        'Place the mouse on the right side of the laptop.',
        'Straighten the colored pencil box.',
    ],
)

inference_strategies = {
    'async': dict(
        args=dict(
            latency_k=4,
        ),
    ),
    'rtc': dict(
        args=dict(
            latency_k=4,
        ),
    ),
}
