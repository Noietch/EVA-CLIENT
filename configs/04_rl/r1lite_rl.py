"""R1 Lite RL workspace with independent policy and critic selection."""

_base_ = ['../01_deploy/r1lite/openpi_qpos.py']

rl_cfg = dict(
    cli_mode='real',
    inference_strategy='async',
    tasks=['placeholder task — replace with the real RL task prompt'],
    policies=[
        dict(
            name='r1lite_openpi_qpos',
            config='../01_deploy/r1lite/openpi_qpos.py',
            host='127.0.0.1',
            port=9000,
        ),
    ],
    critics=[
        dict(
            name='r1lite_critic',
            type='websocket',
            host='127.0.0.1',
            port=9100,
            backend_options={},
        ),
    ],
    data=dict(
        format='lerobot',
        storage=dict(
            log_dir='work_dirs/rl/r1lite',
            fps=15,
            save_queue_max=15,
            async_save=True,
            image_height=360,
            image_width=640,
        ),
    ),
    intervention=dict(control_mode='relative'),
)
