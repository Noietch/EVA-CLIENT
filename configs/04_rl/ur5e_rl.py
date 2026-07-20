"""UR5e RL workspace with independent policy and critic selection."""

_base_ = ['../01_deploy/ur5e/openpi_qpos.py']

rl_cfg = dict(
    cli_mode='real',
    inference_strategy='async',
    tasks=['placeholder task — replace with the real RL task prompt'],
    policies=[
        dict(
            name='ur5e_openpi_qpos',
            config='../01_deploy/ur5e/openpi_qpos.py',
            host='127.0.0.1',
            port=9000,
        ),
    ],
    critics=[
        dict(
            name='ur5e_critic',
            type='websocket',
            host='127.0.0.1',
            port=9100,
            backend_options={},
        ),
    ],
    data=dict(
        format='lerobot',
        storage=dict(
            log_dir='work_dirs/rl/ur5e',
            fps=20,
            save_queue_max=15,
            async_save=True,
        ),
    ),
    intervention=dict(control_mode='relative'),
)
