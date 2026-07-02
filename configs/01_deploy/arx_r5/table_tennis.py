"""ARX R5: table-tennis deploy task preset."""

_base_ = ['openpi_qpos.py']

inference_cfg = dict(
    inference_rate=3,
    publish_rate=30,
    debug_tasks=[
        'play the table tennis',
    ],
)


robot = dict(
    initial_qpos=[
        0.00362,
        -0.00095,
        0.00439,
        -0.03376,
        -0.00057,
        -0.00858,
        0.0,
        0.12684,
        0.98974,
        0.91001,
        -0.40036,
        -0.04864,
        -0.11349,
        0.0,
    ],
)
