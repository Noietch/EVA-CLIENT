"""Shared model and workflow defaults; hardware defaults come from each device config.yaml."""

robot = dict(type="agilex_piper")

transport = dict(
    image_mode="stream",
    image_request_timeout_s=2.0,
    dataset_dir="",
    episode_id=0,
    convert_bgr_to_rgb=True,
    image_height=224,
    image_width=224,
    resize_pad=True,
    image_layout="chw",
    dataset_keys=dict(
        state_key="observations.state.qpos",
        eef_key="observations.state.eef",
        action_key="action",
        video_keys=dict(),
    ),
)

policy = dict(type="openpi", host="127.0.0.1", port=9000, backend_options=dict())

collection = dict(
    controls=dict(
        keyboard=dict(
            motion=dict(control="KeyM", key="M", gesture="tap"),
            record_toggle=dict(control="KeyR", key="R", gesture="tap"),
            record_cancel=dict(control="Escape", key="ESC", gesture="hold", hold_ms=1000),
            home=dict(control="KeyH", key="H", gesture="tap"),
        ),
        vr=dict(
            motion=dict(control="right.secondary", key="B", gesture="tap"),
            record_toggle=dict(control="right.primary", key="A", gesture="tap"),
            record_cancel=dict(control="right.primary", key="A", gesture="hold", hold_ms=1000),
            home=dict(control="left.primary", key="X", gesture="tap"),
            left_arm_toggle=dict(control="left.grip", key="L GRIP", gesture="hold", hold_ms=1000),
            right_arm_toggle=dict(control="right.grip", key="R GRIP", gesture="hold", hold_ms=1000),
        ),
    ),
    storage=dict(log_dir="", fps=30, save_queue_max=15, image_skew_tolerance_sec=None),
    schema=dict(
        min_episode_frames=1,
        max_frame_dt_factor=3.0,
        columns={},
    ),
    teleop=dict(control_source="transport", client=dict()),
    task_set_dir="",
    task_set_name="",
    tasks=dict(),
)

rollout = dict(
    storage=dict(enabled=False, log_dir="", fps=30, save_queue_max=15, async_save=True),
    intervention=dict(control_mode="absolute"),
)

# Console startup workspace. "auto" preserves the workflow-specific defaults:
# EVAL for evaluation configs, otherwise DEBUG.
console = dict(
    initial_tab="auto",
)

# ZMQ control channel: exposes every console button (web:* commands) + read-only
# status/config/frame queries over a REP socket, for a simulator to drive automated
# evaluation. Disabled by default; host stays local unless a deploy opens it up.
control_channel = dict(
    enabled=False,
    host="127.0.0.1",
    port=5757,
)

eval_cfg = {}  # Empty dict marks a non-eval config; eval configs fill this block.

# RL workspace configuration. Empty keeps the RL tab unavailable; dedicated RL
# launch configs provide tasks, policy/critic choices, and data settings.
rl_cfg = {}
# Eval block template (see configs/03_evaluation/*); fill eval_cfg to turn a deploy preset
# into an eval run:
#   eval_cfg = dict(
#       storage=dict(fps=30, save_queue_max=15),
#       trials_per_prompt=5,
#       cli_mode="real",
#       inference_strategy="async",
#       reset_after_each_trial=False,
#       skip_warmup_after_first=True,
#       checkpoints=[dict(name="<ckpt>", config="<deploy_preset.py>", port=9000)],
#       shuffle_ckpts=False,
#       shuffle_seed=42,
#       enable_ssh_forward=False,
#       ssh=dict(host="", user="", port=8000, remote_sync_dir=""),
#       tasks=[dict(prompt_en="pick the apple", milestones=(("grasp", "grasp apple"),))],
#   )

# Root for all generated artifacts (collection logs, eval results). Subpaths under
# it are derived by convention; override per config to redirect output elsewhere.
work_dir = "work_dirs"

# All inference-runtime configuration in one section.
# obs_space / action_space are dicts that load_config replaces with the
# corresponding class instance (JointState / EEFPose) via transform.build_space.
inference_cfg = dict(
    obs_space=dict(type="JointState"),
    action_space=dict(type="JointState"),
    inference_rate=3.0,
    publish_rate=30,
    max_debug_time_s=300.0,
    setup_warmup_chunks=2,
    debug_tasks=["pour soybean", "put cup"],
)

# Multi-preset strategy dict, switchable from the frontend. Each entry =
# dict(type="ClassName", args=dict(...)). Optional args keys per strategy (uncomment in args):
#   all:    execute_horizon=<int>            # chunk crop length, >=1 (omit -> run full chunk)
#   sync/rtc: ignore_gripper_in_sync_wait=False
#   async/naive/rtc: latency_k=<int>         # front-trim of new chunks, >=0
#   act/async: exp_weight_m=0.01             # temporal-ensembling decay, >=0
inference_strategies = {
    "sync": dict(type="BaseInferStrategy", args=dict(execute_horizon=5)),
    "async": dict(
        type="AsyncLinearOverlapInferStrategy",
        args={},
    ),
    "naive": dict(
        type="NaiveAsyncInferStrategy",
        args=dict(latency_k=4),
    ),
    "act": dict(
        type="ActEnsembleInferStrategy",
        args=dict(exp_weight_m=0.01),
    ),
    "rtc": dict(type="RtcInferStrategy", args={}),
}
