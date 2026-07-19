"""Default configuration for the EVA-CLIENT robot inference client.

Every config under configs/01_deploy/*, configs/00_openloop/*,
configs/02_collection/* and configs/03_evaluation/* should start with:

    _base_ = ["../00_base/defaults.py"]

…then override only the fields it actually needs to change. Field-wise deep
merge (via Config.fromfile / _merge_a_into_b) means partial overrides on
nested dicts work without re-stating the whole section.

The 9 top-level items below are the single source of truth for "what happens
when a field is omitted from a preset". load_config() (src/core/config.py)
does NOT apply fallback defaults — it relies on this base file being merged
in via _base_.
"""

robot = dict(
    type="agilex_piper",
    initial_qpos=None,  # None -> use the robot zoo's bundled initial_qpos
    eef_reference_frame="base_link",
    gripper_threshold=0.5,
    gripper_open=1.0,
    gripper_close=0.0,
)

transport = dict(
    type="ros1",  # "ros1" | "ros2" | "zmq" | "dataset"
    node_name="eva_client",
    dataset_dir="",
    episode_id=0,
    image_height=224,
    image_width=224,
    resize_pad=True,
    image_layout="chw",
    convert_bgr_to_rgb=True,  # BGR->RGB swap for JPEG encode / saved video; source of truth for omitted presets
    sub_endpoint="tcp://127.0.0.1:5555",
    pub_endpoint="tcp://127.0.0.1:5556",
    disabled_cameras=[],
    disabled_groups=[],
    dataset_keys=dict(
        state_key="observations.state.qpos",
        eef_key="observations.state.eef",
        action_key="action",
        video_keys={},  # empty -> fallback to "observation.images.{cam}" pattern
    ),
    topics={},  # ros1/ros2 topic mapping; set per-robot in deploy configs ({} for deep-merge)
    ssh=dict(host="", user="", port=0),
)

policy = dict(
    type="openpi",  # "openpi" | "openpi_rtc" | "starvla" | "gr00t" | "mock" | "replay"
    host="127.0.0.1",
    port=9000,
    # Per-backend optional knobs; which keys apply depends on policy.type. Uncomment as needed:
    #   openpi_rtc:  latency_k=1                       # latency shift s
    #   mock/replay: chunk_size=<int>                  # action chunk length
    #   starvla:     camera_key=<str>, unnorm_key=<str>
    #   gr00t:       api_token=<str>, video_keys=<list>, action_keys=<list>,
    #                state_key="state.qpos", language_key="annotation.human.task_description",
    #                timeout_ms=15000
    backend_options={},
)

collection = dict(
    storage=dict(
        log_dir="",
        fps=30,
        save_queue_max=15,
        # Saved video resolution; uncomment BOTH to resize each saved frame to exactly
        # (image_width, image_height). Omitted/commented -> keep the camera's native size.
        # image_height=224,
        # image_width=224,
    ),
    schema=dict(
        robot_type="",
        min_episode_frames=1,
        max_frame_dt_factor=3.0,
        arms={},
        cameras={},
        columns={},
    ),
    teleop=dict(
        type="",
        port="",
        joint_coef=[],
        gripper=dict(
            source="command",
            mode="toggle",
            threshold=400.0,
            open_value=1.0,
            close_value=0.0,
            raw_open=1000.0,
            raw_close=0.0,
        ),
    ),
    transport=dict(
        ros1=dict(primary_camera="", max_frame_skew_sec=0.1, groups={}),
        ros2=dict(primary_camera="", max_frame_skew_sec=0.1, groups={}),
    ),
    tasks=[],
)

rollout = dict(
    storage=dict(
        enabled=False,
        log_dir="",
        fps=30,
        save_queue_max=15,
        async_save=True,
    ),
    intervention=dict(
        enabled=False,
        control_mode="absolute",
    ),
)

operator_control = dict(
    enabled=False,
    button_topic="/eva/operator_button",
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
    setup_warmup_chunks=2,
    debug_tasks=["pour soybean", "put cup"],
)

manual_cfg = dict(
    publish_rate=15,
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
