"""CLI state — session/runtime dataclasses and data helpers shared by the web
servers (console / eval). The interactive terminal UI has been removed; status is
surfaced over HTTP (/api/status), so the former full-screen TUI renderers are gone.
"""

from __future__ import annotations

import dataclasses
import enum
import logging
import queue
import threading
from typing import TYPE_CHECKING, Any

import numpy as np

from core.config import ConfigDict
from core.recorder.episode import EpisodeLogger
from core.types import RawCollectionSnapshot, RolloutInterventionSegment
from policy_client.base import PolicyClient
from robots.base import Robot
from strategy.base_strategy import BaseInferStrategy
from transport.base import TransportBridge

if TYPE_CHECKING:
    from tqdm import tqdm

    from core.app.collection_capture import CollectionCaptureRunner
    from critic_client.runner import CriticRunner
    from transport.dataset import DatasetTransport

logger = logging.getLogger(__name__)


class SessionMode(str, enum.Enum):
    """Operating mode — determines the execution pipeline.

    SELECT: initial unset mode. REAL: continuous real-robot execution. SIM:
    simulation-only inference. MANUAL: hand-drive qpos/eef to verify control and
    obs links (no policy). STEP: preview in sim then confirm on real.
    COLLECT: teleoperation data collection — actions come from a human over the transport
    stream (not a policy), recorded as episodes.
    """

    SELECT = "select"
    REAL = "real"
    SIM = "sim"
    MANUAL = "manual"
    STEP = "step"
    COLLECT = "collect"


class SessionStatus(str, enum.Enum):
    """Session state machine — tracks progress through the setup-run lifecycle.

    UNSET: no setup done yet. READY: setup complete, waiting for start. RUNNING:
    actively publishing actions. EXIT: session termination requested.
    """

    UNSET = "unset"
    READY = "ready"
    RUNNING = "running"
    EXIT = "exit"


class OutputTarget(str, enum.Enum):
    """Action publish target — REAL sends to the physical robot, SIM sends to
    the simulation visualization topic for preview without physical movement."""

    REAL = "real"
    SIM = "sim"


@dataclasses.dataclass
class SessionState:
    """Mutable per-session state — tracks the current mode, execution status,
    in-flight action chunks, step/chunk counters, task selection, and gripper
    overrides. Reset on mode/task/strategy changes; persists across inference cycles.

    Fields:
        mode: Current operating mode (SELECT/REAL/SIM/MANUAL/STEP/COLLECT).
        status: Session state-machine value (UNSET/READY/RUNNING/EXIT).
        action_chunk: In-flight action chunk currently being executed.
        pending_real_chunk: Action chunk staged for the real robot, awaiting confirm.
        sim_preview_qpos: SIM preview — the inferred command qpos drives the 3D canvas
            directly, since the zmq/ros2 transports have no sim echo loop to feed it
            back as feedback.
        chunk_index: Index of the current chunk within the inference stream.
        step_index: Index of the current step within the active chunk.
        is_setup_done: True once run_setup has completed for this session.
        selected_task: Operator-selected DEBUG/run task; None until chosen.
        selected_collect_task: Operator-selected COLLECT task; None until chosen.
        interrupt_requested: True when a motion interrupt has been requested.
        follow_human_gripper: When True, mirror the human teleop gripper state.
        gripper_locks: Per-arm gripper override values keyed by arm name.
        manual_qpos: MANUAL mode — hand-set command qpos that drives the arm.
        manual_real_qpos: Last qpos actually sent to the real robot in MANUAL mode.
        manual_publish_active: True while MANUAL is publishing a staged target to real.
        manual_dispatch: "stage" only previews until SEND; "sync" mirrors every change
            to the real robot.
        last_error: Most recent error message surfaced to the UI.
        last_infer_ms: Wall-clock duration of the last inference call, milliseconds.
        run_start_time: Monotonic timestamp when the current run started.
    """

    mode: SessionMode = SessionMode.SELECT
    status: SessionStatus = SessionStatus.UNSET
    action_chunk: np.ndarray | None = None
    pending_real_chunk: np.ndarray | None = None
    sim_preview_qpos: np.ndarray | None = None
    chunk_index: int = 0
    step_index: int = 0
    is_setup_done: bool = False
    selected_task: str | None = None
    selected_collect_task: str | None = None
    interrupt_requested: bool = False
    follow_human_gripper: bool = False
    gripper_locks: dict[str, float] = dataclasses.field(default_factory=dict)
    manual_qpos: np.ndarray | None = None
    manual_real_qpos: np.ndarray | None = None
    manual_publish_active: bool = False
    manual_dispatch: str = "stage"
    last_error: str = ""
    last_infer_ms: float = 0.0
    run_start_time: float = 0.0


@dataclasses.dataclass
class RuntimeState:
    """Long-lived runtime state — holds the robot description, transport bridge,
    policy client connection, IK solver instance, inference strategy, and the
    inter-thread command queue for motion-interrupt handling. Survives across
    session resets; only rebuilt on full reconnect.

    Fields:
        robot: Robot description (kinematics, observation schema).
        transport: Main transport bridge publishing to / reading from the robot.
        policy: Connected policy client; None until setup connects it.
        policy_metadata: Metadata returned by the policy server (model info, dims).
        last_policy_error: Most recent policy-connection error message.
        ik_solver: IK solver instance used for EEF-to-qpos conversion.
        selected_inference_strategy_key: Key of the active inference strategy.
        command_queue: Inter-thread queue carrying web/CLI commands to the main loop.
        prompt_ready: Event signaling that a prompt has been selected.
        infer_strategy: Active inference strategy instance.
        episode_logger: Logger writing teleop/collection episodes; None when not recording.
        collection_teleop_armed: True only while the console is on COLLECT with the
            local activation switch on; required before teleop can affect hardware.
        collection_teleop_active: True while a teleop collection run is active.
        last_collection_timestamp: Timestamp of the last recorded collection frame.
        collection_capture_runner: Background raw snapshot capture loop for the
            active collection episode.
        rollout_raw_snapshots: Raw snapshots buffered for policy rollout recording.
        rollout_policy_actions: Policy actions captured independently from raw snapshots.
        collection_replay_qpos: qpos sequence loaded for collection replay.
        collection_replay_episode: Episode index currently loaded for collection replay.
        collection_replay_started: Monotonic timestamp when collection replay started.
        collection_replay_fps: Playback rate for collection replay, frames/sec.
        _eval_warmup_done: True once the eval warmup inference has run.
        web_phase: Coarse web-UI phase indicator ("idle"/...).
        last_client_poll: monotonic timestamp of the last /api/status poll. The frontend
            polls status at a fixed cadence, so this doubles as a client-liveness
            heartbeat: the main loop halts an active run if it goes stale (browser crash /
            network drop). 0.0 = never polled.
        setup_stage: human-readable sub-step of an in-flight run_setup (connecting /
            resetting / inferring / warmup), surfaced on /api/status so the UI can show
            what setup is doing. Empty when no setup is running.
        current_clip_id: clip_id minted by the frontend at Start, surfaced on /api/status
            so the video recorder can tag the clip and pair it to the exact result row.
            Cleared on stop.
        current_cell: (scene, pos, trial) of the in-flight run, set at Start and surfaced
            on /api/status so the recorder can stamp the clip's sidecar with its trial
            identity.
        needs_pre_start_reset: True after a stop/ckpt-switch that left the robot un-reset.
            The next start auto-resets.
        ckpt_order: blind multi-ckpt — ckpt_order[slot] -> index into
            config.eval.checkpoints.
        active_ckpt_slot: operator-facing checkpoint slot (0->Model A, 1->Model B, ...).
        eval_output_dir: eval results root (work_dirs/<eval config>/). Each model records
            into a per-model lerobot dataset at <eval_output_dir>/<model_name>/episodes,
            so a model's prior eval episodes (and their embedded scores) auto-resume next
            session.
        active_config: full ConfigDict with the active ckpt's whitelist overrides applied.
            The main loop and web commands read this in preference to the base config (the
            frozen base config can't be mutated in place to switch ckpt settings).
        replay_source: UI-loaded replay/QC data source — a read-only DatasetTransport
            mounted at runtime (independent of the main transport) supplying recorded
            images/state/qpos for the console while the main transport keeps publishing to
            the real robot. Its presence is the runtime "replay is active" signal
            (replaces config.policy.type == replay).
        replay_dataset_dir: Filesystem path of the loaded replay dataset.
        replay_episode_id: Episode index currently loaded for replay.
        replay_n_episodes: Total number of episodes in the replay dataset.
        replay_trajectory: Loaded replay action/qpos trajectory.
        replay_task: Task string of the loaded replay episode.
        replay_action_key: Dataset column key supplying the replay actions.
        replay_action_mode: action publish mode chosen in the REPLAY tab ("joint"/"eef");
            "" = follow config.
        replay_fps: replay playback rate (frames/sec). Defaults to the dataset's recorded
            fps on load so playback matches data collection; the operator can override it
            live.
        replay_exec_steps: Number of action steps to execute per replay frame.
        rollout_episode_logger: explicit rollout save. Separate from episode_logger
            because collection configs use episode_logger for teleop datasets.
        rollout_save_ready: True when a rollout is staged and ready to be saved.
        rollout_save_reason: Human-readable reason/label for the pending rollout save.
        rollout_intervention_active: True while normal rollout is paused under teleop.
        rollout_intervention_pre_qpos: Robot qpos captured before the active intervention.
        rollout_intervention_active_segment: Temporary segment being captured.
        rollout_intervention_segments: Accepted segments saved with the rollout.
        rollout_intervention_next_segment_index: Monotonic segment id inside the rollout.
        rollout_intervention_enabled: Runtime HIL ON/OFF gate for rollout halt.
        rollout_exclusion_active: Reason and source timestamp for a rollout interval
            excluded until the first policy action is published.
        rollout_exclusions: Completed excluded intervals removed when saving rollout data.
        hil_control_mode: HIL relay mode, either "absolute" or "relative".
    """

    robot: Robot
    transport: TransportBridge
    policy: PolicyClient | None = None
    policy_metadata: dict | None = None
    last_policy_error: str = ""
    ik_solver: Any | None = None
    selected_inference_strategy_key: str | None = None
    command_queue: queue.Queue[str] | None = None
    prompt_ready: threading.Event | None = None
    infer_strategy: BaseInferStrategy | None = None  # pyright: ignore[reportGeneralTypeIssues]
    episode_logger: EpisodeLogger | None = None
    collection_teleop_armed: bool = False
    collection_teleop_active: bool = False
    last_collection_timestamp: float | None = None
    collection_capture_runner: CollectionCaptureRunner | None = None
    rollout_raw_snapshots: queue.Queue[RawCollectionSnapshot] = dataclasses.field(
        default_factory=queue.Queue
    )
    rollout_policy_actions: list[
        tuple[float, np.ndarray, np.ndarray | None, np.ndarray | None, np.ndarray | None]
    ] = dataclasses.field(default_factory=list)
    collection_replay_qpos: np.ndarray | None = None
    collection_replay_episode: int | None = None
    collection_replay_started: float = 0.0
    collection_replay_fps: int = 30
    _eval_warmup_done: bool = False
    web_phase: str = "idle"
    last_client_poll: float = 0.0
    setup_stage: str = ""
    current_clip_id: str | None = None
    current_cell: dict[str, object] | None = None
    needs_pre_start_reset: bool = False
    ckpt_order: list[int] = dataclasses.field(default_factory=list)
    active_ckpt_slot: int | None = None
    eval_output_dir: str = ""
    active_config: ConfigDict | None = None
    replay_source: DatasetTransport | None = None
    replay_dataset_dir: str = ""
    replay_episode_id: int = 0
    replay_n_episodes: int = 0
    replay_trajectory: np.ndarray | None = None
    replay_task: str = ""
    replay_action_key: str = ""
    replay_action_mode: str = ""
    replay_fps: int = 10
    replay_exec_steps: int = 1
    replay_pbar: tqdm | None = None
    rollout_episode_logger: EpisodeLogger | None = None
    rollout_save_ready: bool = False
    rollout_save_reason: str = ""
    rollout_intervention_active: bool = False
    rollout_intervention_pre_qpos: np.ndarray | None = None
    rollout_intervention_active_segment: RolloutInterventionSegment | None = None
    rollout_intervention_segments: list[RolloutInterventionSegment] = dataclasses.field(
        default_factory=list
    )
    rollout_intervention_next_segment_index: int = 0
    rollout_intervention_enabled: bool = False
    rollout_exclusion_active: tuple[str, float] | None = None
    rollout_exclusions: list[tuple[str, float, float]] = dataclasses.field(default_factory=list)
    hil_control_mode: str = "absolute"
    rl_active: bool = False
    rl_selected_policy_slot: int | None = None
    rl_selected_critic_slot: int | None = None
    rl_critic_runner: CriticRunner | None = None
    rl_critic_action_horizon: int | None = None
    rl_critic_error: str = ""
    rl_pending_critic_observation: dict | None = None
    rl_pending_critic_action: np.ndarray | None = None
    rl_pending_critic_timestamp: float | None = None
    rl_live_samples: list[tuple[float, np.ndarray, np.ndarray, str, int]] = (
        dataclasses.field(default_factory=list)
    )
    rl_replay_source: DatasetTransport | None = None
    rl_replay_sources: list[DatasetTransport] = dataclasses.field(default_factory=list)
    rl_replay_lock: threading.Lock = dataclasses.field(default_factory=threading.Lock)
    rl_replay_generation: int = 0
    rl_replay_dataset_dir: str = ""
    rl_replay_episode_id: int | None = None
    rl_replay_timestamps: list[float] = dataclasses.field(default_factory=list)
    # EVAL-tab INIT panel: when init_qpos is set, the arm's reset target becomes this
    # recorded start pose instead of robot.initial_qpos (home). init_ready latches once
    # the operator clicks DONE, gating RUN until the arm is positioned + gripper set.
    init_qpos: np.ndarray | None = None
    init_ready: bool = False


# --- state-transition helpers (log every transition uniformly) ---


def set_status(session: SessionState, new_status: SessionStatus, reason: str = "") -> None:
    """Set session.status and log the transition as ``[STATUS] old -> new (reason)``.

    Logging every status change in one place gives a single grep-able marker for the
    session state machine, so a stuck run can be traced to its last transition.
    """
    old = session.status
    session.status = new_status
    if old is new_status:
        return
    suffix = f" ({reason})" if reason else ""
    logger.info("[STATUS] %s -> %s%s", old.value, new_status.value, suffix)


def set_phase(runtime: RuntimeState, new_phase: str) -> None:
    """Set runtime.web_phase and log the transition as ``[PHASE] old -> new``."""
    old = runtime.web_phase
    runtime.web_phase = new_phase
    if old == new_phase:
        return
    logger.info("[PHASE] %s -> %s", old, new_phase)


def set_setup_stage(runtime: RuntimeState, stage: str) -> None:
    """Set runtime.setup_stage and log it as ``[STAGE] stage``.

    Empty stage marks the end of a setup/reset sequence and is not logged — the next
    real stage (or the completion log) is the meaningful marker. setup and reset are
    the steps most likely to hang, so each entered stage is logged on its own line.
    """
    runtime.setup_stage = stage
    if stage:
        logger.info("[STAGE] %s", stage)


# --- data helpers (no terminal rendering) ---


def format_task_label(task: str | None) -> str:
    """Human-readable task label: None -> "unset", "" -> "<empty>", else the task."""
    if task is None:
        return "unset"
    if task == "":
        return "<empty>"
    return task


def default_inference_strategy_key(config: ConfigDict) -> str:
    """Return the default inference strategy key (first entry in the configured mapping)."""
    return next(iter(config.inference_strategies))


def resolve_inference_strategy_label(
    config: ConfigDict,
    strategy_key: str | None,
) -> str:
    """Resolve a strategy key to a display label, falling back to the default when None."""
    if strategy_key is not None:
        return strategy_key
    return default_inference_strategy_key(config)
