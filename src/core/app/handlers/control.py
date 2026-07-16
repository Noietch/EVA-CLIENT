"""Action publishing, gripper control, motion-interrupt polling, and setup/run/mode flows."""

from __future__ import annotations

import dataclasses
import logging
import queue
import time
from functools import partial
from typing import Any, cast

import numpy as np

from core.app.handlers.imaging import build_linear_trajectory, normalize_action_chunk
from core.app.handlers.io import (
    _first_replay_qpos,
    _loop_fetch_chunk,
    _publish_replay_frame,
    _publish_replay_step_chunk,
    connect_policy,
    ensure_action_chunk,
    fetch_action_chunk,
    invalidate_policy_connection,
    start_inference_loop,
)
from core.app.handlers.recording import (
    collect_stop,
    collect_stop_teleop,
    end_episode,
    mark_rollout_save_ready,
    record_executed_action,
    start_rollout_intervention,
)
from core.app.handlers.utils import (
    _format_diag_vector,
    is_offline_transport,
    is_replay,
    reset_ik_solver,
    reset_infer_strategy,
    reset_run_state,
    reset_session_progress,
)
from core.app.operator_control import resolve_operator_action
from core.app.state import (
    OutputTarget,
    RuntimeState,
    SessionMode,
    SessionState,
    SessionStatus,
    format_task_label,
    resolve_inference_strategy_label,
    set_phase,
    set_setup_stage,
    set_status,
)
from core.config import ConfigDict, StrategyYamlArgs, StrategyYamlSpec
from core.registry import STRATEGY_REGISTRY
from policy_client.base import PolicyRequestError

logger = logging.getLogger(__name__)


_MOTION_INTERRUPTING_VERBS = frozenset(
    {
        "select_mode",
        "select_task",
        "switch_task",
        "select_episode",
        "load_replay_dataset",
        "switch_ckpt",
        "reset",
        "console_reset",
    }
)


MANUAL_MAX_QPOS_STEP = 0.02


def apply_gripper_control_overrides(
    runtime: RuntimeState,
    action: np.ndarray,
    session: SessionState,
) -> np.ndarray:
    """Apply gripper overrides to one action before publishing.

    When follow_human_gripper is set, the gripper dims are copied from the live qpos
    (mirroring the teleop hand); otherwise any per-group gripper locks are written in.

    Args:
        action: [total_action_dim] float32 command vector.

    Returns:
        A copy of ``action`` with gripper dims overridden.
    """
    resolved_action = np.asarray(action, dtype=np.float32).copy()
    robot = runtime.robot
    if session.follow_human_gripper:
        latest_qpos = runtime.transport.get_latest_qpos()
        if latest_qpos is None:
            return resolved_action
        for idx in robot.gripper_indices:
            if idx < len(resolved_action) and idx < len(latest_qpos):
                resolved_action[idx] = float(latest_qpos[idx])
        return resolved_action
    for group_name, lock_value in session.gripper_locks.items():
        robot.set_gripper_value(resolved_action, group_name, lock_value)
    return resolved_action


def iter_target_grippers(robot: Any, side: str | None):
    """Yield (group, abs_index, group_index) for each gripper group matching ``side``.

    Group 0 is the left arm, group 1 the right; ``side`` "l"/"r" selects one, None both.
    ``abs_index`` is the gripper's index into the full qpos vector (group offset added).

    Args:
        side: "l", "r", or None for both grippers.

    Yields:
        (group, abs_index, group_index) for each matched gripper actuator group.
    """
    offset = 0
    for i, group in enumerate(robot.actuator_groups):
        if group.gripper_index is not None:
            is_left = i == 0
            if side is None or (side == "l" and is_left) or (side == "r" and not is_left):
                yield group, offset + group.gripper_index, i
        offset += group.dof


def apply_gripper_command(
    config: ConfigDict,
    runtime: RuntimeState,
    session: SessionState | None,
    side: str | None,
    decide,
    lock: bool | None,
    stream: bool,
) -> bool:
    """Drive the targeted gripper(s) to a per-caller value and publish to the real robot.

    Reads the live qpos, writes each matched gripper dim to ``decide(current_value)``,
    optionally records/clears a session lock, then publishes either once (immediate) or
    held for ``publish_rate`` steps (streamed move). Holds the rest of the qpos live so
    only gripper dims move.

    Args:
        decide: maps a gripper's live value to its target value.
        lock: True persists the value as a session lock, False clears it, None leaves
            locks untouched (used when no session is supplied).
        stream: True streams the held target over several steps; False publishes once.

    Returns:
        True when a gripper command was published, False if no feedback / no gripper.
    """
    current_qpos = runtime.transport.get_latest_qpos()
    if current_qpos is None:
        logger.info("No joint feedback available; cannot control gripper")
        return False
    target_qpos = np.asarray(current_qpos, dtype=np.float32).copy()
    applied = False
    gripper_targets: list[tuple[str, float]] = []
    for group, index, _ in iter_target_grippers(runtime.robot, side):
        if index >= len(target_qpos):
            continue
        value = decide(float(current_qpos[index]))
        target_qpos[index] = value
        gripper_targets.append((group.name, value))
        if session is not None:
            if lock is True:
                session.gripper_locks[group.name] = value
            elif lock is False:
                session.gripper_locks.pop(group.name, None)
        applied = True
    if not applied:
        logger.info("No matching gripper group found")
        return False
    if stream:
        steps = max(config.inference_cfg.publish_rate, 10)
        rate = runtime.transport.create_rate(config.inference_cfg.publish_rate)
        for _ in range(steps):
            publish_action(runtime, target_qpos, target=OutputTarget.REAL.value)
            rate.sleep()
    else:
        publish_action(runtime, target_qpos, target=OutputTarget.REAL.value)
    for group_name, value in gripper_targets:
        runtime.transport.record_collection_gripper_action(group_name, value)
    return True


def set_gripper_open_close(
    config: ConfigDict,
    runtime: RuntimeState,
    open_gripper: bool,
    side: str | None = None,
) -> bool:
    """Stream the targeted gripper(s) to the configured open or close value on the real robot.

    Used by the EVAL INIT panel: it physically drives a gripper to a known state
    without setting a session lock, so the policy regains gripper control once RUN
    starts. Holds the arm at its live qpos and only moves the targeted gripper dims.

    Args:
        open_gripper: True opens (config.robot.gripper_open), False closes
            (config.robot.gripper_close).
        side: "l", "r", or None for both grippers (group 0 is left, group 1 is right).

    Returns:
        True when a gripper move was published, False if no joint feedback is available.
    """
    value = config.robot.gripper_open if open_gripper else config.robot.gripper_close
    return apply_gripper_command(
        config, runtime, None, side, decide=lambda _cur: value, lock=None, stream=True
    )


def toggle_gripper_immediate(
    config: ConfigDict,
    runtime: RuntimeState,
    session: SessionState,
    side: str | None = None,
) -> bool:
    """Toggle the targeted gripper(s) once and persist the choice as a lock.

    Reads the live qpos, flips each targeted gripper between config.robot.gripper_open and
    gripper_close (threshold defaults to their midpoint), stores the new value as a lock
    so later publishes keep it, and writes the qpos back so the 3D canvas / qpos panel
    update instantly even while idle.

    Args:
        side: "l", "r", or None for both grippers.

    Returns:
        True when a gripper was toggled, False if no feedback / no gripper.
    """
    # When no explicit threshold is configured, decide open/closed by proximity to
    # the configured open/close targets rather than a fixed 0.5 (gripper ranges vary).
    if config.robot.gripper_threshold is not None:
        threshold = config.robot.gripper_threshold
    else:
        threshold = (config.robot.gripper_open + config.robot.gripper_close) / 2.0

    def decide(cur: float) -> float:
        return config.robot.gripper_close if cur >= threshold else config.robot.gripper_open

    return apply_gripper_command(
        config, runtime, session, side, decide=decide, lock=True, stream=False
    )


def set_gripper_immediate(
    config: ConfigDict,
    runtime: RuntimeState,
    session: SessionState,
    open_gripper: bool,
    side: str | None = None,
    lock: bool = True,
) -> bool:
    """Drive the targeted gripper to a known open/close state once, by absolute value.

    Unlike ``toggle_gripper_immediate`` this never reads the live gripper to decide a
    direction: the operator picks open or close explicitly, so a noisy/lagging feedback
    value can't flip the wrong way or make a second click a no-op. Only the targeted
    group's gripper dim is written; the rest of the qpos holds its live value.

    Args:
        open_gripper: True commands config.robot.gripper_open, False config.robot.gripper_close.
        side: "l", "r", or None for both grippers (group 0 is left, group 1 is right).
        lock: When True, persist the value as a session lock so RUN keeps overriding the
            policy's gripper dim; when False, publish once and let the policy reclaim it.

    Returns:
        True when a gripper command was published, False if no feedback / no gripper.
    """
    value = config.robot.gripper_open if open_gripper else config.robot.gripper_close
    return apply_gripper_command(
        config, runtime, session, side, decide=lambda _cur: value, lock=lock, stream=False
    )


def publish_action(runtime: RuntimeState, action: np.ndarray, target: str) -> None:
    """Publish one action to the given transport target ("real"/"sim"/"data")."""
    runtime.transport.publish_action(action, target=target)


def _publish_manual_real_qpos(
    config: ConfigDict,
    runtime: RuntimeState,
    session: SessionState,
    target: np.ndarray,
) -> bool:
    start = session.manual_real_qpos
    if start is None:
        start = session.manual_qpos
    if start is None:
        start = runtime.transport.get_latest_qpos()
    if start is None:
        start = target
    start = np.asarray(start, dtype=np.float32)
    gripper_mask = np.asarray(runtime.robot.gripper_mask, dtype=bool)
    joint_delta = np.abs(target - start)[~gripper_mask]
    max_delta = float(np.max(joint_delta)) if joint_delta.size else 0.0
    steps = max(1, int(np.ceil(max_delta / MANUAL_MAX_QPOS_STEP)))
    rate = runtime.transport.create_rate(config.inference_cfg.publish_rate)
    trajectory = build_linear_trajectory(start, target, steps + 1)[1:]
    trajectory[:, gripper_mask] = target[gripper_mask]
    for action in trajectory:
        if poll_motion_commands(config, runtime, session):
            consume_motion_interrupt(session, session.status)
            return False
        publish_action(runtime, action, target=OutputTarget.REAL.value)
        session.manual_real_qpos = np.asarray(action, dtype=np.float32).copy()
        rate.sleep()
    return True


def set_manual_qpos(
    config: ConfigDict,
    runtime: RuntimeState,
    session: SessionState,
    qpos: np.ndarray,
) -> None:
    """Set the MANUAL-mode command qpos and publish it.

    Always publishes to sim (drives the external sim / debug echo). Real-robot
    publishing is only available through manual_send().

    Args:
        qpos: [total_action_dim] joint command vector.
    """
    target = np.asarray(qpos, dtype=np.float32).copy()
    session.manual_qpos = target
    publish_action(runtime, target, target=OutputTarget.SIM.value)


def manual_send(config: ConfigDict, runtime: RuntimeState, session: SessionState) -> None:
    """Stage-mode explicit publish: send the current command qpos to the real robot."""
    if session.manual_qpos is None:
        session.manual_publish_active = False
        logger.info("No manual qpos staged; nothing to send")
        return
    session.manual_publish_active = True
    try:
        if _publish_manual_real_qpos(config, runtime, session, session.manual_qpos):
            logger.info("Manual qpos sent to real")
        else:
            logger.info("Manual qpos send paused")
    finally:
        session.manual_publish_active = False


def manual_home(config: ConfigDict, runtime: RuntimeState, session: SessionState) -> None:
    """MANUAL mode: set and publish the robot's home (initial) qpos as the command."""
    set_manual_qpos(config, runtime, session, runtime.robot.initial_qpos)


def publish_real_action_with_preview(
    config: ConfigDict,
    runtime: RuntimeState,
    action: np.ndarray,
    session: SessionState | None = None,
) -> None:
    """Publish one action to the real robot, mirroring it to sim and recording it.

    The sim preview is skipped when a chunk is pending for real (the sim canvas is
    already frozen on the staged playback). When a session is given, the executed
    action is paired with the current obs in the episode loggers.
    """
    should_preview = runtime.transport.has_sim_publishers() and (
        session is None or session.pending_real_chunk is None
    )
    if should_preview:
        publish_action(runtime, action, target=OutputTarget.SIM.value)
    publish_action(runtime, action, target=OutputTarget.REAL.value)
    if session is not None:
        record_executed_action(config, session, action, runtime)


def publish_action_chunk(
    config: ConfigDict,
    runtime: RuntimeState,
    action_chunk: np.ndarray,
    target: OutputTarget,
    session: SessionState | None = None,
    advance_session_chunk_index: bool = False,
) -> bool:
    """Publish a whole action chunk to one target at the configured publish rate.

    For REAL targets with a session: applies gripper overrides per action, and polls
    for motion interrupts between actions so a queued mode/reset can abort mid-chunk.

    Args:
        action_chunk: [T, D] float32 chunk of joint commands.
        advance_session_chunk_index: also bump session.chunk_index per published step.

    Returns:
        True when the full chunk was published, False if it was interrupted or skipped.
    """
    chunk = normalize_action_chunk(action_chunk)
    if session is not None and target == OutputTarget.REAL:
        chunk = chunk.copy()
        for index, action in enumerate(chunk):
            chunk[index] = apply_gripper_control_overrides(runtime, action, session)
    rate = runtime.transport.create_rate(config.inference_cfg.publish_rate)
    for action in chunk:
        if target is OutputTarget.REAL and session is not None:
            if poll_motion_commands(config, runtime, session):
                has_pending = session.pending_real_chunk is not None
                consume_motion_interrupt(
                    session,
                    SessionStatus.READY,
                    clear_pending_real_chunk=has_pending,
                )
                return False
        if target is OutputTarget.REAL:
            publish_real_action_with_preview(config, runtime, action, session=session)
            if session is not None:
                session.step_index += 1
                if advance_session_chunk_index:
                    session.chunk_index += 1
        else:
            publish_action(runtime, action, target=target.value)
        rate.sleep()
    return True


def unlock_prompt(runtime: RuntimeState) -> None:
    """Signal the prompt_ready event so a waiting main loop wakes up."""
    if runtime.prompt_ready is not None:
        runtime.prompt_ready.set()


def handle_motion_command(
    command: str,
    config: ConfigDict,
    runtime: RuntimeState,
    session: SessionState,
) -> bool:
    """Decide whether a command should interrupt an in-flight motion.

    ``halt`` interrupts and is consumed. Verbs that change mode/prompt/episode/ckpt or
    request a reset also interrupt, but are re-queued so the main loop applies them once
    it unwinds — this is what lets a mode switch abort an in-progress setup. All other
    commands are ignored during the motion.

    Returns:
        True when the motion should be interrupted, False to keep going.
    """
    # During an in-flight motion (setup reset trajectory or a running publish loop)
    # only a few verbs may act. `halt` interrupts and is consumed. Verbs that change
    # the mode/prompt/episode/checkpoint or trigger a reset must also interrupt the
    # motion, but are re-queued so the main loop applies them once it unwinds — this
    # is what lets a mode switch abort an in-progress setup instead of being dropped.
    raw_verb = command.strip().lower().removeprefix("web:")
    verb, _, arg = raw_verb.partition(":")
    if verb == "operator_action":
        action, _, source = arg.partition(":")
        resolved_command = resolve_operator_action(
            runtime,
            session,
            action,
            source=source or "ui",
        )
        return (
            resolved_command is not None
            and resolved_command != command
            and handle_motion_command(resolved_command, config, runtime, session)
        )
    if verb == "rollout_stop":
        was_running = session.status is SessionStatus.RUNNING and session.mode in (
            SessionMode.REAL,
            SessionMode.SIM,
        )
        session.interrupt_requested = True
        logger.info("Rollout stop requested; stopping current motion")
        if was_running and not is_replay(runtime):
            mark_rollout_save_ready(runtime, "stop")
        return True
    if verb == "rollout_save":
        was_running = session.status is SessionStatus.RUNNING and session.mode in (
            SessionMode.REAL,
            SessionMode.SIM,
        )
        if not was_running or is_replay(runtime):
            return False
        session.interrupt_requested = True
        mark_rollout_save_ready(runtime, "save")
        if runtime.command_queue is not None:
            runtime.command_queue.put(command)
        logger.info("Rollout save requested; stopping current motion")
        return True
    if verb == "halt":
        was_running = session.status is SessionStatus.RUNNING and session.mode in (
            SessionMode.REAL,
            SessionMode.SIM,
        )
        session.interrupt_requested = True
        logger.info("Interrupt requested; stopping current motion")
        if was_running and not is_replay(runtime):
            mark_rollout_save_ready(runtime, "stop")
            if session.mode is SessionMode.REAL and runtime.rollout_intervention_enabled:
                start_rollout_intervention(config, runtime, session)
        return True
    if verb == "stop" and runtime.web_phase == "running":
        session.interrupt_requested = True
        set_phase(runtime, "stopping")
        if runtime.command_queue is not None:
            runtime.command_queue.put(command)
        logger.info("Interrupting current motion to apply '%s'", verb)
        return True
    if verb in _MOTION_INTERRUPTING_VERBS:
        session.interrupt_requested = True
        if runtime.command_queue is not None:
            runtime.command_queue.put(command)
        logger.info("Interrupting current motion to apply '%s'", verb)
        return True
    return False


def poll_motion_commands(config: ConfigDict, runtime: RuntimeState, session: SessionState) -> bool:
    """Drain queued commands during a motion, applying interrupts.

    Each command is routed through handle_motion_command; the first interrupting verb
    stops the drain. Returns True when the current motion should be interrupted.
    """
    if runtime.command_queue is None:
        return session.interrupt_requested
    interrupted = session.interrupt_requested
    while True:
        try:
            command = runtime.command_queue.get_nowait()
        except queue.Empty:
            break
        if handle_motion_command(command, config, runtime, session):
            interrupted = True
            unlock_prompt(runtime)
            break
        unlock_prompt(runtime)
    return interrupted or session.interrupt_requested


def consume_motion_interrupt(
    session: SessionState,
    fallback_status: SessionStatus,
    clear_pending_real_chunk: bool = False,
) -> None:
    """Clear the interrupt flag and settle the session status after a motion abort.

    Moves to ``fallback_status`` unless an EXIT was requested (which is preserved).

    Args:
        clear_pending_real_chunk: also drop any staged real chunk.
    """
    next_status = SessionStatus.EXIT if session.status is SessionStatus.EXIT else fallback_status
    session.interrupt_requested = False
    session.status = next_status
    if clear_pending_real_chunk:
        session.pending_real_chunk = None


def reset_target_qpos(runtime: RuntimeState) -> np.ndarray:
    """Joint target the arm resets to: the EVAL INIT start pose when set, else home.

    The INIT start pose carries only arm joints; its gripper dims are overwritten with
    the live gripper so a reset never loads/moves the gripper (the operator drives that
    with the INIT panel's OPEN/CLOSE buttons).
    """
    if runtime.init_qpos is None:
        return runtime.robot.initial_qpos
    target = np.asarray(runtime.init_qpos, dtype=np.float32).copy()
    live = runtime.transport.get_latest_qpos()
    if live is not None:
        for idx in runtime.robot.gripper_indices:
            if idx < len(target) and idx < len(live):
                target[idx] = float(live[idx])
    return target


def reset_with_setup(
    config: ConfigDict,
    runtime: RuntimeState,
    session: SessionState,
    output_target: OutputTarget,
) -> bool:
    """Move the robot to its home pose by streaming a reset trajectory to one target.

    Waits up to 3 s for joint feedback (a real target with no feedback fails; a sim
    target skips the trajectory), then publishes a 100-step linear move to home. For a
    REAL target it polls for motion interrupts between steps.

    Returns:
        True when the reset completed (or was safely skipped), False if interrupted or
        the real robot never reported feedback.

    Raises:
        SystemExit: the transport shut down while waiting for feedback.
    """
    rate = runtime.transport.create_rate(config.inference_cfg.publish_rate)
    logger.info("Resetting to initial joint position on %s target", output_target.value)
    set_setup_stage(runtime, "waiting for joint feedback")
    current_qpos = runtime.transport.get_latest_qpos()
    deadline = time.monotonic() + 3.0
    while current_qpos is None and not runtime.transport.is_shutdown():
        if output_target is OutputTarget.REAL and poll_motion_commands(config, runtime, session):
            consume_motion_interrupt(session, SessionStatus.UNSET)
            return False
        if time.monotonic() > deadline:
            if output_target is OutputTarget.REAL:
                session.last_error = (
                    "Setup failed: no joint feedback from the real robot "
                    "(is it connected and publishing?)"
                )
                logger.warning(session.last_error)
                return False
            logger.warning("No joint feedback received within 3 s; skipping reset trajectory")
            return True
        rate.sleep()
        current_qpos = runtime.transport.get_latest_qpos()
    if current_qpos is None:
        raise SystemExit(0)
    target_qpos = reset_target_qpos(runtime)
    reset_trajectory = build_linear_trajectory(current_qpos, target_qpos, steps=100)
    set_setup_stage(runtime, "publishing reset trajectory")
    logger.info("Publishing %d reset steps to %s", len(reset_trajectory), output_target.value)
    for action in reset_trajectory:
        if output_target is OutputTarget.REAL and poll_motion_commands(config, runtime, session):
            consume_motion_interrupt(session, SessionStatus.UNSET)
            return False
        if output_target is OutputTarget.REAL:
            publish_real_action_with_preview(config, runtime, action)
        else:
            publish_action(runtime, action, target=output_target.value)
        rate.sleep()
    return True


def transition_to_first_frame(
    config: ConfigDict, runtime: RuntimeState, session: SessionState
) -> bool:
    """Smoothly move the real robot from its current qpos to the replay's first frame.

    Real-robot replay starts from home (after reset), but the recorded trajectory's
    first pose is usually elsewhere; jumping straight to it would snap the arm. This
    linearly interpolates over the overlapping dims (qpos feedback width may exceed the
    recorded action width — non-overlapping dims hold their current value) and streams
    the transition to the real robot before frame-by-frame replay begins.

    Returns False if interrupted, True otherwise (including when nothing to do).
    """
    first_qpos = _first_replay_qpos(config, runtime)
    if first_qpos is None:
        first_qpos = getattr(runtime.policy, "first_action", None)
    if first_qpos is None:
        return True
    current_qpos = runtime.transport.get_latest_qpos()
    if current_qpos is None:
        return True
    target = np.asarray(current_qpos, dtype=np.float32).copy()
    overlap = min(target.shape[0], first_qpos.shape[0])
    target[:overlap] = first_qpos[:overlap]
    trajectory = build_linear_trajectory(current_qpos, target, steps=100)
    set_setup_stage(runtime, "moving to trajectory start")
    rate = runtime.transport.create_rate(config.inference_cfg.publish_rate)
    for action in trajectory:
        if poll_motion_commands(config, runtime, session):
            consume_motion_interrupt(session, SessionStatus.UNSET)
            return False
        publish_real_action_with_preview(config, runtime, action)
        rate.sleep()
    return True


def run_setup(
    config: ConfigDict,
    runtime: RuntimeState,
    session: SessionState,
    fail_fast: bool = False,
) -> bool:
    """Prepare the robot for a run: connect the policy, reset to home, prime the chunk.

    fail_fast: when True (console-driven setup), the policy connection is attempted
    once and returns immediately on failure instead of blocking the single-threaded
    main loop for ~30 s of retries — that block also stalls every other queued web
    command (e.g. switching to sim). Eval-driven setup keeps the retry behavior.
    """
    assert session.selected_task is not None
    session.last_error = ""
    has_inflight = (
        session.status is not SessionStatus.UNSET
        or session.action_chunk is not None
        or session.pending_real_chunk is not None
    )
    if has_inflight:
        logger.info("Re-running setup; clearing in-flight state first")
        end_episode(runtime)
        reset_session_progress(session)
        reset_infer_strategy(runtime)

    replay = runtime.replay_source
    if replay is not None:
        set_setup_stage(runtime, "rewinding episode")
        replay.seek(0)
        session.action_chunk = None
        session.chunk_index = 0
        session.step_index = 0
        session.pending_real_chunk = None
        session.sim_preview_qpos = None
        reset_ik_solver(config, runtime)
        if session.mode in (SessionMode.REAL, SessionMode.STEP) and not is_offline_transport(
            runtime
        ):
            set_setup_stage(runtime, "resetting to home position")
            if not reset_with_setup(config, runtime, session, OutputTarget.REAL):
                session.last_error = "Setup interrupted during reset trajectory"
                set_setup_stage(runtime, "")
                return False
            reset_ik_solver(config, runtime)
            if not transition_to_first_frame(config, runtime, session):
                session.last_error = "Setup interrupted while moving to trajectory start"
                set_setup_stage(runtime, "")
                return False
            replay.seek(0)
        set_status(session, SessionStatus.READY, reason="replay setup done")
        session.is_setup_done = True
        set_setup_stage(runtime, "")
        qpos = replay.get_scene_qpos() if hasattr(replay, "get_scene_qpos") else None
        logger.info(
            "[REPLAY_SETUP] episode=%d frame=%d/%d qpos=%s",
            replay.episode_id,
            replay.frame_index,
            replay.n_steps,
            _format_diag_vector(qpos) if qpos is not None else "none",
        )
        return True

    set_setup_stage(
        runtime,
        f"connecting to {config.policy.type} {config.policy.host}:{config.policy.port}",
    )
    connected = connect_policy(
        config,
        runtime,
        session,
        retry_until_connected=not fail_fast,
        max_retries=0 if fail_fast else 30,
    )
    if not connected:
        policy_cfg = config.policy
        detail = runtime.last_policy_error or "connection refused"
        session.last_error = (
            f"Setup failed: cannot reach {policy_cfg.type} policy server at "
            f"{policy_cfg.host}:{policy_cfg.port} ({detail})"
        )
        set_setup_stage(runtime, "")
        return False
    if runtime.policy is None:
        session.last_error = "Setup failed: policy client is unavailable"
        set_setup_stage(runtime, "")
        return False
    logger.info("Running setup for mode %s", session.mode.value)
    if session.mode is SessionMode.SIM and not is_offline_transport(runtime):
        logger.info("SIM mode: skipping reset trajectory (no real robot)")
    else:
        set_setup_stage(runtime, "resetting to home position")
        if not reset_with_setup(config, runtime, session, OutputTarget.REAL):
            session.last_error = "Setup interrupted during reset trajectory"
            set_setup_stage(runtime, "")
            return False
    reset_ik_solver(config, runtime)
    set_setup_stage(runtime, "validating policy (inference)")
    try:
        if runtime.infer_strategy is None:
            fetch_action_chunk(config, runtime, session.selected_task, session)
        else:
            runtime.infer_strategy.take_or_fetch_chunk(
                session.selected_task,
                partial(_loop_fetch_chunk, config, runtime, session),
            )
    except PolicyRequestError as error:
        session.last_error = f"Setup failed: policy request error: {error}"
        invalidate_policy_connection(config, runtime, session, error)
        set_setup_stage(runtime, "")
        return False
    except Exception as error:
        reset_run_state(config, runtime, session)
        session.last_error = f"Setup failed: {error}"
        set_setup_stage(runtime, "")
        logger.info("Action chunk preparation failed: %s", error)
        return False
    logger.info("Setup complete: policy validated")
    # Discard the validation chunk — a fresh warmup runs when execution starts.
    if runtime.infer_strategy is not None:
        runtime.infer_strategy.reset()
    session.action_chunk = None
    session.chunk_index = 0
    session.pending_real_chunk = None
    # Real-robot replay: ease the arm from home onto the recorded trajectory's first
    # frame so the first published action doesn't snap it across joint space.
    if (
        is_replay(runtime)
        and not is_offline_transport(runtime)
        and session.mode is SessionMode.REAL
    ):
        if not transition_to_first_frame(config, runtime, session):
            session.last_error = "Setup interrupted while moving to trajectory start"
            set_setup_stage(runtime, "")
            return False
    set_status(session, SessionStatus.READY, reason="setup complete")
    session.is_setup_done = True
    set_setup_stage(runtime, "")
    return True


def run_reset(config: ConfigDict, runtime: RuntimeState, session: SessionState) -> None:
    """Reset the arm to home without contacting the policy server.

    Closes any open episode, clears session/strategy state, and streams a reset
    trajectory to the real robot (skipped in SIM). Leaves the session ready for a fresh
    setup; does not prefetch a new action chunk.
    """
    set_setup_stage(runtime, "resetting to home position")
    try:
        end_episode(runtime)
        reset_session_progress(session)
        reset_infer_strategy(runtime)
        session.last_error = ""
        logger.info(
            "Running reset for mode %s without contacting the policy server",
            session.mode.value,
        )
        if session.mode is SessionMode.SIM and not is_offline_transport(runtime):
            logger.info("SIM mode: skipping reset trajectory (no real robot)")
        elif not reset_with_setup(config, runtime, session, OutputTarget.REAL):
            logger.warning("Reset interrupted before reaching target joint position")
            return
        reset_ik_solver(config, runtime)
        logger.info("Reset complete; run setup when you want to prefetch a new action chunk")
    finally:
        set_setup_stage(runtime, "")


def _resolve_init_pose(
    config: ConfigDict,
    selected_task: str | None,
    qpos: list[float] | None,
    fallback: np.ndarray | None = None,
) -> np.ndarray | None:
    """Resolve the init start-pose qpos: an explicit operator list, else the config one.

    Args:
        selected_task: the active task, used to look up its bound init_pose.
        qpos: operator-supplied joint values from the editable INIT box; None/empty
            falls back to the selected task's init_pose, then the robot's
            initial_qpos (``fallback``).
        fallback: robot home qpos used when nothing is configured.

    Returns:
        [n] float32 qpos, or None when nothing is configured. The length must cover at
        least the arm joints; gripper dims are overwritten on reset from the live qpos.
    """
    values = qpos if qpos else None
    if values is None:
        eval_cfg = config.eval
        task_pose = None
        if eval_cfg is not None and selected_task is not None:
            for t in eval_cfg.tasks:
                if t.prompt_en == selected_task:
                    task_pose = t.get("init_pose")
                    break
        values = task_pose
    if not values:
        return None if fallback is None else np.asarray(fallback, dtype=np.float32)
    return np.asarray(values, dtype=np.float32)


def run_init_move(
    config: ConfigDict,
    runtime: RuntimeState,
    session: SessionState,
    qpos: list[float] | None = None,
) -> None:
    """EVAL INIT panel: move the arm to the configured start-pose qpos.

    Sets runtime.init_qpos (which retargets every later reset) and streams the arm
    there now. Gripper dims of the start pose are ignored — the gripper holds its live
    value and is driven by the panel's OPEN/CLOSE buttons.

    Args:
        qpos: operator-supplied joint values from the editable INIT box. Empty falls
            back to the selected task's configured init_pose, then the robot home.
    """
    target = _resolve_init_pose(
        config, session.selected_task, qpos, fallback=runtime.robot.initial_qpos
    )
    if target is None:
        session.last_error = "No start-pose qpos given"
        return
    expected = runtime.robot.total_action_dim
    if target.shape[0] != expected:
        session.last_error = f"Init pose has {target.shape[0]} values, expected {expected}"
        logger.warning(session.last_error)
        return
    runtime.init_qpos = target
    runtime.init_ready = False
    set_setup_stage(runtime, "moving to init pose")
    try:
        if session.mode is SessionMode.SIM and not is_offline_transport(runtime):
            logger.info("SIM mode: skipping init move (no real robot)")
        elif not reset_with_setup(config, runtime, session, OutputTarget.REAL):
            session.last_error = "Init move interrupted before reaching the start pose"
            logger.warning(session.last_error)
            return
        session.last_error = ""
        logger.info("Init move complete; adjust the gripper, then click DONE")
    finally:
        set_setup_stage(runtime, "")


def run_init_done(runtime: RuntimeState, session: SessionState) -> None:
    """Latch initialization complete so RUN can proceed from the start pose."""
    if runtime.init_qpos is None:
        session.last_error = "Move to the init pose before finishing initialization"
        return
    runtime.init_ready = True
    session.last_error = ""
    logger.info("Initialization complete; ready to RUN from the start pose")


def publish_next_action(config: ConfigDict, runtime: RuntimeState, session: SessionState) -> bool:
    """Publish one step of the active run, advancing the session counters.

    Dispatches by context: replay sources play the next recorded frame/chunk; a running
    async/rtc strategy pops one buffered action; otherwise the sync chunk flow ensures a
    chunk and publishes the next action (waiting for the robot to reach it in sync mode).
    SIM mode publishes to sim and updates the preview qpos; other modes drive the real
    robot with a sim mirror.

    Returns:
        True when an action was published, False when nothing was published this tick
        (buffer empty, replay ended, or an interrupt/limit hit).
    """
    if is_replay(runtime):
        if session.mode is SessionMode.STEP:
            return _publish_replay_step_chunk(config, runtime, session)
        return _publish_replay_frame(config, runtime, session)

    strategy = runtime.infer_strategy

    sim_only = session.mode is SessionMode.SIM

    # Continuous mode: pop individual actions from the buffer
    if strategy is not None and strategy.is_loop_running():
        action = strategy.pop_next_action()
        if action is None:
            return False  # buffer empty, skip this tick
        action = apply_gripper_control_overrides(runtime, action, session)
        logger.info(
            "[PUBLISH_ACTION] mode=%s strategy=loop step=%d action=%s",
            session.mode.value,
            session.step_index,
            _format_diag_vector(action),
        )
        if sim_only:
            publish_action(runtime, action, target=OutputTarget.SIM.value)
            session.sim_preview_qpos = action.copy()
            record_executed_action(config, session, action, runtime)
        else:
            publish_real_action_with_preview(config, runtime, action, session=session)
        session.step_index += 1
        return True

    # Fallback: chunk-based flow (sync mode or no strategy)
    if not ensure_action_chunk(config, runtime, session) or session.action_chunk is None:
        return False
    action = apply_gripper_control_overrides(
        runtime, session.action_chunk[session.chunk_index], session
    )
    logger.info(
        "[PUBLISH_ACTION] mode=%s strategy=chunk step=%d chunk=%d/%d action=%s",
        session.mode.value,
        session.step_index,
        session.chunk_index + 1,
        len(session.action_chunk),
        _format_diag_vector(action),
    )
    if sim_only:
        publish_action(runtime, action, target=OutputTarget.SIM.value)
        session.sim_preview_qpos = action.copy()
        record_executed_action(config, session, action, runtime)
    else:
        publish_real_action_with_preview(config, runtime, action, session=session)

    is_sync = strategy is not None and not strategy.runs_background_loop
    if is_sync and strategy is not None and not is_offline_transport(runtime):
        if not _wait_until_reached(
            runtime,
            action,
            config.inference_cfg.publish_rate,
            config=config,
            session=session,
            threshold=strategy.sync_wait_threshold,
            max_ticks=strategy.sync_wait_max_ticks,
            ignore_gripper_error=strategy.sync_wait_ignore_gripper,
        ):
            consume_motion_interrupt(session, SessionStatus.READY)
            return False

    session.chunk_index += 1
    session.step_index += 1
    return True


def _wait_until_reached(
    runtime: RuntimeState,
    target_action: np.ndarray,
    publish_rate: int,
    config: ConfigDict | None = None,
    session: SessionState | None = None,
    threshold: float = 0.05,
    max_ticks: int = 100,
    ignore_gripper_error: bool = False,
) -> bool:
    """Poll qpos until robot reaches target or timeout.  Used by sync mode.

    Returns ``True`` if the target was reached (or timed out normally),
    ``False`` if an interrupt was detected via the command queue.
    """
    rate = runtime.transport.create_rate(publish_rate)
    step_index = session.step_index if session is not None else -1
    chunk_index = session.chunk_index if session is not None else -1
    logger.info(
        "[SYNC_WAIT_BEGIN] step=%d chunk=%d target=%s threshold=%.4f max_ticks=%d "
        "ignore_gripper_error=%s",
        step_index,
        chunk_index,
        _format_diag_vector(target_action),
        threshold,
        max_ticks,
        ignore_gripper_error,
    )
    last_actual = None
    last_error = None
    last_arm_error = None
    last_gripper_error = None
    for tick in range(max_ticks):
        if config is not None and session is not None:
            if poll_motion_commands(config, runtime, session):
                logger.info("[SYNC_WAIT_END] reason=interrupt step=%d tick=%d", step_index, tick)
                return False
        actual_qpos = runtime.transport.get_latest_qpos()
        if actual_qpos is None:
            if tick > 0 and tick % 20 == 0:
                logger.info(
                    "[SYNC_WAIT_PROGRESS] step=%d tick=%d actual_qpos=None",
                    step_index,
                    tick,
                )
            rate.sleep()
            continue
        overlap = min(len(target_action), len(actual_qpos))
        diff = target_action[:overlap] - actual_qpos[:overlap]
        gripper_mask = np.asarray(runtime.robot.gripper_mask[:overlap], dtype=bool)
        if len(gripper_mask) != overlap:
            gripper_mask = np.zeros(overlap, dtype=bool)
        arm_diff = diff[~gripper_mask]
        gripper_diff = diff[gripper_mask]
        arm_error = float(np.linalg.norm(arm_diff)) if len(arm_diff) else 0.0
        gripper_error = float(np.linalg.norm(gripper_diff)) if len(gripper_diff) else 0.0
        error = arm_error if ignore_gripper_error else float(np.linalg.norm(diff))
        last_actual = actual_qpos
        last_error = error
        last_arm_error = arm_error
        last_gripper_error = gripper_error
        if error < threshold:
            logger.info(
                "[SYNC_WAIT_END] reason=reached step=%d tick=%d error=%.4f "
                "arm_error=%.4f gripper_error=%.4f actual=%s",
                step_index,
                tick,
                error,
                arm_error,
                gripper_error,
                _format_diag_vector(actual_qpos),
            )
            return True
        if tick > 0 and tick % 20 == 0:
            logger.info(
                "[SYNC_WAIT_PROGRESS] step=%d tick=%d error=%.4f arm_error=%.4f "
                "gripper_error=%.4f actual=%s",
                step_index,
                tick,
                error,
                arm_error,
                gripper_error,
                _format_diag_vector(actual_qpos),
            )
        rate.sleep()
    logger.info(
        "[SYNC_WAIT_END] reason=timeout step=%d chunk=%d error=%s arm_error=%s "
        "gripper_error=%s actual=%s",
        step_index,
        chunk_index,
        f"{last_error:.4f}" if last_error is not None else "none",
        f"{last_arm_error:.4f}" if last_arm_error is not None else "none",
        f"{last_gripper_error:.4f}" if last_gripper_error is not None else "none",
        _format_diag_vector(last_actual) if last_actual is not None else "none",
    )
    return True


def run_one_chunk_on_sim(config: ConfigDict, runtime: RuntimeState, session: SessionState) -> bool:
    """STEP mode: infer one chunk, animate it on sim, and stage it for real publish.

    Plays the remaining chunk on the sim target so the 3D arm moves immediately, keeps
    it as pending_real_chunk so REAL can replay the identical chunk, and leaves the
    session READY.

    Returns:
        True when a chunk was animated, False if no chunk could be prepared.
    """
    if not ensure_action_chunk(config, runtime, session) or session.action_chunk is None:
        return False
    # Animate the inferred chunk on the sim target so the 3D arm moves immediately,
    # and keep it as pending so REAL can replay the same chunk on the robot.
    remaining_chunk = session.action_chunk[session.chunk_index :].copy()
    rate = runtime.transport.create_rate(config.inference_cfg.publish_rate)
    for action in remaining_chunk:
        session.sim_preview_qpos = action.copy()
        publish_action(runtime, action, target=OutputTarget.SIM.value)
        rate.sleep()
    session.pending_real_chunk = remaining_chunk
    session.chunk_index = len(session.action_chunk)
    session.status = SessionStatus.READY
    logger.info("Animated %d actions in sim; press REAL to run on the robot", len(remaining_chunk))
    return True


def run_pending_chunk_on_real(
    config: ConfigDict, runtime: RuntimeState, session: SessionState
) -> bool:
    """STEP mode: publish the staged sim chunk to the real robot.

    Replays pending_real_chunk on REAL, then clears it and returns the session to READY.
    Keeps sim_preview_qpos frozen on the sim playback result so REAL only drives the
    robot without altering the visualization.

    Returns:
        True when the chunk was published, False if none was staged or it was interrupted.
    """
    if session.pending_real_chunk is None:
        logger.info("No sim chunk is waiting for real publish")
        return False
    step_index = session.step_index
    if not publish_action_chunk(
        config, runtime, session.pending_real_chunk, OutputTarget.REAL, session=session
    ):
        return False
    logger.info("Published %d actions to real", len(session.pending_real_chunk))
    session.pending_real_chunk = None
    if is_replay(runtime):
        session.step_index = step_index
    # Keep sim_preview_qpos so the 3D stays frozen on the SIM playback result:
    # REAL only drives the robot and must not alter the visualization.
    session.status = SessionStatus.READY
    return True


def update_inference_params(
    config: ConfigDict, runtime: RuntimeState, session: SessionState, params: dict
) -> None:
    """Hot-patch inference timing + strategy args onto the live runtime config.

    Writes a new frozen ConfigDict onto runtime.active_config (the main loop reads it in
    preference to the base config) so inference_rate / publish_rate and the per-strategy
    args take effect on the next inference/publish without a restart. publish_rate is
    re-read by the main loop each iteration; inference_rate + strategy args need the
    active strategy rebuilt, which this does when one is selected.

    Args:
        params: {inference_rate?, publish_rate?, strategies?: {key: {execute_horizon?,
            latency_k?}}} — any subset; absent fields keep current values.
    """
    if params.get("replay_fps") is not None:
        runtime.replay_fps = max(1, int(params["replay_fps"]))
    if params.get("replay_exec_steps") is not None:
        runtime.replay_exec_steps = max(1, int(params["replay_exec_steps"]))

    strategies = dict(config.inference_strategies)
    for key, args in (params.get("strategies") or {}).items():
        if key not in strategies:
            continue
        spec = strategies[key]
        merged = dict(spec.get("args") or {})
        merged.update({k: v for k, v in args.items() if v is not None})
        strategies[key] = StrategyYamlSpec(type=spec["type"], args=cast(StrategyYamlArgs, merged))

    import copy

    new_config = copy.deepcopy(config)
    new_config.inference_cfg.inference_rate = float(
        params.get("inference_rate", config.inference_cfg.inference_rate)
    )
    new_config.inference_cfg.publish_rate = int(
        params.get("publish_rate", config.inference_cfg.publish_rate)
    )
    new_config.inference_strategies = strategies
    runtime.active_config = new_config

    key = runtime.selected_inference_strategy_key
    if key is not None and key in strategies:
        was_running = session.status is SessionStatus.RUNNING
        replace_inference_strategy(new_config, runtime, key)
        if was_running:
            start_inference_loop(new_config, runtime, session)
    logger.info(
        "Updated inference params: inference_rate=%.2f publish_rate=%d replay_fps=%d"
        " replay_exec_steps=%d",
        new_config.inference_cfg.inference_rate,
        new_config.inference_cfg.publish_rate,
        runtime.replay_fps,
        runtime.replay_exec_steps,
    )


def replace_inference_strategy(
    config: ConfigDict, runtime: RuntimeState, strategy_key: str
) -> None:
    """Tear down the current strategy and build the one named by ``strategy_key``.

    Builds the strategy from its config spec at the configured inference_rate.
    """
    if runtime.infer_strategy is not None:
        runtime.infer_strategy.close()
    runtime.selected_inference_strategy_key = strategy_key
    spec = config.inference_strategies[strategy_key]
    args = dict(spec["args"])
    cls = STRATEGY_REGISTRY.get(spec["type"])
    # inference_rate is a global setting; inject only for classes that declare it
    # (background-loop strategies). Sync strategies don't have this field.
    if "inference_rate" in {f.name for f in dataclasses.fields(cls)}:  # pyright: ignore[reportArgumentType]
        args.setdefault("inference_rate", config.inference_cfg.inference_rate)
    runtime.infer_strategy = STRATEGY_REGISTRY.build(spec["type"], **args)


def select_inference_strategy(
    strategy_key: str,
    config: ConfigDict,
    runtime: RuntimeState,
    session: SessionState,
) -> None:
    """Select the inference strategy and reset session progress + IK to apply it.

    No-op when the requested strategy is already active.
    """
    if runtime.selected_inference_strategy_key == strategy_key:
        logger.info(
            "Inference strategy is already %s",
            resolve_inference_strategy_label(config, strategy_key),
        )
        return
    replace_inference_strategy(config, runtime, strategy_key)
    reset_session_progress(session)
    reset_ik_solver(config, runtime)
    label = resolve_inference_strategy_label(config, strategy_key)
    logger.info("Selected inference strategy %s; status is now unset", label)


def select_mode(
    mode: SessionMode, config: ConfigDict, session: SessionState, runtime: RuntimeState
) -> None:
    """Switch the operating mode and reset session progress + strategy.

    Entering MANUAL also seeds the command qpos from the robot's live observed state
    (falling back to home when no feedback has arrived) so the staged target starts
    aligned with the arm.
    """
    # Leaving COLLECT disarms teleop: stop any open recording episode, then close the
    # collection stream so the arm is no longer driven by the human after the switch.
    if mode is not SessionMode.COLLECT and runtime.collection_teleop_active:
        if session.mode is SessionMode.COLLECT and session.status is SessionStatus.RUNNING:
            collect_stop(config, runtime, session)
        collect_stop_teleop(config, runtime, session)
    session.mode = mode
    reset_session_progress(session)
    reset_infer_strategy(runtime)
    if mode is SessionMode.MANUAL:
        # Seed the command qpos from the robot's live observed state so the staged
        # target starts aligned with where the arm actually is. On a real transport
        # get_latest_qpos() returns sensor feedback; fall back to the home pose when
        # no feedback has arrived yet.
        current = runtime.transport.get_latest_qpos()
        seed = current if current is not None else runtime.robot.initial_qpos
        session.manual_qpos = np.asarray(seed, dtype=np.float32).copy()
        session.manual_real_qpos = np.asarray(seed, dtype=np.float32).copy()
    logger.info("Selected mode %s; status is now unset", mode.value)


def select_task(
    task: str, config: ConfigDict, session: SessionState, runtime: RuntimeState
) -> None:
    """Set the active DEBUG/run task and reset session progress + strategy + IK.

    No-op when the task is unchanged.
    """
    if session.selected_task == task:
        logger.info("Task is already %s", format_task_label(task))
        return
    session.selected_task = task
    reset_run_state(config, runtime, session)
    logger.info("Selected task %s; status is now unset", format_task_label(session.selected_task))


__all__ = [
    "_MOTION_INTERRUPTING_VERBS",
    "MANUAL_MAX_QPOS_STEP",
    "apply_gripper_control_overrides",
    "iter_target_grippers",
    "apply_gripper_command",
    "set_gripper_open_close",
    "toggle_gripper_immediate",
    "set_gripper_immediate",
    "publish_action",
    "_publish_manual_real_qpos",
    "set_manual_qpos",
    "manual_send",
    "manual_home",
    "publish_real_action_with_preview",
    "publish_action_chunk",
    "unlock_prompt",
    "handle_motion_command",
    "poll_motion_commands",
    "consume_motion_interrupt",
    "reset_target_qpos",
    "reset_with_setup",
    "transition_to_first_frame",
    "run_setup",
    "run_reset",
    "_resolve_init_pose",
    "run_init_move",
    "run_init_done",
    "publish_next_action",
    "_wait_until_reached",
    "run_one_chunk_on_sim",
    "run_pending_chunk_on_real",
    "update_inference_params",
    "replace_inference_strategy",
    "select_inference_strategy",
    "select_mode",
    "select_task",
]
