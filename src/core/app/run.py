"""Main application loop — web command parsing, dispatch, and the top-level run()
entry that wires transport, robot, and policy together. Commands arrive over HTTP
(console / eval servers) via the command_queue; there is no interactive terminal UI.
"""

from __future__ import annotations

import json
import logging
import queue
import threading
import time

import numpy as np

from core.app.cli import maybe_start_inference_cli
from core.app.console.server import build_console_context, start_console_server
from core.app.control_channel import maybe_start_control_channel
from core.app.handlers import (
    _anchor_buffer_to_current_qpos,
    accept_rollout_intervention_segment,
    begin_rollout_save_episode,
    clear_replay,
    collect_cancel,
    collect_start,
    collect_start_teleop,
    collect_step,
    collect_stop,
    collect_stop_teleop,
    connect_policy,
    discard_rollout_intervention_segment,
    disconnect_policy,
    enable_default_recording,
    end_episode,
    is_replay,
    iter_target_grippers,
    load_replay_dataset,
    manual_home,
    manual_send,
    mark_rollout_save_ready,
    maybe_build_episode_logger,
    publish_next_action,
    rebuild_eval_episode_logger,
    record_rollout_intervention_step,
    reset_ik_solver,
    reset_infer_strategy,
    reset_session_progress,
    rollback_rollout_intervention,
    run_init_done,
    run_init_move,
    run_one_chunk_on_sim,
    run_pending_chunk_on_real,
    run_reset,
    run_setup,
    run_warmup_and_start,
    save_rollout_episode,
    select_episode,
    select_inference_strategy,
    select_mode,
    select_task,
    set_gripper_immediate,
    set_gripper_open_close,
    set_manual_qpos,
    set_replay_fps,
    start_episode,
    start_inference_loop,
    start_rollout_intervention,
    stop_rollout_intervention,
    toggle_gripper_immediate,
    update_inference_params,
)
from core.app.operator_control import (
    maybe_start_operator_button_listener,
    resolve_operator_event,
)
from core.app.state import (
    RuntimeState,
    SessionMode,
    SessionState,
    SessionStatus,
    set_phase,
    set_status,
)
from core.config import ConfigDict
from core.registry import FUNCTIONS
from core.utils.paths import resolve_output_dir
from core.utils.ssh_forward import free_local_ports, stop_ssh_forward
from robots import build_robot
from transport.base import build_transport

logger = logging.getLogger(__name__)

# Halt an active run if the frontend hasn't polled /api/status within this window.
# The poll cadence is ~250ms, so this tolerates dozens of missed polls before tripping.
CLIENT_WATCHDOG_TIMEOUT_S = 3.0
_HIL_CONTROL_MODES = frozenset({"absolute", "relative"})

# Headless heartbeat cadence. While RUNNING we log every beat (proof of life); while
# idle we log far less often so a parked headless process doesn't spam the log.
_HEARTBEAT_INTERVAL_S = 5.0
_HEARTBEAT_IDLE_INTERVAL_S = 30.0


def _maybe_log_heartbeat(
    runtime: RuntimeState,
    session: SessionState,
    last_heartbeat: float,
) -> float:
    """Emit a throttled one-line status heartbeat; return the (possibly new) timestamp.

    Running: every _HEARTBEAT_INTERVAL_S. Idle: every _HEARTBEAT_IDLE_INTERVAL_S. This
    is the headless "still alive" signal that the browser's status poll used to give.
    """
    running = session.status is SessionStatus.RUNNING
    interval = _HEARTBEAT_INTERVAL_S if running else _HEARTBEAT_IDLE_INTERVAL_S
    now = time.monotonic()
    if now - last_heartbeat < interval:
        return last_heartbeat
    logger.info(
        "[HB] mode=%s status=%s phase=%s step=%d chunk=%d infer=%.1fms policy=%s error=%s",
        session.mode.value,
        session.status.value,
        runtime.web_phase,
        session.step_index,
        session.chunk_index,
        session.last_infer_ms,
        "ok" if runtime.policy is not None else "none",
        session.last_error or "-",
    )
    return now


def _target_loop_rate_hz(
    config: ConfigDict,
    runtime: RuntimeState,
) -> float:
    if runtime.replay_source is not None:
        return runtime.replay_fps
    return config.inference_cfg.publish_rate


# --- gripper lock commands ---


def _handle_gripper_lock_command(
    parts: list[str],
    config: ConfigDict,
    runtime: RuntimeState,
    session: SessionState,
) -> bool:
    gripper_groups = [g for g in runtime.robot.actuator_groups if g.gripper_index is not None]
    side = parts[0].lower()
    if side in {"lg", "left", "l"}:
        group_index = 0
    elif side in {"rg", "right", "r"}:
        group_index = 1
    else:
        return False
    if group_index >= len(gripper_groups):
        logger.info("Robot has no %s gripper group", "left" if group_index == 0 else "right")
        return True
    group_name = gripper_groups[group_index].name
    if len(parts) >= 2:
        try:
            value = float(parts[1])
            session.gripper_locks[group_name] = value
            logger.info("Gripper '%s' locked to %.3f", group_name, value)
            return True
        except ValueError:
            logger.error("Invalid gripper value: '%s'", parts[1])
            return True
    if group_name in session.gripper_locks:
        del session.gripper_locks[group_name]
        logger.info("Gripper '%s' unlocked", group_name)
    else:
        current_qpos = runtime.transport.get_latest_qpos()
        if current_qpos is None:
            logger.info("No joint feedback available; cannot toggle gripper")
            return True
        for group, index, _ in iter_target_grippers(runtime.robot, None):
            if group.name == group_name:
                cur = float(current_qpos[index])
                thresh = (
                    config.robot.gripper_threshold
                    if config.robot.gripper_threshold is not None
                    else 0.5
                )
                value = config.robot.gripper_close if cur >= thresh else config.robot.gripper_open
                session.gripper_locks[group_name] = value
                logger.info("Gripper '%s' locked to %.3f", group_name, value)
                break
    return True


# --- web-mode command dispatch ---


def _activate_ckpt_slot(
    slot: int,
    eval_cfg: ConfigDict,
    runtime: RuntimeState,
    session: SessionState,
) -> ConfigDict | None:
    """Make ckpt ``slot`` the active model, doing the FULL activation every time.

    Shared by bootstrap (initial Model A) and switch_ckpt so a directly-launched model is
    initialized identically to a switched-to one — adopt the ckpt's config, drop the old
    policy/session/strategy/IK state, force-reconnect to the ckpt's endpoint, re-select
    mode + strategy, repoint recording at the per-model dataset, and arm a pre-start reset.

    The ckpt brings its own real policy endpoint (host/port), so web eval dials the actual
    server; the eval section is grafted on (ckpt configs are plain robot configs with
    eval=None) while prompts/scenes come from eval.

    Returns the adopted active config on success, or None if the policy connect failed
    (caller sets web_phase=idle). Does NOT touch web_phase otherwise.
    """
    ckpt = eval_cfg.checkpoints[runtime.ckpt_order[slot]]
    import copy
    config = copy.deepcopy(ckpt.config)
    config.eval = eval_cfg
    runtime.active_config = config
    runtime.active_ckpt_slot = slot
    runtime.policy = None
    runtime.policy_metadata = None
    runtime.last_policy_error = ""
    reset_session_progress(session)
    reset_infer_strategy(runtime)
    reset_ik_solver(config, runtime)
    # Select mode + strategy BEFORE connecting. select_inference_strategy is a no-op when
    # the key is already selected, so clear it first to FORCE a real rebuild every time —
    # otherwise the very first activation (bootstrap) would skip building the strategy
    # object (key already defaulted to "async"), leaving the model connected but unable to
    # drive the run loop until a later switch rebuilt it. This is what made Model A require
    # a B->A round-trip. connect_policy then reads the freshly-built strategy for RTC/latency.
    select_mode(SessionMode(eval_cfg.cli_mode), config, session, runtime)
    runtime.selected_inference_strategy_key = None
    select_inference_strategy(eval_cfg.inference_strategy, config, runtime, session)
    if not connect_policy(
        config,
        runtime,
        session,
        force_reconnect=True,
        retry_until_connected=True,
        max_retries=30,
    ):
        return None
    rebuild_eval_episode_logger(config, runtime)
    runtime.needs_pre_start_reset = True
    return config


def _should_start_eval_ssh(eval_cfg: ConfigDict | None) -> bool:
    if not eval_cfg or not eval_cfg.get("checkpoints") or not eval_cfg.get("enable_ssh_forward"):
        return False
    ssh = eval_cfg.get("ssh") or {}
    return bool(ssh.get("host") and ssh.get("user") and ssh.get("port", 0) > 0)


def _should_start_deploy_ssh(config: ConfigDict) -> bool:
    if config.eval:
        return False
    ssh = config.transport.get("ssh") or {}
    return bool(ssh.get("host") and ssh.get("user") and ssh.get("port", 0) > 0)


def _format_inference_strategy_choices(config: ConfigDict) -> str:
    return "  ".join(f"{i + 1}.{k}" for i, k in enumerate(config.inference_strategies))


def _select_only_inference_strategy_if_needed(
    config: ConfigDict,
    runtime: RuntimeState,
    session: SessionState,
) -> bool:
    if is_replay(runtime) or runtime.selected_inference_strategy_key is not None:
        return True
    if len(config.inference_strategies) != 1:
        return False
    select_inference_strategy(
        next(iter(config.inference_strategies)),
        config,
        runtime,
        session,
    )
    return True


def _handle_web_command(
    command: str,
    config: ConfigDict,
    runtime: RuntimeState,
    session: SessionState,
) -> None:
    """Programmatic command channel for the web server.

    Format: `web:<verb>[:<arg>]`. Verbs: bootstrap, switch_task, start, stop, reset.
    """
    payload = command[len("web:") :]
    verb, _, arg = payload.partition(":")
    verb = verb.strip()
    arg = arg.strip()
    logger.info(
        "[CMD] %s%s (phase=%s status=%s)",
        verb,
        f" {arg}" if arg else "",
        runtime.web_phase,
        session.status.value,
    )

    if verb == "bootstrap":
        # One-time setup at server startup: connect policy + select REAL + select strategy.
        # The actual run_setup() (with reset trajectory) happens lazily on first start.
        if not config.eval:
            logger.warning("web:bootstrap called but not config.eval — ignoring")
            return
        eval_cfg = config.eval
        # Blind multi-ckpt: start on slot 0 (Model A). Run the SAME full activation a
        # switch_ckpt does, so a directly-launched Model A is initialized identically to a
        # switched-to model (force-reconnect + reset session/strategy/IK + rebuild logger +
        # arm pre-start reset). ckpt_order is set by the web server before bootstrap fires.
        if eval_cfg.checkpoints and runtime.ckpt_order:
            adopted = _activate_ckpt_slot(0, eval_cfg, runtime, session)
            if adopted is None:
                logger.error("Bootstrap: policy connect failed")
                set_phase(runtime, "idle")
                return
            config = adopted
        else:
            # No checkpoint list (single anonymous model): connect the base config directly.
            if not connect_policy(
                config, runtime, session, retry_until_connected=True, max_retries=30
            ):
                logger.error("Bootstrap: policy connect failed")
                set_phase(runtime, "idle")
                return
            select_mode(SessionMode(eval_cfg.cli_mode), config, session, runtime)
            select_inference_strategy(eval_cfg.inference_strategy, config, runtime, session)
            rebuild_eval_episode_logger(config, runtime)
        set_phase(runtime, "ready")
        logger.info(
            "Bootstrap: mode=%s strategy=%s policy=connected",
            eval_cfg.cli_mode,
            eval_cfg.inference_strategy,
        )
        return

    if verb == "warmup":
        # Fully prepare the robot: run setup (policy validate + reset to home + prime
        # the IK/strategy). After this the robot is ready and the next start is instant.
        if runtime.web_phase != "ready":
            logger.warning("Ignored warmup: phase=%s", runtime.web_phase)
            return
        if session.selected_task is None:
            logger.warning("Ignored warmup: no task selected")
            return
        set_phase(runtime, "starting")
        run_setup(config, runtime, session)
        runtime.needs_pre_start_reset = False
        set_phase(runtime, "ready")
        logger.info("Warmup complete: robot ready")
        return

    if verb == "switch_task":
        # Allowed only when no run is in flight. Frontend should already gate this.
        if runtime.web_phase in {"starting", "running", "stopping", "resetting"}:
            logger.warning("Ignored switch_task while phase=%s", runtime.web_phase)
            return
        stop_rollout_intervention(config, runtime, session, required=False)
        discard_rollout_intervention_segment(runtime)
        select_task(arg, config, session, runtime)
        return

    if verb == "start":
        if runtime.web_phase != "ready":
            logger.warning("Ignored start: phase=%s (must be ready)", runtime.web_phase)
            return
        # Bind this trial's task NOW (from the cell stamped by /api/eval_start), before
        # the no-task gate and before the reset trajectory goes out. Otherwise eval_start
        # can carry a prompt but still be rejected because no prompt was selected in the
        # DEBUG task picker.
        cell = runtime.current_cell or {}
        cell_prompt = cell.get("prompt")
        if isinstance(cell_prompt, str) and session.selected_task != cell_prompt:
            select_task(cell_prompt, config, session, runtime)
        if session.selected_task is None:
            logger.warning("Ignored start: no task selected")
            return
        logger.info("[START] task=%r", session.selected_task)
        # The separate web:switch_task the
        # frontend queued when the operator clicked the trial could still be sitting in the
        # queue and would interrupt the in-flight reset trajectory (switch_task is a
        # motion-interrupting verb) — cutting the init short. Selecting it here makes that
        # queued switch_task a no-op (task unchanged) so the trajectory completes.
        # run_setup() must FULLY complete its init before we run: clear any stale interrupt
        # flag (a leftover from a prior aborted motion would otherwise cut the reset
        # trajectory short), then run setup and only start if it reported success. This
        # guarantees the reset-to-initial-position trajectory was sent in full and the
        # policy was validated — no skipped steps, no phantom "running" without motion.
        # Skip it when a prior SETUP (web:warmup) already homed + validated and nothing
        # since left the arm un-reset (needs_pre_start_reset) or cleared setup (a prompt
        # switch resets is_setup_done) — so RUN after SETUP starts instantly, not re-homing.
        session.interrupt_requested = False
        set_phase(runtime, "starting")
        if runtime.needs_pre_start_reset or not session.is_setup_done:
            if not run_setup(config, runtime, session):
                logger.warning(
                    "[START] setup did not complete (%s); staying ready", session.last_error
                )
                set_phase(runtime, "ready")
                return
            runtime.needs_pre_start_reset = False
        _dispatch_run(config, runtime, session)
        # If start succeeded, SessionStatus.RUNNING; otherwise revert to ready.
        if session.status is SessionStatus.RUNNING:
            set_phase(runtime, "running")
            # Stamp eval trial identity onto the episode so the recorded dataset carries
            # prompt/trial + clip_id; scores join back to the result row via clip_id.
            if runtime.episode_logger is not None and runtime.current_clip_id is not None:
                cell = runtime.current_cell or {}
                runtime.episode_logger.set_episode_meta(
                    prompt=cell.get("prompt"),
                    trial=cell.get("trial"),
                    clip_id=runtime.current_clip_id,
                )
        else:
            set_phase(runtime, "ready")
        return

    if verb == "stop":
        if runtime.web_phase in {"resetting", "ready"}:
            logger.info("Ignored duplicate stop: phase=%s", runtime.web_phase)
            return
        if runtime.web_phase not in {"running", "stopping"}:
            logger.warning("Ignored stop: phase=%s (must be running)", runtime.web_phase)
            return
        if runtime.web_phase != "stopping":
            set_phase(runtime, "stopping")
        _dispatch_halt(config, runtime, session)
        # Close the recording NOW so the episodes.jsonl row (index/prompt/trial/clip_id +
        # timestamp) lands on disk immediately and the just-stopped trial is scorable
        # without waiting for the next RUN. Parquet/video flush in the background.
        end_episode(runtime)
        # Stop always leaves the arm un-reset. Either auto-reset now, or mark a deferred
        # reset that the next start will pick up.
        assert runtime.command_queue is not None
        if config.eval and config.eval.reset_after_each_trial:
            set_phase(runtime, "resetting")
            runtime.command_queue.put("web:reset")
        else:
            runtime.needs_pre_start_reset = True
            set_phase(runtime, "ready")
        return

    if verb == "reset":
        # Manual reset: allowed whenever the arm isn't actively moving under policy control.
        if runtime.web_phase in {"starting", "running", "stopping"}:
            logger.warning("Ignored reset: phase=%s", runtime.web_phase)
            return
        set_phase(runtime, "resetting")
        run_reset(config, runtime, session)
        runtime.needs_pre_start_reset = False
        set_phase(runtime, "ready")
        return

    if verb == "init_move":
        # EVAL INIT panel: move the arm to the start pose. arg is a JSON body carrying an
        # optional operator-supplied qpos list (else the configured default is used).
        # Allowed only when the arm isn't moving under policy control.
        if runtime.web_phase in {"starting", "running", "stopping", "resetting"}:
            logger.warning("Ignored init_move while phase=%s", runtime.web_phase)
            return
        params = json.loads(arg) if arg else {}
        raw_qpos = params.get("qpos")
        qpos = [float(x) for x in raw_qpos] if raw_qpos else None
        set_phase(runtime, "resetting")
        run_init_move(config, runtime, session, qpos=qpos)
        runtime.needs_pre_start_reset = False
        set_phase(runtime, "ready")
        return

    if verb == "init_gripper":
        # EVAL INIT panel OPEN/CLOSE: drive a gripper to a known state without a lock.
        # arg form "<open|close>[:<l|r>]"; no side -> both grippers.
        if runtime.web_phase in {"starting", "running", "stopping", "resetting"}:
            logger.warning("Ignored init_gripper while phase=%s", runtime.web_phase)
            return
        action, _, side = arg.partition(":")
        set_gripper_open_close(config, runtime, open_gripper=(action == "open"), side=side or None)
        return

    if verb == "init_done":
        run_init_done(runtime, session)
        return

    if verb == "switch_ckpt":
        # Blind multi-ckpt: operator picks a slot (0->Model A...). Runs the SAME full
        # activation bootstrap uses for the initial model.
        if runtime.web_phase in {"starting", "running", "stopping", "resetting"}:
            logger.warning("Ignored switch_ckpt while phase=%s", runtime.web_phase)
            return
        if not config.eval or not config.eval.checkpoints:
            logger.warning("Ignored switch_ckpt: no checkpoints configured")
            return
        eval_cfg = config.eval
        slot = int(arg)
        if slot < 0 or slot >= len(runtime.ckpt_order):
            logger.warning("Ignored switch_ckpt: slot %d out of range", slot)
            return
        stop_rollout_intervention(config, runtime, session, required=False)
        discard_rollout_intervention_segment(runtime)
        if _activate_ckpt_slot(slot, eval_cfg, runtime, session) is None:
            logger.error("switch_ckpt: connect failed for slot %d", slot)
            set_phase(runtime, "idle")
            return
        set_phase(runtime, "ready")
        logger.info("Switched to ckpt slot %d (label Model %s)", slot, chr(ord("A") + slot))
        return

    # --- console verbs (eval-independent; drive the debug mainline directly) ---

    if verb == "operator_action":
        intent, _, source = arg.partition(":")
        resolved_command = resolve_operator_event(
            config,
            runtime,
            session,
            intent=intent,
            source=source or "ui",
        )
        if resolved_command is not None:
            handle_command(resolved_command, config, runtime, session)
        return

    if verb == "operator_button":
        button, _, source = arg.partition(":")
        resolved_command = resolve_operator_event(
            config,
            runtime,
            session,
            button=button,
            source=source or "cat",
        )
        if resolved_command is not None:
            handle_command(resolved_command, config, runtime, session)
        return

    if verb == "connect":
        connect_policy(config, runtime, session)
        return

    if verb == "disconnect":
        disconnect_policy(config, runtime, session)
        return

    if verb == "select_mode":
        try:
            mode = SessionMode(arg)
        except ValueError:
            logger.warning("console select_mode: unknown mode %r", arg)
        else:
            select_mode(mode, config, session, runtime)
        return

    if verb == "select_episode":
        select_episode(int(arg), config, session, runtime)
        return

    if verb == "load_replay_dataset":
        params = json.loads(arg) if arg else {}
        keys = params.get("keys", {})
        load_replay_dataset(
            str(params.get("dataset_dir", "")).strip(),
            int(params.get("episode", 0)),
            config,
            session,
            runtime,
            state_key=keys.get("state", ""),
            action_key=keys.get("action", ""),
            video_keys=params.get("video_keys") or {},
            action_mode=params.get("action_mode", ""),
        )
        return

    if verb == "clear_replay":
        clear_replay(config, session, runtime)
        return

    if verb == "set_replay_fps":
        set_replay_fps(runtime, int(arg))
        return

    if verb == "replay_seek":
        replay = runtime.replay_source
        if replay is not None and hasattr(replay, "seek"):
            total = int(getattr(replay, "n_steps", 0) or 0)
            frame = max(0, min(int(arg), total - 1)) if total > 0 else 0
            replay.seek(frame)  # type: ignore[attr-defined]
            session.step_index = frame
            session.sim_preview_qpos = None
            set_status(session, SessionStatus.READY, reason="replay seek")
        return

    if verb == "select_strategy":
        if arg in config.inference_strategies:
            select_inference_strategy(arg, config, runtime, session)
        else:
            logger.warning("console select_strategy: unknown strategy %r", arg)
        return

    if verb == "update_infer_params":
        params = json.loads(arg) if arg else {}
        update_inference_params(config, runtime, session, params)
        return

    if verb == "setup":
        if session.selected_task is None or session.mode is SessionMode.SELECT:
            logger.warning("console setup: select task and mode first")
            return
        if not _select_only_inference_strategy_if_needed(config, runtime, session):
            session.last_error = (
                f"Select inference strategy first: {_format_inference_strategy_choices(config)}"
            )
            logger.warning("console setup: select inference strategy first")
            return
        run_setup(config, runtime, session, fail_fast=True)
        return

    if verb == "run":
        runtime.collection_replay_qpos = None
        runtime.collection_replay_episode = None
        if getattr(runtime, "rollout_intervention_active", False):
            segment = getattr(runtime, "rollout_intervention_active_segment", None)
            if segment is not None and segment.invalid_reason:
                session.last_error = segment.invalid_reason
                return
            if not stop_rollout_intervention(config, runtime, session, required=True):
                return
            if not accept_rollout_intervention_segment(runtime, session):
                return
            runtime.rollout_save_ready = False
            runtime.rollout_save_reason = ""
            session.action_chunk = None
            session.chunk_index = 0
            if not run_warmup_and_start(config, runtime, session):
                return
            session.run_start_time = time.monotonic()
            set_status(session, SessionStatus.RUNNING, reason="resume after teleop intervention")
            return
        if (
            session.mode in (SessionMode.REAL, SessionMode.SIM)
            and session.status is SessionStatus.READY
            and session.is_setup_done
            and session.step_index > 0
        ):
            replay = runtime.replay_source
            if replay is not None and hasattr(replay, "seek"):
                total = int(getattr(replay, "n_steps", 0) or 0)
                if total > 0 and session.step_index >= total:
                    session.step_index = 0
                replay.seek(session.step_index)  # type: ignore[attr-defined]
            else:
                # Async/policy continue: re-anchor the action buffer to the robot's
                # current measured pose and restart the inference loop, so resume
                # blends from where the robot really is instead of dumping the stale
                # actions queued before the pause (which caused a recoil).
                _anchor_buffer_to_current_qpos(runtime)
                start_inference_loop(config, runtime, session)
            session.last_error = ""
            runtime.rollout_save_ready = False
            runtime.rollout_save_reason = ""
            session.run_start_time = time.monotonic()
            set_status(session, SessionStatus.RUNNING, reason="continue")
            return
        # Continuous run for real/sim — reuse the 's' dispatch (sets RUNNING; the main
        # loop then publishes). For STEP 's' instead previews one chunk.
        _dispatch_run(config, runtime, session)
        return

    if verb == "halt":
        # Stop continuous run / cancel a pending sim chunk — reuse the 'c' dispatch.
        was_running = session.status is SessionStatus.RUNNING and session.mode in (
            SessionMode.REAL,
            SessionMode.SIM,
        )
        _dispatch_halt(config, runtime, session)
        if was_running and not is_replay(runtime):
            mark_rollout_save_ready(runtime, "stop")
            if session.mode is SessionMode.REAL and config.rollout.intervention.enabled:
                start_rollout_intervention(config, runtime, session)
        return

    if verb == "rollout_intervention_abandon":
        if not getattr(runtime, "rollout_intervention_active", False):
            session.last_error = "No active rollout intervention to abandon"
            return
        if not stop_rollout_intervention(config, runtime, session, required=True):
            return
        if not rollback_rollout_intervention(config, runtime, session):
            discard_rollout_intervention_segment(runtime)
            return
        discard_rollout_intervention_segment(runtime)
        runtime.rollout_save_ready = False
        runtime.rollout_save_reason = ""
        session.action_chunk = None
        session.chunk_index = 0
        if not run_warmup_and_start(config, runtime, session):
            return
        session.run_start_time = time.monotonic()
        set_status(session, SessionStatus.RUNNING, reason="resume after abandoned intervention")
        return

    if verb == "rollout_intervention_mode":
        if arg not in _HIL_CONTROL_MODES:
            allowed = ", ".join(sorted(_HIL_CONTROL_MODES))
            session.last_error = f"Unsupported HIL control mode {arg!r}; expected: {allowed}"
            return
        if runtime.rollout_intervention_active:
            session.last_error = "Stop or resume the active intervention before changing HIL mode"
            return
        setter = getattr(runtime.transport, "set_hil_control_mode", None)
        if setter is None:
            session.last_error = "Transport does not support HIL control mode"
            return
        setter(arg)
        runtime.hil_control_mode = arg
        session.last_error = ""
        return

    if verb == "console_reset":
        stop_rollout_intervention(config, runtime, session, required=False)
        discard_rollout_intervention_segment(runtime)
        if not is_replay(runtime) and session.mode in (SessionMode.REAL, SessionMode.SIM):
            mark_rollout_save_ready(runtime, "reset")
        run_reset(config, runtime, session)
        return

    if verb == "rollout_save":
        if getattr(runtime, "rollout_intervention_active", False):
            session.last_error = "Continue or abandon the active intervention before saving"
            return
        save_rollout_episode(runtime, session)
        return

    if verb == "rollout_stop":
        was_running = session.status is SessionStatus.RUNNING and session.mode in (
            SessionMode.REAL,
            SessionMode.SIM,
        )
        _dispatch_halt(config, runtime, session)
        if was_running and not is_replay(runtime):
            mark_rollout_save_ready(runtime, "stop")
        return

    if verb == "tab_switch":
        # Switching tabs is a soft reset: stop continuous publishing and clear the
        # session/runtime state machine so modes never bleed across tabs. Entering
        # COLLECT is the exception: it arms teleop immediately, while recording stays
        # controlled by START RECORD.
        # A REPLAY/QC dataset stays mounted on runtime.replay_source until explicitly
        # cleared; leaving it mounted keeps is_replay() true on the next tab, so EVAL/DEBUG
        # would publish the recorded trajectory instead of querying the policy. Unmount it
        # whenever the destination isn't the REPLAY tab (clear_replay also reconnects the
        # policy that load_replay_dataset dropped).
        if arg != "replay" and runtime.replay_source is not None:
            clear_replay(config, session, runtime)
        if arg != "collect":
            runtime.collection_teleop_armed = False
        runtime.collection_replay_qpos = None
        runtime.collection_replay_episode = None
        stop_rollout_intervention(config, runtime, session, required=False)
        discard_rollout_intervention_segment(runtime)
        if session.mode is SessionMode.COLLECT and session.status is SessionStatus.RUNNING:
            collect_stop(config, runtime, session)
        if runtime.collection_teleop_active:
            collect_stop_teleop(config, runtime, session)
        _dispatch_halt(config, runtime, session)
        end_episode(runtime)
        reset_session_progress(session)
        reset_infer_strategy(runtime)
        reset_ik_solver(config, runtime)
        session.last_error = ""
        set_status(session, SessionStatus.UNSET, reason="tab switch")
        if arg == "collect" and getattr(runtime, "collection_teleop_armed", False):
            collect_start_teleop(config, runtime, session)
        elif arg == "eval" and config.eval is not None:
            # EVAL runs on the mode fixed by the config (cli_mode, normally REAL); the
            # generic SELECT reset below would force the operator to re-pick a mode that
            # the eval flow doesn't even expose, so START would fail with "select mode first".
            session.mode = SessionMode(config.eval.cli_mode)
        else:
            session.mode = SessionMode.SELECT
        return

    if verb == "step_infer":
        if not session.is_setup_done:
            logger.warning("console step_infer: run setup first")
            return
        if is_replay(runtime) and session.mode is SessionMode.STEP:
            replay = runtime.replay_source
            logger.info(
                "[REPLAY_STEP_REQUEST] mode=%s status=%s step=%d replay_frame=%s",
                session.mode.value,
                session.status.value,
                session.step_index,
                None if replay is None else getattr(replay, "frame_index", None),
            )
            advanced = publish_next_action(config, runtime, session)
            logger.info(
                "[REPLAY_STEP_DONE] advanced=%s step=%d replay_frame=%s",
                advanced,
                session.step_index,
                None if replay is None else getattr(replay, "frame_index", None),
            )
            return
        # Breakpoint single-step: infer one chunk and animate it on the sim arm now,
        # keeping it as pending_real_chunk so REAL replays the same chunk.
        run_one_chunk_on_sim(config, runtime, session)
        return

    if verb == "step_commit":
        run_pending_chunk_on_real(config, runtime, session)
        return

    if verb == "step_cancel":
        if session.pending_real_chunk is not None:
            session.pending_real_chunk = None
            session.sim_preview_qpos = None
            set_status(session, SessionStatus.READY, reason="step cancelled")
        return

    # --- manual mode verbs ---

    if verb == "manual_qpos":
        values = [float(x) for x in arg.split(",") if x != ""]
        set_manual_qpos(config, runtime, session, np.asarray(values, dtype=np.float32))
        return

    if verb == "manual_dispatch":
        session.manual_dispatch = "stage"
        logger.info("Manual dispatch is now %s", session.manual_dispatch)
        return

    if verb == "manual_send":
        manual_send(config, runtime, session)
        return

    if verb == "manual_home":
        manual_home(config, runtime, session)
        return

    # --- data collection (teleoperation) verbs ---

    if verb == "collect_start":
        collect_start(config, runtime, session)
        return

    if verb == "collect_stop":
        collect_stop(config, runtime, session)
        return

    if verb == "collect_cancel":
        collect_cancel(runtime, session)
        return

    if verb == "gripper":
        # arg forms: "<side>:<open|close>:<0|1>" drives that side to the explicit state
        # immediately (lock=1 keeps overriding the policy during RUN). Legacy forms
        # "<side>:<value>" lock to a number and bare "<side>" toggles.
        parts = arg.split(":")
        side = parts[0]
        if len(parts) >= 2 and parts[1] in {"open", "close"}:
            lock = len(parts) < 3 or parts[2] != "0"
            set_gripper_immediate(
                config,
                runtime,
                session,
                open_gripper=(parts[1] == "open"),
                side=side or None,
                lock=lock,
            )
        elif len(parts) >= 2:
            _handle_gripper_lock_command([side, parts[1]], config, runtime, session)
        else:
            toggle_gripper_immediate(config, runtime, session, side=side or None)
        return

    logger.warning("Unknown web command: %s", command)


# --- command dispatch ---


def _dispatch_run(config: ConfigDict, runtime: RuntimeState, session: SessionState) -> None:
    """Mode-specific run/step: start continuous publishing for REAL/SIM, or preview/
    commit a chunk for STEP. Shared by the web run/start verbs and the queue's ``s`` verb.
    """
    if session.mode is SessionMode.SELECT:
        session.last_error = "Select mode first: 1.real  2.sim  3.sim_real"
        return
    if not _select_only_inference_strategy_if_needed(config, runtime, session):
        session.last_error = (
            f"Select inference strategy first: {_format_inference_strategy_choices(config)}"
        )
        return
    if not session.is_setup_done:
        session.last_error = "Setup is required before running the next step"
        return
    if session.mode in (SessionMode.REAL, SessionMode.SIM):
        if session.status is SessionStatus.UNSET:
            session.last_error = "Setup is required before entering running state"
            return
        if session.status is SessionStatus.RUNNING:
            session.last_error = f"Mode {session.mode.value} is already running"
            return
        if session.status is not SessionStatus.READY:
            session.last_error = f"Unexpected status {session.status.value}; run setup first"
            return
        if not run_warmup_and_start(config, runtime, session):
            return
        if not is_replay(runtime):
            start_episode(runtime, session)
            begin_rollout_save_episode(config, runtime, session)
        session.run_start_time = time.monotonic()
        set_status(session, SessionStatus.RUNNING, reason="start")
        return
    if session.pending_real_chunk is None:
        run_one_chunk_on_sim(config, runtime, session)
        return
    run_pending_chunk_on_real(config, runtime, session)


def _dispatch_halt(config: ConfigDict, runtime: RuntimeState, session: SessionState) -> None:
    """Stop continuous REAL/SIM publishing or cancel a pending STEP chunk. Shared by the
    web halt/stop verbs, the client watchdog, and the queue's ``c`` verb.
    """
    if (
        session.mode in (SessionMode.REAL, SessionMode.SIM)
        and session.status is SessionStatus.RUNNING
    ):
        set_status(session, SessionStatus.READY, reason="continuous publish cancelled")
    elif session.mode is SessionMode.STEP and session.pending_real_chunk is not None:
        session.pending_real_chunk = None
        set_status(session, SessionStatus.READY, reason="sim chunk cancelled")


def handle_command(
    command: str,
    config: ConfigDict,
    runtime: RuntimeState,
    session: SessionState,
) -> None:
    """Dispatch one queued command. Web verbs (``web:`` prefix) route to the web
    handler; the bare run/halt verbs ``s``/``c`` drive the mode-specific execution.
    """
    if command.startswith("web:"):
        _handle_web_command(command, config, runtime, session)
        return
    normalized = command.strip().lower()
    if normalized in {"s", "step"}:
        _dispatch_run(config, runtime, session)
        return
    if normalized in {"c", "cancel", "stop"}:
        _dispatch_halt(config, runtime, session)
        return
    logger.warning("Ignored non-web command %r; CLI input has been removed", normalized)


# --- main application loop ---


def run(
    config: ConfigDict,
    web_port: int,
    config_path: str | None = None,
    headless: bool = False,
) -> None:
    """Main entry. Runs the unified console web server.

    The console hosts every workflow as a tab (DEBUG/REPLAY/COLLECT/MANUAL/EVAL/RESULT).
    When ``config.eval`` is set, the EVAL/RESULT tabs activate and — if it carries
    checkpoints — blind multi-ckpt SSH forwarding + result sync + bootstrap kick in.

    In ``headless`` mode no web server is started: the HTTP console, 3D scene, camera
    streams and episode preview are all skipped. Control comes solely over the ZMQ
    control channel (forced on), and status is surfaced via structured logs + a
    periodic heartbeat — for low-power / simulator-driven evaluation.
    """
    robot = build_robot(config)
    if config.robot.initial_qpos is not None:
        override = np.asarray(config.robot.initial_qpos, dtype=np.float32)
        if override.shape != robot.initial_qpos.shape:
            raise ValueError(
                f"robot.initial_qpos override has {override.shape[0]} dims, "
                f"expected {robot.initial_qpos.shape[0]} for robot '{config.robot.type}'"
            )
        robot.initial_qpos = override

    logger.info(
        "Creating transport: type=%s robot=%s",
        config.transport.type,
        config.robot.type,
    )
    transport = build_transport(config, robot)
    logger.info(
        "Transport ready: type=%s obs_mode=%s",
        config.transport.type,
        config.inference_cfg.obs_space.type,
    )
    if transport.has_sim_publishers():
        logger.info("Sim joint target available")

    session = SessionState()
    runtime = RuntimeState(
        robot=robot,
        transport=transport,
        selected_inference_strategy_key=None,
        infer_strategy=None,
    )
    runtime.hil_control_mode = str(
        getattr(
            transport,
            "hil_control_mode",
            config.rollout.intervention.get("control_mode", "absolute"),
        )
    )
    _select_only_inference_strategy_if_needed(config, runtime, session)
    # Record every trial as an episode so results can be previewed (URDF/camera/chart),
    # rooting the dataset inside the eval results tree. Dataset replay has nothing to
    # record (it plays back an existing dataset), so it is left untouched.
    output_dir = str(resolve_output_dir(config, config_path))
    runtime.eval_output_dir = output_dir
    if config.transport.type != "dataset":
        config = enable_default_recording(config, output_dir)
    maybe_build_episode_logger(config, runtime)

    # Auto-connect a mock policy at startup so setup/step work offline. An eval config
    # with checkpoints instead adopts the real ckpt policy during web:bootstrap below.
    has_eval_ckpts = config.eval and bool(config.eval.checkpoints)
    if config.policy.type == "mock" and not has_eval_ckpts:
        connect_policy(config, runtime, session)

    command_queue: queue.Queue[str] = queue.Queue()
    prompt_ready = threading.Event()
    prompt_ready.set()
    runtime.command_queue = command_queue
    runtime.prompt_ready = prompt_ready
    maybe_start_operator_button_listener(config, runtime)
    loop_rate = transport.create_rate(config.inference_cfg.publish_rate)
    loop_rate_hz = config.inference_cfg.publish_rate

    ssh_proc = None
    result_syncer = None
    should_start_eval_ssh = _should_start_eval_ssh(config.eval)
    # SSH forwarding is opt-in; local checkpoint endpoints must not be killed as stale tunnels.
    if should_start_eval_ssh:
        eval_cfg = config.eval
        assert eval_cfg is not None
        ports = sorted({c.port for c in eval_cfg.checkpoints})
        free_local_ports([web_port, *ports])
        ssh_proc = FUNCTIONS.build("ssh_forward", eval_cfg.ssh, ports)
        if eval_cfg.ssh.get("remote_sync_dir"):
            local_root = resolve_output_dir(config, config_path).parent
            result_syncer = FUNCTIONS.build("result_sync", eval_cfg.ssh, local_root)
    elif _should_start_deploy_ssh(config):
        # Deploy (non-eval): forward the policy port to the remote inference server.
        free_local_ports([web_port, config.policy.port])
        ssh_proc = FUNCTIONS.build("ssh_forward", config.transport.ssh, [config.policy.port])
    else:
        free_local_ports([web_port])

    if headless:
        # Low-power path: no HTTP server, no 3D scene / camera streams / preview.
        # Build a light ConsoleContext so the ZMQ control-channel queries still work,
        # and force the control channel on (it is the ONLY control entry with no web).
        build_console_context(
            config,
            runtime,
            session,
            output_dir=output_dir,
            with_scene=False,
            with_obs_reader=False,
            with_preview=False,
        )
        channel_cfg = config.get("control_channel") or {}
        if not channel_cfg.get("enabled", False):
            # Headless with no channel would be uncontrollable — force it on.
            config.control_channel = ConfigDict(
                {**dict(channel_cfg), "enabled": True}
            )
        maybe_start_control_channel(config, runtime)
        ch = config.control_channel
        disp_host = "127.0.0.1" if ch.get("host") == "0.0.0.0" else ch.get("host", "127.0.0.1")
        logger.info(
            "HEADLESS mode: web disabled; control entry = ZMQ tcp://%s:%d",
            disp_host,
            int(ch.get("port", 5757)),
        )
        # Second entry: an in-process interactive CLI when stdin is a terminal. Both it
        # and the ZMQ channel feed the same command_queue, so a human at the terminal and
        # a simulator over ZMQ drive the same session. Skipped when backgrounded/piped.
        # Select REAL synchronously here (before the CLI banner) so its [CMD]/[STATUS]
        # logs land now, not asynchronously on top of the freshly-printed "eva> " prompt.
        select_mode(SessionMode.REAL, config, session, runtime)
        maybe_start_inference_cli(config, runtime, session)
    else:
        start_console_server(config, runtime, session, port=web_port, output_dir=output_dir)
        logger.info("Console web server started on port %d", web_port)
        # Optional ZMQ control channel — must start AFTER the console server so it can
        # share runtime.console_ctx. No-op unless control_channel.enabled.
        maybe_start_control_channel(config, runtime)
    # An eval config bootstraps the eval state machine (connect ckpt policy, select
    # mode/strategy); plain console configs stay idle until the operator drives a tab.
    if config.eval:
        command_queue.put("web:bootstrap")

    # Headless heartbeat: no browser polls /api/status, so a periodic one-line status
    # log is the "still alive" signal. State transitions already log via set_status /
    # set_phase; this fills the gaps during a long RUNNING stretch.
    last_heartbeat = 0.0

    try:
        while session.status is not SessionStatus.EXIT and not transport.is_shutdown():
            # Multi-ckpt: a switch_ckpt swaps in the active ckpt's full config.
            effective = runtime.active_config or config
            target_hz = _target_loop_rate_hz(effective, runtime)
            if target_hz != loop_rate_hz:
                loop_rate = transport.create_rate(target_hz)
                loop_rate_hz = target_hz
            while True:
                try:
                    cmd = command_queue.get_nowait()
                except queue.Empty:
                    break
                handle_command(cmd, effective, runtime, session)
                effective = runtime.active_config or config
                prompt_ready.set()

            if (
                session.mode in (SessionMode.REAL, SessionMode.SIM)
                and session.status is SessionStatus.RUNNING
            ):
                # Client-liveness watchdog: the frontend polls /api/status every ~250ms,
                # so a stale timestamp means the operator's browser crashed or the network
                # dropped. Halt the run rather than keep commanding the real robot blind.
                stale = time.monotonic() - runtime.last_client_poll
                if runtime.last_client_poll > 0.0 and stale > CLIENT_WATCHDOG_TIMEOUT_S:
                    logger.warning("[WATCHDOG] no client poll for %.1fs; halting active run", stale)
                    _dispatch_halt(effective, runtime, session)
                else:
                    publish_next_action(effective, runtime, session)

            if getattr(runtime, "rollout_intervention_active", False):
                record_rollout_intervention_step(runtime, session)

            if session.mode is SessionMode.COLLECT and session.status is SessionStatus.RUNNING:
                collect_step(effective, runtime)

            # Reconcile web_phase with the real session state. A run can leave RUNNING via
            # paths that DON'T go through the eval web:stop verb — the client-liveness
            # watchdog above or an inference error — and those only reset
            # session.status, leaving web_phase latched at "running". A latched phase then
            # rejects every later start/switch_ckpt/switch_task ("Ignored ... while
            # phase=running"), which manifests as "can't start a trial until I fiddle with
            # the model switch". If the session isn't actually running anymore but the phase
            # still says it is, drop it back to ready.
            if runtime.web_phase == "running" and session.status is not SessionStatus.RUNNING:
                set_phase(runtime, "ready")
                runtime.needs_pre_start_reset = True

            if headless:
                last_heartbeat = _maybe_log_heartbeat(runtime, session, last_heartbeat)

            loop_rate.sleep()
    finally:
        stop_rollout_intervention(config, runtime, session, required=False)
        discard_rollout_intervention_segment(runtime)
        end_episode(runtime)
        if runtime.episode_logger is not None:
            runtime.episode_logger.finalize()
        if runtime.rollout_episode_logger is not None:
            if runtime.rollout_episode_logger.has_active_episode:
                runtime.rollout_episode_logger.cancel_episode("shutdown")
            runtime.rollout_episode_logger.finalize()
        if runtime.infer_strategy is not None:
            runtime.infer_strategy.close()
        transport.close()
        if ssh_proc is not None:
            stop_ssh_forward(ssh_proc)
        if result_syncer is not None:
            result_syncer.stop()
