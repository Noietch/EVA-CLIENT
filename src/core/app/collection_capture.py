"""Background capture loop for teleop collection episodes."""

from __future__ import annotations

import gc
import logging
import threading
import time
from typing import Any

logger = logging.getLogger(__name__)

_gc_collect_thread: threading.Thread | None = None


def _collect_cycles() -> None:
    started = time.monotonic()
    collected = gc.collect()
    logger.info(
        "[CAPTURE_GC] state=collected objects=%d duration_ms=%.1f",
        collected,
        (time.monotonic() - started) * 1000.0,
    )


def _suspend_cyclic_gc() -> None:
    global _gc_collect_thread
    if _gc_collect_thread is not None:
        _gc_collect_thread.join()
        _gc_collect_thread = None
    gc.disable()
    logger.info("[CAPTURE_GC] state=suspended")


def _resume_cyclic_gc() -> None:
    global _gc_collect_thread
    gc.enable()
    _gc_collect_thread = threading.Thread(
        target=_collect_cycles,
        name="eva-capture-gc",
        daemon=True,
    )
    _gc_collect_thread.start()
    logger.info("[CAPTURE_GC] state=resumed")


class CollectionCaptureRunner:
    """Move collection raw snapshot capture off the control/state-machine loop."""

    def __init__(
        self,
        runtime: Any,
        fps: float,
        max_raw_snapshots_per_tick: int,
    ) -> None:
        if fps <= 0:
            raise ValueError(f"collection capture fps must be positive, got {fps}")
        if max_raw_snapshots_per_tick <= 0:
            raise ValueError("collection capture max_raw_snapshots_per_tick must be positive")
        self._runtime = runtime
        self._interval_s = 1.0 / float(fps)
        self._max_raw_snapshots_per_tick = max_raw_snapshots_per_tick
        self._stop_event = threading.Event()
        self._thread = threading.Thread(
            target=self._run,
            name="eva-collection-capture",
            daemon=True,
        )
        self._error: BaseException | None = None
        self._captured = 0
        self._capture_ticks = 0
        self._deadline_misses = 0
        self._max_tick_ms = 0.0
        self._previous_source_time: float | None = None
        self._previous_receive_time: float | None = None
        self._max_source_gap = 0.0
        self._max_receive_gap = 0.0
        self._source_gap_count = 0

    def start(self) -> None:
        """Start the background capture thread."""
        self._thread.start()

    def stop(self) -> None:
        """Stop the capture thread and surface any background failure."""
        self._stop_event.set()
        if self._thread.is_alive():
            self._thread.join()
        self._drain_available_raw(self._max_raw_snapshots_per_tick)
        logger.info(
            "[RAW_CAPTURE] stopped snapshots=%d ticks=%d deadline_misses=%d max_tick_ms=%.1f",
            self._captured,
            self._capture_ticks,
            self._deadline_misses,
            self._max_tick_ms,
        )
        logger_obj = self._runtime.episode_logger
        rollout = getattr(self._runtime, "rollout_episode_logger", None)
        if rollout is not None and getattr(rollout, "has_active_episode", False):
            logger_obj = rollout
        set_meta = getattr(logger_obj, "set_episode_meta", None)
        if callable(set_meta):
            set_meta(
                capture_max_source_gap_sec=round(self._max_source_gap, 6),
                capture_max_receive_gap_sec=round(self._max_receive_gap, 6),
                capture_source_gap_count=self._source_gap_count,
            )
        if self._error is not None:
            raise RuntimeError("collection capture runner failed") from self._error

    @property
    def is_running(self) -> bool:
        """True while the capture thread is alive."""
        return self._thread.is_alive()

    def _run(self) -> None:
        try:
            deadline = time.monotonic()
            while not self._stop_event.is_set():
                deadline += self._interval_s
                started = time.monotonic()
                self._capture_tick()
                finished = time.monotonic()
                tick_ms = (finished - started) * 1000.0
                self._capture_ticks += 1
                self._max_tick_ms = max(self._max_tick_ms, tick_ms)
                remaining = deadline - finished
                if remaining <= 0.0:
                    self._deadline_misses += 1
                    if self._deadline_misses == 1 or self._deadline_misses % 100 == 0:
                        logger.warning(
                            "[RAW_CAPTURE_TIMING] deadline_misses=%d tick_ms=%.1f max_tick_ms=%.1f",
                            self._deadline_misses,
                            tick_ms,
                            self._max_tick_ms,
                        )
                    deadline = finished
                    continue
                self._stop_event.wait(remaining)
        except BaseException as exc:
            self._error = exc
            logger.exception("Collection capture runner failed")

    def _track_capture_timing(self, timestamp: float) -> None:
        received = time.monotonic()
        if self._previous_source_time is not None and self._previous_receive_time is not None:
            source_gap = timestamp - self._previous_source_time
            receive_gap = received - self._previous_receive_time
            self._max_source_gap = max(self._max_source_gap, source_gap)
            self._max_receive_gap = max(self._max_receive_gap, receive_gap)
            if source_gap > max(0.1, 3 * self._interval_s):
                self._source_gap_count += 1
                logger.warning(
                    "[CAPTURE_GAP] source_gap_sec=%.6f receive_gap_sec=%.6f "
                    "previous_source=%.6f source=%.6f",
                    source_gap,
                    receive_gap,
                    self._previous_source_time,
                    timestamp,
                )
        self._previous_source_time = timestamp
        self._previous_receive_time = received

    def _capture_tick(self) -> bool:
        episode_logger = self._runtime.episode_logger
        rollout_logger = getattr(self._runtime, "rollout_episode_logger", None)
        collection_mode = bool(
            episode_logger is not None
            and getattr(episode_logger, "is_collection_enabled", False)
            and getattr(episode_logger, "has_active_episode", False)
        )
        rollout_mode = bool(
            rollout_logger is not None and getattr(rollout_logger, "has_active_episode", False)
        )
        if not collection_mode and not rollout_mode:
            return False
        recorded = False
        for _ in range(self._max_raw_snapshots_per_tick):
            if self._stop_event.is_set():
                break
            snapshot = self._runtime.transport.acquire_collection_raw()
            if snapshot is None:
                break
            self._runtime.last_collection_timestamp = float(snapshot.timestamp)
            self._track_capture_timing(float(snapshot.timestamp))
            if rollout_mode:
                self._runtime.rollout_raw_snapshots.put(snapshot)
            else:
                assert episode_logger is not None
                episode_logger.ingest_collection_snapshot(snapshot)
            self._captured += 1
            recorded = True
        return recorded

    def _drain_available_raw(self, max_snapshots: int) -> bool:
        episode_logger = self._runtime.episode_logger
        rollout_logger = getattr(self._runtime, "rollout_episode_logger", None)
        collection_mode = bool(
            episode_logger is not None
            and getattr(episode_logger, "is_collection_enabled", False)
            and getattr(episode_logger, "has_active_episode", False)
        )
        rollout_mode = bool(
            rollout_logger is not None and getattr(rollout_logger, "has_active_episode", False)
        )
        if not collection_mode and not rollout_mode:
            return False
        recorded = False
        for _ in range(max_snapshots):
            snapshot = self._runtime.transport.acquire_collection_raw()
            if snapshot is None:
                return recorded
            self._runtime.last_collection_timestamp = float(snapshot.timestamp)
            self._track_capture_timing(float(snapshot.timestamp))
            if rollout_mode:
                self._runtime.rollout_raw_snapshots.put(snapshot)
            else:
                assert episode_logger is not None
                episode_logger.ingest_collection_snapshot(snapshot)
            self._captured += 1
            recorded = True
        return recorded


def start_collection_capture(
    runtime: Any,
    fps: float,
    max_raw_snapshots_per_tick: int,
) -> None:
    """Attach and start one collection capture runner on the runtime."""
    if getattr(runtime, "collection_capture_runner", None) is not None:
        raise RuntimeError("collection capture runner is already active")
    rollout_logger = getattr(runtime, "rollout_episode_logger", None)
    logger_obj = (
        rollout_logger
        if rollout_logger is not None and rollout_logger.has_active_episode
        else runtime.episode_logger
    )
    prepare_collection_capture(runtime, logger_obj)
    _suspend_cyclic_gc()
    try:
        runner = CollectionCaptureRunner(runtime, fps, max_raw_snapshots_per_tick)
        runtime.collection_capture_runner = runner
        runner.start()
    except BaseException:
        runtime.collection_capture_runner = None
        _resume_cyclic_gc()
        raise


def stop_collection_capture(runtime: Any) -> None:
    """Stop and detach the active collection capture runner if present."""
    runner = getattr(runtime, "collection_capture_runner", None)
    if runner is None:
        _finish_transport_capture(runtime)
        return
    try:
        runner.stop()
    finally:
        try:
            _finish_transport_capture(runtime)
        finally:
            runtime.collection_capture_runner = None
            _resume_cyclic_gc()


def prepare_collection_capture(runtime: Any, logger_obj: Any) -> None:
    """Point transport scratch storage at the active logger's filesystem."""
    prepare = getattr(runtime.transport, "prepare_collection_capture", None)
    if prepare is not None:
        directory = getattr(logger_obj, "_log_dir", None)
        prepare(None if directory is None else str(directory))


def _finish_transport_capture(runtime: Any) -> None:
    finish = getattr(runtime.transport, "finish_collection_capture", None)
    if finish is not None:
        finish()
