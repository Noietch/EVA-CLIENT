"""Background capture loop for teleop collection episodes."""

from __future__ import annotations

import logging
import threading
from typing import Any

logger = logging.getLogger(__name__)


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

    def start(self) -> None:
        """Start the background capture thread."""
        self._thread.start()

    def stop(self) -> None:
        """Stop the capture thread and surface any background failure."""
        self._stop_event.set()
        if self._thread.is_alive():
            self._thread.join()
        self._drain_available_raw(self._max_raw_snapshots_per_tick)
        logger.info("[RAW_CAPTURE] stopped snapshots=%d", self._captured)
        if self._error is not None:
            raise RuntimeError("collection capture runner failed") from self._error

    @property
    def is_running(self) -> bool:
        """True while the capture thread is alive."""
        return self._thread.is_alive()

    def _run(self) -> None:
        try:
            while not self._stop_event.is_set():
                self._capture_tick()
                self._stop_event.wait(self._interval_s)
        except BaseException as exc:
            self._error = exc
            logger.exception("Collection capture runner failed")

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
            if rollout_mode:
                self._runtime.rollout_raw_snapshots.put(snapshot)
                self._captured += 1
            else:
                assert episode_logger is not None
                episode_logger.ingest_collection_snapshot(snapshot)
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
            if rollout_mode:
                self._runtime.rollout_raw_snapshots.put(snapshot)
                self._captured += 1
            else:
                assert episode_logger is not None
                episode_logger.ingest_collection_snapshot(snapshot)
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
    runner = CollectionCaptureRunner(runtime, fps, max_raw_snapshots_per_tick)
    runtime.collection_capture_runner = runner
    runner.start()


def stop_collection_capture(runtime: Any) -> None:
    """Stop and detach the active collection capture runner if present."""
    runner = getattr(runtime, "collection_capture_runner", None)
    if runner is None:
        return
    try:
        runner.stop()
    finally:
        runtime.collection_capture_runner = None
