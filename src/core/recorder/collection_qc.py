"""Serial background QC for newly committed collection episodes."""

from __future__ import annotations

import logging
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path

from core.utils.dataset_lock import dataset_lock
from core.utils.lerobot import LeRobotDatasetIO
from robots.base import Robot

logger = logging.getLogger(__name__)


class CollectionQcWorker:
    """Check saved takes without blocking capture or applying an obsolete result."""

    def __init__(self, robot: Robot) -> None:
        self.robot = robot
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="collection_qc")
        self._futures: list[Future[None]] = []
        self._lock = threading.Lock()

    def submit(self, dataset_dir: Path, episode_index: int) -> None:
        """Called under the save transaction's dataset lock, after it commits."""
        stat = LeRobotDatasetIO(dataset_dir).episode_parquet(episode_index).stat()
        revision = (stat.st_ino, stat.st_mtime_ns)
        with self._lock:
            self._futures = [
                future
                for future in self._futures
                if not future.done() or future.exception() is not None
            ]
            self._futures.append(
                self._executor.submit(self._check, dataset_dir, episode_index, revision)
            )

    def _check(self, dataset_dir: Path, episode_index: int, revision: tuple[int, int]) -> None:
        # Import after recorder initialization; the dataset checks also load the
        # robot catalog used by the review application.
        from tools.datasets.auto_qc import EpisodeAutoQc, episode_seconds, short_episode_fence

        io = LeRobotDatasetIO(dataset_dir)
        path = io.episode_parquet(episode_index)
        with dataset_lock(dataset_dir).acquire():
            stat = path.stat()
            if (stat.st_ino, stat.st_mtime_ns) != revision:
                return
            rows = io._episode_rows_by_index()
            row = rows[episode_index]
            checker = EpisodeAutoQc(dataset_dir, self.robot)
            fence = short_episode_fence(
                [episode_seconds(item, checker.fps) for item in rows.values()]
            )

        try:
            issues = checker.check(row, fence)
        except (OSError, ValueError, RuntimeError) as error:
            with dataset_lock(dataset_dir).acquire():
                stat = path.stat()
                if (stat.st_ino, stat.st_mtime_ns) != revision:
                    return
                io.mark_qc(episode_index, "unreviewed", str(error), "auto_qc_error", auto=True)
            logger.exception("Automatic QC failed for %s episode %d", dataset_dir, episode_index)
            return

        with dataset_lock(dataset_dir).acquire():
            stat = path.stat()
            if (stat.st_ino, stat.st_mtime_ns) != revision:
                return
            checker.record(row, issues)
        logger.info(
            "[COLLECTION_QC] dataset=%s episode=%d findings=%d",
            dataset_dir,
            episode_index,
            len(issues),
        )

    def wait(self, timeout: float | None = None) -> None:
        """Wait for submitted checks, surfacing unexpected worker failures."""
        deadline = None if timeout is None else time.monotonic() + timeout
        with self._lock:
            futures = list(self._futures)
        for future in futures:
            remaining = None if deadline is None else max(0.0, deadline - time.monotonic())
            future.result(timeout=remaining)

    def close(self) -> None:
        try:
            self.wait()
        finally:
            self._executor.shutdown(wait=True)
