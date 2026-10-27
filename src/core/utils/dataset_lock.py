"""One exclusive lock per collection dataset, shared across processes.

Several EVA processes record into the same collection dataset, so every read-modify-write
of its ``meta/*`` files and every episode/frame index allocation must be serialized.
The lock is an ``flock`` on a persistent file in the dataset.  Locking the directory
itself does not provide reliable mutual exclusion on the shared filesystem.
"""

from __future__ import annotations

import fcntl
import logging
import os
import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

logger = logging.getLogger(__name__)

# Waiting is expected while a peer encodes an episode; a long wait is worth reporting.
_WARN_AFTER_SEC = 1.0

_registry_lock = threading.RLock()
_locks: dict[Path, DatasetLock] = {}


class DatasetLock:
    """Reentrant, thread-safe exclusive lock for one dataset directory.

    Every caller in the process shares one instance per directory, and therefore one
    descriptor: a second ``flock`` here would block against the first. Threads hold the
    reentrant ``_thread_lock`` for their whole with-block and only the outermost acquire
    takes the kernel lock.
    """

    def __init__(self, dataset_dir: Path) -> None:
        self._dataset_dir = dataset_dir
        self._lock_path = dataset_dir / ".dataset.lock"
        self._thread_lock = threading.RLock()
        self._descriptor: int | None = None
        self._depth = 0

    @contextmanager
    def acquire(self) -> Iterator[None]:
        """Hold the dataset lock for the duration of the with-block."""
        started = time.monotonic()
        self._thread_lock.acquire()
        try:
            if self._depth == 0:
                self._dataset_dir.mkdir(parents=True, exist_ok=True)
                descriptor = os.open(
                    self._lock_path,
                    os.O_CREAT | os.O_RDWR,
                    0o600,
                )
                try:
                    fcntl.flock(descriptor, fcntl.LOCK_EX)
                except BaseException:
                    os.close(descriptor)
                    raise
                self._descriptor = descriptor
            self._depth += 1
            waited = time.monotonic() - started
            if waited > _WARN_AFTER_SEC:
                logger.warning(
                    "collection dataset lock waited %.1fs: %s", waited, self._dataset_dir
                )
            try:
                yield
            finally:
                self._depth -= 1
        finally:
            if self._depth == 0 and self._descriptor is not None:
                os.close(self._descriptor)
                self._descriptor = None
            self._thread_lock.release()


def dataset_lock(dataset_dir: str | Path) -> DatasetLock:
    """Return the process-wide lock object for one collection dataset directory."""
    resolved = Path(dataset_dir).resolve()
    with _registry_lock:
        lock = _locks.get(resolved)
        if lock is None:
            lock = DatasetLock(resolved)
            _locks[resolved] = lock
        return lock
