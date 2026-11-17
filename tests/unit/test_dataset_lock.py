from __future__ import annotations

import fcntl
import multiprocessing as mp
import os
import signal
import threading
import time
from pathlib import Path

import pytest

from core.utils.dataset_lock import dataset_lock

pytestmark = pytest.mark.unit

_HOLD_SEC = 0.5


def _hold(dataset_dir: str, held, seconds: float = _HOLD_SEC) -> None:
    with dataset_lock(dataset_dir).acquire():
        held.set()
        time.sleep(seconds)


def _hold_until_killed(dataset_dir: str, held) -> None:
    with dataset_lock(dataset_dir).acquire():
        held.set()
        time.sleep(30.0)


def _probe_raw_lock(dataset_dir: str, result) -> None:
    descriptor = os.open(os.path.join(dataset_dir, ".dataset.lock"), os.O_RDWR)
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        result.put(True)
    except BlockingIOError:
        result.put(False)
    finally:
        os.close(descriptor)


def _kernel_lock_is_free(dataset_dir: Path) -> bool:
    """Whether a fresh descriptor can take the dataset lock right now."""
    descriptor = os.open(dataset_dir / ".dataset.lock", os.O_RDWR)
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        return True
    except BlockingIOError:
        return False
    finally:
        os.close(descriptor)


def _child_can_take_the_kernel_lock(dataset_dir: Path) -> bool:
    """Ask a child process whether the dataset lock is free right now."""
    context = mp.get_context("fork")
    result = context.Queue()
    child = context.Process(target=_probe_raw_lock, args=(str(dataset_dir), result))
    child.start()
    child.join(10.0)
    return bool(result.get(timeout=1.0))


def test_second_process_waits_for_the_holder(tmp_path: Path) -> None:
    dataset_dir = tmp_path / "set"
    context = mp.get_context("fork")
    held = context.Event()
    holder = context.Process(target=_hold, args=(str(dataset_dir), held))
    holder.start()
    assert held.wait(10.0)
    assert not _kernel_lock_is_free(dataset_dir)

    started = time.monotonic()
    with dataset_lock(dataset_dir).acquire():
        waited = time.monotonic() - started
        assert not _kernel_lock_is_free(dataset_dir)
    holder.join(10.0)
    assert holder.exitcode == 0
    assert waited >= 0.05


def test_lock_file_is_persistent_and_private(tmp_path: Path) -> None:
    dataset_dir = tmp_path / "set"
    with dataset_lock(dataset_dir).acquire():
        lock_path = dataset_dir / ".dataset.lock"
        assert lock_path.is_file()
        assert lock_path.stat().st_mode & 0o777 == 0o600
    assert (dataset_dir / ".dataset.lock").is_file()


def test_failed_kernel_acquire_closes_its_descriptor(tmp_path: Path, monkeypatch) -> None:
    lock = dataset_lock(tmp_path / "set")

    def fail(_descriptor, _operation):
        raise OSError("lock unavailable")

    monkeypatch.setattr(fcntl, "flock", fail)
    with pytest.raises(OSError, match="lock unavailable"):
        with lock.acquire():
            pass
    assert lock._descriptor is None
    assert lock._depth == 0


def test_lock_is_released_when_the_holder_is_killed(tmp_path: Path) -> None:
    dataset_dir = tmp_path / "set"
    context = mp.get_context("fork")
    held = context.Event()
    holder = context.Process(target=_hold_until_killed, args=(str(dataset_dir), held))
    holder.start()
    assert held.wait(10.0)
    os.kill(holder.pid, signal.SIGKILL)
    holder.join(10.0)
    started = time.monotonic()
    with dataset_lock(dataset_dir).acquire():
        pass
    assert time.monotonic() - started < 1.0


def test_nested_acquire_releases_only_at_the_outermost_exit(tmp_path: Path) -> None:
    dataset_dir = tmp_path / "set"
    lock = dataset_lock(dataset_dir)
    with lock.acquire():
        with lock.acquire():
            assert not _child_can_take_the_kernel_lock(dataset_dir)
        assert not _child_can_take_the_kernel_lock(dataset_dir)
    assert _child_can_take_the_kernel_lock(dataset_dir)


def test_threads_serialize_within_one_process(tmp_path: Path) -> None:
    lock = dataset_lock(tmp_path / "set")
    events: list[str] = []

    def worker(name: str, hold: float) -> None:
        with lock.acquire():
            events.append(f"{name}-enter")
            time.sleep(hold)
            events.append(f"{name}-exit")

    threads = [
        threading.Thread(target=worker, args=("first", 0.2)),
        threading.Thread(target=worker, args=("second", 0.0)),
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(10.0)
    assert events in (
        ["first-enter", "first-exit", "second-enter", "second-exit"],
        ["second-enter", "second-exit", "first-enter", "first-exit"],
    )


def test_separate_datasets_use_separate_locks(tmp_path: Path) -> None:
    held_dir = tmp_path / "held"
    assert dataset_lock(tmp_path / "one") is dataset_lock(tmp_path / "one" / ".." / "one")
    context = mp.get_context("fork")
    held = context.Event()
    holder = context.Process(target=_hold_until_killed, args=(str(held_dir), held))
    holder.start()
    assert held.wait(10.0)
    started = time.monotonic()
    with dataset_lock(tmp_path / "other").acquire():
        pass
    assert time.monotonic() - started < 1.0
    holder.terminate()
    holder.join(10.0)
