"""SSH local-forward helper for web eval — tunnels each ckpt's serving port from a
remote login endpoint to the same local port, mirroring scripts/run_ssh_port.sh but
driven by the eval config (ssh_host/ssh_user/ssh_port + each checkpoint's port).

Created By Yi Yang
"""

from __future__ import annotations

import logging
import os
import signal
import subprocess
import threading
import time
from pathlib import Path

import psutil

from core.registry import FUNCTIONS

logger = logging.getLogger(__name__)


def _pids_on_port(port: int) -> list[int]:
    """int port -> list of local PIDs currently LISTENing on that TCP port (excluding self).

    Uses psutil instead of shelling out to lsof: on hosts with many NFS mounts lsof
    stats every mount and can hang for tens of seconds. Degrades to [] when the OS
    denies the socket scan (e.g. macOS dev boxes require root), so startup never
    crashes — port freeing is only load-bearing on the Linux eval-SSH path.
    """
    self_pid = os.getpid()
    try:
        conns = psutil.net_connections(kind="tcp")
    except (psutil.AccessDenied, PermissionError):
        return []
    pids = {
        conn.pid
        for conn in conns
        if conn.status == psutil.CONN_LISTEN
        and conn.laddr
        and conn.laddr.port == port
        and conn.pid not in (None, self_pid)
    }
    return sorted(pids)


def free_local_ports(ports: list[int]) -> None:
    """list[int] ports -> None; kill any local listeners on those ports so a fresh
    bind/forward never collides with a leftover server or tunnel from a prior run.
    """
    for port in sorted(set(ports)):
        pids = _pids_on_port(port)
        if not pids:
            continue
        logger.info("[ssh_forward] freeing port %d held by pids %s", port, pids)
        for pid in pids:
            _kill(pid, signal.SIGTERM)
        time.sleep(0.5)
        for pid in _pids_on_port(port):
            logger.warning("[ssh_forward] port %d still held by pid %d, sending SIGKILL", port, pid)
            _kill(pid, signal.SIGKILL)


def _kill(pid: int, sig: int) -> None:
    """(int pid, int sig) -> None; signal pid, ignoring the race where it already exited."""
    try:
        os.kill(pid, sig)
    except ProcessLookupError:
        pass


@FUNCTIONS.register("ssh_forward")
def start_ssh_forward(
    ssh: dict, ports: list[int]
) -> subprocess.Popen[bytes] | None:
    """Start one multiplexed `ssh -fN -L` forwarding each port 1:1 to remote localhost.

    Args:
        ssh: eval.ssh block with ``host`` / ``user`` / ``port`` (remote login endpoint).
        ports: local ports to map 1:1 onto the remote side.

    Returns the ssh Popen handle, or None when host/ports are missing (forwarding off).
    """
    ssh_host = str(ssh.get("host", ""))
    ssh_user = str(ssh.get("user", ""))
    ssh_port = int(ssh.get("port", 0))
    if not ssh_host or not ports:
        return None
    target = f"{ssh_user}@{ssh_host}" if ssh_user else ssh_host
    args = [
        "ssh",
        "-N",
        "-o",
        "ExitOnForwardFailure=yes",
        "-o",
        "ServerAliveInterval=30",
        "-o",
        "StrictHostKeyChecking=accept-new",
    ]
    for p in ports:
        args += ["-L", f"{p}:localhost:{p}"]
    args += ["-p", str(ssh_port), target]
    logger.info("[ssh_forward] forwarding ports %s via %s (ssh port %d)", ports, target, ssh_port)
    proc = subprocess.Popen(args)
    logger.info("[ssh_forward] tunnels up (pid=%d)", proc.pid)
    return proc


def stop_ssh_forward(proc: subprocess.Popen[bytes] | None) -> None:
    """Terminate a forwarder started by start_ssh_forward."""
    if proc is None:
        return
    proc.terminate()
    proc.wait(timeout=5)
    logger.info("[ssh_forward] tunnels down")


class ResultSyncer:
    """Periodic rsync pusher: local_dir -> user@host:remote_dir (every interval_s)."""

    def __init__(
        self,
        local_dir: Path,
        remote_dir: str,
        ssh_host: str,
        ssh_user: str,
        ssh_port: int,
        interval_s: float,
    ) -> None:
        self.local_dir = local_dir
        self.remote_dir = remote_dir.rstrip("/")
        self.ssh_host = ssh_host
        self.ssh_user = ssh_user
        self.ssh_port = ssh_port
        self.interval_s = interval_s
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def _target(self) -> str:
        login = f"{self.ssh_user}@{self.ssh_host}" if self.ssh_user else self.ssh_host
        # trailing slash on src: copy contents of local_dir into remote_dir
        return f"{login}:{self.remote_dir}/"

    def _sync_once(self) -> None:
        """Run one rsync push; logs and swallows failures so the loop keeps going."""
        src = f"{str(self.local_dir).rstrip('/')}/"
        # --update: never overwrite a remote file that is newer than the local one,
        # so manual edits on the sync target (e.g. re-scored results.jsonl) survive.
        args = [
            "rsync",
            "-az",
            "--update",
            "-e",
            f"ssh -p {self.ssh_port} -o StrictHostKeyChecking=accept-new",
            src,
            self._target(),
        ]
        result = subprocess.run(args, capture_output=True, text=True)
        if result.returncode != 0:
            logger.warning(
                "[result_sync] rsync failed (%d): %s", result.returncode, result.stderr.strip()
            )
        else:
            logger.info("[result_sync] synced %s -> %s", src, self._target())

    def _run(self) -> None:
        while not self._stop.wait(self.interval_s):
            self._sync_once()

    def start(self) -> None:
        """Ensure the remote dir exists, then start the periodic sync thread."""
        login = f"{self.ssh_user}@{self.ssh_host}" if self.ssh_user else self.ssh_host
        subprocess.run(
            [
                "ssh",
                "-p",
                str(self.ssh_port),
                "-o",
                "StrictHostKeyChecking=accept-new",
                login,
                f"mkdir -p {self.remote_dir}",
            ],
            capture_output=True,
            text=True,
        )
        self._thread = threading.Thread(target=self._run, name="eva-result-sync", daemon=True)
        self._thread.start()
        logger.info(
            "[result_sync] started: %s -> %s (every %.0fs)",
            self.local_dir,
            self._target(),
            self.interval_s,
        )

    def stop(self) -> None:
        """Stop the loop and do one final sync to flush the latest results."""
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2)
        self._sync_once()
        logger.info("[result_sync] stopped (final sync done)")


@FUNCTIONS.register("result_sync")
def start_result_sync(
    ssh: dict,
    local_dir: Path,
    interval_s: float = 20.0,
) -> ResultSyncer | None:
    """Start a ResultSyncer, or None when host/remote_sync_dir missing (sync off).

    Args:
        ssh: eval.ssh block with ``host`` / ``user`` / ``port`` / ``remote_sync_dir``.
        local_dir: local results root to push.
        interval_s: seconds between rsync pushes.
    """
    ssh_host = str(ssh.get("host", ""))
    remote_dir = str(ssh.get("remote_sync_dir", ""))
    if not ssh_host or not remote_dir:
        return None
    syncer = ResultSyncer(
        local_dir,
        remote_dir,
        ssh_host,
        str(ssh.get("user", "")),
        int(ssh.get("port", 0)),
        interval_s,
    )
    syncer.start()
    return syncer
