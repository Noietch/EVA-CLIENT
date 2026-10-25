"""Local DEBUG sessions and optional, separately recorded rollout datasets."""

from __future__ import annotations

import datetime as dt
import json
import logging
import os
import re
import threading
import uuid
from pathlib import Path
from typing import Any

from core.app.state import RuntimeState, SessionMode, SessionState
from core.config import ConfigDict, deployment_models

logger = logging.getLogger(__name__)
_ID = re.compile(r"[a-f0-9]{32}")


def _now() -> str:
    return dt.datetime.now().astimezone().isoformat(timespec="seconds")


def _checked_id(value: str) -> str:
    if not isinstance(value, str) or not _ID.fullmatch(value):
        raise ValueError("invalid local session or rollout ID")
    return value


class LocalHistoryStore:
    """Atomic JSON session metadata; recording files live in per-rollout datasets."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root).expanduser().resolve()
        self._lock = threading.RLock()

    def _path(self, session_id: str) -> Path:
        return self.root / "sessions" / _checked_id(session_id) / "session.json"

    def dataset_path(self, rollout_id: str) -> Path:
        return self.root / "rollouts" / _checked_id(rollout_id)

    def _read(self, session_id: str) -> dict[str, Any]:
        data = json.loads(self._path(session_id).read_text(encoding="utf-8"))
        if data.get("id") != session_id or not isinstance(data.get("rollouts"), list):
            raise ValueError("invalid local session file")
        return data

    def _write(self, data: dict[str, Any]) -> None:
        path = self._path(data["id"])
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(f".{uuid.uuid4().hex}.tmp")
        try:
            temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
            temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)

    def begin(self, session_id: str | None, **fields: Any) -> tuple[str, str]:
        with self._lock:
            timestamp = _now()
            if session_id:
                data = self._read(session_id)
            else:
                session_id = uuid.uuid4().hex
                data = {
                    "id": session_id,
                    "title": fields["prompt"][:80],
                    "created_at": timestamp,
                    "rollouts": [],
                }
            rollout_id = uuid.uuid4().hex
            record = {
                "id": rollout_id,
                "started_at": timestamp,
                "ended_at": None,
                "status": "running",
                "record_status": "recording" if fields["record"] else "off",
                "owner_pid": os.getpid(),
                **fields,
            }
            data["rollouts"].append(record)
            data["updated_at"] = timestamp
            self._write(data)
            return session_id, rollout_id

    def update(self, session_id: str, rollout_id: str, **fields: Any) -> None:
        with self._lock:
            data = self._read(session_id)
            record = next(item for item in data["rollouts"] if item["id"] == rollout_id)
            record.update(fields)
            data["updated_at"] = _now()
            self._write(data)

    def resume(self, session_id: str, rollout_id: str, *, record: bool) -> int:
        """Reopen the current run after its previous recording has finished saving."""
        with self._lock:
            data = self._read(session_id)
            row = next(
                item
                for item in data["rollouts"]
                if item["id"] == rollout_id and not item.get("deleted_at")
            )
            if row.get("record_status") == "saving":
                raise ValueError("The previous recording is still saving; try again shortly")
            previous_steps = int(row.get("steps") or 0)
            row.update(status="running", ended_at=None, error="", owner_pid=os.getpid())
            row["record"] = bool(row.get("record") or record)
            if record:
                row["record_status"] = "recording"
            row["continuations"] = int(row.get("continuations", 0)) + 1
            data["updated_at"] = _now()
            self._write(data)
            return previous_steps

    def set_deleted(self, session_id: str, rollout_id: str, deleted: bool) -> None:
        """Soft-delete one run, keeping its metadata and recording recoverable."""
        _checked_id(rollout_id)
        with self._lock:
            data = self._read(session_id)
            record = next((row for row in data["rollouts"] if row["id"] == rollout_id), None)
            if record is None:
                raise KeyError("local run not found")
            if bool(record.get("deleted_at")) == deleted:
                return  # A retried delete/restore must succeed after its first write.
            if deleted:
                current = next(
                    row for row in self.read(session_id)["rollouts"] if row["id"] == rollout_id
                )
                if current["status"] == "running" or current["record_status"] in {
                    "recording",
                    "saving",
                }:
                    raise ValueError("wait for the run and recording to finish")
            live = self.dataset_path(rollout_id)
            archived = self.root / "trash" / rollout_id
            source, target = (live, archived) if deleted else (archived, live)
            moved = source.exists()
            if moved:
                if target.exists():
                    raise ValueError("recording destination already exists")
                target.parent.mkdir(parents=True, exist_ok=True)
                source.rename(target)
            record["deleted_at"] = _now() if deleted else None
            data["updated_at"] = _now()
            try:
                self._write(data)
            except Exception:
                if moved:
                    target.rename(source)
                raise

    def read(self, session_id: str) -> dict[str, Any]:
        with self._lock:
            data = self._read(session_id)
        data["rollouts"] = [row for row in data["rollouts"] if not row.get("deleted_at")]
        for record in data["rollouts"]:
            path = self.dataset_path(record["id"])
            metadata = path / "meta/episodes.jsonl"
            record["record_status"] = record.get("record_status", "off")
            record["dataset_dir"] = str(path) if record.get("record") else ""
            record["playable"] = bool(record.get("record") and metadata.is_file())
            if (
                record["playable"]
                and record["status"] != "running"
                and record["record_status"] not in {"saving", "error"}
            ):
                record["record_status"] = "saved"
            if record["status"] == "running" or record["record_status"] == "saving":
                try:
                    pid = int(record.get("owner_pid", 0))
                    if pid <= 0:
                        raise ValueError("missing owner")
                    os.kill(pid, 0)
                except (OSError, ValueError):
                    if record["status"] == "running":
                        record["status"] = "interrupted"
                    if record["record_status"] in {"recording", "saving"}:
                        record["record_status"] = "interrupted"
                    if record["playable"]:
                        record["record_status"] = "saved"
        return data

    def list(self, limit: int = 100) -> list[dict[str, Any]]:
        sessions = self.root / "sessions"
        if not sessions.is_dir():
            return []
        paths = sorted(
            sessions.glob("*/session.json"), key=lambda p: p.stat().st_mtime_ns, reverse=True
        )
        result = []
        for path in paths:
            try:
                data = self.read(path.parent.name)
            except (OSError, ValueError, KeyError, TypeError):
                continue
            records = data["rollouts"]
            if not records:
                continue
            result.append(
                {key: data[key] for key in ("id", "title", "created_at", "updated_at")}
                | {
                    "rollouts": len(records),
                    "model": records[-1].get("model", ""),
                    "status": records[-1]["status"],
                    "recorded": sum(bool(item.get("playable")) for item in records),
                    "runs": [
                        {
                            key: record.get(key)
                            for key in (
                                "id",
                                "prompt",
                                "model",
                                "mode",
                                "strategy",
                                "started_at",
                                "status",
                                "steps",
                                "record",
                                "record_status",
                                "playable",
                            )
                        }
                        for record in records
                    ],
                }
            )
            if len(result) >= limit:
                break
        return result


def history_enabled(config: ConfigDict, runtime: RuntimeState) -> bool:
    return bool((config.get("console") or {}).get("session_history", False)) and not (
        config.get("eval") or getattr(runtime, "rl_active", False)
    )


def history_store(config: ConfigDict, runtime: RuntimeState) -> LocalHistoryStore:
    root = str((config.get("console") or {}).get("history_dir", "~/.eva-client"))
    store = runtime.local_history_store
    if store is None or store.root != Path(root).expanduser().resolve():
        store = LocalHistoryStore(root)
        runtime.local_history_store = store
    return store


def history_status(config: ConfigDict, runtime: RuntimeState) -> dict[str, Any]:
    if runtime.debug_record_enabled is None:
        runtime.debug_record_enabled = bool(
            (config.get("console") or {}).get("record_rollouts", False)
        )
    saving = False
    current = runtime.local_current_rollout_id
    if current and runtime.local_session_id and not runtime.local_rollout_id:
        try:
            row = next(
                item
                for item in history_store(config, runtime).read(runtime.local_session_id)[
                    "rollouts"
                ]
                if item["id"] == current
            )
            saving = row["record_status"] == "saving"
        except (OSError, ValueError, KeyError, StopIteration):
            runtime.local_current_rollout_id = current = None
    return {
        "enabled": history_enabled(config, runtime),
        "record": runtime.debug_record_enabled,
        "session_id": runtime.local_session_id,
        "rollout_id": runtime.local_rollout_id,
        "current_rollout_id": current,
        "saving": saving,
        "generation": runtime.local_run_generation,
    }


def begin_local_rollout(
    config: ConfigDict, runtime: RuntimeState, session: SessionState
) -> bool | None:
    if not history_enabled(config, runtime) or runtime.replay_source is not None:
        return False
    if session.mode not in {SessionMode.REAL, SessionMode.SIM, SessionMode.STEP}:
        return False
    if runtime.local_rollout_id:
        return True
    store = history_store(config, runtime)
    record = history_status(config, runtime)["record"]
    models = deployment_models(config)
    index = min(runtime.active_model_index, len(models) - 1)
    try:
        if runtime.local_current_rollout_id and runtime.local_session_id:
            session_id, rollout_id = runtime.local_session_id, runtime.local_current_rollout_id
            runtime.local_rollout_previous_steps = store.resume(
                session_id, rollout_id, record=record
            )
        else:
            session_id, rollout_id = store.begin(
                None,
                prompt=session.selected_task or "",
                model=models[index].name,
                robot=config.robot.type,
                mode=session.mode.value,
                strategy=runtime.selected_inference_strategy_key,
                record=record,
            )
            runtime.local_rollout_previous_steps = 0
    except (OSError, ValueError, KeyError, StopIteration) as error:
        session.last_error = f"Unable to save local session: {error}"
        logger.exception("Unable to create local execution history")
        return None
    runtime.local_session_id = session_id
    runtime.local_rollout_id = rollout_id
    runtime.local_current_rollout_id = rollout_id
    runtime.debug_record_dataset_dir = str(store.dataset_path(rollout_id))
    runtime.local_rollout_start_step = session.step_index
    return True


def finish_local_rollout(
    config: ConfigDict, runtime: RuntimeState, session: SessionState, reason: str = "stopped"
) -> None:
    if not getattr(runtime, "local_rollout_id", None):
        return
    from core.app.handlers.recording import (
        discard_rollout_episode,
        mark_rollout_save_ready,
        save_rollout_episode,
    )

    store = history_store(config, runtime)
    session_id = runtime.local_session_id
    rollout_id = runtime.local_rollout_id
    assert session_id is not None
    runtime.local_rollout_id = None  # finish once, including reentrant interrupts
    error = session.last_error
    record = bool(runtime.debug_record_enabled)
    try:
        store.update(
            session_id,
            rollout_id,
            status="error" if error else reason,
            ended_at=_now(),
            error=error,
            steps=runtime.local_rollout_previous_steps
            + max(0, session.step_index - runtime.local_rollout_start_step),
            record_status="saving" if record else "off",
        )
    except (OSError, ValueError, KeyError) as exc:
        logger.exception("Unable to finish local execution history")
        discard_rollout_episode(runtime)
        session.last_error = f"Unable to save local session: {exc}"
        return
    if not record:
        return
    try:
        logger_obj = runtime.rollout_episode_logger
        if logger_obj is None or not logger_obj.has_active_episode:
            store.update(session_id, rollout_id, record_status="empty")
            return

        def completed(job: Any) -> None:
            store.update(
                session_id,
                rollout_id,
                record_status="saved" if job.status == "saved" else "error",
                record_error=str(job.error or ""),
            )

        logger_obj.set_save_completion_callback(completed)
        logger_obj.set_episode_meta(local_session_id=session_id, local_rollout_id=rollout_id)
        mark_rollout_save_ready(runtime, reason)
        if not runtime.rollout_save_ready:
            store.update(session_id, rollout_id, record_status="empty")
        elif not save_rollout_episode(runtime, session):
            store.update(
                session_id, rollout_id, record_status="error", record_error=session.last_error
            )
    except Exception as exc:
        logger.exception("Unable to save local rollout %s", rollout_id)
        try:
            store.update(session_id, rollout_id, record_status="error", record_error=str(exc))
        except (OSError, ValueError, KeyError):
            logger.exception("Unable to persist recording failure")
    finally:
        session.last_error = error
