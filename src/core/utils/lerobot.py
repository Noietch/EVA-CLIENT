"""LeRobot v2 dataset I/O — read/inspect/annotate a dataset directory.

LeRobotDatasetIO wraps a dataset root and provides the parquet/video path
resolution, episode counting, trajectory loading, key inference, and QC /
language-annotation read-write helpers that the console and replay paths use.
The DatasetTransport (transport/dataset.py) holds one of these for its own reads.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


class LeRobotDatasetIO:
    """Read/inspect/annotate helpers for one LeRobot v2 dataset directory.

    All methods operate on ``self.root`` (the dataset root). Pure dataset I/O —
    no transport / robot state. Used by the console (server.py), the io handler,
    and the dataset transport.
    """

    # Episode quality-check language annotation. Mirrors the eval score write path:
    # the text lives as a top-level key in meta/episodes.jsonl, as a constant per-frame
    # string column in the episode's data parquet (so it travels inside the LeRobot
    # table like task_index), and the column is declared in meta/info.json's features.
    _ANNOTATION_KEY = "language_annotation"

    def __init__(self, dataset_dir: str | Path) -> None:
        self.root = Path(dataset_dir)

    def episode_parquet(self, episode_id: int) -> Path:
        """Resolve the parquet path for one episode from the dataset's info.json layout."""
        info_path = self.root / "meta" / "info.json"
        if info_path.exists():
            with open(info_path) as f:
                info = json.load(f)
            chunks_size = info.get("chunks_size", 1000)
            episode_chunk = episode_id // chunks_size
            episode_file = self.episode_file_stem(episode_id)
            data_path_tpl = info.get(
                "data_path",
                "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
            )
            return self.root / data_path_tpl.format(
                episode_chunk=episode_chunk,
                episode_index=episode_id,
                episode_file=episode_file,
            )
        return self.root / f"episode_{episode_id:06d}.parquet"

    def episode_video_paths(
        self,
        episode_id: int,
        video_keys: dict[str, str],
    ) -> dict[str, Path]:
        """Resolve available mp4 paths for one episode from dataset video keys.

        Args:
            episode_id: Episode index to resolve.
            video_keys: Mapping from caller camera key to dataset video column key.

        Returns:
            Existing video paths keyed by the caller camera key.
        """
        info_path = self.root / "meta" / "info.json"
        if info_path.exists():
            with open(info_path) as f:
                info = json.load(f)
            chunks_size = info.get("chunks_size", 1000)
            episode_chunk = episode_id // chunks_size
            episode_file = self.episode_file_stem(episode_id)
            video_path_tpl = info.get(
                "video_path",
                "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
            )
        else:
            episode_chunk = 0
            episode_file = f"episode_{episode_id:06d}"
            video_path_tpl = (
                "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4"
            )

        out: dict[str, Path] = {}
        for cam_key, video_key in video_keys.items():
            if not video_key:
                continue
            path = self.root / video_path_tpl.format(
                episode_chunk=episode_chunk,
                video_key=video_key,
                episode_index=episode_id,
                episode_file=episode_file,
            )
            if path.exists():
                out[cam_key] = path
        return out

    def _episode_rows_by_index(self) -> dict[int, dict]:
        path = self.root / "meta" / "episodes.jsonl"
        rows: dict[int, dict] = {}
        if not path.exists():
            return rows
        with path.open() as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                rows[int(rec["episode_index"])] = rec
        return rows

    def episode_file_stem(self, episode_id: int) -> str:
        row = self._episode_rows_by_index().get(episode_id, {})
        stem = row.get("file_stem")
        if stem:
            return str(stem)
        return f"episode_{episode_id:06d}"

    def tasks_by_index(self) -> dict[int, str]:
        # tasks.jsonl maps task_index -> task text; a parquet's task_index column then
        # resolves each episode's prompt without relying on episodes.jsonl.
        path = self.root / "meta" / "tasks.jsonl"
        tasks: dict[int, str] = {}
        if not path.exists():
            return tasks
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                tasks[int(rec["task_index"])] = rec.get("task", "")
        return tasks

    @staticmethod
    def resolve_task(table: object, tasks_by_index: dict[int, str]) -> str:
        # The episode's prompt = task text of its first row's task_index (one task per episode).
        if "task_index" not in table.column_names:  # type: ignore[attr-defined]
            return ""
        task_indices = table.column("task_index").to_pylist()  # type: ignore[attr-defined]
        if not task_indices:
            return ""
        return tasks_by_index.get(int(task_indices[0]), "")

    def load_trajectory(
        self,
        episode_id: int,
        action_key: str = "action",
    ) -> tuple[np.ndarray, str]:
        """Load one episode's recorded action trajectory and task text.

        Standalone reader used by the replay policy so it can source a trajectory without
        a dataset transport (e.g. when publishing onto a real robot over a zmq transport).

        Args:
            episode_id: episode index to load.
            action_key: parquet column holding the recorded action vectors.

        Returns:
            actions: [n_steps, action_dim] float32 recorded action sequence.
            task: episode prompt text ("" if the dataset records none).
        """
        parquet_path = self.episode_parquet(episode_id)
        if not parquet_path.exists():
            raise FileNotFoundError(f"Parquet not found: {parquet_path}")
        table = pq.read_table(str(parquet_path))
        actions = np.array(table.column(action_key).to_pylist(), dtype=np.float32)
        task = self.resolve_task(table, self.tasks_by_index())
        return actions, task

    def count_episodes(self) -> int:
        """Total episode count of the dataset dir.

        Authoritative count is info.json total_episodes; episodes.jsonl may be absent or
        list tasks (which outnumber episodes), so never count tasks here.
        """
        info_path = self.root / "meta" / "info.json"
        if info_path.exists():
            with open(info_path) as f:
                total = json.load(f).get("total_episodes")
            if total is not None:
                return int(total)
        return len(list(self.root.glob("data/**/*.parquet")))

    def mark_qc(self, episode: int, verdict: str, note: str = "") -> bool:
        """Write a quality-check verdict onto an episode's meta/episodes.jsonl row.

        Merges ``qc_verdict`` ("pass"/"fail") and ``qc_note`` into the matching row in
        place, leaving the recorded trajectory untouched. An empty ``verdict`` keeps the
        existing verdict so a note can be saved on its own. Returns False when the dataset
        has no episodes.jsonl or the episode index is absent.
        """
        path = self.root / "meta" / "episodes.jsonl"
        if not path.exists():
            return False
        rows = []
        with path.open() as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
        patched = False
        for row in rows:
            if int(row.get("episode_index", -1)) == episode:
                if verdict:
                    row["qc_verdict"] = verdict
                row["qc_note"] = note
                patched = True
                break
        if not patched:
            return False
        with path.open("w") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        return True

    def read_annotation(self, episode: int) -> str:
        """Read an episode's stored language annotation from meta/episodes.jsonl.

        Returns "" when the dataset, the episode row, or the annotation key is absent,
        so the console can prefill the editor (empty means "not yet annotated").
        """
        path = self.root / "meta" / "episodes.jsonl"
        if not path.exists():
            return ""
        with path.open() as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                if int(row.get("episode_index", -1)) == episode:
                    return str(row.get(self._ANNOTATION_KEY, "") or "")
        return ""

    def annotate_language(self, episode: int, annotation: str) -> bool:
        """Write a quality-check language annotation onto an episode, lerobot-native.

        Mirrors the eval score write path across three places so the annotation lives
        inside the recorded data, and is re-editable (each call overwrites in place):
          1. meta/episodes.jsonl: the episode row's ``language_annotation`` key.
          2. data parquet: a constant per-frame ``language_annotation`` string column.
          3. meta/info.json: declares ``language_annotation`` in ``features``.

        Returns False when the dataset has no episodes.jsonl or the episode is absent.
        """
        path = self.root / "meta" / "episodes.jsonl"
        if not path.exists():
            return False
        rows = []
        with path.open() as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
        patched = False
        for row in rows:
            if int(row.get("episode_index", -1)) == episode:
                row[self._ANNOTATION_KEY] = annotation
                patched = True
                break
        if not patched:
            return False
        with path.open("w") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        self._write_annotation_parquet_column(episode, annotation)
        self._declare_annotation_feature()
        return True

    def _write_annotation_parquet_column(self, episode: int, annotation: str) -> None:
        # Rewrite the episode parquet's constant language_annotation string column in place.
        parquet_path = self.episode_parquet(episode)
        if not parquet_path.exists():
            return
        table = pq.read_table(str(parquet_path))
        arr = pa.array([annotation] * table.num_rows, type=pa.string())
        if self._ANNOTATION_KEY in table.column_names:
            table = table.set_column(
                table.column_names.index(self._ANNOTATION_KEY), self._ANNOTATION_KEY, arr
            )
        else:
            table = table.append_column(self._ANNOTATION_KEY, arr)
        pq.write_table(table, str(parquet_path))

    def _declare_annotation_feature(self) -> None:
        # Declare language_annotation in info.json features (idempotent), matching the
        # constant-per-frame string column written into each episode's parquet.
        info_path = self.root / "meta" / "info.json"
        if not info_path.exists():
            return
        with open(info_path) as f:
            info = json.load(f)
        features = info.setdefault("features", {})
        if self._ANNOTATION_KEY not in features:
            features[self._ANNOTATION_KEY] = {"dtype": "string", "shape": [1], "names": None}
            with open(info_path, "w") as f:
                json.dump(info, f, indent=2, ensure_ascii=False)

    def infer_keys(self) -> dict:
        """Guess image / state / action column roles from the dataset's info.json features.

        Reads ``meta/info.json`` and classifies each feature by name/dtype so the console
        can prefill the dataset_keys picker (and let the user override when a role has
        several candidates). Pure inspection — no parquet/video is opened.

        Classification:
            image  : dtype == "video", or the key contains "image"/"images".
            action : default prefers "action"/"actions" or an "action"-prefixed key.
            state  : default prefers observation.state / observation.qpos.
        Action and state candidates list every non-image column so a dataset whose
        action/state column is named unconventionally can still be picked.

        Returns:
            {role: {"candidates": [col, ...], "default": col | None}} for image/state/action.
            Defaults prefer the most conventional name; None when a role has no candidate.
        """
        info_path = self.root / "meta" / "info.json"
        if not info_path.exists():
            raise FileNotFoundError(f"info.json not found: {info_path}")
        with open(info_path) as f:
            features = json.load(f).get("features", {})

        images: list[str] = []
        non_image: list[str] = []
        action_named: list[str] = []
        state_named: list[str] = []
        for key, spec in features.items():
            dtype = spec.get("dtype", "") if isinstance(spec, dict) else ""
            low = key.lower()
            if dtype == "video" or "image" in low:
                images.append(key)
                continue
            non_image.append(key)
            if low in {"action", "actions"} or low.startswith("action"):
                action_named.append(key)
            if "state" in low or low in {"observation.qpos", "observation.state"}:
                state_named.append(key)

        def _pick(candidates: list[str], preferred: list[str]) -> str | None:
            for p in preferred:
                if p in candidates:
                    return p
            return candidates[0] if candidates else None

        return {
            "image": {"candidates": images, "default": images[0] if images else None},
            "state": {
                "candidates": non_image,
                "default": _pick(state_named, ["observation.state", "observation.qpos"])
                or (state_named[0] if state_named else None),
            },
            "action": {
                "candidates": non_image,
                "default": _pick(action_named, ["action", "actions"])
                or (action_named[0] if action_named else None),
            },
        }
