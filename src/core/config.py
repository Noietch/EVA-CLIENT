"""Minimal config loader wrapping Config.fromfile + a few derived fields.

The previous 750-line dataclass + hand-written YAML loader is replaced by:

  1. ``configs/_base_/defaults.py`` — single source of truth for default values
     (every preset inherits via ``_base_ = ["../../_base_/defaults.py"]``).
  2. ``Config.fromfile`` (vendored mmengine) — handles .py lazy config loading
     and deep ``_base_`` merge.
  3. ``load_config`` (this file) — wraps the loaded ConfigDict, builds
     ``obs_space`` / ``action_space`` into JointState / EEFPose instances,
     applies path-dependent derived fields, and validates.

The returned ConfigDict supports dotted attribute access
(``cfg.transport.image_height``) and behaves like a plain dict otherwise.
"""

from __future__ import annotations

from pathlib import Path
from typing import TypedDict

from core.cfg import Config, ConfigDict


class StrategyYamlArgs(TypedDict, total=False):
    """One strategy's YAML args block (all keys optional). Used by strategy/*.py
    only as a type hint on the spec parameter; ConfigDict ignores it at runtime.
    """

    execute_horizon: int
    latency_k: int
    exp_weight_m: float
    sync_wait_ignore_gripper: bool
    sync_wait_threshold: float
    sync_wait_max_ticks: int


class StrategyYamlSpec(TypedDict):
    """One entry in ``inference_strategies``: type-name + the args block."""

    type: str
    args: StrategyYamlArgs


_COLLECTION_REQUIRED_COLUMNS = ("qpos", "eef", "action_qpos", "action_eef")


def load_config(path: str | Path) -> ConfigDict:
    """Load a .py config (with ``_base_`` inheritance) into a ConfigDict.

    Args:
        path: Filesystem path to a ``.py`` config file. ``_base_`` paths inside
            the file are resolved relative to the file's own directory.

    Returns:
        ConfigDict with dotted attribute access. ``inference_cfg.obs_space``
        and ``inference_cfg.action_space`` are JointState / EEFPose instances
        (not dicts). Derived fields (``log.log_dir`` fallback) are filled in.
    """
    p = Path(path).expanduser()
    cfg = Config.fromfile(str(p)).to_dict()
    cfg = ConfigDict(cfg)
    _normalize_eval_cfg(cfg)
    _normalize_rl_cfg(cfg)
    _coerce_spaces(cfg)
    _coerce_calibration(cfg, p)
    _coerce_calibration_poses(cfg)
    _apply_derived(cfg, p)
    _validate(cfg)
    _resolve_eval_checkpoints(cfg, p)
    _resolve_rl_policies(cfg, p)
    return cfg


def resolve_video_key(dataset_keys: ConfigDict | dict, cam_key: str) -> str | None:
    """cam_key -> dataset video column name, or None when the dataset omits this camera.

    Empty ``video_keys`` dict falls back to the ``observation.images.{cam}``
    LeRobot v2.1 convention. Free-function form of the old
    ``DatasetKeyMapping.resolve_video_key`` method.
    """
    video_keys = dataset_keys.get("video_keys") or {}
    if video_keys:
        return video_keys.get(cam_key)
    return f"observation.images.{cam_key}"


def _normalize_eval_cfg(cfg: ConfigDict) -> None:
    """Expose the authored eval_cfg block through the runtime's legacy cfg.eval name."""
    eval_cfg = cfg.get("eval_cfg")
    legacy_eval = cfg.get("eval")
    if legacy_eval not in (None, {}):
        if eval_cfg not in (None, {}) and eval_cfg != legacy_eval:
            raise ValueError("config defines both eval and eval_cfg with different values")
        eval_cfg = legacy_eval
    if eval_cfg == {}:
        eval_cfg = None
    cfg.eval_cfg = eval_cfg
    cfg.eval = eval_cfg


def _normalize_rl_cfg(cfg: ConfigDict) -> None:
    """Expose the authored rl_cfg block through cfg.rl."""
    rl_cfg = cfg.get("rl_cfg")
    if rl_cfg == {}:
        rl_cfg = None
    cfg.rl_cfg = rl_cfg
    cfg.rl = rl_cfg


def _coerce_spaces(cfg: ConfigDict) -> None:
    """Replace obs_space / action_space dicts with JointState / EEFPose instances.

    No-op when the config has no inference_cfg section (e.g. ckpts/*.py which list
    checkpoints only and inherit from no defaults).
    """
    icfg = cfg.get("inference_cfg")
    if not icfg:
        return
    # Deferred import: core.app.handlers depends on core.config, so a module-level
    # import here would form a cycle. build_space is only needed at load time.
    from core.app.handlers.space import build_space

    if "obs_space" in icfg:
        icfg.obs_space = build_space(icfg.obs_space)
    if "action_space" in icfg:
        icfg.action_space = build_space(icfg.action_space)


def _coerce_calibration(cfg: ConfigDict, path: Path) -> None:
    """Materialize ``calibration.cameras`` into ``CameraCalibration`` instances.

    Two sources compose deterministically: if ``calibration.results_path`` is set,
    load that raiden-schema calibration_results.json first; then overlay the
    inline ``calibration.cameras`` dict per-camera-name. Inline entries win field-
    by-field, so an operator can pin one camera's ``attach_link`` without re-
    stating K/dist. No-op when the config carries no ``calibration`` section.
    """
    calib = cfg.get("calibration")
    if calib is None:
        return
    # Deferred import — same cycle-break trick as _coerce_spaces (build_space).
    import json as _json

    import numpy as np

    from robots.base import CameraCalibration

    merged: dict[str, dict] = {}
    results_path = calib.get("results_path")
    if results_path:
        rp = Path(results_path).expanduser()
        if not rp.is_absolute():
            rp = (path.parent / rp).resolve()
        if rp.exists():
            raw = _json.loads(rp.read_text())
            for name, entry in (raw.get("cameras") or {}).items():
                merged[name] = dict(entry)
    for name, entry in (calib.get("cameras") or {}).items():
        merged.setdefault(name, {}).update(dict(entry))

    def _to_np(value: object) -> np.ndarray | None:
        if value is None:
            return None
        return np.asarray(value, dtype=np.float64)

    coerced: dict[str, CameraCalibration] = {}
    for name, entry in merged.items():
        K = _to_np(entry.get("K"))
        dist = _to_np(entry.get("dist"))
        if K is None or dist is None:
            raise ValueError(f"calibration.cameras.{name} missing required K/dist")
        if K.shape != (3, 3):
            raise ValueError(
                f"calibration.cameras.{name}.K must be 3x3, got {K.shape}"
            )
        T = _to_np(entry.get("T_cam_link"))
        if T is not None and T.shape != (4, 4):
            raise ValueError(
                f"calibration.cameras.{name}.T_cam_link must be 4x4, got {T.shape}"
            )
        size = entry.get("image_size")
        coerced[name] = CameraCalibration(
            name=name,
            K=K,
            dist=dist.reshape(-1),
            attach_link=str(entry.get("attach_link") or ""),
            T_cam_link=T,
            image_size=(int(size[0]), int(size[1])) if size else None,
        )
    calib.cameras = coerced


def _coerce_calibration_poses(cfg: ConfigDict) -> None:
    """Convert ``calibration_poses`` list entries into numpy arrays.

    Empty means "use the robot's default_calibration_poses() at runtime"; the
    CALIBRATE session resolver handles that fallback. Anything else must be a
    list of qpos vectors; we normalize to a tuple of float32 np.ndarrays.
    """
    import numpy as np

    raw = cfg.get("calibration_poses")
    if raw is None:
        cfg.calibration_poses = ()
        return
    if not raw:
        cfg.calibration_poses = ()
        return
    coerced: list[np.ndarray] = []
    for i, entry in enumerate(raw):
        arr = np.asarray(entry, dtype=np.float32).reshape(-1)
        if arr.ndim != 1:
            raise ValueError(f"calibration_poses[{i}] must be a 1-D qpos vector")
        coerced.append(arr)
    cfg.calibration_poses = tuple(coerced)


def _apply_derived(cfg: ConfigDict, path: Path) -> None:
    """Compute fields that depend on the source file path or on other fields.

    Safe on partial configs: sections are accessed with .get() and skipped when missing.
    A collection config with an empty storage.log_dir falls back to <work_dir>/<stem>.
    """
    work_dir = cfg.get("work_dir") or "work_dirs"
    coll = cfg.get("collection") or {}
    schema = coll.get("schema") or {}
    if (schema.get("columns") or {}) and not (coll.get("storage") or {}).get("log_dir"):
        cfg.collection.storage["log_dir"] = str(Path(work_dir) / path.stem)


def _validate(cfg: ConfigDict) -> None:
    """Validate configured collection and RL storage contracts."""
    coll = cfg.get("collection") or {}
    schema = coll.get("schema") or {}
    columns = set(schema.get("columns") or {})
    if columns:
        missing = sorted(set(_COLLECTION_REQUIRED_COLUMNS) - columns)
        if missing:
            raise ValueError(f"collection.schema.columns missing required keys: {missing}")
        if not (schema.get("cameras") or {}):
            raise ValueError("collection.schema.cameras must define at least one camera")
        if not (schema.get("arms") or {}):
            raise ValueError("collection.schema.arms must define at least one arm")

    rl_cfg = cfg.get("rl_cfg")
    if rl_cfg and str(rl_cfg.data.format) != "lerobot":
        raise ValueError("rl.data.format must be 'lerobot' in this version")


def _resolve_eval_checkpoints(cfg: ConfigDict, path: Path) -> None:
    """Load an eval config's inline checkpoint list into full ConfigDicts.

    For an eval config (``cfg.eval_cfg`` non-empty with a ``checkpoints`` list), each
    checkpoint entry carries a ``config`` string pointing at a deploy preset. This
    loads that preset into a full ConfigDict (recursively via ``load_config``),
    overrides its policy endpoint with the checkpoint's ``host``/``port``, and
    replaces the ``config`` string in place — run loop deep-copies it per ckpt swap.

    The ``eval_cfg.ssh`` block (host/user/port/remote_sync_dir) is left untouched —
    consumers pass it straight to the FUNCTIONS-registered forwarder.

    No-op when the config has no eval_cfg section or lists no checkpoints (e.g. deploy
    presets loaded standalone, or recursively as a checkpoint's own config).

    Args:
        cfg: Loaded config; mutated in place.
        path: Source file path; checkpoint ``config`` refs resolve relative to its dir.
    """
    eval_cfg = cfg.get("eval_cfg")
    if not eval_cfg or not eval_cfg.get("checkpoints"):
        return

    for ckpt in eval_cfg["checkpoints"]:
        ref = str(ckpt["config"])
        sub = load_config((path.parent / ref).resolve())
        sub.policy.host = str(ckpt.get("host", "127.0.0.1"))
        sub.policy.port = int(ckpt["port"])
        ckpt["config"] = sub


def _resolve_rl_policies(cfg: ConfigDict, path: Path) -> None:
    """Load each RL policy choice into a full deploy ConfigDict."""
    rl_cfg = cfg.get("rl_cfg")
    if not rl_cfg:
        return
    for model in rl_cfg.policies:
        ref = model.config
        if isinstance(ref, ConfigDict):
            continue
        sub = load_config((path.parent / str(ref)).resolve())
        if "host" in model:
            sub.policy.host = str(model.host)
        if "port" in model:
            sub.policy.port = int(model.port)
        model.config = sub


__all__ = [
    "ConfigDict",
    "StrategyYamlArgs",
    "StrategyYamlSpec",
    "load_config",
    "resolve_video_key",
]
