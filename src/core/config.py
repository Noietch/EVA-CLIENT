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

import csv
import math
import posixpath
import re
from pathlib import Path, PurePosixPath
from typing import TypedDict

from core.cfg import Config, ConfigDict
from core.utils.s3_upload import S3UploadConfig


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
_TELEOP_CONTROL_SOURCES = frozenset({"transport", "client"})
_ROLLOUT_INTERVENTION_SOURCES = frozenset({"transport", "teleop_client"})
_CONSOLE_INITIAL_TABS = frozenset(
    {"auto", "debug", "manual", "collect", "replay", "rl", "eval", "result"}
)
_SCENE_DATASET_RE = re.compile(
    r"^(?P<base>.+)_scene_(?P<index>[1-9][0-9]*)(?:_(?P<date>[0-9]{8}))?$"
)
_PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _resolve_collection_task_set_path(path: str | Path) -> Path:
    root = Path(path).expanduser()
    if not root.is_absolute():
        root = _PROJECT_ROOT / root
    return root.resolve()


def load_collection_task_set(
    path: str | Path,
    dataset_name: str | None = None,
) -> dict[str, list[tuple[str, int]]]:
    """Load normalized collection tasks from a task-set directory."""
    root = _resolve_collection_task_set_path(path)
    tasks_path = root / "tasks.csv"
    if not tasks_path.is_file():
        raise FileNotFoundError(f"collection task-set is missing {tasks_path}")

    prompt_targets: dict[str, int] = {}
    try:
        with tasks_path.open("r", encoding="utf-8-sig", newline="") as handle:
            rows = csv.DictReader(handle)
            required = {"prompt_en", "total_epsiodes_count"}
            missing = sorted(required - set(rows.fieldnames or ()))
            if missing:
                raise ValueError(f"{tasks_path} is missing required columns: {', '.join(missing)}")
            for row_number, row in enumerate(rows, start=2):
                prompt = str(row.get("prompt_en", "") or "").strip()
                if not prompt:
                    raise ValueError(f"{tasks_path}:{row_number} prompt_en must not be empty")
                try:
                    target = int(str(row.get("total_epsiodes_count", "") or "").strip())
                except ValueError as error:
                    raise ValueError(
                        f"{tasks_path}:{row_number} total_epsiodes_count must be an integer"
                    ) from error
                previous_target = prompt_targets.get(prompt)
                if previous_target is None:
                    prompt_targets[prompt] = target
                elif previous_target == -1 or target == -1:
                    prompt_targets[prompt] = -1
                else:
                    prompt_targets[prompt] = previous_target + target
    except (OSError, UnicodeError, csv.Error) as error:
        raise ValueError(f"unable to read collection task-set {tasks_path}") from error

    if not prompt_targets:
        raise ValueError(f"{tasks_path} must contain at least one task")
    name = str(dataset_name or root.name).strip()
    if not _is_safe_dataset_name_component(name):
        raise ValueError("collection task-set dataset_name must be a safe path component")
    return {name: list(prompt_targets.items())}


def _is_safe_dataset_name_component(value: str) -> bool:
    candidate = value.strip()
    return bool(candidate) and Path(candidate).name == candidate and candidate not in {".", ".."}


def _is_safe_posix_directory(value: str) -> bool:
    path = PurePosixPath(value)
    normalized = posixpath.normpath(value)
    canonical = value == "/" or normalized == value.rstrip("/")
    return path.is_absolute() and canonical and all(part not in {".", ".."} for part in path.parts)


def _is_safe_loopback_directory(value: str) -> bool:
    path = Path(value)
    if (
        not value
        or path == Path(".")
        or any(char in value for char in "\r\n")
        or any(part in {".", ".."} for part in path.parts)
        or path.as_posix() != value.rstrip("/")
    ):
        return False
    return path.expanduser().resolve() != Path("/")


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
    _normalize_collection_task_set(cfg)
    _coerce_spaces(cfg)
    _apply_derived(cfg, p)
    _validate(cfg)
    _resolve_eval_checkpoints(cfg, p)
    _resolve_rl_policies(cfg, p)
    return cfg


def _normalize_collection_task_set(cfg: ConfigDict) -> None:
    """Load a mounted task set; otherwise retain configured tasks."""
    collection = cfg.get("collection") or {}
    task_set_dir = str(collection.get("task_set_dir", "") or "").strip()
    if not task_set_dir:
        return
    root = _resolve_collection_task_set_path(task_set_dir)
    if not root.is_dir():
        return
    authored_tasks = collection.get("tasks") or {}
    dataset_name = str(collection.get("task_set_name", "") or "").strip()
    if not dataset_name and len(authored_tasks) == 1:
        dataset_name = str(next(iter(authored_tasks)))
    collection["tasks"] = load_collection_task_set(root, dataset_name or None)


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


def _apply_derived(cfg: ConfigDict, path: Path) -> None:
    """Compute fields that depend on the source file path or on other fields.

    Safe on partial configs: sections are accessed with .get() and skipped when missing.
    A collection config with an empty storage.log_dir falls back to <work_dir>/<stem>.
    """
    work_dir = cfg.get("work_dir") or "work_dirs"
    coll = cfg.get("collection") or {}
    _validate_teleop(cfg)
    schema = coll.get("schema") or {}
    if (schema.get("columns") or {}) and not (coll.get("storage") or {}).get("log_dir"):
        cfg.collection.storage["log_dir"] = str(Path(work_dir) / path.stem)


def _validate(cfg: ConfigDict) -> None:
    """Validate configured transport, collection, and RL storage contracts."""
    transport = cfg.get("transport") or {}
    image_mode = str(transport.get("image_mode", "stream"))
    if image_mode not in {"stream", "on_demand"}:
        raise ValueError(
            f"transport.image_mode must be 'stream' or 'on_demand', got {image_mode!r}"
        )
    timeout = float(transport.get("image_request_timeout_s", 2.0))
    if timeout <= 0.0:
        raise ValueError("transport.image_request_timeout_s must be positive")

    console = cfg.get("console") or {}
    initial_tab = str(console.get("initial_tab", "auto"))
    if initial_tab not in _CONSOLE_INITIAL_TABS:
        raise ValueError(
            "console.initial_tab must be one of "
            f"{sorted(_CONSOLE_INITIAL_TABS)}, got {initial_tab!r}"
        )

    coll = cfg.get("collection") or {}
    schema = coll.get("schema") or {}
    tasks = coll.get("tasks", {})
    if tasks is None:
        tasks = {}
    if not isinstance(tasks, dict):
        raise ValueError("collection.tasks must map dataset names to (prompt, target) lists")
    prompt_datasets: dict[str, str] = {}
    for dataset_name, prompts in tasks.items():
        normalized_name = str(dataset_name).strip()
        if not _is_safe_dataset_name_component(normalized_name):
            raise ValueError("collection.tasks dataset names must be non-empty path components")
        if not isinstance(prompts, (list, tuple)) or not prompts:
            raise ValueError(f"collection.tasks.{dataset_name} must be a non-empty prompt list")
        for entry in prompts:
            if not isinstance(entry, (list, tuple)) or len(entry) != 2:
                raise ValueError(
                    f"collection.tasks.{dataset_name} entries must be (prompt, target) pairs"
                )
            prompt, target = entry
            if not isinstance(prompt, str):
                raise ValueError(f"collection.tasks.{dataset_name} prompts must be strings")
            normalized_prompt = str(prompt).strip()
            if not normalized_prompt:
                raise ValueError(f"collection.tasks.{dataset_name} prompts must not be empty")
            previous_dataset = prompt_datasets.get(normalized_prompt)
            if previous_dataset is not None:
                previous_scene = _SCENE_DATASET_RE.fullmatch(previous_dataset)
                current_scene = _SCENE_DATASET_RE.fullmatch(normalized_name)
                same_scene_family = (
                    previous_scene is not None
                    and current_scene is not None
                    and previous_scene.group("base") == current_scene.group("base")
                    and previous_dataset != normalized_name
                )
                if not same_scene_family:
                    raise ValueError(
                        f"collection prompt must belong to one dataset: {normalized_prompt!r}"
                    )
            else:
                prompt_datasets[normalized_prompt] = normalized_name
            if (
                isinstance(target, bool)
                or not isinstance(target, int)
                or target == 0
                or target < -1
            ):
                raise ValueError(
                    f"collection.tasks.{dataset_name} target must be -1 or a positive integer"
                )
    if "task_requirements" in coll:
        raise ValueError(
            "collection.task_requirements was removed; put each target beside its prompt"
        )
    sftp = (coll.get("storage") or {}).get("sftp") or {}
    if sftp:
        if not str(sftp.get("host", "")).strip():
            raise ValueError("collection.storage.sftp.host must not be empty")
        try:
            sftp_port = int(sftp.get("port", 22))
        except (TypeError, ValueError) as error:
            raise ValueError("collection.storage.sftp.port must be an integer") from error
        if not 1 <= sftp_port <= 65535:
            raise ValueError("collection.storage.sftp.port must be in [1, 65535]")
        remote_dir = str(sftp.get("remote_dir", "")).strip()
        if not _is_safe_posix_directory(remote_dir):
            raise ValueError("collection.storage.sftp.remote_dir must be a canonical absolute path")
    s3 = (coll.get("storage") or {}).get("s3") or {}
    if s3:
        try:
            S3UploadConfig(
                endpoint=str(s3.get("endpoint", "")),
                bucket=str(s3.get("bucket", "")),
                prefix=str(s3.get("prefix", "")),
                sign_service=str(s3.get("sign_service", "")),
                secure=s3.get("secure", False),
            )
        except ValueError as error:
            raise ValueError(f"collection.storage.s3: {error}") from error
    loopback = (coll.get("storage") or {}).get("loopback") or {}
    if loopback:
        remote_dir = str(loopback.get("remote_dir", "")).strip()
        if not _is_safe_loopback_directory(remote_dir):
            raise ValueError(
                "collection.storage.loopback.remote_dir must be a canonical non-root path"
            )
    rl_cfg = cfg.get("rl_cfg")
    columns = set(schema.get("columns") or {})
    if columns:
        missing = sorted(set(_COLLECTION_REQUIRED_COLUMNS) - columns)
        if missing:
            raise ValueError(f"collection.schema.columns missing required keys: {missing}")
        if not (schema.get("cameras") or {}):
            raise ValueError("collection.schema.cameras must define at least one camera")
        if not (schema.get("arms") or {}):
            raise ValueError("collection.schema.arms must define at least one arm")
        storage = coll.get("storage") or {}
        image_skew_tolerance = storage.get("image_skew_tolerance_sec")
        if image_skew_tolerance is not None:
            _positive_finite(
                image_skew_tolerance,
                "collection.storage.image_skew_tolerance_sec",
            )

    if rl_cfg and str(rl_cfg.data.format) != "lerobot":
        raise ValueError("rl.data.format must be 'lerobot' in this version")

    # Validate rollout intervention input source
    for field, section in (("rollout", cfg.get("rollout") or {}), ("rl", rl_cfg or {})):
        intervention = section.get("intervention") or {}
        source = intervention.get("source")
        if not source:
            continue
        source = str(source)
        if source not in _ROLLOUT_INTERVENTION_SOURCES:
            raise ValueError(
                f"{field}.intervention.source must be one of "
                f"{sorted(_ROLLOUT_INTERVENTION_SOURCES)}, got {source!r}"
            )
        if source == "teleop_client" and str((coll.get("teleop") or {}).get("control_source")) != (
            "client"
        ):
            raise ValueError(
                f"{field}.intervention.source='teleop_client' requires "
                "collection.teleop.control_source='client'"
            )


def _positive_finite(value: object, field: str) -> float:
    try:
        parsed = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError) as error:
        raise ValueError(f"{field} must be numeric") from error
    if not math.isfinite(parsed) or parsed <= 0.0:
        raise ValueError(f"{field} must be positive and finite")
    return parsed


def _teleop_arm_group_names(cfg: ConfigDict) -> tuple[str, ...]:
    collection = cfg.get("collection") or {}
    schema_arm_names = tuple((collection.get("schema") or {}).get("arms") or {})
    if schema_arm_names:
        return schema_arm_names
    robot_type = str((cfg.get("robot") or {}).get("type", "")).strip()
    if not robot_type:
        raise ValueError("client-driven teleop requires robot.type")
    import robots  # noqa: F401
    from core.registry import ROBOT_REGISTRY

    robot = ROBOT_REGISTRY.build(robot_type)
    return tuple(group.name for group in robot.arm_groups)


def _validate_teleop(cfg: ConfigDict) -> None:
    """Validate the input-source selection without importing optional VR code eagerly."""
    collection = cfg.get("collection") or {}
    teleop = collection.get("teleop") or {}
    source = str(teleop.get("control_source", "transport"))
    if source not in _TELEOP_CONTROL_SOURCES:
        raise ValueError(
            "collection.teleop.control_source must be one of "
            f"{sorted(_TELEOP_CONTROL_SOURCES)}, got {source!r}"
        )
    client = teleop.get("client") or {}
    if source == "transport":
        if client:
            raise ValueError("transport-driven teleop cannot configure collection.teleop.client")
        return
    if teleop.get("type"):
        raise ValueError("client-driven teleop must use client={...}, not legacy teleop.type")
    from teleop_client import validate_client_config

    validate_client_config(client, arm_group_names=_teleop_arm_group_names(cfg))
    safety = teleop.get("safety") or {}
    for name in (
        "max_qpos_step",
        "max_position_error_m",
        "max_orientation_error_rad",
    ):
        _positive_finite(safety.get(name), f"collection.teleop.safety.{name}")


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
