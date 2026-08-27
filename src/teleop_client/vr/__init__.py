"""VR implementation of the generic teleop client contract."""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping, Sequence
from typing import Any, cast

import numpy as np

from teleop_client.base import TeleopClient
from teleop_client.vr.client import VrTeleopClient


@dataclasses.dataclass(frozen=True)
class _VrClientConfig:
    endpoint: str
    ack_endpoint: str
    input_timeout_s: float
    heartbeat_timeout_s: float
    arms: tuple[dict[str, object], ...]


def _mapping(value: object, field: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{field} must be an object")
    return cast(Mapping[str, object], value)


def _finite(value: object, field: str) -> float:
    try:
        parsed = float(cast(Any, value))
    except (TypeError, ValueError) as error:
        raise ValueError(f"{field} must be numeric") from error
    if not np.isfinite(parsed):
        raise ValueError(f"{field} must be finite")
    return parsed


def _positive(value: object, field: str) -> float:
    parsed = _finite(value, field)
    if parsed <= 0.0:
        raise ValueError(f"{field} must be positive")
    return parsed


def _filter_alpha(value: object, field: str) -> float:
    parsed = _finite(value, field)
    if not 0.0 < parsed <= 1.0:
        raise ValueError(f"{field} must be in (0, 1]")
    return parsed


def _rotation(value: object, field: str) -> np.ndarray:
    try:
        rotation = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{field} must be a proper 3x3 rotation") from error
    if (
        rotation.shape != (3, 3)
        or not np.all(np.isfinite(rotation))
        or not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-6)
        or not np.isclose(np.linalg.det(rotation), 1.0, atol=1e-6)
    ):
        raise ValueError(f"{field} must be a proper 3x3 rotation")
    return rotation


def _gripper(value: object, field: str) -> dict[str, object]:
    raw = _mapping(value, field)
    mode = str(raw.get("mode", ""))
    if mode not in {"binary", "analog", "linear", "toggle"}:
        raise ValueError(f"{field}.mode is invalid")
    threshold = _finite(raw.get("threshold", 0.5), f"{field}.threshold")
    if not 0.0 <= threshold <= 1.0:
        raise ValueError(f"{field}.threshold must be in [0, 1]")
    return {
        "mode": mode,
        "threshold": threshold,
        "open_value": _finite(raw.get("open_value"), f"{field}.open_value"),
        "close_value": _finite(raw.get("close_value"), f"{field}.close_value"),
    }


def _workspace(value: object, field: str) -> dict[str, object] | None:
    if value is None or (isinstance(value, Mapping) and not value):
        return None
    raw = _mapping(value, field)
    try:
        minimum = np.asarray(raw.get("min"), dtype=np.float64)
        maximum = np.asarray(raw.get("max"), dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{field} bounds are invalid") from error
    if (
        minimum.shape != (3,)
        or maximum.shape != (3,)
        or not np.all(np.isfinite(minimum))
        or not np.all(np.isfinite(maximum))
        or np.any(minimum >= maximum)
    ):
        raise ValueError(f"{field} bounds are invalid")
    return {"min": minimum, "max": maximum}


def _parse_client_config(
    config: Mapping[str, object],
    *,
    arm_group_names: Sequence[str],
) -> _VrClientConfig:
    endpoint = str(config.get("endpoint", ""))
    ack_endpoint = str(config.get("ack_endpoint", ""))
    if not endpoint.startswith("tcp://") or not ack_endpoint.startswith("tcp://"):
        raise ValueError("VR client endpoint and ack_endpoint must use tcp://")
    if endpoint == ack_endpoint:
        raise ValueError("VR client endpoint and ack_endpoint must differ")

    shared_rotation = _rotation(
        config.get("base_from_xr_rotation"),
        "collection.teleop.client.base_from_xr_rotation",
    )
    shared_position_scale = _positive(
        config.get("position_scale"),
        "collection.teleop.client.position_scale",
    )
    shared_filter_alpha = _filter_alpha(
        config.get("eef_filter_alpha", 1.0),
        "collection.teleop.client.eef_filter_alpha",
    )
    shared_gripper = _gripper(
        config.get("gripper") or {},
        "collection.teleop.client.gripper",
    )

    arms_raw = _mapping(config.get("arms") or {}, "collection.teleop.client.arms")
    if not arms_raw:
        raise ValueError("collection.teleop.client.arms must not be empty")
    arms_by_name = {str(name): value for name, value in arms_raw.items()}
    configured_names = tuple(arms_by_name)
    expected_names = tuple(str(name) for name in arm_group_names)
    if len(set(expected_names)) != len(expected_names):
        raise ValueError("teleop arm group names must be unique")
    if expected_names and set(configured_names) != set(expected_names):
        raise ValueError(
            "collection.teleop.client.arms must match expected arm groups: "
            f"expected {sorted(expected_names)}, got {sorted(configured_names)}"
        )

    ordered_names = expected_names or configured_names
    hands: list[str] = []
    arms: list[dict[str, object]] = []
    for group_name in ordered_names:
        field = f"collection.teleop.client.arms.{group_name}"
        raw = _mapping(arms_by_name[group_name], field)
        hand = str(raw.get("controller", ""))
        if hand not in {"left", "right"}:
            raise ValueError(f"{field}.controller must be 'left' or 'right'")
        hands.append(hand)
        arm_gripper = _mapping(raw.get("gripper") or {}, f"{field}.gripper")
        arms.append(
            {
                "group_name": group_name,
                "controller": hand,
                "workspace": _workspace(raw.get("workspace"), f"{field}.workspace"),
                "base_from_xr_rotation": _rotation(
                    raw.get("base_from_xr_rotation", shared_rotation),
                    f"{field}.base_from_xr_rotation",
                ),
                "position_scale": _positive(
                    raw.get("position_scale", shared_position_scale),
                    f"{field}.position_scale",
                ),
                "eef_filter_alpha": _filter_alpha(
                    raw.get("eef_filter_alpha", shared_filter_alpha),
                    f"{field}.eef_filter_alpha",
                ),
                "gripper": _gripper(
                    {**shared_gripper, **arm_gripper},
                    f"{field}.gripper",
                ),
            }
        )
    if len(set(hands)) != len(hands):
        raise ValueError("VR arm controller bindings must be unique")

    return _VrClientConfig(
        endpoint=endpoint,
        ack_endpoint=ack_endpoint,
        input_timeout_s=_positive(
            config.get("input_timeout_s"),
            "collection.teleop.client.input_timeout_s",
        ),
        heartbeat_timeout_s=_positive(
            config.get("heartbeat_timeout_s", 2.0),
            "collection.teleop.client.heartbeat_timeout_s",
        ),
        arms=tuple(arms),
    )


def validate_client_config(
    config: Mapping[str, object],
    *,
    arm_group_names: Sequence[str],
) -> None:
    """Validate and normalize all VR-owned configuration fields."""
    _parse_client_config(config, arm_group_names=arm_group_names)


def build_client(
    config: Mapping[str, object],
    *,
    arm_group_names: Sequence[str],
) -> TeleopClient:
    """Build a VR client from the same parser used during config validation."""
    parsed = _parse_client_config(config, arm_group_names=arm_group_names)
    return VrTeleopClient(
        endpoint=parsed.endpoint,
        ack_endpoint=parsed.ack_endpoint,
        input_timeout_s=parsed.input_timeout_s,
        heartbeat_timeout_s=parsed.heartbeat_timeout_s,
        arms=parsed.arms,
    )


__all__ = ["VrTeleopClient", "build_client", "validate_client_config"]
