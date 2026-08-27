"""Input-source clients that produce canonical robot teleoperation commands."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from teleop_client.base import (
    CanonicalEefCommand,
    QposCommand,
    TeleopClient,
    TeleopClientError,
    TeleopContext,
    TeleopInputToken,
    TeleopOperatorEvent,
    TeleopResult,
    TeleopResultKind,
    TeleopStatus,
)


def _client_config(value: object) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError("collection.teleop.client must be an object")
    return value


def _client_type(config: Mapping[str, object]) -> str:
    client_type = str(config.get("type", "")).strip()
    if not client_type:
        raise ValueError("collection.teleop.client.type must not be empty")
    return client_type


def validate_client_config(
    config: object,
    *,
    arm_group_names: Sequence[str],
) -> None:
    """Validate one configured teleop client through its owning implementation."""
    normalized = _client_config(config)
    client_type = _client_type(normalized)
    if client_type == "vr_webxr":
        from teleop_client.vr import validate_client_config as validate_vr_config

        validate_vr_config(normalized, arm_group_names=arm_group_names)
        return
    raise ValueError(f"unsupported teleop client type: {client_type!r}")


def build_client(
    config: object,
    *,
    arm_group_names: Sequence[str],
) -> TeleopClient:
    """Build one teleop client without exposing implementation details to Core."""
    normalized = _client_config(config)
    client_type = _client_type(normalized)
    if client_type == "vr_webxr":
        from teleop_client.vr import build_client as build_vr_client

        return build_vr_client(normalized, arm_group_names=arm_group_names)
    raise ValueError(f"unsupported teleop client type: {client_type!r}")


__all__ = [
    "CanonicalEefCommand",
    "QposCommand",
    "TeleopClient",
    "TeleopClientError",
    "TeleopContext",
    "TeleopInputToken",
    "TeleopOperatorEvent",
    "TeleopResult",
    "TeleopResultKind",
    "TeleopStatus",
    "build_client",
    "validate_client_config",
]
