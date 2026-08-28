"""Input-source-neutral contract between teleop clients and the EVA application."""

from __future__ import annotations

import dataclasses
import enum
from typing import Protocol, TypeAlias

import numpy as np


class TeleopClientError(RuntimeError):
    """Raised when a client cannot safely continue after producing motion."""


@dataclasses.dataclass(frozen=True)
class TeleopContext:
    """Robot feedback supplied to a client for one application control tick."""

    measured_qpos: np.ndarray
    measured_eef: np.ndarray
    now: float


@dataclasses.dataclass(frozen=True)
class CanonicalEefCommand:
    """Absolute canonical EEF targets, one 8D pose per configured arm."""

    value: np.ndarray
    active_arms: tuple[bool, ...] = ()


@dataclasses.dataclass(frozen=True)
class QposCommand:
    """Absolute robot joint target produced directly by a teleop client."""

    value: np.ndarray


TeleopCommand: TypeAlias = CanonicalEefCommand | QposCommand


class TeleopResultKind(str, enum.Enum):
    """Input-source-neutral outcome of one client poll."""

    COMMAND = "command"
    IDLE = "idle"
    REJECTED = "rejected"


@dataclasses.dataclass(frozen=True)
class TeleopInputToken:
    """Opaque health epoch plus the source-frame timestamp for one poll result."""

    generation: int
    frame_received_at: float | None = None

    def __post_init__(self) -> None:
        if self.generation < 0:
            raise ValueError("teleop input generation must be non-negative")
        if self.frame_received_at is not None and self.frame_received_at < 0.0:
            raise ValueError("teleop source frame timestamp must be non-negative")


@dataclasses.dataclass(frozen=True)
class TeleopResult:
    """One latest-only client result with an explicit non-fatal outcome."""

    kind: TeleopResultKind
    command: TeleopCommand | None = None
    message: str = ""
    source_token: TeleopInputToken | None = None

    def __post_init__(self) -> None:
        has_command = self.command is not None
        if (self.kind is TeleopResultKind.COMMAND) != has_command:
            raise ValueError("teleop command outcome must contain exactly one command")
        if self.kind is TeleopResultKind.REJECTED:
            if not self.message.strip():
                raise ValueError("teleop rejected outcome requires a message")
        elif self.message:
            raise ValueError("teleop result message is only valid for rejected outcomes")

    @classmethod
    def idle(cls, *, source_token: TeleopInputToken | None = None) -> TeleopResult:
        return cls(TeleopResultKind.IDLE, source_token=source_token)

    @classmethod
    def rejected(
        cls,
        message: str,
        *,
        source_token: TeleopInputToken | None = None,
    ) -> TeleopResult:
        return cls(TeleopResultKind.REJECTED, message=message, source_token=source_token)

    @classmethod
    def from_command(
        cls,
        command: TeleopCommand,
        *,
        source_token: TeleopInputToken | None = None,
    ) -> TeleopResult:
        return cls(TeleopResultKind.COMMAND, command=command, source_token=source_token)


@dataclasses.dataclass(frozen=True)
class TeleopOperatorEvent:
    """Reliable high-level operator intent normalized by a teleop client."""

    session_id: str
    event_id: int
    intent: str
    received_at: float


@dataclasses.dataclass(frozen=True)
class TeleopStatus:
    """Input-source-neutral health and engagement snapshot for the Console."""

    source_type: str
    connected: bool
    input_age_ms: float | None = None
    engaged_groups: tuple[str, ...] = ()
    held_groups: tuple[str, ...] = ()
    authorized_groups: tuple[str, ...] = ()
    pressed_controls: tuple[str, ...] = ()
    hold_progress: tuple[tuple[str, float], ...] = ()
    neutral: bool = False
    source_error: str = ""


class TeleopClient(Protocol):
    """Lifecycle and polling contract implemented by every in-process client."""

    def start(self) -> None: ...

    def poll(self, context: TeleopContext) -> TeleopResult: ...

    def validate_result(self, result: TeleopResult) -> bool: ...

    def reset(self, *, require_neutral: bool = False) -> None: ...

    def drain_events(self) -> tuple[TeleopOperatorEvent, ...]: ...

    def acknowledge_event(
        self,
        event: TeleopOperatorEvent,
        *,
        accepted: bool,
        message: str,
    ) -> None: ...

    def status(self, now: float | None = None) -> TeleopStatus: ...

    def close(self) -> None: ...


__all__ = [
    "CanonicalEefCommand",
    "QposCommand",
    "TeleopClient",
    "TeleopClientError",
    "TeleopCommand",
    "TeleopContext",
    "TeleopInputToken",
    "TeleopOperatorEvent",
    "TeleopResult",
    "TeleopResultKind",
    "TeleopStatus",
]
