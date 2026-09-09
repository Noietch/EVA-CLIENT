"""Relative retargeting from normalized VR controllers to canonical EEF poses."""

from __future__ import annotations

import dataclasses

import numpy as np
from scipy.spatial.transform import Rotation


@dataclasses.dataclass(frozen=True)
class VrPose:
    """An absolute controller pose expressed in the XR reference space."""

    position: np.ndarray
    orientation_xyzw: np.ndarray


@dataclasses.dataclass(frozen=True)
class VrControllerState:
    hand: str
    valid: bool
    grip_engaged: bool
    pose: VrPose | None
    trigger: float = 0.0
    profiles: tuple[str, ...] = ()


@dataclasses.dataclass(frozen=True)
class WorkspaceBounds:
    minimum: np.ndarray
    maximum: np.ndarray

    def __post_init__(self) -> None:
        minimum = np.asarray(self.minimum, dtype=np.float64)
        maximum = np.asarray(self.maximum, dtype=np.float64)
        if minimum.shape != (3,) or maximum.shape != (3,):
            raise ValueError("workspace bounds must be 3D vectors")
        if not np.all(np.isfinite(minimum)) or not np.all(np.isfinite(maximum)):
            raise ValueError("workspace bounds must be finite")
        if np.any(minimum >= maximum):
            raise ValueError("workspace minimum must be below maximum")
        object.__setattr__(self, "minimum", minimum)
        object.__setattr__(self, "maximum", maximum)

    def contains(self, position: np.ndarray) -> bool:
        value = np.asarray(position, dtype=np.float64)
        return bool(np.all(value >= self.minimum) and np.all(value <= self.maximum))


@dataclasses.dataclass(frozen=True)
class GripperMapping:
    mode: str = "binary"
    threshold: float = 0.5
    open_value: float = 1.0
    close_value: float = 0.0

    def __post_init__(self) -> None:
        values = (self.threshold, self.open_value, self.close_value)
        if not all(np.isfinite(value) for value in values):
            raise ValueError("gripper mapping values must be finite")
        if not 0.0 <= self.threshold <= 1.0:
            raise ValueError("gripper threshold must be in [0, 1]")
        if self.mode not in {"binary", "analog", "linear", "toggle"}:
            raise ValueError(f"unsupported VR gripper mode: {self.mode!r}")

    def map(self, trigger: float) -> float:
        if self.mode == "binary":
            return self.close_value if trigger >= self.threshold else self.open_value
        if self.mode in {"analog", "linear"}:
            return self.open_value + trigger * (self.close_value - self.open_value)
        raise ValueError("toggle gripper mapping requires retargeter state")


@dataclasses.dataclass(frozen=True)
class ArmRetargetStatus:
    target_eef: np.ndarray
    latched: bool
    active: bool


class EefLowPassFilter:
    """Low-pass an EEF pose while keeping gripper commands responsive.

    ``alpha`` is the fraction of the new target applied per update.  A value of
    ``1`` disables smoothing; smaller values remove high-frequency controller
    jitter at the cost of a little more response latency.
    """

    def __init__(self, alpha: float = 1.0) -> None:
        if not np.isfinite(alpha) or not 0.0 < alpha <= 1.0:
            raise ValueError("EEF filter alpha must be finite and in (0, 1]")
        self._alpha = float(alpha)
        self._previous: np.ndarray | None = None

    def reset(self) -> None:
        self._previous = None

    def apply(self, target: np.ndarray) -> np.ndarray:
        value = np.asarray(target, dtype=np.float64)
        if value.shape != (8,) or not np.all(np.isfinite(value)):
            raise ValueError(f"EEF filter target must be finite shape (8,), got {value.shape}")
        if self._previous is None or self._alpha >= 1.0:
            filtered = value.copy()
        else:
            previous = self._previous
            filtered = previous.copy()
            filtered[:3] = previous[:3] + self._alpha * (value[:3] - previous[:3])
            quaternion = value[3:7].copy()
            if float(np.dot(quaternion, previous[3:7])) < 0.0:
                quaternion *= -1.0
            blended = (1.0 - self._alpha) * previous[3:7] + self._alpha * quaternion
            norm = float(np.linalg.norm(blended))
            filtered[3:7] = previous[3:7] if norm <= 1e-8 else blended / norm
            # Gripper transitions should not be delayed by pose smoothing.
            filtered[7] = value[7]
        self._previous = filtered.copy()
        return filtered.astype(np.float32)


class ArmRetargeter:
    """Accumulate grip-gated absolute XR motion into an EEF target."""

    def __init__(
        self,
        *,
        base_from_xr_rotation: np.ndarray,
        position_scale: float,
        gripper: GripperMapping,
        workspace: WorkspaceBounds | None = None,
        eef_filter_alpha: float = 1.0,
    ) -> None:
        rotation = np.asarray(base_from_xr_rotation, dtype=np.float64)
        if rotation.shape != (3, 3):
            raise ValueError(f"base_from_xr_rotation must be 3x3, got {rotation.shape}")
        if not np.all(np.isfinite(rotation)):
            raise ValueError("base_from_xr_rotation must be finite")
        if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-6):
            raise ValueError("base_from_xr_rotation must be orthonormal")
        if not np.isclose(np.linalg.det(rotation), 1.0, atol=1e-6):
            raise ValueError("base_from_xr_rotation must be a proper rotation")
        if not np.isfinite(position_scale) or position_scale <= 0.0:
            raise ValueError("position_scale must be positive and finite")
        self._base_from_xr = rotation
        self._position_scale = float(position_scale)
        self._gripper = gripper
        self._workspace = workspace
        self._eef_filter = EefLowPassFilter(eef_filter_alpha)
        self._home_eef: np.ndarray | None = None
        self._last_target: np.ndarray | None = None
        self._grip_engaged = False
        self._trigger_initialized = False
        self._trigger_pressed = False
        self._gripper_closed: bool | None = None
        self._gripper_target: float | None = None
        self._reference_position: np.ndarray | None = None
        self._reference_rotation: np.ndarray | None = None
        self._accumulated_position = np.zeros(3, dtype=np.float64)
        self._accumulated_rotation = np.eye(3, dtype=np.float64)

    def reset(self) -> None:
        self._home_eef = None
        self._last_target = None
        self._grip_engaged = False
        self._trigger_initialized = False
        self._trigger_pressed = False
        self._gripper_closed = None
        self._gripper_target = None
        self._reference_position = None
        self._reference_rotation = None
        self._accumulated_position = np.zeros(3, dtype=np.float64)
        self._accumulated_rotation = np.eye(3, dtype=np.float64)
        self._eef_filter.reset()

    @staticmethod
    def _measured(value: np.ndarray) -> np.ndarray:
        measured = np.asarray(value, dtype=np.float64)
        if measured.shape != (8,) or not np.all(np.isfinite(measured)):
            raise ValueError(f"measured_eef must be finite shape (8,), got {measured.shape}")
        return measured

    def _gripper_command(self, controller: VrControllerState, measured: float) -> float:
        if self._gripper.mode != "toggle":
            if controller.valid:
                return self._gripper.map(controller.trigger)
            if self._last_target is not None:
                return float(self._last_target[7])
            return measured
        if self._gripper_target is None:
            self._gripper_closed = abs(measured - self._gripper.close_value) < abs(
                measured - self._gripper.open_value
            )
            self._gripper_target = measured
        pressed = controller.trigger >= self._gripper.threshold
        rising = self._trigger_initialized and pressed and not self._trigger_pressed
        self._trigger_initialized = True
        self._trigger_pressed = pressed
        if controller.valid and rising:
            self._gripper_closed = not bool(self._gripper_closed)
            self._gripper_target = (
                self._gripper.close_value if self._gripper_closed else self._gripper.open_value
            )
        return self._gripper_target

    def update(self, controller: VrControllerState, measured_eef: np.ndarray) -> ArmRetargetStatus:
        measured = self._measured(measured_eef)
        gripper = self._gripper_command(controller, float(measured[7]))
        if not controller.grip_engaged:
            self._grip_engaged = False
            self._reference_position = None
            self._reference_rotation = None
            target = measured.copy() if self._last_target is None else self._last_target.copy()
            target[7] = gripper
            filtered = self._eef_filter.apply(target)
            self._last_target = filtered.astype(np.float64)
            return ArmRetargetStatus(filtered, latched=False, active=False)
        if not controller.valid or controller.pose is None:
            raise ValueError(f"VR {controller.hand} controller is not tracked")
        return self._retarget(controller, measured, gripper)

    def _retarget(
        self, controller: VrControllerState, measured: np.ndarray, gripper: float
    ) -> ArmRetargetStatus:
        assert controller.pose is not None
        position_xr = np.asarray(controller.pose.position, dtype=np.float64)
        rotation_xr = Rotation.from_quat(controller.pose.orientation_xyzw).as_matrix()
        # The grip long-press unlock is the calibration boundary. The first
        # authorized frame anchors controller motion to the robot's live pose.
        latched = self._home_eef is None
        if latched:
            self._home_eef = measured.copy()
        assert self._home_eef is not None
        if self._reference_position is None or self._reference_rotation is None:
            self._reference_position = position_xr.copy()
            self._reference_rotation = rotation_xr.copy()
        else:
            incremental_position = position_xr - self._reference_position
            incremental_rotation = rotation_xr @ self._reference_rotation.T
            self._accumulated_position += incremental_position
            self._accumulated_rotation = incremental_rotation @ self._accumulated_rotation
            self._reference_position = position_xr.copy()
            self._reference_rotation = rotation_xr.copy()
        self._grip_engaged = True
        target_position = self._home_eef[:3] + self._position_scale * (
            self._base_from_xr @ self._accumulated_position
        )
        if self._workspace is not None and not self._workspace.contains(target_position):
            raise ValueError(f"VR target outside workspace: {target_position.tolist()}")
        delta_rotation_base = self._base_from_xr @ self._accumulated_rotation @ self._base_from_xr.T
        home_wxyz = self._home_eef[3:7]
        home_rotation = Rotation.from_quat(
            [home_wxyz[1], home_wxyz[2], home_wxyz[3], home_wxyz[0]]
        ).as_matrix()
        qx, qy, qz, qw = Rotation.from_matrix(delta_rotation_base @ home_rotation).as_quat()
        target_quaternion = np.asarray([qw, qx, qy, qz], dtype=np.float64)
        if self._last_target is not None and np.dot(target_quaternion, self._last_target[3:7]) < 0:
            target_quaternion = -target_quaternion
        target = np.concatenate(
            [target_position, target_quaternion, np.asarray([gripper], dtype=np.float64)]
        ).astype(np.float32)
        filtered = self._eef_filter.apply(target)
        self._last_target = filtered.astype(np.float64)
        return ArmRetargetStatus(filtered, latched, active=True)


__all__ = [
    "ArmRetargetStatus",
    "ArmRetargeter",
    "EefLowPassFilter",
    "GripperMapping",
    "VrControllerState",
    "VrPose",
    "WorkspaceBounds",
]
