"""Thin adapter around the official ARX X5 SDK-V2 package."""

from __future__ import annotations

import time
from typing import Any


ARX_X5_2025_GRIPPER_MOTOR_ID = 8
ARX_X5_MOTOR_CLEAR_ERROR_COMMAND = b"\xff\xff\xff\xff\xff\xff\xff\xfb"


def clear_gripper_error(can_port: str, *, timeout_s: float = 1.0) -> dict[str, Any]:
    """Clear and verify the X5-2025 gripper motor fault without enabling arm joints."""
    from bimanual.hardware.can_bus import CANBus
    from bimanual.hardware.motors import MotorB

    bus = CANBus(str(can_port))
    try:
        bus.open()
        gripper = MotorB(ARX_X5_2025_GRIPPER_MOTOR_ID, bus)
        initial_rx_count = int(gripper.rx_count)
        if not bus.send(ARX_X5_2025_GRIPPER_MOTOR_ID, ARX_X5_MOTOR_CLEAR_ERROR_COMMAND):
            raise RuntimeError(f"failed to send gripper clear-error frame on {can_port}")

        deadline = time.monotonic() + max(float(timeout_s), 0.0)
        while int(gripper.rx_count) <= initial_rx_count and time.monotonic() < deadline:
            time.sleep(0.01)
        if int(gripper.rx_count) <= initial_rx_count:
            raise RuntimeError(f"no gripper feedback after clear-error frame on {can_port}")

        feedback = gripper.feedback
        error = int(feedback.error)
        error_name = str(gripper.decode_error(error))
        if error != 0:
            raise RuntimeError(
                f"gripper fault persists on {can_port}: error={error} ({error_name})"
            )
        return {
            "can_port": str(can_port),
            "error": error,
            "error_name": error_name,
            "position": float(feedback.position),
            "temperature": float(feedback.temperature),
        }
    finally:
        bus.close()


class SingleArm:
    """Expose the subset of ``bimanual.SingleArm`` used by the EVA bridge."""

    def __init__(self, config: dict[str, Any]) -> None:
        from bimanual import SingleArm as OfficialSingleArm

        self._arm = OfficialSingleArm(
            {
                "can_port": str(config.get("can_port", "can0")),
                "type": int(config.get("type", 0)),
            }
        )

    def get_joint_positions(self) -> Any:
        return self._arm.get_joint_positions()

    def get_joint_currents(self) -> Any:
        return self._arm.get_joint_currents()

    def get_status(self) -> dict[str, Any]:
        return dict(self._arm.get_status())

    @property
    def offline_joints(self) -> tuple[Any, ...]:
        joints = self._arm.offline_joints
        if joints is None:
            return ()
        try:
            return tuple(joints)
        except TypeError:
            return (joints,)

    @property
    def fault(self) -> Any | None:
        return self._arm.fault

    @property
    def gripper_error(self) -> int:
        return int(self._arm._ctrl.gripper.feedback.error)

    @property
    def gripper_error_name(self) -> str:
        gripper = self._arm._ctrl.gripper
        return str(gripper.decode_error(gripper.feedback.error))

    def get_gripper_pos(self) -> float:
        return float(self._arm.get_gripper_pos())

    def set_joint_positions(self, positions: list[float], *, duration: float = 0.0) -> bool:
        return bool(self._arm.set_joint_positions(positions, duration=duration))

    def set_gripper_pos(self, position: float, *, duration: float | None = None) -> bool:
        return bool(self._arm.set_gripper_pos(position, duration=duration))

    def go_home(
        self,
        duration: float = 2.0,
        *,
        wait: bool = False,
        gripper: bool = True,
    ) -> float:
        return float(self._arm.go_home(duration=duration, wait=wait, gripper=gripper))

    def protect_mode(self) -> bool:
        return bool(self._arm.protect_mode())

    def disable(self) -> bool:
        return bool(self._arm.disable())

    def close(self) -> None:
        self._arm.close()
