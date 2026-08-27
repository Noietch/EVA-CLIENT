from __future__ import annotations

import sys
from types import ModuleType, SimpleNamespace

from examples.hardware.x5 import sdk as x5_sdk


def test_clear_gripper_error_uses_only_2025_gripper_motor(monkeypatch) -> None:
    events: list[object] = []

    class FakeBus:
        def __init__(self, channel: str) -> None:
            self.channel = channel
            events.append(("bus", channel))

        def open(self) -> None:
            events.append("open")

        def send(self, can_id: int, data: bytes) -> bool:
            events.append(("send", can_id, data))
            FakeMotor.instance.rx_count += 1
            return True

        def close(self) -> None:
            events.append("close")

    class FakeMotor:
        instance: FakeMotor

        def __init__(self, motor_id: int, bus: FakeBus) -> None:
            self.motor_id = motor_id
            self.bus = bus
            self.rx_count = 0
            self.feedback = SimpleNamespace(
                error=0,
                position=-3.2,
                temperature=36.0,
            )
            FakeMotor.instance = self
            events.append(("motor", motor_id))

        @staticmethod
        def decode_error(error: int) -> str:
            return "no fault" if error == 0 else "fault"

    can_bus_module = ModuleType("bimanual.hardware.can_bus")
    can_bus_module.CANBus = FakeBus
    motors_module = ModuleType("bimanual.hardware.motors")
    motors_module.MotorB = FakeMotor
    monkeypatch.setitem(sys.modules, "bimanual.hardware.can_bus", can_bus_module)
    monkeypatch.setitem(sys.modules, "bimanual.hardware.motors", motors_module)

    result = x5_sdk.clear_gripper_error("can1")

    assert result == {
        "can_port": "can1",
        "error": 0,
        "error_name": "no fault",
        "position": -3.2,
        "temperature": 36.0,
    }
    assert events == [
        ("bus", "can1"),
        "open",
        ("motor", 8),
        ("send", 8, b"\xff\xff\xff\xff\xff\xff\xff\xfb"),
        "close",
    ]
