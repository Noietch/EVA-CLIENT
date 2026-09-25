"""The collection orchestrator reaches controller vibration through POST /api/haptic."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from core.app.console.server import ConsoleRequestHandler

pytestmark = pytest.mark.unit


def _handler(send_haptic: Mock) -> SimpleNamespace:
    runtime = SimpleNamespace(teleop_client=SimpleNamespace(send_haptic=send_haptic))
    return SimpleNamespace(ctx=SimpleNamespace(runtime=runtime), _send_json=Mock())


def test_haptic_route_forwards_the_requested_hand() -> None:
    send_haptic = Mock(return_value=True)
    handler = _handler(send_haptic)

    ConsoleRequestHandler._post_haptic(
        handler, {"hand": "left", "intensity": 0.35, "duration_ms": 1000.0}
    )

    assert send_haptic.call_args.kwargs == {
        "hand": "left",
        "intensity": 0.35,
        "duration_ms": 1000.0,
    }
    assert handler._send_json.call_args.args == (200, {"ok": True, "hands": ["left"]})


def test_haptic_route_reports_a_disconnected_headset() -> None:
    handler = _handler(Mock(return_value=False))

    ConsoleRequestHandler._post_haptic(handler, {"hand": "right", "intensity": 0.35})

    assert handler._send_json.call_args.args == (200, {"ok": False, "hands": []})


def test_haptic_route_rejects_an_unknown_hand() -> None:
    handler = _handler(Mock(return_value=True))

    ConsoleRequestHandler._post_haptic(handler, {"hand": "middle"})

    assert handler._send_json.call_args.args[0] == 400
