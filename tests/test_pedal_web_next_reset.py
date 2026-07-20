"""Tests for the standalone pedal-to-control-API helper."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

_SCRIPT = Path(__file__).resolve().parents[1] / "examples/hardware/pedal/pedal_web_next_reset.py"
_SPEC = importlib.util.spec_from_file_location("pedal_web_next_reset", _SCRIPT)
assert _SPEC is not None and _SPEC.loader is not None
pedal_web = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(pedal_web)


def test_first_press_starts_then_single_press_emits_cancel_after_one_second():
    actions = []
    controller = pedal_web.WebCollectionController(1.0, actions.append)

    assert controller.on_press(10.0) == "start"
    assert actions == ["start"]
    assert controller.on_press(11.0) is None
    assert controller.on_timeout(11.99) is None
    assert controller.on_timeout(12.0) == "cancel"
    assert actions == ["start", "cancel"]


def test_second_press_inside_one_second_emits_accept_instead_of_cancel():
    actions = []
    controller = pedal_web.WebCollectionController(1.0, actions.append)

    controller.on_press(20.0)
    assert controller.on_press(21.0) is None
    assert controller.on_press(21.9) == "accept"
    assert controller.on_timeout(22.0) is None
    assert actions == ["start", "accept"]


def test_api_endpoint_preserves_host_and_uses_requested_path():
    assert pedal_web.api_endpoint("http://localhost:8093/", "/api/reset") == (
        "http://localhost:8093/api/reset"
    )


def test_api_endpoint_rejects_non_http_urls():
    try:
        pedal_web.api_endpoint("ws://localhost:8092", "/api/reset")
    except ValueError as error:
        assert "http" in str(error)
    else:
        raise AssertionError("non-HTTP URL should fail")


def test_request_api_sends_operator_intent_as_json(monkeypatch):
    captured = {}

    class _Response:
        status = 200

        def __enter__(self):
            return self

        def __exit__(self, *_):
            return None

        def read(self):
            return b""

    def fake_urlopen(request, timeout):
        captured["url"] = request.full_url
        captured["body"] = request.data
        captured["content_type"] = request.get_header("Content-type")
        captured["timeout"] = timeout
        return _Response()

    monkeypatch.setattr(pedal_web, "urlopen", fake_urlopen)

    assert (
        pedal_web.request_api(
            "http://localhost:8081",
            "/api/operator_action",
            2.0,
            {"intent": "accept"},
        )
        == 200
    )
    assert captured == {
        "url": "http://localhost:8081/api/operator_action",
        "body": b'{"intent": "accept"}',
        "content_type": "application/json",
        "timeout": 2.0,
    }
    assert json.loads(captured["body"]) == {"intent": "accept"}
