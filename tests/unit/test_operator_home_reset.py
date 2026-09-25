"""Pressing HOME also asks the collection orchestrator to rebuild the scene."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from core.app.operator_control import handle_teleop_operator_event

pytestmark = pytest.mark.unit


def _runtime(*, armed: bool) -> SimpleNamespace:
    return SimpleNamespace(
        teleop_client=SimpleNamespace(acknowledge_event=Mock()),
        teleop_execution=SimpleNamespace(active=False),
        rl_active=False,
        collection_teleop_armed=armed,
        collection_teleop_active=False,
    )


def _session() -> SimpleNamespace:
    return SimpleNamespace(collection_scene_reset_requests=0, last_error="")


def _press_home(
    runtime: SimpleNamespace, session: SimpleNamespace, dispatch: Mock | None = None
) -> None:
    handle_teleop_operator_event(
        SimpleNamespace(intent="home", session_id="s", event_id=1),
        {},
        runtime,
        session,
        dispatch=dispatch or (lambda *_: None),
    )


def test_home_requests_one_scene_reset() -> None:
    session = _session()
    dispatch = Mock()

    _press_home(_runtime(armed=False), session, dispatch)

    assert dispatch.call_args.args[0] == "web:collect_home"
    assert session.collection_scene_reset_requests == 1


def test_home_while_armed_does_not_request_a_reset() -> None:
    session = _session()

    _press_home(_runtime(armed=True), session)

    assert session.collection_scene_reset_requests == 0
