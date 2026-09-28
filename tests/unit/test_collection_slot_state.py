"""Collection consoles on different hosts keep independent slot cursors."""

import pytest

from core.app.console.collection_slots import (
    CollectionSlotState,
    load_slot_state,
    save_slot_state,
)
from core.config import ConfigDict
from core.utils.collection_paths import slot_state_path

pytestmark = pytest.mark.unit


def test_slot_selection_is_independent_across_hosts(tmp_path, monkeypatch):
    config = ConfigDict(collection=ConfigDict(storage=ConfigDict(log_dir=str(tmp_path))))

    monkeypatch.setattr("core.utils.collection_paths.socket.gethostname", lambda: "host-a")
    host_a_path = slot_state_path(config, "dataset")
    save_slot_state(host_a_path, CollectionSlotState([], "slot-a"))

    monkeypatch.setattr("core.utils.collection_paths.socket.gethostname", lambda: "host-b")
    host_b_path = slot_state_path(config, "dataset")
    save_slot_state(host_b_path, CollectionSlotState([], "slot-b"))

    assert host_a_path != host_b_path
    assert load_slot_state(host_a_path).selected_slot_id == "slot-a"
    assert load_slot_state(host_b_path).selected_slot_id == "slot-b"
