"""SIM preparation results must never change a newer console selection."""

import copy
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
from unittest import mock

import pytest

from core.app.console import server
from tests.integration._harness import console_config, serve_console

pytestmark = pytest.mark.integration


def select(console, slot_id, **fields):
    return console.post("/api/select_collection_slot", {
        "dataset": "cup_set", "slot_id": slot_id, "manual": True, **fields,
    })


def test_metadata_update_cannot_reselect_an_obsolete_slot_or_dataset(tmp_path):
    config = console_config(collection={"enabled": True})
    config.collection.storage.data_root = str(tmp_path)
    with serve_console(config) as console:
        rows = console.get("/api/collection_slots?dataset=cup_set&all=1").json["slots"]
        first, second = rows[0]["slot_id"], rows[1]["slot_id"]
        assert select(console, first).status == 200
        ledger = server.slot_state_path(config, "cup_set")
        before = ledger.read_bytes()
        assert select(console, first, metadata_only=True, sim_metadata={"world": "a"}).status == 200
        assert ledger.read_bytes() == before
        assert select(console, second).status == 200
        for metadata in ({"world": "obsolete"}, None):
            assert select(console, first, metadata_only=True, sim_metadata=metadata).status == 409
            assert console.session.collection_slot_id == second
            assert console.session.collection_sim_metadata is None
        assert console.post("/api/select_collect_task", {
            "dataset": "pouring_set", "task": "pour soybean", "task_index": 0,
        }).status == 200
        stale = select(console, second, metadata_only=True, sim_metadata={"world": "b"})
        assert stale.status == 409
        assert console.session.selected_collect_set == "pouring_set"


def test_metadata_invalidation_is_rejected_during_recording(tmp_path):
    config = console_config(collection={"enabled": True})
    config.collection.storage.data_root = str(tmp_path)
    with serve_console(config) as console:
        slots = console.get("/api/collection_slots?dataset=cup_set&all=1").json["slots"]
        slot = slots[0]["slot_id"]
        assert select(console, slot, sim_metadata={"world": "active"}).status == 200
        console.runtime.episode_logger = SimpleNamespace(has_active_episode=True)
        assert select(console, slot, metadata_only=True, sim_metadata=None).status == 409
        assert console.session.collection_sim_metadata == {"world": "active"}
        console.runtime.episode_logger = None


def test_new_selection_waits_for_binding_then_clears_the_previous_metadata(tmp_path):
    config = console_config(collection={"enabled": True})
    config.collection.storage.data_root = str(tmp_path)
    entered, release = threading.Event(), threading.Event()
    original_copy = copy.deepcopy

    def delayed_copy(value, *args, **kwargs):
        if isinstance(value, dict) and value.get("world") == "delayed":
            entered.set()
            assert release.wait(2)
        return original_copy(value, *args, **kwargs)

    with serve_console(config) as console:
        rows = console.get("/api/collection_slots?dataset=cup_set&all=1").json["slots"]
        first, second = rows[0]["slot_id"], rows[1]["slot_id"]
        assert select(console, first).status == 200
        with (
            mock.patch.object(server.copy, "deepcopy", delayed_copy),
            ThreadPoolExecutor(2) as pool,
        ):
            binding = pool.submit(select, console, first, metadata_only=True,
                                  sim_metadata={"world": "delayed"})
            try:
                assert entered.wait(2)
                selection = pool.submit(select, console, second)
                time.sleep(0.05)
                assert not selection.done()
            finally:
                release.set()
            assert binding.result(timeout=2).status == 200
            assert selection.result(timeout=2).status == 200
        assert console.session.collection_slot_id == second
        assert console.session.collection_sim_metadata is None
