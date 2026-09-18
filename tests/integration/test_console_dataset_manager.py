import pytest

from core.app.console import server
from tests.integration._harness import console_config, serve_console

pytestmark = pytest.mark.integration


def test_dataset_manager_selection_is_validated_and_local_stats_are_slot_based(
    monkeypatch, tmp_path
):
    monkeypatch.setattr(
        server,
        "_collection_slots_snapshot",
        lambda ctx, name: {
            "dataset_dir": str(tmp_path / name),
            "counts": {
                "total": 10,
                "pending": 3,
                "qc_pending": 3,
                "passed": 2,
                "failed": 1,
                "unreviewed": 4,
            },
        },
    )
    monkeypatch.setattr(
        server,
        "_collection_transfer_path",
        lambda config, name: (tmp_path / name, f"datasets/{name}"),
    )
    monkeypatch.setattr(server, "_scene_plan_root", lambda config, name: tmp_path / "plans" / name)
    calls = []
    monkeypatch.setattr(
        server.DatasetTransfer,
        "start",
        lambda self, action, targets, storage=None: calls.append(
            (action, [t["name"] for t in targets])
        ),
    )
    with serve_console(console_config(collection={"enabled": True})) as console:
        response = console.get("/api/dataset_manager")
        data = response.json
        assert data["datasets"][0]["collected"] == 7
        assert data["datasets"][0]["unreviewed"] == 4
        assert (
            console.post(
                "/api/dataset_manager", {"action": "refresh", "datasets": ["cup_set"]}
            ).status
            == 200
        )
        assert calls == [("refresh", ["cup_set"])]
        for names in ([], ["../cup_set"], "cup_set", ["unknown"]):
            assert (
                console.post(
                    "/api/dataset_manager", {"action": "refresh", "datasets": names}
                ).status
                == 400
            )
