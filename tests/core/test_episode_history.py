"""History endpoint data contract and cache tests."""

from __future__ import annotations

import json

from core.app.handlers import recording


def _history_row(index: int, task: str = "task", length: int = 1, **updates):
    row = {"episode_index": index, "tasks": [task], "length": length}
    row.update(updates)
    return row


def _write_history(path, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


def _assert_episode_page(page, indices, **expected) -> None:
    assert [row["episode_index"] for row in page["episodes"]] == indices
    for key, value in expected.items():
        assert page[key] == value


def test_load_episode_history_paginates_with_an_offset_cursor(tmp_path):
    history_path = tmp_path / "meta" / "episodes.jsonl"
    _write_history(
        history_path,
        [
            _history_row(3, "first", 4),
            _history_row(9, "second", 5),
            _history_row(12, "third", 6),
        ],
    )

    first = recording.load_episode_history(tmp_path, limit=2)
    _assert_episode_page(
        first,
        [3, 9],
        total=3,
        since=0,
        next_since=2,
        has_more=True,
    )
    assert first["version"]

    second = recording.load_episode_history(tmp_path, since=first["next_since"], limit=2)
    _assert_episode_page(second, [12], next_since=3, has_more=False)
    assert second["version"] == first["version"]


def test_load_episode_history_filters_by_task_before_pagination(tmp_path):
    history_path = tmp_path / "meta" / "episodes.jsonl"
    _write_history(
        history_path,
        [
            _history_row(3, "first", 4),
            _history_row(9, "second", 5),
            _history_row(12, "first", 6),
        ],
    )

    first = recording.load_episode_history(tmp_path, task="first", limit=1)
    _assert_episode_page(first, [3], total=2, next_since=1, has_more=True)

    second = recording.load_episode_history(
        tmp_path, task="first", since=first["next_since"], limit=1
    )
    _assert_episode_page(second, [12], total=2, has_more=False)


def test_history_cursor_appends_tail_and_resets_after_prefix_rewrite(tmp_path):
    history_path = tmp_path / "meta" / "episodes.jsonl"
    rows = [
        _history_row(0, length=2),
        _history_row(1, length=3),
    ]
    _write_history(history_path, rows)
    initial = recording.load_episode_history(tmp_path)

    rows.append({"episode_index": 2, "tasks": ["task"], "length": 4})
    _write_history(history_path, rows)
    appended = recording.load_episode_history(
        tmp_path,
        since=initial["next_since"],
        cursor=initial["cursor"],
    )

    assert appended["reset"] is False
    _assert_episode_page(appended, [2])

    rows[0]["qc_verdict"] = "fail"
    _write_history(history_path, rows)
    rewritten = recording.load_episode_history(
        tmp_path,
        since=appended["next_since"],
        cursor=appended["cursor"],
    )

    assert rewritten["reset"] is True
    _assert_episode_page(rewritten, [0, 1, 2])
    assert rewritten["episodes"][0]["qc_verdict"] == "fail"


def test_history_cursor_adds_row_after_pending_job_leaves_queue(tmp_path):
    history_path = tmp_path / "meta" / "episodes.jsonl"
    _write_history(
        history_path,
        [
            _history_row(0, length=2),
            _history_row(1, length=3),
        ],
    )
    saving = recording.load_episode_history(
        tmp_path,
        exclude_episode_indices={1},
    )

    saved = recording.load_episode_history(
        tmp_path,
        since=saving["next_since"],
        cursor=saving["cursor"],
    )

    assert saving["total"] == 1
    _assert_episode_page(saved, [1])
    assert saved["reset"] is False


def test_load_episode_history_invalidates_cache_after_qc_rewrite(tmp_path, monkeypatch):
    history_path = tmp_path / "meta" / "episodes.jsonl"
    _write_history(
        history_path,
        [_history_row(0)],
    )
    first = recording.load_episode_history(tmp_path)

    reads = 0
    original_reader = recording._read_episode_history_rows

    def count_reads(path):
        nonlocal reads
        reads += 1
        return original_reader(path)

    monkeypatch.setattr(recording, "_read_episode_history_rows", count_reads)
    cached = recording.load_episode_history(tmp_path)
    assert cached["episodes"] == first["episodes"]
    assert reads == 0

    _write_history(
        history_path,
        [
            {
                **_history_row(0),
                "qc_verdict": "fail",
            }
        ],
    )
    refreshed = recording.load_episode_history(tmp_path)
    assert reads == 1
    assert refreshed["version"] != first["version"]
    assert refreshed["episodes"][0]["qc_verdict"] == "fail"


def test_zero_limit_counts_without_projecting_rows(tmp_path, monkeypatch):
    history_path = tmp_path / "meta" / "episodes.jsonl"
    _write_history(
        history_path,
        [_history_row(index) for index in range(4)],
    )

    def fail_projection(*args, **kwargs):
        raise AssertionError("count-only reads must not project history rows")

    monkeypatch.setattr(recording, "_read_episode_history_rows", fail_projection)
    result = recording.load_episode_history(tmp_path, limit=0)

    assert result["episodes"] == []
    assert result["total"] == 4
    assert result["next_since"] == 0


def test_zero_limit_with_cursor_detects_a_prefix_rewrite(tmp_path):
    history_path = tmp_path / "meta" / "episodes.jsonl"
    rows = [
        _history_row(0),
        _history_row(1),
    ]
    _write_history(history_path, rows)
    initial = recording.load_episode_history(tmp_path)

    rows[0]["qc_verdict"] = "fail"
    _write_history(history_path, rows)
    checked = recording.load_episode_history(
        tmp_path,
        since=initial["next_since"],
        limit=0,
        cursor=initial["cursor"],
    )

    assert checked["episodes"] == []
    assert checked["reset"] is True
    assert checked["next_since"] == 0
    assert checked["has_more"] is True
