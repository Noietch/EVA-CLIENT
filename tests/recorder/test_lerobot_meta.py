from __future__ import annotations

from core.recorder.lerobot_meta import history_row, summarize_quality_issues


def test_quality_issue_status_summary_preserves_codes_and_exact_counts() -> None:
    issues = [
        {"severity": "red", "code": "image_skew_exceeded", "detail": f"frame {i}"}
        for i in range(396)
    ]
    issues.append(
        {"severity": "red", "code": "frame_count_mismatch", "detail": "camera short"}
    )

    summaries, count = summarize_quality_issues(issues)

    assert count == 397
    assert summaries == [
        {
            "severity": "red",
            "code": "image_skew_exceeded",
            "detail": "frame 0",
            "count": 396,
        },
        {
            "severity": "red",
            "code": "frame_count_mismatch",
            "detail": "camera short",
            "count": 1,
        },
    ]


def test_history_row_returns_compact_quality_summary() -> None:
    row = history_row(
        {
            "episode_index": 12,
            "length": 1451,
            "quality": "red",
            "quality_issues": [
                {"severity": "red", "code": "image_skew_exceeded", "detail": "first"},
                {"severity": "red", "code": "image_skew_exceeded", "detail": "second"},
            ],
        },
        0,
    )

    assert row["quality_issue_count"] == 2
    assert row["quality_issues"] == [
        {
            "severity": "red",
            "code": "image_skew_exceeded",
            "detail": "first",
            "count": 2,
        }
    ]
