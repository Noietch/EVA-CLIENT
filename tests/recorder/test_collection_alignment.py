from __future__ import annotations

import numpy as np

from core.recorder.collection_alignment import align_collection_samples
from core.types import CollectionRawBatch, CollectionRawImage, CollectionRawSample
from robots.base import ActuatorGroup, CameraSpec, ObservationSchema, Robot


def _robot() -> Robot:
    return Robot(
        name="fake",
        actuator_groups=(ActuatorGroup("arm", 3, ("j0", "j1", "gripper"), gripper_index=2),),
        initial_qpos=np.zeros(3, dtype=np.float32),
        observation_schema=ObservationSchema(
            cameras=(CameraSpec("front", "cam_high"),),
            state_composition=("arm",),
        ),
    )


def _sample(timestamp: float, value) -> CollectionRawSample:
    return CollectionRawSample(timestamp=timestamp, value=value)


def test_align_collection_samples_interpolates_vectors_on_fixed_grid():
    batch = CollectionRawBatch(
        images={
            "cam_high": [
                _sample(1.00, np.full((2, 2, 3), 10, dtype=np.uint8)),
                _sample(1.10, np.full((2, 2, 3), 20, dtype=np.uint8)),
                _sample(1.20, np.full((2, 2, 3), 30, dtype=np.uint8)),
            ]
        },
        vectors={
            "state_qpos": [
                _sample(1.00, np.array([0.0, 0.0, 0.0], dtype=np.float32)),
                _sample(1.20, np.array([2.0, 4.0, 1.0], dtype=np.float32)),
            ],
            "action_qpos": [
                _sample(1.00, np.array([10.0, 20.0, 0.0], dtype=np.float32)),
                _sample(1.20, np.array([12.0, 24.0, 1.0], dtype=np.float32)),
            ],
        },
    )

    frames, report = align_collection_samples(
        batch,
        robot=_robot(),
        camera_keys=("cam_high",),
        vector_fields=("state_qpos", "action_qpos"),
        fps=10.0,
        image_skew_sec=0.06,
    )

    assert [frame.timestamp for frame in frames] == [1.0, 1.1, 1.2]
    np.testing.assert_allclose(frames[1].state_qpos, [1.0, 2.0, 0.0])
    np.testing.assert_allclose(frames[1].action_qpos, [11.0, 22.0, 0.0])
    assert frames[1].images["cam_high"][0, 0, 0] == 20
    assert report.issues == []


def test_align_collection_samples_decodes_lazy_image_only_when_selected():
    decoded = []

    def lazy(value: int) -> CollectionRawImage:
        def decode() -> np.ndarray:
            decoded.append(value)
            return np.full((2, 2, 3), value, dtype=np.uint8)

        return CollectionRawImage(decode)

    batch = CollectionRawBatch(
        images={
            "cam_high": [
                _sample(0.0, lazy(10)),
                _sample(0.1, lazy(20)),
            ]
        },
        vectors={
            "state_qpos": [
                _sample(0.0, np.zeros(3, dtype=np.float32)),
                _sample(0.1, np.ones(3, dtype=np.float32)),
            ],
        },
    )

    frames, report = align_collection_samples(
        batch,
        robot=_robot(),
        camera_keys=("cam_high",),
        vector_fields=("state_qpos",),
        fps=10.0,
        image_skew_sec=0.06,
    )

    assert report.issues == []
    assert decoded == [10, 20]
    assert frames[0].images["cam_high"][0, 0, 0] == 10
    assert frames[1].images["cam_high"][0, 0, 0] == 20


def test_align_collection_samples_reports_missing_required_vector_stream():
    batch = CollectionRawBatch(
        images={
            "cam_high": [
                _sample(0.0, np.zeros((2, 2, 3), dtype=np.uint8)),
                _sample(0.1, np.zeros((2, 2, 3), dtype=np.uint8)),
            ]
        },
        vectors={
            "state_qpos:arm": [
                _sample(0.0, np.zeros(3, dtype=np.float32)),
                _sample(0.1, np.ones(3, dtype=np.float32)),
            ],
        },
    )

    frames, report = align_collection_samples(
        batch,
        robot=_robot(),
        camera_keys=("cam_high",),
        vector_fields=("state_qpos", "action_qpos"),
        fps=10.0,
        image_skew_sec=0.06,
    )

    assert frames == []
    assert [(issue.code, issue.detail) for issue in report.issues] == [
        ("missing_vector_stream", "action_qpos:arm has no samples")
    ]


def test_align_collection_samples_holds_gripper_dimensions_discrete():
    batch = CollectionRawBatch(
        images={
            "cam_high": [
                _sample(0.0, np.zeros((2, 2, 3), dtype=np.uint8)),
                _sample(0.1, np.zeros((2, 2, 3), dtype=np.uint8)),
            ]
        },
        vectors={
            "state_qpos": [
                _sample(0.0, np.array([0.0, 0.0, 0.0], dtype=np.float32)),
                _sample(0.1, np.array([1.0, 1.0, 1.0], dtype=np.float32)),
            ],
        },
    )

    frames, _ = align_collection_samples(
        batch,
        robot=_robot(),
        camera_keys=("cam_high",),
        vector_fields=("state_qpos",),
        fps=20.0,
        image_skew_sec=0.06,
    )

    np.testing.assert_allclose(frames[1].state_qpos, [0.5, 0.5, 0.0])
    np.testing.assert_allclose(frames[2].state_qpos, [1.0, 1.0, 1.0])


def test_align_collection_samples_reports_image_skew_over_one_hz_tolerance():
    batch = CollectionRawBatch(
        images={
            "cam_high": [
                _sample(0.0, np.zeros((2, 2, 3), dtype=np.uint8)),
                _sample(0.2, np.ones((2, 2, 3), dtype=np.uint8)),
            ]
        },
        vectors={
            "state_qpos": [
                _sample(0.0, np.zeros(3, dtype=np.float32)),
                _sample(0.2, np.ones(3, dtype=np.float32)),
            ],
        },
    )

    frames, report = align_collection_samples(
        batch,
        robot=_robot(),
        camera_keys=("cam_high",),
        vector_fields=("state_qpos",),
        fps=10.0,
        image_skew_sec=1.0 / (2.0 * 9.0),
    )

    assert [frame.timestamp for frame in frames] == [0.0, 0.1, 0.2]
    assert [issue.code for issue in report.issues] == ["image_skew_exceeded"]
    assert "cam_high" in report.issues[0].detail
    assert "0.100000" in report.issues[0].detail


def test_align_collection_samples_reports_raw_image_gap_stats():
    batch = CollectionRawBatch(
        images={
            "cam_high": [
                _sample(0.00, np.zeros((2, 2, 3), dtype=np.uint8)),
                _sample(0.02, np.zeros((2, 2, 3), dtype=np.uint8)),
                _sample(0.04, np.zeros((2, 2, 3), dtype=np.uint8)),
                _sample(0.24, np.zeros((2, 2, 3), dtype=np.uint8)),
            ]
        },
        vectors={
            "state_qpos": [
                _sample(0.0, np.zeros(3, dtype=np.float32)),
                _sample(0.24, np.ones(3, dtype=np.float32)),
            ],
        },
    )

    _frames, report = align_collection_samples(
        batch,
        robot=_robot(),
        camera_keys=("cam_high",),
        vector_fields=("state_qpos",),
        fps=10.0,
        image_skew_sec=0.06,
    )

    stats = report.image_stream_stats["cam_high"]
    assert stats["sample_count"] == 4
    assert stats["estimated_hz"] == 12.5
    assert stats["max_gap_sec"] == 0.20
    assert stats["p99_gap_sec"] == 0.20
    assert stats["gaps_over_tolerance"] == 1
    assert stats["max_gap_start_time"] == 0.04
    assert stats["max_gap_end_time"] == 0.24


def test_align_collection_samples_concatenates_group_vector_streams():
    robot = Robot(
        name="fake_dual",
        actuator_groups=(
            ActuatorGroup("left", 2, ("l0", "l1")),
            ActuatorGroup("right", 2, ("r0", "r1")),
        ),
        initial_qpos=np.zeros(4, dtype=np.float32),
        observation_schema=ObservationSchema(
            cameras=(CameraSpec("front", "cam_high"),),
            state_composition=("left", "right"),
        ),
    )
    batch = CollectionRawBatch(
        images={
            "cam_high": [
                _sample(0.0, np.zeros((2, 2, 3), dtype=np.uint8)),
                _sample(0.1, np.zeros((2, 2, 3), dtype=np.uint8)),
            ]
        },
        vectors={
            "state_qpos:left": [
                _sample(0.0, np.array([0.0, 1.0], dtype=np.float32)),
                _sample(0.1, np.array([1.0, 2.0], dtype=np.float32)),
            ],
            "state_qpos:right": [
                _sample(0.0, np.array([10.0, 11.0], dtype=np.float32)),
                _sample(0.1, np.array([11.0, 12.0], dtype=np.float32)),
            ],
        },
    )

    frames, report = align_collection_samples(
        batch,
        robot=robot,
        camera_keys=("cam_high",),
        vector_fields=("state_qpos",),
        fps=20.0,
        image_skew_sec=0.06,
    )

    assert report.issues == []
    np.testing.assert_allclose(frames[1].state_qpos, [0.5, 1.5, 10.5, 11.5])


def test_align_collection_samples_interpolates_arm_and_holds_group_gripper():
    robot = Robot(
        name="fake_gripper",
        actuator_groups=(
            ActuatorGroup(
                "arm",
                7,
                ("j0", "j1", "j2", "j3", "j4", "j5", "gripper"),
                gripper_index=6,
            ),
        ),
        initial_qpos=np.zeros(7, dtype=np.float32),
        observation_schema=ObservationSchema(
            cameras=(CameraSpec("front", "cam_high"),),
            state_composition=("arm",),
        ),
    )
    batch = CollectionRawBatch(
        images={
            "cam_high": [
                _sample(0.0, np.zeros((2, 2, 3), dtype=np.uint8)),
                _sample(0.1, np.zeros((2, 2, 3), dtype=np.uint8)),
                _sample(0.2, np.zeros((2, 2, 3), dtype=np.uint8)),
            ]
        },
        vectors={
            "state_qpos:arm": [
                _sample(0.0, np.array([0, 0, 0, 0, 0, 0], dtype=np.float32)),
                _sample(0.2, np.array([2, 4, 6, 8, 10, 12], dtype=np.float32)),
            ],
            "gripper:state_qpos:arm": [
                _sample(0.0, np.array([100], dtype=np.float32)),
                _sample(0.15, np.array([0], dtype=np.float32)),
            ],
        },
    )

    frames, report = align_collection_samples(
        batch,
        robot=robot,
        camera_keys=("cam_high",),
        vector_fields=("state_qpos",),
        fps=10.0,
        image_skew_sec=0.06,
    )

    assert report.issues == []
    assert [frame.timestamp for frame in frames] == [0.0, 0.1, 0.2]
    np.testing.assert_allclose(frames[1].state_qpos, [1, 2, 3, 4, 5, 6, 100])
    np.testing.assert_allclose(frames[2].state_qpos, [2, 4, 6, 8, 10, 12, 0])


def test_align_collection_samples_normalizes_mixed_group_gripper_shapes():
    robot = Robot(
        name="fake_gripper",
        actuator_groups=(
            ActuatorGroup(
                "arm",
                7,
                ("j0", "j1", "j2", "j3", "j4", "j5", "gripper"),
                gripper_index=6,
            ),
        ),
        initial_qpos=np.zeros(7, dtype=np.float32),
        observation_schema=ObservationSchema(
            cameras=(CameraSpec("front", "cam_high"),),
            state_composition=("arm",),
        ),
    )
    batch = CollectionRawBatch(
        images={
            "cam_high": [
                _sample(0.0, np.zeros((2, 2, 3), dtype=np.uint8)),
                _sample(0.1, np.zeros((2, 2, 3), dtype=np.uint8)),
                _sample(0.2, np.zeros((2, 2, 3), dtype=np.uint8)),
            ]
        },
        vectors={
            "action_qpos:arm": [
                _sample(0.0, np.array([0, 0, 0, 0, 0, 0], dtype=np.float32)),
                _sample(0.1, np.array([1, 2, 3, 4, 5, 6, 100], dtype=np.float32)),
                _sample(0.2, np.array([2, 4, 6, 8, 10, 12, 0], dtype=np.float32)),
            ],
        },
    )

    frames, report = align_collection_samples(
        batch,
        robot=robot,
        camera_keys=("cam_high",),
        vector_fields=("action_qpos",),
        fps=20.0,
        image_skew_sec=0.06,
    )

    assert report.issues == []
    np.testing.assert_allclose([frame.timestamp for frame in frames], [0.0, 0.05, 0.1, 0.15, 0.2])
    np.testing.assert_allclose(frames[1].action_qpos, [0.5, 1, 1.5, 2, 2.5, 3, 0])
    np.testing.assert_allclose(frames[3].action_qpos, [1.5, 3, 4.5, 6, 7.5, 9, 100])


def test_align_collection_samples_holds_initial_gripper_when_stream_missing():
    robot = Robot(
        name="fake_gripper",
        actuator_groups=(
            ActuatorGroup(
                "arm",
                7,
                ("j0", "j1", "j2", "j3", "j4", "j5", "gripper"),
                gripper_index=6,
            ),
        ),
        initial_qpos=np.array([0, 0, 0, 0, 0, 0, 42], dtype=np.float32),
        observation_schema=ObservationSchema(
            cameras=(CameraSpec("front", "cam_high"),),
            state_composition=("arm",),
        ),
    )
    batch = CollectionRawBatch(
        images={
            "cam_high": [
                _sample(0.0, np.zeros((2, 2, 3), dtype=np.uint8)),
                _sample(0.1, np.zeros((2, 2, 3), dtype=np.uint8)),
            ]
        },
        vectors={
            "state_qpos:arm": [
                _sample(0.0, np.array([0, 0, 0, 0, 0, 0], dtype=np.float32)),
                _sample(0.1, np.array([1, 2, 3, 4, 5, 6], dtype=np.float32)),
            ],
        },
    )

    frames, report = align_collection_samples(
        batch,
        robot=robot,
        camera_keys=("cam_high",),
        vector_fields=("state_qpos",),
        fps=10.0,
        image_skew_sec=0.06,
    )

    assert report.issues == []
    np.testing.assert_allclose(frames[1].state_qpos, [1, 2, 3, 4, 5, 6, 42])


def test_align_collection_samples_does_not_crop_start_to_first_gripper_sample():
    robot = Robot(
        name="fake_gripper",
        actuator_groups=(
            ActuatorGroup(
                "arm",
                7,
                ("j0", "j1", "j2", "j3", "j4", "j5", "gripper"),
                gripper_index=6,
            ),
        ),
        initial_qpos=np.array([0, 0, 0, 0, 0, 0, 42], dtype=np.float32),
        observation_schema=ObservationSchema(
            cameras=(CameraSpec("front", "cam_high"),),
            state_composition=("arm",),
        ),
    )
    batch = CollectionRawBatch(
        images={
            "cam_high": [
                _sample(0.0, np.zeros((2, 2, 3), dtype=np.uint8)),
                _sample(0.1, np.zeros((2, 2, 3), dtype=np.uint8)),
                _sample(0.2, np.zeros((2, 2, 3), dtype=np.uint8)),
            ]
        },
        vectors={
            "state_qpos:arm": [
                _sample(0.0, np.array([0, 0, 0, 0, 0, 0], dtype=np.float32)),
                _sample(0.2, np.array([2, 4, 6, 8, 10, 12], dtype=np.float32)),
            ],
            "gripper:state_qpos:arm": [
                _sample(0.15, np.array([0], dtype=np.float32)),
            ],
        },
    )

    frames, report = align_collection_samples(
        batch,
        robot=robot,
        camera_keys=("cam_high",),
        vector_fields=("state_qpos",),
        fps=10.0,
        image_skew_sec=0.06,
    )

    assert report.issues == []
    assert [frame.timestamp for frame in frames] == [0.0, 0.1, 0.2]
    np.testing.assert_allclose(frames[0].state_qpos, [0, 0, 0, 0, 0, 0, 42])
    np.testing.assert_allclose(frames[1].state_qpos, [1, 2, 3, 4, 5, 6, 42])
    np.testing.assert_allclose(frames[2].state_qpos, [2, 4, 6, 8, 10, 12, 0])


def test_align_collection_samples_does_not_extrapolate_action_qpos():
    batch = CollectionRawBatch(
        images={
            "cam_high": [
                _sample(0.0, np.zeros((2, 2, 3), dtype=np.uint8)),
                _sample(0.1, np.zeros((2, 2, 3), dtype=np.uint8)),
                _sample(0.2, np.zeros((2, 2, 3), dtype=np.uint8)),
            ]
        },
        vectors={
            "state_qpos": [
                _sample(0.0, np.array([0, 0, 0], dtype=np.float32)),
                _sample(0.2, np.array([2, 4, 1], dtype=np.float32)),
            ],
            "action_qpos": [
                _sample(0.2, np.array([12, 24, 1], dtype=np.float32)),
            ],
        },
    )

    frames, report = align_collection_samples(
        batch,
        robot=_robot(),
        camera_keys=("cam_high",),
        vector_fields=("state_qpos", "action_qpos"),
        fps=10.0,
        image_skew_sec=0.06,
    )

    assert report.issues == []
    assert [frame.timestamp for frame in frames] == [0.2]
    np.testing.assert_allclose(frames[0].action_qpos, [12, 24, 1])


def test_align_collection_samples_latches_action_qpos_to_episode_end():
    batch = CollectionRawBatch(
        end_time=0.2,
        images={
            "cam_high": [
                _sample(0.0, np.zeros((2, 2, 3), dtype=np.uint8)),
                _sample(0.1, np.zeros((2, 2, 3), dtype=np.uint8)),
                _sample(0.2, np.zeros((2, 2, 3), dtype=np.uint8)),
            ]
        },
        vectors={
            "state_qpos": [
                _sample(0.0, np.array([0, 0, 0], dtype=np.float32)),
                _sample(0.2, np.array([2, 4, 1], dtype=np.float32)),
            ],
            "action_qpos": [
                _sample(0.0, np.array([10, 20, 0], dtype=np.float32)),
                _sample(0.1, np.array([11, 22, 1], dtype=np.float32)),
            ],
        },
    )

    frames, report = align_collection_samples(
        batch,
        robot=_robot(),
        camera_keys=("cam_high",),
        vector_fields=("state_qpos", "action_qpos"),
        fps=10.0,
        image_skew_sec=0.06,
    )

    assert report.issues == []
    assert [frame.timestamp for frame in frames] == [0.0, 0.1, 0.2]
    np.testing.assert_allclose(frames[2].action_qpos, [11, 22, 1])


def test_align_collection_samples_respects_episode_start_stop_bounds():
    batch = CollectionRawBatch(
        start_time=0.0,
        end_time=0.2,
        images={
            "cam_high": [
                _sample(-0.1, np.zeros((2, 2, 3), dtype=np.uint8)),
                _sample(0.0, np.zeros((2, 2, 3), dtype=np.uint8)),
                _sample(0.1, np.zeros((2, 2, 3), dtype=np.uint8)),
                _sample(0.2, np.zeros((2, 2, 3), dtype=np.uint8)),
                _sample(0.3, np.zeros((2, 2, 3), dtype=np.uint8)),
            ]
        },
        vectors={
            "state_qpos": [
                _sample(-0.1, np.array([-1, -1, 0], dtype=np.float32)),
                _sample(0.0, np.array([0, 0, 0], dtype=np.float32)),
                _sample(0.2, np.array([2, 4, 1], dtype=np.float32)),
                _sample(0.3, np.array([3, 6, 1], dtype=np.float32)),
            ],
            "action_qpos": [
                _sample(-0.1, np.array([9, 18, 0], dtype=np.float32)),
                _sample(0.0, np.array([10, 20, 0], dtype=np.float32)),
                _sample(0.2, np.array([12, 24, 0], dtype=np.float32)),
                _sample(0.3, np.array([13, 26, 1], dtype=np.float32)),
            ],
        },
    )

    frames, report = align_collection_samples(
        batch,
        robot=_robot(),
        camera_keys=("cam_high",),
        vector_fields=("state_qpos", "action_qpos"),
        fps=10.0,
        image_skew_sec=0.06,
    )

    assert report.issues == []
    assert [frame.timestamp for frame in frames] == [0.0, 0.1, 0.2]
    np.testing.assert_allclose(frames[0].state_qpos, [0, 0, 0])
    np.testing.assert_allclose(frames[-1].action_qpos, [12, 24, 0])
