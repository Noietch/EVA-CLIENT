import numpy as np
import pytest

from core.recorder.camera_quality import frozen_camera_issues
from core.recorder.collection_alignment import align_collection_samples
from core.types import CollectionRawBatch, CollectionRawSample, Observation
from robots.base import ActuatorGroup, CameraSpec, ObservationSchema, Robot

pytestmark = pytest.mark.unit


def robot():
    return Robot(
        name="test",
        initial_qpos=np.zeros(4),
        actuator_groups=tuple(
            ActuatorGroup(arm, 2, ("joint", "grip"), gripper_index=1)
            for arm in ("left_arm", "right_arm")
        ),
        observation_schema=ObservationSchema(
            cameras=(
                CameraSpec("front", "front"),
                CameraSpec("left", "left", attached_to="left_arm"),
                CameraSpec("right", "right", attached_to="right_arm"),
            ),
            state_composition=("left_arm", "right_arm"),
        ),
    )


def frames(changed=("front", "left"), state=(0.1, 0, 0, 0)):
    return [
        Observation(
            timestamp=float(i),
            state_qpos=np.asarray(state) * i,
            images={
                key: np.full((2, 2, 3), int(i > 0 and key in changed), dtype=np.uint8)
                for key in ("front", "left", "right")
            },
        )
        for i in range(3)
    ]


def test_stationary_arm_camera_allowed_while_other_arm_moves():
    assert frozen_camera_issues(frames(), robot(), ("front", "left", "right")) == []


@pytest.mark.parametrize(
    "changed,expected", [((), {"front", "left"}), (("front",), {"left"}), (("left",), {"front"})]
)
def test_frozen_external_and_moving_wrist_are_red(changed, expected):
    issues = frozen_camera_issues(frames(changed), robot(), ("front", "left", "right"))
    assert {detail.split(":")[0] for _, detail in issues} == expected


def test_gripper_and_small_feedback_noise_do_not_require_wrist_motion():
    assert (
        frozen_camera_issues(
            frames(("front",), (0.001, 1, 0, 1)), robot(), ("front", "left", "right")
        )
        == []
    )


def test_disabled_camera_not_checked():
    assert frozen_camera_issues(frames(()), robot(), ("right",)) == []


@pytest.mark.parametrize("timestamps", [[0, 0.1], [1.1, 1.2, 2], [0, 0.1, 1.5, 2]])
def test_timeout_at_end_start_or_reconnected_middle_is_not_trimmed_away(timestamps):
    batch = CollectionRawBatch(
        start_time=0,
        end_time=2,
        images={
            "front": [
                CollectionRawSample(t, np.zeros((2, 2, 3), dtype=np.uint8)) for t in timestamps
            ]
        },
        vectors={
            field: [CollectionRawSample(t, np.zeros(4)) for t in (0, 2)]
            for field in ("state_qpos", "action_qpos")
        },
    )
    _, report = align_collection_samples(
        batch,
        robot=robot(),
        camera_keys=("front",),
        vector_fields=("state_qpos", "action_qpos"),
        fps=10,
        image_skew_sec=0.2,
    )
    assert "camera_frame_timeout" in {issue.code for issue in report.issues}
