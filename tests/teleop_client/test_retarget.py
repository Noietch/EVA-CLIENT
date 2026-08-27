from __future__ import annotations

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from teleop_client.vr.retarget import (
    ArmRetargeter,
    EefLowPassFilter,
    GripperMapping,
    VrControllerState,
    VrPose,
    WorkspaceBounds,
)


def test_eef_low_pass_filter_smooths_pose_but_not_gripper() -> None:
    filt = EefLowPassFilter(alpha=0.25)
    first = np.asarray([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0])
    second = np.asarray([1.0, -1.0, 0.5, 0.0, 0.0, 0.0, 1.0, 0.0])

    np.testing.assert_allclose(filt.apply(first), first)
    smoothed = filt.apply(second)

    np.testing.assert_allclose(smoothed[:3], [0.25, -0.25, 0.125])
    np.testing.assert_allclose(smoothed[3:7], [0.9486833, 0.0, 0.0, 0.3162278], atol=1e-6)
    assert smoothed[7] == 0.0


def test_eef_low_pass_filter_reset_drops_previous_pose() -> None:
    filt = EefLowPassFilter(alpha=0.5)
    first = np.asarray([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0])
    moved = first.copy()
    moved[0] = 1.0
    filt.apply(first)
    filt.apply(moved)
    filt.reset()

    np.testing.assert_allclose(filt.apply(moved), moved)


def _controller(position=(0.0, 0.0, 0.0), *, grip_engaged=True, trigger=0.0, valid=True):
    return VrControllerState(
        hand="left",
        valid=valid,
        grip_engaged=grip_engaged,
        pose=(
            VrPose(
                np.asarray(position, dtype=np.float32),
                np.asarray([0.0, 0.0, 0.0, 1.0], dtype=np.float32),
            )
            if valid
            else None
        ),
        trigger=trigger,
    )


def _retarget(*, rotation=None, workspace=None, eef_filter_alpha=1.0):
    return ArmRetargeter(
        base_from_xr_rotation=np.eye(3) if rotation is None else rotation,
        position_scale=0.5,
        gripper=GripperMapping(mode="binary", threshold=0.5),
        workspace=workspace,
        eef_filter_alpha=eef_filter_alpha,
    )


def _toggle_retarget():
    return ArmRetargeter(
        base_from_xr_rotation=np.eye(3),
        position_scale=0.5,
        gripper=GripperMapping(mode="toggle", threshold=0.5),
    )


def _measured():
    return np.asarray([0.4, 0.1, 0.3, 1.0, 0.0, 0.0, 0.0, 1.0])


def test_first_frame_latches_without_jump_and_maps_relative_motion() -> None:
    rotation = Rotation.from_euler("z", 90, degrees=True).as_matrix()
    retarget = _retarget(rotation=rotation)
    first = retarget.update(_controller(), _measured())
    moved = retarget.update(_controller((0.2, 0.0, 0.0)), _measured())

    assert first.latched
    np.testing.assert_allclose(first.target_eef, _measured())
    np.testing.assert_allclose(moved.target_eef[:3], [0.4, 0.2, 0.3], atol=1e-6)


def test_trigger_still_controls_gripper_without_grip_input() -> None:
    retarget = _retarget()
    first = retarget.update(_controller(trigger=0.0), _measured())
    moved = retarget.update(_controller((0.2, 0.0, 0.0), trigger=1.0), _measured())

    assert first.target_eef[7] == 1.0
    assert moved.target_eef[7] == 0.0
    np.testing.assert_allclose(moved.target_eef[:3], [0.5, 0.1, 0.3], atol=1e-6)


def test_linear_gripper_mapping_uses_pico_trigger_value() -> None:
    mapping = GripperMapping(mode="linear", open_value=1.0, close_value=0.0)

    assert mapping.map(0.0) == 1.0
    assert mapping.map(0.25) == 0.75
    assert mapping.map(1.0) == 0.0


def test_trigger_toggles_gripper_in_x5_mode() -> None:
    retarget = _toggle_retarget()
    initial = retarget.update(_controller(trigger=0.0), _measured())
    pressed = retarget.update(_controller(trigger=1.0), _measured())
    held = retarget.update(_controller(trigger=1.0), _measured())
    released = retarget.update(_controller(trigger=0.0), _measured())
    pressed_again = retarget.update(_controller(trigger=1.0), _measured())

    assert initial.target_eef[7] == 1.0
    assert pressed.target_eef[7] == 0.0
    assert held.target_eef[7] == 0.0
    assert released.target_eef[7] == 0.0
    assert pressed_again.target_eef[7] == 1.0


def test_workspace_rejection_is_local_to_retarget_layer() -> None:
    retarget = _retarget(
        workspace=WorkspaceBounds(
            minimum=np.asarray([0.0, 0.0, 0.0]),
            maximum=np.asarray([0.45, 1.0, 1.0]),
        )
    )
    retarget.update(_controller(), _measured())

    try:
        retarget.update(_controller((0.2, 0.0, 0.0)), _measured())
    except ValueError as error:
        assert "outside workspace" in str(error)
    else:
        raise AssertionError("workspace violation was accepted")


def test_tracking_loss_drops_one_tick_without_reanchoring() -> None:
    retarget = _retarget()
    retarget.update(_controller(), _measured())

    try:
        retarget.update(_controller(valid=False), _measured())
    except ValueError as error:
        assert "not tracked" in str(error)
    else:
        raise AssertionError("untracked controller was accepted")

    recovered = retarget.update(_controller((0.2, 0.0, 0.0)), _measured())
    np.testing.assert_allclose(recovered.target_eef[:3], [0.5, 0.1, 0.3], atol=1e-6)


def test_grip_off_freezes_accumulated_delta_and_reengagement_continues() -> None:
    retarget = _retarget()
    retarget.update(_controller((0.0, 0.0, 0.0)), _measured())
    moved = retarget.update(_controller((0.2, 0.0, 0.0)), _measured())
    np.testing.assert_allclose(moved.target_eef[:3], [0.5, 0.1, 0.3])

    disengaged = retarget.update(
        _controller((0.8, 0.0, 0.0), grip_engaged=False),
        _measured(),
    )
    assert not disengaged.active
    np.testing.assert_allclose(disengaged.target_eef[:3], [0.5, 0.1, 0.3])

    reengaged = retarget.update(
        _controller((1.0, 0.0, 0.0)),
        _measured(),
    )
    assert reengaged.active
    assert not reengaged.latched
    np.testing.assert_allclose(reengaged.target_eef[:3], [0.5, 0.1, 0.3])

    continued = retarget.update(_controller((1.2, 0.0, 0.0)), _measured())
    np.testing.assert_allclose(continued.target_eef[:3], [0.6, 0.1, 0.3])


def test_reset_clears_accumulated_delta_and_home() -> None:
    retarget = _retarget()
    retarget.update(_controller(), _measured())
    retarget.update(_controller((0.4, 0.0, 0.0)), _measured())
    retarget.reset()

    first = retarget.update(_controller((0.4, 0.0, 0.0)), _measured())
    assert first.latched
    np.testing.assert_allclose(first.target_eef[:3], _measured()[:3])


def test_arm_retargeter_filters_incremental_target() -> None:
    retarget = _retarget(eef_filter_alpha=0.5)
    retarget.update(_controller(), _measured())
    moved = retarget.update(_controller((0.2, 0.0, 0.0)), _measured())

    np.testing.assert_allclose(moved.target_eef[:3], [0.45, 0.1, 0.3])


def test_grip_reengagement_accumulates_orientation_delta() -> None:
    retarget = _retarget()
    retarget.update(_controller(), _measured())
    quarter_turn = Rotation.from_euler("z", 90, degrees=True).as_quat()
    first_turn = retarget.update(
        VrControllerState(
            hand="left",
            valid=True,
            grip_engaged=True,
            pose=VrPose(
                np.zeros(3, dtype=np.float32),
                quarter_turn.astype(np.float32),
            ),
        ),
        _measured(),
    )
    assert first_turn.target_eef[3] == pytest.approx(0.7071068, abs=1e-5)

    retarget.update(_controller(grip_engaged=False), _measured())
    half_turn = Rotation.from_euler("z", 180, degrees=True).as_quat()
    retarget.update(
        VrControllerState(
            hand="left",
            valid=True,
            grip_engaged=True,
            pose=VrPose(np.zeros(3, dtype=np.float32), half_turn.astype(np.float32)),
        ),
        _measured(),
    )
    three_quarter_turn = Rotation.from_euler("z", 270, degrees=True).as_quat()
    continued = retarget.update(
        VrControllerState(
            hand="left",
            valid=True,
            grip_engaged=True,
            pose=VrPose(
                np.zeros(3, dtype=np.float32),
                three_quarter_turn.astype(np.float32),
            ),
        ),
        _measured(),
    )
    assert abs(float(continued.target_eef[3])) == pytest.approx(0.0, abs=1e-5)
    assert abs(float(continued.target_eef[6])) == pytest.approx(1.0, abs=1e-5)
