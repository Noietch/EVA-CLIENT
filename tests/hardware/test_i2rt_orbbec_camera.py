from types import SimpleNamespace

import pytest

from examples.hardware.i2rt.orbbec_camera import (
    enable_auto_exposure,
    parse_orbbec_camera_specs,
)


def test_parse_orbbec_camera_specs_supports_serial_and_index() -> None:
    cameras = parse_orbbec_camera_specs(
        ["cam_left_wrist=CV2L360000CL", "cam_right_wrist=index:1"],
        width=640,
        height=480,
        fps=30,
        color_format="MJPG",
        warmup_frames=45,
    )

    assert cameras[0].serial == "CV2L360000CL"
    assert cameras[0].device_index is None
    assert cameras[1].serial is None
    assert cameras[1].device_index == 1
    assert cameras[1].warmup_frames == 45


def test_parse_orbbec_camera_specs_rejects_duplicate_key() -> None:
    with pytest.raises(ValueError, match="Duplicate Orbbec camera key"):
        parse_orbbec_camera_specs(
            ["cam_left_wrist=left", "cam_left_wrist=other"],
        )


def test_enable_auto_exposure_when_property_is_writable() -> None:
    auto_exposure = object()
    write = object()
    read_write = object()

    class _Device:
        def __init__(self) -> None:
            self.values: list[tuple[object, bool]] = []

        def is_property_supported(self, prop: object, permission: object) -> bool:
            return prop is auto_exposure and permission is read_write

        def set_bool_property(self, prop: object, value: bool) -> None:
            self.values.append((prop, value))

    device = _Device()
    sdk = SimpleNamespace(
        OBPropertyID=SimpleNamespace(OB_PROP_COLOR_AUTO_EXPOSURE_BOOL=auto_exposure),
        OBPermissionType=SimpleNamespace(
            PERMISSION_WRITE=write,
            PERMISSION_READ_WRITE=read_write,
        ),
    )

    assert enable_auto_exposure(device, sdk) is True
    assert device.values == [(auto_exposure, True)]
