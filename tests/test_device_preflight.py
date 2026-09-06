"""CAN preflight must fail before constructing a motor process."""

from types import SimpleNamespace

import pytest

from core.devices.preflight import check_yam_can

pytestmark = pytest.mark.unit


def test_yam_can_requires_up_can_interfaces(tmp_path):
    command = ["python", "-m", "examples.hardware.yam.node", "--follower-can", "left_arm=can0"]
    with pytest.raises(ValueError, match="adapter connection"):
        check_yam_can(command, net_root=tmp_path)
    interface = tmp_path / "can0"
    interface.mkdir()
    (interface / "type").write_text("280\n")
    (interface / "flags").write_text("0x80\n")
    with pytest.raises(ValueError, match="can0 is DOWN"):
        check_yam_can(command, net_root=tmp_path)
    (interface / "flags").write_text("0x81\n")
    check_yam_can(command, net_root=tmp_path)
    (interface / "type").write_text("1\n")
    with pytest.raises(ValueError, match="not a CAN"):
        check_yam_can(command, net_root=tmp_path)


def test_camera_and_vr_do_not_require_motor_can(tmp_path):
    check_yam_can(
        ["python", "-m", "examples.hardware.yam.node", "--camera-only"], net_root=tmp_path
    )
    check_yam_can(["python", "-m", "examples.input_sources.vr_webxr.node"], net_root=tmp_path)


def test_yam_requires_explicit_channels(tmp_path):
    with pytest.raises(ValueError, match="explicit follower"):
        check_yam_can(["python", "-m", "examples.hardware.yam.node"], net_root=tmp_path)


@pytest.mark.parametrize("authorized", [True, False])
def test_start_prepares_can_without_prompting(tmp_path, monkeypatch, authorized):
    from core.devices import preflight

    interface = tmp_path / "can0"
    interface.mkdir()
    (interface / "type").write_text("280")
    (interface / "flags").write_text("0x80")
    calls = []

    def run(argv, **kwargs):
        calls.append(argv)
        assert kwargs["stdin"] == preflight.subprocess.DEVNULL
        assert kwargs["timeout"] == 120
        if authorized:
            (interface / "flags").write_text("0x81")
        return SimpleNamespace(
            returncode=0 if authorized else 1, stderr="sudo requires authorization"
        )

    monkeypatch.setattr(preflight.subprocess, "run", run)
    command = ["python", "-m", "examples.hardware.yam.node", "--follower-can", "left_arm=can0"]
    if authorized:
        check_yam_can(command, net_root=tmp_path, prepare=True)
        check_yam_can(command, net_root=tmp_path, prepare=True)
    else:
        with pytest.raises(ValueError, match="authorization"):
            check_yam_can(command, net_root=tmp_path, prepare=True)
    assert len(calls) == 1
    assert calls[0][0] == "bash"
    assert calls[0][1].endswith("/yam/prepare_can.sh")
    assert calls[0][2:] == ["can0"]
