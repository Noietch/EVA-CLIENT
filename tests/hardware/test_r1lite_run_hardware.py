import os
import stat
import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


def _write_executable(path: Path, content: str) -> None:
    path.write_text(content)
    path.chmod(path.stat().st_mode | stat.S_IXUSR)


def test_run_hardware_sources_ros_setup_that_reads_unset_variables(tmp_path):
    fake_ros_setup = tmp_path / "setup.bash"
    fake_ros_setup.write_text(
        """
echo "[fake-ros] setup reached" >&2
if [ -n "$AMENT_TRACE_SETUP_FILES" ]; then
  echo "trace enabled" >&2
fi
export FAKE_ROS_SETUP_SOURCED=1
"""
    )

    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    _write_executable(
        fake_bin / "ssh",
        """#!/usr/bin/env bash
exit 0
""",
    )

    env = os.environ.copy()
    env.update(
        {
            "PATH": f"{fake_bin}:{env['PATH']}",
            "ROS_SETUP_BASH": str(fake_ros_setup),
            "FASTRTPS_DEFAULT_PROFILES_FILE": str(tmp_path / "missing-fastdds.xml"),
        }
    )
    env.pop("AMENT_TRACE_SETUP_FILES", None)

    result = subprocess.run(
        ["bash", "examples/hardware/r1_lite/run_hardware.sh"],
        cwd=REPO_ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 1
    assert "[fake-ros] setup reached" in result.stderr
    assert "missing FastDDS profile" in result.stderr


def test_run_hardware_uses_bundled_fastdds_profile_by_default(tmp_path):
    fake_ros_setup = tmp_path / "setup.bash"
    fake_ros_setup.write_text("export FAKE_ROS_SETUP_SOURCED=1\n")

    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    _write_executable(
        fake_bin / "ssh",
        """#!/usr/bin/env bash
echo "[fake-ssh] $*" >&2
echo "[fake-ssh] FASTRTPS_DEFAULT_PROFILES_FILE=${FASTRTPS_DEFAULT_PROFILES_FILE}" >&2
test -f "${FASTRTPS_DEFAULT_PROFILES_FILE}"
exit 42
""",
    )

    env = os.environ.copy()
    env["HOME"] = str(tmp_path)
    env["PATH"] = f"{fake_bin}:{env['PATH']}"
    env["ROS_SETUP_BASH"] = str(fake_ros_setup)
    env.pop("FASTRTPS_DEFAULT_PROFILES_FILE", None)

    result = subprocess.run(
        ["bash", "examples/hardware/r1_lite/run_hardware.sh"],
        cwd=REPO_ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 42
    assert "[fake-ssh]" in result.stderr
    assert "[fake-ssh] FASTRTPS_DEFAULT_PROFILES_FILE=" in result.stderr
    assert "examples/hardware/r1_lite/fastdds_r1lite_super_client.xml" in result.stderr
