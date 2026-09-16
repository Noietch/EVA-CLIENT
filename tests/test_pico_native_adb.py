"""USB forwarding for the native PICO client."""

import os
import subprocess

import pytest

from core.devices import REPOSITORY_ROOT

pytestmark = pytest.mark.unit


def test_prepare_only_preserves_existing_reverse_and_adopts_new_headset(tmp_path):
    adb = tmp_path / "adb"
    adb.write_text(
        "#!/bin/bash\n"
        "if [[ $1 == -P ]]; then shift 2; fi\n"
        "if [[ $1 == devices ]]; then\n"
        "  printf 'List of devices attached\\nOLD\\tdevice\\nNEW\\tdevice\\n'\n"
        "elif [[ $1 == -s && $3 == reverse && $4 == --list ]]; then\n"
        "  [[ $2 == OLD ]] && printf 'UsbFfs tcp:43876 tcp:43876\\n'\n"
        "elif [[ $1 == -s && $3 == reverse ]]; then\n"
        "  printf '%s\\n' \"$2\" >> \"$MOCK_ADB_ADDS\"\n"
        "else exit 1; fi\n"
    )
    adb.chmod(0o755)
    added = tmp_path / "adds"
    script = REPOSITORY_ROOT / "examples/input_sources/eva-pico/start.sh"
    env = dict(os.environ, ADB=str(adb), MOCK_ADB_ADDS=str(added))

    first = subprocess.run(["bash", str(script), "--prepare-only"], env=env, capture_output=True)
    assert first.returncode == 0, first.stderr
    assert added.read_text().splitlines() == ["NEW"]

    added.unlink()
    env["PICO_SERIAL"] = "OLD"
    second = subprocess.run(["bash", str(script), "--prepare-only"], env=env, capture_output=True)
    assert second.returncode == 0, second.stderr
    assert not added.exists()
