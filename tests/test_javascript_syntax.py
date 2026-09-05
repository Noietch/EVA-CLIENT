"""Parse production JavaScript modules with Node's ECMAScript parser."""

import shutil
import subprocess
from pathlib import Path

import pytest

pytestmark = pytest.mark.static
ROOT = Path(__file__).resolve().parents[1]


def test_javascript_modules_parse():
    node = shutil.which("node")
    assert node is not None, "Install Node.js to run JavaScript syntax validation"
    for directory in (
        "src/core/app/console/static/js",
        "tools/datasets/static",
        "examples/input_sources/vr_webxr/static",
    ):
        for path in sorted((ROOT / directory).glob("*.js")):
            result = subprocess.run(
                [node, "--input-type=module", "--check"],
                input=path.read_text(),
                capture_output=True,
                text=True,
                timeout=10,
            )
            assert result.returncode == 0, f"{path}: {result.stderr}"
