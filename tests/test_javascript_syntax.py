"""Parse production JavaScript modules with Node's ECMAScript parser."""

import json
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


def test_dataset_object_modeling_filter(tmp_path):
    template = (ROOT / "tools/datasets/templates/index.html").read_text(encoding="utf-8")
    for option in (
        'id="object-modeling-filter"',
        '<option value="missing">未填写建模方法</option>',
        '<option value="A">A · image2assets</option>',
        '<option value="B">B · agent-primitive</option>',
        '<option value="C">C · agent-cad</option>',
        '<option value="D">D · agent-blender</option>',
    ):
        assert option in template

    source = ROOT / "tools/datasets/static/entity-ui.js"
    module = tmp_path / "entity-ui.mjs"
    module.write_text(source.read_text(encoding="utf-8"), encoding="utf-8")
    script = f"""
import {{ objectMatchesFilters }} from {json.dumps(module.as_uri())};
const objects = [
  {{object_id: "OBJ-A", object_name: "image asset", photos: ["a"], modeling_method: "A"}},
  {{object_id: "OBJ-B", object_name: "primitive", photos: [], modeling_method: "B"}},
  {{object_id: "OBJ-C", object_name: "cad model", photos: ["c"], modeling_method: "C"}},
  {{object_id: "OBJ-D", object_name: "blender model", photos: ["d"], modeling_method: "D"}},
  {{object_id: "OBJ-NONE", object_name: "not modeled", photos: [], modeling_method: ""}},
];
const ids = (query = "", photo = "", modeling = "") => objects
  .filter((object) => objectMatchesFilters(object, query, photo, modeling))
  .map((object) => object.object_id);
const expect = (actual, wanted) => {{
  if (JSON.stringify(actual) !== JSON.stringify(wanted)) throw new Error(JSON.stringify(actual));
}};
expect(ids("", "", "A"), ["OBJ-A"]);
expect(ids("", "", "B"), ["OBJ-B"]);
expect(ids("", "", "C"), ["OBJ-C"]);
expect(ids("", "", "D"), ["OBJ-D"]);
expect(ids("", "", "missing"), ["OBJ-NONE"]);
expect(ids("", "ready", ""), ["OBJ-A", "OBJ-C", "OBJ-D"]);
expect(ids("", "missing", "B"), ["OBJ-B"]);
expect(ids("primitive", "", ""), ["OBJ-B"]);
"""
    result = subprocess.run(
        [shutil.which("node"), "--input-type=module", "-e", script],
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert result.returncode == 0, result.stderr
