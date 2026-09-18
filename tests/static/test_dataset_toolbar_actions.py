"""Check the dataset manager toolbars wire each action to a label and the engine."""

import re
from pathlib import Path

import pytest

from tools.datasets.dataset_transfer import CatalogTransfer, DatasetTransfer

pytestmark = pytest.mark.static
ROOT = Path(__file__).resolve().parents[2]


def _actions(path: Path) -> list[str]:
    return re.findall(r'data-dm-action="([a-z_]+)"', path.read_text(encoding="utf-8"))


def test_dataset_page_toolbar_actions_are_labelled_and_runnable():
    page = ROOT / "tools/datasets/templates/index.html"
    labels = (ROOT / "tools/datasets/static/editor.js").read_text(encoding="utf-8")
    actions = _actions(page)
    assert "download_data" in actions
    assert set(actions) <= CatalogTransfer.ACTIONS
    keys = re.findall(r'data-i18n[\w-]*="([\w.]+)"', page.read_text(encoding="utf-8"))
    for key in dict.fromkeys(keys):
        assert f'"{key}": [' in labels, key


def test_console_toolbar_offers_the_same_data_download():
    actions = _actions(ROOT / "src/core/app/console/static/index.html")
    assert "download_data" in actions
    assert set(actions) <= DatasetTransfer.ACTIONS


def test_dataset_toolbars_act_on_the_selection_alone():
    """Every action, the cloud refresh and automatic QC included, reads the selection."""
    cards = [
        ROOT / "tools/datasets/static/dataset-transfer.js",
        ROOT / "src/core/app/console/static/js/dataset_manager.js",
    ]
    for path in cards:
        source = path.read_text(encoding="utf-8")
        assert "const names = [...selected];" in source, path
        assert re.search(r'start\("refresh"\)', source), path
        # Nothing may fan out to every dataset once the operator picks rows.
        assert not re.search(r'start\("refresh",', source), path
        assert "sweepAll" not in source, path
        assert "!selected.size && " not in source, path
        assert "!datasets.length" not in source, path
