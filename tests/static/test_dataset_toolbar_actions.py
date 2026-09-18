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
    for key in dict.fromkeys(re.findall(r'data-i18n[\w-]*="([\w.]+)"', page.read_text(encoding="utf-8"))):
        assert f'"{key}": [' in labels, key


def test_console_toolbar_offers_the_same_data_download():
    actions = _actions(ROOT / "src/core/app/console/static/index.html")
    assert "download_data" in actions
    assert set(actions) <= DatasetTransfer.ACTIONS
