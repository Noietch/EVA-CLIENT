"""Require explicit test layers while keeping tests grouped by component."""

import pytest


def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    layers = {"unit", "integration", "static", "e2e"}
    for item in items:
        selected = {marker.name for marker in item.iter_markers()} & layers
        if len(selected) != 1:
            raise pytest.UsageError(
                f"{item.nodeid}: declare exactly one test layer, got {selected}"
            )
