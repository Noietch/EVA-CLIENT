"""Pytest fixtures for web UI HTTP integration tests.

Thin wrapper over ``_harness`` (which holds the reusable WebHarness + server
lifecycle). Drives the stdlib ``http.server`` backends end-to-end over real TCP,
exactly as the browser does. See ``_harness`` and work_dirs/COMMONS for details.
"""

from __future__ import annotations

from collections.abc import Generator

import pytest
from _harness import WebHarness, console_config, serve_console


@pytest.fixture
def console(request: pytest.FixtureRequest) -> Generator[WebHarness]:
    """A live console server bound to an ephemeral port, with a synchronous pump.

    Override the config indirectly:
    ``@pytest.mark.parametrize("console", [console_config(...)], indirect=True)``.
    """
    config = getattr(request, "param", None) or console_config()
    with serve_console(config) as harness:
        yield harness
