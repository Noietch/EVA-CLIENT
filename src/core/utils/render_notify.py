"""Ask the resident EVA-SIM render service to render a saved collection episode.

The service replays the episode in place on its own GPU and republishes the dataset's
video metadata, so the console can replay a fresh episode while collection continues.
Collection must not depend on that service being up, so a failed request is reported
once and dropped.
"""

from __future__ import annotations

import json
import logging
import threading
import urllib.error
import urllib.request

logger = logging.getLogger(__name__)

POST_TIMEOUT_SEC = 2.0

_report_failure = True


def notify_render_service(base_url: str, dataset_dir: str, episode_index: int) -> None:
    """Request one episode's render without blocking the caller's thread."""
    threading.Thread(
        target=_post,
        args=(base_url, dataset_dir, episode_index),
        name="render_notify",
        daemon=True,
    ).start()


def _post(base_url: str, dataset_dir: str, episode_index: int) -> None:
    global _report_failure
    body = json.dumps({"dataset_dir": dataset_dir, "episode_index": episode_index}).encode()
    request = urllib.request.Request(
        f"{base_url.rstrip('/')}/render",
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=POST_TIMEOUT_SEC) as response:
            response.read()
    except urllib.error.HTTPError as error:
        _report_once(
            f"render service rejected episode {episode_index}: {error.code} {error.reason}"
        )
    except (urllib.error.URLError, OSError) as error:
        _report_once(f"render service {base_url} unreachable: {error}")


def _report_once(message: str) -> None:
    global _report_failure
    if _report_failure:
        _report_failure = False
        logger.warning("%s (further render-service warnings suppressed)", message)
