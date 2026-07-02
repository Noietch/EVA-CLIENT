"""CLI action handlers — observation building, policy connection, action publishing,
inference loop management, and setup/reset/run workflows for the EVA interactive client.

Split across four modules by responsibility; this package re-exports the full surface
so ``from core.app.handlers import <name>`` keeps working for run.py / console.server /
tests unchanged:

  ``utils``      — path/diagnostic formatting + stateless session/strategy/IK resets.
  ``recording``  — observation building + episode/collection recording.
  ``io``         — policy connection, inference fetch, IK solving, dataset replay.
  ``control``    — publishing, gripper control, motion polling, setup/reset/run/mode.
"""

from __future__ import annotations

from core.app.handlers.control import *  # noqa: F401,F403
from core.app.handlers.io import *  # noqa: F401,F403
from core.app.handlers.recording import *  # noqa: F401,F403
from core.app.handlers.utils import *  # noqa: F401,F403
