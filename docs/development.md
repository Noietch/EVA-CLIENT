[← Back to README](../README.md)

# 🔬 Development

```bash
uv pip install -e ".[dev]"       # pulls pytest, ruff, pyright
pytest                           # tests
pytest -m "not slow"             # skip slow IK tests
ruff check . && ruff format .    # lint + format
pyright                          # type check
pre-commit install               # optional git hooks
```

Tests are grouped under `tests/config/`, `tests/comms/`, `tests/ik/` (slow),
`tests/strategy/`, `tests/transport/`, `tests/recorder/`, and
`tests/integration/web/`. No ROS install or hardware required — the robot side
is faked. Manual verification scripts under `tests/manual/` (run as
`PYTHONPATH=src python tests/manual/<script>.py`) exercise end-to-end eval
flow, rollout, LeRobot meta, and viser chunk tracking.
