[← Back to README](../README.md)

# 🔬 Development

```bash
uv pip install -e ".[dev]"       # pulls pytest, ruff, pyright
pytest                           # tests
ruff check . && ruff format .    # lint + format
pyright                          # type check
pre-commit install               # optional git hooks
```

Tests are grouped by layer: `tests/unit/`, `tests/integration/`, and
`tests/static/`. Each case declares exactly one layer marker. No ROS
install or hardware required — the robot side
is faked. Manual verification scripts under `tests/manual/` (run as
`PYTHONPATH=src python tests/manual/<script>.py`) exercise end-to-end eval
flow, rollout, LeRobot meta, and viser chunk tracking.
