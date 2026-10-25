[← Back to README](../README.md)

# Development

Use Python 3.11 and the repository's existing `.venv`. Development tools are
listed in the root `dev` extra. For a new checkout, follow [Installation](installation.md).
For an existing environment, add only missing dependencies after confirming the
plan preserves all installed versions and files. Do not run `uv run`, rebuild
the environment, or use synchronization/reinstallation commands.

## Checks

Run checks from the repository root:

```bash
# Relevant test layers; choose the layer affected by your change.
make test-unit
make test-integration
make test-static

# All Python tests, with JUnit output under temp/test-results/.
make test

# Read-only Ruff checks; the same checks run in CI.
make lint

# Type checking uses the repository's pyproject.toml settings.
.venv/bin/pyright
```

The Makefile defaults to `.venv/bin/python`. It disables GPU visibility, selects
CPU JAX, limits numerical library threads to two, uses Matplotlib's `Agg`
backend, and disables pytest plugin autoload. These settings keep checks
independent from machine-specific ROS pytest plugins and shared GPU activity.
`PYTHON=/path/to/python make test-unit` selects a different existing interpreter.

Use a focused selection for a small change, for example:

```bash
make test-unit PYTEST='.venv/bin/python -m pytest tests/unit/test_control_channel.py'
```

Python tests declare one layer marker: `unit`, `integration`, or `static`.
Integration coverage includes local HTTP/socket services, recording formats,
fake hardware, collection, evaluation, replay, and dataset transfers. It does
not prove real robot, camera, simulator, remote policy, or publishing behavior.
Several browser unit tests are separate Node.js files:

```bash
node --test tests/unit/*.test.cjs
```

Manual scripts in `tests/manual/` exercise specialized flows and may create
artifacts or require particular services. Read the selected script's inputs
and side effects before running it. Invoke it directly, for example
`PYTHONPATH=src .venv/bin/python tests/manual/viser_track_chunk.py`.

## Source map

| Area | Location |
|------|----------|
| CLI entry point | `src/main.py` |
| Runtime, state machine, command dispatch | `src/core/app/` |
| Device catalog, workstation overrides, detached service | `src/core/devices/` |
| Python preset loading and validation | `src/core/config.py`, `src/core/cfg/` |
| Recording, SIM export, metadata, QC | `src/core/recorder/` |
| Browser UI and HTTP API | `src/core/app/console/` |
| Robot models and kinematics | `src/robots/` |
| Transport / policy / inference strategies | `src/transport/`, `src/policy_client/`, `src/strategy/` |
| Dataset browser, review, transfer | `tools/datasets/` |
| LeRobot v3.0, MCAP, HDF5 conversion | `tools/conversion/` |
| Hardware and teleoperation defaults | `examples/hardware/*/config.yaml`, `examples/input_sources/*/config.yaml` |

Hardware SDK projects have their own Python environments. Root Ruff excludes
the vendored ARX SDK directories; avoid formatting third-party SDKs. If you
use pre-commit, its configuration includes hooks that modify files; it is
separate from `make lint`, and `pre-commit` is not part of the root `dev` extra.

For a documentation-only change, check referenced paths, config names, links,
and command contracts against the source. Runtime and hardware tests are only
needed when the changed behavior requires them.
