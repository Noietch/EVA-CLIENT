from __future__ import annotations

import ast
import os
import subprocess
import sys
from pathlib import Path

_ROOT = Path(__file__).parents[2]


def _imports(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            names.add(node.module)
    return names


def _contains_call(nodes: list[ast.stmt], name: str) -> bool:
    return any(
        isinstance(candidate, ast.Call)
        and isinstance(candidate.func, ast.Name)
        and candidate.func.id == name
        for node in nodes
        for candidate in ast.walk(node)
    )


def test_teleop_client_has_no_application_or_simulator_dependency() -> None:
    forbidden = ("core", "transport", "recorder")
    for path in (_ROOT / "src/teleop_client").rglob("*.py"):
        invalid = sorted(name for name in _imports(path) if name.startswith(forbidden))
        assert invalid == [], f"{path.relative_to(_ROOT)} imports {invalid}"


def test_handler_does_not_own_recording() -> None:
    imports = _imports(_ROOT / "src/core/app/handlers/teleop.py")
    forbidden = {
        "core.app.handlers.control",
        "core.app.handlers.recording",
        "core.recorder",
    }
    assert imports.isdisjoint(forbidden)


def test_run_starts_input_listener_and_keeps_final_cleanup() -> None:
    tree = ast.parse((_ROOT / "src/core/app/run.py").read_text(encoding="utf-8"))
    run_function = next(
        node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "run"
    )

    assert _contains_call(run_function.body, "start_teleop_input")
    assert any(
        isinstance(node, ast.Try) and _contains_call(node.finalbody, "close_teleop")
        for node in ast.walk(run_function)
    )

    teleop_tree = ast.parse((_ROOT / "src/core/app/handlers/teleop.py").read_text(encoding="utf-8"))
    activate = next(
        node
        for node in teleop_tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "activate_teleop"
    )
    assert _contains_call(activate.body, "_ensure_teleop_setup")


def test_run_drains_teleop_events_independently_of_active_collection() -> None:
    tree = ast.parse((_ROOT / "src/core/app/run.py").read_text(encoding="utf-8"))
    run_function = next(
        node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "run"
    )
    event_branches = [
        node
        for node in ast.walk(run_function)
        if isinstance(node, ast.If) and _contains_call(node.body, "drain_teleop_events")
    ]

    assert len(event_branches) == 1
    condition = ast.unparse(event_branches[0].test)
    assert "runtime.teleop_client is not None" in condition
    assert not _contains_call(event_branches[0].body, "step_teleop")


def test_core_teleop_code_has_no_vr_specific_knowledge() -> None:
    forbidden_tokens = (
        "teleop_client.vr",
        "vr_webxr",
        "base_from_xr_rotation",
    )
    for relative in ("src/core/config.py", "src/core/app/handlers/teleop.py"):
        source = (_ROOT / relative).read_text(encoding="utf-8")
        present = [token for token in forbidden_tokens if token in source]
        assert present == [], f"{relative} contains VR-specific tokens: {present}"


def test_non_vr_application_imports_without_vr_implementation() -> None:
    code = """
import builtins
import os
original = builtins.__import__
def blocked(name, *args, **kwargs):
    if name.startswith('teleop_client.vr') or name.startswith('examples.input_sources.vr_webxr'):
        raise ModuleNotFoundError(name)
    return original(name, *args, **kwargs)
builtins.__import__ = blocked
import core.config
import core.app.handlers.teleop
core.config.load_config(os.environ['EVA_TEST_DEFAULTS'])
"""
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(_ROOT / "src")
    environment["EVA_TEST_DEFAULTS"] = str(_ROOT / "configs/00_base/defaults.py")
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=_ROOT,
        env=environment,
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert result.returncode == 0, result.stderr
