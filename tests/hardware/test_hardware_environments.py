from __future__ import annotations

import re
import tomllib
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
HARDWARE_ROOT = REPO_ROOT / "examples" / "hardware"
FORBIDDEN_SCRIPT_PATTERNS = (
    re.compile(r"source\s+\.venv/bin/activate"),
    re.compile(r'["\']\$\{REPO_DIR\}/\.venv/bin/activate["\']'),
    re.compile(r'["\']\$\{PROJECT_DIR\}/\.venv["\']'),
    re.compile(r'["\']\$\{PWD\}/\.venv["\']'),
    re.compile(r'["\']\$\{PWD\}/pyproject\.toml["\']'),
    re.compile(r"\.\[(dev|camera|franka|arx_r5|ur5e|viz)\]"),
)


def _adapter_dirs() -> list[Path]:
    return sorted(
        path
        for path in HARDWARE_ROOT.iterdir()
        if path.is_dir() and not path.name.startswith("__")
    )


def _root_shell_scripts(adapter_dir: Path) -> list[Path]:
    return sorted(
        path
        for path in adapter_dir.iterdir()
        if path.is_file() and path.suffix == ".sh"
    )


def _load_pyproject(pyproject_path: Path) -> dict:
    return tomllib.loads(pyproject_path.read_text())


def test_every_top_level_hardware_adapter_has_its_own_pyproject() -> None:
    missing = [
        adapter_dir.name
        for adapter_dir in _adapter_dirs()
        if not (adapter_dir / "pyproject.toml").is_file()
    ]

    assert missing == []


def test_hardware_adapter_project_names_are_unique() -> None:
    names = []
    for adapter_dir in _adapter_dirs():
        pyproject_path = adapter_dir / "pyproject.toml"
        if not pyproject_path.is_file():
            continue
        names.append(_load_pyproject(pyproject_path)["project"]["name"])

    assert len(names) == len(set(names))


def test_hardware_adapter_projects_declare_python_version() -> None:
    missing = []
    for adapter_dir in _adapter_dirs():
        pyproject_path = adapter_dir / "pyproject.toml"
        if not pyproject_path.is_file():
            continue
        project = _load_pyproject(pyproject_path).get("project", {})
        if not project.get("requires-python"):
            missing.append(adapter_dir.name)

    assert missing == []


@pytest.mark.parametrize(
    "script_path",
    [
        script_path
        for adapter_dir in _adapter_dirs()
        for script_path in _root_shell_scripts(adapter_dir)
    ],
    ids=lambda path: str(path.relative_to(REPO_ROOT)),
)
def test_hardware_entrypoint_scripts_do_not_reference_root_environment(
    script_path: Path,
) -> None:
    script = script_path.read_text()

    assert not any(pattern.search(script) for pattern in FORBIDDEN_SCRIPT_PATTERNS)
