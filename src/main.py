"""EVA Client entry point — parses CLI arguments and launches the web application."""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

import argparse
import logging

from core.app.run import run
from core.config import ConfigDict, load_config
from core.devices import REPOSITORY_ROOT, DeviceWorkspace


def parse_args() -> tuple[ConfigDict, int, str | None, bool]:
    """Parse CLI arguments into an ConfigDict plus runtime options.

    Loads the ConfigDict from --config.

    Returns:
        Tuple of (config, web_port, config_path, headless), where config_path is the
        --config value or None.
    """
    parser = argparse.ArgumentParser(
        description="EVA - Unified robot VLA inference debugging client"
    )
    parser.add_argument("--config", type=str, help="Path to business workflow configuration")
    parser.add_argument("--web-port", type=int, default=8080, help="Web server port (default 8080)")
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Low-power mode: no web server; control over the ZMQ channel, status via logs",
    )
    args = parser.parse_args()

    import robots  # noqa: F401

    config = load_config(args.config or REPOSITORY_ROOT / "configs/00_base/defaults.py")
    if config.transport.type != "dataset":
        config = DeviceWorkspace().configure(config)
    if not args.config:
        config.console.initial_tab = "manual"

    return config, args.web_port, args.config, args.headless


def main() -> None:
    """Configure logging, parse CLI arguments, and launch the web application."""
    if sys.argv[1:2] == ["device"]:
        from core.devices.cli import main as device_main

        device_main(sys.argv[2:])
        return
    logging.addLevelName(logging.WARNING, "WARN")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s.%(msecs)03d %(levelname)-5s %(message)s",
        datefmt="%H:%M:%S",
        force=True,
    )
    config, web_port, config_path, headless = parse_args()
    if run(config, web_port=web_port, config_path=config_path, headless=headless):
        os.execv(sys.executable, [sys.executable, *sys.argv])


if __name__ == "__main__":
    main()
