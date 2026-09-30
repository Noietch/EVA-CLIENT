"""Expose a running EVA Client to coding agents over MCP stdio.

The MCP process owns no robot resources. Each tool sends a bounded JSON request
to EVA Client's existing ZMQ control channel. Direct robot motions are queued for
the EVA main loop and return an operation id immediately, while status and camera
queries remain read-only. Configure the coding-agent MCP host to launch
``eva-mcp --eva-endpoint tcp://127.0.0.1:5757``.
"""

from __future__ import annotations

import argparse
import base64
from typing import Any

import zmq
from mcp.server.mcpserver import Image, MCPServer

_INSTRUCTIONS = """
EVA Client exposes two peer control modes. Direct tools do not depend on a policy
model and remain available whenever EVA has live robot feedback; a policy model
is optional and may be offline. Use eva_status first. Policy failure never removes
direct-control capability. Direct EEF motion uses IK plus joint interpolation and
is not collision-aware, so prefer small observable steps and use robot_stop
whenever the scene or state is uncertain.
""".strip()


class EvaControlClient:
    """Short-lived ZMQ request client for the EVA control channel."""

    def __init__(self, endpoint: str, timeout_ms: int = 5000) -> None:
        self.endpoint = endpoint
        self.timeout_ms = timeout_ms

    def request(self, message: dict[str, Any]) -> dict[str, Any]:
        context = zmq.Context.instance()
        socket = context.socket(zmq.REQ)
        socket.linger = 0
        socket.connect(self.endpoint)
        try:
            socket.send_json(message)
            if socket.poll(self.timeout_ms) == 0:
                raise TimeoutError(f"EVA control channel timed out at {self.endpoint}")
            response = socket.recv_json()
        finally:
            socket.close()
        if not isinstance(response, dict):
            raise RuntimeError("EVA control channel returned a non-object response")
        return response

    def require_ok(self, message: dict[str, Any]) -> dict[str, Any]:
        response = self.request(message)
        if not response.get("ok", False):
            raise RuntimeError(str(response.get("error", "EVA request failed")))
        return response


def build_server(client: EvaControlClient) -> MCPServer:
    """Build the stdio MCP server around one EVA control-channel client."""
    server = MCPServer("EVA Client", instructions=_INSTRUCTIONS)

    @server.tool()
    def eva_status() -> dict[str, Any]:
        """Discover robot, direct-control, model, motion, and operation status."""
        return client.require_ok({"agent_query": {"action": "status"}})

    @server.tool()
    def robot_get_state() -> dict[str, Any]:
        """Read current joint groups and EEF state when kinematics is initialized."""
        return client.require_ok({"agent_query": {"action": "robot_state"}})

    @server.tool()
    def camera_capture(camera: str) -> Image:
        """Capture the named EVA camera and return one JPEG image to the agent."""
        response = client.require_ok(
            {"agent_query": {"action": "camera_capture", "camera": camera}}
        )
        return Image(data=base64.b64decode(response["data"]), format="jpeg")

    @server.tool()
    def robot_solve_ik(
        group: str,
        position: list[float],
        quaternion_wxyz: list[float],
        frame: str | None = None,
        gripper: float | None = None,
    ) -> dict[str, Any]:
        """Solve an EEF target without moving; poll robot_get_operation for the result."""
        return client.require_ok(
            {
                "agent_command": {
                    "action": "solve_ik",
                    "arguments": {
                        "group": group,
                        "position": position,
                        "quaternion_wxyz": quaternion_wxyz,
                        "frame": frame,
                        "gripper": gripper,
                    },
                }
            }
        )

    @server.tool()
    def robot_move_eef(
        group: str,
        position: list[float],
        quaternion_wxyz: list[float],
        duration_s: float | None = None,
        frame: str | None = None,
        gripper: float | None = None,
    ) -> dict[str, Any]:
        """Move one EEF using IK plus joint interpolation; this is not collision-aware."""
        return client.require_ok(
            {
                "agent_command": {
                    "action": "move_eef",
                    "arguments": {
                        "group": group,
                        "position": position,
                        "quaternion_wxyz": quaternion_wxyz,
                        "duration_s": duration_s,
                        "frame": frame,
                        "gripper": gripper,
                    },
                }
            }
        )

    @server.tool()
    def robot_move_joints(
        group: str,
        positions: list[float],
        duration_s: float | None = None,
    ) -> dict[str, Any]:
        """Move one actuator group in joint space while all other groups hold position."""
        return client.require_ok(
            {
                "agent_command": {
                    "action": "move_joints",
                    "arguments": {
                        "group": group,
                        "positions": positions,
                        "duration_s": duration_s,
                    },
                }
            }
        )

    @server.tool()
    def robot_set_gripper(group: str, state: str) -> dict[str, Any]:
        """Open or close the named actuator group's gripper directly."""
        return client.require_ok(
            {
                "agent_command": {
                    "action": "set_gripper",
                    "arguments": {"group": group, "state": state},
                }
            }
        )

    @server.tool()
    def robot_get_operation(operation_id: str | None = None) -> dict[str, Any]:
        """Read the latest direct-control operation state and result."""
        return client.require_ok(
            {
                "agent_query": {
                    "action": "operation",
                    "operation_id": operation_id,
                }
            }
        )

    @server.tool()
    def robot_stop() -> dict[str, Any]:
        """Cancel queued/running direct motion and halt policy execution immediately."""
        return client.require_ok({"agent_stop": True})

    @server.tool()
    def policy_run(instruction: str) -> dict[str, Any]:
        """Ask the configured policy model to run a task; direct control remains available."""
        for command in (
            "web:select_mode:real",
            f"web:switch_task:{instruction}",
            "web:setup",
            "web:run",
        ):
            client.require_ok({"cmd": command})
        return {
            "ok": True,
            "status": "queued",
            "instruction": instruction,
            "direct_control_capable": True,
            "next": "Poll eva_status; use robot_stop then direct tools to take over.",
        }

    @server.tool()
    def policy_stop() -> dict[str, Any]:
        """Stop policy execution without disabling the robot's direct-control tools."""
        response = client.require_ok({"cmd": "web:halt"})
        response["direct_control_capable"] = True
        return response

    return server


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="EVA Client MCP server")
    parser.add_argument(
        "--eva-endpoint",
        default="tcp://127.0.0.1:5757",
        help="Running EVA Client ZMQ control endpoint",
    )
    parser.add_argument(
        "--timeout-ms",
        type=int,
        default=5000,
        help="Per-request EVA control timeout in milliseconds",
    )
    return parser.parse_args()


def main() -> None:
    """Run the local MCP server over stdio for a coding-agent host."""
    args = _parse_args()
    build_server(EvaControlClient(args.eva_endpoint, args.timeout_ms)).run()


if __name__ == "__main__":
    main()
