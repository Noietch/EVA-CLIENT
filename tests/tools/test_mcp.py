from __future__ import annotations

import pytest

pytest.importorskip("mcp")

import anyio
from mcp import Client

from tools.mcp.server import build_server


class FakeControlClient:
    def __init__(self) -> None:
        self.messages: list[dict] = []

    def require_ok(self, message: dict) -> dict:
        self.messages.append(message)
        if "agent_query" in message and message["agent_query"]["action"] == "status":
            return {
                "ok": True,
                "direct_control_available": True,
                "control_modes": ["direct"],
                "policy": {"connected": False},
            }
        return {"ok": True, "operation": {"operation_id": "op-1", "status": "queued"}}


def test_server_exposes_direct_and_policy_tools() -> None:
    async def check() -> None:
        fake = FakeControlClient()
        async with Client(build_server(fake)) as client:
            result = await client.list_tools()
            names = {tool.name for tool in result.tools}
            assert {
                "eva_status",
                "robot_get_state",
                "camera_capture",
                "robot_solve_ik",
                "robot_move_eef",
                "robot_move_joints",
                "robot_set_gripper",
                "robot_get_operation",
                "robot_stop",
                "policy_run",
                "policy_stop",
            } <= names

            result = await client.call_tool("eva_status", {})
            assert result.structured_content is not None
            assert result.structured_content["direct_control_available"] is True
            assert result.structured_content["policy"]["connected"] is False

    anyio.run(check)


def test_move_eef_maps_to_typed_agent_command() -> None:
    async def check() -> None:
        fake = FakeControlClient()
        async with Client(build_server(fake)) as client:
            await client.call_tool(
                "robot_move_eef",
                {
                    "group": "left_arm",
                    "position": [0.1, 0.2, 0.3],
                    "quaternion_wxyz": [1.0, 0.0, 0.0, 0.0],
                },
            )
        message = fake.messages[-1]
        assert message["agent_command"]["action"] == "move_eef"
        assert message["agent_command"]["arguments"]["group"] == "left_arm"

    anyio.run(check)
