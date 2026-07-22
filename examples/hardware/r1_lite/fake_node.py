#!/usr/bin/env python3
"""ROS2-only fake R1 Lite node with an embedded local control UI."""

from __future__ import annotations

import argparse
import dataclasses
import io
import json
import logging
import threading
import time
import webbrowser
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

import numpy as np
from PIL import Image

LOGGER = logging.getLogger(__name__)

LEFT_ARM_JOINTS = tuple(f"left_arm_joint{i}" for i in range(1, 7))
RIGHT_ARM_JOINTS = tuple(f"right_arm_joint{i}" for i in range(1, 7))
GRIPPER_OPEN = 100.0
GRIPPER_CLOSE = 0.0
JOINT_STEP = 0.05
OPERATOR_ACTION_TOPIC = "/eva/operator_action"
SUPPORTED_OPERATOR_ACTIONS = frozenset(
    {
        "rollout_toggle",
        "rollout_reset",
        "intervention_toggle",
        "intervention_accept",
        "gripper_left_open",
        "gripper_left_close",
        "gripper_right_open",
        "gripper_right_close",
    }
)

CAMERA_TOPICS = {
    "head": "/hdas/camera_head/right_raw/image_raw_color/compressed",
    "left_wrist": "/hdas/camera_wrist_left/color/image_raw/compressed",
    "right_wrist": "/hdas/camera_wrist_right/color/image_raw/compressed",
}

CAMERA_OBSERVATION_KEYS = {
    "head": "cam_high",
    "left_wrist": "cam_left_wrist",
    "right_wrist": "cam_right_wrist",
}

CAMERA_COLORS = {
    "head": (170, 70, 45),
    "left_wrist": (45, 95, 170),
    "right_wrist": (45, 160, 95),
}
CAMERA_FRAME_CACHE_SIZE = 30


@dataclasses.dataclass(frozen=True)
class ArmTopics:
    """ROS2 topic names for one R1 Lite arm group."""

    feedback: str
    command: str
    eef: str
    gripper_feedback: str
    gripper_command: str
    hil: str


ARM_TOPICS = {
    "left_arm": ArmTopics(
        feedback="/hdas/feedback_arm_left",
        command="/motion_target/target_joint_state_arm_left",
        eef="/relaxed_ik/motion_control/pose_ee_arm_left",
        gripper_feedback="/hdas/feedback_gripper_left",
        gripper_command="/motion_target/target_position_gripper_left",
        hil="/eva/hil/input_joint_state_arm_left",
    ),
    "right_arm": ArmTopics(
        feedback="/hdas/feedback_arm_right",
        command="/motion_target/target_joint_state_arm_right",
        eef="/relaxed_ik/motion_control/pose_ee_arm_right",
        gripper_feedback="/hdas/feedback_gripper_right",
        gripper_command="/motion_target/target_position_gripper_right",
        hil="/eva/hil/input_joint_state_arm_right",
    ),
}

JOINT_NAMES = {
    "left_arm": LEFT_ARM_JOINTS,
    "right_arm": RIGHT_ARM_JOINTS,
}


@dataclasses.dataclass(frozen=True)
class Ros2MessageTypes:
    """ROS2 message constructors used by the fake runtime."""

    compressed_image: type
    joint_state: type
    pose_stamped: type
    string: type


class R1LiteFakeState:
    """Thread-safe synthetic R1 Lite feedback, command, and UI HIL state."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._feedback = {
            "left_arm": np.zeros(6, dtype=np.float32),
            "right_arm": np.zeros(6, dtype=np.float32),
        }
        self._commands = {
            "left_arm": np.zeros(6, dtype=np.float32),
            "right_arm": np.zeros(6, dtype=np.float32),
        }
        self._hil = {
            "left_arm": np.zeros(6, dtype=np.float32),
            "right_arm": np.zeros(6, dtype=np.float32),
        }
        self._grippers = {
            "left_arm": float(GRIPPER_OPEN),
            "right_arm": float(GRIPPER_OPEN),
        }

    def snapshot(self) -> dict[str, Any]:
        """Return a JSON-serializable copy of feedback, gripper, and HIL state.

        Returns:
            snapshot: dict with arm feedback joints [6], gripper scalar, command
                joints [6], and HIL joints [6] for each group.
        """
        with self._lock:
            return {
                "feedback": {
                    group: {
                        "joints": self._feedback[group].astype(float).tolist(),
                        "gripper": float(self._grippers[group]),
                    }
                    for group in ARM_TOPICS
                },
                "hil": {group: self._hil[group].astype(float).tolist() for group in ARM_TOPICS},
                "command": {
                    group: self._commands[group].astype(float).tolist() for group in ARM_TOPICS
                },
                "joint_step": JOINT_STEP,
            }

    def feedback_copy(self) -> tuple[dict[str, np.ndarray], dict[str, float]]:
        """Return copies of feedback arm joints and gripper positions.

        Returns:
            arms: group -> arm joint vector [6].
            grippers: group -> gripper position scalar.
        """
        with self._lock:
            arms = {group: value.copy() for group, value in self._feedback.items()}
            grippers = dict(self._grippers)
        return arms, grippers

    def hil_copy(self) -> dict[str, np.ndarray]:
        """Return copies of the current HIL joint vectors.

        Returns:
            hil: group -> HIL joint vector [6].
        """
        with self._lock:
            return {group: value.copy() for group, value in self._hil.items()}

    def command_copy(self) -> dict[str, np.ndarray]:
        """Return copies of current arm command targets.

        Returns:
            commands: group -> command target vector [6].
        """
        with self._lock:
            return {group: value.copy() for group, value in self._commands.items()}

    def sync_hil_to_feedback(self) -> dict[str, np.ndarray]:
        """Copy current feedback arm joints into the HIL input state.

        Returns:
            hil: group -> synchronized HIL joint vector [6].
        """
        with self._lock:
            for group_name, feedback in self._feedback.items():
                self._hil[group_name] = feedback.copy()
            return {group: value.copy() for group, value in self._hil.items()}

    def set_hil_joint(self, group_name: str, joint_index: int, value: float) -> np.ndarray:
        """Set one HIL slider joint and return the group's full HIL vector.

        Args:
            group_name: "left_arm" or "right_arm".
            joint_index: arm-local joint index in [0, 5].
            value: new joint position.

        Returns:
            positions: [6] float32 HIL joint vector.
        """
        self._require_group(group_name)
        self._require_joint_index(joint_index)
        with self._lock:
            self._hil[group_name][joint_index] = float(value)
            return self._hil[group_name].copy()

    def apply_hil_delta(self, group_name: str, joint_index: int, delta: float) -> np.ndarray:
        """Add a delta to one HIL slider joint and return the full HIL vector.

        Args:
            group_name: "left_arm" or "right_arm".
            joint_index: arm-local joint index in [0, 5].
            delta: signed joint increment.

        Returns:
            positions: [6] float32 HIL joint vector.
        """
        self._require_group(group_name)
        self._require_joint_index(joint_index)
        with self._lock:
            self._hil[group_name][joint_index] += float(delta)
            return self._hil[group_name].copy()

    def apply_command_delta(self, group_name: str, joint_index: int, delta: float) -> np.ndarray:
        """Add a delta to one feedback joint and return the full command vector.

        Args:
            group_name: "left_arm" or "right_arm".
            joint_index: arm-local joint index in [0, 5].
            delta: signed joint increment.

        Returns:
            positions: [6] float32 command vector.
        """
        self._require_group(group_name)
        self._require_joint_index(joint_index)
        with self._lock:
            self._commands[group_name][joint_index] += float(delta)
            self._feedback[group_name] = self._commands[group_name].copy()
            return self._commands[group_name].copy()

    def apply_arm_command(self, group_name: str, positions: np.ndarray) -> None:
        """Apply an incoming motion target command to fake arm feedback.

        Args:
            group_name: "left_arm" or "right_arm".
            positions: [6] arm joint target, or [7] group target with gripper.
        """
        self._require_group(group_name)
        vector = np.asarray(positions, dtype=np.float32).reshape(-1)
        if vector.shape == (7,):
            with self._lock:
                self._commands[group_name] = vector[:6].copy()
                self._feedback[group_name] = vector[:6].copy()
                self._grippers[group_name] = float(vector[6])
            return
        if vector.shape != (6,):
            raise ValueError(
                f"{group_name} arm command must have 6 or 7 positions, got {vector.shape}"
            )
        with self._lock:
            self._commands[group_name] = vector.copy()
            self._feedback[group_name] = vector.copy()

    def apply_gripper_command(self, group_name: str, value: float) -> None:
        """Apply an incoming gripper command to fake gripper feedback.

        Args:
            group_name: "left_arm" or "right_arm".
            value: gripper position scalar.
        """
        self._require_group(group_name)
        with self._lock:
            self._grippers[group_name] = float(value)

    def _require_group(self, group_name: str) -> None:
        if group_name not in ARM_TOPICS:
            allowed = ", ".join(ARM_TOPICS)
            raise ValueError(f"unsupported group {group_name!r}; expected one of: {allowed}")

    def _require_joint_index(self, joint_index: int) -> None:
        if joint_index < 0 or joint_index >= 6:
            raise ValueError(f"joint_index must be in [0, 5], got {joint_index}")


class R1LiteFakeControlApi:
    """UI-facing command API over the synthetic state and ROS2 bridge."""

    def __init__(self, state: R1LiteFakeState, bridge: Any) -> None:
        self._state = state
        self._bridge = bridge

    def snapshot(self) -> dict[str, Any]:
        """Return current UI state.

        Returns:
            snapshot: JSON-serializable state from R1LiteFakeState.
        """
        return self._state.snapshot()

    def operator_action(self, action: str) -> dict[str, str]:
        """Publish one semantic EVA operator action.

        Args:
            action: Device-independent operator action.

        Returns:
            result: JSON-serializable acknowledgement.
        """
        value = str(action).strip().lower()
        if value not in SUPPORTED_OPERATOR_ACTIONS:
            raise ValueError(f"unsupported operator action {action!r}")
        self._bridge.publish_operator_action(value)
        return {"ok": "true", "action": value}

    def gripper(self, side: str, command: str) -> dict[str, str]:
        """Publish one gripper operator label.

        Args:
            side: left or right.
            command: open or close.

        Returns:
            result: JSON-serializable acknowledgement.
        """
        side_value = str(side).strip().lower()
        command_value = str(command).strip().lower()
        if side_value not in {"left", "right"}:
            raise ValueError(f"unsupported gripper side {side!r}")
        if command_value not in {"open", "close"}:
            raise ValueError(f"unsupported gripper command {command!r}")
        return self.operator_action(f"gripper_{side_value}_{command_value}")

    def set_joint(self, group_name: str, joint_index: int, value: float) -> dict[str, Any]:
        """Set one HIL joint and publish the full HIL vector.

        Args:
            group_name: "left_arm" or "right_arm".
            joint_index: arm-local joint index in [0, 5].
            value: new joint position.

        Returns:
            result: group and updated [6] HIL positions.
        """
        positions = self._state.set_hil_joint(group_name, int(joint_index), float(value))
        self._bridge.publish_hil(group_name, positions)
        return {"group": group_name, "positions": positions.astype(float).tolist()}

    def adjust_joint(self, group_name: str, joint_index: int, delta: float) -> dict[str, Any]:
        """Add a delta to one HIL joint and publish the full HIL vector.

        Args:
            group_name: "left_arm" or "right_arm".
            joint_index: arm-local joint index in [0, 5].
            delta: signed joint increment.

        Returns:
            result: group and updated [6] HIL positions.
        """
        positions = self._state.apply_hil_delta(group_name, int(joint_index), float(delta))
        self._bridge.publish_hil(group_name, positions)
        return {"group": group_name, "positions": positions.astype(float).tolist()}

    def adjust_command(self, group_name: str, joint_index: int, delta: float) -> dict[str, Any]:
        """Add a delta to one command joint and publish a motion target.

        Args:
            group_name: "left_arm" or "right_arm".
            joint_index: arm-local joint index in [0, 5].
            delta: signed joint increment.

        Returns:
            result: group and updated [6] command positions.
        """
        positions = self._state.apply_command_delta(
            group_name,
            int(joint_index),
            float(delta),
        )
        self._bridge.publish_command(group_name, positions)
        return {"group": group_name, "positions": positions.astype(float).tolist()}

    def sync_hil_to_feedback(self) -> dict[str, str]:
        """Sync all HIL joint vectors to the current fake feedback and publish them.

        Returns:
            result: JSON-serializable acknowledgement.
        """
        synced = self._state.sync_hil_to_feedback()
        for group_name, positions in synced.items():
            self._bridge.publish_hil(group_name, positions)
        return {"ok": "true"}


class R1LiteRos2Fake:
    """ROS2 publisher/subscriber bridge for the synthetic R1 Lite state."""

    def __init__(
        self,
        *,
        node: Any,
        message_types: Ros2MessageTypes,
        state: R1LiteFakeState,
        rate_hz: float,
        image_height: int,
        image_width: int,
        qos: Any,
        camera_qos: Any = None,
        control_node: Any = None,
        command_callback_group: Any = None,
        timer_callback_group: Any = None,
        create_timer: bool = True,
        publish_cameras: bool = True,
        publish_robot: bool = True,
        subscribe_commands: bool = True,
    ) -> None:
        if rate_hz <= 0.0:
            raise ValueError(f"rate_hz must be positive, got {rate_hz}")
        if image_height <= 0 or image_width <= 0:
            raise ValueError(f"image size must be positive, got {image_width}x{image_height}")

        self._node = node
        self._control_node = control_node or node
        self._types = message_types
        self._state = state
        self._image_height = image_height
        self._image_width = image_width
        self._frame_index = 0
        self._camera_frame_cache = {}
        if publish_cameras:
            self._camera_frame_cache = {
                camera_name: [
                    _jpeg_image(
                        CAMERA_COLORS[camera_name],
                        image_height,
                        image_width,
                        frame_index,
                    )
                    for frame_index in range(CAMERA_FRAME_CACHE_SIZE)
                ]
                for camera_name in CAMERA_TOPICS
            }

        self._camera_publishers = {}
        if publish_cameras:
            self._camera_publishers = {
                name: node.create_publisher(
                    message_types.compressed_image,
                    topic,
                    camera_qos if camera_qos is not None else qos,
                )
                for name, topic in CAMERA_TOPICS.items()
            }
        self._arm_publishers = {}
        self._command_publishers = {}
        self._gripper_publishers = {}
        self._eef_publishers = {}
        self._operator_action_publisher = None
        self._hil_publishers = {}
        if publish_robot:
            self._arm_publishers = {
                group: node.create_publisher(message_types.joint_state, topics.feedback, qos)
                for group, topics in ARM_TOPICS.items()
            }
            self._gripper_publishers = {
                group: node.create_publisher(
                    message_types.joint_state, topics.gripper_feedback, qos
                )
                for group, topics in ARM_TOPICS.items()
            }
            self._eef_publishers = {
                group: node.create_publisher(message_types.pose_stamped, topics.eef, qos)
                for group, topics in ARM_TOPICS.items()
            }
            self._operator_action_publisher = self._control_node.create_publisher(
                message_types.string,
                OPERATOR_ACTION_TOPIC,
                10,
            )
            self._hil_publishers = {
                group: self._control_node.create_publisher(
                    message_types.joint_state, topics.hil, qos
                )
                for group, topics in ARM_TOPICS.items()
            }
            self._command_publishers = {
                group: self._control_node.create_publisher(
                    message_types.joint_state,
                    topics.command,
                    qos,
                )
                for group, topics in ARM_TOPICS.items()
            }

        if subscribe_commands:
            for group_name, topics in ARM_TOPICS.items():
                self._control_node.create_subscription(
                    message_types.joint_state,
                    topics.command,
                    lambda msg, group=group_name: self._on_arm_command(group, msg),
                    qos,
                    callback_group=command_callback_group,
                )
                self._control_node.create_subscription(
                    message_types.joint_state,
                    topics.gripper_command,
                    lambda msg, group=group_name: self._on_gripper_command(group, msg),
                    qos,
                    callback_group=command_callback_group,
                )

        self._timer = None
        if create_timer:
            self._timer = node.create_timer(
                1.0 / rate_hz,
                self.publish_once,
                callback_group=timer_callback_group,
            )

    def publish_operator_action(self, action: str) -> None:
        """Publish one semantic event on /eva/operator_action.

        Args:
            action: Device-independent action accepted by EVA operator_control.
        """
        if action not in SUPPORTED_OPERATOR_ACTIONS:
            raise ValueError(f"unsupported operator action {action!r}")
        if self._operator_action_publisher is None:
            raise RuntimeError("operator action publisher is disabled")
        message = self._types.string()
        message.data = action
        self._operator_action_publisher.publish(message)

    def publish_hil(self, group_name: str, positions: np.ndarray) -> None:
        """Publish one HIL input JointState for a group.

        Args:
            group_name: "left_arm" or "right_arm".
            positions: [6] arm joint vector.
        """
        if group_name not in ARM_TOPICS:
            raise ValueError(f"unsupported HIL group {group_name!r}")
        vector = np.asarray(positions, dtype=np.float32).reshape(-1)
        if vector.shape != (6,):
            raise ValueError(f"{group_name} HIL input must have 6 positions, got {vector.shape}")
        if group_name not in self._hil_publishers:
            raise RuntimeError(f"HIL publisher is disabled for {group_name}")
        message = self._joint_state(group_name, vector, self._stamp())
        self._hil_publishers[group_name].publish(message)

    def publish_command(self, group_name: str, positions: np.ndarray) -> None:
        """Publish one direct motion target command for a group.

        Args:
            group_name: "left_arm" or "right_arm".
            positions: [6] arm joint vector.
        """
        if group_name not in ARM_TOPICS:
            raise ValueError(f"unsupported command group {group_name!r}")
        vector = np.asarray(positions, dtype=np.float32).reshape(-1)
        if vector.shape != (6,):
            raise ValueError(f"{group_name} command must have 6 positions, got {vector.shape}")
        if group_name not in self._command_publishers:
            raise RuntimeError(f"command publisher is disabled for {group_name}")
        message = self._joint_state(group_name, vector, self._stamp())
        self._command_publishers[group_name].publish(message)

    def publish_once(self) -> None:
        """Publish one full synthetic sensor frame."""
        stamp = self._stamp()
        arms, grippers = self._state.feedback_copy()
        commands = self._state.command_copy()

        for camera_name, publisher in self._camera_publishers.items():
            message = self._types.compressed_image()
            message.header.stamp = stamp
            message.format = "jpeg"
            frames = self._camera_frame_cache[camera_name]
            message.data = frames[self._frame_index % len(frames)]
            publisher.publish(message)

        for group_name, arm in arms.items():
            if group_name not in self._arm_publishers:
                continue
            self._arm_publishers[group_name].publish(self._joint_state(group_name, arm, stamp))
            self._gripper_publishers[group_name].publish(
                self._gripper_state(group_name, grippers[group_name], stamp)
            )
            self._eef_publishers[group_name].publish(
                self._eef_pose(group_name, arm, grippers[group_name], stamp)
            )
            self._command_publishers[group_name].publish(
                self._joint_state(group_name, commands[group_name], stamp)
            )

        self._frame_index += 1

    def _on_arm_command(self, group_name: str, msg: Any) -> None:
        self._state.apply_arm_command(group_name, np.asarray(msg.position, dtype=np.float32))

    def _on_gripper_command(self, group_name: str, msg: Any) -> None:
        position = list(msg.position)
        if not position:
            raise ValueError(f"{group_name} gripper command has no position")
        self._state.apply_gripper_command(group_name, float(position[0]))

    def _stamp(self) -> Any:
        return self._node.get_clock().now().to_msg()

    def _joint_state(self, group_name: str, positions: np.ndarray, stamp: Any) -> Any:
        message = self._types.joint_state()
        message.header.stamp = stamp
        message.name = list(JOINT_NAMES[group_name])
        message.position = [float(v) for v in positions]
        return message

    def _gripper_state(self, group_name: str, value: float, stamp: Any) -> Any:
        message = self._types.joint_state()
        message.header.stamp = stamp
        side = "left" if group_name == "left_arm" else "right"
        message.name = [f"{side}_gripper_joint"]
        message.position = [float(value)]
        return message

    def _eef_pose(self, group_name: str, positions: np.ndarray, gripper: float, stamp: Any) -> Any:
        message = self._types.pose_stamped()
        message.header.stamp = stamp
        sign = 1.0 if group_name == "left_arm" else -1.0
        message.pose.position.x = 0.35 + 0.015 * float(positions[0])
        message.pose.position.y = sign * (0.22 + 0.01 * float(positions[1]))
        message.pose.position.z = 0.45 + 0.005 * float(positions[2]) + 0.0002 * float(gripper)
        message.pose.orientation.w = 1.0
        message.pose.orientation.x = 0.0
        message.pose.orientation.y = 0.0
        message.pose.orientation.z = 0.0
        return message


def _jpeg_image(
    color: tuple[int, int, int],
    height: int,
    width: int,
    frame_index: int,
) -> bytes:
    """Build one synthetic RGB JPEG image with a moving bright band.

    Args:
        color: base RGB color.
        height: image height in pixels.
        width: image width in pixels.
        frame_index: running frame counter.

    Returns:
        data: JPEG bytes.
    """
    pulse = 0.5 + 0.5 * float(np.sin(frame_index * 0.05))
    rgb = np.asarray(color, dtype=np.float32) * (0.45 + 0.55 * pulse)
    image = np.empty((height, width, 3), dtype=np.uint8)
    image[:] = rgb.astype(np.uint8)
    band = int((frame_index * 4) % height)
    image[band : min(band + max(height // 18, 2), height), :, :] = 255

    buffer = io.BytesIO()
    Image.fromarray(image).save(buffer, format="JPEG", quality=85)
    return buffer.getvalue()


def make_ui_handler(api: R1LiteFakeControlApi) -> type[BaseHTTPRequestHandler]:
    """Create a request handler class bound to one fake control API.

    Args:
        api: UI command API.

    Returns:
        handler_class: BaseHTTPRequestHandler subclass for ThreadingHTTPServer.
    """

    class R1LiteFakeUiHandler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            if self.path == "/":
                body = UI_HTML.encode("utf-8")
                self.send_response(HTTPStatus.OK)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return
            if self.path == "/api/state":
                self._send_json(api.snapshot())
                return
            self.send_error(HTTPStatus.NOT_FOUND, "unknown path")

        def do_POST(self) -> None:
            try:
                payload = self._read_json()
                if self.path == "/api/operator":
                    self._send_json(api.operator_action(payload["action"]))
                    return
                if self.path == "/api/gripper":
                    self._send_json(api.gripper(payload["side"], payload["command"]))
                    return
                if self.path == "/api/joint":
                    self._send_json(
                        api.set_joint(
                            payload["group"],
                            int(payload["index"]),
                            float(payload["value"]),
                        )
                    )
                    return
                if self.path == "/api/joint_delta":
                    self._send_json(
                        api.adjust_joint(
                            payload["group"],
                            int(payload["index"]),
                            float(payload["delta"]),
                        )
                    )
                    return
                if self.path == "/api/command_delta":
                    self._send_json(
                        api.adjust_command(
                            payload["group"],
                            int(payload["index"]),
                            float(payload["delta"]),
                        )
                    )
                    return
                if self.path == "/api/sync_hil_to_feedback":
                    self._send_json(api.sync_hil_to_feedback())
                    return
                self.send_error(HTTPStatus.NOT_FOUND, "unknown path")
            except (KeyError, TypeError, ValueError) as exc:
                self._send_json({"error": str(exc)}, status=HTTPStatus.BAD_REQUEST)

        def _read_json(self) -> dict[str, Any]:
            length = int(self.headers.get("Content-Length", "0"))
            raw = self.rfile.read(length)
            return json.loads(raw.decode("utf-8"))

        def _send_json(self, payload: dict[str, Any], status: HTTPStatus = HTTPStatus.OK) -> None:
            body = json.dumps(payload).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    return R1LiteFakeUiHandler


UI_HTML = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>R1 Lite Fake</title>
<style>
:root {
  color-scheme: light;
  font-family: Inter, ui-sans-serif, system-ui, -apple-system,
    BlinkMacSystemFont, "Segoe UI", sans-serif;
  background: #f4f5f7;
  color: #18202a;
}
body {
  margin: 0;
}
main {
  max-width: 980px;
  margin: 0 auto;
  padding: 18px;
}
h1 {
  margin: 0 0 14px;
  font-size: 22px;
  font-weight: 650;
}
.toolbar, .arms {
  display: grid;
  gap: 12px;
}
.toolbar {
  grid-template-columns: repeat(auto-fit, minmax(170px, 1fr));
  margin-bottom: 16px;
}
.panel {
  background: #ffffff;
  border: 1px solid #d8dde5;
  border-radius: 8px;
  padding: 12px;
}
.panel h2 {
  margin: 0 0 10px;
  font-size: 14px;
  font-weight: 650;
}
.buttons {
  display: flex;
  flex-wrap: wrap;
  gap: 8px;
}
button {
  min-height: 34px;
  border: 1px solid #aeb7c2;
  border-radius: 6px;
  background: #f8fafc;
  color: #18202a;
  font: inherit;
  font-size: 13px;
  cursor: pointer;
}
button:hover {
  background: #edf2f7;
}
.buttons button {
  min-width: 52px;
  padding: 0 10px;
}
.arms {
  grid-template-columns: repeat(auto-fit, minmax(320px, 1fr));
}
.joint {
  display: grid;
  grid-template-columns: 34px 34px minmax(120px, 1fr) 34px 62px;
  align-items: center;
  gap: 8px;
  min-height: 42px;
}
.joint span {
  font-variant-numeric: tabular-nums;
  font-size: 12px;
}
.joint input {
  width: 100%;
}
.status {
  margin-top: 8px;
  color: #526170;
  font-size: 12px;
  font-variant-numeric: tabular-nums;
}
</style>
</head>
<body>
<main>
  <h1>R1 Lite Fake</h1>
  <section class="toolbar">
    <div class="panel">
      <h2>Operator</h2>
      <div class="buttons">
        <button data-action="rollout_toggle">A · Run/Save</button>
        <button data-action="rollout_reset">B · Reset</button>
        <button data-action="intervention_toggle">X · Takeover/Abandon</button>
        <button data-action="intervention_accept">Y · Accept</button>
        <button id="sync-hil">Sync HIL</button>
      </div>
    </div>
    <div class="panel">
      <h2>Left Gripper</h2>
      <div class="buttons">
        <button data-gripper-side="left" data-gripper-command="open">Open</button>
        <button data-gripper-side="left" data-gripper-command="close">Close</button>
      </div>
    </div>
    <div class="panel">
      <h2>Right Gripper</h2>
      <div class="buttons">
        <button data-gripper-side="right" data-gripper-command="open">Open</button>
        <button data-gripper-side="right" data-gripper-command="close">Close</button>
      </div>
    </div>
  </section>
  <section id="arms" class="arms"></section>
</main>
<script>
const groups = [
  ["left_arm", "Left Arm"],
  ["right_arm", "Right Arm"],
];
const jointStep = 0.05;

function postJson(path, payload) {
  return fetch(path, {
    method: "POST",
    headers: {"Content-Type": "application/json"},
    body: JSON.stringify(payload),
  }).then((response) => {
    if (!response.ok) {
      return response.json().then((body) => {
        throw new Error(body.error || response.statusText);
      });
    }
    return response.json();
  });
}

function buildArm(group, title) {
  const panel = document.createElement("div");
  panel.className = "panel";
  panel.innerHTML = `<h2>${title}</h2>`;
  for (let index = 0; index < 6; index += 1) {
    const row = document.createElement("div");
    row.className = "joint";
    row.innerHTML = `
      <span>J${index + 1}</span>
      <button data-delta="-1">-</button>
      <input type="range" min="-3.14" max="3.14" step="0.01" value="0">
      <button data-delta="1">+</button>
      <span data-value>0.00</span>
    `;
    const slider = row.querySelector("input");
    const value = row.querySelector("[data-value]");
    slider.addEventListener("input", () => {
      value.textContent = Number(slider.value).toFixed(2);
      postJson("/api/joint", {group, index, value: Number(slider.value)}).catch(console.error);
    });
    row.querySelectorAll("button").forEach((button) => {
      button.addEventListener("click", () => {
        postJson("/api/joint_delta", {
          group,
          index,
          delta: Number(button.dataset.delta) * jointStep,
        }).then(refresh).catch(console.error);
      });
    });
    panel.appendChild(row);
  }
  const status = document.createElement("div");
  status.className = "status";
  status.dataset.status = group;
  panel.appendChild(status);
  return panel;
}

function install() {
  const arms = document.querySelector("#arms");
  groups.forEach(([group, title]) => arms.appendChild(buildArm(group, title)));
  document.querySelectorAll("[data-action]").forEach((button) => {
    button.addEventListener("click", () => {
      postJson("/api/operator", {action: button.dataset.action}).catch(console.error);
    });
  });
  document.querySelector("#sync-hil").addEventListener("click", () => {
    postJson("/api/sync_hil_to_feedback", {}).then(refresh).catch(console.error);
  });
  document.querySelectorAll("[data-gripper-side]").forEach((button) => {
    button.addEventListener("click", () => postJson("/api/gripper", {
      side: button.dataset.gripperSide,
      command: button.dataset.gripperCommand,
    }).catch(console.error));
  });
}

function refresh() {
  return fetch("/api/state")
    .then((response) => response.json())
    .then((state) => {
      groups.forEach(([group]) => {
        const panel = document.querySelector(`[data-status="${group}"]`).parentElement;
        panel.querySelectorAll(".joint").forEach((row, index) => {
          const slider = row.querySelector("input");
          const value = row.querySelector("[data-value]");
          const next = state.hil[group][index];
          slider.value = next;
          value.textContent = Number(next).toFixed(2);
        });
        const feedback = state.feedback[group];
        const joints = feedback.joints.map((v) => Number(v).toFixed(2)).join(" ");
        const gripper = Number(feedback.gripper).toFixed(1);
        document.querySelector(`[data-status="${group}"]`).textContent =
          `feedback ${joints} | gripper ${gripper}`;
      });
    });
}

install();
refresh();
setInterval(refresh, 500);
</script>
</body>
</html>
"""


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="ROS2 fake R1 Lite node with local UI.")
    parser.add_argument("--role", choices=("all", "cameras", "robot"), default="all")
    parser.add_argument("--rate", type=float, default=30.0)
    parser.add_argument("--ui-host", default="127.0.0.1")
    parser.add_argument("--ui-port", type=int, default=8765)
    parser.add_argument("--image-height", type=int, default=360)
    parser.add_argument("--image-width", type=int, default=640)
    parser.add_argument("--no-open-ui", action="store_true")
    return parser


def _make_live_qos(depth: int = 10) -> Any:
    from rclpy.qos import QoSDurabilityPolicy, QoSProfile, QoSReliabilityPolicy

    return QoSProfile(
        depth=depth,
        reliability=QoSReliabilityPolicy.BEST_EFFORT,
        durability=QoSDurabilityPolicy.VOLATILE,
    )


def _run_sensor_publish_loop(
    bridge: R1LiteRos2Fake,
    rate_hz: float,
    stop_event: threading.Event,
    is_running=None,
) -> None:
    """Publish fake sensor frames at a fixed rate outside the ROS2 executor.

    Args:
        bridge: ROS2 fake bridge whose ``publish_once`` emits one sensor frame.
        rate_hz: target sensor publish rate.
        stop_event: set to stop the loop.
    """
    period_s = 1.0 / rate_hz
    next_tick = time.monotonic()
    while not stop_event.is_set() and (is_running is None or is_running()):
        bridge.publish_once()
        next_tick += period_s
        delay_s = next_tick - time.monotonic()
        if delay_s <= 0.0:
            next_tick = time.monotonic()
            continue
        stop_event.wait(delay_s)


def main() -> None:
    """Run the ROS2 fake node and embedded HTTP UI until interrupted."""
    args = build_arg_parser().parse_args()

    import rclpy
    from geometry_msgs.msg import PoseStamped
    from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
    from rclpy.executors import MultiThreadedExecutor
    from rclpy.node import Node
    from sensor_msgs.msg import CompressedImage, JointState
    from std_msgs.msg import String

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    if not rclpy.ok():
        rclpy.init()

    publish_cameras = args.role in {"all", "cameras"}
    publish_robot = args.role in {"all", "robot"}
    subscribe_commands = args.role in {"all", "robot"}
    if args.role == "cameras":
        sensor_node = Node("eva_r1lite_fake_cameras")
        control_node = None
    elif args.role == "robot":
        sensor_node = Node("eva_r1lite_fake_robot")
        control_node = Node("eva_r1lite_fake_control")
    else:
        sensor_node = Node("eva_r1lite_fake_sensor")
        control_node = Node("eva_r1lite_fake_control")
    state = R1LiteFakeState()
    command_callback_group = MutuallyExclusiveCallbackGroup()
    timer_callback_group = MutuallyExclusiveCallbackGroup()
    bridge = R1LiteRos2Fake(
        node=sensor_node,
        control_node=control_node,
        message_types=Ros2MessageTypes(
            compressed_image=CompressedImage,
            joint_state=JointState,
            pose_stamped=PoseStamped,
            string=String,
        ),
        state=state,
        rate_hz=args.rate,
        image_height=args.image_height,
        image_width=args.image_width,
        qos=_make_live_qos(),
        camera_qos=_make_live_qos(depth=1),
        command_callback_group=command_callback_group,
        timer_callback_group=timer_callback_group,
        create_timer=False,
        publish_cameras=publish_cameras,
        publish_robot=publish_robot,
        subscribe_commands=subscribe_commands,
    )
    server = None
    if publish_robot:
        api = R1LiteFakeControlApi(state, bridge)
        server = ThreadingHTTPServer((args.ui_host, args.ui_port), make_ui_handler(api))
        server_thread = threading.Thread(target=server.serve_forever, daemon=True)
        server_thread.start()

        url = f"http://{args.ui_host}:{args.ui_port}/"
        sensor_node.get_logger().info(f"R1 Lite fake UI: {url}")
        if not args.no_open_ui:
            opened = webbrowser.open(url)
            if not opened:
                sensor_node.get_logger().warning(f"Could not open browser automatically: {url}")

    stop_event = threading.Event()
    sensor_thread = threading.Thread(
        target=_run_sensor_publish_loop,
        args=(bridge, args.rate, stop_event, rclpy.ok),
        daemon=True,
    )
    sensor_thread.start()
    executor = None
    if control_node is not None:
        executor = MultiThreadedExecutor(num_threads=4)
        executor.add_node(control_node)

    try:
        if executor is None:
            while rclpy.ok():
                time.sleep(0.5)
        else:
            executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        stop_event.set()
        sensor_thread.join(timeout=2.0)
        if executor is not None:
            executor.shutdown()
        if server is not None:
            server.shutdown()
            server.server_close()
        if control_node is not None:
            control_node.destroy_node()
        sensor_node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
