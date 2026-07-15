#!/usr/bin/env python3
"""Publish CAT Joy-Con X/Y Joy messages as EVA operator button events."""

from __future__ import annotations

import argparse
from collections.abc import Sequence

import rclpy
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from rclpy.utilities import remove_ros_args
from sensor_msgs.msg import Joy
from std_msgs.msg import String

RIGHT_JOY_BUTTONS = {
    2: "x",
    3: "y",
    4: "right_gripper_open",
    5: "right_gripper_close",
}

LEFT_JOY_BUTTONS = {
    4: "left_gripper_open",
    5: "left_gripper_close",
}


class JoyButtonEdgeAdapter:
    """Convert selected Joy button rising edges into operator button labels."""

    def __init__(self, button_map: dict[int, str]) -> None:
        self.button_map = dict(button_map)
        self.previous_buttons: list[int] = []

    def update(self, buttons: Sequence[int]) -> list[str]:
        current = [int(value) for value in buttons]
        labels: list[str] = []
        for index, label in self.button_map.items():
            was_pressed = (
                bool(self.previous_buttons[index]) if index < len(self.previous_buttons) else False
            )
            is_pressed = bool(current[index]) if index < len(current) else False
            if is_pressed and not was_pressed:
                labels.append(label)
        self.previous_buttons = current
        return labels


class CatOperatorButtonNode(Node):
    """Subscribe to Joy-Con Joy messages and publish operator event labels."""

    def __init__(self, left_joy_topic: str, right_joy_topic: str, output_topic: str) -> None:
        super().__init__("eva_cat_takeover_button_agent")
        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            durability=DurabilityPolicy.VOLATILE,
        )
        self.left_adapter = JoyButtonEdgeAdapter(LEFT_JOY_BUTTONS)
        self.right_adapter = JoyButtonEdgeAdapter(RIGHT_JOY_BUTTONS)
        self.pub = self.create_publisher(String, output_topic, 10)
        self.create_subscription(
            Joy,
            left_joy_topic,
            lambda msg: self.on_joy(self.left_adapter, msg),
            qos,
        )
        self.create_subscription(
            Joy,
            right_joy_topic,
            lambda msg: self.on_joy(self.right_adapter, msg),
            qos,
        )
        self.get_logger().info(
            f"Publishing CAT operator buttons from {left_joy_topic}, "
            f"{right_joy_topic} to {output_topic}"
        )

    def publish_button(self, label: str) -> None:
        message = String()
        message.data = label
        self.pub.publish(message)
        self.get_logger().info(f"operator button {label}")

    def on_joy(self, adapter: JoyButtonEdgeAdapter, msg: Joy) -> None:
        for label in adapter.update(msg.buttons):
            self.publish_button(label)


def parse_args(args: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--left-joy-topic", default="/joy_left")
    parser.add_argument("--right-joy-topic", default="/joy_right")
    parser.add_argument("--topic", default="/eva/operator_button")
    return parser.parse_args(remove_ros_args(args)[1:])


def main(args: list[str] | None = None) -> None:
    parsed = parse_args(args)
    rclpy.init(args=args)
    node = CatOperatorButtonNode(parsed.left_joy_topic, parsed.right_joy_topic, parsed.topic)
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
