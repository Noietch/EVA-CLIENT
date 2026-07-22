#!/usr/bin/env python3
"""Reset the R1 Lite torso by publishing its configured fixed pose."""

from __future__ import annotations

import time
from typing import Any

import rclpy
from sensor_msgs.msg import JointState

NODE_NAME = "reset_r1_lite_torso"
TORSO_TOPIC = "/motion_target/target_joint_state_torso"
TORSO_JOINTS = ("torso_joint1", "torso_joint2", "torso_joint3")
TORSO_POSITION = (-0.657, 1.516, 0.845)
PUBLISH_COUNT = 100
PUBLISH_RATE_HZ = 30.0
QUEUE_SIZE = 10


def make_torso_joint_state(stamp: Any) -> JointState:
    """Create the fixed torso JointState command.

    Args:
        stamp: ROS message timestamp.

    Returns:
        message: JointState with the three torso joint names and target positions.
    """
    message = JointState()
    message.header.stamp = stamp
    message.name = list(TORSO_JOINTS)
    message.position = list(TORSO_POSITION)
    return message


def main() -> None:
    """Publish the fixed R1 Lite torso pose 100 times and exit."""
    rclpy.init()
    node = rclpy.create_node(NODE_NAME)
    publisher = node.create_publisher(JointState, TORSO_TOPIC, QUEUE_SIZE)
    period_sec = 1.0 / PUBLISH_RATE_HZ

    for _ in range(PUBLISH_COUNT):
        stamp = node.get_clock().now().to_msg()
        publisher.publish(make_torso_joint_state(stamp))
        time.sleep(period_sec)

    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
