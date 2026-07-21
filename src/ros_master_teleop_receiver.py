"""Subscribe to ROS master arms and write qpos samples to stdout."""

from __future__ import annotations

import argparse

import rospy
from sensor_msgs.msg import JointState


parser = argparse.ArgumentParser()
parser.add_argument("--left-topic", required=True)
parser.add_argument("--right-topic", required=True)
args = parser.parse_args()


def forward(index: int, message: JointState) -> None:
    try:
        print(index, *message.position, flush=True)
    except BrokenPipeError:
        rospy.signal_shutdown("bridge exited")


rospy.init_node("eva_client_master_teleop", anonymous=True)
rospy.Subscriber(args.left_topic, JointState, lambda message: forward(0, message), queue_size=1)
rospy.Subscriber(args.right_topic, JointState, lambda message: forward(1, message), queue_size=1)
rospy.spin()
