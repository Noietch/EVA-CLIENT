"""Forward AgileX master-arm ROS qpos to an EVA Sim ZMQ action endpoint."""

from __future__ import annotations

import argparse
import os
import signal
import subprocess
import threading
import time

import numpy as np
import zmq
from openpi_client import msgpack_numpy

parser = argparse.ArgumentParser()
parser.add_argument("--action-endpoint", required=True)
parser.add_argument("--left-topic", required=True)
parser.add_argument("--right-topic", required=True)
parser.add_argument("--ros-python", required=True)
args = parser.parse_args()


def stop(*_) -> None:
    raise SystemExit


signal.signal(signal.SIGTERM, stop)

positions: list[np.ndarray | None] = [None, None]
lock = threading.Lock()
packer = msgpack_numpy.Packer()
socket = zmq.Context.instance().socket(zmq.PUB)
socket.connect(args.action_endpoint)


def forward(index: int, values: list[str]) -> None:
    with lock:
        positions[index] = np.asarray(values, dtype=np.float32)
        left, right = positions
        if left is None or right is None:
            return
        action = np.concatenate((left, right))
    socket.send(
        packer.pack({"t": time.monotonic(), "action": action, "target": "real"})
    )

env = os.environ | {
    "PYTHONPATH": os.path.dirname(__file__) + ":/opt/ros/noetic/lib/python3/dist-packages"
}
receiver = subprocess.Popen(
    [
        args.ros_python,
        "-m",
        "ros_master_teleop_receiver",
        "--left-topic",
        args.left_topic,
        "--right-topic",
        args.right_topic,
    ],
    env=env,
    stdout=subprocess.PIPE,
    text=True,
)
try:
    for line in receiver.stdout:
        index, *values = line.split()
        if len(values) == 7:
            forward(int(index), values)
finally:
    receiver.terminate()
    receiver.wait()
