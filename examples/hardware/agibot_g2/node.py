from __future__ import annotations

import argparse
import logging
import signal
import threading
import time

import cv2
import numpy as np
import zmq

try:
    import agibot_gdk
    # import agibot_gdk_mock as agibot_gdk
except ImportError:
    raise ImportError("Please install agibot_gdk first.") from None

from transport.zmq import (
    WireObservation,
    pack_observation,
    unpack_action,
)

logger = logging.getLogger(__name__)


JOINT_NAMES = [
    "idx21_arm_l_joint1",
    "idx22_arm_l_joint2",
    "idx23_arm_l_joint3",
    "idx24_arm_l_joint4",
    "idx25_arm_l_joint5",
    "idx26_arm_l_joint6",
    "idx27_arm_l_joint7",
    "idx31_gripper_l_inner_joint1",
    "idx61_arm_r_joint1",
    "idx62_arm_r_joint2",
    "idx63_arm_r_joint3",
    "idx64_arm_r_joint4",
    "idx65_arm_r_joint5",
    "idx66_arm_r_joint6",
    "idx67_arm_r_joint7",
    "idx71_gripper_r_inner_joint1",
    "idx11_head_joint1",
    "idx12_head_joint2",
    "idx13_head_joint3",
    "idx01_body_joint1",
    "idx02_body_joint2",
    "idx03_body_joint3",
    "idx04_body_joint4",
    "idx05_body_joint5",
]


class AgibotHardwareNode:
    def __init__(
        self,
        obs_endpoint: str,
        action_endpoint: str,
        obs_hz: float = 30.0,
        control_hz: float = 100.0,
    ):
        self._is_running = True
        self._obs_hz = obs_hz
        self._control_hz = control_hz

        self._ctx = zmq.Context.instance()

        self._obs_pub = self._ctx.socket(zmq.PUB)
        self._obs_pub.bind(obs_endpoint)

        self._action_sub = self._ctx.socket(zmq.SUB)
        self._action_sub.bind(action_endpoint)
        self._action_sub.setsockopt(zmq.SUBSCRIBE, b"")
        self._action_sub.setsockopt(zmq.RCVTIMEO, 0)

        if agibot_gdk.gdk_init() != agibot_gdk.GDKRes.kSuccess:
            raise RuntimeError("GDK Initialization Failed")
        logger.info("GDK initialized successfully.")

        self.robot = agibot_gdk.Robot()
        self.camera = agibot_gdk.Camera()
        time.sleep(2.0)

        self.cam_mapping = {
            agibot_gdk.CameraType.kHeadColor: "top_head",
            agibot_gdk.CameraType.kHandLeftColor: "hand_left",
            agibot_gdk.CameraType.kHandRightColor: "hand_right",
        }

        self.target_qpos = self._read_current_qpos()

        self._obs_thread = threading.Thread(target=self._observation_loop, daemon=True)
        self._obs_thread.start()

    def _read_current_qpos(self) -> np.ndarray:
        name_to_pos = {}

        joint_states = self.robot.get_joint_states()
        for state in joint_states.get("states", []):
            name_to_pos[state["name"]] = state["motor_position"]

        end_state = self.robot.get_end_state()
        for side in ["left_end_state", "right_end_state"]:
            if side in end_state:
                side_state = end_state[side]
                for name, joint in zip(
                    side_state.get("names", []), side_state.get("end_states", []), strict=False
                ):
                    name_to_pos[name] = joint["position"]

        qpos = np.zeros(len(JOINT_NAMES), dtype=np.float32)
        for i, name in enumerate(JOINT_NAMES):
            qpos[i] = name_to_pos[name]

        return qpos

    def _get_bgr_image(self, cam_type) -> np.ndarray | None:
        img_msg = self.camera.get_latest_image(cam_type, timeout_ms=10.0)
        if img_msg is None or img_msg.data is None:
            return None

        img_array = np.frombuffer(img_msg.data, dtype=np.uint8)
        return cv2.imdecode(img_array, cv2.IMREAD_COLOR)

    def _observation_loop(self):
        period = 1.0 / self._obs_hz
        next_tick = time.monotonic()

        while self._is_running:
            images = {}
            for cam_type, key_name in self.cam_mapping.items():
                img = self._get_bgr_image(cam_type)
                if img is None:
                    raise ValueError(f"Failed to fetch image for {key_name}")
                images[key_name] = img

            current_qpos = self._read_current_qpos()
            state_dict = {
                "left_arm": current_qpos[0:8],
                "right_arm": current_qpos[8:16],
                "head": current_qpos[16:19],
                "body": current_qpos[19:24],
            }

            obs = WireObservation(
                t=time.monotonic(),
                images=images,
                state=state_dict,
                eef=None,
                action=None,
                action_eef=None,
            )
            self._obs_pub.send(pack_observation(obs))

            next_tick += period
            sleep_s = next_tick - time.monotonic()
            if sleep_s > 0:
                time.sleep(sleep_s)
            else:
                next_tick = time.monotonic()

    def _drain_actions(self):
        while True:
            try:
                payload = self._action_sub.recv(zmq.NOBLOCK)
                action_msg = unpack_action(payload)
                if action_msg.target == "real":
                    self.target_qpos = np.asarray(action_msg.action, dtype=float)
            except zmq.Again:
                break
            except Exception as e:
                logger.warning(f"Action parse error: {e}")
                break

    def serve_forever(self):
        control_period = 1.0 / self._control_hz
        logger.info(f"Entering {self._control_hz}Hz servo control loop...")

        next_tick = time.monotonic()
        while self._is_running:
            self._drain_actions()

            req = agibot_gdk.JointServoControlReq()
            req.control_period = control_period
            req.joint_names = JOINT_NAMES
            req.joint_positions = self.target_qpos.tolist()

            try:
                if self.robot.joint_servo_control(req) != 0:
                    logger.warning("joint_servo_control returned non-zero error code.")
            except Exception as e:
                logger.error(f"Servo control exception: {e}")

            next_tick += control_period
            sleep_s = next_tick - time.monotonic()
            if sleep_s > 0:
                time.sleep(sleep_s)
            else:
                next_tick = time.monotonic()

    def close(self):
        logger.info("Shutting down Agibot Node...")
        self._is_running = False

        if hasattr(self, "_obs_thread"):
            self._obs_thread.join(timeout=2.0)

        try:
            self.camera.close_camera()
        except Exception as e:
            logger.debug(f"Error closing camera: {e}")

        agibot_gdk.gdk_release()
        self._action_sub.close(linger=0)
        self._obs_pub.close(linger=0)


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description="Agibot GDK ZMQ Node")
    parser.add_argument("--obs-endpoint", default="tcp://127.0.0.1:5555")
    parser.add_argument("--action-endpoint", default="tcp://127.0.0.1:5556")
    args = parser.parse_args()

    node = AgibotHardwareNode(args.obs_endpoint, args.action_endpoint)

    def _stop(*_args):
        node.close()

    signal.signal(signal.SIGINT, _stop)
    signal.signal(signal.SIGTERM, _stop)

    try:
        node.serve_forever()
    finally:
        node.close()


if __name__ == "__main__":
    main()
