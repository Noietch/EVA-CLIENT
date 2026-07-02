import time

import cv2


class GDKRes:
    kSuccess = 0
    kError = 1


class CameraType:
    kHeadColor = 0
    kHandLeftColor = 1
    kHandRightColor = 2


def gdk_init():
    print("[MOCK GDK] Initialized successfully.")
    return GDKRes.kSuccess


def gdk_release():
    print("[MOCK GDK] Released successfully.")
    return GDKRes.kSuccess


class MockImageMsg:
    def __init__(self, data, width=640, height=480):
        self.width = width
        self.height = height
        self.data = data


class Camera:
    def __init__(self):
        print("[MOCK GDK] Camera instantiated.")
        self.frame_cache = {}

        video_paths = {
            CameraType.kHeadColor: "observation.images.top_head/episode_000000.mp4",
            CameraType.kHandLeftColor: "observation.images.hand_left/episode_000000.mp4",
            CameraType.kHandRightColor: "observation.images.hand_right/episode_000000.mp4",
        }

        for cam_type, path in video_paths.items():
            cap = cv2.VideoCapture(path)
            if cap.isOpened():
                ret, frame = cap.read()
                if ret:
                    success, encoded_img = cv2.imencode(".jpg", frame)
                    if success:
                        self.frame_cache[cam_type] = {
                            "data": encoded_img.tobytes(),
                            "width": frame.shape[1],
                            "height": frame.shape[0],
                        }
                        print(
                            f"[MOCK GDK] Successfully cached first frame for cam_type {cam_type}."
                        )
                    else:
                        print(f"[MOCK GDK] Failed to encode frame for cam_type {cam_type}.")
                else:
                    print(f"[MOCK GDK] Failed to read first frame from {path}.")
            else:
                print(f"[MOCK GDK] Failed to open video file {path}.")

            cap.release()

    def get_latest_image(self, cam_type, timeout_ms=10.0):
        time.sleep(0.01)

        cached = self.frame_cache.get(cam_type)
        if cached:
            return MockImageMsg(data=cached["data"], width=cached["width"], height=cached["height"])
        return None

    def close_camera(self):
        print("[MOCK GDK] Camera closed.")


class JointServoControlReq:
    def __init__(self):
        self.control_period = 0.01
        self.joint_names = []
        self.joint_positions = []
        self.joint_velocities = []


class Robot:
    def __init__(self):
        print("[MOCK GDK] Robot instantiated.")

        self.joint_names = [
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

        initial_qpos = [
            0.739033,
            -0.717023,
            -1.524419,
            -1.537612,
            0.27811,
            -0.925845,
            -0.839257,
            -0.785,
            -0.739033,
            -0.717023,
            1.524419,
            -1.537612,
            -0.27811,
            -0.925845,
            0.839257,
            -0.785,
            0.0,
            0.0,
            0.11464,
            -0.83423,
            1.2172,
            0.10025,
            0.0,
            0.0,
        ]

        self.current_state = dict(zip(self.joint_names, initial_qpos, strict=False))

    def get_joint_states(self):
        states = []
        for name in self.joint_names:
            current_pos = self.current_state[name]
            states.append(
                {
                    "name": name,
                    "position": current_pos,
                    "velocity": 0.0,
                    "effort": 0.0,
                    "motor_position": current_pos,
                    "motor_velocity": 0.0,
                    "motor_current": 0.0,
                    "error_code": 0,
                }
            )

        return {
            "nums": len(self.joint_names),
            "timestamp": int(time.time() * 1e9),
            "states": states,
        }

    def get_end_state(self):
        return {
            "left_end_state": {
                "names": ["idx31_gripper_l_inner_joint1"],
                "end_states": [{"position": self.current_state["idx31_gripper_l_inner_joint1"]}],
            },
            "right_end_state": {
                "names": ["idx71_gripper_r_inner_joint1"],
                "end_states": [{"position": self.current_state["idx71_gripper_r_inner_joint1"]}],
            },
        }

    def joint_servo_control(self, req, enable_low_latency=False):

        for name, target_pos in zip(req.joint_names, req.joint_positions, strict=False):
            if name in self.current_state:
                self.current_state[name] = target_pos

        return 0
