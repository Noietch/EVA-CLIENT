from bimanual import BimanualArm
from bimanual.config import get_dual_arm_config
import numpy as np


def test_dual_arm(dual_arm: BimanualArm):
    #single_arm.go_home()
    while(1):
        xyzrpy = np.array([0.0, 0.0, 0.1,0.0, 0.0, 0.0])  # x, y, z 位置

        dual_arm.set_ee_pose_xyzrpy({
            "left": xyzrpy,
            "right": xyzrpy,
        })

        while(1):
            print("testing ...")

        #print(dual_arm.get_ee_pose())
        #print(dual_arm.get_joint_positions())

        #positions = [0.5, 1.0, -0.5]  # 指定每个关节的位置
        #joint_names = ["joint1", "joint2", "joint3"]  # 对应关节的名称

        #success = single_arm0.set_joint_positions(positions=positions, joint_names=joint_names)

if __name__ == "__main__":
    left_arm_config, right_arm_config = get_dual_arm_config()
    dual_arm = BimanualArm(left_arm_config, right_arm_config)
    test_dual_arm(dual_arm)
