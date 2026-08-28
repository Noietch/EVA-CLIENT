from bimanual import SingleArm
from typing import Dict, Any
import numpy as np

def test_single_arm(single_arm: SingleArm, duration: float = 10.0, dt: float = 0.01):
    #single_arm.go_home()
    while(1):
        position = np.array([0.0, 0.0, 0.1])  # x, y, z 位置
        quaternion = np.array([1.0, 0.0, 0.0, 0.0])  # 四元数表示方向

        success = single_arm.set_ee_pose(pos=position, quat=quaternion)
        
        while(1):
            print("testing ...")

        #print(single_arm.get_ee_pose())
        #print(single_arm.get_joint_positions())
         
        #positions = [0.5, 1.0, -0.5]  # 指定每个关节的位置
        #joint_names = ["joint1", "joint2", "joint3"]  # 对应关节的名称

        #success = single_arm.set_joint_positions(positions=positions, joint_names=joint_names)
        
if __name__ == "__main__":
    arm_config: Dict[str, Any] = {
        "can_port": "can1",
        "type": 0,
        # Add necessary configuration parameters for the left arm
    }
    single_arm = SingleArm(arm_config)
    test_single_arm(single_arm)