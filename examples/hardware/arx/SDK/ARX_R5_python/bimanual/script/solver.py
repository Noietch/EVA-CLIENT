import kinematic_solver as solver
import numpy as np

_instance = None

def forward_kinematics(joint_angles):
    global _instance
    if _instance is None:
        _instance = solver.KinematicSolver()
    return _instance.forward_kinematics(joint_angles)

def inverse_kinematics(target_pose):
    global _instance
    if _instance is None:
        _instance = solver.KinematicSolver()
    return _instance.inverse_kinematics(target_pose)

