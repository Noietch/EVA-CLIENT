from bimanual import SingleArm
from typing import Dict, Any
import numpy as np
import numpy.typing as npt
import curses
import time

arm_config: Dict[str, Any] = {
    "can_port": "can1",
    "type": 0,
    # Add necessary configuration parameters for the left arm
}
single_arm = SingleArm(arm_config)

# 用curses捕获键盘输入
def keyboard_control(stdscr):

    curses.curs_set(0)  # 不显示光标
    stdscr.nodelay(1)   # 设置为非阻塞模式
    stdscr.timeout(10)  # 设置读取键盘的超时
    global target_pose
    curses.mousemask(0)  # 禁用鼠标事件
    xyzrpy = np.zeros(6)
    gripper =0
    # “热爱无需多言"
    big_text = [
        "  AAAAA        RRRRR         X   X     ",
        " A     A       R    R         X X      ",
        " AAAAAAA       RRRRR           X       ",
        " A     A       R  R           X X      ",
        " A     A       R   RR        X   X     ",
        " A     A       R    R       X     X    "
    ]
    
    while True:
        key = stdscr.getch()  # 获取键盘输入
        stdscr.clear()
        # 对每个返回值中的元素进行格式化
        ee_pose = single_arm.get_ee_pose_xyzrpy()
        joint_pos = single_arm.get_joint_positions()
        joint_vel = single_arm.get_joint_velocities()
        joint_curr = single_arm.get_joint_currents()

        # 格式化并显示
        stdscr.addstr(0, 0, f"EE_POSE: [{' '.join([f'{val:.3f}' for val in ee_pose])}]")
        stdscr.addstr(2, 0, f"JOINT_POS: [{' '.join([f'{val:.3f}' for val in joint_pos])}]")
        stdscr.addstr(4, 0, f"JOINT_VEL: [{' '.join([f'{val:.3f}' for val in joint_vel])}]")
        stdscr.addstr(6, 0, f"JOINT_CURR: [{' '.join([f'{val:.3f}' for val in joint_curr])}]")
        
        if key == ord('q'):  # 按 'q' 退出程序
            break
        if key == -1:  # 按 'q' 退出程序
            continue     
        elif key == ord('i'): 
            single_arm.gravity_compensation()
            value=single_arm.get_ee_pose_xyzrpy()

        elif key == ord('w'): 
            xyzrpy[0] += 0.005  # 机械臂前移
            single_arm.set_ee_pose_xyzrpy(xyzrpy=xyzrpy)
        elif key == ord('s'): 
            xyzrpy[0] -= 0.005  # 机械臂后移
            single_arm.set_ee_pose_xyzrpy(xyzrpy=xyzrpy)
        elif key == ord('a'):
            xyzrpy[1] += 0.005  # 机械臂左移
            single_arm.set_ee_pose_xyzrpy(xyzrpy=xyzrpy)
        elif key == ord('d'):  
            xyzrpy[1] -= 0.005  # 机械臂右移
            single_arm.set_ee_pose_xyzrpy(xyzrpy=xyzrpy)
        elif key == curses.KEY_UP: 
            xyzrpy[2] += 0.005  # 机械臂上移
            single_arm.set_ee_pose_xyzrpy(xyzrpy=xyzrpy)
        elif key == curses.KEY_DOWN: 
            xyzrpy[2] -= 0.005  # 机械臂下移
            single_arm.set_ee_pose_xyzrpy(xyzrpy=xyzrpy)
        elif key == curses.KEY_LEFT: 
            xyzrpy[1] += 0.005  # 机械臂左移
            single_arm.set_ee_pose_xyzrpy(xyzrpy=xyzrpy)  
        elif key == curses.KEY_RIGHT: 
             xyzrpy[1] -= 0.005  # 机械臂右移
             single_arm.set_ee_pose_xyzrpy(xyzrpy=xyzrpy)  
        elif key == ord(','): 
            xyzrpy[5] += 0.02  # 机械臂yaw减少
            single_arm.set_ee_pose_xyzrpy(xyzrpy=xyzrpy)
        elif key == ord('/'): 
            xyzrpy[5] -= 0.02  # 机械臂yaw增加
            single_arm.set_ee_pose_xyzrpy(xyzrpy=xyzrpy)
        elif key == ord('m'): 
            xyzrpy[3] += 0.02  # 机械臂roll增加
            single_arm.set_ee_pose_xyzrpy(xyzrpy=xyzrpy)
        elif key == ord('n'): 
            xyzrpy[3] -= 0.02  # 机械臂roll减少
            single_arm.set_ee_pose_xyzrpy(xyzrpy=xyzrpy)
        elif key == ord('l'): 
            xyzrpy[4] += 0.02  # 机械臂pitch增加
            single_arm.set_ee_pose_xyzrpy(xyzrpy=xyzrpy)
        elif key == ord('.'): 
            xyzrpy[4] -= 0.02  # 机械臂pitch减少
            single_arm.set_ee_pose_xyzrpy(xyzrpy=xyzrpy)
        elif key == ord('c'): 
            gripper -= 0.2  # 闭合
            single_arm.set_catch_pos(pos=gripper)
        elif key == ord('o'): 
            gripper += 0.2  # 张开
            single_arm.set_catch_pos(pos=gripper)
        elif key == ord('r'): 
            xyzrpy = np.zeros(6)
            single_arm.go_home()
            print('回到原点\n')

        # height, width = stdscr.getmaxyx()

        # 更新屏幕显示当前目标位姿
        # stdscr.addstr(0, 0, f"Current Target Pose: {xyzrpy}")
        # for i, line in enumerate(big_text):
        #     stdscr.addstr(height // 2 - 3 + i, (width - len(line)) // 2, line)

        stdscr.refresh()

if __name__ == "__main__":
    curses.wrapper(keyboard_control)
