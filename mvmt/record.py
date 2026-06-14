import rtde_control
import rtde_receive
from robotq_gripper import RobotiqGripper
import math
import time
import threading
import os

left_control = rtde_control.RTDEControlInterface("192.168.57.101")
left_receive = rtde_receive.RTDEReceiveInterface("192.168.57.101")
left_control.teachMode()

right_control = rtde_control.RTDEControlInterface("192.168.56.101")
right_receive = rtde_receive.RTDEReceiveInterface("192.168.56.101")
right_control.teachMode()

right_gripper = RobotiqGripper()
right_gripper.connect("192.168.56.101", 63352)
right_gripper.activate()   

left_arm_pos_list = []
right_arm_pos_list = []

experiment_name = input("Experiment Name: ")
step = 0
while (inp := input("Move Description: ")) != "exit":
    step += 1
    left_q  = left_receive.getActualTCPPose()    # [6 joint angles, rad]
    right_q = right_receive.getActualTCPPose()
    label = f"Step {step}, desc: {inp}"
    print(label)
    left_arm_pos_list.append((label, left_q))
    print(left_q)
    right_arm_pos_list.append((label, right_q))
    print(right_q)

print(f"Done. Captured {len(left_arm_pos_list)} samples.")

experiment_content = str(left_arm_pos_list) + "\n" + str(right_arm_pos_list)

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
experiment_dir = os.path.join(parent_dir, "experiments")
path = os.path.join(experiment_dir, experiment_name)
with open(path, "w") as f:
    f.write(experiment_content)

right_gripper.disconnect()
