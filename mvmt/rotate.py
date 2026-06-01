import rtde_control
import rtde_receive
from robotq_gripper import RobotiqGripper
import math

left_control = rtde_control.RTDEControlInterface("192.168.57.101")
left_receive = rtde_receive.RTDEReceiveInterface("192.168.57.101")

right_control = rtde_control.RTDEControlInterface("192.168.56.101")
right_receive = rtde_receive.RTDEReceiveInterface("192.168.56.101")


right_gripper = RobotiqGripper()
right_gripper.connect("192.168.56.101", 63352)
right_gripper.activate()  

def rotate_wrist(arm_control, pos, amt):
    print(f'before: {right_receive.getActualTCPPose()}')
    #current = right_receive.getActualQ()
    joints = right_receive.getActualQ()
    #joints = arm_control.getInverseKinematics(pos, current)
    joints[-1] += amt
    print(f'after: {arm_control.getForwardKinematics(joints)}')
    return arm_control.getForwardKinematics(joints)



joints = right_receive.getActualQ()
joints[-1] += math.pi/2
right_control.moveJ(joints, asynchronous=False)


pos = right_receive.getActualTCPPose()
pos[2] += 8.25/100
pos[1] -= 8.25/100
right_control.moveL(pos)