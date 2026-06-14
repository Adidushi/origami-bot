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


# current_position = right_receive.getActualTCPPose()
# print(current_position)

# cartesian coordinates for left arm
left_bl = [-0.3671114816601733, 0.2297810553041695, -0.156323520188935840072]#[-0.8721046247426338,0.5354361477067348,-0.10357438069952617]
left_tl = [-0.3671114816601733-0.265, 0.2297810553041695, -0.156323520188935840072]#[-1.1437630313007467,0.5324130580232466,-0.10355972975433139]
left_tr = [-0.3671114816601733-0.265, 0.2297810553041695+0.365, -0.156323520188935840072]#[-1.1437630313007467,0.9042242687670197,-0.10356224029837532]
left_br = [-0.3671114816601733, 0.2297810553041695+0.365, -0.156323520188935840072]#[-0.8701698237244103,0.9042242687670197,-0.10356224029837532]
bottom_end_effector_pose = [0, 3.14, 0]

mvmt_data = [0.5, 0.5, 0]

poses = [
    left_bl + bottom_end_effector_pose,
    left_tr + bottom_end_effector_pose,
    left_tl + bottom_end_effector_pose,
    left_br + bottom_end_effector_pose
]

# for pose in poses:
    # left_control.moveL(pose)
    # input("enter for next")


def rotate_wrist(arm_control, pos, amt):
    joints = arm_control.getInverseKinematics(pos)
    joints[-1] += amt
    return arm_control.getForwardKinematics(joints)




# right_arm_folding_pos = [0.14009739625789402, -0.37063495427298887, 0.09706246724302361, 0.03254764983663611, -1.7035613922875321, -0.0487767912745752]
# rtde_c.moveJ_IK(pose = right_arm_folding_pos, speed=0.5, acceleration=0.5)

left_arm = {
    'bottom_right': [-0.3811126487291087, 0.5916647198316248, -0.25001351102176717, -0.0017339176957592841, 3.139994045314482, 0.00013657689149166203],
    'top_right': [-0.6356999775610952, 0.5832977377925311, -0.25001351102176717, -0.053227757033574456, 3.0608363549416917, 0.046481588744146055],
    'top_left': [-0.6356387966936057, 0.22944310560649264, -0.25001351102176717, -0.053413480764674906, 3.060169822572126, 0.045263435720190294],
    'bottom_left': [-0.369466455186429, 0.22944420875840602, -0.25001351102176717, -0.05344966348740266, 3.0601583065307283, 0.045224651625265795]
}

right_arm = {
    'top_right': [-0.1320102015841494, -0.37458099868781475, 0.016177005139537792, 7.130459517859422e-06, 3.14001135938399, 2.4104940535007586e-06],
    'top_left': [-0.1320042761452743, -0.7272697364921029, 0.01618955490848681, -2.1510189625228577e-05, 3.14001739575538, -3.328512474240181e-05],
    'bottom_left': [0.12452098632867296, -0.7213056515565214, 0.016175737793551137, 6.894154764248805e-06, 3.139980309941688, -4.381002936334777e-05],
    'bottom_right': [0.12452041218730804, -0.3696886048388927, 0.016193874742788494, -3.390196218503921e-05, 3.1399894497564644, -4.516041120261115e-05]
}

# lpose = left_arm['top_left']
# lpose[2] += 8/100
# left_control.moveL(lpose)

# input('yeet')

# rpose = right_arm['bottom_right']
# rpose[2] += 8/100
# right_control.moveL(rpose)

# for pose in right_arm.values():
    # right_control.moveL(pose, speed=0.3)

right_gripper.open()

joints = right_receive.getActualQ()
joints[-1] = 0
right_control.moveJ(joints)

pos = right_receive.getActualTCPPose()
pos[0] += 15/100
right_control.moveL(pos)

pos = right_arm['bottom_right']
#pos[2] += 10/100 # temp line for debug
pos[2] += 7.65/100
right_control.moveL(pos)

left_pos = left_arm['top_left']
left_pos[2] += 7.65/100
left_control.moveL(left_pos)

right_gripper.close()
input('advance?')

# input('1')
# attempt to rotate lol
# print(f'before move: {pos}')
# joints = right_receive.getActualQ()
# joints[-1] += math.pi/2
# input(f'1.5: {joints}')
# right_control.moveJ(joints)
# pos[3] += math.pi/2
# pos = right_receive.getActualTCPPose()
# print(f'before move: {pos}')

# pos = rotate_wrist(right_control, math.pi/2)
# right_control.moveL(pos)
# print(f'after move: {pos}')
# input('2')
# move +x 4cm
pos[0] += 4/100
right_control.moveL(pos)
# move around ry pi/2
pos[4] += math.pi/2
right_control.moveL(pos)
# open gripper
right_gripper.open()
# move -x 2cm
pos[0] -= 2/100
right_control.moveL(pos)
# close gripper
right_gripper.close()

# fold
radius = 9.50/100
angle = -1 * math.pi/4


# poses = [
#     (-0.5 * radius, 0.5 * radius),
#     (-0.5 * radius, 0.5 * radius),
#     (-0.5 * radius, -0.5 * radius),
#     (-0.5 * radius, -0.5 * radius)
# ]

# for pose in poses:
#     joints = right_receive.getActualQ()
#     joints[-1] += angle
#     right_control.moveJ(joints)

#     pos = right_receive.getActualTCPPose()
#     pos[1] += pose[0]
#     pos[2] += pose[1]
#     right_control.moveL(pos)
#     print(pos)
#     # input('advance?')


poses = [
    [0.14455670902501033, -0.4171414963796778, 0.14367779896784005, 0.6144824954075364, -1.4835426886965852, -0.6135300246668695],
    [0.14456742466626873, -0.46458551220396765, 0.19113158370842376, 1.2101184686104587, -1.2103547846453826, -1.20845800645584],
    [0.1446362691868944, -0.512027513533371, 0.1436820066897817, 1.7613427711876397, -0.7299543455686673, -1.7584837794824175],
    [0.14468408650575754, -0.5594727288560295, 0.09619033930373834, 2.2229241311048273, -0.0005453450726498538, -2.2194132031899336]
]

# for pose in poses:
#     right_control.moveL(pose)
#     pos = pose
poses_with_vars = [pose+[0.5,0.5,0] for pose in poses]

right_control.moveL(poses_with_vars)





# move with radius 8.25 in circular motion
#radius = 8.25/100
# move -16.5 in y and flip 180 deg in rx
#pos[1] -= 16.5/100
# pos[3] += math.pi
# move slightly up so it prioritizes upper circle
#pos[2] += 1/100
#right_control.servoC(pos, blend=radius)
# move back down
#pos[2] -= 1/100
#right_control.moveL(pos)






right_gripper.disconnect()