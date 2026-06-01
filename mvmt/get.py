import rtde_control
import rtde_receive

left_control = rtde_control.RTDEControlInterface("192.168.57.101")
left_receive = rtde_receive.RTDEReceiveInterface("192.168.57.101")


current_position = left_receive.getActualTCPPose()
print(current_position)

# cartesian coordinates for left arm
left_bl = [-0.3671114816601733, 0.2297810553041695, -0.156323520188935840072]#[-0.8721046247426338,0.5354361477067348,-0.10357438069952617]
left_tl = [-0.3671114816601733-0.265, 0.2297810553041695, -0.156323520188935840072]#[-1.1437630313007467,0.5324130580232466,-0.10355972975433139]
left_tr = [-0.3671114816601733-0.265, 0.2297810553041695+0.365, -0.156323520188935840072]#[-1.1437630313007467,0.9042242687670197,-0.10356224029837532]
left_br = [-0.3671114816601733, 0.2297810553041695+0.365, -0.156323520188935840072]#[-0.8701698237244103,0.9042242687670197,-0.10356224029837532]
bottom_end_effector_pose = [0, 3.14, 0]

mvmt_data = [0.5, 0.5]

poses = [
    left_bl + bottom_end_effector_pose + mvmt_data,
    left_tr + bottom_end_effector_pose + mvmt_data,
    left_tl + bottom_end_effector_pose + mvmt_data,
    left_br + bottom_end_effector_pose + mvmt_data
]

left_control.moveL(poses)







# right_arm_folding_pos = [0.14009739625789402, -0.37063495427298887, 0.09706246724302361, 0.03254764983663611, -1.7035613922875321, -0.0487767912745752]
# rtde_c.moveJ_IK(pose = right_arm_folding_pos, speed=0.5, acceleration=0.5)

