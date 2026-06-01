from mvmt.robotq_gripper import RobotiqGripper

g = RobotiqGripper()
g.connect("192.168.56.101", 63352)
g.activate()                  # idempotent — fine to call every run
g.move(255)            # open
# g.move(255,80,80)          # close
# print("position:", g.position())
g.disconnect()