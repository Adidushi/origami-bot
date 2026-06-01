import rtde_control
rtde_c = rtde_control.RTDEControlInterface("192.168.56.101")

sidefold_pos = [-0.4627435843097132, -2.153088232079977, -1.930733323097229, -2.192392965356344, 1.0508142709732056, -1.5545461813556116]

rtde_c.moveL(sidefold_pos)
