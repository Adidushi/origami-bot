"""Project configuration: robot endpoints, board geometry and taught calibrations.

The corner poses below were recorded from the arms (see ``mvmt/get.py`` /
``mvmt/record.py``).  They are fed to `ArmCalibration.from_taught_corners` to
fit each arm's world-to-base transform.  Re-teach them whenever the board or
tool changes, then update these dictionaries.
"""
from __future__ import annotations

import math

from .coords import ArmCalibration

# --------------------------------------------------------------------------- #
# Network
# --------------------------------------------------------------------------- #
#: IP address of the left arm's controller.
LEFT_ARM_IP = "192.168.57.101"
#: IP address of the right arm's controller.
RIGHT_ARM_IP = "192.168.56.101"
#: TCP port of the Robotiq gripper's socket server.
GRIPPER_PORT = 63352

# --------------------------------------------------------------------------- #
# Board / world geometry (metres)
# --------------------------------------------------------------------------- #
#: Board extent along world +x (metres).
BOARD_WIDTH = 0.37
#: Board extent along world +y (metres).
BOARD_HEIGHT = 0.27
#: Default paper width (metres).
PAPER_WIDTH = 0.21
#: Default paper height (metres).
PAPER_HEIGHT = 0.297

# --------------------------------------------------------------------------- #
# Taught corner poses (UR TCP poses) recorded from the arms via mvmt/get.py.
# --------------------------------------------------------------------------- #
#: Left-arm TCP poses at the four board corners.
LEFT_ARM_CORNERS = {
    "bottom_right": [-0.3811126487291087, 0.5916647198316248, -0.17351351102176717,
                     -0.0017339176957592841, 3.139994045314482, 0.00013657689149166203],
    "top_right": [-0.6356999775610952, 0.5832977377925311, -0.17351351102176717,
                  -0.053227757033574456, 3.0608363549416917, 0.046481588744146055],
    "top_left": [-0.6356387966936057, 0.22944310560649264, -0.17351351102176717,
                 -0.053413480764674906, 3.060169822572126, 0.045263435720190294],
    "bottom_left": [-0.369466455186429, 0.22944420875840602, -0.17351351102176717,
                    -0.05344966348740266, 3.0601583065307283, 0.045224651625265795],
}

#: Right-arm TCP poses at the four board corners.
RIGHT_ARM_CORNERS = {
    "top_right": [-0.1320102015841494, -0.37458099868781475, 0.092677005139537792,
                  7.130459517859422e-06, 3.14001135938399, 2.4104940535007586e-06],
    "top_left": [-0.1320042761452743, -0.7272697364921029, 0.09268955490848681,
                 -2.1510189625228577e-05, 3.14001739575538, -3.328512474240181e-05],
    "bottom_left": [0.12452098632867296, -0.7213056515565214, 0.092675737793551137,
                    6.894154764248805e-06, 3.139980309941688, -4.381002936334777e-05],
    "bottom_right": [0.12452041218730804, -0.3696886048388927, 0.092693874742788494,
                     -3.390196218503921e-05, 3.1399894497564644, -4.516041120261115e-05],
}

LEFT_ARM_MAGNET_PLATFORM_BOTTOM_RIGHT_CORNER = {
    "bottom_right": [-0.10, 0.05, -0.038],
    "bottom_left": [-0.234, 0.05, -0.038]
}

LEFT_ARM_MAGNET_PLATFORM_BOTTOM_RIGHT_CORNER = [-0.09, 0.05, -0.038]
CREASER_POS = [LEFT_ARM_MAGNET_PLATFORM_BOTTOM_RIGHT_CORNER[0]-16.5/100, LEFT_ARM_MAGNET_PLATFORM_BOTTOM_RIGHT_CORNER[1]+16/100, LEFT_ARM_MAGNET_PLATFORM_BOTTOM_RIGHT_CORNER[2]+4.8/100]
CREASER_GRIP_OPEN_POS = 0.65

FLIP_PAPER_CLEARANCE = 0.3
FLIP_PAPER_OVERROTATION = 0.5


# --------------------------------------------------------------------------- #
# Start joint positions (radians), applied at the beginning of every program run.
# --------------------------------------------------------------------------- #
#: Right arm start position: [0°, -90°, -90°, -90°, 90°, 0°]
RIGHT_ARM_START_JOINTS: list[float] = [math.radians(a) for a in [0, -90, -90, -90, 90, 0]]
#: Left arm start position:
#LEFT_ARM_START_JOINTS: list[float] | None = None  # e.g. [math.radians(a) for a in [0, -90, -90, -90, 90, 0]]
LEFT_ARM_START_JOINTS: list[float] = [math.radians(a) for a in [0, -90, 90, -90, -90, 0]]



def left_calibration() -> ArmCalibration:
    """Fit the left arm's world-to-base calibration from its taught corners."""
    return ArmCalibration.from_taught_corners(LEFT_ARM_CORNERS, BOARD_WIDTH, BOARD_HEIGHT)


def right_calibration() -> ArmCalibration:
    """Fit the right arm's world-to-base calibration from its taught corners."""
    return ArmCalibration.from_taught_corners(RIGHT_ARM_CORNERS, BOARD_WIDTH, BOARD_HEIGHT)
