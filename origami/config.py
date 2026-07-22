"""Project configuration: robot endpoints, board geometry and taught calibrations.

`LEFT_ARM_CORNERS`, `RIGHT_ARM_CORNERS`, and `MAGNET_PLATFORM_POSITIONS` are
loaded straight from ``calibration.json`` (next to this file), which is
tracked as the current source of truth for the calibration and tha platform positions. Whenever
the arms, board, or table drift, run ``python -m origami.calibrate`` to
re-calibrate everything and and writes the updated positions back to calibration.json.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

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
# Taught calibration (UR TCP poses / world positions), loaded from
# calibration.json -- see `origami.calibrate`.
# --------------------------------------------------------------------------- #
#: Path to the calibration data written by `origami.calibrate`.
CALIBRATION_FILE = Path(__file__).with_name("calibration.json")

_calibration = json.loads(CALIBRATION_FILE.read_text())
#: Left-arm TCP poses at the four board corners.
LEFT_ARM_CORNERS = _calibration["LEFT_ARM_CORNERS"]
#: Right-arm TCP poses at the four board corners.
RIGHT_ARM_CORNERS = _calibration["RIGHT_ARM_CORNERS"]
#: Magnet platform corner positions in world space (left-arm frame).
MAGNET_PLATFORM_POSITIONS = _calibration["MAGNET_PLATFORM_POSITIONS"]

CREASER_POS = [MAGNET_PLATFORM_POSITIONS["bottom_right"][0]-16.5/100, MAGNET_PLATFORM_POSITIONS["bottom_right"][1]+16/100, MAGNET_PLATFORM_POSITIONS["bottom_right"][2]+4.7/100]
CREASER_GRIP_OPEN_POS = 0.65

FLIP_PAPER_CLEARANCE = 0.35
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
