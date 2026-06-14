"""Origami folding framework for UR arms + Robotiq gripper on a magnetic board.

Work in **world coordinates** -- a 3D frame centred on the board where x and y
span the board surface and z is height above it (z=0 = board surface).  The
framework converts world targets to robot TCP poses, drives the arms, and keeps
an analytic model of the paper and magnets in sync after every operation.

Quick start (simulation, no hardware)
--------------------------------------
>>> from origami import Workspace
>>> ws = Workspace.simulated()
"""
from __future__ import annotations

from . import actions, config, geometry
from .arm import Arm, ArmConfig
from .backends import (
    RobotiqGripperBackend,
    RTDEArmBackend,
    SimulatedArmBackend,
    SimulatedGripperBackend,
)
from .coords import ArmCalibration
from .geometry import FoldLine
from .magnets import BlockMagnet, LBracketMagnet, Magnet, MagnetRegistry
from .paper import Fold, Paper
from .workspace import Workspace

__all__ = [
    # submodules
    "actions",
    "config",
    "geometry",
    # coordinates
    "ArmCalibration",
    # geometry
    "FoldLine",
    # paper
    "Paper",
    "Fold",
    # magnets
    "Magnet",
    "BlockMagnet",
    "LBracketMagnet",
    "MagnetRegistry",
    # arms / backends
    "Arm",
    "ArmConfig",
    "RTDEArmBackend",
    "RobotiqGripperBackend",
    "SimulatedArmBackend",
    "SimulatedGripperBackend",
    # workspace
    "Workspace",
]
