"""Origami folding framework for UR arms + Robotiq gripper on a magnetic board.

Work in **board coordinates** (a 2D ``(x, y)`` frame on the magnetic board, plus
a *height above the board*); the framework converts to robot poses, drives the
arms, and keeps an analytic model of the paper and magnets in sync after every
operation.

The transform and pose algebra is built on `spatialmath` (with
`scipy` for calibration fitting), not hand-rolled.

Quick start (simulation, no hardware)
-------------------------------------
>>> from origami import Workspace, demos
>>> ws = Workspace.simulated()
>>> _ = demos.fold_dart(ws, verbose=False)
>>> ws.paper                                  # doctest: +ELLIPSIS
Paper('paper', ...)

Top-level exports cover the common pieces; submodules hold the details.
"""
from __future__ import annotations

from . import actions, config, demos, geometry
from .arm import Arm, ArmConfig
from .backends import (
    RobotiqGripperBackend,
    RTDEArmBackend,
    SimulatedArmBackend,
    SimulatedGripperBackend,
)
from .calibration import BoardCalibration
from .geometry import FoldLine
from .magnets import BlockMagnet, LBracketMagnet, Magnet, MagnetRegistry
from .paper import Fold, Paper
from .workspace import Workspace

__all__ = [
    # submodules
    "actions",
    "config",
    "demos",
    "geometry",
    # geometry
    "FoldLine",
    # calibration
    "BoardCalibration",
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
