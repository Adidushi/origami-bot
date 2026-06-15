"""World coordinate system and per-arm calibration.

World space
-----------
All high-level code works in a single right-handed *world frame* centred on
the board:

    x  -- horizontal across the board surface (metres, origin at bottom-left)
    y  -- vertical along the board surface (metres, origin at bottom-left)
    z  -- height above the board surface (metres, z=0 is the board, positive is up)

This frame extends freely beyond the physical board footprint: magnet trays,
paper stacks, and any other fixtures can have world coordinates with x or y
outside [0, board_width] × [0, board_height] and a z that reflects their
actual height off the table.

Arm calibration
---------------
Each robot arm sits at a fixed pose relative to the board and has its own
*base frame*.  An `ArmCalibration` stores the rigid transform from world space
into that arm's base frame.  You fit one per arm by jogging the arm to known
board positions and recording the TCP pose at each.

Once fitted, `ArmCalibration` converts any world (x, y, z) into the
base-frame XYZ needed for motion planning, and constructs full 6-DOF TCP poses
with the gripper always pointing straight down onto the board.
"""
from __future__ import annotations

import numpy as np
from spatialmath import SE3, SO3

from . import geometry as geo


class ArmCalibration:
    """Rigid transform from world space into one arm's base frame.

    Parameters
    ----------
    world_to_arm : spatialmath.SE3
        Transform that maps a world-space point ``(x, y, z)`` into the
        robot arm's base frame.

    Notes
    -----
    The columns of ``world_to_arm.R`` are the world ``+x``, ``+y`` and ``+z``
    axes expressed in the arm's base frame.  The gripper approaches the board
    pointing straight down (tool ``+z`` aligned with world ``-z``).
    """

    _CORNER_ORDER = ("bottom_left", "bottom_right", "top_left", "top_right")

    def __init__(self, world_to_arm: SE3) -> None:
        self.world_to_arm = world_to_arm

    # ------------------------------------------------------------------ #
    # Construction
    # ------------------------------------------------------------------ #
    @classmethod
    def from_point_correspondences(cls, world_points, arm_points) -> "ArmCalibration":
        """Fit the calibration from matched world / arm-base point pairs.

        Parameters
        ----------
        world_points : array_like, shape (N, 2) or (N, 3)
            Known positions in world coordinates.  Two-column input is treated
            as lying on the board surface (z = 0).
        arm_points : array_like, shape (N, 3)
            Corresponding positions in the arm's base frame -- typically the
            XYZ translation of ``getActualTCPPose()`` recorded while touching
            each world point.

        Returns
        -------
        ArmCalibration

        Raises
        ------
        ValueError
            If fewer than three correspondences are supplied.
        """
        world = np.atleast_2d(np.asarray(world_points, dtype=float))
        if world.shape[1] == 2:
            world = np.hstack([world, np.zeros((world.shape[0], 1))])
        if world.shape[0] < 3:
            raise ValueError("need at least 3 correspondences to fit an arm calibration")
        return cls(geo.fit_rigid_transform(world, np.asarray(arm_points, dtype=float)))

    @classmethod
    def from_taught_corners(cls, corner_poses: dict[str, list[float]],
                            board_width: float, board_height: float) -> "ArmCalibration":
        """Fit the calibration from TCP poses recorded at the board corners.

        Jog the arm to each corner and record its TCP pose (UR format), then
        call this.  At least three corners are required; all four improve the fit.

        Parameters
        ----------
        corner_poses : dict
            Maps corner names (``'bottom_left'``, ``'bottom_right'``,
            ``'top_left'``, ``'top_right'``) to recorded UR TCP poses.
        board_width : float
            Board extent along world ``+x`` (metres).
        board_height : float
            Board extent along world ``+y`` (metres).

        Returns
        -------
        ArmCalibration

        Notes
        -----
        Corner world coordinates: ``bottom_left=(0,0,0)``,
        ``bottom_right=(width,0,0)``, ``top_left=(0,height,0)``,
        ``top_right=(width,height,0)`` -- all on the board surface.

        Z-axis handling
        ~~~~~~~~~~~~~~~
        The z coordinate of each taught corner pose is the exact board surface
        height in that arm's base frame (the corners are taught by touching the
        board).  Because the board is flat, all corners share the same arm-frame
        z, and the arms are mounted with their z-axis perpendicular to the board,
        so world +z and arm +z are the same direction.

        This method uses those facts directly instead of inferring the z-axis
        from the Procrustes fit (which is unconstrained in z because all world
        points are at z = 0):

        1. The board surface height ``board_z_arm`` is taken as the mean of the
           corner z coordinates — the ground-truth offset for world z = 0.
        2. World +z is forced to equal arm +z (``R[:, 2] = [0, 0, 1]``).
        3. The x-axis from the Procrustes fit is projected onto the board plane
           and y is recomputed as ``z × x`` to keep the frame right-handed.
        4. The translation z-component is pinned to ``board_z_arm`` exactly.
        """
        corner_xy = {
            "bottom_left": (0.0, 0.0),
            "bottom_right": (board_width, 0.0),
            "top_left": (0.0, board_height),
            "top_right": (board_width, board_height),
        }
        world_points, arm_points = [], []
        for name in cls._CORNER_ORDER:
            if name in corner_poses:
                world_points.append(corner_xy[name])
                arm_points.append(np.asarray(corner_poses[name], dtype=float)[:3])

        # Board surface z in arm frame — read directly from the corner poses.
        board_z_arm = float(np.mean([
            np.asarray(corner_poses[n], dtype=float)[2]
            for n in cls._CORNER_ORDER if n in corner_poses
        ]))

        # Procrustes fit gives us the x/y rotation from corner positions.
        base = cls.from_point_correspondences(world_points, arm_points)
        R = np.asarray(base.world_to_arm.R, dtype=float)

        # World +z = arm +z: project the fitted x onto the board plane,
        # recompute y = z × x, keep z = [0, 0, 1].
        x = np.array([R[0, 0], R[1, 0], 0.0])
        x /= np.linalg.norm(x)
        z = np.array([0.0, 0.0, 1.0])
        y = np.cross(z, x)
        R_new = np.column_stack([x, y, z])

        # Least-squares translation from the centroid equation, then pin z.
        src = np.atleast_2d(np.asarray(world_points, dtype=float))
        src = np.hstack([src, np.zeros((src.shape[0], 1))])
        dst = np.asarray(arm_points, dtype=float)
        t = dst.mean(axis=0) - R_new @ src.mean(axis=0)
        t[2] = board_z_arm

        return cls(SE3.Rt(SO3(R_new, check=False), t))

    # ------------------------------------------------------------------ #
    # Coordinate conversion (internal — Arm is the public interface)
    # ------------------------------------------------------------------ #
    def _world_to_arm_xyz(self, x: float, y: float, z: float = 0.0) -> np.ndarray:
        """World (x,y,z) → XYZ in the arm's base frame."""
        return np.asarray(self.world_to_arm * np.array([x, y, z]),
                          dtype=float).reshape(3)

    def _arm_to_world_xyz(self, arm_xyz) -> np.ndarray:
        """XYZ in the arm's base frame → world (x,y,z)."""
        return np.asarray(self.world_to_arm.inv() * np.asarray(arm_xyz, dtype=float),
                          dtype=float).reshape(3)

    # ------------------------------------------------------------------ #
    # Gripper orientation and full TCP pose
    # ------------------------------------------------------------------ #
    def gripper_orientation(self, tool_rotation: float = 0.0) -> SO3:
        """Base-frame orientation for a downward-pointing gripper.

        The gripper's tool ``+z`` points into the board; its tool ``+x`` lies
        in the board plane and can be rotated about the board normal to align
        the fingers with a fold edge or magnet handle.

        Parameters
        ----------
        tool_rotation : float, optional
            Spin about the board normal (radians).  Default ``0``.

        Returns
        -------
        spatialmath.SO3
        """
        R = np.asarray(self.world_to_arm.R, dtype=float)
        world_x, world_y, world_z = R[:, 0], R[:, 1], R[:, 2]
        tool_x = np.cos(tool_rotation) * world_x + np.sin(tool_rotation) * world_y
        tool_z = -world_z
        tool_y = np.cross(tool_z, tool_x)
        return SO3(np.column_stack([tool_x, tool_y, tool_z]), check=False)

    def tcp_pose(self, x: float, y: float, z: float,
                 tool_rotation: float = 0.0) -> list[float]:
        """UR TCP pose ``[x, y, z, rx, ry, rz]`` for a world target.

        Parameters
        ----------
        x, y, z : float
            Target in world coordinates (metres).
        tool_rotation : float, optional
            Spin about the board normal (radians).  Default ``0``.

        Returns
        -------
        list of float
            Ready to pass to ``moveL``.
        """
        position = self._world_to_arm_xyz(x, y, z)
        transform = SE3.Rt(self.gripper_orientation(tool_rotation), position)
        return geo.se3_to_pose(transform)

    # ------------------------------------------------------------------ #
    # Diagnostics
    # ------------------------------------------------------------------ #
    def fit_residuals(self, world_points, arm_points) -> np.ndarray:
        """Per-point fit error (metres) between predicted and measured arm positions.

        Useful for spotting a mis-taught corner after fitting.

        Parameters
        ----------
        world_points : array_like, shape (N, 2) or (N, 3)
            World-space points (two-column input treated as z = 0).
        arm_points : array_like, shape (N, 3)
            Corresponding measured arm base-frame positions.

        Returns
        -------
        numpy.ndarray, shape (N,)
        """
        world = np.atleast_2d(np.asarray(world_points, dtype=float))
        if world.shape[1] == 2:
            world = np.hstack([world, np.zeros((world.shape[0], 1))])
        predicted = np.asarray(self.world_to_arm * world.T, dtype=float).T  # batch transform
        return np.linalg.norm(predicted - np.asarray(arm_points, dtype=float), axis=1)
