"""Calibration between the board workspace and a robot base frame.

A `BoardCalibration` captures the rigid transform from *board* coordinates
to a particular arm's *base* frame, fitted from a handful of taught
correspondences.  With it you can turn any board target -- an ``(x, y)`` location
together with a *height above the board* and a tool rotation -- into a ready-to-
send Universal-Robots TCP pose, with the gripper oriented to approach straight
down onto the board.

Crucially, the *height above board* is an explicit input everywhere: targets are
**not** assumed to lie on the board surface, so the same calibration serves both
in-contact moves (height 0) and elevated moves such as picking up a magnet by a
raised handle.
"""
from __future__ import annotations

import numpy as np
from spatialmath import SE3, SO3

from . import geometry as geo


class BoardCalibration:
    """Rigid ``board -> base`` transform plus a downward-tool convention.

    Parameters
    ----------
    board_to_base : spatialmath.SE3
        Transform taking a board-coordinate point (metres; ``z`` = height above
        the board surface) to the robot base frame.

    Attributes
    ----------
    board_to_base : spatialmath.SE3
        The stored transform (board frame expressed in the base frame).

    Notes
    -----
    The board frame is right-handed with ``z`` pointing **up** out of the
    surface.  The columns of ``board_to_base.R`` are therefore the board ``x``,
    ``y`` and (upward) ``z`` axes written in base coordinates.  The gripper is
    assumed to work pointing down, i.e. its tool ``+z`` axis runs along the board
    ``-z`` direction.
    """

    #: Canonical board coordinates of the four corners, used by
    #: `from_taught_corners()`.  Filled in that method from the board size.
    _CORNER_ORDER = ("bottom_left", "bottom_right", "top_left", "top_right")

    def __init__(self, board_to_base: SE3) -> None:
        self.board_to_base = board_to_base

    # ------------------------------------------------------------------ #
    # Construction
    # ------------------------------------------------------------------ #
    @classmethod
    def from_point_correspondences(cls, board_points, base_points) -> "BoardCalibration":
        """Fit a calibration from matched board / base points.

        Parameters
        ----------
        board_points : array_like, shape (N, 2) or (N, 3)
            Points in board coordinates.  Two-column input is treated as lying on
            the board surface (height 0).
        base_points : array_like, shape (N, 3)
            The corresponding points in the robot base frame, typically the
            translation part of ``getActualTCPPose()`` recorded while touching
            each board point.

        Returns
        -------
        BoardCalibration

        Raises
        ------
        ValueError
            If fewer than three correspondences are supplied.

        See Also
        --------
        origami.geometry.fit_rigid_transform : The underlying Procrustes fit.
        """
        board = np.atleast_2d(np.asarray(board_points, dtype=float))
        if board.shape[1] == 2:
            board = np.hstack([board, np.zeros((board.shape[0], 1))])
        if board.shape[0] < 3:
            raise ValueError("need at least 3 correspondences to fit a calibration")
        return cls(geo.fit_rigid_transform(board, np.asarray(base_points, dtype=float)))

    @classmethod
    def from_taught_corners(cls, corner_poses: dict[str, list[float]],
                            board_width: float, board_height: float) -> "BoardCalibration":
        """Fit a calibration from taught corner *poses*.

        Parameters
        ----------
        corner_poses : dict
            Maps any of ``'bottom_left'``, ``'bottom_right'``, ``'top_left'``,
            ``'top_right'`` to a recorded UR pose at that board corner.  At least
            three corners must be present.
        board_width : float
            Board extent along ``+x`` (metres); the right-hand corners sit here.
        board_height : float
            Board extent along ``+y`` (metres); the top corners sit here.

        Returns
        -------
        BoardCalibration

        Notes
        -----
        Corners are assigned board coordinates
        ``bottom_left=(0, 0)``, ``bottom_right=(width, 0)``,
        ``top_left=(0, height)``, ``top_right=(width, height)`` -- all on the
        board surface.
        """
        corner_xy = {
            "bottom_left": (0.0, 0.0),
            "bottom_right": (board_width, 0.0),
            "top_left": (0.0, board_height),
            "top_right": (board_width, board_height),
        }
        board_points, base_points = [], []
        for name in cls._CORNER_ORDER:
            if name in corner_poses:
                board_points.append(corner_xy[name])
                base_points.append(np.asarray(corner_poses[name], dtype=float)[:3])
        return cls.from_point_correspondences(board_points, base_points)

    # ------------------------------------------------------------------ #
    # Point mapping
    # ------------------------------------------------------------------ #
    def board_point_to_base(self, x: float, y: float, height_above_board: float = 0.0) -> np.ndarray:
        """Map a board point to a base-frame position.

        Parameters
        ----------
        x, y : float
            Board-surface coordinates (metres).
        height_above_board : float, optional
            Height of the point above the board surface (metres).  Default ``0``
            (on the surface).

        Returns
        -------
        numpy.ndarray, shape (3,)
            The point in the robot base frame.
        """
        return np.asarray(self.board_to_base * np.array([x, y, height_above_board]),
                          dtype=float).reshape(3)

    def base_point_to_board(self, base_xyz) -> np.ndarray:
        """Map a base-frame position back to board coordinates.

        Parameters
        ----------
        base_xyz : array_like, shape (3,)
            A point in the robot base frame.

        Returns
        -------
        numpy.ndarray, shape (3,)
            Board coordinates ``(x, y, height_above_board)``.
        """
        return np.asarray(self.board_to_base.inv() * np.asarray(base_xyz, dtype=float),
                          dtype=float).reshape(3)

    # ------------------------------------------------------------------ #
    # Orientation and full poses
    # ------------------------------------------------------------------ #
    def gripper_orientation(self, tool_rotation: float = 0.0) -> SO3:
        """Base-frame orientation for a downward-pointing gripper.

        Parameters
        ----------
        tool_rotation : float, optional
            Rotation of the gripper about the board normal (radians).  Use it to
            line the fingers up with a fold edge.  Default ``0``.

        Returns
        -------
        spatialmath.SO3
            Orientation whose tool ``+z`` points into the board and whose tool
            ``+x`` lies in the board plane, rotated by ``tool_rotation``.
        """
        R = np.asarray(self.board_to_base.R, dtype=float)
        board_x, board_y, board_z = R[:, 0], R[:, 1], R[:, 2]
        tool_x = np.cos(tool_rotation) * board_x + np.sin(tool_rotation) * board_y
        tool_z = -board_z
        tool_y = np.cross(tool_z, tool_x)
        return SO3(np.column_stack([tool_x, tool_y, tool_z]), check=False)

    def gripper_transform(self, x: float, y: float, height_above_board: float,
                          tool_rotation: float = 0.0) -> SE3:
        """Full base-frame pose (as a `spatialmath.SE3`) of the gripper.

        Parameters
        ----------
        x, y : float
            Board-surface target (metres).
        height_above_board : float
            Height of the tool above the board surface (metres).
        tool_rotation : float, optional
            Gripper rotation about the board normal (radians).  Default ``0``.

        Returns
        -------
        spatialmath.SE3
        """
        position = self.board_point_to_base(x, y, height_above_board)
        return SE3.Rt(self.gripper_orientation(tool_rotation), position)

    def tcp_pose_at(self, x: float, y: float, height_above_board: float,
                    tool_rotation: float = 0.0) -> list[float]:
        """UR TCP pose for a board target.

        Parameters
        ----------
        x, y : float
            Board-surface target (metres).
        height_above_board : float
            Height of the tool above the board surface (metres).  Pass ``0`` for
            surface contact; pass a positive value for elevated targets.
        tool_rotation : float, optional
            Gripper rotation about the board normal (radians).  Default ``0``.

        Returns
        -------
        list of float
            UR pose ``[x, y, z, rx, ry, rz]`` ready for ``moveL`` / ``moveJ_IK``.
        """
        return geo.se3_to_pose(self.gripper_transform(x, y, height_above_board, tool_rotation))

    # ------------------------------------------------------------------ #
    # Diagnostics
    # ------------------------------------------------------------------ #
    def fit_residuals(self, board_points, base_points) -> np.ndarray:
        """Per-correspondence fit error, in metres.

        Parameters
        ----------
        board_points : array_like, shape (N, 2) or (N, 3)
            Board points (two columns are treated as surface points).
        base_points : array_like, shape (N, 3)
            Corresponding base-frame points.

        Returns
        -------
        numpy.ndarray, shape (N,)
            Euclidean distance between each predicted and measured base point.
            Useful for spotting a mis-taught corner.
        """
        board = np.atleast_2d(np.asarray(board_points, dtype=float))
        if board.shape[1] == 2:
            board = np.hstack([board, np.zeros((board.shape[0], 1))])
        predicted = np.asarray(self.board_to_base * board.T, dtype=float).T
        return np.linalg.norm(predicted - np.asarray(base_points, dtype=float), axis=1)
