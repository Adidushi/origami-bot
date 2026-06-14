"""High-level, board-aware control of a single robot arm.

`Arm` lets you command an arm in **board coordinates** -- "hover above
this point", "press down here", "grip" -- instead of hand-authoring 6-DOF UR
poses.  It combines a `BoardCalibration` (which knows
how board coordinates map to this arm's base frame) with an arm backend and an
optional gripper backend, real or simulated.

Heights are always explicit.  A target is an ``(x, y)`` board location plus a
*height above the board*; nothing is implicitly assumed to lie on the board
surface.  The only convenience that defaults to the surface is
`Arm.press_onto_board()`, whose whole purpose is to make board contact.
"""
from __future__ import annotations

from dataclasses import dataclass

from . import backends as backends_mod
from .backends import ArmBackend, GripperBackend
from .calibration import BoardCalibration


@dataclass
class ArmConfig:
    """Per-arm motion defaults, in SI units (metres, m/s, m/s^2, radians).

    Parameters
    ----------
    travel_height : float, optional
        Default height above the board used for collision-free transit moves.
        Default ``0.10``.
    contact_depth : float, optional
        How far *below* the board surface to drive when pressing, to guarantee
        firm contact (metres, positive number).  Default ``0.002``.
    linear_speed : float, optional
        Default tool speed for linear moves (m/s).  Default ``0.25``.
    linear_acceleration : float, optional
        Default tool acceleration for linear moves (m/s^2).  Default ``0.5``.
    """

    travel_height: float = 0.10
    contact_depth: float = 0.002
    linear_speed: float = 0.25
    linear_acceleration: float = 0.5


class Arm:
    """One robot arm, commanded in board coordinates.

    Parameters
    ----------
    name : str
        Label for the arm (e.g. ``'left'``).
    backend : origami.backends.ArmBackend
        Motion backend (real or simulated).
    calibration : origami.calibration.BoardCalibration
        Board-to-base calibration for this arm.
    gripper : origami.backends.GripperBackend or None, optional
        Gripper backend, if this arm has one.
    config : ArmConfig or None, optional
        Motion defaults; a default `ArmConfig` is used if omitted.

    See Also
    --------
    Arm.real, Arm.simulated : Convenience constructors.
    """

    def __init__(self, name: str, backend: ArmBackend, calibration: BoardCalibration,
                 gripper: GripperBackend | None = None, config: ArmConfig | None = None) -> None:
        self.name = name
        self.backend = backend
        self.calibration = calibration
        self.gripper = gripper
        self.config = config or ArmConfig()

    # ------------------------------------------------------------------ #
    # Constructors
    # ------------------------------------------------------------------ #
    @classmethod
    def real(cls, name: str, arm_ip: str, calibration: BoardCalibration,
             gripper_ip: str | None = None, gripper_port: int = 63352,
             config: ArmConfig | None = None) -> "Arm":
        """Build an arm driving real hardware over RTDE.

        Parameters
        ----------
        name : str
            Label for the arm.
        arm_ip : str
            IP address of the arm controller.
        calibration : origami.calibration.BoardCalibration
            Board-to-base calibration for this arm.
        gripper_ip : str or None, optional
            IP of the Robotiq gripper server; ``None`` for no gripper.
        gripper_port : int, optional
            Gripper server port.  Default ``63352``.
        config : ArmConfig or None, optional
            Motion defaults.

        Returns
        -------
        Arm
        """
        gripper = backends_mod.RobotiqGripperBackend(gripper_ip, gripper_port) if gripper_ip else None
        return cls(name, backends_mod.RTDEArmBackend(arm_ip), calibration, gripper, config)

    @classmethod
    def simulated(cls, name: str, calibration: BoardCalibration,
                  config: ArmConfig | None = None, start_pose=None) -> "Arm":
        """Build a fully simulated arm (no hardware required).

        Parameters
        ----------
        name : str
            Label for the arm.
        calibration : origami.calibration.BoardCalibration
            Board-to-base calibration for this arm.
        config : ArmConfig or None, optional
            Motion defaults.
        start_pose : sequence of float or None, optional
            Initial TCP pose for the simulated backend.

        Returns
        -------
        Arm
        """
        return cls(name, backends_mod.SimulatedArmBackend(start_pose, name=name), calibration,
                   backends_mod.SimulatedGripperBackend(name=name), config)

    # ------------------------------------------------------------------ #
    # Motion (board coordinates)
    # ------------------------------------------------------------------ #
    def move_to_board_point(self, x: float, y: float, height_above_board: float,
                            tool_rotation: float = 0.0,
                            speed: float | None = None, acceleration: float | None = None) -> bool:
        """Move the tool to a board target at an explicit height.

        This is the general motion primitive; ``height_above_board`` is required
        so the caller always states the working height.

        Parameters
        ----------
        x, y : float
            Board-surface coordinates of the target (metres).
        height_above_board : float
            Height of the tool above the board surface (metres).  ``0`` is on the
            surface; positive values are above it.
        tool_rotation : float, optional
            Gripper rotation about the board normal (radians).  Default ``0``.
        speed : float or None, optional
            Override for tool speed (m/s); uses the config default if ``None``.
        acceleration : float or None, optional
            Override for tool acceleration (m/s^2); config default if ``None``.

        Returns
        -------
        bool
            ``True`` on success.
        """
        pose = self.calibration.tcp_pose_at(x, y, height_above_board, tool_rotation)
        return self.backend.move_linear(pose, speed or self.config.linear_speed,
                                        acceleration or self.config.linear_acceleration)

    def hover_above(self, x: float, y: float, tool_rotation: float = 0.0,
                    hover_height: float | None = None) -> bool:
        """Move to a safe transit height directly above a board point.

        Parameters
        ----------
        x, y : float
            Board-surface coordinates (metres).
        tool_rotation : float, optional
            Gripper rotation about the board normal (radians).  Default ``0``.
        hover_height : float or None, optional
            Height above the board (metres); uses
            `ArmConfig.travel_height` if ``None``.

        Returns
        -------
        bool
        """
        height = self.config.travel_height if hover_height is None else hover_height
        return self.move_to_board_point(x, y, height, tool_rotation)

    def press_onto_board(self, x: float, y: float, tool_rotation: float = 0.0,
                         contact_depth: float | None = None) -> bool:
        """Press the tool down onto the board surface at a point.

        This is the one operation that intentionally targets the board surface,
        driving slightly below it to ensure contact.

        Parameters
        ----------
        x, y : float
            Board-surface coordinates (metres).
        tool_rotation : float, optional
            Gripper rotation about the board normal (radians).  Default ``0``.
        contact_depth : float or None, optional
            Distance below the surface to drive (metres); uses
            `ArmConfig.contact_depth` if ``None``.

        Returns
        -------
        bool
        """
        depth = self.config.contact_depth if contact_depth is None else contact_depth
        return self.move_to_board_point(x, y, -depth, tool_rotation)

    def descend_and_press(self, x: float, y: float, tool_rotation: float = 0.0) -> bool:
        """Hover above a board point, then press onto the surface.

        Parameters
        ----------
        x, y : float
            Board-surface coordinates (metres).
        tool_rotation : float, optional
            Gripper rotation about the board normal (radians).  Default ``0``.

        Returns
        -------
        bool
            Result of the final press.
        """
        self.hover_above(x, y, tool_rotation)
        return self.press_onto_board(x, y, tool_rotation)

    def lift_off_board(self, x: float | None = None, y: float | None = None,
                       tool_rotation: float = 0.0, hover_height: float | None = None) -> bool:
        """Raise the tool to transit height.

        Parameters
        ----------
        x, y : float or None, optional
            Board point to lift over.  If either is ``None``, the arm's current
            board ``(x, y)`` is used (a straight-up lift).
        tool_rotation : float, optional
            Gripper rotation about the board normal (radians).  Default ``0``.
        hover_height : float or None, optional
            Height above the board (metres); config default if ``None``.

        Returns
        -------
        bool
        """
        if x is None or y is None:
            x, y = self.current_board_xy()
        return self.hover_above(x, y, tool_rotation, hover_height)

    # ------------------------------------------------------------------ #
    # Gripper
    # ------------------------------------------------------------------ #
    def grip(self) -> None:
        """Close the gripper, if this arm has one."""
        if self.gripper is not None:
            self.gripper.grip()

    def release(self) -> None:
        """Open the gripper, if this arm has one."""
        if self.gripper is not None:
            self.gripper.release()

    # ------------------------------------------------------------------ #
    # State
    # ------------------------------------------------------------------ #
    def current_tcp_pose(self) -> list[float]:
        """Return the current TCP pose ``[x, y, z, rx, ry, rz]``.

        Returns
        -------
        list of float
        """
        return self.backend.current_tcp_pose()

    def current_board_point(self) -> tuple[float, float, float]:
        """Return the tool's current position in board coordinates.

        Returns
        -------
        tuple of float
            ``(x, y, height_above_board)`` in metres.
        """
        board = self.calibration.base_point_to_board(self.current_tcp_pose()[:3])
        return float(board[0]), float(board[1]), float(board[2])

    def current_board_xy(self) -> tuple[float, float]:
        """Return the tool's current board ``(x, y)``, ignoring height.

        Returns
        -------
        tuple of float
        """
        x, y, _ = self.current_board_point()
        return x, y

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return f"Arm('{self.name}')"
