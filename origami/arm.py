"""Single-arm control in world coordinates.

World space is centred on the board: x/y across the surface, z = height above
it (z=0 is the board surface).  `ArmCalibration` converts world targets into
the arm's own base-frame TCP poses so the robot executes them correctly.

API summary
-----------
State
    current_tcp_pose()      raw [x,y,z,rx,ry,rz] in the arm base frame
    current_world_pos()     (x, y, z) in world space
    get_tool_pos()          (x, y, z) in world space (alias for current_world_pos)

Motion — arm (TCP) frame
    move_to_tcp(pose, motion)   move to a raw TCP pose; motion='linear'|'joint'

Motion — world frame
    move_to_world(x,y,z, motion)     move to world position
    move_offset_world(dx,dy,dz)      relative move in world space
    move_up(d) / move_down(d)        vertical offsets in world z

Gripper
    grip() / release()

Joint control
    rotate_joint(joint, delta)  rotate one joint by a delta angle (radians)

Coordinate conversion (no motion)
    world_to_tcp(x,y,z)     world position → full TCP pose
    tcp_to_world(pose)      TCP pose → world (x,y,z)

Convenience moves
    move_to_clearance(x,y)  joint-space transit to safe height above (x,y)
    press(x,y)              linear descent onto board surface at (x,y)
    lift()                  linear rise to clearance from current (x,y)
"""
from __future__ import annotations

from dataclasses import dataclass

from . import backends as backends_mod
from .backends import ArmBackend, GripperBackend
from .coords import ArmCalibration


@dataclass
class ArmConfig:
    """Per-arm motion defaults.

    Parameters
    ----------
    clearance_z : float
        Safe transit height in world z (metres above board).  Default 0.10.
    contact_depth : float
        How far below z=0 to drive when pressing onto the board to guarantee
        firm contact (metres, positive).  Default 0.002.
    speed : float
        Default TCP speed for linear moves (m/s).  Default 0.25.
    acceleration : float
        Default TCP acceleration (m/s²).  Default 0.5.
    joint_speed : float
        Default joint speed for joint moves (rad/s).  Default 1.0.
    joint_acceleration : float
        Default joint acceleration (rad/s²).  Default 1.4.
    """

    clearance_z: float = 0.10
    contact_depth: float = 0
    speed: float = 0.25
    acceleration: float = 0.5
    joint_speed: float = 1.0
    joint_acceleration: float = 1.4


class Arm:
    """One robot arm commanded in world coordinates.

    Parameters
    ----------
    name : str
        Label (e.g. ``'left'``).
    backend : ArmBackend
        Motion backend (real hardware or simulated).
    calibration : ArmCalibration
        This arm's world-to-base-frame calibration.
    gripper : GripperBackend or None, optional
        Gripper backend, if fitted.
    config : ArmConfig or None, optional
        Motion defaults.

    See Also
    --------
    Arm.real, Arm.simulated : Convenience constructors.
    """

    def __init__(self, name: str, backend: ArmBackend, calibration: ArmCalibration,
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
    def real(cls, name: str, arm_ip: str, calibration: ArmCalibration,
             gripper_ip: str | None = None, gripper_port: int = 63352,
             config: ArmConfig | None = None) -> "Arm":
        """Build an arm driving real hardware over RTDE."""
        gripper = backends_mod.RobotiqGripperBackend(gripper_ip, gripper_port) if gripper_ip else None
        return cls(name, backends_mod.RTDEArmBackend(arm_ip), calibration, gripper, config)

    @classmethod
    def simulated(cls, name: str, calibration: ArmCalibration,
                  config: ArmConfig | None = None, start_pose=None) -> "Arm":
        """Build a fully simulated arm (no hardware required)."""
        return cls(name, backends_mod.SimulatedArmBackend(start_pose, name=name), calibration,
                   backends_mod.SimulatedGripperBackend(name=name), config)

    # ------------------------------------------------------------------ #
    # State
    # ------------------------------------------------------------------ #
    def current_tcp_pose(self) -> list[float]:
        """Current TCP pose ``[x, y, z, rx, ry, rz]`` in the arm's base frame."""
        return self.backend.current_tcp_pose()

    def current_world_pos(self) -> tuple[float, float, float]:
        """Current tool position in world space ``(x, y, z)``."""
        w = self.calibration._arm_to_world_xyz(self.backend.current_tcp_pose()[:3])
        return float(w[0]), float(w[1]), float(w[2])

    def get_tool_pos(self) -> tuple[float, float, float]:
        """Current tool position in world space ``(x, y, z)``.

        Convenience alias for `current_world_pos` with a shorter, more
        intent-revealing name for use in action planning.
        """
        return self.current_world_pos()

    # ------------------------------------------------------------------ #
    # Motion — arm (TCP) frame
    # ------------------------------------------------------------------ #
    def move_to_tcp(self, pose: list[float],
                    speed: float | None = None,
                    acceleration: float | None = None,
                    motion: str = "linear") -> bool:
        """Move to a raw TCP pose ``[x, y, z, rx, ry, rz]`` in the arm base frame.

        Parameters
        ----------
        pose : list of float
            Target TCP pose.
        speed, acceleration : float or None
            Override config defaults.
        motion : {'linear', 'joint'}
            ``'linear'`` (default) — TCP traces a straight Cartesian line
            (``moveL``); precise, use near paper and magnets.
            ``'joint'`` — joint angles interpolate, curved TCP path
            (``moveJ`` with IK); faster for large transit moves.
        """
        spd = speed or self.config.speed
        acc = acceleration or self.config.acceleration
        if motion == "joint":
            return self.backend.move_joint_space(pose, spd, acc)
        return self.backend.move_linear(pose, spd, acc)

    # ------------------------------------------------------------------ #
    # Motion — world frame
    # ------------------------------------------------------------------ #
    def move_to_world(self, x: float, y: float, z: float,
                      tool_rotation: float = 0.0,
                      speed: float | None = None,
                      acceleration: float | None = None,
                      motion: str = "linear") -> bool:
        """Move to a world position.

        Parameters
        ----------
        x, y, z : float
            World coordinates (metres).  z = 0 is the board surface.
        tool_rotation : float, optional
            Gripper spin about the board normal (radians).  Default 0.
        speed, acceleration : float or None
            Override config defaults.
        motion : {'linear', 'joint'}
            Passed through to `move_to_tcp`.  Default ``'linear'``.
        """
        return self.move_to_tcp(
            self.world_to_tcp(x, y, z, tool_rotation),
            speed, acceleration, motion,
        )

    def move_offset_world(self, dx: float, dy: float, dz: float,
                          tool_rotation: float | None = None) -> bool:
        """Move by ``(dx, dy, dz)`` relative to the current world position.

        Parameters
        ----------
        dx, dy, dz : float
            Offset in world coordinates (metres).
        tool_rotation : float or None, optional
            New gripper rotation.  ``None`` preserves the current TCP
            orientation exactly (no recomputation).
        """
        tcp = self.current_tcp_pose()
        x, y, z = self.tcp_to_world(tcp)
        if tool_rotation is None:
            new_xyz = self.calibration._world_to_arm_xyz(x + dx, y + dy, z + dz)
            return self.move_to_tcp(list(new_xyz) + tcp[3:])
        return self.move_to_world(x + dx, y + dy, z + dz, tool_rotation)

    def move_up(self, distance: float) -> bool:
        """Rise ``distance`` metres in world z from the current position."""
        return self.move_offset_world(0.0, 0.0, distance)

    def move_down(self, distance: float) -> bool:
        """Descend ``distance`` metres in world z from the current position."""
        return self.move_offset_world(0.0, 0.0, -distance)

    # ------------------------------------------------------------------ #
    # Gripper
    # ------------------------------------------------------------------ #
    def grip(self) -> None:
        """Close the gripper."""
        if self.gripper is not None:
            self.gripper.grip()

    def release(self) -> None:
        """Open the gripper."""
        if self.gripper is not None:
            self.gripper.release()

    def goto(self, percentage: float) -> None:
        """Set the gripper to a given opening percentage (0 = open, 1 = closed)."""
        if self.gripper is not None:
            self.gripper.goto(percentage)

    # ------------------------------------------------------------------ #
    # Joint control
    # ------------------------------------------------------------------ #
    def rotate_joint(self, joint: int, delta: float) -> bool:
        """Rotate one joint by ``delta`` radians from its current angle.

        Parameters
        ----------
        joint : int
            Joint index 0–5 (0 = base, 5 = wrist).
        delta : float
            Angle change in radians (positive = counter-clockwise).
        """
        angles = list(self.backend.current_joint_angles())
        angles[joint] += delta
        return self.backend.move_joints(
            angles, self.config.joint_speed, self.config.joint_acceleration
        )

    # ------------------------------------------------------------------ #
    # Coordinate conversion (no motion)
    # ------------------------------------------------------------------ #
    def world_to_tcp(self, x: float, y: float, z: float,
                     tool_rotation: float = 0.0) -> list[float]:
        """Convert a world position to a full TCP pose (no motion).

        Parameters
        ----------
        x, y, z : float
            World coordinates.
        tool_rotation : float, optional
            Gripper spin about the board normal (radians).  Default 0.

        Returns
        -------
        list of float
            ``[x, y, z, rx, ry, rz]`` ready for ``moveL``.
        """
        return self.calibration.tcp_pose(x, y, z, tool_rotation)

    def tcp_to_world(self, pose: list[float]) -> tuple[float, float, float]:
        """Convert a TCP pose to world coordinates (no motion).

        Parameters
        ----------
        pose : list of float
            TCP pose ``[x, y, z, rx, ry, rz]`` in the arm base frame.

        Returns
        -------
        tuple of float
            World ``(x, y, z)``.
        """
        w = self.calibration._arm_to_world_xyz(pose[:3])
        return float(w[0]), float(w[1]), float(w[2])

    # ------------------------------------------------------------------ #
    # Convenience moves
    # ------------------------------------------------------------------ #
    def move_to_clearance(self, x: float, y: float,
                          tool_rotation: float = 0.0,
                          motion: str = "linear") -> bool:
        """Move to safe transit height above ``(x, y)``.

        Defaults to joint-space motion (faster transit, curved path is fine
        at clearance height where obstacles are absent).

        Parameters
        ----------
        x, y : float
            World target in board plane (metres).
        tool_rotation : float, optional
            Gripper spin about the board normal.  Default 0.
        motion : {'joint', 'linear'}, optional
            Motion type.  Default ``'joint'``.
        """
        return self.move_to_world(x, y, self.config.clearance_z, tool_rotation, motion=motion)

    def press(self, x: float, y: float, tool_rotation: float = 0.0) -> bool:
        """Drive the tool onto the board surface at ``(x, y)``.

        Descends ``contact_depth`` below z=0 to guarantee firm contact.
        Uses linear motion for a straight, predictable descent.

        Parameters
        ----------
        x, y : float
            Board position (metres).
        tool_rotation : float, optional
            Gripper spin about the board normal.  Default 0.
        """
        return self.move_to_world(x, y, -self.config.contact_depth, tool_rotation)

    def lift(self) -> bool:
        """Rise straight up to clearance height from the current ``(x, y)``.

        Uses linear motion so the TCP moves vertically.
        """
        x, y, _ = self.get_tool_pos()
        return self.move_to_world(x, y, self.config.clearance_z)

    def __repr__(self) -> str:  # pragma: no cover
        return f"Arm('{self.name}')"
