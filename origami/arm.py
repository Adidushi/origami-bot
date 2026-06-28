"""Single-arm control in world coordinates.

World space is centred on the board: x/y across the surface, z = height above
it (z=0 is the board surface).  `ArmCalibration` converts world targets into
the arm's own base-frame TCP poses so the robot executes them correctly.

All Cartesian motion uses linear interpolation (moveL).  Explicit joint-space
control is available only through `move_to_joints` and `rotate_joint`.

API summary
-----------
State
    current_tcp_pose()      raw [x,y,z,rx,ry,rz] in the arm base frame
    current_world_pos()     (x, y, z) in world space
    get_tool_pos()          (x, y, z) in world space (alias for current_world_pos)

Motion — arm (TCP) frame
    move_to_tcp(pose)            move to a raw TCP pose via moveL

Motion — world frame
    move_to_world(x,y,z)         move to world position via moveL
    move_offset_world(dx,dy,dz)  relative move in world space via moveL
    move_up(d) / move_down(d)    vertical offsets in world z via moveL

Gripper
    grip() / release()

Joint control
    move_to_joints(angles)        move to absolute joint angles
    rotate_joint(joint, delta)    rotate one joint by a delta angle (radians)

Coordinate conversion (no motion)
    world_to_tcp(x,y,z)     world position → full TCP pose
    tcp_to_world(pose)      TCP pose → world (x,y,z)

Convenience moves
    move_to_clearance(x,y)  linear transit to safe height above (x,y)
    press(x,y)              linear descent onto board surface at (x,y)
    lift()                  linear rise to clearance from current (x,y)
"""
from __future__ import annotations

import math
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
    home: list[float, float, float] | None = None


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
    
    def go_home(self):
        home_pos = self.config.home
        self.move_to_joints(home_pos)


    # ------------------------------------------------------------------ #
    # Motion — arm (TCP) frame
    # ------------------------------------------------------------------ #
    def move_to_tcp(self, pose: list[float],
                    speed: float | None = None,
                    acceleration: float | None = None) -> bool:
        """Move to a raw TCP pose ``[x, y, z, rx, ry, rz]`` in the arm base frame via moveL.

        Parameters
        ----------
        pose : list of float
            Target TCP pose.
        speed, acceleration : float or None
            Override config defaults.
        """
        spd = speed or self.config.speed
        acc = acceleration or self.config.acceleration
        return self.backend.move_linear(pose, spd, acc)

    def movej_to_tcp(self, pose: list[float], speed: float | None = None, acceleration: float | None = None) -> bool:
        """Move to a raw TCP pose ``[x, y, z, rx, ry, rz]`` in the arm base frame via moveJ_IK (as we're giving it a TCP pose
        and calculating joint angles in order to move in joint space).

        Parameters
        ----------
        pose : list of float
            Target TCP pose.
        speed, acceleration : float or None
            Override config defaults.
        """
        spd = speed or self.config.speed
        acc = acceleration or self.config.acceleration
        return self.backend.move_joint_space(pose, spd, acc)
    
    # ------------------------------------------------------------------ #
    # Motion — world frame
    # ------------------------------------------------------------------ #
    def move_to_world(self, x: float, y: float, z: float,
                      tool_rotation: float = 0.0,
                      speed: float | None = None,
                      acceleration: float | None = None,
                      sideways: bool = False) -> bool:
        """Move to a world position via moveL.

        Parameters
        ----------
        x, y, z : float
            World coordinates (metres).  z = 0 is the board surface.
        tool_rotation : float, optional
            Gripper spin about the board normal (radians).  Default 0.
        speed, acceleration : float or None
            Override config defaults.
        sideways : bool, optional
            When ``True`` the gripper is oriented horizontally for a side
            approach (see `ArmCalibration.gripper_orientation`).
        """
        return self.move_to_tcp(
            self.world_to_tcp(x, y, z, tool_rotation, sideways),
            speed, acceleration,
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

    def goto(self, percentage: float, blocking = True) -> None:
        """Set the gripper to a given opening percentage (0 = open, 1 = closed)."""
        if self.gripper is not None:
            self.gripper.goto(percentage, blocking)

    # ------------------------------------------------------------------ #
    # Joint control
    # ------------------------------------------------------------------ #
    def move_to_joints(self, angles: list[float],
                       speed: float | None = None,
                       acceleration: float | None = None) -> bool:
        """Move directly to absolute joint angles ``[j0..j5]`` (radians)."""
        spd = speed or self.config.joint_speed
        acc = acceleration or self.config.joint_acceleration
        return self.backend.move_joints(angles, spd, acc)

    def rotate_joint(self, joint: int, delta: float, joint_speed: float | None = None) -> bool:
        """Rotate one joint by ``delta`` radians from its current angle.

        Parameters
        ----------
        joint : int
            Joint index 0–5 (0 = base, 5 = wrist).
        delta : float
            Angle change in radians (positive = counter-clockwise).
        joint_speed : float or None, optional
            Speed for the joint movement.  ``None`` uses the default speed.
        """
        angles = list(self.backend.current_joint_angles())
        angles[joint] += delta
        spd = joint_speed or self.config.joint_speed
        return self.backend.move_joints(
            angles, spd, self.config.joint_acceleration
        )

    def get_joint_angles(self) -> list[float]:
        """Return the current joint angles ``[j0..j5]`` in radians."""
        return self.backend.current_joint_angles()

    # ------------------------------------------------------------------ #
    # Coordinate conversion (no motion)
    # ------------------------------------------------------------------ #
    def world_to_tcp(self, x: float, y: float, z: float,
                     tool_rotation: float = 0.0,
                     sideways: bool = False) -> list[float]:
        """Convert a world position to a full TCP pose (no motion).

        Parameters
        ----------
        x, y, z : float
            World coordinates.
        tool_rotation : float, optional
            Gripper spin about the board normal (radians).  Default 0.
        sideways : bool, optional
            When ``True`` the gripper is oriented horizontally.

        Returns
        -------
        list of float
            ``[x, y, z, rx, ry, rz]`` ready for ``moveL``.
        """
        return self.calibration.tcp_pose(x, y, z, tool_rotation, sideways)

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
                          sideways: bool = False) -> bool:
        """Move to safe transit height above ``(x, y)`` via moveL.

        Parameters
        ----------
        x, y : float
            World target in board plane (metres).
        tool_rotation : float, optional
            Gripper spin about the board normal.  Default 0.
        sideways : bool, optional
            When ``True`` the gripper is oriented horizontally.
        """
        return self.move_to_world(x, y, self.config.clearance_z, tool_rotation,
                                  sideways=sideways)

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
