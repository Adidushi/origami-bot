"""Hardware backends for the arms and gripper.

The high-level `Arm` class talks to the world through two small
interfaces -- one for arm motion, one for the gripper -- so the same choreography
runs against either real hardware or an in-memory simulator:

* `RTDEArmBackend` / `RobotiqGripperBackend` wrap the real
  ``rtde_control`` / ``rtde_receive`` interfaces and the Robotiq gripper.  Their
  third-party dependencies are imported lazily, so this module loads fine on a
  machine with no robot libraries installed.
* `SimulatedArmBackend` / `SimulatedGripperBackend` keep their state
  in memory and log every command, enabling development and dry-runs with no
  hardware attached.
"""
from __future__ import annotations

from typing import Protocol, Sequence, runtime_checkable

import numpy as np


@runtime_checkable
class ArmBackend(Protocol):
    """Minimal arm-motion interface required by `Arm`."""

    def move_linear(self, pose: Sequence[float], speed: float, acceleration: float) -> bool:
        """Move the TCP in a straight line to ``pose`` (TCP space, [x,y,z,rx,ry,rz])."""
        ...

    def move_joint_space(self, pose: Sequence[float], speed: float, acceleration: float) -> bool:
        """Move to a TCP pose via joint interpolation (moveJ with IK).

        The TCP traces a curved path; use for fast transit moves where the
        exact Cartesian path does not matter.
        """
        ...

    def move_joints(self, angles: Sequence[float], speed: float, acceleration: float) -> bool:
        """Move to a set of joint angles [j0..j5] (radians)."""
        ...

    def current_tcp_pose(self) -> list[float]:
        """Return the current TCP pose ``[x, y, z, rx, ry, rz]`` in the arm base frame."""
        ...

    def current_joint_angles(self) -> list[float]:
        """Return the current joint angles ``[j0..j5]`` in radians."""
        ...


@runtime_checkable
class GripperBackend(Protocol):
    """Minimal gripper interface required by `Arm`."""

    def grip(self) -> None:
        """Close the gripper."""
        ...

    def release(self) -> None:
        """Open the gripper."""
        ...

    def opening(self) -> int:
        """Return the current gripper position (0 = open, 255 = closed)."""
        ...

    def goto(self, percentage: float) -> None:
        """Move the gripper to a given opening percentage (0 = open, 1 = closed)."""
        ...


# --------------------------------------------------------------------------- #
# Real hardware
# --------------------------------------------------------------------------- #
class RTDEArmBackend:
    """Adapter around ``rtde_control`` / ``rtde_receive`` for one UR arm.

    Parameters
    ----------
    ip : str
        IP address of the arm's controller.

    Notes
    -----
    ``rtde_control`` and ``rtde_receive`` are imported lazily so this module
    loads fine on machines without robot libraries installed.
    """

    def __init__(self, ip: str) -> None:
        import rtde_control
        import rtde_receive

        self.ip = ip
        self.control = rtde_control.RTDEControlInterface(ip)
        self.receive = rtde_receive.RTDEReceiveInterface(ip)

    def move_linear(self, pose, speed: float, acceleration: float) -> bool:
        return bool(self.control.moveL(list(pose), speed, acceleration))

    def move_joint_space(self, pose, speed: float, acceleration: float) -> bool:
        return bool(self.control.moveJ(list(pose), speed, acceleration))

    def move_joints(self, angles, speed: float, acceleration: float) -> bool:
        return bool(self.control.moveJ(list(angles), speed, acceleration))

    def current_tcp_pose(self) -> list[float]:
        return list(self.receive.getActualTCPPose())

    def current_joint_angles(self) -> list[float]:
        return list(self.receive.getActualQ())


class RobotiqGripperBackend:
    """Adapter around the community Robotiq gripper module.

    Parameters
    ----------
    ip : str
        IP address hosting the gripper's socket server.
    port : int, optional
        TCP port of the gripper server.  Default ``63352``.
    """

    def __init__(self, ip: str, port: int = 63352) -> None:
        from mvmt.robotq_gripper import RobotiqGripper

        self._gripper = RobotiqGripper()
        self._gripper.connect(ip, port)
        self._gripper.activate()

    def grip(self) -> None:
        self._gripper.close()

    def release(self) -> None:
        self._gripper.open()

    def goto(self, percentage: float) -> None:
        """Move the gripper to a given opening percentage (0 = open, 1 = closed)."""
        self._gripper.move(int(percentage * 255))

    def opening(self) -> int:
        return int(self._gripper.position())

    def disconnect(self) -> None:
        self._gripper.disconnect()


# --------------------------------------------------------------------------- #
# Simulation
# --------------------------------------------------------------------------- #
class SimulatedArmBackend:
    """In-memory arm that records its pose and logs every command.

    Parameters
    ----------
    pose : sequence of float or None, optional
        Initial TCP pose.  Defaults to a plausible hovering pose.
    name : str, optional
        Label used in logs.  Default ``'sim'``.
    """

    def __init__(self, pose: Sequence[float] | None = None, name: str = "sim") -> None:
        self.name = name
        self._pose = list(pose) if pose is not None else [0.0, -0.4, 0.2, 0.0, float(np.pi), 0.0]
        self._joints = [0.0] * 6
        self.log: list[tuple[str, list[float]]] = []

    def move_linear(self, pose, speed: float, acceleration: float) -> bool:
        self._pose = [float(v) for v in pose]
        self.log.append(("move_linear", self._pose.copy()))
        return True

    def move_joint_space(self, pose, speed: float, acceleration: float) -> bool:
        self._pose = [float(v) for v in pose]
        self.log.append(("move_joint_space", self._pose.copy()))
        return True

    def move_joints(self, angles, speed: float, acceleration: float) -> bool:
        self._joints = [float(a) for a in angles]
        self.log.append(("move_joints", self._joints.copy()))
        return True

    def current_tcp_pose(self) -> list[float]:
        return list(self._pose)

    def current_joint_angles(self) -> list[float]:
        return list(self._joints)


class SimulatedGripperBackend:
    """In-memory gripper.

    Parameters
    ----------
    name : str, optional
        Label used in logs.  Default ``'sim'``.
    """

    def __init__(self, name: str = "sim") -> None:
        self.name = name
        self._opening = 0
        self.log: list[str] = []

    def grip(self) -> None:
        self._opening = 255
        self.log.append("grip")

    def release(self) -> None:
        self._opening = 0
        self.log.append("release")

    def opening(self) -> int:
        return self._opening
    
    def goto(self, percentage: float) -> None:
        """Move the gripper to a given opening percentage (0 = open, 1 = closed)."""
        self._opening = int(percentage * 255)
        self.log.append(f"goto {percentage}")