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
        """Move the TCP in a straight line to ``pose``.

        Parameters
        ----------
        pose : sequence of float
            Target UR pose ``[x, y, z, rx, ry, rz]``.
        speed : float
            Tool speed (m/s).
        acceleration : float
            Tool acceleration (m/s^2).

        Returns
        -------
        bool
            ``True`` on success.
        """
        ...

    def current_tcp_pose(self) -> list[float]:
        """Return the current TCP pose ``[x, y, z, rx, ry, rz]``.

        Returns
        -------
        list of float
        """
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
        """Return the current gripper position (0 open ... 255 closed).

        Returns
        -------
        int
        """
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

    Attributes
    ----------
    control : rtde_control.RTDEControlInterface
        Motion interface.
    receive : rtde_receive.RTDEReceiveInterface
        State interface.

    Notes
    -----
    ``rtde_control`` and ``rtde_receive`` are imported lazily inside ``__init__``
    so importing this module never requires the robot libraries.
    """

    def __init__(self, ip: str) -> None:
        import rtde_control
        import rtde_receive

        self.ip = ip
        self.control = rtde_control.RTDEControlInterface(ip)
        self.receive = rtde_receive.RTDEReceiveInterface(ip)

    def move_linear(self, pose, speed: float, acceleration: float) -> bool:
        """Move the TCP linearly to ``pose`` (see `ArmBackend.move_linear()`)."""
        return bool(self.control.moveL(list(pose), speed, acceleration))

    def current_tcp_pose(self) -> list[float]:
        """Return the live TCP pose from ``rtde_receive``."""
        return list(self.receive.getActualTCPPose())


class RobotiqGripperBackend:
    """Adapter around the community Robotiq gripper module.

    Parameters
    ----------
    ip : str
        IP address hosting the gripper's socket server.
    port : int, optional
        TCP port of the gripper server.  Default ``63352``.

    Notes
    -----
    Wraps `mvmt.robotq_gripper.RobotiqGripper`, connecting and activating
    on construction.
    """

    def __init__(self, ip: str, port: int = 63352) -> None:
        from mvmt.robotq_gripper import RobotiqGripper

        self._gripper = RobotiqGripper()
        self._gripper.connect(ip, port)
        self._gripper.activate()

    def grip(self) -> None:
        """Fully close the gripper."""
        self._gripper.close()

    def release(self) -> None:
        """Fully open the gripper."""
        self._gripper.open()

    def opening(self) -> int:
        """Return the current gripper position (0 open ... 255 closed)."""
        return int(self._gripper.position())

    def disconnect(self) -> None:
        """Close the gripper's network connection."""
        self._gripper.disconnect()


# --------------------------------------------------------------------------- #
# Simulation
# --------------------------------------------------------------------------- #
class SimulatedArmBackend:
    """In-memory arm that records its pose and logs every move.

    Parameters
    ----------
    pose : sequence of float or None, optional
        Initial TCP pose.  Defaults to a plausible "hovering, pointing down"
        pose.
    name : str, optional
        Label for the arm (used in logs).  Default ``'sim'``.

    Attributes
    ----------
    log : list of tuple
        Chronological list of ``(command, pose)`` pairs issued to this backend.
    """

    def __init__(self, pose: Sequence[float] | None = None, name: str = "sim") -> None:
        self.name = name
        self._pose = list(pose) if pose is not None else [0.0, -0.4, 0.2, 0.0, float(np.pi), 0.0]
        self.log: list[tuple[str, list[float]]] = []

    def move_linear(self, pose, speed: float, acceleration: float) -> bool:
        """Record a linear move and update the stored pose.

        Returns
        -------
        bool
            Always ``True``.
        """
        self._pose = [float(v) for v in pose]
        self.log.append(("move_linear", self._pose))
        return True

    def current_tcp_pose(self) -> list[float]:
        """Return the last commanded TCP pose."""
        return list(self._pose)


class SimulatedGripperBackend:
    """In-memory gripper tracking an opening of 0 (open) to 255 (closed).

    Parameters
    ----------
    name : str, optional
        Label for the gripper (used in logs).  Default ``'sim'``.

    Attributes
    ----------
    log : list of str
        Chronological list of ``'grip'`` / ``'release'`` commands.
    """

    def __init__(self, name: str = "sim") -> None:
        self.name = name
        self._opening = 0
        self.log: list[str] = []

    def grip(self) -> None:
        """Record a close command."""
        self._opening = 255
        self.log.append("grip")

    def release(self) -> None:
        """Record an open command."""
        self._opening = 0
        self.log.append("release")

    def opening(self) -> int:
        """Return the current opening (0 open ... 255 closed)."""
        return self._opening
