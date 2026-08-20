from __future__ import annotations

from enum import Enum
from pathlib import Path

import numpy as np
from ikpy.chain import Chain
from ur_analytic_ik import ur5e

from .rotation import rot_matrix_to_rot_vec, rot_vec_to_rot_matrix

#: Bundled UR5e URDF (base_link -> ee_link), used to build the FK chain below.
_URDF_PATH = Path(__file__).with_name("resources") / "ur5e.urdf"

#: The 8-link ikpy chain: an origin link, the 6 UR5e joints, and a fixed
#: end-effector link. Only the 6 joint links are active.
_CHAIN = Chain.from_urdf_file(
    str(_URDF_PATH), base_elements=["base_link"],
    active_links_mask=[False, True, True, True, True, True, True, False])


class Joint(Enum):
    """One of the UR5e's six revolute joints, in kinematic-chain order.

    The value is the joint's index into a UR joint-angle array ``[j0..j5]``.
    """
    BASE  = 0
    SHOULDER = 1
    ELBOW         = 2
    WRIST_1       = 3
    WRIST_2       = 4
    WRIST_3       = 5
    #: The fixed tool/TCP frame beyond wrist_3 (the ``ee_link`` in the URDF).
    #: This is the actual end effector, not the wrist_3 joint axis.
    TCP           = 6


def _to_transform(pose) -> np.ndarray:
    p = np.asarray(pose, dtype=float).reshape(6)
    transform = np.eye(4)
    transform[:3, :3] = rot_vec_to_rot_matrix(p[3:])
    transform[:3, 3] = p[:3]
    return transform


def forward_kinematics(joint_angles, joint: Joint, tcp_offset=None) -> list[float]:
    """Compute the pose of ``joint`` for a given set of UR5e joint angles.

    Parameters
    ----------
    joint_angles : array_like, shape (6,)
        UR5e joint angles ``[j0..j5]``, in radians.
    joint : Joint
        Which joint's pose to return.
    tcp_offset : array_like, shape (6,), optional
        Real gripper's fixed offset from the flange, as a UR-style pose
        (``Arm.backend.get_tcp_offset()``). Only affects ``Joint.TCP``.

    Returns
    -------
    list of float
        Pose ``[x, y, z, rx, ry, rz]`` of ``joint``'s frame, in the arm's
        base frame.
    """
    angles = np.asarray(joint_angles, dtype=float).reshape(6)
    if joint is Joint.TCP:
        tcp_transform = _to_transform(tcp_offset) if tcp_offset is not None else np.eye(4)
        transform = ur5e.forward_kinematics_with_tcp(*angles.tolist(), tcp_transform)
    else:
        # _CHAIN's links are [origin, j0..j5, fixed ee link]; pad both ends to match.
        chain_angles = [0.0, *angles.tolist(), 0.0]
        transform = _CHAIN.forward_kinematics(chain_angles, full_kinematics=True)[joint.value + 1]
    xyz = transform[:3, 3].tolist()
    rotvec = rot_matrix_to_rot_vec(transform[:3, :3]).tolist()
    return xyz + rotvec


def inverse_kinematics(pose, tcp_offset=None) -> list[list[float]]:
    """Compute every analytical inverse-kinematics solution for ``pose``.

    Uses the UR5e's closed-form solution (``ur_analytic_ik``), which yields
    up to 8 discrete joint-angle solutions for a reachable pose.

    Parameters
    ----------
    pose : array_like, shape (6,)
        Desired real gripper (TCP) pose ``[x, y, z, rx, ry, rz]`` in the
        arm's base frame.
    tcp_offset : array_like, shape (6,), optional
        Real gripper's fixed offset from the flange, as a UR-style pose
        (``Arm.backend.get_tcp_offset()``). Without it, ``pose`` is solved
        as a bare-flange target, overshooting the true tool tip forward by
        the gripper's length whenever a real offset is configured.

    Returns
    -------
    list of list of float
        Every valid solution, each a 6-element joint-angle list (radians).
        Empty if ``pose`` is unreachable.
    """
    eef_pose = _to_transform(pose)
    if tcp_offset is not None:
        solutions = ur5e.inverse_kinematics_with_tcp(eef_pose, _to_transform(tcp_offset))
    else:
        solutions = ur5e.inverse_kinematics(eef_pose)
    return [list(solution) for solution in solutions]


# --------------------------------------------------------------------------- #
# Branch filters
# --------------------------------------------------------------------------- #
# The UR5e's (up to) 8 analytic IK solutions for a reachable pose are exactly
# the 8 combinations of 3 independent binary choices ("branches"): shoulder
# left/right, elbow up/down, and wrist flip/no-flip. Each is readable directly
# off one joint angle's sign -- no forward kinematics needed to filter by it.
# Empirically verified (via forward_kinematics, comparing solutions for the
# same target pose): j1 < 0 always gives the higher elbow position regardless
# of the shoulder branch, and every pose's 8 solutions realize each of the 8
# sign(j0), sign(j1), sign(j4) combinations exactly once. "left"/"right" and
# "flip"/"no-flip" follow this arm's sign convention -- re-check with
# forward_kinematics if it doesn't match intuition for a different mount.

def shoulder_left(solutions: list[list[float]]) -> list[list[float]]:
    """Keep only ``solutions`` with the base joint (j0) rotated positive."""
    return [s for s in solutions if s[0] > 0]


def shoulder_right(solutions: list[list[float]]) -> list[list[float]]:
    """Keep only ``solutions`` with the base joint (j0) rotated negative."""
    return [s for s in solutions if s[0] < 0]


def elbow_up(solutions: list[list[float]]) -> list[list[float]]:
    """Keep only ``solutions`` with the elbow configured up (shoulder joint j1 negative)."""
    return [s for s in solutions if s[1] < 0]


def elbow_down(solutions: list[list[float]]) -> list[list[float]]:
    """Keep only ``solutions`` with the elbow configured down (shoulder joint j1 positive)."""
    return [s for s in solutions if s[1] > 0]


def wrist_flip(solutions: list[list[float]]) -> list[list[float]]:
    """Keep only ``solutions`` with the wrist joint (j4) rotated negative."""
    return [s for s in solutions if s[4] < 0]


def wrist_no_flip(solutions: list[list[float]]) -> list[list[float]]:
    """Keep only ``solutions`` with the wrist joint (j4) rotated positive."""
    return [s for s in solutions if s[4] > 0]
