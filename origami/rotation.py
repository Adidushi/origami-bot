"""Tool-orientation utilities for UR robot arms.

Convention
----------
The rotation matrix R from a UR TCP pose has columns:
  R[:, 0]  = tool x-axis expressed in the arm's base frame
  R[:, 1]  = tool y-axis expressed in the arm's base frame
  R[:, 2]  = tool z-axis expressed in the arm's base frame

Tool frame (gripper definition)
  tool z  -- points away from the tooltip
  tool y  -- direction of the finger-opening gap between gripper fingers
  tool x  -- flat face of the gripper; derived as cross(tool_y, tool_z)

All direction vectors are expressed in the arm's base frame.
"""
from __future__ import annotations

from enum import Enum

import math

import numpy as np
from scipy.spatial.transform import Rotation


# ---------------------------------------------------------------------------
# Core math utilities  (generic — no tool or base-frame dependency)
# ---------------------------------------------------------------------------

def rot_vec_to_rot_matrix(rotvec) -> np.ndarray:
    """Rotation vector → 3×3 rotation matrix.

    Converts an axis-angle rotation vector (direction = rotation axis,
    magnitude = rotation angle in radians) to the equivalent rotation matrix.

    Note: for a UR TCP pose ``[x, y, z, rx, ry, rz]`` the rotation vector is
    ``[rx, ry, rz]`` and the resulting matrix columns are the tool x/y/z axes
    expressed in the arm's base frame.

    Parameters
    ----------
    rotvec : array_like, shape (3,)
        Axis-angle rotation vector.

    Returns
    -------
    numpy.ndarray, shape (3, 3)
        Equivalent rotation matrix.
    """
    return Rotation.from_rotvec(np.asarray(rotvec, dtype=float).reshape(3)).as_matrix()


def rot_matrix_to_rot_vec(R: np.ndarray) -> np.ndarray:
    """3×3 rotation matrix → rotation vector.

    Inverse of ``rot_vec_to_rot_matrix``.  Direction = rotation axis,
    magnitude = rotation angle in radians.

    Note: the result slots directly into the ``[rx, ry, rz]`` part of a UR
    TCP pose.

    Parameters
    ----------
    R : numpy.ndarray, shape (3, 3)

    Returns
    -------
    numpy.ndarray, shape (3,)
        Axis-angle rotation vector.
    """
    return Rotation.from_matrix(np.asarray(R, dtype=float)).as_rotvec()


def compose_rotation_vectors(first, second) -> np.ndarray:
    """Compose two axis-angle rotations using Rodrigues' composition formula.

    Given two rotations expressed as rotation vectors (direction = axis,
    magnitude = angle in radians), returns the single rotation vector that is
    equivalent to applying ``first`` and then ``second``.  In rotation-matrix
    terms the result equals ``R_second @ R_first``.

    The half-angle (quaternion) form of Rodrigues' formula is used, following
    https://math.stackexchange.com/questions/382760.  With ``first`` having
    angle ``a`` about unit axis ``â`` and ``second`` having angle ``b`` about
    unit axis ``b̂``, the combined rotation ``(g, ĉ)`` satisfies::

        cos(g/2)          = cos(a/2) cos(b/2) - sin(a/2) sin(b/2) (â · b̂)
        sin(g/2) ĉ        = sin(a/2) cos(b/2) â + cos(a/2) sin(b/2) b̂
                            + sin(a/2) sin(b/2) (b̂ × â)

    Parameters
    ----------
    first : array_like, shape (3,)
        Rotation vector applied first.
    second : array_like, shape (3,)
        Rotation vector applied second.

    Returns
    -------
    numpy.ndarray, shape (3,)
        Combined axis-angle rotation vector.
    """
    v1 = np.asarray(first, dtype=float).reshape(3)
    v2 = np.asarray(second, dtype=float).reshape(3)

    a = float(np.linalg.norm(v1))
    b = float(np.linalg.norm(v2))

    # Unit axes; a zero-angle rotation has an arbitrary axis, so use a
    # placeholder that is multiplied by sin(0) = 0 and therefore drops out.
    axis1 = v1 / a if a > 0.0 else np.zeros(3)
    axis2 = v2 / b if b > 0.0 else np.zeros(3)

    ca, sa = math.cos(a / 2.0), math.sin(a / 2.0)
    cb, sb = math.cos(b / 2.0), math.sin(b / 2.0)

    # Quaternion scalar and vector parts of the composed rotation.
    w = ca * cb - sa * sb * float(np.dot(axis1, axis2))
    xyz = sa * cb * axis1 + ca * sb * axis2 + sa * sb * np.cross(axis2, axis1)

    sin_half = float(np.linalg.norm(xyz))
    if sin_half < 1e-12:
        # Combined rotation is (near) identity.
        return np.zeros(3)

    angle = 2.0 * math.atan2(sin_half, w)
    return (xyz / sin_half) * angle


def extract_rot_vec_from_tcp(pose) -> np.ndarray:
    """Extract the rotation vector from a UR TCP pose.

    Parameters
    ----------
    pose : array_like, shape (6,)
        UR TCP pose ``[x, y, z, rx, ry, rz]``.

    Returns
    -------
    numpy.ndarray, shape (3,)
        Rotation vector ``[rx, ry, rz]``.

    Raises
    ------
    ValueError
        If ``pose`` does not have exactly 6 elements.
    """
    p = np.asarray(pose, dtype=float).reshape(-1)
    if p.size != 6:
        raise ValueError(f"TCP pose must have 6 elements, got {p.size}")
    return p[3:]


def extract_rot_matrix_from_tcp(pose) -> np.ndarray:
    """Extract the 3×3 rotation matrix from a UR TCP pose.

    Convenience wrapper combining ``extract_rot_vec_from_tcp`` and
    ``rot_vec_to_rot_matrix``.  For the UR tool frame the resulting columns are
    the tool x/y/z axes expressed in the arm's base frame.

    Parameters
    ----------
    pose : array_like, shape (6,)
        UR TCP pose ``[x, y, z, rx, ry, rz]``.

    Returns
    -------
    numpy.ndarray, shape (3, 3)
        Column 0 = tool x, column 1 = tool y, column 2 = tool z,
        all expressed in the arm's base frame.
    """
    return rot_vec_to_rot_matrix(extract_rot_vec_from_tcp(pose))


# ---------------------------------------------------------------------------
# BaseAxis — one of the six cardinal directions in the arm's base frame
# ---------------------------------------------------------------------------

class BaseAxis(Enum):
    """A cardinal direction in the arm's base frame.

    Each member is one of the six axis-aligned unit directions.  The enum
    value is a plain tuple for exact equality comparisons; use the
    ``.vector`` property to get the corresponding numpy array.

    Attributes
    ----------
    POS_X, NEG_X, POS_Y, NEG_Y, POS_Z, NEG_Z
    """
    POS_X = ( 1.,  0.,  0.)
    NEG_X = (-1.,  0.,  0.)
    POS_Y = ( 0.,  1.,  0.)
    NEG_Y = ( 0., -1.,  0.)
    POS_Z = ( 0.,  0.,  1.)
    NEG_Z = ( 0.,  0., -1.)

    @property
    def vector(self) -> np.ndarray:
        """This axis as a numpy unit vector, shape (3,)."""
        return np.array(self.value, dtype=float)

    def __str__(self) -> str:
        _labels = {
            "POS_X": "base_x",  "NEG_X": "-base_x",
            "POS_Y": "base_y",  "NEG_Y": "-base_y",
            "POS_Z": "base_z",  "NEG_Z": "-base_z",
        }
        return _labels[self.name]


# ---------------------------------------------------------------------------
# ToolOrientation — current or desired orientation of the tool
#
# Axis-alignment assumption
# -------------------------
# ToolOrientation only works correctly when the tool frame is axis-aligned
# with the base frame — meaning each tool axis (x, y, z) points in exactly
# one cardinal base-frame direction.  In practice this means the tool is
# always at multiples of 90°, never at a diagonal angle.  The rotation
# matrix in this case is a signed permutation matrix: one ±1 per row and
# column, all other entries ~0.
#
# If this does not hold, closest_base_axis() will silently snap to the
# wrong axis.
# TODO: for non-axis-aligned orientations switch to a continuous
#       parameterisation (Euler angles, quaternions) rather than snapping.
# ---------------------------------------------------------------------------

class ToolOrientation:
    """Describes the current or desired orientation of the tool frame via
    cardinal axes of the arm's base frame.

    Stores which base-frame cardinal direction each tool axis (x, y, z) points
    in.  Requires the tool to be axis-aligned — see note above.

    Two axes (tooltip direction + gripper roll) fully determine the rotation;
    tool x is derived as cross(tool_y, tool_z).  All three are stored
    explicitly for clarity.

    Constructors
    ------------
    from_tcp_pose(pose)            — from a UR TCP pose [x, y, z, rx, ry, rz]
    from_rotation_matrix(R)        — from a 3×3 rotation matrix
    from_labels(tooltip, gripper)  — from human-readable orientation names

    Conversion
    ----------
    to_rotation_matrix()   → 3×3 numpy array
    to_rot_vec()           → rotation vector for a TCP pose

    Attributes
    ----------
    tool_x, tool_y, tool_z : BaseAxis
        Which cardinal base-frame direction each tool axis points. 
        Represents Column 0, 1, 2 of the rotation matrix of the tool frame expressed in terms of the base frame's axes.
    """

    # Maps label → tool z-axis direction (tooltip pointing direction).
    TOOLTIP_DIRECTIONS: dict[str, BaseAxis] = {
        "forward":  BaseAxis.NEG_X,  # tooltip → -base_x
        "backward": BaseAxis.POS_X,  # tooltip → +base_x
        "right":    BaseAxis.POS_Y,  # tooltip → +base_y
        "left":     BaseAxis.NEG_Y,  # tooltip → -base_y
        "up":       BaseAxis.POS_Z,  # tooltip → +base_z
        "down":     BaseAxis.NEG_Z,  # tooltip → -base_z
    }

    # Maps label → tool y-axis direction (gripper roll / finger-gap direction).
    GRIPPER_ORIENTATIONS: dict[str, BaseAxis] = {
        "flat":          BaseAxis.POS_Y,  # finger gap along +base_y (default)
        "flipped_flat":  BaseAxis.NEG_Y,  # finger gap along -base_y (180° from flat)
        "vertical_up":   BaseAxis.POS_Z,  # finger gap pointing up (+base_z)
        "vertical_down": BaseAxis.NEG_Z,  # finger gap pointing down (-base_z)
        "inward":        BaseAxis.NEG_X,  # finger gap toward arm base (-base_x)
        "outward":       BaseAxis.POS_X,  # finger gap away from arm (+base_x)
    }

    # Reverse lookups: BaseAxis tuple value → label.
    _TOOLTIP_BY_VEC: dict[tuple, str] = {ax.value: lbl for lbl, ax in TOOLTIP_DIRECTIONS.items()}
    _GRIPPER_BY_VEC: dict[tuple, str] = {ax.value: lbl for lbl, ax in GRIPPER_ORIENTATIONS.items()}

    def __init__(self, tool_x: BaseAxis, tool_y: BaseAxis, tool_z: BaseAxis) -> None:
        self.tool_x = tool_x
        self.tool_y = tool_y
        self.tool_z = tool_z

    # -- axis snapping -------------------------------------------------------

    @staticmethod
    def closest_base_axis(v) -> BaseAxis:
        """Map a vector to the cardinal base-frame axis it most closely aligns with.

        Only valid when the tool is axis-aligned — see the ToolOrientation note above.

        Parameters
        ----------
        v : array_like, shape (3,)
            Any non-zero vector expressed in the base frame.

        Returns
        -------
        BaseAxis
        """
        v = np.asarray(v, dtype=float).reshape(3)
        # Largest abs-value component gives the axis; its sign gives direction.
        idx = int(np.argmax(np.abs(v)))
        result = [0., 0., 0.]
        result[idx] = float(np.sign(v[idx]))
        return BaseAxis(tuple(result))

    # -- constructors --------------------------------------------------------

    @classmethod
    def from_rotation_matrix(cls, R: np.ndarray) -> "ToolOrientation":
        """Construct from a 3×3 rotation matrix.

        Snaps each column to the nearest cardinal axis (assumes axis-aligned
        tool — see class note).

        Parameters
        ----------
        R : numpy.ndarray, shape (3, 3)
            Column 0 = tool x, column 1 = tool y, column 2 = tool z.
        """
        R = np.asarray(R, dtype=float)
        return cls(
            tool_x=cls.closest_base_axis(R[:, 0]),
            tool_y=cls.closest_base_axis(R[:, 1]),
            tool_z=cls.closest_base_axis(R[:, 2]),
        )

    @classmethod
    def from_tcp_pose(cls, pose) -> "ToolOrientation":
        """Construct ToolOrientation from a UR TCP pose ``[x, y, z, rx, ry, rz]``.

        Parameters
        ----------
        pose : array_like, shape (6,)
        """
        return cls.from_rotation_matrix(extract_rot_matrix_from_tcp(pose))

    @classmethod
    def from_labels(cls, tooltip: str, gripper: str) -> "ToolOrientation":
        """Construct ToolOrientation from human-readable orientation label names.

        Both labels are required: together they fully determine the rotation
        (tool x = cross(tool_y, tool_z) for a right-handed frame).

        Parameters
        ----------
        tooltip : str
            Direction the tooltip (tool z) points.  One of:
            ``"forward"``, ``"backward"``, ``"right"``, ``"left"``,
            ``"up"``, ``"down"``.
        gripper : str
            Gripper roll — direction the finger gap faces (tool y).  One of:
            ``"flat"``, ``"flipped_flat"``, ``"vertical_up"``,
            ``"vertical_down"``, ``"inward"``, ``"outward"``.

        Raises
        ------
        ValueError
            Unknown label, or the two axes are parallel (degenerate rotation).
        """
        if tooltip not in cls.TOOLTIP_DIRECTIONS:
            raise ValueError(
                f"unknown tooltip direction {tooltip!r}; "
                f"choose from {list(cls.TOOLTIP_DIRECTIONS)}"
            )
        if gripper not in cls.GRIPPER_ORIENTATIONS:
            raise ValueError(
                f"unknown gripper orientation {gripper!r}; "
                f"choose from {list(cls.GRIPPER_ORIENTATIONS)}"
            )

        tool_z = cls.TOOLTIP_DIRECTIONS[tooltip]
        tool_y = cls.GRIPPER_ORIENTATIONS[gripper]

        cross = np.cross(tool_y.vector, tool_z.vector)
        if np.linalg.norm(cross) < 0.5:
            raise ValueError(
                f"tooltip={tooltip!r} and gripper={gripper!r} are parallel — "
                "choose axes along different base-frame directions"
            )

        return cls(
            tool_x=cls.closest_base_axis(cross),
            tool_y=tool_y,
            tool_z=tool_z,
        )

    # -- constructors from TCP pose with axis reorientation ------------------

    @classmethod
    def point_axis_to(
        cls,
        pose,
        tool_axis: str,
        direction: BaseAxis,
        locked_axis: str,
    ) -> "ToolOrientation":
        """Build a new orientation by pointing one tool axis at a target
        base-frame direction while keeping a second tool axis locked to its
        current orientation from the pose.

        The third axis is derived to maintain a right-handed frame.

        Parameters
        ----------
        pose : array_like, shape (6,)
            Current TCP pose ``[x, y, z, rx, ry, rz]`` in the arm base frame.
        tool_axis : str
            Which tool axis to reorient: ``'x'``, ``'y'``, or ``'z'``.
        direction : BaseAxis
            Desired base-frame direction for ``tool_axis``.
        locked_axis : str
            Which tool axis to preserve from the current pose: ``'x'``, ``'y'``,
            or ``'z'``.  Must differ from ``tool_axis``.

        Returns
        -------
        ToolOrientation

        Raises
        ------
        ValueError
            If ``tool_axis`` and ``locked_axis`` are the same, if either is not
            ``'x'``, ``'y'``, or ``'z'``, or if ``direction`` is parallel to
            the locked axis (degenerate frame).

        Examples
        --------
        Point tool z toward base -x while keeping current tool y orientation::

            new_orient = ToolOrientation.point_axis_to(
                arm.current_tcp_pose(), 'z', BaseAxis.NEG_X, 'y'
            )
        """
        if tool_axis not in ('x', 'y', 'z'):
            raise ValueError(f"tool_axis must be 'x', 'y', or 'z', got {tool_axis!r}")
        if locked_axis not in ('x', 'y', 'z'):
            raise ValueError(f"locked_axis must be 'x', 'y', or 'z', got {locked_axis!r}")
        if tool_axis == locked_axis:
            raise ValueError(
                f"tool_axis and locked_axis must differ, both are {tool_axis!r}"
            )
    
        cur = cls.from_tcp_pose(pose)
        locked_vec = getattr(cur, f"tool_{locked_axis}").vector

        if abs(np.dot(direction.vector, locked_vec)) > 0.5:
            raise ValueError(
                f"tool_{tool_axis} direction {direction} is parallel to locked "
                f"tool_{locked_axis} — choose a non-parallel direction"
            )

        axes: dict[str, np.ndarray] = {
            tool_axis: direction.vector,
            locked_axis: locked_vec,
        }
        # Derive the third axis using the right-hand rule: x=cross(y,z), y=cross(z,x), z=cross(x,y)
        (derived,) = {'x', 'y', 'z'} - {tool_axis, locked_axis}
        if derived == 'x':
            axes['x'] = np.cross(axes['y'], axes['z'])
        elif derived == 'y':
            axes['y'] = np.cross(axes['z'], axes['x'])
        else:
            axes['z'] = np.cross(axes['x'], axes['y'])

        return cls(
            tool_x=cls.closest_base_axis(axes['x']),
            tool_y=cls.closest_base_axis(axes['y']),
            tool_z=cls.closest_base_axis(axes['z']),
        )

    @classmethod
    def point_tooltip_preserve_gripper_orientation(
        cls, pose, tooltip: str
    ) -> "ToolOrientation":
        """Point the tooltip (tool z) in a new direction while preserving gripper roll.

        Parameters
        ----------
        pose : array_like, shape (6,)
            Current TCP pose ``[x, y, z, rx, ry, rz]`` in the arm base frame.
        tooltip : str
            Desired tooltip direction.  One of: ``'forward'``, ``'backward'``,
            ``'right'``, ``'left'``, ``'up'``, ``'down'``.

        Returns
        -------
        ToolOrientation

        Raises
        ------
        ValueError
            Unknown tooltip label, or new direction is parallel to the current
            gripper-roll axis.

        Examples
        --------
        Point the tooltip straight down, keeping the current gripper roll::

            new_orient = ToolOrientation.point_tooltip_preserve_gripper_orientation(
                arm.current_tcp_pose(), 'down'
            )
        """
        if tooltip not in cls.TOOLTIP_DIRECTIONS:
            raise ValueError(
                f"unknown tooltip direction {tooltip!r}; "
                f"choose from {list(cls.TOOLTIP_DIRECTIONS)}"
            )
        return cls.point_axis_to(pose, 'z', cls.TOOLTIP_DIRECTIONS[tooltip], 'y')

    @classmethod
    def point_gripper_preserve_tooltip(
        cls, pose, gripper: str
    ) -> "ToolOrientation":
        """Set the gripper roll (tool y) while preserving the current tooltip direction.

        Parameters
        ----------
        pose : array_like, shape (6,)
            Current TCP pose ``[x, y, z, rx, ry, rz]`` in the arm base frame.
        gripper : str
            Desired gripper roll.  One of: ``'flat'``, ``'flipped_flat'``,
            ``'vertical_up'``, ``'vertical_down'``, ``'inward'``, ``'outward'``.

        Returns
        -------
        ToolOrientation

        Raises
        ------
        ValueError
            Unknown gripper label, or new direction is parallel to the current
            tooltip axis.

        Examples
        --------
        Set the gripper to vertical-up roll, keeping the current tooltip direction::

            new_orient = ToolOrientation.point_gripper_preserve_tooltip(
                arm.current_tcp_pose(), 'vertical_up'
            )
        """
        if gripper not in cls.GRIPPER_ORIENTATIONS:
            raise ValueError(
                f"unknown gripper orientation {gripper!r}; "
                f"choose from {list(cls.GRIPPER_ORIENTATIONS)}"
            )
        return cls.point_axis_to(pose, 'y', cls.GRIPPER_ORIENTATIONS[gripper], 'z')

    # -- conversion ----------------------------------------------------------

    def to_rotation_matrix(self) -> np.ndarray:
        """Return the 3×3 rotation matrix for this orientation.

        Returns
        -------
        numpy.ndarray, shape (3, 3)
            Column 0 = tool x, column 1 = tool y, column 2 = tool z.
        """
        return np.column_stack([
            self.tool_x.vector,
            self.tool_y.vector,
            self.tool_z.vector,
        ])

    def to_rot_vec(self) -> list[float]:
        """Return the rotation vector ``[rx, ry, rz]`` for this orientation.

        Slots directly into the last three elements of a UR TCP pose.

        Returns
        -------
        list of float, shape (3,)
        """
        return rot_matrix_to_rot_vec(self.to_rotation_matrix()).tolist()

    # -- human-readable labels -----------------------------------------------

    @property
    def tooltip(self) -> str | None:
        """Label for the tooltip direction (tool z), or None if not in the map."""
        return self._TOOLTIP_BY_VEC.get(self.tool_z.value)

    @property
    def gripper(self) -> str | None:
        """Label for the gripper roll orientation (tool y), or None if not in the map."""
        return self._GRIPPER_BY_VEC.get(self.tool_y.value)

    def describe(self) -> dict:
        """Human-readable summary of this orientation.

        Returns
        -------
        dict
            Keys: ``tooltip``, ``gripper``, ``tool_x``, ``tool_y``, ``tool_z``.
            The tool axis values are ``BaseAxis`` string labels (e.g. ``"base_x"``).
        """
        return {
            "tooltip": self.tooltip,
            "gripper": self.gripper,
            "tool_x":  str(self.tool_x),
            "tool_y":  str(self.tool_y),
            "tool_z":  str(self.tool_z),
        }

    def __repr__(self) -> str:
        return f"ToolOrientation(tooltip={self.tooltip!r}, gripper={self.gripper!r})"
