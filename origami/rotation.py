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
        """Construct from a UR TCP pose ``[x, y, z, rx, ry, rz]``.

        Parameters
        ----------
        pose : array_like, shape (6,)
        """
        return cls.from_rotation_matrix(extract_rot_matrix_from_tcp(pose))

    @classmethod
    def from_labels(cls, tooltip: str, gripper: str) -> "ToolOrientation":
        """Construct from human-readable orientation label names.

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

    # -- constructors from TCP pose with one axis fixed ----------------------

    @classmethod
    def from_tcp_with_axis_fixed(
        cls,
        pose,
        tool_axis: str,
        direction: BaseAxis,
    ) -> "ToolOrientation":
        """Build from a TCP pose, fixing one tool axis to a desired base-frame
        direction while preserving the remaining axes from the current pose.

        Preserved-axis priority (matches UR tool frame semantics):

        - Fixing tool_z: keep tool_y (gripper roll), fall back to tool_x.
        - Fixing tool_y: keep tool_z (tooltip), fall back to tool_x.
        - Fixing tool_x: keep tool_y, fall back to tool_z.

        The fallback is needed only when the "preferred" axis happens to be
        parallel to the newly fixed axis (they cannot both remain in the frame).

        Parameters
        ----------
        pose : array_like, shape (6,)
            Current TCP pose ``[x, y, z, rx, ry, rz]`` in the arm base frame.
        tool_axis : str
            Which tool axis to fix: ``'x'``, ``'y'``, or ``'z'``.
        direction : BaseAxis
            Desired base-frame direction for the fixed axis.

        Returns
        -------
        ToolOrientation

        Raises
        ------
        ValueError
            If ``tool_axis`` is not ``'x'``, ``'y'``, or ``'z'``.

        Examples
        --------
        Point the tooltip straight down while keeping the current gripper roll::

            new_orient = ToolOrientation.from_tcp_with_axis_fixed(
                arm.current_tcp_pose(), 'z', BaseAxis.NEG_Z
            )
        """
        if tool_axis not in ('x', 'y', 'z'):
            raise ValueError(f"tool_axis must be 'x', 'y', or 'z', got {tool_axis!r}")

        cur = cls.from_tcp_pose(pose)

        if tool_axis == 'z':
            new_z = direction
            # Prefer to keep current tool_y (gripper roll); fall back to tool_x.
            if abs(np.dot(cur.tool_y.vector, new_z.vector)) < 0.5:
                new_y = cur.tool_y
            else:
                new_y = cls.closest_base_axis(np.cross(new_z.vector, cur.tool_x.vector))
            new_x = cls.closest_base_axis(np.cross(new_y.vector, new_z.vector))
            return cls(tool_x=new_x, tool_y=new_y, tool_z=new_z)

        if tool_axis == 'y':
            new_y = direction
            # Prefer to keep current tool_z (tooltip); fall back to tool_x.
            if abs(np.dot(cur.tool_z.vector, new_y.vector)) < 0.5:
                new_z = cur.tool_z
            else:
                new_z = cls.closest_base_axis(np.cross(cur.tool_x.vector, new_y.vector))
            new_x = cls.closest_base_axis(np.cross(new_y.vector, new_z.vector))
            return cls(tool_x=new_x, tool_y=new_y, tool_z=new_z)

        # tool_axis == 'x'
        new_x = direction
        # Prefer to keep current tool_y; fall back to tool_z.
        if abs(np.dot(cur.tool_y.vector, new_x.vector)) < 0.5:
            new_y = cur.tool_y
        else:
            new_y = cls.closest_base_axis(np.cross(cur.tool_z.vector, new_x.vector))
        new_z = cls.closest_base_axis(np.cross(new_x.vector, new_y.vector))
        return cls(tool_x=new_x, tool_y=new_y, tool_z=new_z)

    @classmethod
    def from_tcp_with_tooltip(cls, pose, tooltip: str) -> "ToolOrientation":
        """Fix the tooltip direction (tool z) while preserving gripper roll from pose.

        Wrapper around ``from_tcp_with_axis_fixed`` using the ``TOOLTIP_DIRECTIONS``
        label map.

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
            Unknown tooltip label.

        Examples
        --------
        Reorient the tooltip straight down, keeping the current gripper roll::

            new_orient = ToolOrientation.from_tcp_with_tooltip(
                arm.current_tcp_pose(), 'down'
            )
        """
        if tooltip not in cls.TOOLTIP_DIRECTIONS:
            raise ValueError(
                f"unknown tooltip direction {tooltip!r}; "
                f"choose from {list(cls.TOOLTIP_DIRECTIONS)}"
            )
        return cls.from_tcp_with_axis_fixed(pose, 'z', cls.TOOLTIP_DIRECTIONS[tooltip])

    @classmethod
    def from_tcp_with_gripper(cls, pose, gripper: str) -> "ToolOrientation":
        """Fix the gripper roll (tool y) while preserving tooltip direction from pose.

        Wrapper around ``from_tcp_with_axis_fixed`` using the ``GRIPPER_ORIENTATIONS``
        label map.

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
            Unknown gripper label.

        Examples
        --------
        Set the gripper to vertical-up roll, keeping the current tooltip direction::

            new_orient = ToolOrientation.from_tcp_with_gripper(
                arm.current_tcp_pose(), 'vertical_up'
            )
        """
        if gripper not in cls.GRIPPER_ORIENTATIONS:
            raise ValueError(
                f"unknown gripper orientation {gripper!r}; "
                f"choose from {list(cls.GRIPPER_ORIENTATIONS)}"
            )
        return cls.from_tcp_with_axis_fixed(pose, 'y', cls.GRIPPER_ORIENTATIONS[gripper])

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
