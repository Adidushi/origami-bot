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

from enum import Enum, auto

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


# ---------------------------------------------------------------------------
# TooltipDirection / GripperOrientation -- base-frame orientation labels
# ---------------------------------------------------------------------------

class TooltipDirection(Enum):
    """Direction the tooltip (tool z) points, directions
    are based on the base frame's axes."""

    FORWARD  = BaseAxis.NEG_X   # toward the wall
    BACKWARD = BaseAxis.POS_X   # away from the wall - towards us
    RIGHT    = BaseAxis.POS_Y
    LEFT     = BaseAxis.NEG_Y
    UP       = BaseAxis.POS_Z
    DOWN     = BaseAxis.NEG_Z

    @property
    def axis(self) -> BaseAxis:
        """Returns BaseAxis this direction resolves to."""
        return self.value

    @property
    def vector(self) -> np.ndarray:
        """Returns the unit vector this direction resolves to."""
        return self.value.vector

    def __str__(self) -> str:
        return f"TooltipDirection.{self.name}"


# A chosen absolute up, from which a relative up can be defined inside the circle
# defined by on the tool x-y plane by rotating about tool z.
_UP_REFERENCE = BaseAxis.POS_Z

# Stands in when the tooltip direction is parallel to the up reference, therefore making it impossible 
# to define a relative up on the tool x-y plane via a +Z absolute reference so we need a different point of reference.
_UP_REFERENCE_FALLBACK = BaseAxis.NEG_X


class GripperOrientation(Enum):
    """Gripper roll: where the finger gap (tool y) faces, relative to the tooltip.

    The gripper has one DOF, a rotation about tool z, which defines a circle in the
    tool x-y plane. Since gripper orientation is determined relative to tool z by
    the position of tool y on that circle, splitting the circle into 4 quadrants/labels gives
    us convenient directions to set the gripper orientation with.

    Note
    ----
    These labels cannot name fixed base directions, because tool y is constrained by
    tool z — it has to stay perpendicular to it and only has proper meaning/orientation when
    defined relative to the tooltip / tool z:
    - A base direction is almost never on the tool x-y plane / circle, and therefore
    cannot be used to create a proper tool frame as not being on the circle means its not
    perpendicular to tool z.
    - Beyond that, since the tool y (gripper orientation) is defined relative to tool z (constrained by it),
    a specific gripper orientation (FLAT, VERTICAL_UP, etc.) changes completely depending on the tooltip direction. 
    For example, a FLAT gripper orientation with a FORWARD tooltip direction is not the same as a FLAT gripper orientation with a RIGHT tooltip direction. 
    TLDR: The gripper orientation is always defined relative to the tooltip direction, and therefore cannot be defined in absolute terms.

    """

    FLAT          = auto()
    FLIPPED_FLAT  = auto()
    VERTICAL_UP   = auto()
    VERTICAL_DOWN = auto()

    def resolve_relative_orientation(self, tool_z) -> np.ndarray:
        """Resolves the given gripper orientation instance to the correct tool_y / unit vector s.t. relative to tool_z (tooltip direction)
        it is in the correct gripper orientation
        
        Parameters
        ----------
        tool_z : array_like, shape (3,)
            Unit tooltip direction, in base-frame coordinates.

        Returns
        -------
        numpy.ndarray, shape (3,)
            Resolved tool_y unit vector corresponding to this gripper orientation relative to tool_z, in base-frame coordinates.
        """
        tool_z = np.asarray(tool_z, dtype=float).reshape(3)
        normalized_tool_z = tool_z / float(np.linalg.norm(tool_z))

        # Crossing tool z with the relative up is what gives us a reference sideways
        # vector on the circle / tool x-y plane.

        # If the reference/absolute up is parallel with tool z we don't have a good reference to define a relative up on the tool x-y plane / circle, 
        # so we need to use a different reference.
        reference_up = _UP_REFERENCE.vector
        if abs(float(np.dot(reference_up, normalized_tool_z))) > 1.0 - 1e-6:
            reference_up = _UP_REFERENCE_FALLBACK.vector

        # We want a relative up direction on the tool x-y plane / circle, achieved by
        # taking our absolute/reference up direction and casting it onto that plane / circle.

        # We do this by removing the parts of the reference parallel to tool z, giving
        # a vector pointing conceptually in the same direction but orthogonal to
        # tool z, and since every vector perpendicular to tool z lies on the tool x-y
        # plane / circle, that is exactly what we wanted.
        up = reference_up - float(np.dot(reference_up, normalized_tool_z)) * normalized_tool_z
        up /= float(np.linalg.norm(up))

        # The cross product of tool z and the relative up gives us a sideways vector on the
        # circle / tool x-y plane.
        side = np.cross(normalized_tool_z, up)

        # dict/mapping of gripper orientation to the corresponding relative tool y direction on the tool x-y plane / circle.
        # so we can resolve a given instance of class enum to its correct vector
        return {
            GripperOrientation.FLAT:          side,
            GripperOrientation.FLIPPED_FLAT: -side,
            GripperOrientation.VERTICAL_UP:    up,
            GripperOrientation.VERTICAL_DOWN: -up,
        }[self]

    def __str__(self) -> str:
        return f"GripperOrientation.{self.name}"


# ---------------------------------------------------------------------------
# ArmOrientation -- a tool orientation composed from the labels above
# ---------------------------------------------------------------------------

class ArmOrientation:
    """A desired tool orientation, described with respect to the arm's base frame - held internally as 
    a 3x3 matrix, where the matrix columns [C_1, C_2, C_3] = [tool_x, tool_y, tool_z], i.e. each of the matrice's columns 
    describe the end position of the corresponding tool axis, with each column/desired tool position/axis expressed in terms 
    of/relative to the arm's base axes.   

    Instantiation
    -------------
    Created through the ``from_*`` builders. Typically ``from_directions``, from a
    tooltip direction (where tool +z points) and a gripper orientation (where tool
    +y points). It can also be built from an arbitrary rotation matrix or rotation
    vector, or the current TCP pose — see Typical / Alternative usage.

    Typical usage
    -------------
    Created from a tooltip direction and gripper orientation, giving a standard
    pose that is axis-aligned to the base frame. From there, relative rotations
    off that pose are applied with ``tilt_tooltip`` (see Method chaining).

    Alternative usage
    -----------------
    Created from a custom matrix or rotation vector (e.g. ``from_tcp_pose``). This
    can be used to simply tilt an existing orientation, or to describe a
    standalone rotation that is then composed with another orientation to build a
    more complex target — ``ArmOrientation`` supports composition via ``@`` and
    ``compose``.

    Method chaining
    ---------------
    The transform methods (``tooltip_direction``, ``gripper_orientation``,
    ``tilt_tooltip``, ``compose``) mutate the orientation in place and return it,
    so calls chain into a single expression.

    Examples
    --------
    Tooltip forward, gripper flat, aimed 30 deg toward the right:

        ArmOrientation.from_directions(TooltipDirection.FORWARD, GripperOrientation.FLAT).tilt_tooltip(
            TooltipDirection.RIGHT, degrees=30)

    Re-aim the current tool orientation downward, keeping its gripper roll:

        ArmOrientation.from_tcp_pose(arm.current_tcp_pose()).tooltip_direction(
            TooltipDirection.DOWN)

    Compose a standalone rotation onto a base orientation::

        base @ ArmOrientation.from_rotvec([0, 0, math.pi / 4])
    """


    @staticmethod
    def _matrix_from_directions(tool_z: np.ndarray, tool_y: np.ndarray) -> np.ndarray:
        """Build the tool frame matrix from its z and y axis directions.

        - ``tool_z`` (tooltip direction) and ``tool_y`` (gripper roll) must be
          orthogonal unit vectors; ``tool_x`` is derived via ``cross(tool_y, tool_z)``.
        - Raises ``ValueError`` if the two are not orthogonal (their dot product is
          non-zero), since then they do not define a valid frame.

        Parameters
        ----------
        tool_z, tool_y : numpy.ndarray, shape (3,)
            Orthogonal unit directions of tool z and tool y in the base frame.

        Returns
        -------
        numpy.ndarray, shape (3, 3)
            The frame matrix, columns = tool x, y, z.
        """
        # tool z and tool y are treated as orthogonal (and so form a valid frame)
        # when the magnitude of their dot product is at or below this.
        _ORTHOGONAL_TOLERANCE = 1e-6
        if abs(float(np.dot(tool_z, tool_y))) > _ORTHOGONAL_TOLERANCE:
            raise ValueError("tool z (tooltip direction) and tool y (gripper "
                             "orientation) must be orthogonal to form a valid frame")
        return np.column_stack([np.cross(tool_y, tool_z), tool_y, tool_z])

    # -- builders (constructors) ---------------------------------------------
    @classmethod
    def from_matrix(cls, matrix: np.ndarray) -> "ArmOrientation":
        """Create from an existing 3x3 rotation matrix (columns = tool x, y, z)."""
        obj = cls.__new__(cls)
        obj._matrix = np.asarray(matrix, dtype=float).reshape(3, 3)
        return obj

    @classmethod
    def from_directions(cls, tooltip_direction: TooltipDirection,
                        gripper_orientation: GripperOrientation) -> "ArmOrientation":
        """Create an axis-aligned orientation from a tooltip direction and gripper orientation.

        Parameters
        ----------
        tooltip_direction : TooltipDirection
            Where the tooltip (tool +z) points.
        gripper_orientation : GripperOrientation
            Where the finger gap (tool +y) points, resolved against the tooltip.
        """
        tool_z = tooltip_direction.vector
        _matrix = cls._matrix_from_directions(
            tool_z=tool_z, tool_y=gripper_orientation.resolve_relative_orientation(tool_z))
        return cls.from_matrix(_matrix)
    
    @classmethod
    def from_rotvec(cls, rotvec) -> "ArmOrientation":
        """Create from a rotation vector ``[rx, ry, rz]``."""
        return cls.from_matrix(rot_vec_to_rot_matrix(rotvec))

    @classmethod
    def from_tcp_pose(cls, pose) -> "ArmOrientation":
        """Create from the tool orientation in a UR TCP pose ``[x, y, z, rx, ry, rz]``."""
        return cls.from_matrix(extract_rot_matrix_from_tcp(pose))

    # -- tool axes -----------------------------------------------------------

    @property
    def tool_x(self) -> np.ndarray:
        """The tool x axis (flat face of the gripper), in base-frame coordinates."""
        return self._matrix[:, 0]

    @property
    def tool_y(self) -> np.ndarray:
        """The tool y axis (finger-gap direction), in base-frame coordinates."""
        return self._matrix[:, 1]

    @property
    def tool_z(self) -> np.ndarray:
        """The tool z axis (tooltip direction), in base-frame coordinates."""
        return self._matrix[:, 2]

    # -- transforms (mutate in place, return self for chaining) --------------

    def tooltip_direction(self, direction: TooltipDirection) -> "ArmOrientation":
        """Point the tooltip in ``direction``, keeping the current gripper roll.

        - The whole frame swings rigidly so that tool z is on ``direction``, so tool y and tool x keep
          the same relationship to the tooltip they had before to maintain the gripper roll.
        - Works from any orientation, tilted ones included.
        - Math Explanation: https://imgur.com/a/5iOEfoZ

        Parameters
        ----------
        direction : TooltipDirection
            The base-frame direction to point the tooltip in.

        Returns
        -------
        ArmOrientation
            ``self``, for method chaining.
        """
        # Both are unit vectors, so a dot b = cos of the angle between source and target locations of tool z.
        cos_angle = float(np.dot(self.tool_z, direction.vector))
        # If cosine of angle between them is 1 then the angle is 0, so no rotation is needed.
        if cos_angle >= 1.0 - 1e-12:
            return self
        # If cosine of angle is -1 then the angle between them is 180 degrees. Since source and target tool z vectors are opposite/parallel that means
        # they live on the same line, so any axis perpendicular to tool z (e.g. tool y) is perpendicular to both source (tool z) and target (destination) so no cross product is needed.
        # so we can just rotate about tool y by 180 degrees to get from source to target.
        if cos_angle <= -1.0 + 1e-12:
            self._matrix = rot_vec_to_rot_matrix(self.tool_y * math.pi) @ self._matrix
            return self

        # The axis of rotation we want must be perpendicular/orthogonal to both source and targte tool z vectors, so we can get it by taking the cross product of the two.
        rotation_axis = np.cross(self.tool_z, direction.vector)
        unit_rotation_axis = rotation_axis / float(np.linalg.norm(rotation_axis))
        angle = np.clip(math.acos(cos_angle), 0, math.pi)
        rotation_vector = unit_rotation_axis * angle
        self._matrix = rot_vec_to_rot_matrix(rotation_vector) @ self._matrix
        return self

    def gripper_orientation(self, orientation: GripperOrientation) -> "ArmOrientation":
        """Roll the gripper to ``orientation``, keeping the current tooltip direction.

        Parameters
        ----------
        orientation : GripperOrientation
            The roll to set the gripper to.

        Returns
        -------
        ArmOrientation
            ``self``, for method chaining.
        """
        tool_z = self.tool_z
        self._matrix = self._matrix_from_directions(
            tool_z=tool_z, tool_y=orientation.resolve_relative_orientation(tool_z))
        return self

    def tilt_tooltip(self, direction: TooltipDirection, degrees: float) -> "ArmOrientation":
        """Tilt the tooltip ``degrees`` in the direction of ``direction``. Keeps the gripper orientation the same relative to the tooltip.

        - This lets you aim the tooltip relative to where it currently is towards a cardinal direction.
            -  e.g. from its current orientation, tilt 30 degrees to the right.
        - ``direction`` must differ from the direction the tooltip already points
          along, since then there is nothing to tilt toward, otherwise raises
          ``ValueError``.
        - Math Explanation: https://imgur.com/a/5iOEfoZ though in this case we don't need to compute 
        the angle between tool_z and the "target" direction since we only tilt by a given specified ``degrees`` 
        rather than actually reorienting the tooltip to point exactly along the target direction.

        Parameters
        ----------
        direction : TooltipDirection
            The base-frame direction to tilt the tooltip toward.
        degrees : float
            How far to tilt, in degrees.

        Returns
        -------
        ArmOrientation
            ``self``, for method chaining.
        """
        # Note: since rotation axis is perpendicular to tool z, we don't get any rotation along tool z, so the gripper roll is preserved relative to the tooltip.
        rotation_axis = np.cross(self.tool_z, direction.vector)
        unit_rotation_axis = rotation_axis / float(np.linalg.norm(rotation_axis))
        # If the dot product of the tooltip direction and the target direction is 1, then the angle between them is 0 degrees, meaning they are already aligned and there is nothing to tilt toward.
        if np.dot(self.tool_z, direction.vector) > 1.0 - 1e-6:
            raise ValueError(
                f"cannot tilt toward {direction}: tooltip already points along it")
        tilt = rot_vec_to_rot_matrix(unit_rotation_axis * math.radians(degrees))
        self._matrix = tilt @ self._matrix
        return self

    def rotate_gripper(self, degrees: float) -> "ArmOrientation":
        """Rotate the gripper by ``degrees``.

        - This is done by composing the current orientation with a rotation about
          the tooltip direction (tool z). Spinning about that axis holds the tooltip
          pointing exactly where it already does and carries tool y (gripper "slit" orientation)
          and tool x around with it, so the tool's aim is untouched and only the
          gripper's roll changes.

        Parameters
        ----------
        degrees : float
            How far to rotate the gripper, in degrees.

        Returns
        -------
        ArmOrientation
            ``self``, for method chaining.
        """
        # We rotate about the tooltip direction (tool z) to spin the gripper around it, which preserves the tooltip direction and only changes the gripper roll.
        rotate_gripper = rot_vec_to_rot_matrix(self.tool_z * math.radians(degrees))
        # This rotate gripper is relative to the current tool frame, which means we need to apply it on the current tool frame via left composition
        # in order to achieve the desired gripper rotation. 
        self._matrix = rotate_gripper @ self._matrix
        return self

    def compose(self, other: "ArmOrientation") -> "ArmOrientation":
        """Compose ``other`` onto this orientation in place (``self.matrix @ other.matrix``).

        ``other`` is applied first, then this orientation. Equivalent to
        ``self @ other`` but more convenient for method chaining and also applies
        this method in place rather than returning a new instance.

        Returns
        -------
        ArmOrientation
            ``self``, for method chaining.
        """
        self._matrix = self._matrix @ other._matrix
        return self

    # -- conversion ----------------------------------------------------------

    def to_rotation_matrix(self) -> np.ndarray:
        """The 3x3 rotation matrix (columns = tool x, y, z in the base frame)."""
        return self._matrix.copy()

    def to_rotation_vector(self) -> list[float]:
        """The rotation vector ``[rx, ry, rz]`` for a TCP pose."""
        return rot_matrix_to_rot_vec(self._matrix).tolist()

    # -- dunders -------------------------------------------------------------

    def __matmul__(self, other: "ArmOrientation") -> "ArmOrientation":
        """Compose two orientations, ``self @ other`` (other applied first, then self)."""
        if not isinstance(other, ArmOrientation):
            return NotImplemented
        return ArmOrientation.from_matrix(self._matrix @ other._matrix)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, ArmOrientation):
            return NotImplemented
        return np.allclose(self._matrix, other._matrix)

    def __repr__(self) -> str:
        rx, ry, rz = self.to_rotation_vector()
        matrix = np.array2string(self._matrix, precision=3, suppress_small=True)
        return f"ArmOrientation(rotvec=[{rx:.3f}, {ry:.3f}, {rz:.3f}], matrix=\n{matrix})"
