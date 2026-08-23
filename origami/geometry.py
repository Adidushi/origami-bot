"""Geometric primitives for the origami framework.

This module is the mathematical foundation of the package.  Rather than
hand-rolling rotation/transform algebra, it builds on top of established
libraries:

* `spatialmath` -- rigid-body transforms (`SE2`, `SE3`, `SO3`) and the exponential /
  logarithm maps used to convert to and from the Universal-Robots rotation-vector
  pose convention.
* `scipy` -- robust point-cloud registration via orthogonal Procrustes
  (`scipy.spatial.transform.Rotation.align_vectors()`).
* `numpy` -- array plumbing only.

Coordinate conventions
----------------------
Board coordinates
    A right-handed frame attached to the magnetic board.  ``x`` and ``y`` lie in
    the board surface (metres); ``z`` (also called *height above board*) points
    straight up out of the surface.  Paper, magnets and tool targets are all
    expressed here.
UR TCP pose
    The Universal-Robots / RTDE convention ``[x, y, z, rx, ry, rz]`` where the
    translation is in metres and ``(rx, ry, rz)`` is a *rotation vector*
    (axis-angle whose magnitude is the rotation angle in radians).

See Also
--------
origami.calibration : Maps board coordinates onto a specific robot base frame.
"""
from __future__ import annotations

from typing import Sequence

import numpy as np
from scipy.spatial.transform import Rotation
from spatialmath import SE2, SE3, SO3

ArrayLike = Sequence[float] | np.ndarray


def _as_xy(point: ArrayLike) -> np.ndarray:
    """Coerce ``point`` to a length-2 ``float`` board-plane vector.

    Parameters
    ----------
    point : array_like
        Any 2-element sequence ``(x, y)``.

    Returns
    -------
    numpy.ndarray
        Shape ``(2,)`` float array.

    Raises
    ------
    ValueError
        If ``point`` does not contain exactly two elements.
    """
    v = np.asarray(point, dtype=float).reshape(-1)
    if v.size != 2:
        raise ValueError(f"expected a 2D (x, y) point, got {v.size} elements")
    return v


# --------------------------------------------------------------------------- #
# UR rotation-vector pose  <->  spatialmath SE3
# --------------------------------------------------------------------------- #
def pose_to_se3(pose: ArrayLike) -> SE3:
    """Convert a UR TCP pose to a `spatialmath.SE3`.

    Parameters
    ----------
    pose : array_like
        UR pose ``[x, y, z, rx, ry, rz]`` (metres and a rotation vector).

    Returns
    -------
    spatialmath.SE3
        The equivalent rigid-body transform.

    Notes
    -----
    The rotation is recovered with the exponential map
    `spatialmath.SO3.EulerVec()`, the inverse of `spatialmath.SO3.eulervec()`.

    Examples
    --------
    >>> T = pose_to_se3([0.1, -0.4, 0.2, 0.0, np.pi, 0.0])
    >>> np.allclose(T.t, [0.1, -0.4, 0.2])
    True
    """
    p = np.asarray(pose, dtype=float).reshape(-1)
    if p.size != 6:
        raise ValueError(f"a UR pose has 6 elements, got {p.size}")
    return SE3.Rt(SO3.EulerVec(p[3:]), p[:3])


def se3_to_pose(transform: SE3) -> list[float]:
    """Convert a `spatialmath.SE3` to a UR TCP pose.

    Parameters
    ----------
    transform : spatialmath.SE3
        A rigid-body transform.

    Returns
    -------
    list of float
        UR pose ``[x, y, z, rx, ry, rz]`` ready to hand to ``moveL``.

    Notes
    -----
    The rotation vector is the logarithm map `spatialmath.SO3.eulervec()`.
    """
    t = np.asarray(transform.t, dtype=float).reshape(3)
    rotvec = np.asarray(SO3(transform.R, check=False).eulervec(), dtype=float).reshape(3)
    return [float(t[0]), float(t[1]), float(t[2]),
            float(rotvec[0]), float(rotvec[1]), float(rotvec[2])]


# --------------------------------------------------------------------------- #
# Rigid registration (board -> base calibration fit)
# --------------------------------------------------------------------------- #
def fit_rigid_transform(source_points: ArrayLike, target_points: ArrayLike) -> SE3:
    """Best-fit rigid transform mapping ``source_points`` onto ``target_points``.

    Solves the orthogonal Procrustes problem: find the rotation ``R`` and
    translation ``t`` minimising ``sum ||(R @ s_i + t) - d_i||^2`` over all
    corresponding point pairs.

    Parameters
    ----------
    source_points : array_like, shape (N, 3)
        Points in the source frame (e.g. board coordinates).
    target_points : array_like, shape (N, 3)
        The corresponding points in the target frame (e.g. robot base frame).

    Returns
    -------
    spatialmath.SE3
        Transform ``T`` such that ``T * source_points[i] ≈ target_points[i]``.

    Raises
    ------
    ValueError
        If the inputs are not matching ``(N, 3)`` arrays with ``N >= 2``.

    Notes
    -----
    The rotation is obtained from
    `scipy.spatial.transform.Rotation.align_vectors()` applied to the
    mean-centred point sets; the translation then aligns the centroids.  At least
    three non-collinear correspondences are required for a unique solution.
    """
    src = np.atleast_2d(np.asarray(source_points, dtype=float))
    dst = np.atleast_2d(np.asarray(target_points, dtype=float))
    if src.shape != dst.shape or src.shape[1] != 3 or src.shape[0] < 2:
        raise ValueError("expected two matching (N, 3) arrays with N >= 2")
    src_centroid = src.mean(axis=0)
    dst_centroid = dst.mean(axis=0)
    rotation, _ = Rotation.align_vectors(dst - dst_centroid, src - src_centroid)
    R = rotation.as_matrix()
    t = dst_centroid - R @ src_centroid
    return SE3.Rt(SO3(R, check=False), t)


# --------------------------------------------------------------------------- #
# Polygon helpers
# --------------------------------------------------------------------------- #
def polygon_centroid(points: ArrayLike) -> np.ndarray:
    """Area centroid of a simple polygon in the board plane.

    Parameters
    ----------
    points : array_like, shape (N, 2)
        Polygon vertices in order (clockwise or counter-clockwise).

    Returns
    -------
    numpy.ndarray, shape (2,)
        The centroid.  For a degenerate (zero-area) polygon the vertex mean is
        returned instead.

    Notes
    -----
    Uses the standard shoelace centroid formula.
    """
    pts = np.atleast_2d(np.asarray(points, dtype=float))
    x, y = pts[:, 0], pts[:, 1]
    x_next, y_next = np.roll(x, -1), np.roll(y, -1)
    cross = x * y_next - x_next * y
    area = cross.sum() / 2.0
    if abs(area) < 1e-12:
        return pts.mean(axis=0)
    cx = ((x + x_next) * cross).sum() / (6.0 * area)
    cy = ((y + y_next) * cross).sum() / (6.0 * area)
    return np.array([cx, cy])


def rotate_points_about(points: ArrayLike, angle: float, pivot: ArrayLike) -> np.ndarray:
    """Rotate board-plane points about a pivot.

    Parameters
    ----------
    points : array_like, shape (N, 2) or (2,)
        Point(s) to rotate.
    angle : float
        Rotation angle in radians (counter-clockwise).
    pivot : array_like, shape (2,)
        Centre of rotation.

    Returns
    -------
    numpy.ndarray, shape (N, 2)
        The rotated points.

    Notes
    -----
    Implemented as the `spatialmath.SE2` conjugation
    ``T = translate(pivot) · rotate(angle) · translate(-pivot)``.
    """
    pivot = _as_xy(pivot)
    pts = np.atleast_2d(np.asarray(points, dtype=float))
    T = SE2(pivot[0], pivot[1], 0.0) * SE2(0.0, 0.0, angle) * SE2(-pivot[0], -pivot[1], 0.0)
    out = (T * pts.T)  # spatialmath maps a (2, N) block of column vectors
    return np.asarray(out).T


# --------------------------------------------------------------------------- #
# Fold line
# --------------------------------------------------------------------------- #
class FoldLine:
    """An oriented line in the board plane, used as a fold / crease axis.

    A fold reflects part of the sheet across this line.  Internally the line is
    stored as a `spatialmath.SE2` *line frame* whose origin lies on the
    line and whose x-axis runs along the line direction; reflecting a point is
    then "flip the sign of its y-coordinate in that frame".  The line's left side
    (the ``+y`` half-plane of the frame) has positive
    `signed_offset()`.

    Parameters
    ----------
    point : array_like, shape (2,)
        Any point lying on the line.
    direction : array_like, shape (2,)
        A non-zero direction vector along the line (need not be unit length).

    Attributes
    ----------
    frame : spatialmath.SE2
        The line frame (origin on the line, x-axis along ``direction``).

    See Also
    --------
    origami.paper.Paper.fold : Applies a fold about a ``FoldLine``.
    """

    def __init__(self, point: ArrayLike, direction: ArrayLike) -> None:
        p = _as_xy(point)
        d = _as_xy(direction)
        if np.linalg.norm(d) < 1e-12:
            raise ValueError("fold-line direction must be non-zero")
        self.frame = SE2(float(p[0]), float(p[1]), float(np.arctan2(d[1], d[0])))

    # -- constructors ---------------------------------------------------- #
    @classmethod
    def through_points(cls, first: ArrayLike, second: ArrayLike) -> "FoldLine":
        """Construct the line passing through two points.

        Parameters
        ----------
        first, second : array_like, shape (2,)
            Two distinct points on the line.

        Returns
        -------
        FoldLine
        """
        first = _as_xy(first)
        second = _as_xy(second)
        return cls(first, second - first)

    @classmethod
    def at_angle(cls, point: ArrayLike, angle: float) -> "FoldLine":
        """Construct a line through ``point`` at a given heading.

        Parameters
        ----------
        point : array_like, shape (2,)
            A point on the line.
        angle : float
            Heading of the line in radians, measured from the board ``+x`` axis.

        Returns
        -------
        FoldLine
        """
        return cls(point, [np.cos(angle), np.sin(angle)])

    # -- geometry -------------------------------------------------------- #
    @property
    def point(self) -> np.ndarray:
        """numpy.ndarray, shape (2,): The stored point on the line."""
        return np.asarray(self.frame.t, dtype=float).reshape(2)

    @property
    def direction(self) -> np.ndarray:
        """numpy.ndarray, shape (2,): Unit vector along the line."""
        return np.asarray(self.frame.R[:, 0], dtype=float).reshape(2)

    @property
    def normal(self) -> np.ndarray:
        """numpy.ndarray, shape (2,): Unit left-hand normal (points to the +side)."""
        return np.asarray(self.frame.R[:, 1], dtype=float).reshape(2)

    @property
    def angle(self) -> float:
        """float: Heading of the line in radians from the board ``+x`` axis."""
        return float(self.frame.theta())

    def signed_offset(self, point: ArrayLike) -> float:
        """Signed perpendicular distance from ``point`` to the line.

        Parameters
        ----------
        point : array_like, shape (2,)

        Returns
        -------
        float
            Positive if ``point`` lies on the left of the line (looking along its
            direction), negative on the right, zero if exactly on it.
        """
        local = np.asarray(self.frame.inv() * _as_xy(point)).reshape(2)
        return float(local[1])

    def side_of(self, point: ArrayLike, tolerance: float = 1e-9) -> int:
        """Which side of the line a point lies on.

        Parameters
        ----------
        point : array_like, shape (2,)
        tolerance : float, optional
            Half-width of the "on the line" band.  Default ``1e-9``.

        Returns
        -------
        int
            ``+1`` left of the line, ``-1`` right, ``0`` on it.
        """
        offset = self.signed_offset(point)
        if abs(offset) <= tolerance:
            return 0
        return 1 if offset > 0 else -1

    def reflect(self, point: ArrayLike) -> np.ndarray:
        """Mirror a single point across the line.

        Parameters
        ----------
        point : array_like, shape (2,)

        Returns
        -------
        numpy.ndarray, shape (2,)
            The reflected point.
        """
        local = np.asarray(self.frame.inv() * _as_xy(point)).reshape(2)
        local[1] = -local[1]
        return np.asarray(self.frame * local).reshape(2)

    def reflect_many(self, points: ArrayLike) -> np.ndarray:
        """Mirror several points across the line.

        Parameters
        ----------
        points : array_like, shape (N, 2)

        Returns
        -------
        numpy.ndarray, shape (N, 2)
            The reflected points.
        """
        pts = np.atleast_2d(np.asarray(points, dtype=float))
        return np.array([self.reflect(p) for p in pts])

    def project(self, point: ArrayLike) -> np.ndarray:
        """Foot of the perpendicular from ``point`` onto the line.

        Parameters
        ----------
        point : array_like, shape (2,)

        Returns
        -------
        numpy.ndarray, shape (2,)
            The closest point on the line.
        """
        local = np.asarray(self.frame.inv() * _as_xy(point)).reshape(2)
        local[1] = 0.0
        return np.asarray(self.frame * local).reshape(2)

    def segment_intersection(self, start: ArrayLike, end: ArrayLike,
                             tolerance: float = 1e-9) -> np.ndarray | None:
        """Point where the segment ``start``--``end`` crosses the line, if any.

        Parameters
        ----------
        start, end : array_like, shape (2,)
            Endpoints of the segment.
        tolerance : float, optional
            Half-width of the "on the line" band used to treat an endpoint as
            lying exactly on the line.  Default ``1e-9``.

        Returns
        -------
        numpy.ndarray, shape (2,) or None
            The crossing point, or ``None`` when the segment lies entirely on
            one side of the line (no proper crossing).

        Notes
        -----
        If an endpoint lies on the line that endpoint is returned.  A segment
        that lies wholly within the line band is treated as non-crossing
        (``None``) -- there is no single intersection point.
        """
        a = _as_xy(start)
        b = _as_xy(end)
        oa = self.signed_offset(a)
        ob = self.signed_offset(b)
        a_on = abs(oa) <= tolerance
        b_on = abs(ob) <= tolerance
        if a_on and b_on:
            return None
        if a_on:
            return a
        if b_on:
            return b
        if (oa > 0.0) == (ob > 0.0):
            return None
        t = oa / (oa - ob)
        return a + t * (b - a)

    def clipped_segment(self, centre: ArrayLike, span: float) -> tuple[np.ndarray, np.ndarray]:
        """A finite segment of the line, centred near ``centre``.

        Useful for turning the infinite fold line into two drawable / reachable
        endpoints (e.g. for creasing along the line).

        Parameters
        ----------
        centre : array_like, shape (2,)
            A reference point; its projection onto the line is the segment midpoint.
        span : float
            Total length of the returned segment (metres).

        Returns
        -------
        tuple of numpy.ndarray
            The two endpoints ``(start, end)``.
        """
        mid = self.project(centre)
        half = self.direction * (span / 2.0)
        return mid - half, mid + half

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        p = self.point
        return f"FoldLine(point=[{p[0]:.4f}, {p[1]:.4f}], angle={self.angle:.4f})"
