"""Paper state model.

A `Paper` tracks the named reference points of a sheet (its corners, and
any landmarks created along the way) in board coordinates, together with the list
of `Fold` operations that have been applied.  Every manipulation --
folding, rotating, sliding the sheet -- updates the landmark positions
analytically, so the rest of the system always knows where each reference point
currently is.

This is deliberately a model of *key reference points* rather than a full
multi-layer mesh: a paper dart only needs a handful of landmarks (corners, the
nose, fold endpoints) to drive the arms, and exact reflections keep them precise.

Terminology
-----------
There is a single operation, `Paper.fold()`, which both creases the paper and
moves the selected region across the crease.  A "crease" is simply the line a
fold was made along; it is recorded as a `Fold`.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Callable, Iterable

import numpy as np

from . import geometry as geo
from .geometry import FoldLine


@dataclass
class Fold:
    """A record of a fold that has been applied to the paper.

    Parameters
    ----------
    start, end : numpy.ndarray, shape (2,)
        Board-coordinate endpoints of the crease (a finite segment of the fold
        line, clipped to the sheet).
    style : {'valley', 'mountain'}
        Whether the crease was folded as a valley (toward the viewer) or a
        mountain (away).  Purely descriptive metadata.
    label : str
        Human-readable name for the fold (e.g. ``'left nose'``).
    flaps : list of numpy.ndarray, optional
        The folded-over region(s) of this fold, each an ``(M, 2)`` polygon in
        board coordinates.  Because these regions have been reflected across the
        crease they now expose the paper's reverse ("back") face, which the UI
        renders distinctly.
    """

    start: np.ndarray
    end: np.ndarray
    style: str = "valley"
    label: str = ""
    flaps: list[np.ndarray] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.start = np.asarray(self.start, dtype=float).reshape(2)
        self.end = np.asarray(self.end, dtype=float).reshape(2)
        self.flaps = [np.asarray(f, dtype=float).reshape(-1, 2) for f in self.flaps]

    def as_line(self) -> FoldLine:
        """Return the infinite `FoldLine` of this crease.

        Returns
        -------
        origami.geometry.FoldLine
        """
        return FoldLine.through_points(self.start, self.end)


@dataclass
class Paper:
    """Mutable paper state in board coordinates (metres).

    Parameters
    ----------
    landmarks : dict, optional
        Maps landmark names to ``(x, y)`` board positions.  Corners of a fresh
        sheet are named ``bottom_left``/``bottom_right``/``top_right``/``top_left``.
    folds : list of Fold, optional
        Folds already applied (usually left empty and built up via `fold()`).
    history : list of str, optional
        Human-readable log of operations, appended to automatically.
    name : str, optional
        A label for the sheet.

    See Also
    --------
    Paper.square, Paper.rectangle : Convenience constructors.
    """

    landmarks: dict[str, np.ndarray] = field(default_factory=dict)
    folds: list[Fold] = field(default_factory=list)
    history: list[str] = field(default_factory=list)
    name: str = "paper"

    def __post_init__(self) -> None:
        self.landmarks = {k: np.asarray(v, dtype=float).reshape(2) for k, v in self.landmarks.items()}

    # ------------------------------------------------------------------ #
    # Construction
    # ------------------------------------------------------------------ #
    @classmethod
    def square(cls, side_length: float, origin=(0.0, 0.0), name: str = "paper") -> "Paper":
        """Create a square sheet.

        Parameters
        ----------
        side_length : float
            Length of each side (metres).
        origin : array_like, shape (2,), optional
            Board position of the bottom-left corner.  Default ``(0, 0)``.
        name : str, optional
            Sheet label.

        Returns
        -------
        Paper
        """
        return cls.rectangle(side_length, side_length, origin=origin, name=name)

    @classmethod
    def rectangle(cls, width: float, height: float, origin=(0.0, 0.0), name: str = "paper") -> "Paper":
        """Create a rectangular sheet with axis-aligned corners.

        Parameters
        ----------
        width : float
            Extent along ``+x`` (metres).
        height : float
            Extent along ``+y`` (metres).
        origin : array_like, shape (2,), optional
            Board position of the bottom-left corner.  Default ``(0, 0)``.
        name : str, optional
            Sheet label.

        Returns
        -------
        Paper
        """
        ox, oy = np.asarray(origin, dtype=float).reshape(2)
        corners = {
            "bottom_left": (ox, oy),
            "bottom_right": (ox + width, oy),
            "top_right": (ox + width, oy + height),
            "top_left": (ox, oy + height),
        }
        return cls(landmarks=corners, name=name)

    # ------------------------------------------------------------------ #
    # Queries
    # ------------------------------------------------------------------ #
    def landmark(self, name: str) -> np.ndarray:
        """Return a copy of a landmark's current position.

        Parameters
        ----------
        name : str
            Landmark name (e.g. ``'top_left'``).

        Returns
        -------
        numpy.ndarray, shape (2,)
        """
        return self.landmarks[name].copy()

    def landmark_array(self) -> np.ndarray:
        """All landmark positions stacked into an array.

        Returns
        -------
        numpy.ndarray, shape (N, 2)
            Positions in landmark-insertion order.
        """
        return np.array(list(self.landmarks.values()))

    def centroid(self) -> np.ndarray:
        """Area centroid of the current landmark polygon.

        Returns
        -------
        numpy.ndarray, shape (2,)
        """
        return geo.polygon_centroid(self.landmark_array())

    def bounding_box(self) -> tuple[np.ndarray, np.ndarray]:
        """Axis-aligned bounding box of the landmarks.

        Returns
        -------
        tuple of numpy.ndarray
            ``(lower_corner, upper_corner)`` as ``(2,)`` arrays.
        """
        pts = self.landmark_array()
        return pts.min(axis=0), pts.max(axis=0)

    def copy(self) -> "Paper":
        """Return a deep, independent copy of this paper state.

        Returns
        -------
        Paper
        """
        return copy.deepcopy(self)

    # ------------------------------------------------------------------ #
    # Rigid moves (whole sheet)
    # ------------------------------------------------------------------ #
    def translate(self, offset) -> "Paper":
        """Slide the whole sheet by a board-plane offset (in place).

        Parameters
        ----------
        offset : array_like, shape (2,)
            Displacement ``(dx, dy)`` in metres.

        Returns
        -------
        Paper
            ``self``, to allow chaining.
        """
        off = np.asarray(offset, dtype=float).reshape(2)
        for name in self.landmarks:
            self.landmarks[name] = self.landmarks[name] + off
        for fold in self.folds:
            fold.start, fold.end = fold.start + off, fold.end + off
            fold.flaps = [flap + off for flap in fold.flaps]
        self.history.append(f"translate by {off.tolist()}")
        return self

    def rotate(self, angle: float, pivot=None) -> "Paper":
        """Rotate the whole sheet about a pivot (in place).

        Parameters
        ----------
        angle : float
            Rotation angle in radians (counter-clockwise).
        pivot : array_like, shape (2,), optional
            Centre of rotation.  Defaults to the sheet centroid.

        Returns
        -------
        Paper
            ``self``, to allow chaining.
        """
        pivot = self.centroid() if pivot is None else np.asarray(pivot, dtype=float).reshape(2)
        for name in self.landmarks:
            self.landmarks[name] = geo.rotate_points_about(self.landmarks[name], angle, pivot)[0]
        for fold in self.folds:
            fold.start = geo.rotate_points_about(fold.start, angle, pivot)[0]
            fold.end = geo.rotate_points_about(fold.end, angle, pivot)[0]
            fold.flaps = [geo.rotate_points_about(flap, angle, pivot) for flap in fold.flaps]
        self.history.append(f"rotate {angle:.4f} rad about {np.round(pivot, 4).tolist()}")
        return self

    # ------------------------------------------------------------------ #
    # Folding
    # ------------------------------------------------------------------ #
    def fold(self, fold_line: FoldLine,
             moving_region: Iterable[str] | Callable[[np.ndarray], bool] | None = None,
             style: str = "valley", label: str = "") -> "Paper":
        """Fold the sheet along ``fold_line``, reflecting the moving region (in place).

        Parameters
        ----------
        fold_line : origami.geometry.FoldLine
            The crease to fold along.
        moving_region : iterable of str, callable, or None, optional
            Selects which landmarks travel with the moving flap:

            * ``None`` (default) -- every landmark strictly on the left
              (``+``) side of ``fold_line``.
            * iterable of names -- exactly those landmarks.
            * callable -- a predicate ``f(xy) -> bool`` evaluated per landmark.
        style : {'valley', 'mountain'}, optional
            Crease style recorded with the fold.  Default ``'valley'``.
        label : str, optional
            Human-readable name for the fold.

        Returns
        -------
        Paper
            ``self``, to allow chaining.

        Notes
        -----
        Selected landmarks are mirrored exactly across ``fold_line``.  Where the
        crease passes through the interior of the sheet -- i.e. an edge of the
        landmark polygon has one endpoint that moves and one that stays -- a new
        landmark is inserted at that boundary intersection so the page keeps its
        true folded outline instead of being skewed by the moved corner alone.
        A `Fold` record (the crease, clipped to those boundary intersections
        when available) is appended to `folds`.
        """
        names = self._select_landmarks(fold_line, moving_region)
        self._reflect_prior_folds(fold_line, names)
        crease_points, flaps = self._apply_fold(fold_line, names)
        start, end = self._crease_endpoints(fold_line, crease_points)
        self.folds.append(Fold(start=start, end=end, style=style, label=label, flaps=flaps))
        self.history.append(f"fold {style} '{label}' moving {names}")
        return self

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #
    def _select_landmarks(self, fold_line: FoldLine, moving_region) -> list[str]:
        """Resolve a ``moving_region`` selector to a list of landmark names.

        Parameters
        ----------
        fold_line : origami.geometry.FoldLine
        moving_region : iterable of str, callable, or None
            See `fold()`.

        Returns
        -------
        list of str
        """
        if moving_region is None:
            return [n for n, p in self.landmarks.items() if fold_line.side_of(p) > 0]
        if callable(moving_region):
            return [n for n, p in self.landmarks.items() if moving_region(p)]
        return list(moving_region)

    def _apply_fold(self, fold_line: FoldLine, names) -> tuple[list[np.ndarray], list[np.ndarray]]:
        """Reflect the moving landmarks and insert crease boundary points.

        The landmark dict is treated as a closed polygon in insertion order.
        Each polygon edge with exactly one moving endpoint is crossed by the
        crease; a new landmark is inserted at that crossing (which lies on the
        fold line and is therefore unaffected by the reflection).  This keeps
        the page outline correct instead of letting a lone moved corner skew
        the whole sheet.

        Parameters
        ----------
        fold_line : origami.geometry.FoldLine
        names : iterable of str
            The landmarks that move with the flap.

        Returns
        -------
        crease_points : list of numpy.ndarray
            The crease/boundary intersection points that were inserted, in
            polygon order.
        flaps : list of numpy.ndarray
            The folded-over region(s), each an ``(M, 2)`` polygon (the reflected
            moving landmarks together with the crease points that bound them).
        """
        moving = set(names)
        items = list(self.landmarks.items())
        rebuilt: dict[str, np.ndarray] = {}
        crease_points: list[np.ndarray] = []
        # Ordered polygon entries tagged by kind, used to carve out the flap(s).
        ordered: list[tuple[np.ndarray, str]] = []

        def _add_crease(point: np.ndarray, *neighbours: np.ndarray) -> None:
            # Skip intersections that coincide with an adjacent vertex, to avoid
            # zero-length edges / duplicate landmarks piling up where creases meet.
            for neighbour in neighbours:
                if float(np.linalg.norm(point - neighbour)) <= 1e-9:
                    return
            name = self._unique_landmark_name(rebuilt)
            rebuilt[name] = point
            crease_points.append(point)
            ordered.append((point, "crease"))

        count = len(items)
        for index in range(count):
            name, position = items[index]
            is_moving = name in moving
            placed = fold_line.reflect(position) if is_moving else position.copy()
            rebuilt[name] = placed
            if is_moving:
                kind = "move"
            elif abs(fold_line.signed_offset(placed)) <= 1e-9:
                # A pre-existing landmark sitting on the crease is a shared corner
                # of the flap, so treat it as a flap boundary vertex.
                kind = "crease"
            else:
                kind = "static"
            ordered.append((placed, kind))
            if count < 2:
                continue
            next_name, next_position = items[(index + 1) % count]
            if is_moving == (next_name in moving):
                continue
            crossing = fold_line.segment_intersection(position, next_position)
            if crossing is not None:
                next_placed = (fold_line.reflect(next_position)
                               if next_name in moving else next_position)
                _add_crease(crossing, placed, next_placed)

        self.landmarks = rebuilt
        return crease_points, self._extract_flaps(ordered)

    def _reflect_prior_folds(self, fold_line: FoldLine, names: Iterable[str]) -> None:
        """Move previously recorded fold geometry with a new fold operation.

        Existing crease segments and back-face flap polygons are metadata tied to
        the current paper pose, so when a new fold moves one half-plane, those
        stored points must move with it.
        """
        selected = [self.landmarks[name] for name in names if name in self.landmarks]
        if not selected:
            return
        offsets = [fold_line.signed_offset(point) for point in selected]
        mean_offset = float(np.mean(offsets))
        if abs(mean_offset) <= 1e-9:
            return
        moving_sign = 1 if mean_offset > 0.0 else -1

        def _move_if_selected(point: np.ndarray) -> np.ndarray:
            return (fold_line.reflect(point)
                    if moving_sign * fold_line.signed_offset(point) > 1e-9
                    else np.asarray(point, dtype=float).copy())

        for fold in self.folds:
            fold.start = _move_if_selected(fold.start)
            fold.end = _move_if_selected(fold.end)
            fold.flaps = [np.array([_move_if_selected(point) for point in flap]) for flap in fold.flaps]

    @staticmethod
    def _extract_flaps(ordered: list[tuple[np.ndarray, str]]) -> list[np.ndarray]:
        """Carve the folded-over flap polygon(s) out of a tagged polygon ring.

        Parameters
        ----------
        ordered : list of (numpy.ndarray, str)
            The rebuilt polygon vertices in order, each tagged ``'move'``,
            ``'static'`` or ``'crease'``.

        Returns
        -------
        list of numpy.ndarray
            One ``(M, 2)`` polygon per folded-over flap.  A flap is a run of
            moving/crease vertices bounded by static paper; when the whole sheet
            moves the single flap is the entire polygon.
        """
        if not any(kind == "move" for _, kind in ordered):
            return []
        if not any(kind == "static" for _, kind in ordered):
            return [np.array([p for p, _ in ordered])]
        # Rotate so the ring starts at a static vertex, giving clean run breaks.
        start = next(i for i, (_, kind) in enumerate(ordered) if kind == "static")
        ring = ordered[start:] + ordered[:start]
        flaps: list[np.ndarray] = []
        run: list[np.ndarray] = []
        has_move = False
        for point, kind in ring:
            if kind == "static":
                if has_move and len(run) >= 3:
                    flaps.append(np.array(run))
                run = []
                has_move = False
                continue
            run.append(point)
            has_move = has_move or kind == "move"
        if has_move and len(run) >= 3:
            flaps.append(np.array(run))
        return flaps

    def _unique_landmark_name(self, existing: dict[str, np.ndarray]) -> str:
        """Generate a fresh ``crease<fold>_<n>`` name not already in use.

        Parameters
        ----------
        existing : dict
            Landmark names already taken in the polygon being rebuilt.

        Returns
        -------
        str
        """
        base = f"crease{len(self.folds)}"
        index = 0
        name = f"{base}_{index}"
        while name in existing or name in self.landmarks:
            index += 1
            name = f"{base}_{index}"
        return name

    def _crease_endpoints(self, fold_line: FoldLine,
                          crease_points: list[np.ndarray] | None = None) -> tuple[np.ndarray, np.ndarray]:
        """Endpoints of the drawable crease segment.

        When the fold actually crosses the sheet boundary the crossing points
        (``crease_points``) span the page exactly, so they are used directly.
        Otherwise the fold line is clipped to the current bounding box.

        Parameters
        ----------
        fold_line : origami.geometry.FoldLine
        crease_points : list of numpy.ndarray or None, optional
            Boundary intersections found while folding.

        Returns
        -------
        tuple of numpy.ndarray
            Two endpoints spanning the sheet along the fold line.
        """
        if crease_points and len(crease_points) >= 2:
            pts = np.array(crease_points)
            coords = pts @ fold_line.direction
            return pts[int(coords.argmin())], pts[int(coords.argmax())]
        lower, upper = self.bounding_box()
        centre = (lower + upper) / 2.0
        span = float(np.linalg.norm(upper - lower)) or 1.0
        return fold_line.clipped_segment(centre, span)

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        marks = ", ".join(f"{k}={np.round(v, 4).tolist()}" for k, v in self.landmarks.items())
        return f"Paper('{self.name}', {marks}, folds={len(self.folds)})"
