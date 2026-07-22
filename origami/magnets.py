"""Magnet state model.

The rig uses 3D-printed magnet holders on a magnetic board as extra "joints" that
pin or weigh down paper while the arms work. Each physical magnet is tracked
here with its board pose and type-specific geometry, so the high-level actions
can reason about what is currently holding the paper and exactly where (and at
what height) an arm must grip to move it.

The model uses two different 2D positions:

* ``anchor_xy`` -- the board-contact point that defines where the magnet itself is
    placed (what actually holds down our paper).
* ``grip_xy`` -- the point (handle position) the robot actually grasps. For some magnets this is
    the same as ``anchor_xy``; for others it is offset from the anchor because the handle is offset 
    from the anchor/contact point.

Two holder types are modelled:

* ``BlockMagnet`` -- a weight block with a holder; gripped directly above its
    anchor.
* ``LBracketMagnet`` -- an L-shaped bracket whose board anchor sits on the
    board (and can act as a fold hinge) while its handle stands off to one side at
    a raised height, which is where the gripper actually grabs it.

Positions use the world coordinate frame (x, y on the board surface, z = height
above board). When a magnet is placed on the board its anchor sits at z = 0.
Tray positions may have z ≠ 0 because trays can be at a different height.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .geometry import FoldLine


@dataclass
class Magnet:
    """Base class for a tracked magnet holder.

    Parameters
    ----------
    identifier : str
        Unique name for this physical magnet.
    anchor_xy : array_like, shape (2,), optional
        World (x, y) of the board-contact point that defines where the magnet itself is
    placed (what actually holds down our paper)
        Default ``(0, 0)``.  The z of the anchor is implied by context: z = 0
        when placed on the board, ``tray_position[2]`` when stowed.
    orientation : float, optional
        Yaw of the magnet about the board normal, in radians.  Default ``0``.
    placed : bool, optional
        ``True`` when the magnet is currently on the board holding paper;
        ``False`` while it waits in its tray.  Default ``False``.
    tray_position : tuple (x, y, z) or None, optional
        World position of the magnet's home / "graveyard" slot.  All three
        coordinates are stored because the tray may sit outside the board
        board area and at a different height from the board surface.

    Attributes
    ----------
    kind : str
        Short type tag (``'block'``, ``'lbracket'``, ...), set by subclasses.
    """

    identifier: str
    anchor_xy: np.ndarray = field(default_factory=lambda: np.zeros(2))
    orientation: float = 0.0
    placed: bool = False
    tray_position: tuple[float, float, float] | None = None

    kind: str = "generic"

    def __post_init__(self) -> None:
            self.tray_position = np.asarray(self.tray_position, dtype=float).reshape(3)
            self.anchor_xy = np.asarray(self.tray_position[:2], dtype=float)

    # -- geometry the arms need ----------------------------------------- #
    def get_anchor_xy(self) -> np.ndarray:
        """numpy.ndarray, shape (2,): World (x, y) of the anchor point."""
        return self.anchor_xy

    @property
    def grip_xy(self) -> np.ndarray:
        """numpy.ndarray, shape (2,): The robot grasp point (the magnet handle).

        For some magnets this is the same as ``anchor_xy``; for others it is
        offset from the anchor because the handle is offset from the anchor/contact point.
        """
        return self.get_anchor_xy()

    def get_grip_xy(self) -> np.ndarray:
        """numpy.ndarray, shape (2,): World (x, y) of the robot grasp point."""
        return self.grip_xy

    # -- geometry the arms need ----------------------------------------- #
    @property
    def grip_height(self) -> float:
        """float: Height above the anchor's surface (metres) at which to grip.

        This is added to the anchor's z to get the world grip z:
        ``world_grip_z = anchor_z + grip_height``.
        Subclasses override with the true grip / holder height.
        """
        return 0.0

    # -- state transitions ---------------------------------------------- #
    def place_at(self, x: float, y: float, orientation: float | None = None) -> "Magnet":
        """Update the magnet's board state so its derived grip point is correct.

        This updates only ``anchor_xy`` and, if requested, the orientation,
        then marks the magnet as placed. Callers can then query ``grip_xy`` (e.g. via chaining on this method) from the updated state in order 
        to know where to actually move the arm to (in order to move the magnet), as that is the property the arm actually works with 
        to interact with the magnet

        Parameters
        ----------
        x, y : float
            Board position of the magnet anchor (contact point with the paper) (z = 0 implied).
        orientation : float or None, optional
            New yaw in radians; unchanged if ``None``.

        Returns
        -------
        Magnet
            ``self``, for method chaining.
        """
        self.anchor_xy = np.array([float(x), float(y)])
        if orientation is not None:
            self.orientation = float(orientation)
        self.placed = True
        return self

    def stow(self) -> "Magnet":
        """Update the magnet's tray state so its derived grip point is correct.

        This clears the ``placed`` flag and updates ``anchor_xy`` to the tray
        position. Callers can then query ``grip_xy`` (e.g. via chaining on this method) from the updated state in order to 
        know where to actually move the arm to (in order to move the magnet), as that is the property the arm actually works with 
        to interact with the magnet.

        Returns
        -------
        Magnet
            ``self``, for method chaining.
        """
        self.placed = False
        if self.tray_position is not None:
            self.anchor_xy = np.asarray(self.tray_position[:2], dtype=float)
        return self

    def describe(self) -> str:
        """One-line human-readable summary of the magnet's state."""
        state = "placed" if self.placed else "stowed"
        anchor = f"anchor=({self.get_anchor_xy()[0]:.3f}, {self.get_anchor_xy()[1]:.3f})"
        grip = self.get_grip_xy()
        if not np.allclose(grip, self.get_anchor_xy()):
            grip_text = f" grip=({grip[0]:.3f}, {grip[1]:.3f}, z={self.grip_height:.3f})"
        else:
            grip_text = f" grip_z={self.grip_height:.3f}"
        return f"{self.kind}:{self.identifier} {state} {anchor} yaw={self.orientation:.2f}{grip_text}"


@dataclass
class BlockMagnet(Magnet):
    """A weight block (with a gripper holder) that pins down a point or area.

    The gripper grasps the holder directly above the block's anchor at
    ``handle_height`` above the board surface.

    Parameters
    ----------
    handle_height : float, optional
        World z of the grip point above the board surface (metres).
        Default ``0.02``.
    """

    handle_height: float = 0.02
    kind: str = "block"

    @property
    def grip_height(self) -> float:
        return self.handle_height


@dataclass
class LBracketMagnet(Magnet):
    """An L-bracket magnet, its grip point (its handle) is offset from the anchor (not directly above it).

    * **Anchor / contact point** -- ``anchor_xy`` sits on the board surface and
      holds the paper in place.
    * **Grip point / handle** -- the gripper picks up a separate handle point
      offset from ``anchor_xy`` in the direction of ``-orientation``.

    Parameters
    ----------
    handle_offset : float, optional
        Distance from ``anchor_xy`` to the grip point in the direction of
        ``-orientation`` (metres). Default ``0.03``.
    handle_height : float, optional
        Height of the grip point above the board surface (metres).
        Default ``0.03``.
    """

    handle_offset: float = 0.03
    handle_height: float = 0.03
    kind: str = "lbracket"

    @property
    def grip_xy(self) -> np.ndarray:
        """numpy.ndarray, shape (2,): World (x, y) of the grip point.

        The grip point is offset from the anchor in the direction of
        ``-orientation``.
        """
        direction = np.array([np.cos(self.orientation), np.sin(self.orientation)])
        return self.get_anchor_xy() - direction * self.handle_offset

    @property
    def grip_height(self) -> float:
        return self.handle_height


class MagnetRegistry:
    """A collection of magnets with lookup and availability queries.

    Parameters
    ----------
    magnets : iterable of Magnet, optional
        Magnets to register up front.
    """

    def __init__(self, magnets: list[Magnet] | None = None) -> None:
        self._magnets: dict[str, Magnet] = {}
        for magnet in magnets or []:
            self.add(magnet)

    def add(self, magnet: Magnet) -> Magnet:
        """Register a magnet.

        Raises
        ------
        ValueError
            If a magnet with the same ``identifier`` is already registered.
        """
        if magnet.identifier in self._magnets:
            raise ValueError(f"duplicate magnet identifier: {magnet.identifier}")
        self._magnets[magnet.identifier] = magnet
        return magnet

    def get(self, identifier: str) -> Magnet:
        """Look up a magnet by identifier."""
        return self._magnets[identifier]

    def placed(self) -> list[Magnet]:
        """All magnets currently on the board."""
        return [m for m in self._magnets.values() if m.placed]

    def available(self, kind: str | None = None) -> list[Magnet]:
        """Magnets still in their tray, optionally filtered by ``kind``."""
        return [m for m in self._magnets.values()
                if not m.placed and (kind is None or m.kind == kind)]

    def of_kind(self, kind: str) -> list[Magnet]:
        """All registered magnets of a given type tag."""
        return [m for m in self._magnets.values() if m.kind == kind]

    def __contains__(self, identifier: str) -> bool:
        return identifier in self._magnets

    def __iter__(self):
        return iter(self._magnets.values())

    def __len__(self) -> int:
        return len(self._magnets)

    def __repr__(self) -> str:  # pragma: no cover
        body = "\n  ".join(m.describe() for m in self)
        return f"MagnetRegistry[\n  {body}\n]"
