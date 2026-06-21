"""Magnet state model.

The rig uses 3D-printed magnet holders on a magnetic board as extra "joints" that
pin or weigh down paper while the arms work.  Each physical magnet is tracked
here with its board pose and type-specific geometry, so the high-level actions
can reason about what is currently holding the paper and exactly where (and at
what height) an arm must grip to move it.

Two holder types are modelled:

* `BlockMagnet` -- a weight block with a holder; gripped directly above
  its centre.
* `LBracketMagnet` -- an L-shaped bracket whose magnetic foot sits on the
  board (and can act as a fold hinge) while its handle stands off to one side at
  a raised height, which is where the gripper actually grabs it.

Positions use the world coordinate frame (x, y on the board surface, z = height
above board).  When a magnet is placed on the board its foot sits at z = 0.
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
    center : array_like, shape (2,), optional
        World (x, y) of the magnet's contact footprint.  For an `LBracketMagnet`
        this is the magnetic foot; for a `BlockMagnet` it is the block centre.
        Default ``(0, 0)``.  The z of the foot is implied by context: z = 0
        when placed on the board, ``tray_position[2]`` when stowed.
    orientation : float, optional
        Yaw of the magnet about the board normal, in radians.  Default ``0``.
    placed : bool, optional
        ``True`` when the magnet is currently on the board holding paper;
        ``False`` while it waits in its tray.  Default ``False``.
    tray_position : tuple (x, y, z) or None, optional
        World position of the magnet's home / "graveyard" slot.  All three
        coordinates are stored because the tray may sit outside the board
        footprint and at a different height from the board surface.

    Attributes
    ----------
    kind : str
        Short type tag (``'block'``, ``'lbracket'``, ...), set by subclasses.
    """

    identifier: str
    center: np.ndarray = field(default_factory=lambda: np.zeros(2))
    orientation: float = 0.0
    placed: bool = False
    tray_position: tuple[float, float, float] | None = None

    kind: str = "generic"

    def __post_init__(self) -> None:
        self.tray_position = np.asarray(self.tray_position, dtype=float).reshape(3)
        self.center = np.asarray(self.tray_position[:2], dtype=float)

    # -- geometry the arms need ----------------------------------------- #
    @property
    def handle_xy(self) -> np.ndarray:
        """numpy.ndarray, shape (2,): World (x, y) of the grip handle.

        For magnets gripped directly above their footprint this equals `center`.
        Subclasses with a spatially separated handle (e.g. `LBracketMagnet`)
        override this to return the actual handle location.
        """
        return self.center

    @property
    def grip_height(self) -> float:
        """float: Height above the foot's surface (metres) at which to grip.

        This is added to the foot's z to get the world grip z:
        ``world_grip_z = foot_z + grip_height``.
        Subclasses override with the true handle / holder height.
        """
        return 0.0

    @property
    def grip_xy(self) -> np.ndarray:
        """numpy.ndarray, shape (2,): Alias for `handle_xy`."""
        return self.handle_xy

    # -- state transitions ---------------------------------------------- #
    def place_at(self, x: float, y: float, orientation: float | None = None) -> "Magnet":
        """Mark the magnet as placed on the board at (x, y).

        Parameters
        ----------
        x, y : float
            Board position of the magnet foot (z = 0 implied).
        orientation : float or None, optional
            New yaw in radians; unchanged if ``None``.

        Returns
        -------
        Magnet
            ``self``, for chaining.
        """
        self.center = np.array([float(x), float(y)])
        if orientation is not None:
            self.orientation = float(orientation)
        self.placed = True
        return self

    def stow(self) -> "Magnet":
        """Mark the magnet as returned to its tray.

        If `tray_position` is set, the foot ``(x, y)`` is moved there.

        Returns
        -------
        Magnet
            ``self``, for chaining.
        """
        self.placed = False
        if self.tray_position is not None:
            self.center = np.asarray(self.tray_position[:2], dtype=float)
        return self

    def describe(self) -> str:
        """One-line human-readable summary of the magnet's state."""
        state = "placed" if self.placed else "stowed"
        foot = f"foot=({self.center[0]:.3f}, {self.center[1]:.3f})"
        handle = self.handle_xy
        if not np.allclose(handle, self.center):
            grip = f" handle=({handle[0]:.3f}, {handle[1]:.3f}, z={self.grip_height:.3f})"
        else:
            grip = f" grip_z={self.grip_height:.3f}"
        return f"{self.kind}:{self.identifier} {state} {foot} yaw={self.orientation:.2f}{grip}"


@dataclass
class BlockMagnet(Magnet):
    """A weight block (with a gripper holder) that pins down a point or area.

    The gripper grasps the holder directly above the block's centre at
    ``holder_height`` above the board surface.

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
    """An L-bracket whose foot pins a board edge and whose handle is gripped.

    * **Magnetic foot** -- centred at `center` on the board surface; holds
      the paper and provides a clean pivot / hinge line.
    * **Handle** -- stands off from the foot along `orientation` by
      `handle_offset` metres and rises to `handle_height` above the board;
      this is what the gripper grasps.

    Parameters
    ----------
    handle_offset : float, optional
        Distance from the foot to the handle along `orientation` (metres).
        Default ``0.03``.
    handle_height : float, optional
        Height of the handle above the board surface (metres).  Default ``0.03``.
    """

    handle_offset: float = 0.03
    handle_height: float = 0.03
    kind: str = "lbracket"

    @property
    def handle_xy(self) -> np.ndarray:
        """numpy.ndarray, shape (2,): World (x, y) of the handle (not the foot).

        The handle is offset from the foot in the direction of `orientation`.
        """
        direction = np.array([np.cos(self.orientation), np.sin(self.orientation)])
        return self.center - direction * self.handle_offset

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
