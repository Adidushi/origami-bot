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

Heights matter: a magnet's grip point is generally **not** on the board surface,
so every magnet exposes an explicit `grip_height`.
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
        Board position of the magnet's contact footprint on the board.  For an
        `LBracketMagnet` this is the magnetic foot; for a
        `BlockMagnet` it is the block centre.  Default ``(0, 0)``.
    orientation : float, optional
        Yaw of the magnet about the board normal, in radians.  Default ``0``.
    placed : bool, optional
        ``True`` when the magnet is currently on the board holding paper;
        ``False`` while it waits in its tray.  Default ``False``.
    tray_position : tuple of float or None, optional
        Board ``(x, y)`` of this magnet's home / "graveyard" slot, if it has one.

    Attributes
    ----------
    kind : str
        Short type tag (``'block'``, ``'lbracket'``, ...), set by subclasses.
    """

    identifier: str
    center: np.ndarray = field(default_factory=lambda: np.zeros(2))
    orientation: float = 0.0
    placed: bool = False
    tray_position: tuple[float, float] | None = None

    kind: str = "generic"

    def __post_init__(self) -> None:
        self.center = np.asarray(self.center, dtype=float).reshape(2)

    # -- geometry the arms need ----------------------------------------- #
    @property
    def grip_xy(self) -> np.ndarray:
        """numpy.ndarray, shape (2,): Board point an arm should grip above.

        For the base magnet this is simply `center`; subclasses with an
        offset handle override it.
        """
        return self.center.copy()

    @property
    def grip_height(self) -> float:
        """float: Height above the board (metres) at which to grip this magnet.

        Subclasses override with the true handle / holder height.  The base
        default is ``0`` (board surface).
        """
        return 0.0

    # -- state transitions ---------------------------------------------- #
    def place_at(self, x: float, y: float, orientation: float | None = None) -> "Magnet":
        """Mark the magnet as placed on the board at a pose.

        Parameters
        ----------
        x, y : float
            Board position of the magnet centre (metres).
        orientation : float or None, optional
            New yaw in radians; unchanged if ``None``.

        Returns
        -------
        Magnet
            ``self``, to allow chaining.
        """
        self.center = np.array([float(x), float(y)])
        if orientation is not None:
            self.orientation = float(orientation)
        self.placed = True
        return self

    def stow(self) -> "Magnet":
        """Mark the magnet as returned to its tray.

        Returns
        -------
        Magnet
            ``self``, to allow chaining.  If a `tray_position` is set, the
            centre is moved there.
        """
        self.placed = False
        if self.tray_position is not None:
            self.center = np.asarray(self.tray_position, dtype=float).reshape(2)
        return self

    def describe(self) -> str:
        """Return a one-line human-readable summary of the magnet's state.

        Returns
        -------
        str
        """
        state = "placed" if self.placed else "stowed"
        return (f"{self.kind}:{self.identifier} {state} "
                f"@({self.center[0]:.3f}, {self.center[1]:.3f}, yaw={self.orientation:.2f})")


@dataclass
class BlockMagnet(Magnet):
    """A weight block (with a gripper holder) that pins down a point or area.

    Parameters
    ----------
    holder_height : float, optional
        Height above the board (metres) at which the gripper grasps the block's
        holder.  Default ``0.02``.

    Other Parameters
    ----------------
    identifier, center, orientation, placed, tray_position
        Inherited from `Magnet`.
    """

    holder_height: float = 0.02
    kind: str = "block"

    @property
    def grip_height(self) -> float:
        """float: Gripping height -- the holder height above the board."""
        return self.holder_height


@dataclass
class LBracketMagnet(Magnet):
    """An L-bracket whose foot pins a board edge and whose handle is gripped.

    The bracket has two relevant parts:

    * the **magnetic foot**, centred at `center` on the board, which holds
      the paper and provides a clean pivot / hinge line;
    * the **handle**, an upright offset a fixed distance from the foot along the
      bracket's orientation and standing at a raised height -- this is what the
      gripper actually grasps.

    Parameters
    ----------
    handle_offset : float, optional
        Distance from the magnetic foot (`center`) to the handle, measured
        along `orientation` (metres).  Default ``0.03``.
    handle_height : float, optional
        Height of the handle's grip point above the board (metres).  Default
        ``0.03``.

    Other Parameters
    ----------------
    identifier, center, orientation, placed, tray_position
        Inherited from `Magnet`.  Here ``center`` is the magnetic foot and
        ``orientation`` is the direction from the foot toward the handle.
    """

    handle_offset: float = 0.03
    handle_height: float = 0.03
    kind: str = "lbracket"

    @property
    def handle_xy(self) -> np.ndarray:
        """numpy.ndarray, shape (2,): Board position of the handle grip point."""
        direction = np.array([np.cos(self.orientation), np.sin(self.orientation)])
        return self.center + direction * self.handle_offset

    @property
    def grip_xy(self) -> np.ndarray:
        """numpy.ndarray, shape (2,): Gripping point -- the handle, not the foot."""
        return self.handle_xy

    @property
    def grip_height(self) -> float:
        """float: Gripping height -- the raised handle height above the board."""
        return self.handle_height

    def hinge_line(self) -> FoldLine:
        """The board line the bracket pins / hinges the paper about.

        Returns
        -------
        origami.geometry.FoldLine
            A line through the magnetic foot along `orientation`.
        """
        return FoldLine.at_angle(self.center, self.orientation)


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

        Parameters
        ----------
        magnet : Magnet

        Returns
        -------
        Magnet
            The same magnet, for convenience.

        Raises
        ------
        ValueError
            If a magnet with the same `identifier` is already
            registered.
        """
        if magnet.identifier in self._magnets:
            raise ValueError(f"duplicate magnet identifier: {magnet.identifier}")
        self._magnets[magnet.identifier] = magnet
        return magnet

    def get(self, identifier: str) -> Magnet:
        """Look up a magnet by identifier.

        Parameters
        ----------
        identifier : str

        Returns
        -------
        Magnet
        """
        return self._magnets[identifier]

    def placed(self) -> list[Magnet]:
        """All magnets currently on the board.

        Returns
        -------
        list of Magnet
        """
        return [m for m in self._magnets.values() if m.placed]

    def available(self, kind: str | None = None) -> list[Magnet]:
        """Magnets still in their tray, optionally filtered by type.

        Parameters
        ----------
        kind : str or None, optional
            If given, restrict to magnets of this `kind`.

        Returns
        -------
        list of Magnet
        """
        return [m for m in self._magnets.values()
                if not m.placed and (kind is None or m.kind == kind)]

    def of_kind(self, kind: str) -> list[Magnet]:
        """All registered magnets of a given type.

        Parameters
        ----------
        kind : str
            Type tag, e.g. ``'block'`` or ``'lbracket'``.

        Returns
        -------
        list of Magnet
        """
        return [m for m in self._magnets.values() if m.kind == kind]

    def __contains__(self, identifier: str) -> bool:
        return identifier in self._magnets

    def __iter__(self):
        return iter(self._magnets.values())

    def __len__(self) -> int:
        return len(self._magnets)

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        body = "\n  ".join(m.describe() for m in self)
        return f"MagnetRegistry[\n  {body}\n]"
