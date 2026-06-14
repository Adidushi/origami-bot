"""High-level paper and magnet choreography.

All functions operate on a `~origami.workspace.Workspace` and keep the
analytic `Paper` / `MagnetRegistry` state in sync after every physical move.

Magnet actions
--------------
    place_magnet      fetch from tray → place on board
    move_magnet       relocate a placed magnet
    remove_magnet     pick from board → return to tray

Paper actions
-------------
    grip_corner       close gripper on a named paper corner
    grip_edge         close gripper on the midpoint of a named edge
    move_paper        translate the whole sheet by (dx, dy)
    rotate_paper      rotate the sheet about a pivot point
    flip_paper        flip the sheet over about its horizontal or vertical centre axis

Positioning constraints
-----------------------
Several paper actions require part of the sheet to *overhang* the board edge
so the gripper can descend onto it from above with nothing beneath it.  Each
function documents which constraint it needs and raises ``ValueError`` clearly
if it is not met.  Use ``move_paper()`` to satisfy a constraint before
calling a grip or fold action.

The minimum required overhang is ``GRIP_OVERHANG_MIN`` (5 mm by default).
"""
from __future__ import annotations

import math

import numpy as np

from .magnets import Magnet
from .workspace import Workspace

# ---------------------------------------------------------------------------
# Physical constants
# ---------------------------------------------------------------------------
#: Clearance above a magnet grip point when approaching / leaving (metres).
MAGNET_APPROACH_CLEARANCE = 0.05

#: World z at which to close the gripper on paper (just above board surface).
PAPER_GRIP_HEIGHT = 0.001

#: Minimum overhang beyond the board boundary (metres) required to grip paper.
GRIP_OVERHANG_MIN = 0.005

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------
_EDGE_CORNERS: dict[str, tuple[str, str]] = {
    "bottom": ("bottom_left", "bottom_right"),
    "top":    ("top_left",    "top_right"),
    "left":   ("bottom_left", "top_left"),
    "right":  ("bottom_right", "top_right"),
}

_CORNER_NAMES = ("bottom_left", "bottom_right", "top_right", "top_left")


def _overhangs_board(workspace: Workspace, pos) -> bool:
    """True if *pos* lies outside the board footprint by at least GRIP_OVERHANG_MIN."""
    bw, bh = workspace.board_width, workspace.board_height
    x, y = float(pos[0]), float(pos[1])
    return (
        x < -GRIP_OVERHANG_MIN
        or x > bw + GRIP_OVERHANG_MIN
        or y < -GRIP_OVERHANG_MIN
        or y > bh + GRIP_OVERHANG_MIN
    )


def _require_overhang(workspace: Workspace, pos, label: str) -> None:
    if not _overhangs_board(workspace, pos):
        raise ValueError(
            f"{label} does not overhang the board — reposition the paper so it "
            f"extends beyond the board boundary by at least "
            f"{GRIP_OVERHANG_MIN * 1000:.0f} mm.  Use move_paper() first."
        )


def _find_grippable_corner(workspace: Workspace) -> str:
    """Return the name of the first paper corner that overhangs the board."""
    for name in _CORNER_NAMES:
        if _overhangs_board(workspace, workspace.paper.landmark(name)):
            return name
    raise ValueError(
        "No paper corner overhangs the board boundary — reposition the paper "
        f"so at least one corner extends beyond the board edge by at least "
        f"{GRIP_OVERHANG_MIN * 1000:.0f} mm.  Use move_paper() first."
    )


def _grip_paper_at(arm, pos) -> None:
    """Transit to above pos, descend to paper height, close gripper, rise to clearance."""
    x, y = float(pos[0]), float(pos[1])
    arm.move_to_clearance(x, y)
    arm.move_to_world(x, y, PAPER_GRIP_HEIGHT)
    arm.grip()
    arm.move_to_world(x, y, MAGNET_APPROACH_CLEARANCE)


def _grip_magnet_at(arm, x: float, y: float, grip_height: float,
                    tool_rotation: float) -> None:
    """Approach a magnet from above, descend to its grip height, close, retreat."""
    clearance = grip_height + MAGNET_APPROACH_CLEARANCE
    arm.move_to_world(x, y, clearance, tool_rotation)
    arm.move_to_world(x, y, grip_height, tool_rotation)
    arm.grip()
    arm.move_to_world(x, y, clearance, tool_rotation)


def _release_magnet_at(arm, x: float, y: float, grip_height: float,
                       tool_rotation: float) -> None:
    """Carry to board position, descend to release height, open, retreat."""
    clearance = grip_height + MAGNET_APPROACH_CLEARANCE
    arm.move_to_world(x, y, clearance, tool_rotation)
    arm.move_to_world(x, y, grip_height, tool_rotation)
    arm.release()
    arm.move_to_world(x, y, clearance, tool_rotation)


# ===========================================================================
# Magnet actions
# ===========================================================================

def place_magnet(workspace: Workspace, magnet: Magnet, x: float, y: float,
                 orientation: float = 0.0, carrying_arm: str = "left") -> None:
    """Fetch a magnet from its tray and set it down on the board.

    Parameters
    ----------
    workspace : Workspace
    magnet : Magnet
        The magnet to place.  Its tray position, grip height and orientation
        are honoured.
    x, y : float
        Target board position for the magnet centre (metres).
    orientation : float, optional
        Target yaw of the magnet about the board normal (radians).
    carrying_arm : {'left', 'right'}, optional
        Arm that carries the magnet.
    """
    arm = workspace.arm(carrying_arm)

    if magnet.tray_position is not None:
        pick = np.asarray(magnet.tray_position, dtype=float)
        _grip_magnet_at(arm, float(pick[0]), float(pick[1]),
                        magnet.grip_height, magnet.orientation)

    _release_magnet_at(arm, x, y, magnet.grip_height, orientation)
    magnet.place_at(x, y, orientation)
    if magnet.identifier not in workspace.magnets:
        workspace.magnets.add(magnet)


def move_magnet(workspace: Workspace, identifier: str, x: float, y: float,
                orientation: float = 0.0, carrying_arm: str = "left") -> None:
    """Relocate an already-placed magnet to a new board position.

    Parameters
    ----------
    workspace : Workspace
    identifier : str
        Identifier of the magnet to move.
    x, y : float
        New board position (metres).
    orientation : float, optional
        New yaw of the magnet (radians).  Default 0.
    carrying_arm : {'left', 'right'}, optional
    """
    arm = workspace.arm(carrying_arm)
    magnet = workspace.magnets.get(identifier)
    handle = magnet.handle_xy

    _grip_magnet_at(arm, float(handle[0]), float(handle[1]),
                    magnet.grip_height, magnet.orientation)
    _release_magnet_at(arm, x, y, magnet.grip_height, orientation)
    magnet.place_at(x, y, orientation)


def remove_magnet(workspace: Workspace, identifier: str,
                  carrying_arm: str = "left") -> None:
    """Pick a placed magnet back up and return it to its tray.

    Parameters
    ----------
    workspace : Workspace
    identifier : str
        Identifier of the magnet to remove.
    carrying_arm : {'left', 'right'}, optional
    """
    arm = workspace.arm(carrying_arm)
    magnet = workspace.magnets.get(identifier)
    handle = magnet.handle_xy

    _grip_magnet_at(arm, float(handle[0]), float(handle[1]),
                    magnet.grip_height, magnet.orientation)

    if magnet.tray_position is not None:
        home = np.asarray(magnet.tray_position, dtype=float)
        _release_magnet_at(arm, float(home[0]), float(home[1]),
                           magnet.grip_height, magnet.orientation)
    else:
        arm.release()
        arm.lift()

    magnet.stow()


# ===========================================================================
# Paper actions
# ===========================================================================

def grip_corner(workspace: Workspace, corner: str, arm: str = "right") -> None:
    """Grip a named paper corner and lift it to clearance height.

    **Constraint**: the corner must overhang the board boundary by at least
    ``GRIP_OVERHANG_MIN`` so there is no board material beneath it for the
    gripper to collide with.

    Parameters
    ----------
    workspace : Workspace
    corner : {'bottom_left', 'bottom_right', 'top_left', 'top_right'}
        Which corner to grip.
    arm : str, optional
        Which arm grips.  Default ``'right'``.

    Raises
    ------
    ValueError
        If the corner does not overhang the board.
    """
    arm_obj = workspace.arm(arm)
    pos = workspace.paper.landmark(corner)
    _require_overhang(workspace, pos, f"corner '{corner}'")
    _grip_paper_at(arm_obj, pos)


def grip_edge(workspace: Workspace, edge: str, arm: str = "right") -> None:
    """Grip the midpoint of a named paper edge and lift it to clearance height.

    **Constraint**: at least one endpoint of the edge must overhang the board
    boundary so the gripper can descend without hitting the board.

    Parameters
    ----------
    workspace : Workspace
    edge : {'bottom', 'top', 'left', 'right'}
        Which edge to grip.
    arm : str, optional
        Which arm grips.  Default ``'right'``.

    Raises
    ------
    ValueError
        If neither endpoint of the edge overhangs the board.
    """
    arm_obj = workspace.arm(arm)
    c1, c2 = _EDGE_CORNERS[edge]
    p1 = workspace.paper.landmark(c1)
    p2 = workspace.paper.landmark(c2)

    if not (_overhangs_board(workspace, p1) or _overhangs_board(workspace, p2)):
        raise ValueError(
            f"edge '{edge}' does not overhang the board — reposition the paper "
            f"so this edge extends beyond the board boundary by at least "
            f"{GRIP_OVERHANG_MIN * 1000:.0f} mm.  Use move_paper() first."
        )

    mid = (p1 + p2) / 2.0
    _grip_paper_at(arm_obj, mid)


def move_paper(workspace: Workspace, dx: float, dy: float,
               carrying_arm: str = "right", corner: str | None = None) -> None:
    """Slide the whole sheet by ``(dx, dy)`` on the board.

    Grips an overhanging corner, carries it to its translated position, and
    updates the paper model.

    **Constraint**: at least one paper corner must overhang the board edge.
    If the sheet is entirely within the board it cannot be gripped.

    Parameters
    ----------
    workspace : Workspace
    dx, dy : float
        Translation in world-frame metres.
    carrying_arm : str, optional
        Which arm carries the paper.  Default ``'right'``.
    corner : str or None, optional
        Which corner to grip; auto-detected (first overhanging corner) if
        ``None``.

    Raises
    ------
    ValueError
        If no corner overhangs the board.
    """
    arm_obj = workspace.arm(carrying_arm)

    if corner is None:
        corner = _find_grippable_corner(workspace)

    pos = workspace.paper.landmark(corner)
    _require_overhang(workspace, pos, f"corner '{corner}'")

    x, y = float(pos[0]), float(pos[1])

    # Grip and lift
    arm_obj.move_to_clearance(x, y)
    arm_obj.move_to_world(x, y, PAPER_GRIP_HEIGHT)
    arm_obj.grip()

    # Carry to translated position
    tx, ty = x + dx, y + dy
    arm_obj.move_to_clearance(tx, ty)
    arm_obj.move_to_world(tx, ty, PAPER_GRIP_HEIGHT)
    arm_obj.release()
    arm_obj.lift()

    workspace.paper.translate([dx, dy])


def rotate_paper(workspace: Workspace, angle: float, pivot=None,
                 folding_arm: str = "right", anchor_arm: str = "left",
                 n_steps: int = 10, corner: str | None = None) -> None:
    """Rotate the paper by *angle* radians about *pivot*.

    The anchor arm presses at the pivot to prevent the sheet from drifting;
    the folding arm grips an overhanging corner and arcs it to its new
    position via *n_steps* linear waypoints.

    **Constraint**: at least one paper corner must overhang the board edge
    so it can be gripped by the folding arm.

    Parameters
    ----------
    workspace : Workspace
    angle : float
        Rotation angle in radians (counter-clockwise).
    pivot : array_like (2,) or None, optional
        World ``(x, y)`` centre of rotation.  Defaults to the paper centroid.
    folding_arm, anchor_arm : str, optional
        Which arms perform each role.  Defaults ``'right'`` and ``'left'``.
    n_steps : int, optional
        Number of arc waypoints (more = smoother path but slower).  Default 10.
    corner : str or None, optional
        Corner for the folding arm to grip; auto-detected if ``None``.

    Raises
    ------
    ValueError
        If no grippable corner is found.
    """
    paper = workspace.paper
    pivot_xy = (paper.centroid() if pivot is None
                else np.asarray(pivot, dtype=float).reshape(2))

    if corner is None:
        corner = _find_grippable_corner(workspace)

    corner_pos = paper.landmark(corner)

    anchor = workspace.arm(anchor_arm)
    folding = workspace.arm(folding_arm)

    # Anchor arm stabilises at the pivot
    anchor.move_to_clearance(float(pivot_xy[0]), float(pivot_xy[1]))
    anchor.press(float(pivot_xy[0]), float(pivot_xy[1]))

    # Folding arm grips the corner; arm ends at MAGNET_APPROACH_CLEARANCE
    _grip_paper_at(folding, corner_pos)

    # Arc the corner to its new position
    radius = float(np.linalg.norm(corner_pos - pivot_xy))
    start_a = math.atan2(
        float(corner_pos[1] - pivot_xy[1]),
        float(corner_pos[0] - pivot_xy[0]),
    )

    for i in range(1, n_steps + 1):
        a = start_a + angle * i / n_steps
        wx = float(pivot_xy[0]) + radius * math.cos(a)
        wy = float(pivot_xy[1]) + radius * math.sin(a)
        folding.move_to_world(wx, wy, MAGNET_APPROACH_CLEARANCE)

    # Set the corner down at the final position
    final_a = start_a + angle
    fx = float(pivot_xy[0]) + radius * math.cos(final_a)
    fy = float(pivot_xy[1]) + radius * math.sin(final_a)
    folding.move_to_world(fx, fy, PAPER_GRIP_HEIGHT)
    folding.release()
    folding.lift()

    # Release anchor
    anchor.lift()

    workspace.paper.rotate(angle, pivot=pivot_xy)


def flip_paper(workspace: Workspace, axis: str = "y",
               folding_arm: str = "right", anchor_arm: str = "left") -> None:
    """Flip the paper over about its horizontal (``'y'``) or vertical (``'x'``) centre axis.

    The folding arm grips the leading edge, lifts it up and over the paper,
    and sets it down on the far side.  The anchor arm presses the opposite
    edge to keep the sheet from sliding during the flip.

    **Constraints** (checked at call time):

    * ``axis='y'`` (bottom-over flip): the *bottom* edge must overhang the
      board boundary.  The folding arm grips the bottom edge from below.
    * ``axis='x'`` (left-over flip): the *left* edge must overhang the board
      boundary.

    Use ``move_paper()`` to satisfy the constraint before calling this function.

    Parameters
    ----------
    workspace : Workspace
    axis : {'y', 'x'}, optional
        ``'y'`` flips the bottom edge up and over (default);
        ``'x'`` flips the left edge over to the right.
    folding_arm, anchor_arm : str, optional
        Which arms perform each role.  Defaults ``'right'`` and ``'left'``.

    Raises
    ------
    ValueError
        If the required edge does not overhang the board, or ``axis`` is not
        ``'x'`` or ``'y'``.
    """
    if axis not in ("x", "y"):
        raise ValueError(f"axis must be 'x' or 'y', got {axis!r}")

    paper = workspace.paper
    folding = workspace.arm(folding_arm)
    anchor = workspace.arm(anchor_arm)

    grip_edge_name = "bottom" if axis == "y" else "left"
    anchor_edge_name = "top"  if axis == "y" else "right"

    c1, c2 = _EDGE_CORNERS[grip_edge_name]
    p1 = paper.landmark(c1)
    p2 = paper.landmark(c2)

    if not (_overhangs_board(workspace, p1) or _overhangs_board(workspace, p2)):
        raise ValueError(
            f"flip_paper(axis='{axis}'): the '{grip_edge_name}' edge must overhang "
            f"the board by at least {GRIP_OVERHANG_MIN * 1000:.0f} mm. "
            f"Use move_paper() to reposition the sheet first."
        )

    mid = (p1 + p2) / 2.0  # midpoint of the leading edge to grip

    # Anchor arm presses the opposite edge
    ac1, ac2 = _EDGE_CORNERS[anchor_edge_name]
    anchor_mid = (paper.landmark(ac1) + paper.landmark(ac2)) / 2.0
    anchor.move_to_clearance(float(anchor_mid[0]), float(anchor_mid[1]))
    anchor.press(float(anchor_mid[0]), float(anchor_mid[1]))

    # Grip the leading edge
    _grip_paper_at(folding, mid)

    centre = paper.centroid()

    # Arc height: enough to clear the farthest point of the paper
    pts = paper.landmark_array()
    max_radius = float(np.max(np.linalg.norm(pts - centre, axis=1)))
    arc_height = max_radius + MAGNET_APPROACH_CLEARANCE

    if axis == "y":
        # Bottom edge arcs up and over to the top position
        final_y = 2.0 * float(centre[1]) - float(mid[1])
        folding.move_to_world(float(mid[0]), float(mid[1]),   arc_height)
        folding.move_to_world(float(mid[0]), float(centre[1]), arc_height)
        folding.move_to_world(float(mid[0]), final_y,          arc_height)
        folding.move_to_world(float(mid[0]), final_y,          PAPER_GRIP_HEIGHT)
    else:
        # Left edge arcs over to the right position
        final_x = 2.0 * float(centre[0]) - float(mid[0])
        folding.move_to_world(float(mid[0]),   float(mid[1]), arc_height)
        folding.move_to_world(float(centre[0]), float(mid[1]), arc_height)
        folding.move_to_world(final_x,          float(mid[1]), arc_height)
        folding.move_to_world(final_x,          float(mid[1]), PAPER_GRIP_HEIGHT)

    folding.release()
    folding.lift()
    anchor.lift()

    # Update paper model: reflect landmarks about the centre axis
    centre = paper.centroid()  # re-read (centroid hasn't changed yet)
    for name in list(paper.landmarks):
        pt = paper.landmarks[name]
        if axis == "y":
            paper.landmarks[name] = np.array([pt[0], 2.0 * float(centre[1]) - pt[1]])
        else:
            paper.landmarks[name] = np.array([2.0 * float(centre[0]) - pt[0], pt[1]])
    paper.history.append(f"flip about {axis}-axis through centroid")
