"""High-level folding and magnet choreography.

These functions coordinate the two arms *and* keep the analytic
`Paper` / `MagnetRegistry` state in
sync, so a fold *recipe* (see `origami.demos.dart`) reads like a short list
of natural instructions.

The strategy for a valley fold is the classic two-arm division of labour:

1. the **anchoring arm** presses down on the stationary side, just across the
   crease, so the sheet pivots cleanly about the fold line;
2. the **folding arm** grips the moving flap's free corner, carries it across the
   crease and presses it down onto the other side;
3. the folding arm then runs along the crease to sharpen it.

Finally the paper model reflects the moving landmarks across the fold line.  In
simulation every move is logged; on hardware the same calls drive the arms.

Magnet handling respects each magnet's own grip point and height, since a
magnet's handle is generally **not** on the board surface.
"""
from __future__ import annotations


import numpy as np

from .magnets import Magnet
from .workspace import Workspace


def place_magnet(workspace: Workspace, magnet: Magnet, x: float, y: float,
                 orientation: float = 0.0, carrying_arm: str = "left") -> None:
    """Fetch a magnet from its tray and set it down on the board.

    The arm grips the magnet at its own grip point and height (e.g. an
    L-bracket's raised handle), carries it over and releases it at the target
    board pose.

    Parameters
    ----------
    workspace : origami.workspace.Workspace
        The workspace; the magnet is added to its registry if not already present.
    magnet : origami.magnets.Magnet
        The magnet to place.  Its grip geometry and height are honoured.
    x, y : float
        Target board position for the magnet centre (metres).
    orientation : float, optional
        Target yaw of the magnet about the board normal (radians).  Default ``0``.
    carrying_arm : {'left', 'right'}, optional
        Arm that carries the magnet.  Default ``'left'``.

    Returns
    -------
    None
    """
    arm = workspace.arm(carrying_arm)

    if magnet.tray_position is not None:
        pick = np.asarray(magnet.tray_position, dtype=float).reshape(2)
        _grip_magnet_at(arm, pick[0], pick[1], magnet.grip_height, magnet.orientation)

    # Carry to the target and release the magnet there.
    _release_magnet_at(arm, x, y, magnet.grip_height, orientation)
    magnet.place_at(x, y, orientation)
    if magnet.identifier not in workspace.magnets:
        workspace.magnets.add(magnet)


def remove_magnet(workspace: Workspace, identifier: str, carrying_arm: str = "left") -> None:
    """Pick a placed magnet back up and return it to its tray.

    Parameters
    ----------
    workspace : origami.workspace.Workspace
        The workspace holding the magnet.
    identifier : str
        Identifier of the magnet to remove.
    carrying_arm : {'left', 'right'}, optional
        Arm that carries the magnet.  Default ``'left'``.

    Returns
    -------
    None
    """
    arm = workspace.arm(carrying_arm)
    magnet = workspace.magnets.get(identifier)
    grip = magnet.grip_xy

    _grip_magnet_at(arm, grip[0], grip[1], magnet.grip_height, magnet.orientation)
    if magnet.tray_position is not None:
        home = np.asarray(magnet.tray_position, dtype=float).reshape(2)
        _release_magnet_at(arm, home[0], home[1], magnet.grip_height, magnet.orientation)
    else:
        arm.release()
        arm.lift_off_board()
    magnet.stow()


# --------------------------------------------------------------------------- #
# Internal motion / selection helpers
# --------------------------------------------------------------------------- #
def _grip_magnet_at(arm, x: float, y: float, grip_height: float, tool_rotation: float) -> None:
    """Approach a magnet from above, descend to its grip height and close.

    Parameters
    ----------
    arm : origami.arm.Arm
    x, y : float
        Board position of the grip point (metres).
    grip_height : float
        Height of the grip point above the board (metres).
    tool_rotation : float
        Gripper rotation about the board normal (radians).
    """
    clearance = grip_height + MAGNET_APPROACH_CLEARANCE
    arm.move_to_board_point(x, y, clearance, tool_rotation)
    arm.move_to_board_point(x, y, grip_height, tool_rotation)
    arm.grip()
    arm.move_to_board_point(x, y, clearance, tool_rotation)


def _release_magnet_at(arm, x: float, y: float, grip_height: float, tool_rotation: float) -> None:
    """Carry to a board point, descend to the release height, open and retreat.

    Parameters
    ----------
    arm : origami.arm.Arm
    x, y : float
        Board position to release at (metres).
    grip_height : float
        Height of the grip point above the board (metres).
    tool_rotation : float
        Gripper rotation about the board normal (radians).
    """
    clearance = grip_height
    arm.move_to_board_point(x, y, clearance, tool_rotation)
    arm.move_to_board_point(x, y, grip_height, tool_rotation)
    arm.release()
    arm.move_to_board_point(x, y, clearance, tool_rotation)
