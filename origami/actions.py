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
    grip_paper        close gripper on a paper edge or corner at an arbitrary (x, y, angle)
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
from typing import Tuple

import numpy as np

from origami.arm import Arm

from .magnets import Magnet
from .workspace import Workspace
import time

# ---------------------------------------------------------------------------
# Physical constants
# ---------------------------------------------------------------------------
#: Clearance above a magnet grip point when approaching / leaving (metres).
MAGNET_APPROACH_CLEARANCE = 0.08

#: World z at which to close the gripper on paper (just above board surface).
PAPER_GRIP_HEIGHT = 0.001

#: Minimum overhang beyond the board boundary (metres) required to grip paper.
GRIP_OVERHANG_MIN = 0.005

#: How far outside the grip point the arm starts its horizontal approach (metres).
PAPER_APPROACH_OFFSET = 0.05

MAGNET_GRIP_OPEN_POS = 0.6
MAGNET_GRIP_CLOSE_POS = 0.8
# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _grip_paper_at(arm, pos) -> None:
    """Transit to above pos, descend to paper height, close gripper, rise to clearance."""
    x, y = float(pos[0]), float(pos[1])
    arm.move_to_clearance(x, y)
    arm.move_to_world(x, y, PAPER_GRIP_HEIGHT)
    arm.grip()
    arm.move_to_world(x, y, MAGNET_APPROACH_CLEARANCE)


def _grip_magnet_at(arm, x: float, y: float, z: float,
                    tool_rotation: float) -> None:
    """Approach a magnet from above, descend to its grip height, close, retreat."""
    clearance = z + MAGNET_APPROACH_CLEARANCE
    arm.move_to_world(x, y, clearance, tool_rotation)
    arm.goto(MAGNET_GRIP_OPEN_POS)  # ensure gripper is open before descending
    time.sleep(1)  # give the gripper a moment to open before moving down
    arm.move_to_world(x, y, z, tool_rotation)
    arm.goto(MAGNET_GRIP_CLOSE_POS)
    # input("Time to measure magnet") # Remove this later - just for testing
    time.sleep(1)  # give the gripper a moment to close before lifting
    arm.move_to_world(x, y, clearance, tool_rotation)


def _release_magnet_at(arm, x: float, y: float, z: float,
                       tool_rotation: float) -> None:
    """Carry to board position, descend to release height, open, retreat."""
    clearance = z + MAGNET_APPROACH_CLEARANCE
    arm.move_to_world(x, y, clearance, tool_rotation)
    arm.move_to_world(x, y, z, tool_rotation)
    arm.goto(MAGNET_GRIP_OPEN_POS)
    # input("Time to measure magnet") # Remove this later - just for testing
    time.sleep(1)  # give the gripper a moment to open before moving
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
                        float(pick[2]) + magnet.grip_height, magnet.orientation)

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
                           float(home[2]) + magnet.grip_height, magnet.orientation)
    else:
        arm.goto(MAGNET_GRIP_OPEN_POS)
        time.sleep(1)  # give the gripper a moment to open before lifting
        arm.lift()

    magnet.stow() # Update the magnet's state to reflect that it's stowed away back in the tray


# ===========================================================================
# Paper actions
# ===========================================================================

def grip_paper(workspace: Workspace, x: float, y: float, grip_angle: float,
               arm: str = "right") -> None:
    """Grip the paper at an edge or corner by approaching horizontally from outside the board.

    The paper is assumed to overhang the board at ``(x, y)``.  The arm transits
    to the approach point at clearance height, descends to paper height, then
    slides in horizontally along ``grip_angle`` to straddle the edge before
    closing the gripper.

    Parameters
    ----------
    workspace : Workspace
    x, y : float
        Board-plane position to grip (metres).  z is fixed to world z = 0
        (the board/paper surface).
    grip_angle : float
        Direction the arm approaches from, as a rotation about the board normal
        (radians).  The arm starts ``PAPER_APPROACH_OFFSET`` outside the board
        in the ``(-cos(grip_angle), -sin(grip_angle))`` direction and slides in
        to ``(x, y)``.  Use ``0`` to approach from the left (``-x``), ``π/2``
        to approach from below (``-y``), etc.
    arm : {'left', 'right'}, optional
        Which arm performs the grip.  Default ``'right'``.
    """
    a = workspace.arm(arm)
    x_start = x - PAPER_APPROACH_OFFSET * math.cos(grip_angle)
    y_start = y - PAPER_APPROACH_OFFSET * math.sin(grip_angle)

    # Step 1: transit to approach start in downward orientation — IK is
    # well-conditioned here so the arm naturally lands in a good config.
    a.move_to_clearance(x_start, y_start, grip_angle)
    # Step 2: ensure elbow-up before asking for the sideways orientation change,
    # so the subsequent moveL starts from the right configuration.
    #a.ensure_elbow_up()
    # Step 3: reorient to sideways at clearance height.
    a.move_to_clearance(x_start, y_start, grip_angle, sideways=True)
    # Step 4: safety net — correct if IK still drifted elbow-down.
    # a.ensure_elbow_up()
    # Step 5: descend to paper height outside the board (nothing to hit here).
    a.move_to_world(x_start, y_start, PAPER_GRIP_HEIGHT, grip_angle, sideways=True)
    # Step 5.5?: open da gripper
    a.goto(.5)
    # Slide horizontally in to the paper edge.
    a.move_to_world(x, y, PAPER_GRIP_HEIGHT, grip_angle, sideways=True)
    a.grip()

def flip_paper(workspace: Workspace,
               arm: str = "right") -> None:
    """Grip the paper at an edge or corner by approaching horizontally from outside the board.

    The paper is assumed to overhang the board at ``(x, y)``.  The arm transits
    to the approach point at clearance height, descends to paper height, then
    slides in horizontally along ``grip_angle`` to straddle the edge before
    closing the gripper.

    Parameters
    ----------
    workspace : Workspace
    x, y : float
        Board-plane position to grip (metres).  z is fixed to world z = 0
        (the board/paper surface).
    grip_angle : float
        Direction the arm approaches from, as a rotation about the board normal
        (radians).  The arm starts ``PAPER_APPROACH_OFFSET`` outside the board
        in the ``(-cos(grip_angle), -sin(grip_angle))`` direction and slides in
        to ``(x, y)``.  Use ``0`` to approach from the left (``-x``), ``π/2``
        to approach from below (``-y``), etc.
    arm : {'left', 'right'}, optional
        Which arm performs the grip.  Default ``'right'``.
    """

    clearance = 0.2

    a = workspace.arm(arm)
    a.move_offset_world(0, 0, clearance)
    a.rotate_joint(5, math.pi)
    a.move_offset_world(0, 0, -clearance)


# ===========================================================================
# Fold actions
# ===========================================================================

def fold_arc(
    workspace: Workspace,
    arm_side: str,
    radius: float,
    axis: str,
    n_steps: int = 8,
) -> None:
    """Fold paper by sweeping the gripped edge through a circular arc about a fold line.

    The fold axis is a line parallel to the y-axis at x = ``fold_axis_x`` on
    the board surface (z = 0).  Call this immediately after ``grip_paper``; the
    arm's current position is taken as the arc start point.

    Each waypoint is placed exactly on the circle of radius
    ``r = |start_x - fold_axis_x|`` centred at (fold_axis_x, y, 0), so the
    gripper traces a true semicircle in the x-z plane.  y stays constant.
    Wrist (joint 5) rotates by the same angular step at each waypoint so the
    gripper stays perpendicular to the paper throughout the fold.

    Sequence per step: rotate wrist → moveL to next arc position.

    Parameters
    ----------
    workspace : Workspace
    arm_side : {'left', 'right'}
        Arm carrying the paper edge.
    fold_axis_x : float
        World x coordinate of the fold crease (metres).
    n_steps : int
        Number of equally-spaced waypoints along the π-radian arc.  Default 8.
    """

    # select arm
    arm = workspace.arm(arm_side)

    # set the starting position
    start_pos = list(arm.current_world_pos())

    # set the mid and end point depending on radius and axis selection
    if (axis == 'x'):
        end_pos = start_pos.copy()
        midpoint = start_pos.copy()
        end_pos[0] += 2 * radius
        midpoint[0] += radius
    else:
        end_pos = start_pos.copy()
        midpoint = start_pos.copy()
        end_pos[1] += 2 * radius
        midpoint[1] += radius

    previous_pos = start_pos

    offsets = list()
    # calculate steps 
    for i in range(1, n_steps+1):
        # if radius is positive, fold to the right, else fold to the left
        right_flag = np.sign(radius) > 0 
        base_angle = i*math.pi/n_steps
        angle = (math.pi - base_angle) if right_flag else base_angle

        current_pos = midpoint.copy()
        current_pos[2] += radius * math.sin(angle)
        if (axis == 'x'):
            current_pos[0] += radius * math.cos(angle)
        else:
            current_pos[1] += radius * math.cos(angle)

        #arm.move_to_world(*current_pos, 0, sideways=True)
        offset = [a-b for a,b in zip(current_pos,previous_pos)]
        print(offset)
        offsets.append(offset)
        # divide by i since rotate joint is relative
        arm.rotate_joint(5, base_angle/i)
        arm.move_offset_world(*offset)
        
        previous_pos = current_pos.copy()
    return offsets

def unfold_arc(
    workspace: Workspace,
    arm_side: str,
    offsets: list[list[float,float,float]]
) -> None:
    """Unfold paper by sweeping the gripped edge through a circular arc about a fold line.
    """
    arm = workspace.arm(arm_side)
    angle = math.pi/len(offsets)
    for offset in offsets[::-1]: # reverse the offsets to unfold
        arm.rotate_joint(5, -angle)
        arm.move_offset_world(*[-coord for coord in offset])
    

# afterwards move this out to a crease tool method?
def grip_crease_tool(workspace: Workspace, x: float, y: float, z: float, grip_angle: float,
               arm: str = "right") -> None:
    """Grip the crease at an edge or corner by approaching horizontally from outside the board.

    The arm transits to the approach point at clearance height, descends to paper height, then
    slides in horizontally along ``grip_angle`` to straddle the edge before
    closing the gripper.

    Parameters
    ----------
    workspace : Workspace
    x, y, z : float
        Board-plane position to grip (metres).  z is fixed to world z = 0
        (the board/paper surface).
    grip_angle : float
        Direction the arm approaches from, as a rotation about the board normal
        (radians).  The arm starts ``PAPER_APPROACH_OFFSET`` outside the board
        in the ``(-cos(grip_angle), -sin(grip_angle))`` direction and slides in
        to ``(x, y)``.  Use ``0`` to approach from the left (``-x``), ``π/2``
        to approach from below (``-y``), etc.
    arm : {'left', 'right'}, optional
        Which arm performs the grip.  Default ``'right'``.
    """
    a = workspace.arm(arm)
    x_start = x - PAPER_APPROACH_OFFSET * math.cos(grip_angle)
    y_start = y - PAPER_APPROACH_OFFSET * math.sin(grip_angle)
    z_start = z
    # Step 1: transit to approach start in downward orientation — IK is
    # well-conditioned here so the arm naturally lands in a good config.
    a.move_to_clearance(x_start, y_start, grip_angle)
    # Step 2: ensure elbow-up before asking for the sideways orientation change,
    # so the subsequent moveL starts from the right configuration.

    # Step 3: reorient to sideways at clearance height.
    a.move_to_clearance(x_start, y_start, grip_angle, sideways=True)
    # Step 4: safety net — correct if IK still drifted elbow-down.

    # Step 5: descend to paper height outside the board (nothing to hit here).
    a.move_to_world(x_start, y_start, z_start, grip_angle, sideways=True)
    # Step 5.5?: open da gripper
    a.goto(.5)
    # Slide horizontally in to the paper edge.
    a.move_to_world(x, y, z, grip_angle, sideways=True)
    a.grip()
    a.move_offset_world(-0.05, 0, 0.05) # move up and back a bit to lift up the creaser tool
    a.go_home()

def return_creaser_tool(workspace: Workspace, x: float, y: float, z: float, grip_angle: float,
               arm: str = "right") -> None:
    """Grip the crease at an edge or corner by approaching horizontally from outside the board.

    The arm transits to the approach point at clearance height, descends to paper height, then
    slides in horizontally along ``grip_angle`` to straddle the edge before
    closing the gripper.

    Parameters
    ----------
    workspace : Workspace
    x, y, z : float
        Board-plane position to grip (metres).  z is fixed to world z = 0
        (the board/paper surface).
    grip_angle : float
        Direction the arm approaches from, as a rotation about the board normal
        (radians).  The arm starts ``PAPER_APPROACH_OFFSET`` outside the board
        in the ``(-cos(grip_angle), -sin(grip_angle))`` direction and slides in
        to ``(x, y)``.  Use ``0`` to approach from the left (``-x``), ``π/2``
        to approach from below (``-y``), etc.
    arm : {'left', 'right'}, optional
        Which arm performs the grip.  Default ``'right'``.
    """
    a = workspace.arm(arm)
    x_start = x - PAPER_APPROACH_OFFSET * math.cos(grip_angle)
    y_start = y - PAPER_APPROACH_OFFSET * math.sin(grip_angle)
    z_start = z
    # Step 1: transit to approach start in downward orientation — IK is
    # well-conditioned here so the arm naturally lands in a good config.
    a.move_to_clearance(x_start, y_start, grip_angle)
    # Step 2: ensure elbow-up before asking for the sideways orientation change,
    # so the subsequent moveL starts from the right configuration.

    # Step 3: reorient to sideways at clearance height.
    a.move_to_clearance(x_start, y_start, grip_angle, sideways=True)
    # Step 4: safety net — correct if IK still drifted elbow-down.

    # Step 5: descend to paper height outside the board (nothing to hit here).
    a.move_to_world(x_start, y_start, z_start, grip_angle, sideways=True)
    # Step 5.5?: open da gripper
    # Slide horizontally in to the paper edge.
    a.move_to_world(x, y, z, grip_angle, sideways=True)
    a.goto(.5)
    a.move_offset_world(-0.1, 0, 0) # move up and back a bit to lift up the creaser tool
    a.grip()
    a.go_home()



def crease(
    workspace: Workspace,
    arm_side: str,
    axis: str,
    start_x: float,
    start_y: float,
    crease_length: float,
) -> None:
    """Creases the paper by picking up the creaser tool (atm its position is hardcoded in this function
    later on we will extract it to a tool registry that workspace has access to), rotate tool tip by 45 degrees in direction of crease
    axis and then move along the crease line for a given length.

    Parameters
    ----------
    workspace : Workspace
    arm_side : {'left', 'right'}
        Arm carrying the creaser tool.
    axis : {'x', 'y'}
        Axis along which to crease the paper.
    start_x : float
        World x coordinate of the start of the crease (metres).
    start_y : float
        World y coordinate of the start of the crease (metres).
    crease_length : float
        Length of the crease (metres).
    """
    if axis == 'x':
        start_x += crease_length/2
    else:
        start_y += crease_length/2
    
    arm = workspace.arm(arm_side)
    arm.go_home()
    arm.grip()
    # pick up the creaser tool (hardcoded position for now)
    crease_x, crease_y, crease_z = -0.21, 0.21, 0.03
    grip_crease_tool(workspace, crease_x, crease_y, crease_z, grip_angle=0, arm=arm_side)


    clearance_offset = 0.1
    crease_height = 6.5/100# * math.sin(math.pi/4) # safe height to


    # rotate tool tip by 45 degrees in direction of crease
    # if axis == 'x':
    #     arm.rotate_joint(4, math.pi/4)
    # else:
    #     arm.rotate_joint(4, -math.pi/4)

    
    arm.move_to_world(start_x, start_y, crease_height+clearance_offset)
    if axis == 'y':
        arm.rotate_joint(5, math.pi/2)
    arm.move_offset_world(0,0,-clearance_offset)
    # move along the crease line for a given length
    if axis == 'x':
        arm.move_offset_world(crease_length/2, 0, 0)
        arm.move_offset_world(0,0,clearance_offset)
        arm.move_offset_world(-crease_length/2, 0, 0)
        arm.move_offset_world(0,0,-clearance_offset)
        arm.move_offset_world(-crease_length/2, 0, 0)
    else:
        arm.move_offset_world(0, crease_length/2, 0)
        arm.move_offset_world(0,0,clearance_offset)
        arm.move_offset_world(0, -crease_length/2, 0)
        arm.move_offset_world(0,0,-clearance_offset)
        arm.move_offset_world(0, -crease_length/2, 0)

    return_creaser_tool(workspace, crease_x, crease_y, crease_z, grip_angle=0, arm=arm_side)

