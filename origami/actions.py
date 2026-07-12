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

import numpy as np

from origami import config
from origami.rotation import ToolOrientation

from .magnets import Magnet
from .workspace import Workspace
from .rotation import compose_rotation_vectors

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

def _grip_magnet_at(arm, x: float, y: float, z: float) -> None:
    """Approach a magnet from above, descend to its grip height, close, retreat."""
    clearance = z + MAGNET_APPROACH_CLEARANCE
    arm.move_to_world(x, y, clearance)
    arm.goto(MAGNET_GRIP_OPEN_POS, blocking=True)
    arm.move_to_world(x, y, z)
    arm.goto(MAGNET_GRIP_CLOSE_POS, blocking=True)
    arm.move_to_world(x, y, clearance)


def _release_magnet_at(arm, x: float, y: float, z: float) -> None:
    """Carry to board position, descend to release height, open, retreat."""
    clearance = z + MAGNET_APPROACH_CLEARANCE
    arm.move_to_world(x, y, clearance)
    arm.move_to_world(x, y, z)
    arm.goto(MAGNET_GRIP_OPEN_POS, blocking=True)
    arm.move_to_world(x, y, clearance)


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
        pick = np.asarray(magnet.handle_xy, dtype=float)
        _grip_magnet_at(arm, float(pick[0]), float(pick[1]),
                        float(magnet.tray_position[2]) + magnet.grip_height)

    # update x and y so handle_xy is calculated correctly for the new position of the magnet
    magnet.place_at(x, y, orientation)
    handle_x, handle_y = magnet.handle_xy
    _release_magnet_at(arm, handle_x, handle_y, magnet.grip_height)
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

    _grip_magnet_at(arm, float(handle[0]), float(handle[1]), magnet.grip_height)
    _release_magnet_at(arm, x, y, magnet.grip_height)
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

    _grip_magnet_at(arm, float(handle[0]), float(handle[1]), magnet.grip_height)

    magnet.stow() # Update the magnet's state to reflect that it's stowed away back in the tray

    if magnet.tray_position is not None:
        home = magnet.handle_xy
        _release_magnet_at(arm, float(home[0]), float(home[1]),
                           float(magnet.tray_position[2]) + magnet.grip_height)
    else:
        arm.goto(MAGNET_GRIP_OPEN_POS, blocking=True) # open the gripper and block until it is fully open before moving the arm up
        arm.lift()



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
    # grip angle is relative to forward direction, which is relative to y axis,
    # however typically angles are defined relative to x axis so that x stuff is via cos and y stuff is via sin,
    # by taking 90 - angle = pi/2 - angle we get the angle relative to the x axis again.
    x_start = x + PAPER_APPROACH_OFFSET * math.cos(math.pi/2-grip_angle)
    y_start = y - PAPER_APPROACH_OFFSET * math.sin(math.pi/2-grip_angle)

    # Step 1: transit to approach start, preserving current orientation.
    a.move_to_clearance(x_start, y_start)
    forward_rotvec = ToolOrientation.from_labels(tooltip="forward", gripper="flat").to_rot_vec()
    # rotate the gripper to point forward and flat (so it can grip the paper) so that it is facing the wall in an easy to start orientation
    # rotate the gripper to point in the direction of the grip angle so that it can approach the paper edge at the correct angle
    forward_rotated_rotvec = compose_rotation_vectors(forward_rotvec, [0, 0, grip_angle])
    a.rotate_absolute(*forward_rotated_rotvec) 
    
    a.move_to_tcp(a.world_to_tcp(x_start, y_start, PAPER_GRIP_HEIGHT)) # move to paper grip height at the approach point
    a.goto(.5) # open the gripper to prepare to grip the paper
    # Slide horizontally in to the paper edge.
    a.move_to_tcp(a.world_to_tcp(x, y, PAPER_GRIP_HEIGHT))
    a.grip() # grip paper

def flip_paper(workspace: Workspace,
               arm: str = "right") -> None:
    """Flip the paper held in the gripper by rotating the wrist joint 180°.

    Lifts the arm to clearance, reorients the tooltip straight down, rotates
    wrist joint 5 by π radians, then reorients the tooltip forward to leave
    the arm in a sideways approach posture.

    Parameters
    ----------
    workspace : Workspace
    arm : {'left', 'right'}, optional
        Which arm performs the flip.  Default ``'right'``.
    """

    a = workspace.arm(arm)
    # move arm out of the way for flipping
    a.move_offset_world(0, 0, config.FLIP_PAPER_CLEARANCE)

    current_tcp = a.current_tcp_pose()
    tool_down = ToolOrientation.point_tooltip_preserve_gripper_orientation(current_tcp, "down")
    tool_down_rot_vec = tool_down.to_rot_vec()
    new_tcp = current_tcp[:3] + tool_down_rot_vec
    print(f"tool_down_rot_vec: {tool_down_rot_vec}")
    print(f"type(tool_down_rot_vec): {type(tool_down_rot_vec)}")
    a.move_to_tcp(new_tcp)
    a.rotate_joint(5, math.pi) # rotate the wrist joint by 180 degrees to flip the paper
    current_tcp = a.current_tcp_pose()
    tool_up_same_gripper_orientation = ToolOrientation.point_tooltip_preserve_gripper_orientation(current_tcp, "forward")
    print(f"tool_up_same_gripper_orientation: {tool_up_same_gripper_orientation}")
    new_tcp = new_tcp[:3] + tool_up_same_gripper_orientation.to_rot_vec()
    a.move_to_tcp(new_tcp)

    # put the paper back
    a.move_offset_world(0, 0, -config.FLIP_PAPER_CLEARANCE)


# ===========================================================================
# Fold actions
# ===========================================================================

def fold_arc(
    workspace: Workspace,
    arm_side: str,
    end_pos: list[float, float, float],
    n_steps: int = 8,
    fold_percent: float = 1.0
) -> None:
    """Fold paper by sweeping the gripped edge through a semicircular arc along the fold axis defined by 
    the vector from start to end positions of the fold.  

    Call this immediately after `grip_paper`; the arm's current position is
    taken as the arc start point.  The gripper sweeps through a true semicircle
    of radius `|radius|` in the plane of the fold axis.



    Wrist (joint 5) rotates by π / n_steps at each step so the gripper stays
    parallel to the paper throughout the fold.

    Sequence per step: rotate wrist + moveL to next arc position blended together.

    Parameters
    ----------
    workspace : Workspace
    arm_side : {'left', 'right'}
        Which arm performs the fold.
    end_pos : list[float, float, float]
        World position of the gripped edge at the end of the fold (metres). This together with current arm position define the fold axis.
    n_steps : int
        Number of waypoints along the arc, including the end point.  More
        waypoints produce a smoother fold at the cost of longer execution
        time.  Default 8.
    """

    # select arm
    arm = workspace.arm(arm_side)

    # set the starting position
    x1, y1, z1 = list(arm.current_tcp_pose())[:3]
    x2, y2 = list(arm.world_to_tcp(*end_pos))[:2]
    radius = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)/2
    rotation_vector = np.cross([x2 - x1, y2 - y1, 0], [0, 0, 1])
    rotation_vector /= np.linalg.norm(rotation_vector)

    poses = list()
    for i in range(1, n_steps+1):
        # calculate x, y, z on half-circle
        theta = math.pi * i / n_steps
        # interpolate x and y between start and end positions using cosine interpolation
        c = (math.cos(theta) + 1) / 2
        x = x1 * c + x2 * (1 - c)
        y = y1 * c + y2 * (1 - c)
        z = math.sin(theta) * radius + z1

        if i / n_steps > fold_percent:
            rx, ry, rz = poses[-1][3:]  # keep the last rotation vector if we are beyond the fold percent
        else:
            # otherwise, calculate rotation vector
            rx, ry, rz = compose_rotation_vectors(arm.current_tcp_pose()[3:], (rotation_vector * i * -math.pi/n_steps))

        x, y, z, rx, ry, rz = float(x), float(y), float(z), float(rx), float(ry), float(rz)
        poses.append([x, y, z, rx, ry, rz])

    arm.backend.move_linear_poses(poses, speed=0.1, acceleration=0.1)

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
               arm: str = "left") -> None:
    """Grip the creaser tool by approaching horizontally from outside.

    Transits to the approach point at clearance height, reorients the tooltip
    sideways along ``grip_angle``, descends to ``z``, slides in to ``(x, y)``
    and closes the gripper, then lifts the tool clear and returns home.

    Parameters
    ----------
    workspace : Workspace
    x, y, z : float
        Position of the creaser tool handle (metres).
    grip_angle : float
        Direction the arm approaches from, as a rotation about the board normal
        (radians).  The arm starts ``PAPER_APPROACH_OFFSET`` outside in the
        ``(-cos(grip_angle), -sin(grip_angle))`` direction and slides in.
        Use ``0`` to approach from the left (``-x``), ``π/2`` from below
        (``-y``), etc.
    arm : {'left', 'right'}, optional
        Which arm performs the grip.  Default ``'right'``.
    """
    a = workspace.arm(arm)
    x_start = x - PAPER_APPROACH_OFFSET * math.cos(grip_angle)
    y_start = y - PAPER_APPROACH_OFFSET * math.sin(grip_angle)
    z_start = z

    # Step 1: transit to approach start, preserving current orientation.
    a.move_to_clearance(x_start, y_start)

    # rotate the gripper to point right and flat (so it can grip the creaser tool)
    right_rotvec = ToolOrientation.from_labels(tooltip="right", gripper="inward").to_rot_vec()
    a.rotate_absolute(*right_rotvec)

    # Step 2: reorient to sideways at clearance height.
    a.move_to_tcp(a.world_to_tcp(x_start, y_start, a.config.clearance_z))
    # Step 3: descend to grip height.
    a.move_to_tcp(a.world_to_tcp(x_start, y_start, z_start))
    a.goto(config.CREASER_GRIP_OPEN_POS)
    # Slide horizontally in to grip point.
    a.move_to_tcp(a.world_to_tcp(x, y, z))
    a.grip()
    a.move_offset_world(-0.07, 0, 0) # move back a bit to remove the creaser tool from the holder
    a.move_offset_world(0, 0, 0.05) # move up a bit to lift up the creaser tool from the platform
    a.go_home()

def return_creaser_tool(workspace: Workspace, x: float, y: float, z: float, grip_angle: float,
               arm: str = "right") -> None:
    """Return the creaser tool to its tray by approaching horizontally and releasing.

    Transits to the approach point at clearance height, reorients the tooltip
    sideways along ``grip_angle``, descends to ``z``, slides in to ``(x, y)``,
    opens the gripper to release the tool, then retreats and returns home.

    Parameters
    ----------
    workspace : Workspace
    x, y, z : float
        Position of the creaser tool tray slot (metres).
    grip_angle : float
        Direction the arm approaches from, as a rotation about the board normal
        (radians).  The arm starts ``PAPER_APPROACH_OFFSET`` outside in the
        ``(-cos(grip_angle), -sin(grip_angle))`` direction and slides in.
        Use ``0`` to approach from the left (``-x``), ``π/2`` from below
        (``-y``), etc.
    arm : {'left', 'right'}, optional
        Which arm performs the return.  Default ``'right'``.
    """
    a = workspace.arm(arm)
    x_start = x - PAPER_APPROACH_OFFSET * math.cos(grip_angle)
    y_start = y - PAPER_APPROACH_OFFSET * math.sin(grip_angle)
    z_start = z

    # Step 1: transit to approach start, preserving current orientation.
    a.move_to_clearance(x_start, y_start)

    right_rotvec = ToolOrientation.from_labels(tooltip="right", gripper="outward").to_rot_vec()
    a.rotate_absolute(*right_rotvec)

    # Step 2: reorient to sideways at clearance height.
    a.move_to_tcp(a.world_to_tcp(x_start, y_start, a.config.clearance_z))
    # Step 3: descend to grip height.
    a.move_to_tcp(a.world_to_tcp(x_start, y_start, z_start))
    # Slide horizontally in to the paper edge.
    a.move_to_tcp(a.world_to_tcp(x, y, z))
    a.goto(config.CREASER_GRIP_OPEN_POS, blocking=True)
    a.move_offset_world(-0.1, 0, 0) # move back a bit to put down the creaser tool properly
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
    
    crease_x, crease_y, crease_z = config.CREASER_POS
    grip_crease_tool(workspace, crease_x, crease_y, crease_z, grip_angle=0, arm=arm_side)


    clearance_offset = 0.103
    crease_height = 6.5/100# * math.sin(math.pi/4) # safe height to


    # rotate tool tip by 45 degrees in direction of crease
    # if axis == 'x':
    #     arm.rotate_joint(4, math.pi/4)
    # else:
    #     arm.rotate_joint(4, -math.pi/4)
    middle_magnet_width = 7/100
    
    arm.move_to_world(start_x, start_y, crease_height+clearance_offset)
    if axis == 'y':
        arm.rotate_joint(5, math.pi/2)
    # move along the crease line for a given length
    if axis == 'x':
        arm.move_offset_world(middle_magnet_width/2, 0, 0) # move past the of the magnet to avoid collisions
        arm.move_offset_world(0,0,-clearance_offset)
        arm.move_offset_world((crease_length-middle_magnet_width)/2, 0, 0)
        arm.move_offset_world(0,0,clearance_offset)
        arm.move_offset_world(-(crease_length/2+middle_magnet_width), 0, 0)
        arm.move_offset_world(0,0,-clearance_offset)
        arm.move_offset_world(-(crease_length-middle_magnet_width)/2, 0, 0)
    else:
        arm.move_offset_world(0, middle_magnet_width/2, 0) # move past the of the magnet to avoid collisions
        arm.move_offset_world(0,0,-clearance_offset)
        arm.move_offset_world(0, (crease_length-middle_magnet_width)/2, 0)
        arm.move_offset_world(0,0,clearance_offset)
        arm.move_offset_world(0, -(crease_length/2+middle_magnet_width), 0)
        arm.move_offset_world(0,0,-clearance_offset)
        arm.move_offset_world(0, -(crease_length-middle_magnet_width)/2, 0)

    return_creaser_tool(workspace, crease_x, crease_y, crease_z, grip_angle=0, arm=arm_side)

