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

from .magnets import Magnet
from .workspace import Workspace
from .rotation import ArmOrientation, TooltipDirection, GripperOrientation, compose_rotation_vectors

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

def _grip_magnet_at(arm, x: float, y: float, z: float, orientation: float = 0.0) -> None:
    """Approach a magnet from above, descend to its grip height, close, retreat."""
    # point via the magnet's saved internal orientation
    magnet_orientation = ArmOrientation.from_tcp_pose(arm.current_tcp_pose()).gripper_orientation(GripperOrientation.FLAT).rotate_gripper(orientation)
    # rotate gripper to pick up magnet at its current orientation so that it can be gripped properly
    arm.rotate_absolute(magnet_orientation)
    print(f"Gripping magnet at ({x:.3f}, {y:.3f}, {z:.3f}) with orientation {orientation:.2f} rad")
    clearance = z + MAGNET_APPROACH_CLEARANCE
    arm.move_to_world(x, y, clearance)
    arm.goto(MAGNET_GRIP_OPEN_POS, blocking=True)
    arm.move_to_world(x, y, z)
    arm.goto(MAGNET_GRIP_CLOSE_POS, blocking=True)
    arm.move_to_world(x, y, clearance)


def _release_magnet_at(arm, x: float, y: float, z: float, orientation: float = 0.0) -> None:
    """Carry to the magnet grip point, descend to release height, open, retreat.
    Parameters
    ----------
    arm : Arm
        Arm to use for the release.
    x, y, z : float
        World coordinates of the magnet grip point (metres).
    """
    # rotate the arm to the desired orientation so that it can be released properly
    new_magnet_orientation = ArmOrientation.from_tcp_pose(arm.current_tcp_pose()).gripper_orientation(GripperOrientation.FLAT).rotate_gripper(orientation)
    arm.rotate_absolute(new_magnet_orientation)

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
        pick_up_point = np.asarray(magnet.get_grip_xy(), dtype=float)
        _grip_magnet_at(arm, float(pick_up_point[0]), float(pick_up_point[1]),
                        float(magnet.tray_position[2]) + magnet.grip_height, magnet.orientation) # The magnet's orientation is 0 at the tray, but for conformity we should still use the magnet's saved orientation here in case it is changed in the future.

    # update the magnet state first so its derived grip point reflects the new anchor,
    # then extract that updated grip point for the release move
    grip_x, grip_y = magnet.place_at(x, y, orientation).get_grip_xy()
    # now that magnet state is updated we can use that value directly from the magnet.
    _release_magnet_at(arm, grip_x, grip_y, magnet.grip_height, magnet.orientation)
    if magnet.identifier not in workspace.magnets:
        workspace.magnets.add(magnet)

    neutral_grip = ArmOrientation.from_tcp_pose(arm.current_tcp_pose()).gripper_orientation(GripperOrientation.FLAT)
    # rotate gripper back to neutral orientation after placing the magnet
    arm.rotate_absolute(neutral_grip) 

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
    if not magnet.placed:
        print(f"ERROR (non-fatal): Magnet {magnet.identifier} tried to remove while not placed. Ignoring command.")
        return
    grip = magnet.get_grip_xy()

    # rotate the arm to the magnet's orientation so that it can be gripped properly
    magnet_orientation = ArmOrientation.from_tcp_pose(arm.current_tcp_pose()).gripper_orientation(GripperOrientation.FLAT).rotate_gripper(magnet.orientation)
    arm.rotate_absolute(magnet_orientation)
    _grip_magnet_at(arm, float(grip[0]), float(grip[1]), magnet.grip_height)

    # update the magnet state first so its derived grip point reflects the tray position,
    # then extract that updated grip point for the release move
    home = magnet.stow().get_grip_xy()

    _release_magnet_at(arm, float(home[0]), float(home[1]),
                        float(magnet.tray_position[2]) + magnet.grip_height, 0.0)
    
    # rotate the arm to the magnet's orientation so that it can be released properly (at home it has orientation 0)
    default_magnet_orientation = ArmOrientation.from_tcp_pose(arm.current_tcp_pose()).gripper_orientation(GripperOrientation.FLAT)
    arm.rotate_absolute(default_magnet_orientation)



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
    grip = magnet.get_grip_xy()
    
    _grip_magnet_at(arm, float(grip[0]), float(grip[1]), magnet.grip_height, orientation)

    # update magnet state to where its going to be placed and extract those (this takes orientation into consideration)
    grip_x, grip_y = magnet.place_at(x, y, orientation).get_grip_xy()
    # now that magnet state is updated we can use that value directly from the magnet.
    _release_magnet_at(arm, grip_x, grip_y, magnet.grip_height, magnet.orientation)

    neutral_grip = ArmOrientation.from_tcp_pose(arm.current_tcp_pose()).gripper_orientation(GripperOrientation.FLAT)
    # rotate gripper back to neutral orientation after placing the magnet
    arm.rotate_absolute(neutral_grip)

def grip_magnet(workspace: Workspace, identifier: str, carrying_arm: str = "left") -> None:
    """Grip a placed magnet and hold it in the gripper.

    Parameters
    ----------
    workspace : Workspace
    identifier : str
        Identifier of the magnet to grip.
    carrying_arm : {'left', 'right'}, optional
    """
    arm = workspace.arm(carrying_arm)
    magnet = workspace.magnets.get(identifier)
    if not magnet.placed:
        raise Exception(f"ERROR (non-fatal): Magnet {magnet.identifier} tried to grip while not placed. Ignoring command.")

    grip = magnet.get_grip_xy()
    _grip_magnet_at(arm, float(grip[0]), float(grip[1]), magnet.grip_height)

    neutral_grip = ArmOrientation.from_tcp_pose(arm.current_tcp_pose()).gripper_orientation(GripperOrientation.FLAT)
    # rotate gripper back to neutral orientation after placing the magnet
    arm.rotate_absolute(neutral_grip)

def release_magnet(workspace: Workspace, identifier: str, x: float, y: float,
                   orientation: float = 0.0, carrying_arm: str = "left") -> None:
    """Release a gripped magnet at a specified board position. 

    Note: requires that the magnet is already gripped in the arm's gripper.  If it is not, use `move_magnet` instead!.

    Parameters
    ----------
    workspace : Workspace
    identifier : str
        Identifier of the magnet to release.
    x, y : float
        Board position to release the magnet (metres).
    orientation : float, optional
        Yaw of the magnet (radians).  Default 0.
    carrying_arm : {'left', 'right'}, optional
    """
    arm = workspace.arm(carrying_arm)
    magnet = workspace.magnets.get(identifier)
    
    # update magnet state to where its going to be placed and extract those (this takes orientation into consideration)
    grip_x, grip_y = magnet.place_at(x, y, orientation).get_grip_xy()
    _release_magnet_at(arm, grip_x, grip_y, magnet.grip_height, orientation)

    neutral_grip = ArmOrientation.from_tcp_pose(arm.current_tcp_pose()).gripper_orientation(GripperOrientation.FLAT)
    # rotate gripper back to neutral orientation after releasing the magnet
    arm.rotate_absolute(neutral_grip)


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

    # rotate the gripper to point forward and flat (so it can grip the paper) so that it is facing the wall in an easy to start orientation
    # forward_rotvec = ArmOrientation.from_directions(tooltip_direction=TooltipDirection.FORWARD, gripper_orientation=GripperOrientation.FLAT).to_rotvec()
    forward_orientation = ArmOrientation.from_directions(tooltip_direction=TooltipDirection.FORWARD, gripper_orientation=GripperOrientation.FLAT)


    # rotate the gripper to point in the direction of the grip angle so that it can approach the paper edge at the correct angle
    # grip_angle_oriented_rotvec = compose_rotation_vectors(forward_rotvec, [0, 0, grip_angle])
    # a.rotate_absolute(grip_angle_oriented_rotvec)
    
    # based on right hand rule since tooltip (index finger) = forward (-x base dir), gripper (middle finger)=flat (in this case pointing left = +y base dir) then rotation axis/thumb = +z base dir with
    # positive rotation angle being left, so in rotvec case pos degree is to the left so we do the same here.
    grip_angle_oriented_orientation = forward_orientation.tilt_tooltip(direction=TooltipDirection.LEFT, degrees=math.degrees(grip_angle))
    a.rotate_absolute(grip_angle_oriented_orientation)

    a.move_to_tcp(a.world_to_tcp(x_start, y_start, PAPER_GRIP_HEIGHT)) # move to paper grip height at the approach point
    a.goto(.5) # open the gripper to prepare to grip the paper
    # Slide horizontally in to the paper edge.
    a.move_to_tcp(a.world_to_tcp(x, y, PAPER_GRIP_HEIGHT))
    a.grip() # grip paper

def flip_paper(workspace: Workspace,
               arm: str = "right") -> None:
    """Flip the paper **currently held in the gripper** by 180° 
    
    This function lifts the arm up to safe clearance height, reorients the arm/tooltip to point straight down (in order to avoid 
    gravity deforming the paper while we flip it / causing it to fold down into itself weirdly) and rotates wrist 3 / the wrist joint (gripper) 
    by 180° before finally reorienting the arm back to horizontal/sideways approach orientation - with the gripper orientation preserved
    from the previous step (rotating wrist joint) in order for paper to remain flipped 

    Parameters
    ----------
    workspace : Workspace
    arm : {'left', 'right'}, optional
        Which arm currently grips the paper and performs the flip.  Default ``'right'``.
    """

    a = workspace.arm(arm)
    # lift arm holding paper to clearance height in order to allow tilting arm down
    a.move_offset_world(0, 0, config.FLIP_PAPER_CLEARANCE)

    current_tcp = a.current_tcp_pose()

    # rotate the arm to point downwards (so that gravity doesn't deform the paper while we flip it)
    point_down = ArmOrientation.from_tcp_pose(current_tcp).tooltip_direction(TooltipDirection.DOWN)
    a.rotate_absolute(point_down)

    # now that the arm is pointing downwards, we can rotate the wrist joint to flip the paper
    rotate_wrist = point_down.rotate_gripper(180)
    a.rotate_absolute(rotate_wrist) # rotate the wrist joint by 180 degrees to flip the paper
    #a.rotate_joint(5, math.pi) # rotate the wrist joint by 180 degrees to flip the paper

    # now that the paper is flipped, we can reorient the arm back to horizontal/sideways approach orientation - with the gripper orientation preserved from the previous step (rotating wrist joint) in order for paper to remain flipped
    point_forwards = rotate_wrist.tooltip_direction(TooltipDirection.FORWARD)
    a.rotate_absolute(point_forwards)

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
    of radius `||arm.current_world_pos() - end_pos||/2` in the plane of the fold axis.



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

    poses = [arm.current_tcp_pose()]
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

    arm.backend.move_linear_poses(poses, speed=0.5, acceleration=0.5)
    return poses

def unfold_arc(
    workspace: Workspace,
    arm_side: str,
    poses: list[list[float,float,float,float,float,float]]
) -> None:
    """Unfold paper by sweeping the gripped edge through a circular arc about a fold line.
    """
    arm = workspace.arm(arm_side)
    arm.backend.move_linear_poses(poses[::-1], speed=0.5, acceleration=0.5)
    

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
    right_orientation = ArmOrientation.from_directions(tooltip_direction=TooltipDirection.RIGHT, gripper_orientation=GripperOrientation.FLAT).to_rotvec()
    a.rotate_absolute(right_orientation)

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

    right_orientation = ArmOrientation.from_directions(tooltip_direction=TooltipDirection.RIGHT, gripper_orientation=GripperOrientation.FLAT).to_rotvec()
    a.rotate_absolute(right_orientation)

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


# TODO: Make crease function generic similar to fold arc
# given two points drops onto the first and creases to the second
# holding the creaser perpendicular to the movement direction
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

