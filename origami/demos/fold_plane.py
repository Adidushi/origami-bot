"""
Usage
-----
    python3 origami/demos/fold_plane.py                # real arms
    python3 origami/demos/fold_plane.py --simulation   # simulation
"""
from __future__ import annotations

import argparse
import math
import os
import sys


# Allow running directly: python3 origami/demos/demo_get.py
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from origami.arm import ArmConfig
from origami.magnets import BlockMagnet, LBracketMagnet
from origami import Paper, Workspace, actions, config
from origami.config import CM

GO_MODE = False

def safe_input(prompt: str) -> None:
    """A safe version of input() that works in environments where stdin may not be available."""
    if not GO_MODE:
        input(prompt)
    else:
        print(prompt + " (skipped due to GO_MODE)")

# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="demo_get — paper edge grip and fold")
    parser.add_argument("--speed", type=float, default=1.0,
                        help="global multiplier applied to every commanded speed (default: 1.0)")
    parser.add_argument("--go", action="store_true",
                        help="skip all input prompts (GO_MODE)") 
    args = parser.parse_args()

    global GO_MODE
    GO_MODE = args.go
    config.SPEED_SCALE = args.speed

    # ---------------------------------------------------------------------------
    # Initialization
    # ---------------------------------------------------------------------------
    arm_configs = [ArmConfig(home=config.LEFT_ARM_START_JOINTS, away=config.LEFT_ARM_AWAY_JOINTS), ArmConfig(home=config.RIGHT_ARM_START_JOINTS, away=config.RIGHT_ARM_AWAY_JOINTS)]
    ws = Workspace.hardware(arm_configs=arm_configs, home=True)

    ws.left.grip()
    ws.right.grip()
    
    # Centre an A4 sheet on the board.  A4 is 297 mm tall on a 270 mm board,
    # giving a natural 13.5 mm overhang on both the top and bottom edges.
    paper_bottom_left_corner_x = config.BOARD_WIDTH  / 2 - config.PAPER_WIDTH  / 2
    paper_bottom_edge_y = config.BOARD_HEIGHT / 2 - config.PAPER_HEIGHT / 2   # < 0
    ws.paper = Paper.rectangle(config.PAPER_WIDTH, config.PAPER_HEIGHT,
                               origin=(paper_bottom_left_corner_x, paper_bottom_edge_y))
    paper_bottom_right_corner_x = paper_bottom_left_corner_x + config.PAPER_WIDTH
    # Grip the bottom-right corner of the paper from the −y side (below board)
    print()
    print(f"paper bottom-left corner   ({paper_bottom_left_corner_x:.4f}, {paper_bottom_edge_y:.4f})")
    print(f"paper bottom-right corner  ({paper_bottom_right_corner_x:.4f}, {paper_bottom_edge_y:.4f})")
    # ---------------------------------------------------------------------------
    # Step 1 — place initial magnets
    # ---------------------------------------------------------------------------
    print("[Step 1] Place initial magnets")
    safe_input("Proceed with step 1? (press Enter to continue)")
    block_a = BlockMagnet(
        identifier="block_a",
        handle_height=1.5 * CM,
        tray_position=(config.MAGNET_PLATFORM_POSITIONS["bottom_right"][0]-1 * CM, config.MAGNET_PLATFORM_POSITIONS["bottom_right"][1]+11.7 * CM, config.MAGNET_PLATFORM_POSITIONS["bottom_right"][2]),
    )

    block_b = BlockMagnet(
        identifier="block_b",
        handle_height=1.5 * CM,
        tray_position=(config.MAGNET_PLATFORM_POSITIONS["bottom_right"][0]-1 * CM-5 * CM, config.MAGNET_PLATFORM_POSITIONS["bottom_right"][1]+11.7 * CM, config.MAGNET_PLATFORM_POSITIONS["bottom_right"][2]),
    )

    lbracket_a = LBracketMagnet(
        identifier="lbracket_a",
        handle_height=0.3 * CM,
        handle_offset=19.7 * CM, 
        orientation = 0,
        tray_position=(config.MAGNET_PLATFORM_POSITIONS["bottom_right"][0]-0.7 * CM, config.MAGNET_PLATFORM_POSITIONS["bottom_right"][1]+3.55 * CM, config.MAGNET_PLATFORM_POSITIONS["bottom_right"][2]),
    )
    
    actions.place_magnet(ws.left, block_a, x=0.275, y=0.10)
    actions.place_magnet(ws.left, block_b, x=0.265, y=0.225)
    ws.left.go_home()

    # ---------------------------------------------------------------------------
    # Step 2 — grip paper edge
    # ---------------------------------------------------------------------------
    print("[Step 2] grab paper for first fold — grip paper edge")
    safe_input("Proceed with step 2? (press Enter to continue)")
    # Grip the paper edge with the right arm (sideways horizontal approach)
    actions.grip_paper(
        arm=ws.right,
        x=paper_bottom_left_corner_x + 1/100,  # approach from just beyond the left edge of the paper
        y=paper_bottom_edge_y + 0.3/100,  # approach from just below the bottom edge of the paper
        grip_angle=0
    )

    # ---------------------------------------------------------------------------
    # Step 3 — fold paper in half  
    # ---------------------------------------------------------------------------
    print("[Step 3] fold paper in half")
    safe_input("Proceed with step 3? (press Enter to continue)")
    # Fold axis at the board centre: folds the right half of the paper over.
    # Radius = grip_x - fold_axis_x ≈ 9.5 cm, matching get.py's radius value.
    end_pos = list(ws.right.current_world_pos())
    end_pos[0] += 2*9.5/100
    poses = actions.fold_arc(
        arm=ws.right,
        end_pos=end_pos,
        n_steps=8,
    )
    # ---------------------------------------------------------------------------
    # Step 4 — place l-bracket magnet to hold the fold  
    # ---------------------------------------------------------------------------
    print("[Step 4] place l-bracket magnet to hold the fold")
    safe_input("Proceed with step 4? (press Enter to continue)")

    # paper is placed s.t. its top edge is aligned with top of board, but since their sizes differ, to get to middle of paper we need to move down by paper's height from top of board, which is not the same as half of board's height
    actions.place_magnet(ws.left, lbracket_a, x=config.BOARD_WIDTH/2+3.5/100, y=config.BOARD_HEIGHT-config.PAPER_HEIGHT/2)
    ws.left.go_home()
    
    # ---------------------------------------------------------------------------
    # Step 5 — crease paper
    # ---------------------------------------------------------------------------
    print("[Step 5] crease paper")
    safe_input("Proceed with step 5? (press Enter to continue)")

    actions.crease_multiple(
        arm=ws.left,
        position_pairs=[
            (
                [config.BOARD_WIDTH/2, paper_bottom_edge_y+config.PAPER_HEIGHT/2+3.5*CM, config.CREASE_HEIGHT],
                [config.BOARD_WIDTH/2, paper_bottom_edge_y+config.PAPER_HEIGHT, config.CREASE_HEIGHT],
            ),
            (
                [config.BOARD_WIDTH/2, paper_bottom_edge_y+config.PAPER_HEIGHT/2-6*CM, config.CREASE_HEIGHT],
                [config.BOARD_WIDTH/2, paper_bottom_edge_y, config.CREASE_HEIGHT],
            )
        ],
        angle=math.radians(0)
    )
    
    # ---------------------------------------------------------------------------
    # Step 6 — remove l-bracket magnet
    # ---------------------------------------------------------------------------
    print("[Step 6] remove l-bracket magnet")
    safe_input("Proceed with step 6? (press Enter to continue)")
    actions.remove_magnet(ws.left, lbracket_a)
    ws.left.go_home()
    # ---------------------------------------------------------------------------
    # Step 7 — unfold paper
    # ---------------------------------------------------------------------------
    print("[Step 7] unfold paper")
    safe_input("Proceed with step 7? (press Enter to continue)")
    actions.unfold_arc(
        arm = ws.right,
        poses=poses
    )

    ws.right.release()
    ws.right.move_offset_world(0,-2/100,0)
    ws.right.go_home()

    # ---------------------------------------------------------------------------
    # Step 8 — remove placed magnets
    # ---------------------------------------------------------------------------
    print("[Step 8] remove placed magnets")
    safe_input("Proceed with step 8? (press Enter to continue)")
    actions.remove_magnet(ws.left, block_a)
    actions.remove_magnet(ws.left, block_b)
    ws.left.go_home()
    
    actions.grip_paper(
        arm=ws.right,
        x=config.BOARD_WIDTH/2,  # approach from just beyond the left edge of the paper
        y=paper_bottom_edge_y + 0.3/100,  # approach from just below the bottom edge of the paper
        grip_angle=0
    )

    # ---------------------------------------------------------------------------
    # Step 9 — flip paper over
    # ---------------------------------------------------------------------------
    print("[Step 9] flip paper over")
    safe_input("Proceed with step 9? (press Enter to continue)")
    actions.flip_paper(arm=ws.right)

    # let go of paper, move back and go home
    ws.right.release()
    ws.right.move_offset_world(0,-2/100,0)
    ws.right.go_home()

    
    
    # ---------------------------------------------------------------------------
    # Step 10 — grab paper for second fold — grip right paper edge
    # ---------------------------------------------------------------------------
    print("[Step 10] grab paper for second fold — grip right paper edge")
    safe_input("Proceed with step 10? (press Enter to continue)")
    actions.place_magnet(
        ws.left,
        block_a,
        x=config.BOARD_WIDTH/2+8/100,
        y=15/100
    )

    actions.place_magnet(
        ws.left,
        block_b,
        x=config.BOARD_WIDTH/2+8/100,
        y=20/100
    )

    ws.left.go_home()

    # Grip the paper edge with the right arm (sideways horizontal approach)
    actions.grip_paper(
        arm=ws.right,
        x=paper_bottom_right_corner_x-0.3*CM,  # approach from just beyond the right edge of the paper
        y=paper_bottom_edge_y-0.3*CM,  # approach from just below the bottom edge of the paper
        grip_angle=math.pi / 4
    )

    # ---------------------------------------------------------------------------
    # Step 11 — fold paper from right corner to middle
    # ---------------------------------------------------------------------------
    print("[Step 11] fold paper from right corner to middle")
    safe_input("Proceed with step 11? (press Enter to continue)")
    # Fold the bottom-right corner up to the middle.
    # end position x is the middle of the board, while the end position y
    # is calculated to be equal in difference to the x difference
    # so that we make a 45 degree fold (square kinda thing)
    end_pos = list(ws.right.current_world_pos())
    end_pos[0] = config.BOARD_WIDTH/2+2/100
    diff = abs(ws.right.current_world_pos()[0] - end_pos[0])
    end_pos[1] += diff
    actions.fold_arc(
        arm=ws.right,
        end_pos=end_pos,
        n_steps=8,
        fold_percent=5/8
    )

    # place the corner folding magnet (long boy)
    actions.place_magnet(ws.left, lbracket_a, x=config.BOARD_WIDTH/2+1/100, y=2/100)
    ws.left.go_home()

    # Open the hand, release the paper and go home (post-fold)
    ws.right.goto(config.CREASER_GRIP_OPEN_POS)
    ws.right.move_offset_world(0, 0, 5/100)
    ws.right.move_away()

    actions.move_magnet(
        ws.left,
        block_b,
        x=config.BOARD_WIDTH/2+1*CM,
        y=8.5/100,
    )

    actions.move_magnet(
        ws.left,
        lbracket_a,
        x=config.BOARD_WIDTH-config.PAPER_WIDTH/2-16*CM,
        y=4*CM
    )
    
    start_pos = [config.BOARD_WIDTH/2+config.PAPER_WIDTH/2-0.5*CM, paper_bottom_edge_y+config.PAPER_WIDTH/2-0.5*CM, config.CREASE_HEIGHT-0.1*CM]
    end_pos = [config.BOARD_WIDTH/2, paper_bottom_edge_y, config.CREASE_HEIGHT-0.1*CM]
    mid_pos = [(start_pos[0]+end_pos[0])/2, (start_pos[1]+end_pos[1])/2, (start_pos[2]+end_pos[2])/2]

    actions.crease_multiple(
        arm=ws.left,
        position_pairs=[
            (mid_pos, start_pos),
            (mid_pos, end_pos)
        ],
        angle=math.radians(0)
    )
    
    actions.remove_magnet(ws.left, block_a)
    actions.remove_magnet(ws.left, block_b)
    actions.remove_magnet(ws.left, lbracket_a)
    ws.left.go_home()
    ws.right.go_home()
    
    # # --------------------------------------------------------------------------- #
    # # Step 11.x - shift paper over in prep for next fold
    # # --------------------------------------------------------------------------- #
    print("[Step 11.1] Grab paper to shift to the right")
    safe_input("Proceed with step 11.1? (press Enter to continue)")
    actions.grip_paper(
            arm=ws.right,
            x=config.BOARD_WIDTH/2,  # approach from just beyond the left edge of the paper
            y=paper_bottom_edge_y + 0.3/100,  # approach from just below the bottom edge of the paper
            grip_angle=0
        )

    print("[Step 11.2] Shift paper to right side of board")
    safe_input("Proceed with step 11.2? (press Enter to continue)")
    # move up, then sideways, then down
    actions.move_paper(
        arm=ws.right,
        x=config.BOARD_WIDTH-config.PAPER_WIDTH/2,
        y=paper_bottom_edge_y,
        orientation=math.pi/2
    )

    # let go of the page, move back and go home
    ws.right.release()
    ws.right.move_offset_world(0,-2/100,0)
    ws.right.go_home()
    ws.right.grip()
    
    
    print("[Step 11.3] Place block magnets in preparation for next fold")
    safe_input("Proceed with step 11.3? (press Enter to continue)")
    actions.place_magnet(
            ws.left,
            block_b,
            x=15/100+(config.BOARD_WIDTH-config.PAPER_WIDTH),
            y=8/100
        )
        
    actions.place_magnet(ws.left, block_a, x=0.275+6/100, y=0.14)

    ws.left.go_home()


    # ---------------------------------------------------------------------------
    # Step 12 — grab paper for third fold — grip bottom-left paper edge
    # ---------------------------------------------------------------------------
    print("[Step 12] grab paper for third fold — grip bottom-left paper edge")
    safe_input("Proceed with step 12? (press Enter to continue)")
    # Grip the paper edge with the left arm (sideways horizontal approach)
    actions.grip_paper(
        arm=ws.right,
        x=config.BOARD_WIDTH-config.PAPER_WIDTH+1/100,  # approach from just beyond the left edge of the paper
        y=paper_bottom_edge_y,  # approach from just below the bottom edge of the paper
        grip_angle= - math.pi / 4
    )

    # ---------------------------------------------------------------------------
    # Step 13 — fold paper from bottom-left corner to middle
    # ---------------------------------------------------------------------------
    print("[Step 13] fold paper from bottom-left corner to middle")
    safe_input("Proceed with step 13? (press Enter to continue)")

    # Fold axis at the board centre: folds the right half of the paper over.
    # Radius = grip_x - fold_axis_x ≈ 9.5 cm, matching get.py's radius value.
    end_pos = list(ws.right.current_world_pos())
    end_pos[0] = config.BOARD_WIDTH-config.PAPER_WIDTH/2-2/100
    diff = abs(ws.right.current_world_pos()[0] - end_pos[0])
    end_pos[1] += diff
    actions.fold_arc(
        arm=ws.right,
        end_pos=end_pos,
        n_steps=8,
        fold_percent=5/8
    )

    # ----------------------------------------------------------------------------
    # Step 14 — place l-bracket magnet to hold the fold and go home
    # ----------------------------------------------------------------------------
    print("[Step 14] place l-bracket magnet to hold the fold")
    safe_input("Proceed with step 14? (press Enter to continue)")
    # place the corner folding magnet (long boy)
    actions.place_magnet(
        ws.left,
        lbracket_a, 
        x=config.BOARD_WIDTH - 11*CM, 
        y=1*CM, 
        orientation=math.radians(45)
    )
    ws.left.go_home()

    # Open the hand, release the paper and go home (post-fold)
    ws.right.goto(config.CREASER_GRIP_OPEN_POS)
    ws.right.move_offset_world(0, 0, 5/100)
    ws.right.move_away()
    ws.right.grip()

    actions.move_magnet(
        ws.left,
        block_b,
        x=config.BOARD_WIDTH-config.PAPER_WIDTH/2-1*CM,
        y=8.5/100,
    )

    actions.move_magnet(
        ws.left,
        lbracket_a,
        x=config.BOARD_WIDTH-config.PAPER_WIDTH/2-1*CM,
        y=15 * CM
    )
    
    start_pos = [config.BOARD_WIDTH-config.PAPER_WIDTH, paper_bottom_edge_y+config.PAPER_WIDTH/2, config.CREASE_HEIGHT]
    end_pos = [config.BOARD_WIDTH-config.PAPER_WIDTH/2, paper_bottom_edge_y, config.CREASE_HEIGHT]
    mid_pos = [(start_pos[0]+end_pos[0])/2, (start_pos[1]+end_pos[1])/2, (start_pos[2]+end_pos[2])/2]

    actions.crease_multiple(
        arm=ws.left,
        position_pairs=[
            (mid_pos, start_pos),
            (mid_pos, end_pos)
        ],
        angle=math.radians(0)
    )

    actions.remove_magnet(ws.left, lbracket_a)
    actions.remove_magnet(ws.left, block_a)
    actions.remove_magnet(ws.left, block_b)
    ws.left.go_home()
    
    # -------------------------------------------------------------------------------
    # Step 15 - move paper to the left side of the board, rotated s.t. airplane tip 
    # is facing to the left side of the board and the airplane slightly overhangs on the bottom
    # Releases paper and goes home.
    # -------------------------------------------------------------------------------
    print("[Step 15] move paper to the left side of the board, rotated s.t. airplane tip is facing to the left side of the board and the airplane slightly overhangs on the bottom")
    safe_input("Proceed with step 15? (press Enter to continue)")

    # move to a known good position to seed a good movement onwards
    ws.left.move_to_joints([-1.4161742369281214, -0.5880364340594788, 1.8365309874164026, -1.2560871702483674, 0.16226720809936523-2*math.pi, 1.5788087844848633])

    actions.grip_paper(
                arm=ws.left,
                x=config.BOARD_WIDTH-config.PAPER_WIDTH/2,  # approach from just beyond the left edge of the paper
                y=paper_bottom_edge_y,  # approach from just below the bottom edge of the paper
                grip_angle=0
            )
    
    # Move paper to the left side of the board, s.t. paper airplane tip is facing to the left side of the board and 
    # the airplane slightly overhangs on the bottom
    paper_airplane_tip = [-3 * CM, 8.5 * CM]
    actions.move_paper(
        arm=ws.left,
        x=paper_airplane_tip[0],
        y=paper_airplane_tip[1],
        orientation=math.radians(0)
    )


    # Open the hand, release the paper and go home (post-move)
    ws.left.goto(0)
    ws.left.move_offset_world(0, 7 * CM, 0)
    # safe_input(f'known good joints: {ws.left.get_joint_angles()}')
    ws.left.move_offset_world(0, 0, 20 * CM)
    ws.left.grip()
    ws.left.go_home()
    
    
    # --------------------------------------------------------------------------------
    # Step 16 - Place Block A magnet to hold the paper airplane in place for the next fold
    # --------------------------------------------------------------------------------
    print("[Step 16] Place Block A magnet to hold the paper airplane in place for the next fold")
    safe_input("Proceed with step 16? (press Enter to continue)")
    actions.place_magnet(
        ws.left,
        block_a, 
        x=config.BOARD_WIDTH/2-8 * CM, 
        y=config.PAPER_WIDTH/2+3 * CM, 
        orientation=0
    )

    actions.place_magnet(
        ws.left,
        block_b, 
        x=config.BOARD_WIDTH/2+3 * CM, 
        y=config.PAPER_WIDTH/2+3 * CM, 
        orientation=0
    )

    ws.left.go_home()
    ws.right.go_home()
    
    # ---------------------------------------------------------------------------
    # Step 17 - Fold right wing of paper airplane via right hand.
    # Right hand grips paper from bottom edge of board
    # ---------------------------------------------------------------------------
    print("[Step 17] Fold right wing of paper airplane via right hand. Start with gripping paper.")
    safe_input("Proceed with step 17? (press Enter to continue)")
    actions.grip_paper(
            arm=ws.right,
            x=config.BOARD_WIDTH/2, 
            y=paper_bottom_edge_y + 0.3 * CM,
            grip_angle=0
    )

    print("[Step 17.5] Fold right wing of paper airplane via right hand. Start with gripping paper.")
    safe_input("Proceed with step 17.5? (press Enter to continue)")
    # 15 CM right from paper airplane tip.
    paper_airplane_tip = [-3 * CM, 8.5 * CM]
    end_pos = [config.BOARD_WIDTH/2, paper_airplane_tip[1], 0]
    actions.fold_arc(
        arm=ws.right,
        end_pos=end_pos,
        n_steps=8,
        fold_percent=5/8
    )

    # ---------------------------------------------------------------------------
    # Step 18 - Transition LBracket magnet to hold previous fold in place 
    # ---------------------------------------------------------------------------
    print("[Step 18] Transition LBracket magnet to hold previous fold in place")
    safe_input("Proceed with step 18? (press Enter to continue)")
    actions.place_magnet(
        ws.left,
        lbracket_a, 
        x=config.BOARD_WIDTH/2-8 * CM, 
        y=config.PAPER_WIDTH/2-5 * CM, 
        orientation=0
    )

    # Now that fold is held in place, let go of paper, move back and go home
    ws.right.goto(config.CREASER_GRIP_OPEN_POS)
    ws.right.move_offset_world(0, 0, 5 * CM)
    ws.right.go_home()
    
    # ---------------------------------------------------------------------------
    # Step 19 - crease folded right wing of paper airplane via left hand.
    # ---------------------------------------------------------------------------
    print("[Step 19] crease folded right wing of paper airplane via left hand.")
    safe_input("Proceed with step 19? (press Enter to continue)")
    start_pos = [config.BOARD_WIDTH/2 - 5*CM, paper_bottom_edge_y + config.PAPER_WIDTH/4, config.CREASE_HEIGHT]
    end_pos = start_pos.copy()
    end_pos[0] += 13 * CM
    ws.right.move_away()
    actions.crease_multiple(
        arm=ws.left,
        position_pairs=[
            (start_pos, end_pos)
        ]
    )

    # ---------------------------------------------------------------------------
    # Step 20 - remove magnets and go home.
    # ---------------------------------------------------------------------------
    print("[Step 20] remove magnets and go home.")
    safe_input("Proceed with step 20? (press Enter to continue)")
    actions.remove_magnet(ws.left, block_a)
    actions.remove_magnet(ws.left, block_b)
    actions.remove_magnet(ws.left, lbracket_a)
    ws.left.go_home()

    

    # ---------------------------------------------------------------------------
    # Step 21 - turn plane 90 degrees.
    # ---------------------------------------------------------------------------
    print("[Step 21] turn plane 90 degrees.")
    safe_input("Proceed with step 21? (press Enter to continue)")

    ws.left.move_to_joints([-0.028294865285054982, -0.1579865974238892, 1.0424016157733362, -0.7840147775462647, -3.1639869848834437, 1.6709191799163818])
    ws.left.goto(.5)
    ws.left.move_offset_world(0, -7 * CM, 0)
    paper_airplane_tip = [-3 * CM, 8.5 * CM]
    actions.grip_paper(
        arm=ws.left,
        x=paper_airplane_tip[0], #config.BOARD_WIDTH-config.PAPER_WIDTH/2,  # approach from just beyond the left edge of the paper
        y=paper_airplane_tip[1], #paper_bottom_edge_y,  # approach from just below the bottom edge of the paper
        grip_angle=-math.pi/2,
        skip_clearance=True
    )
    
    # Move paper to the left side of the board, s.t. paper airplane tip is facing to the left side of the board and 
    # the airplane slightly overhangs on the bottom
    actions.move_paper(
        arm=ws.left,
        x=config.PAPER_WIDTH/2-2*CM,
        y=paper_bottom_edge_y,
        orientation=math.pi/2
    )

    ws.left.goto(config.CREASER_GRIP_OPEN_POS)
    ws.left.move_offset_world(0, -3 * CM, 0)
    ws.left.go_home()
    ws.left.grip()
    # ---------------------------------------------------------------------------
    # Step 22 - move plane to right side for second wing fold
    # ---------------------------------------------------------------------------
    print("[Step 22] move plane to right side for second wing fold.")
    safe_input("Proceed with step 22? (press Enter to continue)")
    paper_airplane_tip = [config.BOARD_WIDTH-5 * CM, 8.5 * CM]
    ws.left.move_to_joints([-0.028294865285054982, -0.1579865974238892, 1.0424016157733362, -0.7840147775462647, -3.1639869848834437, 1.6709191799163818])
    ws.left.goto(.5)
    ws.left.move_offset_world(0, -7 * CM, 0)
    ws.left.move_offset_world(2 * CM, 0, 0)
    actions.grip_paper(
        arm=ws.left,
        x=-1 * CM,
        y=8.5 * CM,
        grip_angle=-math.pi/2,
        skip_clearance=True
    )

    # Move paper to the left side of the board, s.t. paper airplane tip is facing to the left side of the board and 
    # the airplane slightly overhangs on the bottom
    actions.move_paper(
        arm=ws.left,
        x=config.BOARD_WIDTH - 9*CM,
        y=paper_bottom_edge_y,
        orientation=math.pi/2
    )

    ws.left.goto(config.CREASER_GRIP_OPEN_POS)
    ws.left.move_offset_world(0, -5 * CM, 0)
    ws.left.move_offset_world(0, 0, 20 * CM)
    ws.left.go_home()
    ws.left.grip()

    # --------------------------------------------------------------------------------
    # Step 23 - Place Block A magnet to hold the paper airplane in place for the next fold
    # --------------------------------------------------------------------------------
    print("[Step 23] Place Block A magnet to hold the paper airplane in place for the next fold")
    safe_input("Proceed with step 23? (press Enter to continue)")
    actions.place_magnet(
        ws.left,
        block_a, 
        x=config.BOARD_WIDTH/2+8 * CM, 
        y=config.PAPER_WIDTH/2, 
        orientation=math.radians(90)
    )

    actions.place_magnet(
        ws.left,
        block_b, 
        x=config.BOARD_WIDTH/2-3 * CM, 
        y=config.PAPER_WIDTH/2, 
        orientation=math.radians(90)
    )

    ws.left.go_home()
    ws.right.go_home()
    
    # ---------------------------------------------------------------------------
    # Step 24 - Fold right wing of paper airplane via right hand.
    # Right hand grips paper from bottom edge of board
    # ---------------------------------------------------------------------------
    print("[Step 24] Fold right wing of paper airplane via right hand. Start with gripping paper.")
    safe_input("Proceed with step 24? (press Enter to continue)")
    actions.grip_paper(
            arm=ws.right,
            x=config.BOARD_WIDTH/2, 
            y=paper_bottom_edge_y + 0.3 * CM,
            grip_angle=0
    )

    print("[Step 24.5] Fold right wing of paper airplane via right hand. Start with gripping paper.")
    safe_input("Proceed with step 24.5? (press Enter to continue)")
    # 15 CM right from paper airplane tip.
    paper_airplane_tip = [3 * CM, 8.5 * CM]
    end_pos = [config.BOARD_WIDTH/2, paper_airplane_tip[1], 0]
    actions.fold_arc(
        arm=ws.right,
        end_pos=end_pos,
        n_steps=8,
        fold_percent=5/8
    )

    # ---------------------------------------------------------------------------
    # Step 25 - Transition LBracket magnet to hold previous fold in place 
    # ---------------------------------------------------------------------------
    print("[Step 25] Transition LBracket magnet to hold previous fold in place")
    safe_input("Proceed with step 25? (press Enter to continue)")
    actions.place_magnet(
        ws.left,
        lbracket_a, 
        x=config.BOARD_WIDTH/2, 
        y=config.PAPER_WIDTH/2-7 * CM, 
        orientation=0
    )
    ws.left.go_home()

    # Now that fold is held in place, let go of paper, move back and go home
    ws.right.goto(config.CREASER_GRIP_OPEN_POS)
    ws.right.move_offset_world(0, 0, 5 * CM)
    ws.right.go_home()
    
    # ---------------------------------------------------------------------------
    # Step 26 - crease folded right wing of paper airplane via left hand.
    # ---------------------------------------------------------------------------
    print("[Step 26] crease folded right wing of paper airplane via left hand.")
    safe_input("Proceed with step 26? (press Enter to continue)")
    start_pos = [config.BOARD_WIDTH/2+3*CM, paper_bottom_edge_y + config.PAPER_WIDTH/4, config.CREASE_HEIGHT]
    end_pos = start_pos.copy()
    end_pos[0] += 13 * CM
    ws.right.move_away()

    actions.crease_multiple(
        arm=ws.left,
        position_pairs=[
            (start_pos, end_pos)
        ]
    )

    # ---------------------------------------------------------------------------
    # Step 27 - remove magnets and go home.
    # ---------------------------------------------------------------------------
    print("[Step 27] remove magnets and go home.")
    safe_input("Proceed with step 27? (press Enter to continue)")
    actions.remove_magnet(ws.left, block_a)
    actions.remove_magnet(ws.left, block_b)
    actions.remove_magnet(ws.left, lbracket_a)
    ws.left.go_home()
    
    print(f"\n{'=' * 60}")
    print("  Demo complete.")
    print(f"{'=' * 60}\n")

if __name__ == "__main__":
    main()
