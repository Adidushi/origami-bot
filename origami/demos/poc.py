"""Replicates the two working functions from mvmt/get.py using the framework.

  demo_main_movement    ↔  main_movement()
      Move the anchor arm out of the way, then grip the paper edge from outside
      the board using a horizontal sideways approach (all moveL).

  demo_fold_arc         ↔  estimated_circular_motion()
      Starting from wherever demo_main_movement left the arm, sweep the gripped
      edge through an exact circular arc (x changes, y constant, wrist tracks).
      This is contingent on demo_main_movement having run first.

Usage
-----
    python3 origami/demos/demo_get.py              # simulation
    python3 origami/demos/demo_get.py --hardware   # real arms
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


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="demo_get — paper edge grip and fold")
    parser.add_argument("--hardware", action="store_false",
                        help="drive real arms (default: hardware)")
    args = parser.parse_args()

    mode = "HARDWARE" if args.hardware else "SIMULATION"

    print("=" * 60)
    print("  demo_get — grip paper edge and fold")
    print(f"  mode: {mode}")
    print("=" * 60)

    # ---------------------------------------------------------------------------
    # Initialization
    # ---------------------------------------------------------------------------
    arm_configs = [ArmConfig(home=config.LEFT_ARM_START_JOINTS), ArmConfig(home=config.RIGHT_ARM_START_JOINTS)]
    ws = Workspace.hardware(arm_configs=arm_configs, home=True) if args.hardware else Workspace.simulated(arm_configs=arm_configs)

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
    block_a = BlockMagnet(
        identifier="block_a",
        handle_height=0.015,
        tray_position=(config.MAGNET_PLATFORM_POSITIONS["bottom_right"][0]-1/100, config.MAGNET_PLATFORM_POSITIONS["bottom_right"][1]+11.7/100, config.MAGNET_PLATFORM_POSITIONS["bottom_right"][2]),
    )
    ws.magnets.add(block_a)

    block_b = BlockMagnet(
        identifier="block_b",
        handle_height=0.015,
        tray_position=(config.MAGNET_PLATFORM_POSITIONS["bottom_right"][0]-1/100-5/100, config.MAGNET_PLATFORM_POSITIONS["bottom_right"][1]+11.7/100, config.MAGNET_PLATFORM_POSITIONS["bottom_right"][2]),
    )
    ws.magnets.add(block_b)

    lbracket_a = LBracketMagnet(
        identifier="lbracket_a",
        handle_height=0.3/100,
        handle_offset=19.7/100, 
        orientation = 0,
        tray_position=(config.MAGNET_PLATFORM_POSITIONS["bottom_right"][0]-0.7/100, config.MAGNET_PLATFORM_POSITIONS["bottom_right"][1]+3.55/100, config.MAGNET_PLATFORM_POSITIONS["bottom_right"][2]),
    )
    ws.magnets.add(lbracket_a)
    

    actions.place_magnet(ws, block_a, x=0.275, y=0.10, carrying_arm="left")
    actions.place_magnet(ws, block_b, x=0.265, y=0.225, carrying_arm="left")
    ws.left.go_home()

    # ---------------------------------------------------------------------------
    # Step 2 — grip paper edge
    # ---------------------------------------------------------------------------
    print("[Step 2] grab paper for first fold — grip paper edge")
    # Grip the paper edge with the right arm (sideways horizontal approach)
    actions.grip_paper(
        workspace=ws, 
        x=paper_bottom_left_corner_x + 1/100,  # approach from just beyond the left edge of the paper
        y=paper_bottom_edge_y + 0.3/100,  # approach from just below the bottom edge of the paper
        grip_angle=0,
        arm="right"
    )

    # ---------------------------------------------------------------------------
    # Step 3 — fold paper in half  
    # ---------------------------------------------------------------------------
    print("[Step 3] fold paper in half")

    # Fold axis at the board centre: folds the right half of the paper over.
    # Radius = grip_x - fold_axis_x ≈ 9.5 cm, matching get.py's radius value.
    end_pos = list(ws.right.current_world_pos())
    end_pos[0] += 2*9.5/100
    poses = actions.fold_arc(
        ws,
        arm_side="right",
        end_pos=end_pos,
        n_steps=8,
    )
    
    # ---------------------------------------------------------------------------
    # Step 4 — place l-bracket magnet to hold the fold  
    # ---------------------------------------------------------------------------
    print("[Step 4] place l-bracket magnet to hold the fold")
    # in future need to correct orientation of gripper to always close on bottom and top position of magnet holder, right now its fine based on preset magnet and gripper orientations in the POC
    # paper is placed s.t. its top edge is aligned with top of board, but since their sizes differ, to get to middle of paper we need to move down by paper's height from top of board, which is not the same as half of board's height
    actions.place_magnet(ws, lbracket_a, x=config.BOARD_WIDTH/2+3.5/100, y=config.BOARD_HEIGHT-config.PAPER_HEIGHT/2, carrying_arm="left")
    # ---------------------------------------------------------------------------
    # Step 5 — crease paper
    # ---------------------------------------------------------------------------
    print("[Step 5] crease paper")
    actions.crease(
        workspace=ws,
        arm_side="left",
        start_x=paper_bottom_left_corner_x + config.PAPER_WIDTH / 2,  # start just beyond the left edge of the paper
        start_y=paper_bottom_edge_y,
        crease_length=config.PAPER_HEIGHT,
        axis="y"
    )

    # ---------------------------------------------------------------------------
    # Step 6 — remove l-bracket magnet
    # ---------------------------------------------------------------------------
    print("[Step 6] remove l-bracket magnet")
    actions.remove_magnet(ws, lbracket_a, carrying_arm="left")
    ws.left.go_home()
    # ---------------------------------------------------------------------------
    # Step 7 — unfold paper
    # ---------------------------------------------------------------------------
    print("[Step 7] unfold paper")
    actions.unfold_arc(
        ws,
        arm_side="right",
        poses=poses
    )

    ws.right.release()
    ws.right.move_offset_world(0,-2/100,0)
    ws.right.go_home()

    # ---------------------------------------------------------------------------
    # Step 8 — remove placed magnets
    # ---------------------------------------------------------------------------
    print("[Step 8] remove placed magnets")
    actions.remove_magnet(ws, block_a, carrying_arm="left")
    actions.remove_magnet(ws, block_b, carrying_arm="left")
    ws.left.go_home()

    actions.grip_paper(
        workspace=ws, 
        x=config.BOARD_WIDTH/2,  # approach from just beyond the left edge of the paper
        y=paper_bottom_edge_y + 0.3/100,  # approach from just below the bottom edge of the paper
        grip_angle=0,
        arm="right"
    )

    # ---------------------------------------------------------------------------
    # Step 9 — flip paper over
    # ---------------------------------------------------------------------------
    print("[Step 9] flip paper over")
    actions.flip_paper(workspace=ws, arm="right")

    # let go of paper, move back and go home
    ws.right.release()
    ws.right.move_offset_world(0,-2/100,0)
    ws.right.go_home()

    

    # ---------------------------------------------------------------------------
    # Step 10 — grab paper for second fold — grip right paper edge
    # ---------------------------------------------------------------------------
    print("[Step 10] grab paper for second fold — grip right paper edge")

    actions.place_magnet(
        ws,
        block_a,
        x=config.BOARD_WIDTH/2+8/100,
        y=15/100,
        carrying_arm='left'
    )

    actions.place_magnet(
        ws,
        block_b,
        x=config.BOARD_WIDTH/2+8/100,
        y=20/100,
        carrying_arm='left'
    )

    ws.left.go_home()

    # Grip the paper edge with the right arm (sideways horizontal approach)
    actions.grip_paper(
        workspace=ws, 
        x=paper_bottom_right_corner_x,  # approach from just beyond the right edge of the paper
        y=paper_bottom_edge_y,  # approach from just below the bottom edge of the paper
        grip_angle=math.pi / 4,
        arm="right"
    )

    # ---------------------------------------------------------------------------
    # Step 11 — fold paper from right corner to middle
    # ---------------------------------------------------------------------------
    print("[Step 11] fold paper from right corner to middle")

    # Fold the bottom-right corner up to the middle.
    # end position x is the middle of the board, while the end position y
    # is calculated to be equal in difference to the x difference
    # so that we make a 45 degree fold (square kinda thing)
    end_pos = list(ws.right.current_world_pos())
    end_pos[0] = config.BOARD_WIDTH/2+2/100
    diff = abs(ws.right.current_world_pos()[0] - end_pos[0])
    end_pos[1] += diff
    actions.fold_arc(
        ws,
        arm_side="right",
        end_pos=end_pos,
        n_steps=8,
        fold_percent=5/8
    )

    # place the corner folding magnet (long boy)
    actions.place_magnet(ws, lbracket_a, x=config.BOARD_WIDTH/2+1/100, y=2/100, carrying_arm="left")
    ws.left.go_home()

    # Open the hand, release the paper and go home (post-fold)
    ws.right.goto(0.65)
    ws.right.move_offset_world(0, 0, 5/100)

    ws.right.go_home()

    actions.move_magnet(
        ws,
        block_b,
        x=16/100+paper_bottom_left_corner_x,
        y=8/100,
        carrying_arm='right'
        )
    
    ws.right.go_home()

    actions.remove_magnet(ws, lbracket_a, carrying_arm="left")
    actions.remove_magnet(ws, block_a, carrying_arm="left")
    actions.remove_magnet(ws, block_b, carrying_arm="left")
    ws.left.go_home()
    
    print("TODO: add angled crease and refactor crease function to be more general and work with start/end pos")


    # # --------------------------------------------------------------------------- #
    # # Step 11.x - shift paper over in prep for next fold
    # # --------------------------------------------------------------------------- #
    print("[Step 11.1] Grab paper to shift to the right")
    input("Proceed with step 11.1? (press Enter to continue)")
    actions.grip_paper(
            workspace=ws, 
            x=config.BOARD_WIDTH/2,  # approach from just beyond the left edge of the paper
            y=paper_bottom_edge_y + 0.3/100,  # approach from just below the bottom edge of the paper
            grip_angle=0,
            arm="right"
        )

    print("[Step 11.2] Shift paper to right side of board")
    input("Proceed with step 11.2? (press Enter to continue)")
    # move up, then sideways, then down
    ws.right.move_offset_world(0,0,1/100)
    ws.right.move_offset_world((config.BOARD_WIDTH-config.PAPER_WIDTH)/2, 0, 0)
    ws.right.move_offset_world(0,0,-1/100)

    # let go of the page, move back and go home
    ws.right.release()
    ws.right.move_offset_world(0,-2/100,0)
    ws.right.go_home()
    ws.right.grip()
    
    print("[Step 11.3] Place block magnets in preparation for next fold")
    input("Proceed with step 11.3? (press Enter to continue)")
    actions.place_magnet(
            ws,
            block_b,
            x=15/100+(config.BOARD_WIDTH-config.PAPER_WIDTH),
            y=8/100,
            carrying_arm='left'
        )
        
    actions.place_magnet(ws, block_a, x=0.275+6/100, y=0.14, carrying_arm="left")

    ws.left.go_home()

    

    # ---------------------------------------------------------------------------
    # Step 12 — grab paper for third fold — grip bottom-left paper edge
    # ---------------------------------------------------------------------------
    print("[Step 12] grab paper for third fold — grip bottom-left paper edge")
    input("Proceed with step 12? (press Enter to continue)")
    # Grip the paper edge with the left arm (sideways horizontal approach)
    actions.grip_paper(
        workspace=ws, 
        x=config.BOARD_WIDTH-config.PAPER_WIDTH+1/100,  # approach from just beyond the left edge of the paper
        y=paper_bottom_edge_y,  # approach from just below the bottom edge of the paper
        grip_angle= - math.pi / 4,
        arm="right"
    )

    # ---------------------------------------------------------------------------
    # Step 13 — fold paper from bottom-left corner to middle
    # ---------------------------------------------------------------------------
    print("[Step 13] fold paper from bottom-left corner to middle")
    input("Proceed with step 13? (press Enter to continue)")

    # Fold axis at the board centre: folds the right half of the paper over.
    # Radius = grip_x - fold_axis_x ≈ 9.5 cm, matching get.py's radius value.
    end_pos = list(ws.right.current_world_pos())
    end_pos[0] = config.BOARD_WIDTH-config.PAPER_WIDTH/2-2/100
    diff = abs(ws.right.current_world_pos()[0] - end_pos[0])
    end_pos[1] += diff
    actions.fold_arc(
        ws,
        arm_side="right",
        end_pos=end_pos,
        n_steps=8,
        fold_percent=5/8
    )

    # ----------------------------------------------------------------------------
    # Step 14 — place l-bracket magnet to hold the fold
    # ----------------------------------------------------------------------------
    print("[Step 14] place l-bracket magnet to hold the fold")
    input("Proceed with step 14? (press Enter to continue)")
    # place the corner folding magnet (long boy)
    actions.place_magnet(
        ws, 
        lbracket_a, 
        x=config.BOARD_WIDTH-13.5/100, 
        y=2/100, 
        carrying_arm="left",
        orientation=math.radians(45)
    )
    ws.left.go_home()

    # Open the hand, release the paper and go home (post-fold)
    ws.right.goto(0.65)
    ws.right.move_offset_world(0, 0, 5/100)
    ws.right.go_home()
    ws.right.grip()
    actions.remove_magnet(ws, lbracket_a, carrying_arm="left")
    actions.remove_magnet(ws, block_a, carrying_arm="left")
    actions.remove_magnet(ws, block_b, carrying_arm="left")
    ws.left.go_home()

    # ----------------------------------------------------------------------------
    # Step 15 - move paper back to center and down 8.5cm
    # ----------------------------------------------------------------------------
    print("[Step 15] move paper back to center and down 8.5cm")
    input("Proceed with step 15? (press Enter to continue)")
    actions.grip_paper(
                workspace=ws, 
                x=config.BOARD_WIDTH/2+(config.BOARD_WIDTH-config.PAPER_WIDTH)/2,  # approach from just beyond the left edge of the paper
                y=paper_bottom_edge_y + 0.3/100,  # approach from just below the bottom edge of the paper
                grip_angle=0,
                arm="right"
            )

    ws.right.move_offset_world(0,0,1/100)
    ws.right.move_offset_world(-(config.BOARD_WIDTH-config.PAPER_WIDTH)/2, -8.5/100, 0)
    ws.right.move_offset_world(0,0,-1/100)

    # let go of the page, move back and go home
    ws.right.goto(actions.MAGNET_GRIP_OPEN_POS)
    ws.right.move_offset_world(0,-2/100,0)
    ws.right.go_home()
    ws.right.grip()
    
    # ----------------------------------------------------------------------------
    # Step 16 - place magnet in prep for next folds
    # ----------------------------------------------------------------------------
    print("[Step 16] place magnet in prep for next folds")
    input("Proceed with step 16? (press Enter to continue)")
    # places a magnet off of the centerfold line
    # 3.5cm from the back edge of the paper, plus 8.5 of moving the page down
    # left side magnet
    actions.place_magnet(ws, block_a, x=config.BOARD_WIDTH/2-2.5/100, y=config.BOARD_HEIGHT-12/100)
    ws.left.go_home()

    # ---------------------------------------------------------------------------
    # Step 17 — grip paper edge
    # ---------------------------------------------------------------------------
    print("[Step 17] grab paper for first wing fold — grip paper edge")
    input("Proceed with step 17? (press Enter to continue)")
    # Grip the paper edge with the right arm (sideways horizontal approach)
    actions.grip_paper(
        workspace=ws, 
        x=paper_bottom_right_corner_x - 1/100,  # approach from just beyond the right edge of the paper
        y=paper_bottom_edge_y + 0.3/100,  # approach from just below the bottom edge of the paper
        grip_angle=0,
        arm="right"
    )

    # ---------------------------------------------------------------------------
    # Step 18 — fold right paper wing
    # ---------------------------------------------------------------------------
    print("[Step 18] fold right paper wing")
    input("Proceed with step 18? (press Enter to continue)")
    # Fold axis at the board centre: folds the right half of the paper over.
    # Radius = grip_x - fold_axis_x ≈ 9.5/2 cm, to the centerline of page
    end_pos = list(ws.right.current_world_pos())
    end_pos[0] -= 9.5/100
    poses = actions.fold_arc(
        ws,
        arm_side="right",
        end_pos=end_pos,
        n_steps=8,
    )

    # ---------------------------------------------------------------------------
    # Step 19 — place l-bracket magnet to hold the fold  
    # ---------------------------------------------------------------------------
    print("[Step 19] place l-bracket magnet to hold the fold")
    input("Proceed with step 19? (press Enter to continue)")
    # in future need to correct orientation of gripper to always close on bottom and top position of magnet holder, right now its fine based on preset magnet and gripper orientations in the POC
    # paper is placed s.t. its top edge is aligned with top of board, but since their sizes differ, to get to middle of paper we need to move down by paper's height from top of board, which is not the same as half of board's height
    actions.place_magnet(ws, lbracket_a, x=config.BOARD_WIDTH/2+5/100, y=config.BOARD_HEIGHT-20.5/100, carrying_arm="left")
    ws.left.go_home()
    ws.right.goto(actions.MAGNET_GRIP_OPEN_POS)
    ws.right.move_offset_world(0,-2/100,0)
    ws.right.go_home()

    # ---------------------------------------------------------------------------
    # Step 20 — crease paper
    # ---------------------------------------------------------------------------
    print("[Step 20] crease paper")
    input("Proceed with step 20? (press Enter to continue)")
    actions.crease(
        workspace=ws,
        arm_side="left",
        start_x=config.BOARD_WIDTH/2+5/100,  # start just beyond the left edge of the paper
        start_y=config.BOARD_HEIGHT-20.5/100,
        crease_length=config.PAPER_HEIGHT,
        axis="y",
        crease_under_magnet=False
    )

    # ---------------------------------------------------------------------------
    # Step 21 — move block magnet onto last fold and remove L bracket
    # ---------------------------------------------------------------------------
    print("[Step 21] move block magnet onto last fold")
    input("Proceed with step 21? (press Enter to continue)")
    actions.move_magnet(ws, block_a, x=config.BOARD_WIDTH/2+2.5/100, y=config.BOARD_HEIGHT-12/100)
    actions.remove_magnet(ws, lbracket_a, carrying_arm="left")

    # ---------------------------------------------------------------------------
    # Step 22 — grip left paper edge
    # ---------------------------------------------------------------------------
    print("[Step 22] grip left paper edge")
    input("Proceed with step 22? (press Enter to continue)")
    # Grip the paper edge with the right arm (sideways horizontal approach)
    actions.grip_paper(
        workspace=ws, 
        x=paper_bottom_left_corner_x + 1/100,  # approach from just beyond the left edge of the paper
        y=paper_bottom_edge_y + 0.3/100,  # approach from just below the bottom edge of the paper
        grip_angle=0,
        arm="right"
    )

    # ---------------------------------------------------------------------------
    # Step 23 — fold left paper wing
    # ---------------------------------------------------------------------------
    print("[Step 23] fold left paper wing")
    input("Proceed with step 23? (press Enter to continue)")

    # Fold axis at the board centre: folds the left half of the paper over.
    # Radius = grip_x - fold_axis_x ≈ 9.5/2 cm, to the centerline of page
    end_pos = list(ws.right.current_world_pos())
    end_pos[0] += 9.5/100
    poses = actions.fold_arc(
        ws,
        arm_side="right",
        end_pos=end_pos,
        n_steps=8,
    )


    # ---------------------------------------------------------------------------
    # Step 24 — place l-bracket magnet to hold the fold  
    # ---------------------------------------------------------------------------
    print("[Step 24] place l-bracket magnet to hold the fold")
    input("Proceed with step 24? (press Enter to continue)")
    # in future need to correct orientation of gripper to always close on bottom and top position of magnet holder, right now its fine based on preset magnet and gripper orientations in the POC
    # paper is placed s.t. its top edge is aligned with top of board, but since their sizes differ, to get to middle of paper we need to move down by paper's height from top of board, which is not the same as half of board's height
    actions.place_magnet(ws, lbracket_a, x=config.BOARD_WIDTH/2-5/100, y=config.BOARD_HEIGHT-20.5/100, carrying_arm="left")
    ws.left.go_home()
    ws.right.goto(actions.MAGNET_GRIP_OPEN_POS)
    ws.right.move_offset_world(0,-2/100,0)
    ws.right.go_home()
    # ---------------------------------------------------------------------------
    # Step 25 — crease paper
    # ---------------------------------------------------------------------------
    print("[Step 25] crease paper")
    input("Proceed with step 25? (press Enter to continue)")
    actions.crease(
        workspace=ws,
        arm_side="left",
        start_x=config.BOARD_WIDTH/2-5/100,  # start just beyond the left edge of the paper
        start_y=config.BOARD_HEIGHT-20.5/100-config.PAPER_HEIGHT/2,
        crease_length=config.PAPER_HEIGHT,
        axis="y",
        crease_under_magnet=False
    )

    actions.remove_magnet(ws, lbracket_a, carrying_arm="left")
    actions.remove_magnet(ws, block_a, carrying_arm="left")
    ws.left.go_home()

    actions.grip_paper(
        workspace=ws, 
        x=config.BOARD_WIDTH/2,  # approach from just beyond the left edge of the paper
        y=paper_bottom_edge_y + 0.3/100-8.5/100,  # approach from just below the bottom edge of the paper
        grip_angle=0,
        arm="right"
    )

    ws.right.move_offset_world(0,8.5/100,0)

    # ---------------------------------------------------------------------------
    # Step 26 — flip paper over
    # ---------------------------------------------------------------------------
    print("[Step 26] flip paper over")
    input("Proceed with step 26? (press Enter to continue)")
    actions.flip_paper(workspace=ws, arm="right")

    # let go of paper, move back and go home
    ws.right.goto(actions.MAGNET_GRIP_OPEN_POS)
    ws.right.move_offset_world(0,-2/100,0)
    ws.right.go_home()


    print(f"\n{'=' * 60}")
    print("  Demo complete.")
    print(f"{'=' * 60}\n")

if __name__ == "__main__":
    main()
