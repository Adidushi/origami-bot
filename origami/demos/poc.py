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
    parser.add_argument("--calibrate", action="store_true",
                        help="calibrate the boards (default: False)")
    args = parser.parse_args()

    mode = "HARDWARE" if args.hardware else "SIMULATION"
    if args.calibrate:
        calibrate_boards()
        return

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
    origin_x = config.BOARD_WIDTH  / 2 - config.PAPER_WIDTH  / 2
    origin_y = config.BOARD_HEIGHT / 2 - config.PAPER_HEIGHT / 2   # < 0
    ws.paper = Paper.rectangle(config.PAPER_WIDTH, config.PAPER_HEIGHT,
                               origin=(origin_x, origin_y))
    
    # Grip the bottom-right corner of the paper from the −y side (below board)
    grip_x = origin_x+1/100   # LEFT! edge of paper
    grip_y = origin_y+0.3/100                         # bottom edge, overhangs board

    print()
    print(f"paper bottom-left corner   ({grip_x:.4f}, {grip_y:.4f})")
    print(f"paper bottom-right corner  ({grip_x:.4f}, {grip_y:.4f})")
    # print(f"overhang below board       {-grip_y * 1000:.1f} mm")

    # ---------------------------------------------------------------------------
    # Step 1 — place initial magnet
    # ---------------------------------------------------------------------------
    print("[Step 1] Place initial magnets")
    block_a = BlockMagnet(
        identifier="block_a",
        handle_height=0.015,
        tray_position=(config.MAGNET_PLATFORM_POSITIONS["bottom_right"][0]-1.7/100, config.MAGNET_PLATFORM_POSITIONS["bottom_right"][1]+11.2/100, config.MAGNET_PLATFORM_POSITIONS["bottom_right"][2]),
    )

    block_b = BlockMagnet(
        identifier="block_b",
        handle_height=0.015,
        tray_position=(config.MAGNET_PLATFORM_POSITIONS["bottom_right"][0]-1.7/100-5/100, config.MAGNET_PLATFORM_POSITIONS["bottom_right"][1]+11.2/100, config.MAGNET_PLATFORM_POSITIONS["bottom_right"][2]),
    )

    lbracket_a = LBracketMagnet(
        identifier="lbracket_a",
        handle_height=0.003,
        handle_offset=0.2, 
        orientation = 0,
        tray_position=(config.MAGNET_PLATFORM_POSITIONS["bottom_right"][0]-1.15/100, config.MAGNET_PLATFORM_POSITIONS["bottom_right"][1]+3.05/100, config.MAGNET_PLATFORM_POSITIONS["bottom_right"][2]),
    )

    # actions.place_magnet(ws, block_a, x=0.275, y=0.10, carrying_arm="left")
    # actions.place_magnet(ws, block_b, x=0.265, y=0.225, carrying_arm="left")
    # ws.left.go_home()

    # # ---------------------------------------------------------------------------
    # # Step 2 — grip paper edge
    # # ---------------------------------------------------------------------------
    # print("[Step 2] grab paper for first fold — grip paper edge")
    # # Grip the paper edge with the right arm (sideways horizontal approach)
    # actions.grip_paper(
    #     workspace=ws, 
    #     x=grip_x, 
    #     y=grip_y, 
    #     grip_angle=math.pi / 2, 
    #     arm="right"
    # )

    # # ---------------------------------------------------------------------------
    # # Step 3 — fold arc  (replaces estimated_circular_motion)
    # # ---------------------------------------------------------------------------
    # print("[Step 3] estimated_circular_motion — fold arc")

    # # Fold axis at the board centre: folds the right half of the paper over.
    # # Radius = grip_x - fold_axis_x ≈ 9.5 cm, matching get.py's radius value.
    # offsets = actions.fold_arc(
    #     ws,
    #     arm_side="right",
    #     radius=9.5/100,
    #     axis="x",
    #     n_steps=8,
    # )

    # # in future need to correct orientation of gripper to always close on bottom and top position of magnet holder, right now its fine based on preset magnet and gripper orientations in the POC
    # # paper is placed s.t. its top edge is aligned with top of board, but since their sizes differ, to get to middle of paper we need to move down by paper's height from top of board, which is not the same as half of board's height
    # actions.place_magnet(ws, lbracket_a, x=config.BOARD_WIDTH/2+3.5/100, y=config.BOARD_HEIGHT-config.PAPER_HEIGHT/2, carrying_arm="left")

    # actions.crease(
    #     workspace=ws,
    #     arm_side="left",
    #     start_x=origin_x + config.PAPER_WIDTH / 2 + 3/100,  # start just beyond the left edge of the paper
    #     start_y=origin_y,
    #     crease_length=config.PAPER_HEIGHT,
    #     axis="y"
    # )

    # actions.remove_magnet(ws, 'lbracket_a', carrying_arm="left")
    # ws.arm(side='left').go_home()

    # actions.unfold_arc(
    #     ws,
    #     arm_side="right",
    #     offsets=offsets
    # )
    # ws.arm(side='right').release()
    # ws.arm(side='right').move_offset_world(0,-2/100,0)
    # ws.arm(side='right').go_home()

    # actions.remove_magnet(ws, 'block_a', carrying_arm="left")
    # actions.remove_magnet(ws, 'block_b', carrying_arm="left")
    # ws.arm(side='left').go_home()

    grip_x = config.BOARD_WIDTH/2

    actions.grip_paper(
        workspace=ws, 
        x=grip_x, 
        y=grip_y, 
        grip_angle=math.pi / 2, 
        arm="right"
    )

    actions.flip_paper(workspace=ws, arm="right")
    
    ws.arm(side='right').release()
    ws.arm(side='right').move_offset_world(0,-2/100,0)
    ws.arm(side='right').go_home()


    print(f"\n{'=' * 60}")
    print("  Demo complete.")
    print(f"{'=' * 60}\n")


def calibrate_boards():
    arm_configs = [ArmConfig(home=config.LEFT_ARM_START_JOINTS), ArmConfig(home=config.RIGHT_ARM_START_JOINTS)]
    ws = Workspace.hardware(arm_configs=arm_configs, home=True)

    ws.left.grip()
    ws.right.grip()
    
    # Centre an A4 sheet on the board.  A4 is 297 mm tall on a 270 mm board,
    # giving a natural 13.5 mm overhang on both the top and bottom edges.
    origin_x = config.BOARD_WIDTH  / 2 - config.PAPER_WIDTH  / 2
    origin_y = config.BOARD_HEIGHT / 2 - config.PAPER_HEIGHT / 2   # < 0
    ws.paper = Paper.rectangle(config.PAPER_WIDTH, config.PAPER_HEIGHT,
                               origin=(origin_x, origin_y))
    
    input("ready to calibrate")

    ws.left.move_to_world(*ws.left.tcp_to_world(config.LEFT_ARM_CORNERS["top_left"]))
    ws.right.move_to_world(*ws.right.tcp_to_world(config.RIGHT_ARM_CORNERS["bottom_right"]))

    input("calibrate board")

    while input("next?") != "exit":
        ws.left.move_to_world(*config.MAGNET_PLATFORM_POSITIONS["bottom_right"])
        if input("next?") == "exit":
            break
        ws.left.move_to_world(*config.MAGNET_PLATFORM_POSITIONS["bottom_left"])

if __name__ == "__main__":
    main()
