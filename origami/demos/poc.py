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
from origami.magnets import BlockMagnet
from origami import Paper, Workspace, actions, config


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="demo_get — paper edge grip and fold")
    parser.add_argument("--hardware", action="store_true",
                        help="drive real arms (default: simulate)")
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
    ws = Workspace.hardware(arm_configs=arm_configs) if args.hardware else Workspace.simulated(arm_config=arm_configs)
    ws.go_to_start()
    
    # Centre an A4 sheet on the board.  A4 is 297 mm tall on a 270 mm board,
    # giving a natural 13.5 mm overhang on both the top and bottom edges.
    origin_x = config.BOARD_WIDTH  / 2 - config.PAPER_WIDTH  / 2
    origin_y = config.BOARD_HEIGHT / 2 - config.PAPER_HEIGHT / 2   # < 0
    ws.paper = Paper.rectangle(config.PAPER_WIDTH, config.PAPER_HEIGHT,
                               origin=(origin_x, origin_y))
    
    


    # Grip the bottom-right corner of the paper from the −y side (below board)
    grip_x = origin_x   # LEFT! edge of paper
    grip_y = origin_y                         # bottom edge, overhangs board

    print()
    print(f"paper bottom-left corner   ({grip_x:.4f}, {grip_y:.4f})")
    print(f"paper bottom-right corner  ({grip_x:.4f}, {grip_y:.4f})")
    print(f"overhang below board       {-grip_y * 1000:.1f} mm")


    # ---------------------------------------------------------------------------
    # Step 1 — place initial magnet
    # ---------------------------------------------------------------------------
    print("[Step 1] Place initial magnets")
    block_a = BlockMagnet(
        identifier="block_a",
        holder_height=0.015,
        tray_position=(-0.15, 0.05, -0.02),
    )

    block_b = BlockMagnet(
        identifier="block_b",
        holder_height=0.015,
        tray_position=(-0.15, 0.115, -0.02),
    )

    actions.place_magnet(ws, block_a, x=0.255, y=0.135, carrying_arm="left")
    actions.place_magnet(ws, block_b, x=0.255, y=0.225, carrying_arm="left")
    ws.left.go_home()

    # ---------------------------------------------------------------------------
    # Step 2 — grip paper edge
    # ---------------------------------------------------------------------------
    print("[Step 2] grab paper for first fold — grip paper edge")
    # Grip the paper edge with the right arm (sideways horizontal approach)
    actions.grip_paper(
        workspace=ws, 
        x=grip_x, 
        y=grip_y, 
        grip_angle=math.pi / 2, 
        arm="right"
    )

    # ---------------------------------------------------------------------------
    # Step 3 — fold arc  (replaces estimated_circular_motion)
    # ---------------------------------------------------------------------------
    print("[Step 3] estimated_circular_motion — fold arc")

    # Fold axis at the board centre: folds the right half of the paper over.
    # Radius = grip_x - fold_axis_x ≈ 9.5 cm, matching get.py's radius value.
    actions.fold_arc(
        ws,
        arm_side="right",
        radius=9.5/100,
        axis="x",
        n_steps=12,
    )

    actions.crease(
        workspace=ws,
        arm_side="left",
        start_x=origin_x + config.PAPER_WIDTH / 2,
        start_y=origin_y,
        crease_length=config.PAPER_HEIGHT,
        axis="y"
    )

    print(f"\n{'=' * 60}")
    print("  Demo complete.")
    print(f"{'=' * 60}\n")


if __name__ == "__main__":
    main()
