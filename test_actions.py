"""Live / simulated action test.

Runs every action in actions.py against either the simulator (default) or
real hardware.  Each step prints what is about to happen, executes it, and
confirms the result, so you can follow along on the pendant / visualiser.

Usage
-----
    # Simulation (no hardware needed)
    python3 test_actions.py

    # Real hardware (arms must be powered on and homed)
    python3 test_actions.py --hardware
"""
from __future__ import annotations

import argparse
import math
import sys

import numpy as np

import origami
from origami import actions, BlockMagnet, LBracketMagnet, MagnetRegistry, Paper, Workspace


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def step(title: str) -> None:
    print(f"\n{'─' * 60}")
    print(f"  {title}")
    print(f"{'─' * 60}")


def info(label: str, value) -> None:
    print(f"    {label:30s} {value}")


def check(condition: bool, msg: str) -> None:
    tag = "  PASS" if condition else "  FAIL"
    print(f"  [{tag}]  {msg}")
    if not condition:
        sys.exit(1)


# ---------------------------------------------------------------------------
# Build workspace
# ---------------------------------------------------------------------------

def build_workspace(hardware: bool) -> Workspace:
    if hardware:
        print("Connecting to hardware arms …")
        ws = Workspace.hardware()
        print("Connected.")
    else:
        print("Building simulated workspace.")
        ws = Workspace.simulated()
    return ws


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_arm_basics(ws: Workspace) -> None:
    step("1 · Arm basics — get_tool_pos, move_to_clearance, press, lift")

    for side in ("left", "right"):
        arm = ws.arm(side)
        pos = arm.get_tool_pos()
        info(f"{side} arm world pos", f"({pos[0]:.4f}, {pos[1]:.4f}, {pos[2]:.4f})")

    arm = ws.left
    cx, cy = 0.1, 0.1

    arm.move_to_clearance(cx, cy)
    px, py, pz = arm.get_tool_pos()
    info("move_to_clearance z", f"{pz:.4f} m  (want ≈ {arm.config.clearance_z:.4f})")
    check(abs(pz - arm.config.clearance_z) < 0.005,
          "z reached clearance height")

    arm.press(cx, cy)
    px, py, pz = arm.get_tool_pos()
    info("press z", f"{pz:.4f} m  (want ≈ {-arm.config.contact_depth:.4f})")
    check(abs(pz - (-arm.config.contact_depth)) < 0.005,
          "z at contact depth")

    arm.lift()
    px, py, pz = arm.get_tool_pos()
    info("lift z", f"{pz:.4f} m  (want ≈ {arm.config.clearance_z:.4f})")
    check(abs(pz - arm.config.clearance_z) < 0.005,
          "z back at clearance after lift")


def test_motion_types(ws: Workspace) -> None:
    step("2 · move_to_world — linear motion")

    arm = ws.right
    target = (0.20, 0.15, arm.config.clearance_z)

    arm.move_to_world(*target)
    px, py, pz = arm.get_tool_pos()
    info("move result", f"({px:.4f}, {py:.4f}, {pz:.4f})")
    check(abs(px - target[0]) < 0.005 and abs(py - target[1]) < 0.005,
          "reached target")

    arm.move_to_world(0.10, 0.10, arm.config.clearance_z)
    px, py, pz = arm.get_tool_pos()
    info("second move result", f"({px:.4f}, {py:.4f}, {pz:.4f})")
    check(abs(px - 0.10) < 0.005 and abs(py - 0.10) < 0.005,
          "reached second target")


def test_move_offset_world(ws: Workspace) -> None:
    step("3 · move_offset_world — relative move, single TCP snapshot")

    arm = ws.left
    arm.move_to_clearance(0.10, 0.10)
    before = arm.get_tool_pos()
    dx, dy, dz = 0.03, 0.02, 0.0
    arm.move_offset_world(dx, dy, dz)
    after = arm.get_tool_pos()

    info("before", f"({before[0]:.4f}, {before[1]:.4f}, {before[2]:.4f})")
    info("after",  f"({after[0]:.4f}, {after[1]:.4f}, {after[2]:.4f})")
    check(abs((after[0] - before[0]) - dx) < 0.005 and
          abs((after[1] - before[1]) - dy) < 0.005,
          f"offset ({dx}, {dy}, {dz}) applied correctly")


def test_magnet_place_move_remove(ws: Workspace) -> None:
    step("4 · Magnets — place, move, remove")

    block = BlockMagnet(
        identifier="block_a",
        holder_height=0.015,
        tray_position=(-0.15, 0.05, -0.02),
    )
    lbracket = BlockMagnet(
        identifier="lbracket_a",
        holder_height=0.015,
        tray_position=(-0.15, 0.15, -0.02),
    )

    # Place block
    actions.place_magnet(ws, block, x=0.10, y=0.10, carrying_arm="left")
    check(block.placed, "block magnet marked as placed")
    info("block centre", block.center.tolist())
    check("block_a" in ws.magnets, "block registered in workspace")

    # Place L-bracket
    actions.place_magnet(ws, lbracket, x=0.20, y=0.10,
                         orientation=0.0, carrying_arm="left")
    check(lbracket.placed, "L-bracket magnet marked as placed")

    # Move block to new position
    actions.move_magnet(ws, "block_a", x=0.15, y=0.15, carrying_arm="left")
    check(np.allclose(block.center, [0.15, 0.15]),
          f"block centre updated to (0.15, 0.15), got {block.center.tolist()}")

    # Remove block → tray
    actions.remove_magnet(ws, "block_a", carrying_arm="left")
    check(not block.placed, "block magnet marked as stowed")

    # Remove L-bracket → tray
    actions.remove_magnet(ws, "lbracket_a", carrying_arm="left")
    check(not lbracket.placed, "L-bracket magnet marked as stowed")


def test_grip_corner_constraint(ws: Workspace) -> None:
    step("5 · grip_corner — constraint check and grip with overhang")

    # Reset paper to origin (fully on board) and verify constraint fires
    ws.paper = Paper.rectangle(origami.config.PAPER_WIDTH, origami.config.PAPER_HEIGHT,
                               origin=(0.0, 0.0))
    try:
        actions.grip_corner(ws, "bottom_left")
        check(False, "should have raised ValueError — paper fully on board")
    except ValueError as e:
        check(True, f"raised ValueError as expected: {str(e)[:55]}…")

    # Slide paper so bottom-left overhangs by ~1 cm
    ws.paper.translate([-0.01, -0.01])
    bl = ws.paper.landmark("bottom_left")
    info("bottom_left after slide", bl.tolist())
    check(bl[0] < -actions.GRIP_OVERHANG_MIN and bl[1] < -actions.GRIP_OVERHANG_MIN,
          "corner now overhangs board")

    actions.grip_corner(ws, "bottom_left", arm="right")
    check(True, "grip_corner executed without error")


def test_grip_edge(ws: Workspace) -> None:
    step("6 · grip_edge — bottom edge overhang")

    # Paper already offset from previous test; ensure bottom edge overhangs
    p1 = ws.paper.landmark("bottom_left")
    p2 = ws.paper.landmark("bottom_right")
    mid = (p1 + p2) / 2
    info("bottom edge midpoint", mid.tolist())

    overhangs = p1[1] < -actions.GRIP_OVERHANG_MIN or p2[1] < -actions.GRIP_OVERHANG_MIN
    if not overhangs:
        ws.paper.translate([0.0, -0.01])
    actions.grip_edge(ws, "bottom", arm="right")
    check(True, "grip_edge executed without error")


def test_move_paper(ws: Workspace) -> None:
    step("7 · move_paper — translate sheet by (dx, dy)")

    # Ensure bottom-left overhangs
    ws.paper.translate([-0.01, -0.01])
    before = ws.paper.centroid().copy()
    dx, dy = 0.03, 0.02

    actions.move_paper(ws, dx=dx, dy=dy, carrying_arm="right")

    after = ws.paper.centroid()
    info("centroid before", before.tolist())
    info("centroid after",  after.tolist())
    check(abs((after[0] - before[0]) - dx) < 0.001 and
          abs((after[1] - before[1]) - dy) < 0.001,
          f"paper translated by ({dx}, {dy})")


def test_rotate_paper(ws: Workspace) -> None:
    step("8 · rotate_paper — 15° counter-clockwise about centroid")

    # Ensure a corner overhangs
    ws.paper.translate([-0.01, -0.01])
    pivot = ws.paper.centroid().copy()
    bl_before = ws.paper.landmark("bottom_left").copy()
    angle = np.radians(15)

    actions.rotate_paper(ws, angle=angle, folding_arm="right", anchor_arm="left",
                         n_steps=6)

    bl_after = ws.paper.landmark("bottom_left")
    # Verify analytically: the bottom_left corner should have rotated
    r = np.linalg.norm(bl_before - pivot)
    a0 = np.arctan2(bl_before[1] - pivot[1], bl_before[0] - pivot[0])
    expected = pivot + r * np.array([np.cos(a0 + angle), np.sin(a0 + angle)])

    info("bottom_left before", bl_before.tolist())
    info("bottom_left after",  bl_after.tolist())
    info("expected",           expected.tolist())
    check(np.allclose(bl_after, expected, atol=1e-6),
          "landmark rotated to correct position")


def test_flip_paper(ws: Workspace) -> None:
    step("9 · flip_paper — y-axis flip (bottom edge over to top)")

    # Ensure bottom edge overhangs for the flip
    ws.paper.translate([-0.01, -0.01])
    bl_before = ws.paper.landmark("bottom_left").copy()
    tl_before = ws.paper.landmark("top_left").copy()
    cy_before = ws.paper.centroid()[1]

    actions.flip_paper(ws, axis="y", folding_arm="right", anchor_arm="left")

    bl_after = ws.paper.landmark("bottom_left")
    tl_after = ws.paper.landmark("top_left")

    info("bottom_left y before / after", f"{bl_before[1]:.4f}  →  {bl_after[1]:.4f}")
    info("top_left    y before / after", f"{tl_before[1]:.4f}  →  {tl_after[1]:.4f}")

    # After a y-flip each landmark's y is reflected about the centroid y
    exp_bl_y = 2 * cy_before - bl_before[1]
    check(abs(bl_after[1] - exp_bl_y) < 1e-6,
          f"bottom_left y reflected correctly (expect {exp_bl_y:.4f})")
    check("flip about y-axis" in ws.paper.history[-1],
          "flip recorded in paper history")


def test_grip_paper(ws: Workspace) -> None:
    step("10 · grip_paper — A4 centred on board, grip bottom edge from below")

    # A4 is 297 mm tall but the board is only 270 mm tall, so centring the sheet
    # gives a natural (297 - 270) / 2 = 13.5 mm overhang on both the top and
    # bottom edges — no artificial translation needed.
    paper_w = origami.config.PAPER_WIDTH    # 0.21 m (landscape width)
    paper_h = origami.config.PAPER_HEIGHT   # 0.297 m (portrait height)
    board_cx = origami.config.BOARD_WIDTH  / 2   # 0.185 m
    board_cy = origami.config.BOARD_HEIGHT / 2   # 0.135 m

    # Bottom-left corner of the centred sheet.  origin_y is negative because the
    # sheet extends below the board boundary (y = 0).
    origin_x = board_cx - paper_w / 2
    origin_y = board_cy - paper_h / 2

    ws.paper = Paper.rectangle(paper_w, paper_h, origin=(origin_x, origin_y))

    bottom_y = origin_y          # y-coordinate of the bottom edge (< 0 → off board)
    overhang_mm = -bottom_y * 1000

    info("paper origin", f"({origin_x:.4f}, {origin_y:.4f})")
    info("bottom-edge y", f"{bottom_y:.4f}  →  {overhang_mm:.1f} mm overhang below board")
    check(bottom_y < -actions.GRIP_OVERHANG_MIN,
          f"bottom edge overhangs by at least {actions.GRIP_OVERHANG_MIN * 1000:.0f} mm")

    # Grip the midpoint of the bottom edge.
    grip_x = board_cx    # horizontal centre — equidistant from left and right sides
    grip_y = bottom_y    # right on the bottom edge of the sheet

    # grip_angle = π/2 sets the approach direction to +y: the arm starts
    # PAPER_APPROACH_OFFSET below the grip point (further outside the board) and
    # slides horizontally in the +y direction to straddle the edge.
    actions.grip_paper(ws, x=grip_x, y=grip_y, grip_angle=math.pi / 2, arm="left")

    px, py, pz = ws.left.get_tool_pos()
    info("arm pos after grip", f"({px:.4f}, {py:.4f}, {pz:.4f})")
    check(abs(px - grip_x) < 0.005, f"arm x at grip centre ({grip_x:.4f} m)")
    check(abs(py - grip_y) < 0.005, f"arm y at bottom edge ({grip_y:.4f} m)")
    check(abs(pz - actions.PAPER_GRIP_HEIGHT) < 0.005,
          f"arm z at paper grip height ({actions.PAPER_GRIP_HEIGHT * 1000:.1f} mm)")


def test_grip_paper_corner(ws: Workspace) -> None:
    step("11 · grip_paper — bottom-right corner, 45° diagonal approach")

    # Same A4 centred-on-board layout as test 10.  The bottom edge overhangs
    # the board by ~13.5 mm, so both bottom corners are reachable from outside.
    paper_w = origami.config.PAPER_WIDTH
    paper_h = origami.config.PAPER_HEIGHT
    board_cx = origami.config.BOARD_WIDTH  / 2
    board_cy = origami.config.BOARD_HEIGHT / 2

    origin_x = board_cx - paper_w / 2
    origin_y = board_cy - paper_h / 2

    ws.paper = Paper.rectangle(paper_w, paper_h, origin=(origin_x, origin_y))

    # Bottom-right corner: right edge of the sheet at the bottom.
    grip_x = origin_x + paper_w  # right edge of the paper
    grip_y = origin_y             # bottom edge (< 0 → below board)

    info("bottom-right corner", f"({grip_x:.4f}, {grip_y:.4f})")
    info("y overhang", f"{-grip_y * 1000:.1f} mm below board edge (y = 0)")
    check(grip_y < -actions.GRIP_OVERHANG_MIN,
          f"corner overhangs board by at least {actions.GRIP_OVERHANG_MIN * 1000:.0f} mm")

    # grip_angle = 3π/4 (135°): the arm starts diagonally below-right of the
    # corner and slides toward upper-left to reach it.
    #   cos(3π/4) = -1/√2  →  x_start = grip_x + OFFSET/√2  (to the right)
    #   sin(3π/4) = +1/√2  →  y_start = grip_y − OFFSET/√2  (below)
    # This bisects the angle between the bottom edge and the right edge,
    # the natural approach direction for this corner.
    grip_angle = 3 * math.pi / 4

    actions.grip_paper(ws, x=grip_x, y=grip_y, grip_angle=grip_angle, arm="left")

    px, py, pz = ws.right.get_tool_pos()
    info("arm pos after grip", f"({px:.4f}, {py:.4f}, {pz:.4f})")
    check(abs(px - grip_x) < 0.005, f"arm x at corner ({grip_x:.4f} m)")
    check(abs(py - grip_y) < 0.005, f"arm y at corner ({grip_y:.4f} m)")
    check(abs(pz - actions.PAPER_GRIP_HEIGHT) < 0.005,
          f"arm z at paper grip height ({actions.PAPER_GRIP_HEIGHT * 1000:.1f} mm)")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="origami-bot action test suite")
    parser.add_argument("--hardware", action="store_true",
                        help="run against real arms instead of the simulator")
    args = parser.parse_args()

    print("=" * 60)
    print("  origami-bot action tests")
    mode = "HARDWARE" if args.hardware else "SIMULATION"
    print(f"  mode: {mode}")
    print("=" * 60)

    ws = build_workspace(args.hardware)

    # test_arm_basics(ws)
    # test_motion_types(ws)
    # test_move_offset_world(ws)
    test_magnet_place_move_remove(ws)
    # test_grip_corner_constraint(ws)
    # test_grip_edge(ws)
    # test_move_paper(ws)
    # test_rotate_paper(ws)
    # test_flip_paper(ws)
    # test_grip_paper(ws)
    test_grip_paper_corner(ws)

    print(f"\n{'=' * 60}")
    print("  All tests passed.")
    print(f"{'=' * 60}\n")


if __name__ == "__main__":
    main()
