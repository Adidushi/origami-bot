"""CLI to re-teach the arm/board/magnet-platform calibration when the rig drifts.

Jog arms in small increments with the keyboard until they line up with a
known reference point, then confirm with Enter. Keys:

    w/s = x-/x+   a/d = y-/y+   q/e = z-/z+
    +/- = grow/shrink the step size (1mm per press)
    r   = (board calibration only) switch which arm the keys move
    n   = (magnet-platform only) restart the corner sequence
    Enter = confirm and move on

Run with: python -m origami.calibrate
"""
from __future__ import annotations

import json
import math
import sys
import termios
import tty

from . import config
from .arm import Arm, ArmConfig

ORIENTATION = (0.0, math.pi, 0.0)  # tool points straight down, no roll
STEP = 0.005        # metres, starting jog increment
STEP_DELTA = 0.001  # metres, how much +/- resizes the step
SPEED, ACCEL = 0.1, 0.3

# wasd + qe -> (axis index, sign)
AXIS_KEYS = {"w": (0, -1), "s": (0, 1), "a": (1, -1), "d": (1, 1), "q": (2, -1), "e": (2, 1)}


def _read_key() -> str:
    """Block for a single keypress on stdin (no Enter needed)."""
    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    try:
        tty.setcbreak(fd)
        return sys.stdin.read(1)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)


def _wait_for_enter() -> None:
    while _read_key() not in ("\r", "\n"):
        pass


def _jog(arm, label, extra_stop_keys=()):
    """Let wasdqe/+- nudge ``arm``'s TCP relative to wherever it currently is
    until Enter or a key in ``extra_stop_keys`` is pressed.  The arm should
    already be sitting at its starting pose.  Returns the key that stopped it.
    """
    step = STEP
    while True:
        pose = arm.current_tcp_pose()
        print(f"[{label}] step={step * 1000:.1f}mm pos={[round(p, 4) for p in pose[:3]]}",
              end="", flush=True)
        key = _read_key()

        if key in ("\r", "\n") or key in extra_stop_keys:
            print()
            return key
        if key in AXIS_KEYS:
            axis, sign = AXIS_KEYS[key]
            pose[axis] += sign * step
            arm.move_to_tcp(pose, SPEED, ACCEL)
        elif key in ("+", "="):
            step += STEP_DELTA
        elif key in ("-", "_"):
            step = max(STEP_DELTA, step - STEP_DELTA)


def calibrate_board(left_arm, right_arm, left_corners, right_corners):
    """Re-teach the left arm's bottom_left board corner and the right arm's
    top_right corner ('r' switches which arm the keys move; Enter confirms
    both). The board is a fixed-size rectangle, so shifting every other
    corner by the same amount the measured one moved re-aligns all four.

    Returns the updated ``(left_corners, right_corners)`` dicts.
    """
    print("press <Enter> to send both arms home and start...")
    _wait_for_enter()
    left_arm.go_home()
    right_arm.go_home()

    arms = [left_arm, right_arm]
    positions = [list(left_corners["bottom_left"][:3]), list(right_corners["top_right"][:3])]
    labels = ["left arm -> bottom_left", "right arm -> top_right"]

    # send each arm to its starting corner; _jog then nudges relatively from there
    for arm, pos in zip(arms, positions):
        arm.move_to_tcp(pos + list(ORIENTATION), SPEED, ACCEL)

    print("w/s=x-/x+ a/d=y-/y+ q/e=z-/z+ +/-=step r=switch arm Enter=confirm both")
    active = 0
    while _jog(arms[active], labels[active], extra_stop_keys="r") == "r":
        active = 1 - active

    lx, ly, lz = left_arm.current_tcp_pose()[:3]
    ox, oy, oz = left_corners["bottom_left"][:3]
    new_left_corners = {name: [x + lx - ox, y + ly - oy, z + lz - oz, rx, ry, rz]
                        for name, (x, y, z, rx, ry, rz) in left_corners.items()}

    rx_, ry_, rz_ = right_arm.current_tcp_pose()[:3]
    ox, oy, oz = right_corners["top_right"][:3]
    new_right_corners = {name: [x + rx_ - ox, y + ry_ - oy, z + rz_ - oz, rx, ry, rz]
                         for name, (x, y, z, rx, ry, rz) in right_corners.items()}

    return new_left_corners, new_right_corners


def calibrate_magnet_platform(left_arm, right_arm, positions):
    """Re-teach the magnet platform's bottom_right and bottom_left corners
    using the left arm only: Enter marks bottom_right, then bottom_left.
    Pressing 'n' at any point restarts the sequence from bottom_right.

    Returns the updated world-position dict.
    """
    print("press <Enter> to send both arms home and start...")
    _wait_for_enter()
    left_arm.go_home()
    right_arm.go_home()  # out of the way; not otherwise used

    print("w/s=x-/x+ a/d=y-/y+ q/e=z-/z+ +/-=step n=restart Enter=confirm")
    while True:
        new_positions = {}
        for name in ("bottom_right", "bottom_left"):
            # send the arm to the stored corner; _jog then nudges relatively from there
            left_arm.move_to_tcp(list(left_arm.world_to_tcp(*positions[name])[:3]) + list(ORIENTATION),
                                 SPEED, ACCEL)
            if _jog(left_arm, f"magnet platform {name}", extra_stop_keys="n") == "n":
                break  # restart the whole sequence from bottom_right
            # read back the confirmed world position straight from the arm
            x, y, z = left_arm.current_world_pos()
            new_positions[name] = [x, y, z]
        else:
            return new_positions


def main():
    left_arm = Arm.real("left", config.LEFT_ARM_IP, config.left_calibration(),
                        config=ArmConfig(home=config.LEFT_ARM_START_JOINTS))
    right_arm = Arm.real("right", config.RIGHT_ARM_IP, config.right_calibration(),
                         config=ArmConfig(home=config.RIGHT_ARM_START_JOINTS))

    data = json.loads(config.CALIBRATION_FILE.read_text())

    print("Calibrating board corners...")
    data["LEFT_ARM_CORNERS"], data["RIGHT_ARM_CORNERS"] = calibrate_board(
        left_arm, right_arm, data["LEFT_ARM_CORNERS"], data["RIGHT_ARM_CORNERS"])

    print("Calibrating magnet platform...")
    data["MAGNET_PLATFORM_POSITIONS"] = calibrate_magnet_platform(
        left_arm, right_arm, data["MAGNET_PLATFORM_POSITIONS"])

    config.CALIBRATION_FILE.write_text(json.dumps(data, indent=2))
    print(f"wrote {config.CALIBRATION_FILE}")
    left_arm.go_home()
    right_arm.go_home()


if __name__ == "__main__":
    main()
