# `origami/` — paper-folding framework for the dual-UR rig

A framework for folding paper airplanes (starting with a basic **dart**) using
two UR arms, a Robotiq gripper, a magnetic board, and 3D-printed magnet weights.

The core idea: **stop hand-tuning 6-DOF robot poses.** You work in *board
coordinates* — a 2D `(x, y)` frame on the board surface plus a *height above the
board* — and the framework converts to UR TCP poses, drives the arms, and keeps
an analytic model of the **paper** and **magnets** in sync after every fold,
rotation, or move.

All transform/pose algebra is built on **`spatialmath`** (`SE2`/`SE3`/`SO3`,
exp/log maps for the UR rotation-vector convention) with **`scipy`** for
calibration fitting — not hand-rolled.

## Layout

| Module | Responsibility |
|---|---|
| `geometry.py` | `spatialmath`/`scipy`-based core: `FoldLine` (reflect/side/project), pose ↔ `SE3`, rigid-transform fit, polygon helpers |
| `calibration.py` | `BoardCalibration`: fit board→base `SE3`; build downward-gripper TCP poses at an explicit height |
| `paper.py` | `Paper`: named landmarks + `Fold` records, with a single `fold()` plus `rotate`/`translate` |
| `magnets.py` | `Magnet`, `BlockMagnet`, `LBracketMagnet` (foot center + handle offset/height + hinge), `MagnetRegistry` |
| `backends.py` | `RTDEArmBackend`/`RobotiqGripperBackend` (real) and `Simulated*` (in-memory) backends |
| `arm.py` | `Arm`: `move_to_board_point`, `hover_above`, `press_onto_board`, `grip`/`release` — heights always explicit |
| `workspace.py` | `Workspace`: both arms + paper + magnets + board |
| `actions.py` | choreography: `fold_flap_over`, `rotate_sheet`, `place_magnet`, `remove_magnet` |
| `config.py` | IPs, board size, taught corner poses → calibrations |
| `demos/dart.py` | end-to-end dart recipe, runnable in simulation |

## Environment

The framework imports `spatialmath` directly (assumed always available). On this
machine, the **system** Python can't import `spatialmath` because a Debian
`matplotlib-3.5.1-nspkg.pth` forces a stale `mpl_toolkits` onto the path, which
breaks `spatialmath`'s import. A dedicated virtualenv avoids the whole problem:

```bash
python3 -m virtualenv .venv          # (or: python3 -m venv .venv)
. .venv/bin/activate
pip install -r origami/requirements.txt
```

`.venv/` is git-ignored. Everything below assumes that venv is active.

## Quick start (simulation — no robots)

```bash
python -m origami.demos.dart         # fold a dart; prints paper + arm logs
```

```python
from origami import Workspace, FoldLine, actions

ws = Workspace.simulated()           # uses the taught calibrations in config.py
p  = ws.paper                         # a square sheet centred on the board

# Fold the right half over the vertical centre line:
lo, hi = p.bounding_box(); cx = (lo[0]+hi[0])/2
actions.fold_flap_over(
    ws, FoldLine.through_points([cx, lo[1]], [cx, hi[1]]),
    moving_region=["bottom_right", "top_right"],
    anchoring_arm="left", folding_arm="right",
)
print(p)                              # landmarks updated analytically
```

## Running on hardware

```python
from origami import Workspace
ws = Workspace.hardware()             # connects both arms + the Robotiq gripper
ws.right.descend_and_press(0.18, 0.13)        # board coords -> pose -> moveL
ws.right.grip()
```

The same `actions.*` recipes run in both modes — only the backend differs.

## Heights are explicit (not glued to the board)

`Arm.move_to_board_point(x, y, height_above_board, tool_rotation=0.0)` always
takes a height. Only `press_onto_board` targets the surface (that's its job).
Magnets carry their own grip height, so picking up an L-bracket grabs its *raised
handle*, not the board:

```python
from origami import LBracketMagnet, actions

bracket = LBracketMagnet("hinge_a", center=[0.10, 0.10], orientation=1.57,
                         handle_offset=0.04, handle_height=0.03,
                         tray_position=(0.02, 0.02))
actions.place_magnet(ws, bracket, x=0.18, y=0.13, orientation=1.57)
hinge = bracket.hinge_line()          # fold the paper about the pinned edge
```

## Calibration workflow

1. Jog each arm to the four board corners and record `getActualTCPPose()`
   (see `mvmt/get.py` / `mvmt/record.py`).
2. Paste those poses into `config.LEFT_ARM_CORNERS` / `RIGHT_ARM_CORNERS`.
3. `BoardCalibration.from_taught_corners(...)` fits a best-fit rigid `board→base`
   transform via `scipy` Procrustes. Check `calib.fit_residuals(...)`; the
   current hand-taught corners sit within ~1 cm (re-teach the worst corner to
   tighten). Three non-collinear corners are the minimum; four improve the fit.

## Physical-world ideas for consistency

- **Registration hard-stops** on the board so the paper's start pose is
  repeatable run-to-run (a fixed origin keeps the `Paper` model trustworthy).
- **Calibration jig**: three permanently marked board points to recompute each
  arm's transform after any tool/board change.
- **Magnet home tray + "graveyard"** at fixed board coords (`Magnet.tray_position`).
- **L-bracket magnets as hinges** (`LBracketMagnet.hinge_line()`) to pivot a flap
  about a known edge; **block magnets** to weigh layers flat while the other arm
  creases.
- **A dedicated creasing-tool TCP** (a second `BoardCalibration` with a different
  tool offset) for sharp, repeatable creases.
- **Two-arm division of labour**: one arm anchors near the crease, the other
  carries and presses the flap — exactly what `actions.fold_flap_over` does.
