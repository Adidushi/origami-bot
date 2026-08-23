# `origami/` — paper-folding framework for the dual-UR5e rig

This project contains the code to fold a plane using two UR5e arm robots.  
Additionally, it contains the necessary framework to create new patterns for different folds.

## Layout

| Module                | Responsibility                                                                                                          |
|-----------------------|-------------------------------------------------------------------------------------------------------------------------|
| `actions.py`          | choreography: `fold_flap_over`, `rotate_sheet`, `place_magnet`, `remove_magnet`, etc.                                   |
| `arm.py`              | `Arm`: `move_to_world`, `move_to_tcp`, `move_offset_world`, `grip`/`release`                                            |
| `backends.py`         | `RTDEArmBackend`/`RobotiqGripperBackend`                                                                                |
| `calibrate.py`        | Runs calibration method to find new board and platform positions                                                        |
| `calibration.json`    | Taught board positions created by `calibrate.py`                                                                        |
| `config.py`           | IPs, board size, taught corner poses → calibrations                                                                     |
| `coords.py`           | TCP to World frame conversion                                                                                           |
| `geometry.py`         | `spatialmath`/`scipy`-based core: `FoldLine` (reflect/side/project), pose ↔ `SE3`, rigid-transform fit, polygon helpers |
| `magnets.py`          | `Magnet`, `BlockMagnet`, `LBracketMagnet` (anchor_xy + grip_xy + grip height + hinge), `MagnetRegistry`                 |
| `paper.py`            | `Paper`: named landmarks + `Fold` records, with a single `fold()` plus `rotate`/`translate`                             |
| `rotation.py`         | Handles absolute and relative rotations via rotation vector and matrix math                                             |
| `workspace.py`        | `Workspace`: both arms + paper + magnets + board                                                                        |
| `demos/fold_plane.py` | end-to-end plane recipe, runnable on hardware                                                                           |


Additional components include `mvmt/robotq_gripper.py` which is a thin wrapper for the actual gripper tools.

## Environment

We recommend using a virtual environment to prevent conflicts.

```bash
python3 -m virtualenv .venv
. .venv/bin/activate
```
Everything below assumes that venv is active.

## Calibration
Calibration uses the `origami/calibration.json` file as a baseline, and can be adjusted to match any new board
and magnet platform position.  
Use `w/a/s/d/q/e` for 3d position adjustment.  
After calibration, the updated positions will be saved to the calibration file.

## Physical Environment
The physical environment is defined by a "world space", using positive X as the horizontal board direction,
and positive Y as the vertical board direction, where 0,0 is the bottom-left corner of the board.
The Z axis remains the vertical axis.  
This unifies both arms into one comfortable coordinate system, allowing for cross-hand coordination.  
The general workflow is:
- Place paper in comfortable position
- Place magnets to support fold
- Fold
- Crease the fold  
Of course, there are many instances where these guidelines do not apply (for instance, folding in half and re-opening!)

## Requirements
Physical implements required for the system are:
- 1 Main work platform
- 1 Magnet holder platform
  - 2 block magnets (`block_a`, `block_b`)
  - 1 L-Bracket magnet (`lbracket_a`)
  - 1 Creaser card
3d models for all printable objects are included in the project under `models/`

Software requirements are under `origami/requirements.txt` and can be installed as so:
```bash
pip install -r origami/requirements.txt
```

## Quick start
After calibration, place a piece of paper centered on the board, flush to the top (overhanging on the bottom) and run
the demo:
```bash
python -m origami.demos.fold_plane
```
The file can be run with several parameters:
```bash
--speed=1 (any value in (0,1], global speed multiplier)
--go (disables action confirmation, without this flag each step requires manual input)
```
