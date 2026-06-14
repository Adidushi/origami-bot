"""Fold a basic paper dart -- an end-to-end demo that runs in simulation.

Run it with::

    python -m origami.demos.dart

The classic dart recipe (nose pointing toward ``+y``):

1. Fold the two top corners in to the lengthwise centre line, forming the nose.
2. Fold the new slanted top edges in to the centre line again, sharpening it.
3. Mountain-fold the whole thing in half.

(The wing folds follow the same pattern and are left as a follow-up.)

Each step drives the arms (simulated here) via `origami.actions` *and*
updates the analytic `Paper` model, so afterwards
``workspace.paper`` reflects exactly where every landmark ended up.
"""
from __future__ import annotations

import numpy as np

from .. import actions
from ..geometry import FoldLine
from ..workspace import Workspace


def fold_dart(workspace: Workspace | None = None, verbose: bool = True) -> Workspace:
    """Fold a paper dart, in simulation by default.

    Parameters
    ----------
    workspace : origami.workspace.Workspace or None, optional
        Workspace to fold in.  A fresh simulated workspace is created if ``None``.
    verbose : bool, optional
        If ``True`` (default), print the landmark positions after each stage.

    Returns
    -------
    origami.workspace.Workspace
        The workspace, with its paper folded and arm command logs populated.
    """
    workspace = workspace or Workspace.simulated()
    paper = workspace.paper

    lower, upper = paper.bounding_box()
    centre_x = (lower[0] + upper[0]) / 2.0
    nose = np.array([centre_x, upper[1]])               # top-centre point (the nose)
    left_mid = np.array([lower[0], (lower[1] + upper[1]) / 2.0])
    right_mid = np.array([upper[0], (lower[1] + upper[1]) / 2.0])

    def show(message: str) -> None:
        if verbose:
            print(f"\n# {message}")
            for name, position in paper.landmarks.items():
                print(f"    {name:12s} {np.round(position, 4).tolist()}")

    show("start")

    # 1. Fold the top corners in to the lengthwise centre line.
    actions.fold_flap_over(
        workspace, FoldLine.through_points(nose, left_mid), moving_region=["top_left"],
        anchoring_arm="right", folding_arm="left", label="left nose",
    )
    actions.fold_flap_over(
        workspace, FoldLine.through_points(nose, right_mid), moving_region=["top_right"],
        anchoring_arm="left", folding_arm="right", label="right nose",
    )
    show("after corner folds (top_left / top_right now on the centre line)")

    # 2. Fold the slanted top edges in to the centre line, sharpening the nose.
    quarter = lower[0] + (upper[0] - lower[0]) * 0.25
    three_quarter = lower[0] + (upper[0] - lower[0]) * 0.75
    actions.fold_flap_over(
        workspace, FoldLine.through_points(nose, [quarter, lower[1]]),
        moving_region=["top_left"], anchoring_arm="right", folding_arm="left",
        label="left sharpen",
    )
    actions.fold_flap_over(
        workspace, FoldLine.through_points(nose, [three_quarter, lower[1]]),
        moving_region=["top_right"], anchoring_arm="left", folding_arm="right",
        label="right sharpen",
    )
    show("after sharpening folds")

    # 3. Mountain-fold the whole sheet in half along the centre line.
    centre_line = FoldLine.through_points([centre_x, lower[1]], [centre_x, upper[1]])
    paper.fold(centre_line, moving_region=lambda xy: xy[0] < centre_x - 1e-9,
               style="mountain", label="in half")
    show("after folding in half")

    if verbose:
        print(f"\nFolds formed:    {[f.label for f in paper.folds]}")
        print(f"Left-arm moves:  {len(workspace.left.backend.log)}")
        print(f"Right-arm moves: {len(workspace.right.backend.log)}")
    return workspace


if __name__ == "__main__":
    fold_dart()
