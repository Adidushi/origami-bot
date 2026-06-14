"""Interactive web UI for the origami folding framework.

The UI is a thin, dependency-free layer on top of a *simulated*
`origami.workspace.Workspace`.  It serves a single-page canvas application that
visualises the board, the current `origami.paper.Paper` polygon, its creases and
landmarks, and lets you drive the analytic model interactively -- start a new
sheet, fold by clicking, rotate, translate, undo and fold a dart -- without any
robot hardware.

Run it with::

    python -m origami.ui          # then open http://127.0.0.1:8000

Everything is built on the Python standard library (`http.server`), so no extra
packages beyond the framework's own requirements are needed.
"""
from __future__ import annotations

from .server import WorkspaceController, create_server, run

__all__ = ["WorkspaceController", "create_server", "run"]
