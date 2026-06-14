"""Standard-library HTTP backend for the origami UI.

`WorkspaceController` owns a single simulated `Workspace` and exposes the
folding operations as plain methods that return JSON-serialisable dictionaries.
The `http.server`-based handler wraps that controller in a small REST API and
serves the static single-page frontend from ``origami/ui/static``.

No third-party web framework is used -- only the Python standard library -- so
the UI runs anywhere the framework's core requirements are installed.
"""
from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import numpy as np

from ..geometry import FoldLine
from ..paper import Paper
from ..workspace import Workspace

STATIC_DIR = (Path(__file__).resolve().parent / "static").resolve()

#: Tolerance for treating a point as lying exactly on a fold line.
ON_LINE_TOLERANCE = 1e-9

_CONTENT_TYPES = {
    ".html": "text/html; charset=utf-8",
    ".js": "text/javascript; charset=utf-8",
    ".css": "text/css; charset=utf-8",
    ".json": "application/json; charset=utf-8",
    ".svg": "image/svg+xml",
    ".ico": "image/x-icon",
}


class WorkspaceController:
    """Stateful, thread-safe controller around a simulated `Workspace`.

    The controller keeps the current `origami.paper.Paper` as the single source
    of truth and maintains an undo stack of paper snapshots so the UI can step
    back through operations.  Every mutating method records a snapshot first.

    Parameters
    ----------
    paper_width, paper_height : float, optional
        Initial sheet size in metres.  Defaults to the framework's configured
        A4-ish defaults via `Workspace.simulated`.
    """

    def __init__(self, paper_width: float | None = None, paper_height: float | None = None) -> None:
        self._lock = threading.Lock()
        if paper_width is not None and paper_height is not None:
            paper = Paper.rectangle(paper_width, paper_height, origin=(0.0, 0.0))
            self.workspace = Workspace.simulated(paper=paper)
        else:
            self.workspace = Workspace.simulated()
        self._undo: list[Paper] = []

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #
    @property
    def paper(self) -> Paper:
        return self.workspace.paper

    def _snapshot(self) -> None:
        """Push a deep copy of the current paper onto the undo stack."""
        self._undo.append(self.paper.copy())

    # ------------------------------------------------------------------ #
    # State serialisation
    # ------------------------------------------------------------------ #
    def state(self) -> dict[str, Any]:
        """Return a JSON-serialisable snapshot of the whole workspace."""
        with self._lock:
            return self._state_unlocked()

    def _state_unlocked(self) -> dict[str, Any]:
        paper = self.paper
        landmarks = [
            {"name": name, "x": float(pos[0]), "y": float(pos[1])}
            for name, pos in paper.landmarks.items()
        ]
        folds = [
            {
                "start": [float(f.start[0]), float(f.start[1])],
                "end": [float(f.end[0]), float(f.end[1])],
                "style": f.style,
                "label": f.label,
            }
            for f in paper.folds
        ]
        magnets = [
            {
                "identifier": m.identifier,
                "x": float(m.center[0]),
                "y": float(m.center[1]),
                "orientation": float(m.orientation),
            }
            for m in self.workspace.magnets.placed()
        ]
        return {
            "board": {
                "width": float(self.workspace.board_width),
                "height": float(self.workspace.board_height),
            },
            "paper": {
                "name": paper.name,
                "landmarks": landmarks,
                "polygon": [[float(p[0]), float(p[1])] for p in paper.landmark_array()],
                "centroid": [float(c) for c in paper.centroid()],
                "folds": folds,
            },
            "magnets": magnets,
            "history": list(paper.history),
            "arms": {
                "left": len(self.workspace.left.backend.log),
                "right": len(self.workspace.right.backend.log),
            },
            "can_undo": bool(self._undo),
        }

    # ------------------------------------------------------------------ #
    # Operations
    # ------------------------------------------------------------------ #
    def new_sheet(self, shape: str, width: float, height: float) -> dict[str, Any]:
        """Replace the current paper with a fresh square or rectangle."""
        if width <= 0 or height <= 0:
            raise ValueError("sheet dimensions must be positive")
        with self._lock:
            if shape == "square":
                paper = Paper.square(width, origin=(0.0, 0.0))
            else:
                paper = Paper.rectangle(width, height, origin=(0.0, 0.0))
            self.workspace.paper = paper
            self._undo.clear()
            return self._state_unlocked()

    def fold(self, p1, p2, moving_side: int, style: str = "valley",
             label: str = "") -> dict[str, Any]:
        """Fold along the line through ``p1`` and ``p2``.

        ``moving_side`` is ``+1`` or ``-1`` and selects which half-plane of the
        fold line travels across the crease (see `FoldLine.side_of`).
        """
        line = FoldLine.through_points(p1, p2)
        sign = 1 if moving_side >= 0 else -1
        with self._lock:
            self._snapshot()
            self.paper.fold(
                line,
                moving_region=lambda xy: line.side_of(xy) == sign,
                style=style,
                label=label,
            )
            return self._state_unlocked()

    def rotate(self, angle_deg: float, pivot=None) -> dict[str, Any]:
        """Rotate the whole sheet by ``angle_deg`` degrees (counter-clockwise)."""
        with self._lock:
            self._snapshot()
            self.paper.rotate(np.deg2rad(angle_deg), pivot=pivot)
            return self._state_unlocked()

    def translate(self, dx: float, dy: float) -> dict[str, Any]:
        """Slide the whole sheet by ``(dx, dy)`` metres."""
        with self._lock:
            self._snapshot()
            self.paper.translate([dx, dy])
            return self._state_unlocked()

    def undo(self) -> dict[str, Any]:
        """Restore the most recent snapshot, if any."""
        with self._lock:
            if self._undo:
                self.workspace.paper = self._undo.pop()
            return self._state_unlocked()

    def reset(self) -> dict[str, Any]:
        """Reset to a fresh default sheet and clear history."""
        with self._lock:
            self.workspace = Workspace.simulated()
            self._undo.clear()
            return self._state_unlocked()

    def fold_dart(self) -> dict[str, Any]:
        """Fold the classic paper dart on the current sheet.

        This mirrors `origami.demos.dart` but drives the analytic `Paper`
        model directly, so it works purely in the visualiser without any arm
        choreography.
        """
        with self._lock:
            self._snapshot()
            paper = self.paper
            lower, upper = paper.bounding_box()
            centre_x = (lower[0] + upper[0]) / 2.0
            nose = np.array([centre_x, upper[1]])
            left_mid = np.array([lower[0], (lower[1] + upper[1]) / 2.0])
            right_mid = np.array([upper[0], (lower[1] + upper[1]) / 2.0])

            # 1. Fold the top corners in to the lengthwise centre line.
            paper.fold(FoldLine.through_points(nose, left_mid),
                       moving_region=["top_left"], label="left nose")
            paper.fold(FoldLine.through_points(nose, right_mid),
                       moving_region=["top_right"], label="right nose")

            # 2. Sharpen the nose by folding the new slanted edges in again.
            quarter = lower[0] + (upper[0] - lower[0]) * 0.25
            three_quarter = lower[0] + (upper[0] - lower[0]) * 0.75
            paper.fold(FoldLine.through_points(nose, [quarter, lower[1]]),
                       moving_region=["top_left"], label="left sharpen")
            paper.fold(FoldLine.through_points(nose, [three_quarter, lower[1]]),
                       moving_region=["top_right"], label="right sharpen")

            # 3. Mountain-fold the whole sheet in half.
            centre_line = FoldLine.through_points([centre_x, lower[1]], [centre_x, upper[1]])
            paper.fold(centre_line, moving_region=lambda xy: xy[0] < centre_x - ON_LINE_TOLERANCE,
                       style="mountain", label="in half")
            return self._state_unlocked()


# --------------------------------------------------------------------------- #
# HTTP handler
# --------------------------------------------------------------------------- #
def _make_handler(controller: WorkspaceController) -> type[BaseHTTPRequestHandler]:
    """Build a request-handler class bound to ``controller``."""

    class Handler(BaseHTTPRequestHandler):
        # Quieter logging: one concise line per request.
        def log_message(self, fmt: str, *args: Any) -> None:  # noqa: A003
            return

        # -- helpers ------------------------------------------------------ #
        def _send_json(self, payload: dict[str, Any], status: int = 200) -> None:
            body = json.dumps(payload).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _read_json(self) -> dict[str, Any]:
            length = int(self.headers.get("Content-Length", 0) or 0)
            if not length:
                return {}
            raw = self.rfile.read(length)
            return json.loads(raw.decode("utf-8")) if raw else {}

        def _serve_static(self, rel_path: str) -> None:
            # The frontend lives as a flat set of files in STATIC_DIR, so only
            # the final path component is ever meaningful.  Reducing the request
            # to its basename discards any directory or ``..`` segments and makes
            # path traversal outside STATIC_DIR impossible.
            name = Path(rel_path).name
            if not name:
                name = "index.html"
            target = (STATIC_DIR / name).resolve()
            if target.parent != STATIC_DIR or not target.is_file():
                self._send_json({"error": "not found"}, status=404)
                return
            data = target.read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", _CONTENT_TYPES.get(target.suffix, "application/octet-stream"))
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        # -- routes ------------------------------------------------------- #
        def do_GET(self) -> None:  # noqa: N802
            path = self.path.split("?", 1)[0]
            if path == "/api/state":
                self._send_json(controller.state())
                return
            if path == "/" or path == "":
                self._serve_static("index.html")
                return
            if path.startswith("/static/"):
                self._serve_static(path[len("/static/"):])
                return
            self._send_json({"error": "not found"}, status=404)

        def do_POST(self) -> None:  # noqa: N802
            path = self.path.split("?", 1)[0]
            try:
                body = self._read_json()
                if path == "/api/new":
                    state = controller.new_sheet(
                        str(body.get("shape", "rectangle")),
                        float(body["width"]), float(body["height"]))
                elif path == "/api/fold":
                    state = controller.fold(
                        body["p1"], body["p2"], int(body.get("moving_side", 1)),
                        str(body.get("style", "valley")), str(body.get("label", "")))
                elif path == "/api/rotate":
                    state = controller.rotate(float(body["angle_deg"]), body.get("pivot"))
                elif path == "/api/translate":
                    state = controller.translate(float(body["dx"]), float(body["dy"]))
                elif path == "/api/undo":
                    state = controller.undo()
                elif path == "/api/reset":
                    state = controller.reset()
                elif path == "/api/dart":
                    state = controller.fold_dart()
                else:
                    self._send_json({"error": "not found"}, status=404)
                    return
            except (KeyError, ValueError, TypeError) as exc:
                self._send_json({"error": str(exc)}, status=400)
                return
            self._send_json(state)

    return Handler


def create_server(host: str = "127.0.0.1", port: int = 8000,
                  controller: WorkspaceController | None = None) -> ThreadingHTTPServer:
    """Create (but do not start) the UI HTTP server.

    Parameters
    ----------
    host, port : str, int
        Address to bind.
    controller : WorkspaceController or None, optional
        Controller to drive; a fresh one is created if omitted.

    Returns
    -------
    http.server.ThreadingHTTPServer
    """
    controller = controller or WorkspaceController()
    handler = _make_handler(controller)
    server = ThreadingHTTPServer((host, port), handler)
    server.controller = controller  # type: ignore[attr-defined]
    return server


def run(host: str = "127.0.0.1", port: int = 8000) -> None:
    """Start the UI server and serve until interrupted."""
    server = create_server(host, port)
    url = f"http://{host}:{port}"
    print(f"origami UI running at {url}  (Ctrl-C to stop)")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nshutting down")
    finally:
        server.server_close()
