"""Entry point: ``python -m origami.ui``.

Starts the interactive folding UI on a local HTTP server.  Options::

    python -m origami.ui --host 0.0.0.0 --port 8080
"""
from __future__ import annotations

import argparse

from .server import run


def main() -> None:
    parser = argparse.ArgumentParser(description="Interactive origami folding UI (simulation).")
    parser.add_argument("--host", default="127.0.0.1", help="address to bind (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=8000, help="port to bind (default: 8000)")
    args = parser.parse_args()
    run(args.host, args.port)


if __name__ == "__main__":
    main()
