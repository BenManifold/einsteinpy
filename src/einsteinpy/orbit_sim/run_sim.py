#!/usr/bin/env python
"""Launch the black hole orbit simulation.

Run with::

    pip install -e ".[sim]"   # install PyQtGraph, PySide6, PyOpenGL
    python -m einsteinpy.orbit_sim.run_sim

Debug logging::

    python -m einsteinpy.orbit_sim.run_sim --debug
    # or
    EINSTEINPY_ORBIT_DEBUG=1 python -m einsteinpy.orbit_sim.run_sim

Or::

    from einsteinpy.orbit_sim import run
    run()
"""

import argparse
import logging
import os
import sys


def _setup_logging(debug=False):
    """Configure logging for the orbit sim."""
    level = logging.DEBUG if debug else logging.INFO
    fmt = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    logging.basicConfig(
        level=level,
        format=fmt,
        datefmt="%H:%M:%S",
        stream=sys.stderr,
        force=True,
    )
    log = logging.getLogger("einsteinpy.orbit_sim")
    log.setLevel(level)
    if debug:
        log.debug("Debug logging enabled")


def _dbg(evt, hyp, data=None):
    import json
    import time
    with open("/Users/benjaminmanifold/einsteinpy/.cursor/debug-08691c.log", "a") as f:
        f.write(json.dumps({"sessionId": "08691c", "hypothesisId": hyp, "location": "run_sim.py", "message": evt, "data": data or {}, "timestamp": int(time.time() * 1000)}) + "\n")

def run(debug=None):
    """Launch the orbit simulation application."""
    _dbg("run_sim.run() entered", "A")
    if debug is None:
        debug = os.environ.get("EINSTEINPY_ORBIT_DEBUG", "").lower() in ("1", "true", "yes")
    _setup_logging(debug=debug)
    _dbg("after _setup_logging", "D")

    _dbg("before import main", "A")
    from .main import run as _run
    _dbg("after import main", "A")

    _dbg("before _run()", "B")
    return _run()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Black hole orbit simulation")
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging to stderr",
    )
    args = parser.parse_args()
    run(debug=args.debug)
