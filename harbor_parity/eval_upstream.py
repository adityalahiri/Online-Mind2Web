#!/usr/bin/env python3
"""Run upstream src/run.py unchanged, but with the 'fork' multiprocessing start method.

On macOS Python defaults to 'spawn', and run.py's OpenaiEngine (holding a thread lock)
cannot be pickled into the worker processes. Linux already defaults to 'fork', so this
shim only matters on macOS; it does not alter any evaluation logic. Same CLI as run.py.
"""
import multiprocessing
import runpy
import sys
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src"
multiprocessing.set_start_method("fork", force=True)
sys.path.insert(0, str(SRC))
sys.argv[0] = str(SRC / "run.py")
runpy.run_path(str(SRC / "run.py"), run_name="__main__")
