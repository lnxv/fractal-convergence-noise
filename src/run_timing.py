"""Run the timing experiment (baseline + robustness runs), then the analysis and figures.

Runs from inside results/timing/ so outputs land there:  cd results/timing && python ../../src/run_timing.py
Roughly 1-2 hours on a laptop; each run_set call can also be run on its own.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from exp_timing import run_set  # noqa: E402

JOBS = [
    ("base_init", 200, 64, [("init", None)]), ("base_early10", 200, 64, [("early", 10)]),
    ("base_early50", 200, 64, [("early", 50)]), ("base_persist", 200, 64, [("persist", None)]),
    ("long_init", 600, 64, [("init", None)]), ("long_early10", 600, 64, [("early", 10)]),
    ("long_persist", 600, 64, [("persist", None)]),
    ("more_init", 200, 192, [("init", None)]), ("more_early10", 200, 192, [("early", 10)]),
    ("more_persist", 200, 192, [("persist", None)]),
]

if __name__ == "__main__":
    for tag, T, n, groups in JOBS:
        run_set(tag, T=T, trials=n, res=101, groups=groups)
    import analyze_timing
    analyze_timing.main()
