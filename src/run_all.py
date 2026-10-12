"""Run every experiment and build every figure.

    python src/run_all.py            # full run (about 30 minutes on one core)
    QUICK=1 python src/run_all.py    # smaller grids and fewer runs (a few minutes)

Steps, following the plan suggested by G. Montufar:
  1. exp_reproduce : recover the deterministic phenomena of the paper
  2. exp_windows   : add noise gradually (step-size, additive, SGD) near the boundary and measure the change
  3. exp_global    : larger noise over the whole domain
  4. exp_controls  : resolution, horizon and noise-coupling checks
  5. figures
Outputs are written to results/ (or $OUT).
"""
import runpy

for module in ("exp_reproduce", "exp_windows", "exp_global", "exp_controls", "figures"):
    print(f"=== {module}", flush=True)
    runpy.run_module(module, run_name="__main__")
