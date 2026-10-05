# Pure-Python reference implementation

This is the original implementation, pure Python standard library only (no
dependencies), that actually produced the numbers reported in the main
README. Written this way because the development sandbox blocked NumPy's
compiled extensions. Parallelized with `multiprocessing` to stay fast.

Run it with: `python3 baseline.py` or `python3 analysis.py` (no pip installs
needed). Output is PPM images; convert to PNG with `sips -s format png
file.ppm --out file.png` (macOS) or `pip install pillow` + a one-line script
on other platforms.

Kept here as a dependency-free fallback and as the validated reference to
debug the numpy version (`../src/`) against if its output ever looks off.











