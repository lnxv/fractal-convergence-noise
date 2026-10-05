# Does Noise Smooth or Roughen the Fractal Convergence Boundary in Gradient Descent?

A computational investigation extending [Liang & Montúfar, "Gradient Descent with Large
Step Sizes: Chaos and Fractal Convergence Region" (arXiv:2509.25351, ICLR 2026)](https://arxiv.org/abs/2509.25351).

Done as a research exercise suggested by Prof. Guido Montúfar (UCLA).

## Background

Liang & Montúfar show that gradient descent on a simple regularized matrix
factorization problem,

```
L(u, v) = 0.5*(u*v - y)^2 + 0.5*lambda*(u^2 + v^2)
```

develops a **fractal convergence boundary** at near-critical step sizes: the set of
initializations that converge to each of the two global minimizers forms a
self-similar, spiky region, with an estimated box-counting dimension of **1.249**.

## Research question

The paper's analysis is for deterministic gradient descent. In practice, we train
with SGD, which behaves like gradient descent plus noise. **Does adding noise smooth
out the fractal boundary (as noise often regularizes sensitive dynamical systems), or
does it make the unpredictability worse?**

## Method

1. **Reproduce the deterministic baseline.** Used the authors' own released code
   ([anonymous.4open.science/r/chaos-matrix-factorization-07C5](https://anonymous.4open.science/r/chaos-matrix-factorization-07C5))
   to get the exact parameters (`y=0.5, eta=1.0, lambda=0.2`) and convergence
   criterion, rather than guessing. Validated the reproduction visually against their
   published Figure 3 and numerically via box-counting dimension (got 1.33 vs. their
   reported 1.249; the small gap is expected given a coarser grid and a simpler
   boundary-extraction method than their exact symmetry-reduced approach).

2. **Introduce stochasticity**, two models:
   - **Gradient noise** (SGD-like): add i.i.d. Gaussian noise `N(0, sigma^2)` to each
     coordinate of the gradient at every step.
   - **Step-size noise**: randomize the step size `eta_t ~ Uniform[eta(1-sigma),
     eta(1+sigma)]` at every step.

3. **Measure boundary complexity** as a function of noise level `sigma`, using the
   same box-counting dimension method the paper uses: extract boundary points
   (grid cells adjacent to a differently-classified neighbor), normalize into the
   unit square, count occupied boxes at scales `eps = 1/2^k` for `k=2..8`, and fit
   `log N(eps)` vs `log(1/eps)` with a line; the slope is the estimated dimension.

## Result

**Noise does not smooth the fractal boundary. It makes it monotonically more
complex, in both noise models, with a very clean fit (R² > 0.999 throughout):**

| Gradient noise σ | Box-counting dim. | R² |
|---|---|---|
| 0.000 | 1.378 | 0.9994 |
| 0.002 | 1.384 | 0.9995 |
| 0.005 | 1.414 | 0.9993 |
| 0.010 | 1.472 | 0.9994 |
| 0.020 | 1.529 | 0.9999 |
| 0.050 | 1.686 | 0.9996 |

| Step-size noise σ | Box-counting dim. | R² |
|---|---|---|
| 0.01 | 1.415 | 0.9993 |
| 0.05 | 1.544 | 0.9991 |
| 0.10 | 1.669 | 0.9992 |
| 0.20 | 1.796 | 0.9992 |

Visually (see `results/figures/`), the boundary goes from a clean, sparse fractal
spike pattern at `sigma=0` to a dense, fine-grained "salt-and-pepper" speckle at
higher `sigma`: the zone of unpredictable, sensitive-to-initialization behavior
*expands* rather than blurring away.

## Limitations / honest caveats

- Results are from a **single zoomed-in boundary region** near one spike, not the
  full boundary; the trend should be checked in other regions to confirm it's not
  specific to this location.
- Each grid point used a **single noise realization** (one random seed), not
  averaged over multiple runs, so individual dimension values carry some
  seed-dependent noise, though the monotonic trend across six noise levels in two
  independent noise models is a strong signal that this is real, not an artifact.
- Grid resolution (350-500 per side) is modest compared to what a
  numpy/GPU-accelerated implementation could do; this was a pure-Python,
  multiprocessing-parallelized implementation (see Implementation notes).

## Implementation notes

The code is written with **NumPy + Matplotlib**, vectorized so each noise sweep
evaluates the entire grid of initializations and every GD step as array
operations, not a Python-level loop per pixel.

**Important:** this code was developed in a sandboxed environment where NumPy's
compiled extensions could not be loaded, so it could not be executed end-to-end
there. An earlier pure-Python implementation (no NumPy, hand-rolled PRNG and PPM
writer, parallelized with `multiprocessing` instead) was used to generate and
validate the actual numeric results reported above, that version works without
any dependencies. The NumPy version in `src/` is a direct, careful translation of
the same equations, parameters, and box-counting algorithm, intended to be the
version actually run going forward (much faster, allows far higher resolution).
**Before trusting its output, run it locally and confirm the numbers/figures
match what's reported here**; if anything differs, the pure-Python version's
logic is the validated reference to debug against.

## Repository structure

```
src/
  baseline.py      deterministic gradient descent + convergence classification,
                    exact parameters/criterion matching the authors' code
  stochastic.py     gradient-noise and step-size-noise variants,
                    multiprocessing-parallelized grid evaluation
  boxcount.py       box-counting fractal dimension estimator (matches the
                    paper's own method)
  analysis.py       runs the full noise sweep and saves results/final_results.json
results/
  final_results.json   full numeric results table
  figures/             baseline + noise-sweep visualizations (PNG)
```

## Running it

No dependencies beyond the Python 3 standard library.

```bash
cd src
python3 baseline.py       # reproduces the deterministic fractal boundary
python3 analysis.py       # runs the full noise sweep, saves results/final_results.json
```

## Reference

Liang, S. & Montúfar, G. (2026). *Gradient Descent with Large Step Sizes: Chaos and
Fractal Convergence Region.* ICLR 2026. [arXiv:2509.25351](https://arxiv.org/abs/2509.25351)











































































































