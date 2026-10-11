"""Correctness checks. Run with `python -m pytest -q` or `python test_dynamics.py`."""
import numpy as np
from dynamics import Params, loss, gradient, grid, simulate, classify, survival_probability
from metrics import power_law_fit, interface_fraction


def test_gradient_matches_finite_differences():
    p, h = Params(eta=1.17), 1e-6
    for u, v in [(0.3, -0.7), (1.7, 0.2), (-0.55, -0.55)]:
        gu, gv = gradient(u, v, p)
        assert abs(gu - (loss(u + h, v, p) - loss(u - h, v, p)) / (2 * h)) < 1e-7
        assert abs(gv - (loss(u, v + h, p) - loss(u, v - h, p)) / (2 * h)) < 1e-7


def test_minimizers_are_fixed_points():
    p = Params()
    for s in (1.0, -1.0):
        m = s * p.minimizer
        u, v, esc = simulate(np.array([m]), np.array([m]), Params(steps=5))
        assert not esc[0] and abs(u[0] - m) < 1e-12 and abs(v[0] - m) < 1e-12


def test_projected_dynamics():
    """GD on (u, v) induces a closed map on (P, w) = (uv, u^2 + v^2) (Proposition 3 of the paper)."""
    p = Params(eta=1.17, steps=1)
    rng = np.random.default_rng(1)
    u0, v0 = rng.normal(size=50), rng.normal(size=50)
    u1, v1, _ = simulate(u0, v0, p)
    P, w = u0 * v0, u0 ** 2 + v0 ** 2
    r, a = P - p.y, 1 - p.eta * p.lam
    P1 = a * a * P - p.eta * a * r * w + p.eta ** 2 * r * r * P
    w1 = (a * a + p.eta ** 2 * r * r) * w - 4 * p.eta * a * r * P
    assert np.allclose(u1 * v1, P1) and np.allclose(u1 ** 2 + v1 ** 2, w1)


def test_zero_noise_equals_deterministic():
    U, V = grid((-3, 3, -3, 3), 41)
    _, _, e0 = simulate(U, V)
    for mode in ("step", "additive"):
        _, _, e1 = simulate(U, V, mode=mode, sigma=0.0)
        assert np.array_equal(e0, e1)


def test_point_symmetry():
    U, V = grid((-3, 3, -3, 3), 41)
    _, _, esc = simulate(U, V)
    assert np.array_equal(esc, esc[::-1, ::-1])  # (u, v) -> (-u, -v) leaves the loss invariant


def test_reproducible_and_zero_noise_probabilities_are_binary():
    U, V = grid((-3, 3, -3, 3), 21)
    a = survival_probability(U, V, Params(), "step", 0.1, trials=8, seed=7)
    b = survival_probability(U, V, Params(), "step", 0.1, trials=8, seed=7)
    assert np.array_equal(a, b)
    z = survival_probability(U, V, Params(steps=100), "step", 0.0, trials=4)
    assert set(np.unique(z)) <= {0.0, 1.0}


def test_power_law_fit_recovers_exponent():
    x = np.array([0.01, 0.02, 0.05, 0.1])
    f = power_law_fit(x, 3.0 * x ** 0.7)
    assert abs(f["exponent"] - 0.7) < 1e-12 and f["r2"] > 0.999999


def test_interface_fraction():
    m = np.zeros((4, 4), bool); m[:, 2:] = True
    assert abs(interface_fraction(m) - 4 / 24) < 1e-12


if __name__ == "__main__":
    tests = [f for k, f in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
    print(f"PASS {len(tests)} tests")
