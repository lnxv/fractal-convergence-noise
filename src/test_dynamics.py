"""Correctness checks. Run with `python -m pytest -q src` or `python src/test_dynamics.py`."""
import numpy as np
from dynamics import (Params, loss, gradient, grid, simulate, outcome_probabilities,
                      survival_probability, projected_map, critical_step_size)
from metrics import power_law_fit, interface_fraction, erode, box_counting_dimension, selection_uncertain_fraction


def test_gradient_matches_finite_differences():
    p, h = Params(eta=1.17), 1e-6
    for u, v in [(0.3, -0.7), (1.7, 0.2), (-0.55, -0.55)]:
        gu, gv = gradient(u, v, p)
        assert abs(gu - (loss(u + h, v, p) - loss(u - h, v, p)) / (2 * h)) < 1e-7
        assert abs(gv - (loss(u, v + h, p) - loss(u, v - h, p)) / (2 * h)) < 1e-7


def test_minimizers_are_fixed_points_also_under_step_noise():
    p = Params(steps=50)
    for s in (1.0, -1.0):
        m = s * p.minimizer
        for mode, sigma in (("deterministic", 0.0), ("step", 0.5)):
            u, v, esc = simulate(np.array([m]), np.array([m]), p, mode, sigma)
            assert not esc[0] and abs(u[0] - m) < 1e-12 and abs(v[0] - m) < 1e-12


def test_projected_dynamics():
    """GD on (u, v) induces a closed map on (P, w) = (uv, u^2 + v^2) (Proposition 3 of the paper)."""
    p = Params(eta=1.17, steps=1)
    rng = np.random.default_rng(1)
    u0, v0 = rng.normal(size=50), rng.normal(size=50)
    u1, v1, _ = simulate(u0, v0, p)
    P1, w1 = projected_map(u0 * v0, u0 ** 2 + v0 ** 2, p)
    assert np.allclose(u1 * v1, P1) and np.allclose(u1 ** 2 + v1 ** 2, w1)


def test_sgd_gradient_is_unbiased():
    """The two per-sample gradients of the SGD model average to the full gradient."""
    p, s = Params(), 0.3
    u, v = np.array([0.4, 2.1, -1.3]), np.array([1.2, -0.2, 0.7])
    g_lo, g_hi, g = gradient(u, v, p, y=p.y - s), gradient(u, v, p, y=p.y + s), gradient(u, v, p)
    assert np.allclose((g_lo[0] + g_hi[0]) / 2, g[0]) and np.allclose((g_lo[1] + g_hi[1]) / 2, g[1])


def test_zero_noise_equals_deterministic():
    U, V = grid((-3, 3, -3, 3), 41)
    _, _, e0 = simulate(U, V)
    for mode in ("step", "additive", "sgd"):
        _, _, e1 = simulate(U, V, mode=mode, sigma=0.0)
        assert np.array_equal(e0, e1)


def test_point_symmetry():
    U, V = grid((-3, 3, -3, 3), 41)
    _, _, esc = simulate(U, V)
    assert np.array_equal(esc, esc[::-1, ::-1])  # (u, v) -> (-u, -v) leaves the loss invariant


def test_probabilities_reproducible_consistent_and_binary_without_noise():
    U, V = grid((-3, 3, -3, 3), 21)
    a = outcome_probabilities(U, V, Params(), "sgd", 0.1, trials=8, seed=7)
    b = outcome_probabilities(U, V, Params(), "sgd", 0.1, trials=8, seed=7)
    assert all(np.array_equal(a[k], b[k]) for k in a)
    assert np.all(a["plus"] + a["minus"] <= a["survive"] + 1e-12)
    z = survival_probability(U, V, Params(steps=100), "step", 0.0, trials=4)
    assert set(np.unique(z)) <= {0.0, 1.0}


def test_critical_step_size_formula():
    # Theorem 1 at y = 0 reduces to 4 / (|u|^2 + |v|^2).
    u0, v0 = np.array([1.0, 0.5]), np.array([1.0, -2.0])
    assert np.allclose(critical_step_size(u0, v0, 0.0), 4.0 / (u0 ** 2 + v0 ** 2))


def test_metrics():
    x = np.array([0.01, 0.02, 0.05, 0.1])
    f = power_law_fit(x, 3.0 * x ** 0.7)
    assert abs(f["exponent"] - 0.7) < 1e-12 and f["r2"] > 0.999999
    m = np.zeros((4, 4), bool); m[:, 2:] = True
    assert abs(interface_fraction(m) - 4 / 24) < 1e-12
    sq = np.zeros((7, 7), bool); sq[1:6, 1:6] = True
    assert erode(sq).sum() == 9
    t = np.linspace(0, 1, 20000)
    assert abs(box_counting_dimension(np.c_[t, 0.3 * t])["exponent"] - 1) < 0.02
    g = np.random.default_rng(0).uniform(size=(400000, 2))
    assert abs(box_counting_dimension(g)["exponent"] - 2) < 0.05
    probs = {"survive": np.array([1.0, 1.0, 0.2]), "plus": np.array([0.5, 1.0, 0.1]), "minus": np.array([0.5, 0.0, 0.1])}
    assert abs(selection_uncertain_fraction(probs) - 1 / 3) < 1e-12


if __name__ == "__main__":
    tests = [f for k, f in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
    print(f"PASS {len(tests)} tests")
