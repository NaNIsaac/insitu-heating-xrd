import numpy as np
import pytest

from insitu_xrd.avrami import avrami_fit, hold_range, jmak_fit, local_n

N, K = 2.5, 0.2          # Avrami exponent, rate constant [1/min]


def jmak(t, t0=0.0):
    return 1 - np.exp(-np.clip(K * (t - t0), 0, None) ** N)


def test_avrami_fit_exact_curve():
    tm = np.linspace(0.1, 20, 300)
    r = avrami_fit(tm, jmak(tm), (0.15, 0.85))
    assert r["n"] == pytest.approx(N, rel=1e-6)
    assert r["k"] == pytest.approx(K, rel=1e-6)
    assert r["r2"] == pytest.approx(1.0)
    assert r["npts"] > 50


def test_avrami_fit_too_few_points():
    tm = np.linspace(0.1, 20, 4)
    assert np.isnan(avrami_fit(tm, jmak(tm), (0.15, 0.85))["n"])


def test_jmak_fit_free_t0():
    t = np.linspace(0, 25, 250)
    rng = np.random.default_rng(1)
    X = jmak(t, t0=3.0) + rng.normal(0, 0.005, t.size)
    r = jmak_fit(t, X)
    assert r["n_jmak"] == pytest.approx(N, rel=0.05)
    assert r["k_jmak"] == pytest.approx(K, rel=0.02)
    assert r["t0_fit"] == pytest.approx(3.0, abs=0.2)


def test_jmak_fit_fixed_t0():
    t = np.linspace(0, 25, 250)
    r = jmak_fit(t, jmak(t, t0=3.0), t0_fixed=3.0)
    assert r["n_jmak"] == pytest.approx(N, rel=1e-3)
    assert r["k_jmak"] == pytest.approx(K, rel=1e-3)
    assert r["t0_fit"] == 3.0


def test_local_n_constant_for_jmak():
    tm = np.linspace(0.1, 20, 300)
    Xs, n = local_n(tm, jmak(tm))
    mid = (Xs > 0.1) & (Xs < 0.9)
    np.testing.assert_allclose(n[mid], N, rtol=0.02)


def test_hold_range():
    T = np.concatenate([np.linspace(25, 370, 50), np.full(100, 370.0), np.linspace(370, 50, 30)])
    a, b = hold_range(T)
    assert T[a] >= 367 and T[a - 1] < 367
    assert a <= 50 and b >= 149
    assert hold_range(np.full(50, 370.0)) is None      # no ramp → no hold detection
