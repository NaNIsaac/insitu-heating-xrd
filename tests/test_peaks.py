import numpy as np
import pytest

from insitu_xrd.compare import peak_window
from insitu_xrd.peaks import detect_peaks, fit_peak
from insitu_xrd.tracks import level_temperature, window_area

X = np.arange(10.0, 30.0, 0.02)


def gauss(x, amp, mu, fwhm):
    return amp * np.exp(-0.5 * ((x - mu) / (fwhm / 2.3548)) ** 2)


def test_fit_peak_recovers_gaussian_on_linear_background():
    rng = np.random.default_rng(0)
    y = gauss(X, 50, 18.4, 0.2) + 5 + 0.1 * X + rng.normal(0, 0.3, X.size)
    f = fit_peak(X, y, 18.35)
    assert f.center == pytest.approx(18.4, abs=0.005)
    assert f.fwhm == pytest.approx(0.2, rel=0.05)
    assert 0 < f.center_err < 0.01


def test_fit_peak_rejects_bad_windows():
    assert np.isnan(fit_peak(X, np.full(X.size, np.nan), 18.4).center)
    assert np.isnan(fit_peak(X, 5 - gauss(X, 50, 18.4, 0.2), 18.4).center)  # dip, not a peak


def test_detect_peaks_skips_gap():
    y = gauss(X, 40, 15.0, 0.2) + gauss(X, 25, 22.0, 0.25) + 10
    y[(X > 25) & (X < 26)] = np.nan
    _, pk = detect_peaks(X, y)
    np.testing.assert_allclose(X[pk], [15.0, 22.0], atol=0.03)


def test_peak_window_fit_and_fallback():
    y = gauss(X, 40, 21.4, 0.3) + 2
    c, fwhm, hw = peak_window(X, y, 21.3, 0.6, 1.5)
    assert c == pytest.approx(21.4, abs=0.005)
    assert fwhm == pytest.approx(0.3, rel=0.03)
    assert hw == pytest.approx(1.5 * fwhm)


def test_window_area_removes_linear_background():
    y = gauss(X, 10, 20.0, 0.3) + 3 + 0.2 * X
    area = window_area(X, y, 20.0, 0.6)
    assert area == pytest.approx(10 * 0.3 / 2.3548 * np.sqrt(2 * np.pi), rel=0.02)


def test_level_temperature_interpolates_first_rise():
    T = np.linspace(300, 500, 201)
    Xc = 1 / (1 + np.exp(-(T - 412.3) / 5))
    assert level_temperature(T, Xc, 0.5) == pytest.approx(412.3, abs=0.05)
    assert np.isnan(level_temperature(T, np.zeros_like(T), 0.5))
