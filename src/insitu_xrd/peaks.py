"""Peak detection and Gaussian fitting."""

from typing import NamedTuple

import numpy as np
from scipy.optimize import curve_fit
from scipy.signal import find_peaks, savgol_filter

from .geometry import FArray


class PeakFit(NamedTuple):
    center: float
    fwhm: float
    center_err: float


NAN_FIT = PeakFit(np.nan, np.nan, np.nan)


def _gauss_lin(t, amp, mu, sig, b0, b1):
    return amp * np.exp(-0.5 * ((t - mu) / sig) ** 2) + b0 + b1 * t


def fit_peak(x: FArray, y: FArray, x0: float, half_window: float = 0.4) -> PeakFit:
    """Fit a Gaussian on a linear background within half_window of x0."""
    m = (np.abs(x - x0) <= half_window) & np.isfinite(y)
    if m.sum() < 6:
        return NAN_FIT
    xs, ys = x[m], y[m]
    p0 = [ys.max() - ys.min(), x0, half_window / 4, ys.min(), 0.0]
    try:
        (amp, mu, sig, *_), cov = curve_fit(_gauss_lin, xs, ys, p0=p0, maxfev=5000)
    except RuntimeError, ValueError:
        return NAN_FIT
    if amp <= 0 or not xs.min() < mu < xs.max():
        return NAN_FIT
    return PeakFit(mu, 2.3548 * abs(sig), np.sqrt(cov[1, 1]))


def detect_peaks(
    x: FArray,
    y: FArray,
    prominence: float | None = None,
    min_distance: float = 0.15,
    smooth: int = 7,
) -> tuple[FArray, np.ndarray]:
    """Peak indices of a profile that may contain NaN gaps: (smoothed, peak_idx).
    prominence=None means 3 times the std of the residual after removing the background.
    min_distance is in x units."""
    good = np.isfinite(y)
    yy = np.where(good, y, np.nanmedian(y))
    if smooth and smooth >= 5:
        yy = savgol_filter(yy, smooth, 2)
    if prominence is None:
        prominence = 3 * np.std(yy - savgol_filter(yy, 51, 2))
    dist = max(1, round(min_distance / (x[1] - x[0])))
    pk, _ = find_peaks(yy, prominence=prominence, distance=dist)
    return yy, pk[good[pk]]  # drop false peaks on filled gaps
