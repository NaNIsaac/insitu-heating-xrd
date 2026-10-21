"""피크 검출 · 가우시안 피팅."""

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
    """x0 근처를 가우시안 + 선형 배경으로 피팅."""
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
    """NaN(갭) 이 섞인 프로파일에서 피크 인덱스 검출 → (smoothed, peak_idx).
    prominence=None 이면 배경 제거 잔차 표준편차의 3배. min_distance 는 x 단위."""
    good = np.isfinite(y)
    yy = np.where(good, y, np.nanmedian(y))
    if smooth and smooth >= 5:
        yy = savgol_filter(yy, smooth, 2)
    if prominence is None:
        prominence = 3 * np.std(yy - savgol_filter(yy, 51, 2))
    dist = max(1, round(min_distance / (x[1] - x[0])))
    pk, _ = find_peaks(yy, prominence=prominence, distance=dist)
    return yy, pk[good[pk]]  # 갭 위의 가짜 피크 제거
