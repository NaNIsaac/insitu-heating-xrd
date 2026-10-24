"""Peak tracking over time: per-peak intensity vs time/temperature and when each peak appears or
vanishes (for estimating phase transitions).

1) Detect peaks in every frame with find_peaks (prominence relative to the noise).
2) Link peaks of neighboring frames into one track when their positions differ by less than tol
   (follows small shifts from thermal expansion).
3) Keep only tracks seen long enough, and compute in every frame the area of the +-hw window
   around that position (straight-line background removed).
4) Transition: the normalized area (maximum = 1) crossing 0.5 upward is an appearance, downward a
   disappearance. Brief noisy crossings are filtered with the median of min_frames on each side,
   and no decision spans a gap in the images.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d, median_filter
from scipy.optimize import isotonic_regression
from scipy.signal import find_peaks
from scipy.stats import linregress, theilslopes

from .heatmap import beam_ok
from .style import TRACK_COLORS  # noqa: F401  (also used as tk.TRACK_COLORS)


@dataclass(frozen=True, slots=True)
class TrackOptions:
    snr: float = 6.0  # prominence >= snr x noise
    rel: float = 0.03  # prominence >= rel x profile range
    tol: float = 0.12  # largest position difference between neighboring frames still treated as the same peak [deg]
    max_skip: int = 4  # the track survives this many frames without a detection
    min_frames: int = 5  # minimum detected frames (or 3 % of all)
    max_tracks: int = 6
    manual: tuple[float, ...] = ()  # 2θ positions given directly (fixed window)
    # transition level: the normalized area (maximum = 1) crossing this value up or down
    level: float = 0.5


@dataclass(slots=True)
class Track:
    name: str
    pos: np.ndarray  # peak position per frame [deg] (interpolated/extended where not detected)
    hw: float  # integration window half-width [deg]
    detected: np.ndarray  # bool, detected per frame
    area: np.ndarray = field(default_factory=lambda: np.empty(0))
    events: list[tuple[str, int]] = field(default_factory=list)  # ("appears"|"vanishes", frame)
    manual: bool = False
    level: float = 0.5  # transition level (normalized area)
    # T50 as computed by compare (heating, X=0.5) when matched to an experiments.toml peak
    t50: float = np.nan

    @property
    def center(self) -> float:
        return (
            float(np.median(self.pos[self.detected]))
            if self.detected.any()
            else float(np.median(self.pos))
        )


def _filled(prof: np.ndarray, x: np.ndarray) -> np.ndarray | None:
    ok = np.isfinite(prof)
    return np.interp(x, x[ok], prof[ok]) if ok.sum() > 10 else None


def _detect(x: np.ndarray, Z: np.ndarray, valid: np.ndarray, opt: TrackOptions):
    """List of (position, half-width) per frame."""
    dx = x[1] - x[0]
    out = []
    for prof, ok_frame in zip(Z, valid):
        p = _filled(prof, x) if ok_frame else None
        if p is None:
            out.append([])
            continue
        ok = np.isfinite(prof)
        noise = 1.4826 * np.median(np.abs(np.diff(prof[ok]))) / np.sqrt(2)
        ps = gaussian_filter1d(p, 1.0)
        rng = np.percentile(ps, 99) - np.percentile(ps, 1)
        idx, pr = find_peaks(ps, prominence=max(opt.snr * noise, opt.rel * rng, 1e-12), width=3)
        out.append([(x[i], w * dx / 2) for i, w in zip(idx, pr["widths"])])
    return out


def _link(dets, opt: TrackOptions, n: int) -> list[dict]:
    tracks: list[dict] = []
    for k, peaks in enumerate(dets):
        used = set()
        for xp, hw in sorted(peaks, key=lambda p: p[0]):
            best = None
            for j, tr in enumerate(tracks):
                if j in used or k - tr["last"] > opt.max_skip + 1:
                    continue
                d = abs(tr["pos"][-1] - xp)
                if d <= opt.tol and (best is None or d < best[0]):
                    best = (d, j)
            if best is None:
                tracks.append({"frames": [k], "pos": [xp], "hw": [hw], "last": k})
                used.add(len(tracks) - 1)
            else:
                tr = tracks[best[1]]
                tr["frames"].append(k)
                tr["pos"].append(xp)
                tr["hw"].append(hw)
                tr["last"] = k
                used.add(best[1])
    need = max(opt.min_frames, int(0.03 * n))
    return [t for t in tracks if len(t["frames"]) >= need]


def window_area(x: np.ndarray, prof: np.ndarray, c: float, hw: float) -> float:
    """Area of the c +- hw window, with the line joining the window ends subtracted as background."""
    m = (x >= c - hw) & (x <= c + hw)
    if m.sum() < 5:
        return np.nan
    xs, ys = x[m], prof[m]
    ok = np.isfinite(ys)
    if ok.sum() < 5:
        return np.nan
    xs, ys = xs[ok], ys[ok]
    k = max(2, len(xs) // 10)
    x0, y0 = xs[:k].mean(), ys[:k].mean()
    x1, y1 = xs[-k:].mean(), ys[-k:].mean()
    base = y0 + (y1 - y0) * (xs - x0) / ((x1 - x0) or 1)
    return float(np.trapezoid(ys - base, xs))


def track(
    x: np.ndarray, Z: np.ndarray, breaks: np.ndarray, opt: TrackOptions = TrackOptions()
) -> list[Track]:
    """breaks: bool per frame, True = an image gap or folder boundary precedes that frame.
    The area is computed only inside blocks (stretches between gaps) where the peak was detected at least once."""
    n = len(Z)
    block = np.cumsum(breaks)
    valid = beam_ok(Z, block)
    raw = _link(_detect(x, Z, valid, opt), opt, n)
    # longest tracks first so duplicates at the same position are dropped
    raw.sort(key=lambda t: -len(t["frames"]))
    kept: list[dict] = []
    for t in raw:  # drop if a track close in position and overlapping in time is already kept
        c, a, b = np.median(t["pos"]), t["frames"][0], t["frames"][-1]
        if all(
            abs(c - np.median(k["pos"])) > 2 * opt.tol or b < k["frames"][0] or a > k["frames"][-1]
            for k in kept
        ):
            kept.append(t)
    kept = kept[: opt.max_tracks]

    out: list[Track] = []
    for t in kept:
        fr = np.array(t["frames"])
        # outside the detections, hold the first/last position
        pos = np.interp(np.arange(n), fr, t["pos"])
        hw = float(np.clip(2.0 * np.median(t["hw"]), 0.1, 0.35))
        det = np.zeros(n, bool)
        det[fr] = True
        out.append(Track("", pos, hw, det))
    for c in opt.manual:
        out.append(Track("", np.full(n, float(c)), 0.2, np.zeros(n, bool), manual=True))

    out.sort(key=lambda tr: tr.center)
    for i, tr in enumerate(out):
        tr.name = f"P{i + 1}"
        blocks = set(block) if tr.manual else set(block[tr.detected])
        tr.area = np.array(
            [
                window_area(x, Z[k], tr.pos[k], tr.hw)
                if valid[k] and block[k] in blocks
                else np.nan
                for k in range(n)
            ]
        )
        tr.events = _crossings(tr.area, block, opt)
        tr.level = opt.level
    return out


def norm_area(area: np.ndarray) -> np.ndarray:
    """Area normalized to a maximum of 1 (the same scale as the graphs)."""
    mx = np.nanmax(np.abs(area)) if np.isfinite(area).any() else np.nan
    return area / mx if mx else area


def crossings(v: np.ndarray, level: float, m: int, full: bool = True):
    """Points j (v[j-1] to v[j]) where v crosses level, with direction 'appears' (up) or 'vanishes' (down).
    Brief noisy crossings are ignored: of the crossings of the 3-point median, only those where the
    median of the m values before is below the level and of the m values after is above it count
    (reversed for downward). full=True requires m values on both sides."""
    above = median_filter(v, size=3, mode="nearest") >= level
    for j in np.flatnonzero(above[1:] != above[:-1]) + 1:
        if full and (j < m or j + m > len(v)):
            continue
        pre, post = np.median(v[max(0, j - m) : j]), np.median(v[j : j + m])
        if above[j] and pre < level <= post:
            yield "appears", int(j)
        elif not above[j] and post < level <= pre:
            yield "vanishes", int(j)


def level_temperature(T: np.ndarray, X: np.ndarray, level: float, m: int = 5) -> float:
    """Temperature (T50) at which X first crosses level upward; brief spikes ignored, linear interpolation between the frames around it."""
    ok = np.isfinite(T) & np.isfinite(X)
    T, X = T[ok], X[ok]
    if len(X) < 2 * m:
        return np.nan
    j = next((j for kind, j in crossings(X, level, m, full=False) if kind == "appears"), None)
    if j is None:
        return np.nan
    x0, x1 = X[j - 1], X[j]
    w = (level - x0) / (x1 - x0) if x1 != x0 else 0.5
    return float(T[j - 1] + np.clip(w, 0, 1) * (T[j] - T[j - 1]))


def _first_cross(T: np.ndarray, X: np.ndarray, level: float) -> float:
    """T where a monotonically increasing X first crosses level (linear interpolation)."""
    j = np.flatnonzero(X >= level)
    if not len(j) or j[0] == 0:
        return np.nan
    j = j[0]
    return (
        float(np.interp(level, [X[j - 1], X[j]], [T[j - 1], T[j]]))
        if X[j] > X[j - 1]
        else float(T[j])
    )


def transition_point(
    T: np.ndarray,
    A: np.ndarray,
    level: float = 0.5,
    K: float = 3.0,
    n_end: int = 10,
    iters: int = 8,
) -> dict:
    """Transition point of a step-shaped signal A(T). Only a window set by the data is used, so it is
    insensitive to outliers outside, drift and the choice of range.

    1) Normalize roughly with the medians of the first/last n_end points, then isotonic
       (monotone) regression gives a cumulative curve. Its 0.5 point is the center c and the
       distance between the 0.25 and 0.75 points is the width w.
    2) Window [c - K*w, c + K*w]. A Theil-Sen line on each outer tail [c +- 1.5w ... c +- K*w]
       gives the before/after background (local normalization).
    3) A line regression of the normalized values in the core |T - c| <= w; its crossing of level
       is T50 (error = standard error of the x intercept).
    4) Repeat 2-3 with the new c, w until converged.
    The window scales with the transition width (K=3 spans about 0.1 %-99.9 % of a sigmoid), so
    no human-chosen range is involved.
    Returns T50, err, width (dT 25-75), a0/a1 (background before/after at T50), window, ok."""
    out = dict(
        T50=np.nan,
        err=np.nan,
        width=np.nan,
        a0=np.nan,
        a1=np.nan,
        window=(np.nan, np.nan),
        n_core=0,
        ok=False,
    )
    m = np.isfinite(T) & np.isfinite(A)
    if m.sum() < 4 * n_end:
        return out
    o = np.argsort(T[m], kind="stable")
    T, A = T[m][o], A[m][o]
    a0, a1 = np.median(A[:n_end]), np.median(A[-n_end:])
    if a1 == a0:
        return out
    Xi = isotonic_regression((A - a0) / (a1 - a0)).x
    c = _first_cross(T, Xi, level)
    w = _first_cross(T, Xi, 0.75) - _first_cross(T, Xi, 0.25)
    if not (np.isfinite(c) and np.isfinite(w)):
        return out
    dT = float(np.median(np.diff(T))) or 1e-6
    for _ in range(iters):
        w = max(w, 3 * dT)  # keep at least a few frames in the core
        pre = (T >= c - K * w) & (T <= c - 1.5 * w)
        post = (T >= c + 1.5 * w) & (T <= c + K * w)
        core = np.abs(T - c) <= w
        if pre.sum() < 3 or post.sum() < 3 or core.sum() < 3:
            return out
        sp, ip = theilslopes(A[pre], T[pre])[:2]
        sq, iq = theilslopes(A[post], T[post])[:2]
        Xl = (A - (ip + sp * T)) / ((iq + sq * T) - (ip + sp * T))
        lr = linregress(T[core], Xl[core])
        if not lr.slope > 0:
            return out
        c_new = (level - lr.intercept) / lr.slope
        win = (T >= c - K * w) & (T <= c + K * w)
        Xw = isotonic_regression(Xl[win]).x
        t25, t75 = _first_cross(T[win], Xw, 0.25), _first_cross(T[win], Xw, 0.75)
        w_new = t75 - t25
        tc = T[core]
        s = np.sqrt(np.sum((Xl[core] - lr.intercept - lr.slope * tc) ** 2) / max(len(tc) - 2, 1))
        err = (
            s
            / lr.slope
            * np.sqrt(1 / len(tc) + (c_new - tc.mean()) ** 2 / np.sum((tc - tc.mean()) ** 2))
        )
        out = dict(
            T50=float(c_new),
            err=float(err),
            width=float(w_new) if np.isfinite(w_new) else float(w),
            t25=float(t25),
            t75=float(t75),
            a0=float(ip + sp * c_new),
            a1=float(iq + sq * c_new),
            pre=(sp, ip),
            post=(sq, iq),
            window=(float(c - K * w), float(c + K * w)),
            n_core=int(core.sum()),
            ok=True,
        )
        done = abs(c_new - c) < 0.05 and (not np.isfinite(w_new) or abs(w_new - w) < 0.05)
        c, w = c_new, (w_new if np.isfinite(w_new) else w)
        if done:
            break
    return out


def normalize_step(T: np.ndarray, A: np.ndarray, tp: dict) -> np.ndarray:
    """Normalize with the before/after background lines of transition_point: X = (A - pre(T)) / (post(T) - pre(T)).
    Outside the window the background is held at its value at the window edge (the lines are not extrapolated far)."""
    Tc = np.clip(T, *tp["window"])
    (sp, ip), (sq, iq) = tp["pre"], tp["post"]
    lo, hi = ip + sp * Tc, iq + sq * Tc
    with np.errstate(all="ignore"):
        return (A - lo) / (hi - lo)


def _crossings(area: np.ndarray, block: np.ndarray, opt: TrackOptions) -> list[tuple[str, int]]:
    """Frames where the normalized area crosses opt.level upward (appears) or downward (vanishes).
    Judged only inside blocks (between image gaps); repeated crossings in the same direction keep the first."""
    an = norm_area(area)
    events: list[tuple[str, int]] = []
    for b in np.unique(block):
        idx = np.flatnonzero((block == b) & np.isfinite(an))
        if len(idx) < 2 * opt.min_frames:
            continue
        for kind, j in crossings(an[idx], opt.level, opt.min_frames):
            if not events or events[-1][0] != kind:
                events.append((kind, int(idx[j])))
    return events


def label(tr: Track, temp: np.ndarray) -> str:
    """Legend text such as 'P1 18.40° ↑374 °C' (temperature where the normalized area rose above ↑ or fell below ↓ the level), or '(no crossing)' if it never crossed."""
    s = f"{tr.name} {tr.center:.2f}°"
    if tr.manual:
        s += " (manual)"
    if np.isfinite(tr.t50):
        s += f"  T50 {tr.t50:.0f} °C"
    elif tr.events:
        s += "  " + "  ".join(
            f"{'↑' if ev == 'appears' else '↓'}{temp[k]:.0f} °C" for ev, k in tr.events
        )
    else:
        s += "  (no crossing)"
    return s


def to_frames(tracks: list[Track], frames: pd.DataFrame) -> pd.DataFrame:
    """Table of peak position and area per frame."""
    df = frames[["file", "time_kst", "time_rel"]].copy()
    for c in ("temp_profile", "temp_pv", "segment"):
        if c in frames:
            df[c] = frames[c].to_numpy()
    for tr in tracks:
        df[f"{tr.name}_pos"] = tr.pos
        df[f"{tr.name}_area"] = tr.area
        df[f"{tr.name}_detected"] = tr.detected
    return df


def events_table(tracks: list[Track], frames: pd.DataFrame, temp: np.ndarray) -> pd.DataFrame:
    rows = []
    for tr in tracks:
        if not tr.events:
            rows.append(
                {
                    "peak": tr.name,
                    "2theta": round(tr.center, 3),
                    "event": f"no crossing of {tr.level:g}",
                    "time_kst": "",
                    "T_C": np.nan,
                }
            )
        for ev, k in tr.events:
            rows.append(
                {
                    "peak": tr.name,
                    "2theta": round(float(tr.pos[k]), 3),
                    "event": ev,
                    "time_kst": frames["time_kst"].iloc[k][11:19],
                    "time_rel": round(float(frames["time_rel"].iloc[k]), 2),
                    "T_C": temp[k],
                }
            )
    return pd.DataFrame(rows)
