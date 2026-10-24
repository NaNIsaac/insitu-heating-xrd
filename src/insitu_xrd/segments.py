"""Automatic split of a temperature profile into heat / hold / cool segments."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd
from scipy.ndimage import median_filter, uniform_filter1d

from .temperature import PV, SV, TempLog, fmt_kst

Kind = Literal["heat", "hold", "cool", "gap"]
_CODE = {1: "heat", 0: "hold", -1: "cool", 2: "gap"}
EDGE_HOLD = 120.0  # longest time the first/last log value is held outward [s]


@dataclass(frozen=True, slots=True)
class SegmentOptions:
    hold_rate: float = 1.5  # |dT/dt| below this is a hold segment [°C/min]
    min_duration: float = 90.0  # shorter segments are absorbed by a neighbor [s]
    window: float = 60.0  # slope window [s]
    dt: float = 1.0  # resampling step [s]
    context: float = 3 * 3600.0  # extra log read on both sides so segments are not cut off [s]
    fill_gaps: bool = True  # estimate log gaps by extrapolating the neighboring ramp


@dataclass(frozen=True, slots=True)
class Segment:
    kind: Kind
    t0: float  # Unix [s]
    t1: float
    T0: float  # start/end temperature [°C]
    T1: float
    rate: float  # mean heating rate [°C/min]
    estimated: bool = False  # includes values extrapolated over a log gap

    @property
    def duration(self) -> float:
        return self.t1 - self.t0

    @property
    def label(self) -> str:
        match self.kind:
            case "hold":
                return f"Hold {(self.T0 + self.T1) / 2:.0f} °C"
            case "heat" | "cool":
                return f"{self.kind.capitalize()} {abs(self.rate):.0f} °C/min"
            case _:
                return "no log"

    def describe(self) -> list[str]:
        """Label lines for figures: name, temperature, duration, time (and an estimate mark)."""
        lines = [self.label]
        if self.kind in ("heat", "cool"):
            lines.append(f"{self.T0:.0f} → {self.T1:.0f} °C")
        lines += [
            f"{self.duration / 60:.1f} min",
            f"{fmt_kst(self.t0)[11:16]}–{fmt_kst(self.t1)[11:16]}",
        ]
        if self.estimated:
            lines.append("(T partly est.)")
        return lines


def _rle(code: np.ndarray) -> list[list[int]]:
    """List of [code, start, stop) runs."""
    edges = np.flatnonzero(np.diff(code)) + 1
    starts = np.r_[0, edges]
    stops = np.r_[edges, len(code)]
    return [[int(code[a]), int(a), int(b)] for a, b in zip(starts, stops)]


def _merge_short(runs: list[list[int]], min_len: int) -> list[list[int]]:
    """Absorb short runs into the longer neighbor, shortest first; gaps are left alone."""

    def neighbors(k: int) -> list[int]:
        return [j for j in (k - 1, k + 1) if 0 <= j < len(runs) and runs[j][0] != 2]

    while True:
        # a short run with only gaps beside it has nothing to merge into, so it stays
        cand = [
            k for k, r in enumerate(runs) if r[0] != 2 and r[2] - r[1] < min_len and neighbors(k)
        ]
        if not cand:
            return runs
        k = min(cand, key=lambda k: runs[k][2] - runs[k][1])
        j = max(neighbors(k), key=lambda j: runs[j][2] - runs[j][1])
        runs[k][0] = runs[j][0]
        merged: list[list[int]] = []
        for r in runs:
            if merged and merged[-1][0] == r[0]:
                merged[-1][2] = r[2]
            else:
                merged.append(r)
        runs = merged


def _refine(t: np.ndarray, T: np.ndarray, a: int, b: int, c: int, w: int) -> int:
    """Move boundary b within ±w to where the two-sided straight-line fit residual is smallest."""
    lo, hi = max(a + 2, b - w), min(c - 2, b + w)
    if hi <= lo:
        return b
    best, arg = np.inf, b
    for s in range(lo, hi + 1):
        sse = 0.0
        for i0, i1 in ((max(a, s - 2 * w), s), (s, min(c, s + 2 * w))):
            x, y = t[i0:i1], T[i0:i1]
            if len(x) >= 2:
                sse += np.sum((y - np.polyval(np.polyfit(x, y, 1), x)) ** 2)
        if sse < best:
            best, arg = sse, s
    return arg


@dataclass(slots=True)
class Detection:
    t: np.ndarray  # 1 s grid, Unix [s]
    T: np.ndarray  # temperature used for the decision (SV first; estimated over gaps)
    pv: np.ndarray  # measured PV (NaN over gaps)
    sv: np.ndarray
    estimated: np.ndarray  # bool, True where T is an estimate
    segments: list[Segment]

    def at(self, times: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Times to (temperature, whether estimated)."""
        i = np.clip(np.searchsorted(self.t, times), 0, len(self.t) - 1)
        return self.T[i], self.estimated[i]


def _line(t: np.ndarray, T: np.ndarray) -> float:
    """Slope [°C/s]."""
    ok = np.isfinite(T)
    return float(np.polyfit(t[ok], T[ok], 1)[0]) if ok.sum() >= 2 else 0.0


def _fill_gaps(
    t: np.ndarray, T: np.ndarray, hold: float, span: int
) -> tuple[np.ndarray, np.ndarray]:
    """Fill log gaps: if the side after (or before) is a ramp, extend its line but stop at the
    temperature of the opposite side; otherwise interpolate linearly.
    Outside the log start/end (only one side exists) the stopping point is unknown, so the
    nearest value is held instead of extrapolating."""
    T = T.copy()
    est = np.zeros(len(T), bool)
    for c, a, b in _rle(np.isfinite(T).astype(int)):
        if c or (a == 0 and b == len(T)):
            continue
        before = T[a - 1] if a > 0 else np.nan
        after = T[b] if b < len(T) else np.nan
        r_after = _line(t[b : b + span], T[b : b + span]) if b < len(T) else 0.0
        r_before = _line(t[max(0, a - span) : a], T[max(0, a - span) : a]) if a > 0 else 0.0
        tt = t[a:b]
        both = np.isfinite(before) and np.isfinite(after)
        if both and abs(r_after) * 60 >= hold:
            v = after + r_after * (tt - t[b])
            v = np.maximum(v, before) if r_after > 0 else np.minimum(v, before)
        elif both and abs(r_before) * 60 >= hold:
            v = before + r_before * (tt - t[a - 1])
            v = np.minimum(v, after) if r_before > 0 else np.maximum(v, after)
        elif both:
            v = np.interp(tt, [t[a - 1], t[b]], [before, after])
        else:
            # outside the log: hold the nearest value for EDGE_HOLD seconds, NaN beyond
            if np.isfinite(before):
                near = tt - t[a - 1] <= EDGE_HOLD
                v = np.where(near, before, np.nan)
            else:
                near = t[b] - tt <= EDGE_HOLD
                v = np.where(near, after, np.nan)
            T[a:b] = v
            est[a:b] = near
            continue
        T[a:b] = v
        est[a:b] = True
    return T, est


def detect(
    log: TempLog, t_start: float, t_end: float, opt: SegmentOptions = SegmentOptions()
) -> Detection:
    """Split the temperature over [t_start, t_end] (Unix) into heat / hold / cool segments.
    Uses the setpoint (SV) when available since its steps and ramps are clean, else PV.
    Reads opt.context of extra log on both sides for the decision, then clips."""
    lo = min(t_start, max(log.t[0], t_start - opt.context))
    hi = max(t_end, min(log.t[-1], t_end + opt.context))
    grid = np.arange(lo, hi + opt.dt, opt.dt)
    sv, pv = log.at(grid, SV), log.at(grid, PV)
    T = np.where(np.isfinite(sv), sv, pv)
    est = np.zeros(len(grid), bool)
    if not np.isfinite(T).any() or len(grid) < 3:
        return Detection(grid, T, pv, sv, est, [])
    w = max(3, int(round(opt.window / opt.dt)))
    if opt.fill_gaps:
        T, est = _fill_gaps(grid, T, opt.hold_rate, 5 * w)
    valid = np.isfinite(T)

    # remaining gaps are filled only for the slope; they are still marked as gap
    Tf = np.interp(grid, grid[valid], T[valid])
    Ts = median_filter(Tf, size=max(3, w // 2) | 1, mode="nearest")
    rate = uniform_filter1d(np.gradient(Ts, grid), w, mode="nearest") * 60.0

    code = np.where(np.abs(rate) < opt.hold_rate, 0, np.sign(rate)).astype(int)
    code[~valid] = 2
    runs = _merge_short(_rle(code), int(round(opt.min_duration / opt.dt)))

    # refine boundaries, except next to gaps
    for k in range(1, len(runs)):
        if 2 in (runs[k - 1][0], runs[k][0]):
            continue
        b = _refine(grid, Tf, runs[k - 1][1], runs[k][1], runs[k][2], w)
        runs[k - 1][2] = runs[k][1] = b

    # clip to [t_start, t_end]; a very short leftover sliver is attached to its neighbor
    i_lo, i_hi = np.searchsorted(grid, [t_start, t_end])
    i_hi = min(i_hi, len(grid) - 1)
    sliver = max(int(0.02 * (i_hi - i_lo)), 1)
    clipped = []
    for c, a, b in runs:
        a2, b2 = max(a, i_lo), min(b, i_hi + 1)
        if b2 <= a2:
            continue
        if clipped and b2 - a2 < sliver:
            clipped[-1][4] = b2
            continue
        clipped.append([c, a, b, a2, b2])
    if len(clipped) > 1 and clipped[0][4] - clipped[0][3] < sliver:
        clipped[1][3] = clipped[0][3]
        clipped.pop(0)

    segs = []
    for c, a, b, a2, b2 in clipped:
        j = min(b, len(grid) - 1)  # rate over the whole segment
        r = (Tf[j] - Tf[a]) / (grid[j] - grid[a]) * 60.0 if grid[j] > grid[a] and c != 2 else 0.0
        e = min(b2, len(grid) - 1)
        T0, T1 = (np.nan, np.nan) if c == 2 else (Tf[a2], Tf[e])
        segs.append(
            Segment(
                _CODE[c],
                float(grid[a2]),
                float(grid[e]),
                float(T0),
                float(T1),
                float(r),
                bool(est[a2 : e + 1].any()),
            )
        )
    return Detection(grid, T, pv, sv, est, segs)


def to_frame(segs: list[Segment], t_ref: float, scale: float) -> pd.DataFrame:
    """Segment list to a table (time relative to t_ref; scale=60 gives minutes)."""
    return pd.DataFrame(
        [
            {
                "segment": i + 1,
                "kind": s.kind,
                "label": s.label,
                "start_kst": fmt_kst(s.t0),
                "end_kst": fmt_kst(s.t1),
                "start_rel": (s.t0 - t_ref) / scale,
                "end_rel": (s.t1 - t_ref) / scale,
                "duration_min": s.duration / 60.0,
                "T_start": s.T0,
                "T_end": s.T1,
                "rate_C_per_min": s.rate,
                "estimated": s.estimated,
            }
            for i, s in enumerate(segs)
        ]
    )


def assign(times: np.ndarray, segs: list[Segment]) -> np.ndarray:
    """Segment number (from 1) containing each time, 0 if none."""
    out = np.zeros(len(times), int)
    for i, s in enumerate(segs):
        out[(times >= s.t0) & (times <= s.t1) & (out == 0)] = i + 1
    return out


def kinds_at(times: np.ndarray, segs: list[Segment]) -> np.ndarray:
    """Segment kind at each time ('heat' / 'hold' / 'cool' / 'gap', '' outside any segment)."""
    names = np.array([""] + [s.kind for s in segs], dtype=object)
    return names[assign(times, segs)]
