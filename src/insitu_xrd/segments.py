"""온도 프로파일 → 승온 / 유지 / 하온 구간 자동 분할."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd
from scipy.ndimage import median_filter, uniform_filter1d

from .temperature import PV, SV, TempLog, fmt_kst

Kind = Literal["heat", "hold", "cool", "gap"]
_CODE = {1: "heat", 0: "hold", -1: "cool", 2: "gap"}
EDGE_HOLD = 120.0  # 로그 처음/끝 값을 바깥으로 유지하는 최대 시간 [s]


@dataclass(frozen=True, slots=True)
class SegmentOptions:
    hold_rate: float = 1.5  # |dT/dt| 이 이보다 작으면 유지 구간 [°C/min]
    min_duration: float = 90.0  # 이보다 짧은 구간은 이웃에 흡수 [s]
    window: float = 60.0  # 기울기 계산 창 [s]
    dt: float = 1.0  # 재샘플 간격 [s]
    context: float = 3 * 3600.0  # 앞뒤로 이만큼 로그를 더 보고 판단 (구간이 잘려 보이지 않게) [s]
    fill_gaps: bool = True  # 로그 끊긴 구간을 이웃 램프 외삽으로 추정


@dataclass(frozen=True, slots=True)
class Segment:
    kind: Kind
    t0: float  # Unix [s]
    t1: float
    T0: float  # 시작/끝 온도 [°C]
    T1: float
    rate: float  # 평균 승온 속도 [°C/min]
    estimated: bool = False  # 로그 끊긴 구간을 외삽한 값이 섞여 있음

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
        """그림 라벨용 줄들: 이름 / 온도 / 길이 / 시각 (/ 추정 표시)."""
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
    """[code, start, stop) 런 목록."""
    edges = np.flatnonzero(np.diff(code)) + 1
    starts = np.r_[0, edges]
    stops = np.r_[edges, len(code)]
    return [[int(code[a]), int(a), int(b)] for a, b in zip(starts, stops)]


def _merge_short(runs: list[list[int]], min_len: int) -> list[list[int]]:
    """짧은 런을 가장 짧은 것부터 더 긴 이웃에 흡수 (gap 은 건드리지 않음)."""

    def neighbors(k: int) -> list[int]:
        return [j for j in (k - 1, k + 1) if 0 <= j < len(runs) and runs[j][0] != 2]

    while True:
        # 양옆이 gap 뿐인 짧은 런은 흡수할 곳이 없으므로 그대로 둠
        cand = [
            k for k, r in enumerate(runs) if r[0] != 2 and r[2] - r[1] < min_len and neighbors(k)
        ]
        if not cand:
            return runs
        k = min(cand, key=lambda k: runs[k][2] - runs[k][1])
        j = max(neighbors(k), key=lambda j: runs[j][2] - runs[j][1])
        runs[k][0] = runs[j][0]
        merged: list[list[int]] = []
        for r in runs:  # 같은 코드끼리 합치기
            if merged and merged[-1][0] == r[0]:
                merged[-1][2] = r[2]
            else:
                merged.append(r)
        runs = merged


def _refine(t: np.ndarray, T: np.ndarray, a: int, b: int, c: int, w: int) -> int:
    """경계 b 를 ±w 범위에서 양쪽 직선 피팅 잔차가 최소인 지점으로 이동."""
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
    t: np.ndarray  # 1 s 격자, Unix [s]
    T: np.ndarray  # 판단에 쓴 온도 (SV 우선, 끊긴 곳은 추정값)
    pv: np.ndarray  # 측정 PV (끊긴 곳 NaN)
    sv: np.ndarray
    estimated: np.ndarray  # bool, T 가 추정값인 지점
    segments: list[Segment]

    def at(self, times: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """시각 → (온도, 추정 여부)."""
        i = np.clip(np.searchsorted(self.t, times), 0, len(self.t) - 1)
        return self.T[i], self.estimated[i]


def _line(t: np.ndarray, T: np.ndarray) -> float:
    """기울기 [°C/s]."""
    ok = np.isfinite(T)
    return float(np.polyfit(t[ok], T[ok], 1)[0]) if ok.sum() >= 2 else 0.0


def _fill_gaps(
    t: np.ndarray, T: np.ndarray, hold: float, span: int
) -> tuple[np.ndarray, np.ndarray]:
    """끊긴 구간: 뒤(또는 앞)가 램프면 그 직선을 연장하되 반대편 온도에서 멈춤, 아니면 선형 보간.
    로그 처음/끝 바깥(한쪽만 있음)은 어디서 멈출지 모르므로 외삽하지 않고 가장 가까운 값 유지."""
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
            # 로그 처음/끝 바깥: 가장 가까운 값을 EDGE_HOLD 초까지만 유지, 그 너머는 모름(NaN)
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
    """[t_start, t_end] (Unix) 구간의 온도를 승온/유지/하온 구간으로 분할.
    설정값(SV)이 있으면 SV 기준 (계단·램프가 깨끗함), 없으면 PV.
    앞뒤 opt.context 만큼 로그를 더 보고 판단한 뒤 잘라냄."""
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

    # 남은 결측은 기울기 계산용으로만 메우고 gap 처리
    Tf = np.interp(grid, grid[valid], T[valid])
    Ts = median_filter(Tf, size=max(3, w // 2) | 1, mode="nearest")
    rate = uniform_filter1d(np.gradient(Ts, grid), w, mode="nearest") * 60.0

    code = np.where(np.abs(rate) < opt.hold_rate, 0, np.sign(rate)).astype(int)
    code[~valid] = 2
    runs = _merge_short(_rle(code), int(round(opt.min_duration / opt.dt)))

    # 경계 정밀화 (gap 경계는 그대로)
    for k in range(1, len(runs)):
        if 2 in (runs[k - 1][0], runs[k][0]):
            continue
        b = _refine(grid, Tf, runs[k - 1][1], runs[k][1], runs[k][2], w)
        runs[k - 1][2] = runs[k][1] = b

    # [t_start, t_end] 로 자르기. 잘려서 아주 짧게 남은 조각은 이웃에 붙임
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
        j = min(b, len(grid) - 1)  # 전체 구간 기준 속도
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
    """구간 목록 → 표 (t_ref 기준 상대시간, scale=60 이면 분)."""
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
    """각 시각이 속한 구간 번호 (1부터, 없으면 0)."""
    out = np.zeros(len(times), int)
    for i, s in enumerate(segs):
        out[(times >= s.t0) & (times <= s.t1) & (out == 0)] = i + 1
    return out


def kinds_at(times: np.ndarray, segs: list[Segment]) -> np.ndarray:
    """각 시각의 구간 종류 ('heat' / 'hold' / 'cool' / 'gap', 구간 밖은 '')."""
    names = np.array([""] + [s.kind for s in segs], dtype=object)
    return names[assign(times, segs)]
