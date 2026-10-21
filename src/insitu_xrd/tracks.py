"""시간에 따른 피크 추적 → 피크별 세기 vs 시간/온도, 나타남·사라짐 시점 (상변이 추정용).

1) 프레임마다 find_peaks 로 피크 검출 (잡음 대비 prominence 기준)
2) 이웃 프레임의 피크를 위치가 tol 이내면 같은 피크로 이어 붙임 (열팽창으로 조금씩 움직여도 따라감)
3) 충분히 오래 보인 트랙만 남기고, 모든 프레임에서 그 위치 ±hw 창의 면적(직선 배경 제거)을 계산
4) 전이 시점: 정규화 면적(최댓값=1)이 0.5 를 위로 넘으면 나타남, 아래로 내려가면 사라짐
   (잡음으로 잠깐 넘나드는 건 앞뒤 min_frames 중앙값으로 걸러냄, 이미지 공백 경계를 넘어서는 판단 안 함)
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
from .style import TRACK_COLORS  # noqa: F401  (tk.TRACK_COLORS 로도 씀)


@dataclass(frozen=True, slots=True)
class TrackOptions:
    snr: float = 6.0  # prominence ≥ snr × 잡음
    rel: float = 0.03  # prominence ≥ rel × 프로파일 범위
    tol: float = 0.12  # 이웃 프레임 간 같은 피크로 볼 위치 차 [deg]
    max_skip: int = 4  # 이만큼 프레임 동안 안 보여도 같은 트랙 유지
    min_frames: int = 5  # 최소 검출 프레임 수 (또는 전체의 3 %)
    max_tracks: int = 6
    manual: tuple[float, ...] = ()  # 직접 지정한 2θ 위치 (고정 창)
    level: float = 0.5  # 전이 기준: 정규화 면적(최댓값=1)이 이 값을 넘는/내려가는 시점


@dataclass(slots=True)
class Track:
    name: str
    pos: np.ndarray  # 프레임별 피크 위치 [deg] (검출 없는 프레임은 보간/연장)
    hw: float  # 적분 창 반폭 [deg]
    detected: np.ndarray  # bool, 프레임별 검출 여부
    area: np.ndarray = field(default_factory=lambda: np.empty(0))
    events: list[tuple[str, int]] = field(default_factory=list)  # ("appears"|"vanishes", 프레임)
    manual: bool = False
    level: float = 0.5  # 전이 기준 (정규화 면적)
    t50: float = np.nan  # experiments.toml 피크와 맞으면 compare 방식 T50 (승온, X=0.5)

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
    """프레임별 (위치, 반폭) 목록."""
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
    """c ± hw 창의 면적, 창 양끝을 잇는 직선을 배경으로 뺌."""
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
    """breaks: bool (프레임), True = 그 프레임 앞에 이미지 공백/폴더 경계가 있음.
    면적은 그 피크가 한 번이라도 검출된 블록(공백 사이 구간) 안에서만 계산."""
    n = len(Z)
    block = np.cumsum(breaks)
    valid = beam_ok(Z, block)
    raw = _link(_detect(x, Z, valid, opt), opt, n)
    # 변화가 큰(=정보가 많은) 트랙 우선, 같은 위치 중복 제거
    raw.sort(key=lambda t: -len(t["frames"]))
    kept: list[dict] = []
    for t in raw:  # 위치가 가깝고 시간도 겹치는 트랙이 이미 있으면 버림
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
        pos = np.interp(np.arange(n), fr, t["pos"])  # 검출 밖은 처음/끝 위치로 연장
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
    """최댓값 = 1 로 정규화한 면적 (그래프와 같은 기준)."""
    mx = np.nanmax(np.abs(area)) if np.isfinite(area).any() else np.nan
    return area / mx if mx else area


def crossings(v: np.ndarray, level: float, m: int, full: bool = True):
    """v 가 level 을 넘나드는 지점 j (v[j-1] → v[j]) 와 방향 'appears'(위로) / 'vanishes'(아래로).
    잡음으로 잠깐 넘나드는 건 무시: 3점 중앙값으로 넘나든 곳 중 앞 m 개 중앙값은 기준 아래,
    뒤 m 개 중앙값은 기준 위 (아래로는 반대). full=True 면 앞뒤 m 개가 다 있어야 함."""
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
    """X 가 level 을 처음 위로 넘는 온도 (T50). 잠깐 튀는 건 무시, 앞뒤 프레임 사이 선형보간."""
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
    """단조 증가 X 가 level 을 처음 넘는 T (선형 보간)."""
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
    """계단 모양 신호 A(T) 의 전이점 — 데이터가 정하는 창만 써서 바깥 아웃라이어·드리프트·범위 선택에 둔감.

    1) 처음/끝 n_end 중앙값으로 대략 정규화 → isotonic(단조) 회귀 = 누적 곡선.
       그 0.5 지점 = 중심 c, 0.25–0.75 지점 사이 = 폭 w
    2) 창 [c − K·w, c + K·w]. 바깥 꼬리 [c ± 1.5w … c ± K·w] 에 Theil–Sen 직선 → 전·후 배경 (국소 정규화)
    3) 코어 |T − c| ≤ w 의 정규화값을 직선 회귀 → level 교차 = T50 (오차 = x절편 표준오차)
    4) 새 c, w 로 2–3 반복 (수렴까지)
    창 크기는 전이 폭에 비례 (K=3 이면 시그모이드의 ~0.1 %–99.9 %) → 사람이 정한 범위가 아님.
    반환: T50, err, width(ΔT₂₅₋₇₅), a0/a1 (T50 에서의 전·후 배경값), window, ok."""
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
        w = max(w, 3 * dT)  # 코어에 최소 몇 프레임은 들어오게
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
    """transition_point 의 전·후 배경 직선으로 정규화: X = (A − 전(T)) / (후(T) − 전(T)).
    창 밖에서는 배경을 창 끝 값으로 고정 (직선을 멀리 외삽하지 않음)."""
    Tc = np.clip(T, *tp["window"])
    (sp, ip), (sq, iq) = tp["pre"], tp["post"]
    lo, hi = ip + sp * Tc, iq + sq * Tc
    with np.errstate(all="ignore"):
        return (A - lo) / (hi - lo)


def _crossings(area: np.ndarray, block: np.ndarray, opt: TrackOptions) -> list[tuple[str, int]]:
    """정규화 면적이 opt.level 을 위로 넘는 프레임 = 나타남, 아래로 내려가는 프레임 = 사라짐.
    블록(이미지 공백 사이) 안에서만 판단, 같은 방향이 연달아 나오면 첫 번째만."""
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
    """범례용: 'P1 18.40° ↑374 °C' (정규화 면적이 level 을 넘은 ↑ / 내려간 ↓ 온도),
    한 번도 안 넘나들면 '(no crossing)'."""
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
    """프레임별 피크 위치·면적 표."""
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
