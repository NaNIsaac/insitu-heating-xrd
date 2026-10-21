"""결정화 변화 X(t) → Avrami(JMAK) 지수 n, 속도상수 k, (같은 시료의 등온 온도가 2개 이상이면) 활성화 에너지 Ea.

측정 = 승온 → 등온 유지 → 하온. 경고 · 판정 없이 전체 결과만.

설정: experiments.toml 의 [[iso]]  (측정 하나 = 비정질 시료 한 조각)
    [[iso]]
    sample = "x1"                 # 같은 sample 끼리 묶어 Arrhenius
    images = ['D:\\...\\x1_iso370']
    recipe = ['D:\\...\\recipe_....csv']   # (선택) 없으면 온도 없이 n, k 만
    T_iso  = 370                  # (선택) 이름 붙이기용 설정 온도. 실제 온도는 로그에서 계산
    start  = "10:05"              # (선택) 이 시각 사이 프레임만 (X 의 0 / 1 기준이 되는 처음·끝을 정함)
    end    = "11:30"
    t0     = "10:15:02"           # (선택) 변화 시작 시각을 직접 지정. 생략하면 JMAK 피팅으로 구함

    uv run main.py avrami                  ← [[iso]] 전부
    uv run main.py avrami --only x1        ← 이름 또는 sample 로 일부만
    uv run main.py avrami exp2             ← [[iso]] 대신 [[exp2]] (기존 승온 데이터로 시험)

분석
 1) X(t): compare 와 같은 방식 — 비정질로 나눈 R = I/I₀ − 1 의 피크 창(중심 ± fwhm_k·FWHM) 면적.
    X = 0 은 처음 n_norm 프레임, X = 1 은 마지막 n_norm 프레임
 2) t0 (t = 0): 설정의 t0 > 온도 로그의 등온 시작 (승온 끝나 설정 온도 도달) > JMAK 피팅.
    t0 가 정해지면 JMAK  X = 1 − exp(−(k·(t − t0 − τ))ⁿ) 의 τ = 그 뒤 잠복시간
 3) Avrami 플롯: ln[−ln(1−X)] = n·ln(t − t0) + n·ln k   (fit_range 안의 X) → n, k
 1') 하온 시작 전(등온 끝)에서 자름 → X = 1 은 등온 끝 상태
 4) 온도 T_iso: 등온 구간 PV 평균 (등온 구간을 못 찾으면 X 10–90 % 동안 PV 평균)
 5) 같은 sample 의 온도가 2개 이상: ln k vs 1/T → Ea (Avrami k · JMAK k · 1/t50 세 가지)
 + t10 · t50 · t90, 최대 결정화 속도 (dX/dt) 와 그 시각, 국소 Avrami 지수 n(X),
   등온 중 FWHM · 피크 위치 변화, 등온 끝 FWHM 의 겉보기 결정 크기 (Scherrer, 장비 폭 보정 없음)

결과 (--save): out/avrami/<설정이름>_avrami.png / .csv / _arrhenius.csv / _frames.csv
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.ndimage import median_filter
from scipy.optimize import curve_fit
from scipy.stats import linregress

from .compare import (
    DEFAULT_CONFIG,
    Experiment,
    Settings,
    _as_list,
    check_paths,
    entry_name,
    load_exp,
    load_settings,
    peak_areas,
    smooth,
)
from .config import load_toml, out_dir, project_path, saving
from .geometry import Geometry
from .heatmap import beam_ok
from .style import EXP_COLORS, ORIGIN_RC, origin_plotly, write_fit_html
from .temperature import fmt_kst, parse_time
from .tracks import level_temperature

KB_EV = 8.617333e-5  # 볼츠만 상수 [eV/K]
LINESTYLES = ["-", "--", ":", "-."]
MARKERS = ["o", "s", "^", "D", "v"]


# ═════════════════════════════ 설정 ═════════════════════════════
@dataclass(slots=True)
class IsoRun:
    exp: Experiment  # name · images · recipe · start/end
    sample: str
    t0: str | None = None  # 변화 시작 시각 직접 지정 (KST)


@dataclass(slots=True)
class IsoOptions:
    fit_range: tuple[float, float] = (0.15, 0.85)
    flat_min: float = 5.0  # 변화가 끝났는지 볼 창 [min] (그 뒤 X 가 다시 변하면 거기까지만)
    hold_tol: float = 3.0  # 최고 온도 − 이 값 [°C] 에 도달한 때 = 등온 시작


def load_iso(
    path: Path = DEFAULT_CONFIG, section: str = "iso"
) -> tuple[Settings, list[IsoRun], IsoOptions]:
    """설정 파일의 [peaks]·[options] + [[iso]] (section="exp" 면 [[exp]]) 목록."""
    st = load_settings(path, need_exps=False)
    raw = load_toml(path)
    o = raw.get("options", {})
    opt = IsoOptions(fit_range=tuple(float(v) for v in o.get("fit_range", (0.15, 0.85))))
    runs = []
    for i, e in enumerate(raw.get(section, []), start=1):
        sample = str(e.get("sample") or e.get("name") or f"run{i}")
        exp = Experiment(
            name=entry_name(e, i),
            images=[project_path(s) for s in _as_list(e["images"])],
            recipe=[project_path(s) for s in _as_list(e.get("recipe", []))],
            start=e.get("start"),
            end=e.get("end"),
            color=e.get("color"),
        )
        runs.append(IsoRun(exp, sample, e.get("t0")))
    if not runs:
        raise SystemExit(f"❌ {path}: [[{section}]] 항목이 없음 (README 의 '등온 실험' 참고)")
    return st, runs, opt


# ═════════════════════════════ 피팅 ═════════════════════════════
def _jmak(t, n, k, t0):
    return 1 - np.exp(-(np.clip(k * (t - t0), 0, None) ** n))


def jmak_fit(t: np.ndarray, X: np.ndarray, t0_fixed: float | None = None) -> dict:
    """X = 1 − exp(−(k·(t − t0))ⁿ). t0_fixed 가 없으면 t0 도 피팅 (여러 초기값 중 잔차 최소).
    t [min]. 반환 t0 는 같은 시간축의 값."""
    out = dict(n_jmak=np.nan, k_jmak=np.nan, t0_fit=np.nan)
    m = np.isfinite(t) & np.isfinite(X)
    t, X = t[m], X[m]
    t05, t50 = (level_temperature(t, X, v, m=3) for v in (0.05, 0.5))
    if len(t) < 8 or not np.isfinite(t50):
        return out
    t05 = t05 if np.isfinite(t05) else t[0]
    lo_t0 = t[0] - 2 * (t50 - t[0])  # t0 는 첫 프레임보다 조금 앞일 수도 있음
    best = None
    for n0 in (1.0, 2.0, 3.0, 4.0):
        for f in (0.0, 0.5, 1.0):
            t00 = t0_fixed if t0_fixed is not None else max(lo_t0 + 1e-6, t05 - f * (t50 - t05))
            if t50 - t00 <= 0:
                continue
            k0 = np.log(2) ** (1 / n0) / (t50 - t00)
            try:
                if t0_fixed is None:
                    p, _ = curve_fit(
                        _jmak,
                        t,
                        X,
                        p0=[n0, k0, t00],
                        maxfev=20000,
                        bounds=([0.3, k0 / 1e3, lo_t0], [30.0, k0 * 1e3, t50 - 1e-6]),
                    )
                else:
                    (pn, pk), _ = curve_fit(
                        lambda tt, n, k: _jmak(tt, n, k, t0_fixed),
                        t,
                        X,
                        p0=[n0, k0],
                        bounds=([0.3, k0 / 1e3], [30.0, k0 * 1e3]),
                        maxfev=20000,
                    )
                    p = (pn, pk, t0_fixed)
            except RuntimeError, ValueError:
                continue
            sse = float(np.sum((_jmak(t, *p) - X) ** 2))
            if best is None or sse < best[0]:
                best = (sse, p)
    if best is not None:
        out.update(n_jmak=float(best[1][0]), k_jmak=float(best[1][1]), t0_fit=float(best[1][2]))
    return out


def avrami_fit(tm: np.ndarray, X: np.ndarray, rng: tuple[float, float]) -> dict:
    """Avrami 플롯 직선 피팅: ln(−ln(1−X)) = n·ln t + n·ln k. tm = t − t0 [min].
    X 가 처음 rng[1] 을 넘기 전(주된 변화 구간)의 점만 — 변화 뒤 잡음으로 범위에 들어온 점은 뺌."""
    m = (tm > 0) & np.isfinite(X) & (X > rng[0]) & (X < rng[1])
    good = np.flatnonzero(np.isfinite(X))
    if len(good):
        above = np.flatnonzero(median_filter(X[good], size=3, mode="nearest") >= rng[1])
        if len(above):
            m &= tm <= tm[good[above[0]]]
    out = dict(n=np.nan, n_err=np.nan, k=np.nan, r2=np.nan, npts=int(m.sum()))
    if m.sum() < 5:
        return out
    lr = linregress(np.log(tm[m]), np.log(-np.log(1 - X[m])))
    n = float(lr.slope)
    out.update(
        n=n,
        n_err=float(lr.stderr),
        k=float(np.exp(lr.intercept / n)) if n > 0 else np.nan,
        r2=float(lr.rvalue**2),
        t_fit_end=float(tm[m].max()),
    )
    return out


# ═════════════════════════════ 측정 하나 ═════════════════════════════
@dataclass(slots=True)
class IsoResult:
    run: IsoRun
    T_iso: float = np.nan  # 등온 구간 평균 PV [°C]
    t0_src: str = ""  # t0 를 어디서: 지정 / 등온 시작 / JMAK
    t0: float = np.nan  # Unix (변화 시작)
    tm: np.ndarray = field(default_factory=lambda: np.empty(0))  # 프레임별 t − t0 [min]
    T: np.ndarray = field(default_factory=lambda: np.empty(0))
    files: np.ndarray = field(default_factory=lambda: np.empty(0))
    X: dict[str, np.ndarray] = field(default_factory=dict)
    width: dict[str, np.ndarray] = field(default_factory=dict)  # 피크 → 프레임별 FWHM [deg]
    center: dict[str, np.ndarray] = field(default_factory=dict)  # 피크 → 프레임별 중심 2θ [deg]
    fits: dict[str, dict] = field(default_factory=dict)  # 피크 → n, k, t50, ...
    fail: str = ""  # 분석 못 했을 때 이유

    @property
    def ok(self) -> bool:
        return bool(self.fits)


def hold_range(T: np.ndarray, tol: float = 3.0, ramp: float = 20.0) -> tuple[int, int] | None:
    """승온 → 등온 → 하온: 등온 구간 [처음, 마지막] 프레임 = PV 가 (등온 온도 − tol) 이상인 첫 · 마지막 프레임.
    등온 온도 = 상위 PV 의 중앙값. 처음부터 이미 그 온도 근처(승온 없음)면 None."""
    ok = np.isfinite(T)
    if ok.sum() < 5:
        return None
    Tv = T[ok]
    T_hold = float(np.median(Tv[Tv >= np.percentile(Tv, 80)]))
    if Tv[0] > T_hold - ramp:
        return None
    on = np.flatnonzero(ok & (T >= T_hold - tol))
    return int(on[0]), int(on[-1])


def _normalized(
    x, Z, ok, st: Settings, say=print, shapes: dict | None = None
) -> dict[str, np.ndarray]:
    """피크별 X: 처음 n_norm 정상 프레임 = 0, 마지막 n_norm 정상 프레임 = 1.
    shapes 에 dict 를 주면 피크별 (프레임별 중심, FWHM) 도 채움."""
    idx = np.flatnonzero(ok)
    base, final = idx[: st.n_norm], idx[-st.n_norm :]
    _, areas, sh = peak_areas(x, Z, base, final, st, say, widths=shapes is not None)
    if shapes is not None:
        shapes.update(sh)
    out = {}
    for pk, A in areas.items():
        a0, a1 = np.nanmedian(A[base]), np.nanmedian(A[final])
        X = (A - a0) / ((a1 - a0) or np.nan)
        X[~ok] = np.nan
        out[pk] = X
    return out


def _change_end(
    tt: np.ndarray, X: np.ndarray, opt: IsoOptions, drift: float = 0.1, flat: float = 0.1
) -> int | None:
    """주된 변화(X 가 0.5 를 넘는 상승)가 끝나 평평해진 뒤, 이후 데이터가 그 수준에서 drift 넘게 벗어나면
    (승온·냉각으로 세기·위치가 변하는 등) 평평해진 구간 끝 인덱스. 계속 평평하면(등온) None.
    평평함 = flat_min 분 창의 변화 속도가 최대 속도의 flat 배 미만."""
    good = np.flatnonzero(np.isfinite(X))
    if len(good) < 10:
        return None
    xs = median_filter(X[good], size=5, mode="nearest")
    ts = tt[good]
    up = np.flatnonzero(xs >= 0.5)
    w = max(3, int(round(opt.flat_min / np.median(np.diff(ts)))))  # flat_min 분의 프레임 수
    if not len(up) or len(xs) <= w + 1:
        return None
    rate = np.abs(xs[w:] - xs[:-w])  # 창마다 변화량
    for j in range(up[0], len(xs) - w):
        if rate[j] < flat * rate.max():  # 평평해짐
            level = np.median(xs[j : j + w + 1])
            if np.any(np.abs(xs[j + w + 1 :] - level) > drift):
                return int(good[j + w])
            return None
    return None


def _rate(tm: np.ndarray, X: np.ndarray) -> dict:
    """최대 결정화 속도 dX/dt [1/min] 와 그 시각 (5 프레임 중앙값으로 고른 X 의 기울기)."""
    m = np.isfinite(X) & np.isfinite(tm)
    if m.sum() < 7:
        return dict(rate_max=np.nan, t_rate_max=np.nan)
    xs = median_filter(X[m], size=5, mode="nearest")
    v = np.gradient(xs, tm[m])
    i = int(np.nanargmax(v))
    return dict(rate_max=float(v[i]), t_rate_max=float(tm[m][i]))


def local_n(
    tm: np.ndarray, X: np.ndarray, lo: float = 0.03, hi: float = 0.97
) -> tuple[np.ndarray, np.ndarray]:
    """국소 Avrami 지수 n(X) = d ln[−ln(1−X)] / d ln(t − t0)  (X 를 5 프레임 중앙값으로 고른 뒤).
    n 이 X 에 따라 바뀌면 핵생성 · 성장 방식이 도중에 바뀐 것."""
    m = np.isfinite(X) & (tm > 0)
    if m.sum() < 7:
        return np.empty(0), np.empty(0)
    xs = median_filter(X[m], size=5, mode="nearest")
    k = (xs > lo) & (xs < hi)
    if k.sum() < 4:
        return np.empty(0), np.empty(0)
    with np.errstate(all="ignore"):
        y = np.log(-np.log(1 - xs[k]))
        n = np.gradient(y, np.log(tm[m][k]))
    return xs[k], median_filter(n, size=5, mode="nearest")


def analyze_iso(
    run: IsoRun,
    x: np.ndarray,
    Z: np.ndarray,
    t: np.ndarray,
    T: np.ndarray,
    files: np.ndarray,
    st: Settings,
    opt: IsoOptions = IsoOptions(),
) -> IsoResult:
    """프레임 프로파일 Z (시간순, start/end 로 이미 자른 것) → 피크별 X(t), n, k."""
    if (
        np.isfinite(T).any() and not np.isfinite(T).all()
    ):  # 온도 로그 시간대 밖 프레임은 이 측정이 아님
        keep = np.isfinite(T)
        Z, t, T, files = Z[keep], t[keep], T[keep], files[keep]
    ok = beam_ok(Z)
    if ok.sum() < 4 * st.n_norm:
        return IsoResult(run, T=T, files=files, fail=f"프레임이 {ok.sum()} 개뿐")

    # 1) 등온 구간 끝(하온 시작)에서 자름 → X: 처음 n_norm 프레임 (승온 초기, 비정질) = 0,
    #    등온 끝 n_norm 프레임 = 1. 온도로 등온을 못 찾으면 변화가 끝난 뒤 X 가 다시 변하는 곳에서 자름
    hold = hold_range(T, opt.hold_tol)
    tt = (t - t[0]) / 60.0
    if run.exp.end:
        cut = None
    elif hold is not None:
        cut = hold[1]
    else:
        cut = _change_end(
            tt, _normalized(x, Z, ok, st, say=lambda *a: None)[next(iter(st.peaks))], opt
        )
    if cut is not None and cut < len(t) - st.n_norm:
        k = cut + 1
        Z, t, T, files, ok, tt = Z[:k], t[:k], T[:k], files[:k], ok[:k], tt[:k]
    res = IsoResult(run, T=T, files=files)
    sh: dict = {}
    res.X = _normalized(x, Z, ok, st, say=lambda *a: None, shapes=sh)
    for pk, (c, w) in sh.items():
        c[~ok], w[~ok] = np.nan, np.nan
        res.center[pk], res.width[pk] = c, w

    # 2) t0: 지정 > 등온 시작 (온도 로그) > 첫 피크 JMAK 피팅. 모든 피크가 같은 t0
    pk0 = next(iter(res.X))
    h = hold[0] if hold is not None else None
    if run.t0:
        t0, res.t0_src = parse_time(run.t0, fmt_kst(t[0])[:10]), "지정"
    elif h is not None:
        t0, res.t0_src = t[h], "등온 시작"
    else:
        j = jmak_fit(tt, res.X[pk0])
        if not np.isfinite(j["t0_fit"]):
            res.fail = f"{pk0}: X 가 0.5 를 넘지 않음 (변화 없음)"
            return res
        t0, res.t0_src = t[0] + j["t0_fit"] * 60, "JMAK"
    res.t0 = t0
    res.tm = tm = (t - t0) / 60.0
    fixed = res.t0_src != "JMAK"

    # 3) 피크별 Avrami 플롯 + JMAK (t0 가 정해졌으면 τ = 그 뒤 잠복시간)
    for pk, X in res.X.items():
        f = avrami_fit(tm, X, opt.fit_range)
        j = jmak_fit(tm, X, None if fixed else 0.0)
        f.update(n_jmak=j["n_jmak"], k_jmak=j["k_jmak"], tau=j["t0_fit"] if fixed else 0.0)
        f["t50"] = level_temperature(tm, X, 0.5, m=3)
        f["t10"] = level_temperature(tm, X, 0.1, m=3)
        f["t90"] = level_temperature(tm, X, 0.9, m=3)
        f.update(_rate(tm, X))
        w, c = res.width.get(pk), res.center.get(pk)
        f["fwhm_end"] = (
            float(np.nanmedian(w[-st.n_norm :]))
            if w is not None and np.isfinite(w).any()
            else np.nan
        )
        f["center_end"] = (
            float(np.nanmedian(c[-st.n_norm :]))
            if c is not None and np.isfinite(c).any()
            else np.nan
        )
        res.fits[pk] = f

    # 4) 온도: 등온 구간 평균 (하온 전에서 이미 잘림; 없으면 X 10–90 % 동안 평균)
    if h is not None:
        res.T_iso = float(np.nanmean(T[h:]))
    else:
        X0 = res.X[pk0]
        act = (X0 > 0.1) & (X0 < 0.9) & np.isfinite(T)
        if act.sum() >= 2:
            res.T_iso = float(np.mean(T[act]))
    return res


def _arr_fit(T: pd.Series, k: pd.Series) -> tuple[float, float, float, float]:
    """ln k = ln k0 − Ea/(kB·T) → (Ea [eV], 오차, ln k0, R²). 점 2개면 오차·R² 는 NaN."""
    ok = np.isfinite(k) & np.isfinite(T) & (k > 0)
    if T[ok].round().nunique() < 2:
        return (np.nan,) * 4
    lr = linregress(1 / (T[ok] + 273.15), np.log(k[ok]))
    many = ok.sum() > 2
    return (
        -lr.slope * KB_EV,
        lr.stderr * KB_EV if many else np.nan,
        lr.intercept,
        lr.rvalue**2 if many else np.nan,
    )


def arrhenius(table: pd.DataFrame) -> pd.DataFrame:
    """같은 sample · 피크의 측정들: ln k vs 1/T → Ea [eV] (온도 2개 이상일 때만).
    Ea_eV = Avrami 플롯 k 로, Ea_jmak_eV = JMAK 피팅 k 로, Ea_t50_eV = 1/t50 로 (모델 없이)."""
    rows = []
    for (sample, pk), d in table.groupby(["sample", "peak"], sort=False):
        Ea, err, lnk0, r2 = _arr_fit(d["T_iso"], d["k"])
        if not np.isfinite(Ea):
            continue
        Ej, ej, _, _ = _arr_fit(d["T_iso"], d["k_jmak"])
        Et, et, _, _ = _arr_fit(d["T_iso"], 1 / d["t50"].where(d["t50"] > 0))
        rows.append(
            {
                "sample": sample,
                "peak": pk,
                "n_temps": int(d["T_iso"].round().nunique()),
                "Ea_eV": Ea,
                "Ea_err_eV": err,
                "lnk0": lnk0,
                "r2": r2,
                "Ea_jmak_eV": Ej,
                "Ea_jmak_err_eV": ej,
                "Ea_t50_eV": Et,
                "Ea_t50_err_eV": et,
            }
        )
    return pd.DataFrame(
        rows,
        columns=[
            "sample",
            "peak",
            "n_temps",
            "Ea_eV",
            "Ea_err_eV",
            "lnk0",
            "r2",
            "Ea_jmak_eV",
            "Ea_jmak_err_eV",
            "Ea_t50_eV",
            "Ea_t50_err_eV",
        ],
    )


# ═════════════════════════════ 그림 ═════════════════════════════
def _style(results: list[IsoResult]) -> dict[str, tuple[str, str, str]]:
    """측정별 (색, 선 모양, 마커): 색 = sample, 선·마커 = 그 sample 안의 온도 순서."""
    samples = list(dict.fromkeys(r.run.sample for r in results))
    out = {}
    for si, s in enumerate(samples):
        rs = sorted(
            (r for r in results if r.run.sample == s), key=lambda r: np.nan_to_num(r.T_iso, nan=1e9)
        )
        for j, r in enumerate(rs):
            c = r.run.exp.color or EXP_COLORS[si % len(EXP_COLORS)]
            out[r.run.exp.name] = (c, LINESTYLES[j % len(LINESTYLES)], MARKERS[j % len(MARKERS)])
    return out


def plot(results: list[IsoResult], arr: pd.DataFrame, st: Settings, opt: IsoOptions) -> plt.Figure:
    """맨 위: 온도 (t − t0). 피크마다 한 줄: X(t) · Avrami 플롯 · 국소 n(X) · FWHM(t) · Arrhenius."""
    peaks = list(st.peaks)
    sty = _style(results)
    good = [r for r in results if r.ok]
    with plt.rc_context(ORIGIN_RC):
        fig = plt.figure(figsize=(27, 3.6 + 5.0 * len(peaks)))
        gs = fig.add_gridspec(
            1 + len(peaks), 5, height_ratios=[0.7] + [1] * len(peaks), hspace=0.38, wspace=0.3
        )
        ax_T = fig.add_subplot(gs[0, :3])
        for r in good:
            c, ls, _ = sty[r.run.exp.name]
            ax_T.plot(
                r.tm,
                r.T,
                ls=ls,
                color=c,
                lw=2,
                label=f"{r.run.exp.name}  {r.T_iso:.0f} °C"
                if np.isfinite(r.T_iso)
                else r.run.exp.name,
            )
        ax_T.axvline(0, color="0.5", lw=1, ls=":")
        ax_T.set(xlabel="t − t$_0$ (min)", ylabel="T (°C)")
        ax_T.set_title("temperature  (t$_0$ = isothermal start)", fontsize=15, loc="left")
        if good:
            ax_T.legend(fontsize=11, loc="lower right")
        ax_tab = fig.add_subplot(gs[0, 3:])  # 요약 표
        ax_tab.axis("off")
        rows = [
            [
                r.run.exp.name,
                pk,
                f"{r.T_iso:.0f}",
                f"{f['n']:.2f}",
                f"{f['k']:.3g}",
                f"{f['t10']:.1f}",
                f"{f['t50']:.1f}",
                f"{f['t90']:.1f}",
                f"{f['tau']:.1f}",
            ]
            for r in good
            for pk, f in r.fits.items()
        ]
        if rows:
            tb = ax_tab.table(
                cellText=rows,
                colLabels=["run", "peak", "T °C", "n", "k /min", "t10", "t50", "t90", "τ"],
                loc="center",
                cellLoc="center",
            )
            tb.auto_set_font_size(False)
            tb.set_fontsize(12)
            tb.scale(1, 1.35)

        for i, pk in enumerate(peaks, start=1):
            ax_x, ax_a, ax_n, ax_w, ax_r = (fig.add_subplot(gs[i, j]) for j in range(5))
            for r in good:
                c, ls, mk = sty[r.run.exp.name]
                f = r.fits[pk]
                X, tm = r.X[pk], r.tm
                T = f"{r.T_iso:.0f} °C" if np.isfinite(r.T_iso) else "T –"
                lab = f"{r.run.exp.name}  {T}  n={f['n']:.2f}"
                ax_x.plot(tm, X, mk, color=c, ms=3, alpha=0.6, mfc="none")
                if np.isfinite(f["n_jmak"]):
                    ts = np.linspace(np.nanmin(tm), np.nanmax(tm), 400)
                    ax_x.plot(
                        ts,
                        _jmak(ts, f["n_jmak"], f["k_jmak"], f["tau"]),
                        ls=ls,
                        color=c,
                        lw=1.8,
                        label=lab,
                    )
                else:
                    ax_x.plot([], [], ls=ls, color=c, label=lab)
                # Avrami 플롯: 피팅 범위 점은 채우고 나머지는 흐리게
                with np.errstate(all="ignore"):
                    lx, ly = np.log(tm), np.log(-np.log(1 - X))
                inr = (tm > 0) & (X > opt.fit_range[0]) & (X < opt.fit_range[1])
                if np.isfinite(f["n"]):
                    inr &= tm <= f["t_fit_end"]
                ax_a.plot(
                    lx[(tm > 0) & ~inr],
                    ly[(tm > 0) & ~inr],
                    mk,
                    color=c,
                    ms=3,
                    alpha=0.25,
                    mfc="none",
                )
                if inr.any():
                    ax_a.plot(lx[inr], ly[inr], mk, color=c, ms=4)
                if np.isfinite(f["n"]):
                    xs = np.array([lx[inr].min(), lx[inr].max()])
                    ax_a.plot(
                        xs,
                        f["n"] * (xs + np.log(f["k"])),
                        ls=ls,
                        color=c,
                        lw=1.6,
                        label=f"{r.run.exp.name}  n = {f['n']:.2f} ± {f['n_err']:.2f}",
                    )
                # 국소 n(X)
                xn, nn = local_n(tm, X)
                ax_n.plot(xn, nn, ls=ls, color=c, lw=1.8, label=r.run.exp.name)
                # FWHM(t)
                w = r.width.get(pk)
                if w is not None:
                    ax_w.plot(tm, w, mk, color=c, ms=3, alpha=0.3, mfc="none")
                    ax_w.plot(
                        tm,
                        smooth(w, 7),
                        ls=ls,
                        color=c,
                        lw=1.8,
                        label=f"{r.run.exp.name}  end {f['fwhm_end']:.3f}°",
                    )
            ax_x.axhline(0.5, color="0.6", lw=0.9, ls=(0, (5, 3)), zorder=0)
            ax_x.axvline(0, color="0.6", lw=0.9, ls=":", zorder=0)
            ax_x.set(xlabel="t − t$_0$ (min)", ylabel="X", ylim=(-0.1, 1.15))
            ax_x.set_title(f"{pk}  X(t)  (lines: JMAK fit)", fontsize=15, loc="left")
            ax_x.legend(fontsize=11, loc="lower right")
            for v in opt.fit_range:
                ax_a.axhline(np.log(-np.log(1 - v)), color="0.7", lw=0.8, ls=":")
            ax_a.set(xlabel="ln[(t − t$_0$) / min]", ylabel="ln[−ln(1 − X)]")
            ax_a.set_title(
                f"{pk}  Avrami plot  (filled: {opt.fit_range[0]:g} < X < {opt.fit_range[1]:g})",
                fontsize=15,
                loc="left",
            )
            ax_a.legend(fontsize=11, loc="upper left")
            ax_n.set(xlabel="X", ylabel="local n", xlim=(0, 1))
            ax_n.set_title(f"{pk}  local Avrami exponent", fontsize=15, loc="left")
            if good:
                ax_n.legend(fontsize=11)
            ax_w.axvline(0, color="0.6", lw=0.9, ls=":", zorder=0)
            ax_w.set(xlabel="t − t$_0$ (min)", ylabel="FWHM (deg)")
            ax_w.set_title(f"{pk}  FWHM (per-frame fit)", fontsize=15, loc="left")
            if good:
                ax_w.legend(fontsize=11)
            # Arrhenius
            for r in good:
                c, _, mk = sty[r.run.exp.name]
                k = r.fits[pk]["k"]
                if np.isfinite(r.T_iso) and np.isfinite(k):
                    ax_r.plot(1000 / (r.T_iso + 273.15), np.log(k), mk, color=c, ms=8, mec="black")
            for _, a in arr[arr["peak"] == pk].iterrows():
                rs = [r for r in good if r.run.sample == a["sample"] and np.isfinite(r.T_iso)]
                c = sty[rs[0].run.exp.name][0]
                inv = np.array([1 / (r.T_iso + 273.15) for r in rs])
                xs = np.array([inv.min(), inv.max()])
                err = f" ± {a['Ea_err_eV']:.2g}" if np.isfinite(a["Ea_err_eV"]) else ""
                ax_r.plot(
                    1000 * xs,
                    a["lnk0"] - a["Ea_eV"] / KB_EV * xs,
                    color=c,
                    lw=1.5,
                    label=f"{a['sample']}  E$_a$ = {a['Ea_eV']:.2f}{err} eV  (t50: {a['Ea_t50_eV']:.2f})",
                )
            ax_r.set(xlabel="1000 / T (K$^{-1}$)", ylabel="ln(k / min$^{-1}$)")
            ax_r.set_title(f"{pk}  Arrhenius", fontsize=15, loc="left")
            if len(arr[arr["peak"] == pk]):
                ax_r.legend(fontsize=11)
            else:
                ax_r.text(
                    0.5,
                    0.5,
                    "Ea: needs ≥ 2 temperatures\nper sample",
                    transform=ax_r.transAxes,
                    ha="center",
                    va="center",
                    fontsize=14,
                    color="0.4",
                )
    return fig


DASH = {"-": "solid", "--": "dash", ":": "dot", "-.": "dashdot"}
SYMBOL = {"o": "circle", "s": "square", "^": "triangle-up", "D": "diamond", "v": "triangle-down"}


def _fmt(v: float, f: str) -> str:
    return format(v, f) if np.isfinite(v) else "–"


def _ref(fig, row: int, col: int) -> tuple[str, str]:
    """서브플롯의 축 이름 ('x3', 'y3')."""
    ax = fig.get_subplot(row, col)
    return ax.xaxis.plotly_name.replace("axis", ""), ax.yaxis.plotly_name.replace("axis", "")


def _vl(fig, x: float, row: int, col: int, **line) -> None:
    """세로 기준선 (표가 든 그림에선 add_vline 이 안 돼서 직접)."""
    xr, yr = _ref(fig, row, col)
    fig.add_shape(
        type="line", x0=x, x1=x, y0=0, y1=1, xref=xr, yref=f"{yr} domain", line=line, layer="below"
    )


def _hl(fig, y: float, row: int, col: int, **line) -> None:
    xr, yr = _ref(fig, row, col)
    fig.add_shape(
        type="line", y0=y, y1=y, x0=0, x1=1, yref=yr, xref=f"{xr} domain", line=line, layer="below"
    )


def write_html(
    results: list[IsoResult],
    arr: pd.DataFrame,
    st: Settings,
    opt: IsoOptions,
    path: Path,
    title: str = "",
) -> Path:
    """브라우저용 한 장 (compare · overview 와 같은 스타일). 위에서부터
    요약 표 · 온도 · [피크별] X(t) · Avrami 플롯 · 국소 n(X) · FWHM(t) · (온도 2개 이상이면) Arrhenius."""
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    peaks = list(st.peaks)
    nc = len(peaks)
    sty = _style(results)
    good = [r for r in results if r.ok]
    rows_tab = [(r, pk, f) for r in good for pk, f in r.fits.items()]
    has_arr = len(arr) > 0
    kinds = ["X", "avrami", "localn", "fwhm"] + (["arr"] if has_arr else [])
    span = [{"colspan": nc}] + [None] * (nc - 1)
    specs = [[{"type": "table", **span[0]}] + span[1:], span] + [[{}] * nc for _ in kinds]
    names = {
        "X": "X(t)  — lines: JMAK fit",
        "avrami": f"Avrami plot  (filled: {opt.fit_range[0]:g} < X < {opt.fit_range[1]:g})",
        "localn": "local Avrami exponent n(X)",
        "fwhm": "FWHM (per-frame fit)",
        "arr": "Arrhenius  (k from Avrami plot)",
    }
    titles = ["", "Temperature  (t₀ = isothermal start)"] + [
        f"<b>{pk}</b>  {names[k]}" for k in kinds for pk in peaks
    ]
    tab_h = 46 + 30 * len(rows_tab)
    heights = [tab_h, 250] + [380] * len(kinds)
    gap = 105  # 행 사이 [px]: x 축 제목 + 다음 행 제목
    plot_h = sum(heights) + gap * (len(heights) - 1)
    H = plot_h + 70 + 60  # + 위 · 아래 여백
    fig = make_subplots(
        rows=len(heights),
        cols=nc,
        specs=specs,
        subplot_titles=titles,
        row_heights=[h / sum(heights) for h in heights],
        vertical_spacing=gap / plot_h,
        horizontal_spacing=0.08,
    )
    row = {k: i + 3 for i, k in enumerate(kinds)}

    # 요약 표
    head = [
        "run",
        "peak",
        "T (°C)",
        "n",
        "k (1/min)",
        "t10",
        "t50",
        "t90",
        "τ",
        "max dX/dt",
        "FWHM end",
        "D app (nm)",
    ]
    cells = (
        [
            [r.run.exp.name for r, _, _ in rows_tab],
            [pk for _, pk, _ in rows_tab],
            [_fmt(r.T_iso, ".1f") for r, _, _ in rows_tab],
            [f"{_fmt(f['n'], '.2f')} ± {_fmt(f['n_err'], '.2f')}" for _, _, f in rows_tab],
            [_fmt(f["k"], ".3g") for _, _, f in rows_tab],
        ]
        + [[_fmt(f[k], ".1f") for _, _, f in rows_tab] for k in ("t10", "t50", "t90", "tau")]
        + [
            [_fmt(f["rate_max"], ".2f") for _, _, f in rows_tab],
            [_fmt(f["fwhm_end"], ".3f") + "°" for _, _, f in rows_tab],
            [_fmt(f.get("D_app_nm", np.nan), ".1f") for _, _, f in rows_tab],
        ]
    )
    colors = [[sty[r.run.exp.name][0] for r, _, _ in rows_tab]] + [["black"] * len(rows_tab)] * (
        len(head) - 1
    )
    fig.add_trace(
        go.Table(
            columnwidth=[1.1, 0.8, 0.8, 1.1, 0.9, 0.6, 0.6, 0.6, 0.6, 0.9, 0.9, 0.9],
            header=dict(
                values=[f"<b>{h}</b>" for h in head],
                fill_color="#eeeeee",
                line_color="black",
                font=dict(size=15, color="black"),
                height=32,
                align="center",
            ),
            cells=dict(
                values=cells,
                line_color="#999",
                font=dict(size=15, color=colors),
                height=30,
                align="center",
            ),
        ),
        1,
        1,
    )

    # 온도
    for r in good:
        c, ls, _ = sty[r.run.exp.name]
        fig.add_trace(
            go.Scatter(
                x=r.tm,
                y=r.T,
                mode="lines",
                line=dict(color=c, width=2.4, dash=DASH[ls]),
                legendgroup=r.run.exp.name,
                showlegend=False,
                hovertemplate=f"<b>{r.run.exp.name}</b><br>%{{x:.2f}} min · %{{y:.1f}} °C<extra></extra>",
            ),
            2,
            1,
        )
    _vl(fig, 0, 2, 1, color="#888", dash="dot", width=1.2)

    for j, pk in enumerate(peaks, start=1):
        lg = "legend" if j == 1 else f"legend{j}"
        for r in good:
            c, ls, mk = sty[r.run.exp.name]
            f, X, tm, nm = r.fits[pk], r.X[pk], r.tm, r.run.exp.name
            T = f"{r.T_iso:.0f} °C" if np.isfinite(r.T_iso) else "T –"
            hov = f"<b>{nm} {pk}</b><br>"
            # X(t) + JMAK
            fig.add_trace(
                go.Scatter(
                    x=tm,
                    y=X,
                    mode="markers",
                    legendgroup=nm,
                    showlegend=False,
                    marker=dict(color=c, size=5, symbol=SYMBOL[mk] + "-open"),
                    hovertemplate=hov + "%{x:.2f} min · X %{y:.3f}<extra></extra>",
                ),
                row["X"],
                j,
            )
            if np.isfinite(f["n_jmak"]):
                ts = np.linspace(np.nanmin(tm), np.nanmax(tm), 400)
                fig.add_trace(
                    go.Scatter(
                        x=ts,
                        y=_jmak(ts, f["n_jmak"], f["k_jmak"], f["tau"]),
                        mode="lines",
                        name=f"{nm}  {T}  n = {_fmt(f['n'], '.2f')}",
                        legend=lg,
                        legendgroup=nm,
                        line=dict(color=c, width=2.4, dash=DASH[ls]),
                        hoverinfo="skip",
                    ),
                    row["X"],
                    j,
                )
            # Avrami 플롯
            with np.errstate(all="ignore"):
                lx, ly = np.log(tm), np.log(-np.log(1 - X))
            inr = (tm > 0) & (X > opt.fit_range[0]) & (X < opt.fit_range[1])
            if np.isfinite(f["n"]):
                inr &= tm <= f["t_fit_end"]
            out = (tm > 0) & ~inr
            fig.add_trace(
                go.Scatter(
                    x=lx[out],
                    y=ly[out],
                    mode="markers",
                    legendgroup=nm,
                    showlegend=False,
                    marker=dict(color=c, size=5, opacity=0.3, symbol=SYMBOL[mk] + "-open"),
                    hoverinfo="skip",
                ),
                row["avrami"],
                j,
            )
            fig.add_trace(
                go.Scatter(
                    x=lx[inr],
                    y=ly[inr],
                    mode="markers",
                    legendgroup=nm,
                    showlegend=False,
                    marker=dict(color=c, size=7, symbol=SYMBOL[mk]),
                    hovertemplate=hov + "ln t %{x:.2f} · %{y:.2f}<extra></extra>",
                ),
                row["avrami"],
                j,
            )
            if np.isfinite(f["n"]) and inr.any():
                xs = np.array([lx[inr].min(), lx[inr].max()])
                fig.add_trace(
                    go.Scatter(
                        x=xs,
                        y=f["n"] * (xs + np.log(f["k"])),
                        mode="lines",
                        legendgroup=nm,
                        showlegend=False,
                        line=dict(color=c, width=2.4, dash=DASH[ls]),
                        hovertemplate=hov + f"n = {f['n']:.2f} ± {f['n_err']:.2f}<extra></extra>",
                    ),
                    row["avrami"],
                    j,
                )
            # 국소 n(X)
            xn, nn = local_n(tm, X)
            fig.add_trace(
                go.Scatter(
                    x=xn,
                    y=nn,
                    mode="lines",
                    legendgroup=nm,
                    showlegend=False,
                    line=dict(color=c, width=2.4, dash=DASH[ls]),
                    hovertemplate=hov + "X %{x:.2f} · n %{y:.2f}<extra></extra>",
                ),
                row["localn"],
                j,
            )
            # FWHM(t)
            w = r.width.get(pk)
            if w is not None:
                fig.add_trace(
                    go.Scatter(
                        x=tm,
                        y=w,
                        mode="markers",
                        legendgroup=nm,
                        showlegend=False,
                        marker=dict(color=c, size=4, opacity=0.3),
                        hoverinfo="skip",
                    ),
                    row["fwhm"],
                    j,
                )
                fig.add_trace(
                    go.Scatter(
                        x=tm,
                        y=smooth(w, 7),
                        mode="lines",
                        legendgroup=nm,
                        showlegend=False,
                        line=dict(color=c, width=2.4, dash=DASH[ls]),
                        hovertemplate=hov + "%{x:.2f} min · FWHM %{y:.3f}°<extra></extra>",
                    ),
                    row["fwhm"],
                    j,
                )
            if has_arr and np.isfinite(r.T_iso) and np.isfinite(f["k"]):
                fig.add_trace(
                    go.Scatter(
                        x=[1000 / (r.T_iso + 273.15)],
                        y=[np.log(f["k"])],
                        mode="markers",
                        legendgroup=nm,
                        showlegend=False,
                        marker=dict(
                            color=c, size=12, symbol=SYMBOL[mk], line=dict(color="black", width=1.2)
                        ),
                        hovertemplate=hov + f"{r.T_iso:.0f} °C · k {f['k']:.3g}<extra></extra>",
                    ),
                    row["arr"],
                    j,
                )
        if has_arr:
            for _, a in arr[arr["peak"] == pk].iterrows():
                rs = [r for r in good if r.run.sample == a["sample"] and np.isfinite(r.T_iso)]
                inv = np.array([1 / (r.T_iso + 273.15) for r in rs])
                xs = np.array([inv.min(), inv.max()])
                err = f" ± {a['Ea_err_eV']:.2g}" if np.isfinite(a["Ea_err_eV"]) else ""
                fig.add_trace(
                    go.Scatter(
                        x=1000 * xs,
                        y=a["lnk0"] - a["Ea_eV"] / KB_EV * xs,
                        mode="lines",
                        legend=lg,
                        name=f"{a['sample']}  Ea = {a['Ea_eV']:.2f}{err} eV",
                        line=dict(color=sty[rs[0].run.exp.name][0], width=2),
                    ),
                    row["arr"],
                    j,
                )
        _hl(fig, 0.5, row["X"], j, color="#999", dash="dash", width=1)
        _vl(fig, 0, row["X"], j, color="#999", dash="dot", width=1)
        _vl(fig, 0, row["fwhm"], j, color="#999", dash="dot", width=1)
        for v in opt.fit_range:
            _hl(
                fig,
                float(np.log(-np.log(1 - v))),
                row["avrami"],
                j,
                color="#bbb",
                dash="dot",
                width=1,
            )
        fig.update_xaxes(title="t − t₀ (min)", row=row["X"], col=j)
        fig.update_yaxes(title="X", range=[-0.1, 1.15], row=row["X"], col=j)
        fig.update_xaxes(title="ln[(t − t₀) / min]", row=row["avrami"], col=j)
        fig.update_yaxes(title="ln[−ln(1 − X)]", row=row["avrami"], col=j)
        fig.update_xaxes(title="X", range=[0, 1], row=row["localn"], col=j)
        fig.update_yaxes(title="local n", row=row["localn"], col=j)
        fig.update_xaxes(title="t − t₀ (min)", row=row["fwhm"], col=j)
        fig.update_yaxes(title="FWHM (°)", row=row["fwhm"], col=j)
        if has_arr:
            fig.update_xaxes(title="1000 / T (1/K)", row=row["arr"], col=j)
            fig.update_yaxes(title="ln(k / min⁻¹)", row=row["arr"], col=j)
        ax = fig.get_subplot(row["X"], j)  # 범례: X(t) 그래프 안 오른쪽 아래
        fig.update_layout(
            {
                lg: dict(
                    x=ax.xaxis.domain[1] - 0.005,
                    xanchor="right",
                    y=ax.yaxis.domain[0] + 0.004,
                    yanchor="bottom",
                    bgcolor="rgba(255,255,255,0.85)",
                    bordercolor="#999",
                    borderwidth=1,
                    font=dict(size=14),
                )
            }
        )
    fig.update_xaxes(title="t − t₀ (min)", row=2, col=1)
    fig.update_yaxes(title="T (°C)", row=2, col=1)
    fig.update_layout(
        template="simple_white",
        title=dict(text=f"<b>{title}</b>", x=0.01, y=0.995),
        margin=dict(t=70, l=90, r=30, b=60),
    )
    origin_plotly(fig)
    W = 160 + 620 * nc
    return write_fit_html(fig, path, W, H, fit_height=False)


def compare_html(
    results: list[IsoResult],
    st: Settings,
    path: Path,
    group: str,
    colors: dict[str, str] | None = None,
) -> Path:
    """compare 와 같은 모양의 등온 비교: 피크마다 한 열, 위 X(t) (t = 0 = 등온 시작, ◆ = t50) · 아래 FWHM(t)."""
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    from .compare import LABEL_ROWS
    from .style import LEGEND, header

    peaks = list(st.peaks)
    good = [r for r in results if r.ok]
    sty = _style(results)
    col = {
        r.run.exp.name: (colors or {}).get(r.run.exp.name) or sty[r.run.exp.name][0]
        for r in results
    }
    fig = make_subplots(
        rows=2,
        cols=len(peaks),
        subplot_titles=[f"<b>{pk}</b>" for pk in peaks],
        shared_xaxes=True,
        horizontal_spacing=0.06,
        vertical_spacing=0.05,
        row_heights=[0.56, 0.44],
    )
    t_all = np.concatenate([r.tm for r in good]) if good else np.array([0.0, 1.0])
    span = float(np.nanmax(t_all) - np.nanmin(t_all)) or 1.0
    for j, pk in enumerate(peaks, start=1):
        lg = "legend" if j == 1 else f"legend{j}"
        for r in good:
            c, nm, f = col[r.run.exp.name], r.run.exp.name, r.fits[pk]
            T = f"{r.T_iso:.0f} °C" if np.isfinite(r.T_iso) else "T –"
            fig.add_trace(
                go.Scatter(
                    x=r.tm,
                    y=r.X[pk],
                    mode="lines+markers",
                    legend=lg,
                    legendgroup=nm,
                    name=f"{nm}  {T}  t50 {f['t50']:.1f} min  (n {f['n']:.2f})",
                    marker=dict(size=4, color=c),
                    line=dict(color=c, width=1.5),
                    hovertemplate=f"<b>{nm} {pk}</b><br>%{{x:.2f}} min · X %{{y:.3f}}<extra></extra>",
                ),
                1,
                j,
            )
            w = r.width.get(pk)
            if w is not None:
                fig.add_trace(
                    go.Scatter(
                        x=r.tm,
                        y=w,
                        mode="markers",
                        legendgroup=nm,
                        showlegend=False,
                        marker=dict(size=4, color=c, opacity=0.35),
                        hoverinfo="skip",
                    ),
                    2,
                    j,
                )
                fig.add_trace(
                    go.Scatter(
                        x=r.tm,
                        y=smooth(w, 7),
                        mode="lines",
                        legendgroup=nm,
                        showlegend=False,
                        line=dict(color=c, width=2),
                        hovertemplate=f"<b>{nm} {pk}</b><br>%{{x:.2f}} min · FWHM %{{y:.3f}}°"
                        "<extra></extra>",
                    ),
                    2,
                    j,
                )
        fig.add_hline(y=0.5, line=dict(color="#999", dash="dash", width=1), row=1, col=j)
        for row in (1, 2):
            fig.add_vline(x=0, line=dict(color="#555", dash="dot", width=1.2), row=row, col=j)
        # t50: 세로 점선 · ◆ · 위쪽 숫자 상자 (compare 의 T50 과 같은 표시), 가까우면 다음 줄
        last: dict[int, float] = {}
        for r in sorted(good, key=lambda r: np.nan_to_num(r.fits[pk]["t50"], nan=np.inf)):
            v = r.fits[pk]["t50"]
            if not np.isfinite(v):
                continue
            k = next(
                (i for i in range(len(LABEL_ROWS)) if v - last.get(i, -np.inf) > 0.09 * span), 0
            )
            last[k] = v
            c = col[r.run.exp.name]
            for row in (1, 2):
                fig.add_vline(x=v, line=dict(color=c, dash="dash", width=1.4), row=row, col=j)
            fig.add_trace(
                go.Scatter(
                    x=[v],
                    y=[0.5],
                    mode="markers",
                    legendgroup=r.run.exp.name,
                    showlegend=False,
                    marker=dict(
                        symbol="diamond", size=13, color=c, line=dict(color="black", width=1.2)
                    ),
                    hovertemplate=f"<b>{r.run.exp.name} {pk}  t50 {v:.2f} min</b><extra></extra>",
                ),
                1,
                j,
            )
            fig.add_annotation(
                x=v,
                y=LABEL_ROWS[k],
                text=f"<b>{v:.1f}</b>",
                showarrow=False,
                row=1,
                col=j,
                font=dict(color=c, size=16),
                bgcolor="white",
                bordercolor=c,
                borderwidth=1.2,
                borderpad=2,
            )
    leg = {}
    for j in range(1, len(peaks) + 1):
        ax = fig.get_subplot(1, j)
        leg["legend" if j == 1 else f"legend{j}"] = dict(
            x=ax.xaxis.domain[0] + 0.005,
            xanchor="left",  # 왼쪽 위
            y=ax.yaxis.domain[1] - 0.01,
            yanchor="top",
            **LEGEND,
        )
    header(
        fig,
        "isothermal" if group == "pick" else f"[{group}]  isothermal",
        "t = 0 at isothermal start  ·  ◆ / number = t50 (X = 0.5, min)  ·  legend: T · t50 · Avrami n",
    )
    origin_plotly(fig)
    fig.update_layout(margin=dict(t=80, l=90, r=30, b=60), **leg)
    fig.update_xaxes(title="t − t₀ (min)", row=2)
    fig.update_yaxes(range=[-0.1, 1.38], tickvals=[0, 0.2, 0.4, 0.6, 0.8, 1.0], row=1)
    fig.update_yaxes(matches="y", row=1)
    fig.update_yaxes(title="X (normalised area)", row=1, col=1)
    fig.update_yaxes(title="FWHM (°)", row=2, col=1)
    return write_fit_html(fig, path, 150 + 540 * len(peaks), 800)


def compare_iso(
    config: Path,
    group: str | list[Experiment],
    geo: Geometry,
    workers: int | None,
    show: bool = True,
    colors: dict[str, str] | None = None,
) -> list[IsoResult]:
    """compare 의 등온 회차 ([[iso]] ...) 또는 고른 등온 실험들: 온도 대신 시간 (등온 시작부터) 으로 X · FWHM 비교."""
    import webbrowser
    from dataclasses import replace

    if isinstance(group, str):
        st, runs, opt = load_iso(config, group)
    else:  # 고른 실험들 (회차 섞임): 이름 = 회차.이름
        picks, runs = group, []
        for e in picks:
            st, rs, opt = load_iso(config, e.group)
            r = next(r for r in rs if r.exp.name == e.name)
            runs.append(replace(r, exp=replace(r.exp, name=f"{e.group}.{e.name}")))
        group = "pick"
    results = []
    for r in runs:
        d = load_exp(r.exp, st, geo, workers)
        results.append(analyze_iso(r, d.x, d.Z, d.t, d.T, d.files, st, opt))
    rows = [
        {
            "exp": r.run.exp.name,
            "T_iso": round(r.T_iso, 1),
            "t0_from": r.t0_src,
            **{
                f"{k} {pk}": round(r.fits[pk][k], 3)
                for pk in st.peaks
                if pk in r.fits
                for k in ("t50", "n", "k", "fwhm_end")
            },
        }
        for r in results
        if r.ok
    ]
    for r in results:
        if not r.ok:
            print(f"❌ {r.run.exp.name}: 분석 못 함 — {r.fail}")
    summary = pd.DataFrame(rows)
    print(
        f"\n📊 [[{group}]] 등온 — t50 = 등온 시작부터 X 0.5 까지 [min], k [1/min]\n"
        + summary.to_string(index=False)
    )
    d = out_dir("compare")
    stem = (
        "+".join(r.run.exp.name.replace(".", "_") for r in results)
        if group == "pick"
        else f"{Path(config).stem}_{group}"
    )
    page = compare_html(results, st, d / f"{stem}.html", group, colors)
    if saving():
        summary.to_csv(d / f"{stem}_t50.csv", index=False, encoding="utf-8-sig")
        print(f"\n💾 {page}\n💾 {d / f'{stem}_t50.csv'}")
    if show:
        webbrowser.open_new(page.resolve().as_uri())
    return results


# ═════════════════════════════ 실행 ═════════════════════════════
def run(
    config: Path = DEFAULT_CONFIG,
    geo: Geometry = Geometry(),
    workers: int | None = None,
    only: list[str] | None = None,
    show: bool = True,
    section: str | list[str] | None = None,
) -> list[IsoResult]:
    """section: 회차 하나 · 여러 개, 생략하면 iso 로 시작하는 회차 전부 ([[iso]], [[iso2]] ...)
    → 온도가 다른 측정이 다른 회차에 있어도 한꺼번에 Arrhenius."""
    from .compare import exp_groups

    secs = [section] if isinstance(section, str) else list(section or [])
    if not secs:
        secs = [g for g in exp_groups(load_toml(config)) if g.startswith("iso")] or ["iso"]
    st, runs, opt = load_iso(config, secs[0])
    for g in secs[1:]:
        runs += load_iso(config, g)[1]
    section = "+".join(secs)
    if only:
        runs = [r for r in runs if r.exp.name in only or r.sample in only]
        if not runs:
            raise SystemExit(f"❌ --only {' '.join(only)} 에 맞는 [[{section}]] 없음")
    print(
        f"📋 {Path(config)} [[{section}]]  —  측정 {len(runs)}개, 피크 {', '.join(st.peaks)}, "
        f"Avrami 피팅 {opt.fit_range[0]:g} < X < {opt.fit_range[1]:g}"
    )
    check_paths([r.exp for r in runs])
    results = []
    for r in runs:
        d = load_exp(r.exp, st, geo, workers)
        res = analyze_iso(r, d.x, d.Z, d.t, d.T, d.files, st, opt)
        results.append(res)

    for r in results:  # 겉보기 결정 크기 (Scherrer K=0.9, 장비 폭 보정 없음) [nm]
        for f in r.fits.values():
            b, c = np.deg2rad(f["fwhm_end"]), np.deg2rad(f["center_end"] / 2)
            f["D_app_nm"] = 0.9 * geo.wavelength / (b * np.cos(c)) / 10 if b > 0 else np.nan
    table = pd.DataFrame(
        [
            {
                "run": r.run.exp.name,
                "sample": r.run.sample,
                "peak": pk,
                "T_iso": r.T_iso,
                "t0_kst": fmt_kst(r.t0),
                "t0_from": r.t0_src,
                **f,
            }
            for r in results
            for pk, f in r.fits.items()
        ]
    )
    for r in results:
        if not r.ok:
            print(f"❌ {r.run.exp.name}: 분석 못 함 — {r.fail}")
    if table.empty:
        raise SystemExit("❌ 분석된 측정이 없음")
    arr = arrhenius(table)

    stem = f"{Path(config).stem}_{section}_avrami"
    cols = [
        "run",
        "peak",
        "T_iso",
        "t0_from",
        "n",
        "n_err",
        "k",
        "r2",
        "n_jmak",
        "tau",
        "t10",
        "t50",
        "t90",
        "rate_max",
        "t_rate_max",
        "fwhm_end",
        "D_app_nm",
    ]
    print(
        "\n📊 Avrami  (k · rate_max [1/min], 시간 = t0 부터 [min], tau = JMAK 잠복시간, "
        "D_app = 등온 끝 Scherrer 겉보기 크기 [nm])\n" + table[cols].round(3).to_string(index=False)
    )
    if len(arr):
        print("\n📈 Arrhenius\n" + arr.round(3).to_string(index=False))
    d = out_dir("avrami")
    page = write_html(results, arr, st, opt, d / f"{stem}.html", title=f"Avrami  [[{section}]]")
    if saving():
        table.to_csv(d / f"{stem}.csv", index=False, encoding="utf-8-sig")
        arr.to_csv(d / f"{stem}_arrhenius.csv", index=False, encoding="utf-8-sig")
        pd.concat(
            [
                pd.DataFrame(
                    {
                        "run": r.run.exp.name,
                        "file": r.files,
                        "t_min": r.tm,
                        "T_pv": r.T,
                        **{f"X_{pk}": X for pk, X in r.X.items()},
                        **{f"FWHM_{pk}": w for pk, w in r.width.items()},
                        **{f"center_{pk}": c for pk, c in r.center.items()},
                    }
                )
                for r in results
                if r.ok
            ]
        ).to_csv(d / f"{stem}_frames.csv", index=False, encoding="utf-8-sig")
        fig = plot(results, arr, st, opt)
        png = d / f"{stem}.png"
        with plt.rc_context(ORIGIN_RC):
            fig.savefig(png)
        plt.close(fig)
        print(f"\n💾 {png}\n💾 {page}\n💾 {d / f'{stem}.csv'}")
    if show:
        import webbrowser

        webbrowser.open_new(page.resolve().as_uri())
    return results
