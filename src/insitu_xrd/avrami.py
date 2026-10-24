"""Crystallization change X(t) to the Avrami (JMAK) exponent n, rate constant k and, when the same
sample has two or more isothermal temperatures, the activation energy Ea.

A measurement is heating, isothermal hold, cooling. Results are reported in full, without
warnings or verdicts.

Settings: [[iso]] in experiments.toml (one measurement = one piece of amorphous sample)
    [[iso]]
    sample = "x1"                 # runs with the same sample are grouped for Arrhenius
    images = ['<image folder>']
    recipe = ['<recipe>.csv']     # optional; without it only n and k, no temperature
    T_iso  = 370                  # optional nominal temperature for naming; the real one comes from the log
    start  = "10:05"              # optional; only frames between these times (sets the first/last frames that define X = 0 / 1)
    end    = "11:30"
    t0     = "10:15:02"           # optional start time of the change; found by a JMAK fit if omitted

    uv run main.py avrami                  all [[iso]]
    uv run main.py avrami --only x1        a subset by name or sample
    uv run main.py avrami exp2             [[exp2]] instead of [[iso]] (try it on existing heating data)

Analysis
 1) X(t): as in compare, the area of the peak window (center +- fwhm_k*FWHM) of R = I/I0 - 1,
    divided by the amorphous. X = 0 is the first n_norm frames, X = 1 the last n_norm frames.
 2) t0 (t = 0): t0 from the settings > start of the isothermal hold in the temperature log
    (heating ended, set temperature reached) > JMAK fit. Once t0 is fixed, tau in the JMAK form
    X = 1 - exp(-(k*(t - t0 - tau))^n) is the incubation time after it.
 3) Avrami plot: ln[-ln(1-X)] = n*ln(t - t0) + n*ln k for X inside fit_range gives n, k.
 1') Data are cut where cooling starts (end of the hold), so X = 1 is the state at the end of the hold.
 4) Temperature T_iso: mean PV over the hold (if no hold is found, mean PV while X is 10-90 %).
 5) Two or more temperatures for the same sample: ln k vs 1/T gives Ea (from the Avrami k, the
    JMAK k and 1/t50).
 Also t10, t50, t90, the maximum crystallization rate (dX/dt) and when it occurs, the local
 Avrami exponent n(X), the change of FWHM and peak position during the hold, and the apparent
 crystallite size from the FWHM at the end of the hold (Scherrer, no instrument-width correction).

Output (--save): out/avrami/<settings name>_avrami.png / .csv / _arrhenius.csv / _frames.csv
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

KB_EV = 8.617333e-5  # Boltzmann constant [eV/K]
LINESTYLES = ["-", "--", ":", "-."]
MARKERS = ["o", "s", "^", "D", "v"]


@dataclass(slots=True)
class IsoRun:
    exp: Experiment  # name · images · recipe · start/end
    sample: str
    t0: str | None = None  # start time of the change given directly (KST)


@dataclass(slots=True)
class IsoOptions:
    fit_range: tuple[float, float] = (0.15, 0.85)
    # window to judge that the change has ended [min]; if X changes again later, stop there
    flat_min: float = 5.0
    # reaching (maximum temperature - this value) [°C] marks the start of the hold
    hold_tol: float = 3.0


def load_iso(
    path: Path = DEFAULT_CONFIG, section: str = "iso"
) -> tuple[Settings, list[IsoRun], IsoOptions]:
    """[peaks] and [options] from the settings file plus the list of [[iso]] (or [[exp]] for section="exp")."""
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


def _jmak(t, n, k, t0):
    return 1 - np.exp(-(np.clip(k * (t - t0), 0, None) ** n))


def jmak_fit(t: np.ndarray, X: np.ndarray, t0_fixed: float | None = None) -> dict:
    """Fit X = 1 - exp(-(k*(t - t0))^n). Without t0_fixed, t0 is fitted too (smallest residual over
    several starting values). t in min; the returned t0 is on the same time axis."""
    out = dict(n_jmak=np.nan, k_jmak=np.nan, t0_fit=np.nan)
    m = np.isfinite(t) & np.isfinite(X)
    t, X = t[m], X[m]
    t05, t50 = (level_temperature(t, X, v, m=3) for v in (0.05, 0.5))
    if len(t) < 8 or not np.isfinite(t50):
        return out
    t05 = t05 if np.isfinite(t05) else t[0]
    lo_t0 = t[0] - 2 * (t50 - t[0])  # t0 may lie slightly before the first frame
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
    """Straight-line fit of the Avrami plot: ln(-ln(1-X)) = n*ln t + n*ln k, with tm = t - t0 [min].
    Only points before X first exceeds rng[1] (the main change); points that fall in range from noise after the change are dropped."""
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


@dataclass(slots=True)
class IsoResult:
    run: IsoRun
    T_iso: float = np.nan  # mean PV over the hold [°C]
    t0_src: str = ""  # where t0 came from: given / hold start / JMAK
    t0: float = np.nan  # Unix (start of the change)
    tm: np.ndarray = field(default_factory=lambda: np.empty(0))  # t - t0 per frame [min]
    T: np.ndarray = field(default_factory=lambda: np.empty(0))
    files: np.ndarray = field(default_factory=lambda: np.empty(0))
    X: dict[str, np.ndarray] = field(default_factory=dict)
    width: dict[str, np.ndarray] = field(default_factory=dict)  # peak -> FWHM per frame [deg]
    center: dict[str, np.ndarray] = field(default_factory=dict)  # peak -> center 2θ per frame [deg]
    fits: dict[str, dict] = field(default_factory=dict)  # peak -> n, k, t50, ...
    fail: str = ""

    @property
    def ok(self) -> bool:
        return bool(self.fits)


def hold_range(T: np.ndarray, tol: float = 3.0, ramp: float = 20.0) -> tuple[int, int] | None:
    """Heating, hold, cooling: the [first, last] frame of the hold, i.e. the first and last frame with PV
    at or above (hold temperature - tol). The hold temperature is the median of the top PV values.
    None if the run already starts near that temperature (no heating)."""
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
    """X per peak: the first n_norm good frames = 0, the last n_norm good frames = 1.
    If shapes is a dict, it is also filled with (center per frame, FWHM) per peak."""
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
    """Index of the end of the flat stretch after the main change (X rising past 0.5) has finished,
    if later data leave that level by more than drift (intensity or position changing with heating
    or cooling). None if it stays flat (isothermal).
    Flat means the rate of change over a flat_min-minute window is below flat times the maximum rate."""
    good = np.flatnonzero(np.isfinite(X))
    if len(good) < 10:
        return None
    xs = median_filter(X[good], size=5, mode="nearest")
    ts = tt[good]
    up = np.flatnonzero(xs >= 0.5)
    w = max(3, int(round(opt.flat_min / np.median(np.diff(ts)))))
    if not len(up) or len(xs) <= w + 1:
        return None
    rate = np.abs(xs[w:] - xs[:-w])
    for j in range(up[0], len(xs) - w):
        if rate[j] < flat * rate.max():
            level = np.median(xs[j : j + w + 1])
            if np.any(np.abs(xs[j + w + 1 :] - level) > drift):
                return int(good[j + w])
            return None
    return None


def _rate(tm: np.ndarray, X: np.ndarray) -> dict:
    """Maximum crystallization rate dX/dt [1/min] and when it occurs (slope of X smoothed by a 5-frame median)."""
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
    """Local Avrami exponent n(X) = d ln[-ln(1-X)] / d ln(t - t0), after smoothing X with a 5-frame median.
    If n changes with X, the nucleation or growth mode changed along the way."""
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
    """Frame profiles Z (in time order, already cut by start/end) to X(t), n, k per peak."""
    if (
        np.isfinite(T).any() and not np.isfinite(T).all()
    ):  # frames outside the temperature log time range are not this measurement
        keep = np.isfinite(T)
        Z, t, T, files = Z[keep], t[keep], T[keep], files[keep]
    ok = beam_ok(Z)
    if ok.sum() < 4 * st.n_norm:
        return IsoResult(run, T=T, files=files, fail=f"프레임이 {ok.sum()} 개뿐")

    # Cut at the end of the hold (start of cooling). X is 0 for the first n_norm frames (early
    # heating, amorphous) and 1 for the last n_norm frames of the hold. If the temperature shows no
    # hold, cut where X changes again after the change has ended.
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

    # t0: given > start of the hold (temperature log) > JMAK fit of the first peak; shared by all peaks
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

    # Avrami plot and JMAK per peak; with t0 fixed, tau is the incubation time after it
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

    # temperature: mean over the hold (already cut before cooling); otherwise mean while X is 10-90 %
    if h is not None:
        res.T_iso = float(np.nanmean(T[h:]))
    else:
        X0 = res.X[pk0]
        act = (X0 > 0.1) & (X0 < 0.9) & np.isfinite(T)
        if act.sum() >= 2:
            res.T_iso = float(np.mean(T[act]))
    return res


def _arr_fit(T: pd.Series, k: pd.Series) -> tuple[float, float, float, float]:
    """ln k = ln k0 - Ea/(kB*T) to (Ea [eV], error, ln k0, R^2); with only 2 points error and R^2 are NaN."""
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
    """Measurements of the same sample and peak: ln k vs 1/T gives Ea [eV] (only with two or more temperatures).
    Ea_eV uses the Avrami plot k, Ea_jmak_eV the JMAK fit k, Ea_t50_eV uses 1/t50 (model-free)."""
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


def _style(results: list[IsoResult]) -> dict[str, tuple[str, str, str]]:
    """(color, line style, marker) per measurement: color by sample, line and marker by temperature order within the sample."""
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
    """Top: temperature (t - t0). One row per peak: X(t), Avrami plot, local n(X), FWHM(t), Arrhenius."""
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
        ax_tab = fig.add_subplot(gs[0, 3:])
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
                # Avrami plot: points in the fit range filled, the rest faded
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
                xn, nn = local_n(tm, X)
                ax_n.plot(xn, nn, ls=ls, color=c, lw=1.8, label=r.run.exp.name)
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
    """Axis names of a subplot ('x3', 'y3')."""
    ax = fig.get_subplot(row, col)
    return ax.xaxis.plotly_name.replace("axis", ""), ax.yaxis.plotly_name.replace("axis", "")


def _vl(fig, x: float, row: int, col: int, **line) -> None:
    """Vertical reference line, drawn by hand because add_vline does not work in a figure containing a table."""
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
    """One HTML page (same style as compare and overview). From the top: summary table, temperature,
    then per peak X(t), Avrami plot, local n(X), FWHM(t) and, with two or more temperatures, Arrhenius."""
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
    gap = 105  # space between rows [px]: x-axis title plus the next row's title
    plot_h = sum(heights) + gap * (len(heights) - 1)
    H = plot_h + 70 + 60
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
        ax = fig.get_subplot(row["X"], j)  # legend: bottom right inside the X(t) graph
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
    """Isothermal comparison shaped like compare: one column per peak, X(t) above (t = 0 at hold start, diamond = t50) and FWHM(t) below."""
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
        # t50: dashed vertical line, diamond and number box on top (as T50 in compare); close ones go to the next row
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
            xanchor="left",
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
    """Isothermal runs of compare ([[iso]] ...) or chosen isothermal experiments: compare X and FWHM against time (from the hold start) instead of temperature."""
    import webbrowser
    from dataclasses import replace

    if isinstance(group, str):
        st, runs, opt = load_iso(config, group)
    else:  # chosen experiments (runs mixed): name = run.name
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


def run(
    config: Path = DEFAULT_CONFIG,
    geo: Geometry = Geometry(),
    workers: int | None = None,
    only: list[str] | None = None,
    show: bool = True,
    section: str | list[str] | None = None,
) -> list[IsoResult]:
    """section: one run or several; if omitted, every run starting with iso ([[iso]], [[iso2]] ...),
    so measurements at different temperatures in different runs still go into one Arrhenius fit."""
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

    # apparent crystallite size (Scherrer K=0.9, no instrument-width correction) [nm]
    for r in results:
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
