"""Compare the crystallized fraction X (normalized area) of each peak against temperature for several
experiments, with T50.

Write the image folders and temperature logs (recipe) of each experiment in the settings file
(TOML, experiments.toml by default), then
    uv run python main.py compare [experiments.toml]
gives one panel per peak: X vs PV temperature of each experiment, with 'x1  T50 401 °C' in the legend.

Division by the amorphous: R = I / I_amorphous - 1, where I_amorphous is the mean of the first
n_norm frames of the used range. The intensity falloff at the detector edge (multiplicative) and
the amorphous background cancel, leaving only the crystalline peaks.

Integration window (automatic per experiment and peak):
     The 2θ in the settings is only a hint of where to look. In the mean R of the last n_norm frames,
     after crystallization, the peak within the hint ± search is fitted with a Gaussian (plus a
     straight background), giving the center c and FWHM.
     Window = c +- fwhm_k * FWHM (default 1.5: about 99.9 % of the Gaussian area).

X  = (A - A0) / (A1 - A0)
     A  = area of R in that window (straight background through the window ends removed)
     A0 = median of the first n_norm frames of the used range (before the transition),
     A1 = median of the last n_norm frames (after the transition)
T50 = temperature where X first crosses level (default 0.5) upward (linear interpolation between
      the frames around it)

Image folders and temperature logs are used exactly as written in the settings file (they are
not searched for). Integration results are cached in out/cache/ and not recomputed while the
image folder is unchanged.
"""

from __future__ import annotations

import re
import webbrowser
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.ndimage import median_filter

from . import segments as sg
from .config import SETTINGS_FILE, load_toml, out_dir, project_path, saving
from .geometry import Geometry
from .heatmap import Config, beam_ok, integrate_folder, regrid, time_mean
from .integrate import ProfileOptions
from .peaks import fit_peak
from .style import EXP_COLORS, ORIGIN_RC, origin_plotly, place_legend, write_fit_html
from .temperature import PV, TempLog, fmt_kst, parse_time
from .tracks import level_temperature, normalize_step, transition_point, window_area

DEFAULT_CONFIG = SETTINGS_FILE
BASELINE_TOL = (
    # if the first used frame is later than the start of heating by more than this [°C], there is no pre-transition reference
    30.0
)


@dataclass(slots=True)
class Experiment:
    name: str
    images: list[Path]
    recipe: list[Path]
    start: str | None = None
    end: str | None = None
    color: str | None = None
    group: str = ""  # which [[...]] it came from (overview uses isothermal mode for iso)


@dataclass(slots=True)
class Settings:
    peaks: dict[str, float]  # peak name -> hint 2θ (the real peak is searched near it)
    exps: list[Experiment]
    segment: str = "heat"  # "heat" = heating segment only, "all" = everything
    level: float = 0.5
    n_norm: int = 10
    t_range: tuple[float, float] | None = None
    tth_bin: float = 0.02
    search: float = 0.5  # search for the peak within hint 2θ ± search [deg]
    fwhm_k: float = 1.5  # integration window = center ± fwhm_k x FWHM
    path: Path = DEFAULT_CONFIG
    group: str = "exp"  # [[exp]] / [[exp2]] / ... (experiment run)


def _as_list(v) -> list[str]:
    return [v] if isinstance(v, str) else list(v)


def exp_groups(raw: dict) -> list[str]:
    """Experiment runs in the settings file: every [[...]] that has images.
    [[exp]], [[exp2]], [[exp3]] ... come first in number order, then the others such as [[iso]] in file order."""
    keys = [
        k
        for k, v in raw.items()
        if isinstance(v, list) and v and all(isinstance(e, dict) and "images" in e for e in v)
    ]
    num = sorted((k for k in keys if re.fullmatch(r"exp\d*", k)), key=lambda k: int(k[3:] or 1))
    return num + [k for k in keys if k not in num]


def entry_name(e: dict, i: int = 1) -> str:
    """Name of a [[...]] entry: name, else (isothermal) sample_T_iso, sample, run{i}."""
    if e.get("name"):
        return str(e["name"])
    sample = str(e.get("sample") or f"run{i}")
    return f"{sample}_{e['T_iso']:g}" if "T_iso" in e else sample


def load_settings(
    path: Path = DEFAULT_CONFIG, need_exps: bool = True, group: str = "exp"
) -> Settings:
    """Read the settings file (experiments from [[group]]). Paths as written, relative ones against the project folder."""
    path = Path(path)
    raw = load_toml(path)
    o = raw.get("options", {})
    exps = [
        Experiment(
            name=entry_name(e, i),
            images=[project_path(s) for s in _as_list(e["images"])],
            recipe=[project_path(s) for s in _as_list(e.get("recipe", []))],
            start=e.get("start"),
            end=e.get("end"),
            color=e.get("color"),
            group=group,
        )
        for i, e in enumerate(raw.get(group, []), start=1)
    ]
    if not exps and need_exps:
        raise SystemExit(
            f"❌ {path}: [[{group}]] 항목이 없음 (있는 회차: {', '.join(exp_groups(raw)) or '없음'})"
        )
    # a single number is the hint 2θ; the older [start, end] form uses its midpoint
    peaks = {
        k: float(v) if isinstance(v, (int, float)) else (float(v[0]) + float(v[1])) / 2
        for k, v in raw.get("peaks", {}).items()
    }
    if not peaks:
        raise SystemExit(f"❌ {path}: [peaks] 항목이 없음")
    tr = o.get("t_range")
    return Settings(
        peaks=peaks,
        exps=exps,
        segment=o.get("segment", "heat"),
        level=float(o.get("level", 0.5)),
        n_norm=int(o.get("n_norm", 10)),
        t_range=(float(tr[0]), float(tr[1])) if tr else None,
        tth_bin=float(o.get("tth_bin", 0.02)),
        search=float(o.get("search", 0.5)),
        fwhm_k=float(o.get("fwhm_k", 1.5)),
        path=path,
        group=group,
    )


def find_exp(name: str, path: Path = DEFAULT_CONFIG) -> Experiment | None:
    """Experiment name from the settings file to an Experiment (for use like overview x1).
    With a run prefix such as 'exp2.x8' it looks in that run, otherwise from the first run on.
    If no name matches exactly it also looks by sample (iso.x1 finds x1_380 whose sample is "x1" in [[iso]])."""
    if not Path(path).exists():
        return None
    raw = load_toml(path)
    group, _, short = name.partition(".") if name.split(".")[0] in raw else ("", "", name)
    groups = [group] if group else exp_groups(raw)
    every = [
        (g, e, str(r.get("sample", "")))
        for g in groups
        for e, r in zip(load_settings(path, False, g).exps, raw.get(g, []))
    ]
    hits = [(g, e) for g, e, _ in every if e.name == short] or [
        (g, e) for g, e, s in every if s == short
    ]
    if not hits:
        return None
    if len(hits) > 1:
        print(
            f"ℹ️  '{name}' 에 맞는 게 여럿: {', '.join(f'{g}.{e.name}' for g, e in hits)} → "
            f"{hits[0][0]}.{hits[0][1].name} 사용 (다른 것은 그 이름으로)"
        )
    return hits[0][1]


def exp_names(path: Path = DEFAULT_CONFIG) -> str:
    """For error messages: every experiment name in the settings file ('exp.x1, exp2.x1, ..., iso.x1_380')."""
    if not Path(path).exists():
        return ""
    raw = load_toml(path)
    return ", ".join(
        f"{g}.{e.name}" for g in exp_groups(raw) for e in load_settings(path, False, g).exps
    )


def check_paths(exps: list[Experiment]) -> None:
    """Report all missing paths at once, before any computation starts."""
    missing = [f"   {e.name}: {p}" for e in exps for p in (*e.images, *e.recipe) if not p.exists()]
    if missing:
        raise SystemExit(
            "❌ 경로 없음 (experiments.toml 의 images / recipe 확인):\n" + "\n".join(missing)
        )


@dataclass(slots=True)
class ExpResult:
    name: str
    table: pd.DataFrame  # per frame: file, time_kst, T_pv, kind, A_<peak>, X_<peak>
    t50: dict[str, float]
    note: str = ""
    color: str = "#000000"
    ok: bool = True  # False = no pre-transition data, so X cannot be normalized
    windows: dict[str, tuple[float, float, float, float]] | None = (
        None  # peak -> (center, FWHM, window start, end)
    )
    trans: dict[str, dict] | None = None  # peak -> transition_point result (T50 error, width)


def _clip_window(
    x: np.ndarray, ok: np.ndarray, c: float, lo: float, hi: float
) -> tuple[float, float]:
    """Shrink [lo, hi] to the stretch of ok points connected to the center c (gaps (NaN) are crossed, falloff at the edge is not)."""
    i = int(np.clip(np.searchsorted(x, c), 0, len(x) - 1))
    a = i
    while a > 0 and ok[a - 1] and x[a - 1] >= lo:
        a -= 1
    b = i
    while b < len(x) - 1 and ok[b + 1] and x[b + 1] <= hi:
        b += 1
    return max(lo, float(x[a])), min(hi, float(x[b]))


def peak_window(
    x: np.ndarray, prof: np.ndarray, guess: float, search: float, k: float
) -> tuple[float, float, float]:
    """(center, FWHM, integration half-width = k*FWHM) of the peak near the hint 2θ.
    Gaussian plus straight background fit; if it fails, the half-maximum points are found directly."""
    fit = fit_peak(x, prof, guess, half_window=search)
    dx = x[1] - x[0]
    if (
        np.isfinite(fit.fwhm)
        and abs(fit.center - guess) <= search
        and 2 * dx < fit.fwhm < 2 * search
    ):
        return fit.center, fit.fwhm, k * fit.fwhm
    # fallback: subtract the straight background through the range ends and take the left and right half-maximum points
    m = (np.abs(x - guess) <= search) & np.isfinite(prof)
    xs, ys = x[m], prof[m]
    if len(xs) < 6:
        return guess, np.nan, search
    ys = ys - np.interp(xs, [xs[0], xs[-1]], [ys[:3].mean(), ys[-3:].mean()])
    i = int(np.argmax(ys))
    half = ys[i] / 2
    left = np.flatnonzero(ys[:i] < half)
    right = np.flatnonzero(ys[i:] < half)
    if not len(left) or not len(right) or ys[i] <= 0:
        return guess, np.nan, search
    l0, r0 = left[-1], i + right[0]
    xl = np.interp(half, [ys[l0], ys[l0 + 1]], [xs[l0], xs[l0 + 1]])
    xr = np.interp(half, [ys[r0], ys[r0 - 1]], [xs[r0], xs[r0 - 1]])
    fwhm = xr - xl
    return float(xs[i]), float(fwhm), float(k * fwhm)


def frame_widths(
    x: np.ndarray,
    R: np.ndarray,
    c: float,
    fwhm: float,
    A: np.ndarray,
    a_final: float,
    min_frac: float = 0.2,
) -> tuple[np.ndarray, np.ndarray]:
    """Peak (center, FWHM) per frame from a Gaussian plus straight background fit. Only frames where the
    peak really exists (area >= min_frac of the final area); the rest are NaN. Fit range is c ± max(1.5*FWHM, 0.4°)."""
    n = len(R)
    center, width = np.full(n, np.nan), np.full(n, np.nan)
    if not (np.isfinite(c) and np.isfinite(a_final) and a_final > 0):
        return center, width
    hw = max(1.5 * fwhm, 0.4) if np.isfinite(fwhm) else 0.4
    dx = x[1] - x[0]
    for i in np.flatnonzero(np.nan_to_num(A, nan=-np.inf) >= min_frac * a_final):
        f = fit_peak(x, R[i], c, half_window=hw)
        if np.isfinite(f.fwhm) and 2 * dx < f.fwhm < 2 * hw and abs(f.center - c) < hw / 2:
            center[i], width[i] = f.center, f.fwhm
    return center, width


def peak_areas(
    x: np.ndarray,
    Z: np.ndarray,
    base: np.ndarray,
    final: np.ndarray,
    st: Settings,
    say=print,
    widths: bool = False,
):
    """Per peak (center, FWHM, window start, end), area per frame and, with widths=True, (center, FWHM)
    per frame. Returns (windows, areas, shapes).
    Divided by the mean of the base frames (before the transition, amorphous): R = I / I_amorphous - 1,
    so the intensity falloff at the detector edge (multiplicative) and the amorphous background cancel
    and only the crystalline peaks remain as clean peaks.
    The window is set from the mean R of the final frames (after crystallization), by finding the
    peak near the [peaks] hint."""
    pre = time_mean(Z[base])
    cover = np.isfinite(pre) & (pre > 0.3 * np.nanmedian(pre))  # skip 2θ the detector barely covers
    with np.errstate(all="ignore"):
        R = np.where(cover, Z / pre - 1.0, np.nan)
    ref = time_mean(R[final])
    dx = x[1] - x[0]
    windows, areas, shapes = {}, {}, {}
    for pk, guess in st.peaks.items():
        c, fwhm, hw = peak_window(x, ref, guess, st.search, st.fwhm_k)
        lo, hi = _clip_window(x, cover | np.isnan(pre), c, c - hw, c + hw)
        windows[pk] = (c, fwhm, lo, hi)
        cut = " ⚠️ 디텍터 범위 끝에서 잘림" if (lo > c - hw + dx or hi < c + hw - dx) else ""
        say(
            f"   {pk:>6}: 중심 {c:.3f}°  FWHM {fwhm:.3f}°  →  적분 {lo:.2f}–{hi:.2f}° "
            f"(±{st.fwhm_k:g}·FWHM){cut}"
            if np.isfinite(fwhm)
            else f"   {pk:>6}: ⚠️ {guess}° ±{st.search}° 에서 피크를 못 찾음 → {lo:.2f}–{hi:.2f}° 적분"
        )
        areas[pk] = np.array([window_area(x, r, (lo + hi) / 2, (hi - lo) / 2) for r in R])
        if widths:
            shapes[pk] = frame_widths(
                x, R, c, fwhm, areas[pk], float(np.nanmedian(areas[pk][final]))
            )
    return windows, areas, shapes


@dataclass(slots=True)
class Analysis:
    use: np.ndarray  # per frame: used in the computation (heating, temperature range, beam on)
    kind: np.ndarray  # segment kind per frame: heat/hold/cool
    ok: bool  # False = no pre-transition data, cannot normalize
    msg: str
    t50: dict[str, float]
    # peak -> (center, FWHM, window start, end)
    windows: dict[str, tuple[float, float, float, float]]
    A: dict[str, np.ndarray]
    X: dict[str, np.ndarray]
    center: dict[str, np.ndarray]  # peak center per frame [deg] (NaN where there is no peak)
    fwhm: dict[str, np.ndarray]  # FWHM per frame [deg]
    trans: dict[str, dict]  # peak -> transition_point result (T50, err, width, window ...)


def analyze(
    x: np.ndarray,
    Z: np.ndarray,
    t: np.ndarray,
    T: np.ndarray,
    log: TempLog,
    st: Settings,
    verbose: bool = True,
) -> Analysis:
    """Frame profiles Z (in time order) plus frame temperature T (PV) to X and T50 per peak; shared by compare and overview."""
    say = print if verbose else (lambda *a, **k: None)
    has_T = np.isfinite(T)
    kind = np.full(len(t), "", dtype=object)
    heat_T0 = np.nan
    if has_T.any():
        tt = t[has_T]
        # detect over the whole log; cutting to the frame range would misjudge the start temperature of heating
        det = sg.detect(log, min(tt[0], log.t[0]), max(tt[-1], log.t[-1]))
        kind = np.where(has_T, sg.kinds_at(t, det.segments), "")
        heats = [s for s in det.segments if s.kind == "heat"]
        if heats:
            heat_T0 = max(heats, key=lambda s: s.duration).T0

    use = has_T.copy()
    if st.segment == "heat":
        use &= kind == "heat"
    if st.t_range:
        use &= (T >= st.t_range[0]) & (T <= st.t_range[1])
    if use.sum() < 2 * st.n_norm:
        say(f"   ⚠️  사용할 프레임이 {use.sum()} 개뿐 (segment={st.segment}, t_range={st.t_range})")
    # exclude beam-off frames (relative to the median intensity of the used frames)
    use &= beam_ok(Z, use)

    ok, msg = True, ""
    if use.any():
        T_first = float(np.nanmin(T[use]))
        T_need = max(
            st.t_range[0] if st.t_range else -np.inf, heat_T0 if np.isfinite(heat_T0) else -np.inf
        )
        if np.isfinite(T_need) and T_first - T_need > BASELINE_TOL:
            ok = False
            msg = (
                f"이미지가 {T_first:.0f} °C 부터만 있음 (승온은 {T_need:.0f} °C 부터) → "
                "전이 전 기준이 없어 X·T50 계산 안 함"
            )
            say(f"   ❌ {msg}")
    else:
        ok, msg = False, "사용할 프레임 없음"

    t50, Xs = {}, {}
    iu_all = np.flatnonzero(use)
    if not len(iu_all):
        iu_all = np.flatnonzero(has_T)
    windows, As, shapes = peak_areas(
        x, Z, iu_all[: st.n_norm], iu_all[-st.n_norm :], st, say, widths=True
    )
    # reported FWHM = per-frame fit values (of the last frames that set the window)
    for pk, (
        c,
        _,
        lo,
        hi,
    ) in windows.items():
        tail = shapes[pk][1][iu_all[-st.n_norm :]]
        if np.isfinite(tail).any():
            windows[pk] = (c, float(np.nanmedian(tail)), lo, hi)
    trans = {}
    for pk, A in As.items():
        X = np.full(len(A), np.nan)
        iu = np.flatnonzero(use & np.isfinite(A))
        n = min(st.n_norm, max(1, len(iu) // 4))
        # the transition point uses only a window proportional to the transition width (tracks.transition_point),
        # which makes it insensitive to range ends, distant outliers and drift
        tp = (
            transition_point(T[iu], A[iu], st.level, n_end=n) if len(iu) >= 4 * n else {"ok": False}
        )
        # normalized by the before/after background lines of the transition window, so X is at level at T50
        if tp["ok"]:
            X[iu] = normalize_step(T[iu], A[iu], tp)
        # no window (too little data around the transition): fall back to the medians of both ends
        elif len(iu) >= 2:
            a0, a1 = np.median(A[iu[:n]]), np.median(A[iu[-n:]])
            X[iu] = (A[iu] - a0) / ((a1 - a0) or np.nan)
        Xs[pk], trans[pk] = X, tp
        if not ok:
            t50[pk] = np.nan
            continue
        if tp["ok"]:
            t50[pk] = tp["T50"]
            say(
                f"   {pk:>6}: T50 = {tp['T50']:.1f} ± {tp['err']:.1f} °C  (폭 ΔT₂₅₋₇₅ {tp['width']:.1f} °C, "
                f"창 {tp['window'][0]:.0f}–{tp['window'][1]:.0f} °C, 코어 {tp['n_core']} 프레임)"
            )
        else:
            t50[pk] = level_temperature(T[use], X[use], st.level)
            say(
                f"   {pk:>6}: T50 = {t50[pk]:.1f} °C  (⚠️ 전이 전후 데이터가 부족해 창을 못 잡음 → 양 끝 정규화)"
                if np.isfinite(t50[pk])
                else f"   {pk:>6}: T50 없음"
            )
    return Analysis(
        use,
        kind,
        ok,
        msg,
        t50,
        windows,
        As,
        Xs,
        {pk: s[0] for pk, s in shapes.items()},
        {pk: s[1] for pk, s in shapes.items()},
        trans,
    )


@dataclass(slots=True)
class ExpData:
    x: np.ndarray  # 2θ
    Z: np.ndarray  # intensity [frame, 2θ], in time order
    t: np.ndarray  # acquisition time Unix [frame]
    files: np.ndarray
    T: np.ndarray  # PV temperature per frame (NaN outside the log)
    log: TempLog | None  # None if there is no temperature log
    note: str = ""


def load_exp(exp: Experiment, st: Settings, geo: Geometry, workers: int | None) -> ExpData:
    """Integrate (cached) the image folder(s) of an experiment, merge in time order, cut by start/end and match temperatures."""
    print(
        f"\n🧪 {exp.name}: {', '.join(str(p) for p in exp.images)}  |  "
        f"{', '.join(p.name for p in exp.recipe) or '온도 로그 없음'}"
    )
    parts = [
        integrate_folder(
            Config(
                folder=f, geometry=geo, workers=workers, profile=ProfileOptions(tth_bin=st.tth_bin)
            )
        )
        for f in exp.images
    ]
    x = parts[0][0]
    Z = np.vstack([regrid(x, p[0], p[1]) for p in parts])
    t = np.concatenate([p[2] for p in parts])
    files = np.concatenate([p[3] for p in parts])
    order = np.argsort(t, kind="stable")
    Z, t, files = Z[order], t[order], files[order]

    keep = np.ones(len(t), bool)
    day = fmt_kst(t[0])[:10]
    if exp.start:
        keep &= t >= parse_time(exp.start, day)
    if exp.end:
        keep &= t <= parse_time(exp.end, day)
    Z, t, files = Z[keep], t[keep], files[keep]

    if not exp.recipe:  # no temperature log: temperature is NaN (avrami still computes n and k)
        return ExpData(x, Z, t, files, np.full(len(t), np.nan), None, "온도 로그 없음")
    log = TempLog.load(exp.recipe)
    T = log.at(t, PV)
    n_T = int(np.isfinite(T).sum())
    note = ""
    if n_T < len(t):
        note = f"온도 있는 프레임 {n_T}/{len(t)} (로그 {log.span[0][11:]}~{log.span[1][11:]} 밖은 제외)"
        print(f"   ⚠️  {note}")
    return ExpData(x, Z, t, files, T, log, note)


def run_exp(
    exp: Experiment, st: Settings, geo: Geometry, workers: int | None, color: str = "#000000"
) -> ExpResult:
    if not exp.recipe:
        raise SystemExit(f"❌ {exp.name}: recipe(온도 로그) 가 필요함")
    d = load_exp(exp, st, geo, workers)
    x, Z, t, files, T, log, note = d.x, d.Z, d.t, d.files, d.T, d.log, d.note
    an = analyze(x, Z, t, T, log, st)
    if an.msg:
        note = (note + " / " if note else "") + an.msg
    tab = pd.DataFrame(
        {
            "file": files,
            "time_kst": [fmt_kst(v) for v in t],
            "T_pv": T,
            "kind": an.kind,
            "used": an.use,
        }
    )
    for pk in st.peaks:
        tab[f"A_{pk}"], tab[f"X_{pk}"] = an.A[pk], an.X[pk]
        tab[f"center_{pk}"], tab[f"FWHM_{pk}"] = an.center[pk], an.fwhm[pk]
    return ExpResult(
        exp.name, tab, an.t50, note, color=color, ok=an.ok, windows=an.windows, trans=an.trans
    )


def _lab(r: ExpResult, pk: str) -> str:
    t50 = r.t50[pk]
    if not np.isfinite(t50):
        return f"{r.name}  –"
    tp = (r.trans or {}).get(pk, {})
    width = f" (Δ{tp['width']:.0f})" if tp.get("ok") else ""
    return f"{r.name}  {t50:.0f} °C{width}"


NO_BASELINE = "no pre-transition images"


def smooth(y: np.ndarray, size: int = 7) -> np.ndarray:
    """Running median skipping NaN, a trend line for values that jitter per frame such as FWHM."""
    out = np.full(len(y), np.nan)
    ok = np.flatnonzero(np.isfinite(y))
    if len(ok) >= size:
        out[ok] = median_filter(y[ok], size=size, mode="nearest")
    return out


def _fwhm_rows(r: ExpResult, pk: str) -> pd.DataFrame:
    """Frames to draw in FWHM vs T: those with a peak within the used range (heating, temperature range)."""
    d = r.table[r.table["used"]]
    return d[np.isfinite(d[f"FWHM_{pk}"])]


# heights for the T50 numbers (margin above the X axis; close ones go to the next row)
LABEL_ROWS = (1.13, 1.22, 1.31)


def _t50_marks(results: list[ExpResult], pk: str, span: float) -> list[tuple]:
    """Transition points to draw, [(result, T50, T25, T75, label height)], in T50 order with rows assigned so labels do not overlap."""
    marks, last = [], {}
    for r in sorted(results, key=lambda r: r.t50.get(pk, np.inf)):
        t50 = r.t50.get(pk, np.nan)
        if not (r.ok and np.isfinite(t50)):
            continue
        tp = (r.trans or {}).get(pk, {})
        t25, t75 = (tp["t25"], tp["t75"]) if tp.get("ok") else (np.nan, np.nan)
        row = next(
            (k for k in range(len(LABEL_ROWS)) if t50 - last.get(k, -np.inf) > 0.09 * span),
            len(marks) % len(LABEL_ROWS),
        )
        last[row] = t50
        marks.append((r, t50, t25, t75, LABEL_ROWS[row]))
    return marks


def plot(results: list[ExpResult], st: Settings) -> plt.Figure:
    peaks = list(st.peaks)
    with plt.rc_context(ORIGIN_RC):
        fig, axes = plt.subplots(
            2,
            len(peaks),
            figsize=(6.6 * len(peaks), 9.6),
            squeeze=False,
            sharex="col",
            gridspec_kw={"height_ratios": [1, 0.8], "hspace": 0.08},
        )
        legend_axes = []
        for j, pk in enumerate(peaks):
            ax, ax_w = axes[0][j], axes[1][j]
            for r in results:
                if not r.ok:
                    ax.plot([], [], "o", color="0.65", ms=4, label=f"{r.name}  {NO_BASELINE}")
                    continue
                d = r.table[r.table["used"]]
                ax.plot(
                    d["T_pv"], d[f"X_{pk}"], "-o", color=r.color, ms=3, lw=1.3, label=_lab(r, pk)
                )
            for r in results:  # FWHM can be drawn even without a pre-transition reference
                d = _fwhm_rows(r, pk)
                if len(d):
                    ax_w.plot(
                        d["T_pv"], d[f"FWHM_{pk}"], "o", color=r.color, ms=2.5, alpha=0.35, mew=0
                    )
                    ax_w.plot(d["T_pv"], smooth(d[f"FWHM_{pk}"].to_numpy()), color=r.color, lw=1.6)
            ax.axhline(st.level, color="0.6", lw=0.9, ls=(0, (5, 3)), zorder=0)
            if st.t_range:
                ax.set_xlim(*st.t_range)
            lo, hi = ax.get_xlim()
            # transition point: shaded width (25-75 %), dashed vertical line down to the FWHM panel, diamond, temperature above
            for r, t50, t25, t75, ly in _t50_marks(results, pk, hi - lo):
                for a in (ax, ax_w):
                    if np.isfinite(t25):
                        a.axvspan(t25, t75, color=r.color, alpha=0.12, lw=0, zorder=0)
                    a.axvline(t50, color=r.color, ls=(0, (4, 2)), lw=1.3, zorder=1)
                ax.plot(t50, st.level, "D", ms=10, mfc=r.color, mec="black", mew=1.2, zorder=6)
                ax.text(
                    t50,
                    ly,
                    f"{t50:.0f}",
                    ha="center",
                    va="center",
                    fontsize=13,
                    fontweight="bold",
                    color=r.color,
                    zorder=7,
                    bbox=dict(boxstyle="round,pad=0.2", fc="white", ec=r.color, lw=1.2),
                )
            ax.set_ylim(-0.1, 1.38)
            ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
            ax.set_title(pk, fontsize=17.5)
            legend_axes.append(ax)
            ax.tick_params(labelbottom=False)
            ax_w.set_xlabel("PV (°C)")
        axes[0][0].set_ylabel("crystallised fraction X\n(normalised area)")
        axes[1][0].set_ylabel("FWHM (°)\n(per-frame Gaussian fit)")
        fig.suptitle(f"[{st.group}]", x=0.01, ha="left", fontsize=16, fontweight="bold")
        fig.tight_layout()
        # legend in a free spot of the graph, not covering curves, diamonds or T50 numbers
        for a in legend_axes:
            place_legend(
                a,
                fontsize=12.5,
                frameon=True,
                framealpha=0.9,
                edgecolor="0.6",
                handlelength=1.4,
                title="T50  (ΔT$_{25–75}$)",
                title_fontsize=12.5,
            )
    return fig


def write_html(results: list[ExpResult], st: Settings, path: Path) -> Path:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    peaks = list(st.peaks)
    fig = make_subplots(
        rows=2,
        cols=len(peaks),
        subplot_titles=peaks,
        shared_xaxes=True,
        horizontal_spacing=0.06,
        vertical_spacing=0.05,
        row_heights=[0.56, 0.44],
    )
    for j, pk in enumerate(peaks, start=1):
        lg = "legend" if j == 1 else f"legend{j}"
        for r in results:
            d = _fwhm_rows(r, pk)
            if not len(d):
                continue
            fig.add_trace(
                go.Scatter(
                    x=d["T_pv"],
                    y=d[f"FWHM_{pk}"],
                    mode="markers",
                    legendgroup=r.name,
                    showlegend=False,
                    marker=dict(size=4, color=r.color, opacity=0.35),
                    hovertemplate=f"<b>{r.name} {pk}</b><br>T = %{{x:.1f}} °C<br>FWHM = %{{y:.3f}}°<extra></extra>",
                ),
                2,
                j,
            )
            fig.add_trace(
                go.Scatter(
                    x=d["T_pv"],
                    y=smooth(d[f"FWHM_{pk}"].to_numpy()),
                    mode="lines",
                    legendgroup=r.name,
                    showlegend=False,
                    hoverinfo="skip",
                    line=dict(color=r.color, width=2),
                ),
                2,
                j,
            )
        for r in results:
            if not r.ok:
                fig.add_trace(
                    go.Scatter(
                        x=[None],
                        y=[None],
                        mode="markers",
                        legend=lg,
                        name=f"{r.name}  {NO_BASELINE}",
                        marker=dict(color="#aaa", size=6),
                    ),
                    1,
                    j,
                )
                continue
            d = r.table[r.table["used"]]
            fig.add_trace(
                go.Scatter(
                    x=d["T_pv"],
                    y=d[f"X_{pk}"],
                    mode="lines+markers",
                    name=_lab(r, pk),
                    legendgroup=r.name,
                    legend=lg,
                    marker=dict(size=4, color=r.color),
                    line=dict(color=r.color, width=1.5),
                    customdata=np.stack([d["time_kst"].str[11:19], d["file"]], axis=1),
                    hovertemplate=(
                        f"<b>{r.name} {pk}</b><br>T = %{{x:.1f}} °C<br>X = %{{y:.3f}}"
                        "<br>%{customdata[0]} · %{customdata[1]}<extra></extra>"
                    ),
                ),
                1,
                j,
            )
        fig.add_hline(y=st.level, line=dict(color="#999", dash="dash", width=1), row=1, col=j)
        # transition point: shaded width (25-75 %), dashed line, diamond (hover shows T50 ± error and width), temperature above
        span = (st.t_range[1] - st.t_range[0]) if st.t_range else 300.0
        for r, t50, t25, t75, ly in _t50_marks(results, pk, span):
            tp = (r.trans or {}).get(pk, {})
            info = f"±{tp['err']:.1f} °C · ΔT25–75 {tp['width']:.1f} °C" if tp.get("ok") else ""
            for row in (1, 2):
                if np.isfinite(t25):
                    fig.add_vrect(
                        x0=t25,
                        x1=t75,
                        fillcolor=r.color,
                        opacity=0.12,
                        line_width=0,
                        layer="below",
                        row=row,
                        col=j,
                    )
                fig.add_vline(
                    x=t50, line=dict(color=r.color, dash="dash", width=1.4), row=row, col=j
                )
            fig.add_trace(
                go.Scatter(
                    x=[t50],
                    y=[st.level],
                    mode="markers",
                    legendgroup=r.name,
                    showlegend=False,
                    marker=dict(
                        symbol="diamond",
                        size=13,
                        color=r.color,
                        line=dict(color="black", width=1.2),
                    ),
                    hovertemplate=f"<b>{r.name} {pk}  T50 {t50:.1f} °C</b><br>{info}<extra></extra>",
                ),
                1,
                j,
            )
            fig.add_annotation(
                x=t50,
                y=ly,
                text=f"<b>{t50:.0f}</b>",
                showarrow=False,
                row=1,
                col=j,
                font=dict(color=r.color, size=16),
                bgcolor="white",
                bordercolor=r.color,
                borderwidth=1.2,
                borderpad=2,
            )
    y_top = fig.layout.yaxis.domain[0]
    leg = {
        ("legend" if j == 1 else f"legend{j}"): dict(
            x=fig.layout[f"xaxis{'' if j == 1 else j}"].domain[1] - 0.005,
            xanchor="right",
            y=y_top + 0.02,
            yanchor="bottom",
            bgcolor="rgba(255,255,255,0.8)",
            font=dict(size=15),
        )
        for j in range(1, len(peaks) + 1)
    }
    fig.update_layout(
        template="simple_white",
        font=dict(family="Arial", size=17.5),
        title=dict(text=f"<b>[{st.group}]</b>", x=0.01, y=0.99),
        margin=dict(t=70, l=90, r=30, b=60),
        **leg,
    )
    origin_plotly(fig)
    if st.t_range:
        fig.update_xaxes(range=list(st.t_range))
    fig.update_xaxes(title="PV (°C)", row=2)
    fig.update_yaxes(range=[-0.1, 1.38], tickvals=[0, 0.2, 0.4, 0.6, 0.8, 1.0], row=1)
    fig.update_yaxes(matches="y", row=1)
    fig.update_yaxes(title="X (normalised area)", row=1, col=1)
    fig.update_yaxes(title="FWHM (°)", row=2, col=1)
    # fixed size that does not stretch in a large window (~540 px per peak), so screenshots keep the text ratio
    return write_fit_html(fig, path, 150 + 540 * len(peaks), 800)


def run(
    config: Path = DEFAULT_CONFIG,
    geo: Geometry = Geometry(),
    workers: int | None = None,
    show: bool = True,
    groups: list[str] | None = None,
) -> dict[str, list[ExpResult]]:
    """Compare each experiment run ([[exp]], [[exp2]], ..., [[iso]]) in turn; one result file and browser window per run."""
    raw = load_toml(config)
    groups = groups or exp_groups(raw)
    # anything that is not a run is an experiment name (iso3.x1, x8_300 ...)
    picks = [g for g in groups if g not in raw]
    if picks:
        return {"pick": _run_picks(picks, config, geo, workers, show, raw)}
    unknown = [g for g in groups if g not in raw]
    if not groups or unknown:
        raise SystemExit(
            f"❌ {config}: 회차 {', '.join(unknown) or '[[exp]]'} 없음 "
            f"(있는 회차: {', '.join(exp_groups(raw)) or '없음'})"
        )
    # the same name (e.g. x8) gets the same color in every run
    names = list(
        dict.fromkeys(
            entry_name(e, i) for g in exp_groups(raw) for i, e in enumerate(raw[g], start=1)
        )
    )
    colors = {n: EXP_COLORS[i % len(EXP_COLORS)] for i, n in enumerate(names)}
    for g in groups:  # check the paths of all runs together before computing
        check_paths(load_settings(config, group=g).exps)
    out = {}
    for g in groups:
        st = load_settings(config, group=g)
        # isothermal run: compare against time (from the hold start) instead of temperature
        if g.startswith("iso"):
            from .avrami import compare_iso

            print(
                f"\n{'═' * 70}\n📋 [[{g}]]  {st.path}  —  등온 측정 {len(st.exps)}개 (X · FWHM vs 시간)"
            )
            out[g] = compare_iso(config, g, geo, workers, show, colors)
            continue
        print(
            f"\n{'═' * 70}\n📋 [[{g}]]  {st.path}  —  실험 {len(st.exps)}개, "
            f"피크 {', '.join(f'{k} ~{v}°' for k, v in st.peaks.items())}"
            f"  (± {st.search}° 에서 찾아 ±{st.fwhm_k:g}·FWHM 적분)"
        )
        out[g] = _run_group(st, geo, workers, show, colors)
    return out


def _run_picks(
    names: list[str], config: Path, geo: Geometry, workers: int | None, show: bool, raw: dict
):
    """Several experiment names (runs may be mixed) on one page: time comparison if all are isothermal runs, else temperature comparison."""
    from dataclasses import replace

    picked = []
    for nm in names:
        e = find_exp(nm, config)
        if e is None:
            raise SystemExit(
                f"❌ 없음: {nm}  (회차 이름 또는 실험 이름)\n   있는 회차: {', '.join(exp_groups(raw))}"
                f"\n   있는 이름: {exp_names(config)}"
            )
        picked.append(e)
    check_paths(picked)
    names_all = list(dict.fromkeys(f"{e.group}.{e.name}" for e in picked))
    colors = {f"{e.group}.{e.name}": EXP_COLORS[i % len(EXP_COLORS)] for i, e in enumerate(picked)}
    label = " + ".join(names_all)
    print(f"\n{'═' * 70}\n📋 {label}")
    if all(e.group.startswith("iso") for e in picked):
        from .avrami import compare_iso

        return compare_iso(config, picked, geo, workers, show, colors)
    st = replace(
        load_settings(config, need_exps=False),
        exps=[replace(e, name=f"{e.group}.{e.name}") for e in picked],
        group="pick",
    )
    return _run_group(st, geo, workers, show, colors)


def _run_group(
    st: Settings, geo: Geometry, workers: int | None, show: bool, colors: dict[str, str]
) -> list[ExpResult]:
    results = [
        run_exp(e, st, geo, workers, e.color or colors.get(e.name, "#000000")) for e in st.exps
    ]
    d = out_dir("compare")
    stem = (
        st.path.stem
        if st.group == "exp"
        else "+".join(e.name.replace(".", "_") for e in st.exps)
        if st.group == "pick"
        else f"{st.path.stem}_{st.group}"
    )

    def _row(r: ExpResult) -> dict:
        row = {"exp": r.name}
        for k, v in r.t50.items():
            tp = (r.trans or {}).get(k, {})
            row[f"T50 {k}"] = round(v, 1)
            row[f"±{k}"] = round(tp["err"], 2) if tp.get("ok") else np.nan
            row[f"ΔT25-75 {k}"] = round(tp["width"], 1) if tp.get("ok") else np.nan
        for k, w in (r.windows or {}).items():
            row[f"{k} center"], row[f"{k} FWHM"] = round(w[0], 3), round(w[1], 3)
        return {**row, "note": r.note}

    summary = pd.DataFrame([_row(r) for r in results])
    print(
        f"\n📊 [[{st.group}]] T50 (X = {st.level:g} 을 넘는 온도)\n"
        + summary.to_string(index=False)
    )
    page = write_html(results, st, d / f"{stem}.html")
    if saving():
        summary.to_csv(d / f"{stem}_T50.csv", index=False, encoding="utf-8-sig")
        pd.concat([r.table.assign(exp=r.name) for r in results]).to_csv(
            d / f"{stem}_frames.csv", index=False, encoding="utf-8-sig"
        )
        fig = plot(results, st)
        png = d / f"{stem}.png"
        with plt.rc_context(ORIGIN_RC):
            fig.savefig(png)
        plt.close(fig)
        print(f"\n💾 {png}\n💾 {page}\n💾 {d / f'{stem}_T50.csv'}")
    if show:
        webbrowser.open_new(page.resolve().as_uri())
    return results
