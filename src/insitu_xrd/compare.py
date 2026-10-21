"""여러 실험의 피크별 결정화 분율 X (정규화 면적) vs 온도 비교 + T50.

설정 파일(TOML, 기본 experiments.toml)에 실험별 이미지 폴더 · 온도 로그(recipe) 를 적으면
    uv run python main.py compare [experiments.toml]
→ 피크마다 패널 하나: 각 실험의 X vs PV 온도, 범례에 'x1  T50 401 °C'

비정질 기준 나누기: R = I / I_amorphous − 1  (I_amorphous = 사용 구간 처음 n_norm 프레임 평균)
     디텍터 가장자리 세기 감소(곱셈)와 비정질 배경이 상쇄되어 결정 피크만 봉우리로 남음

적분 창 (실험·피크마다 자동):
     설정의 2θ 는 '이 근처를 봐라' 는 참고값. 결정화가 끝난 마지막 n_norm 프레임의 평균 R 에서
     참고값 ± search 안의 피크를 가우시안(+직선 배경)으로 피팅 → 중심 c, 반치폭 FWHM
     → 창 = c ± fwhm_k × FWHM  (기본 1.5 배: 가우시안 면적의 ~99.9 %)

X  = (A − A₀) / (A₁ − A₀)
     A  = 위 창에서 R 의 면적 (창 양끝을 잇는 직선 배경 제거)
     A₀ = 사용 구간 처음 n_norm 프레임의 중앙값 (전이 전), A₁ = 마지막 n_norm 프레임의 중앙값 (전이 후)
T50 = X 가 처음으로 level(기본 0.5) 을 위로 넘는 온도 (앞뒤 프레임 사이 선형보간)

이미지 폴더 · 온도 로그는 설정 파일에 적힌 경로 그대로 사용 (자동으로 찾지 않음).
적분 결과는 out/cache/ 에 저장해 두고 이미지 폴더가 그대로면 다시 계산하지 않음.
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
    30.0  # 첫 사용 프레임이 승온 시작보다 이만큼 [°C] 넘게 늦으면 '전이 전' 기준이 없다고 봄
)


# ═════════════════════════════ 설정 ═════════════════════════════
@dataclass(slots=True)
class Experiment:
    name: str
    images: list[Path]
    recipe: list[Path]
    start: str | None = None
    end: str | None = None
    color: str | None = None
    group: str = ""  # 어느 [[...]] 에서 왔는지 (iso 면 overview 가 등온 모드)


@dataclass(slots=True)
class Settings:
    peaks: dict[str, float]  # 피크 이름 → 참고 2θ (이 근처에서 실제 피크를 찾음)
    exps: list[Experiment]
    segment: str = "heat"  # "heat" = 승온 구간만, "all" = 전체
    level: float = 0.5
    n_norm: int = 10
    t_range: tuple[float, float] | None = None
    tth_bin: float = 0.02
    search: float = 0.5  # 참고 2θ ± search 안에서 피크를 찾음 [deg]
    fwhm_k: float = 1.5  # 적분 창 = 중심 ± fwhm_k × FWHM
    path: Path = DEFAULT_CONFIG
    group: str = "exp"  # [[exp]] / [[exp2]] / ... (실험 회차)


def _as_list(v) -> list[str]:
    return [v] if isinstance(v, str) else list(v)


def exp_groups(raw: dict) -> list[str]:
    """설정 파일의 실험 회차 목록 = images 가 있는 [[...]] 전부.
    [[exp]], [[exp2]], [[exp3]] ... 번호순 먼저, 그다음 [[iso]] 같은 나머지는 파일에 적힌 순서."""
    keys = [
        k
        for k, v in raw.items()
        if isinstance(v, list) and v and all(isinstance(e, dict) and "images" in e for e in v)
    ]
    num = sorted((k for k in keys if re.fullmatch(r"exp\d*", k)), key=lambda k: int(k[3:] or 1))
    return num + [k for k in keys if k not in num]


def entry_name(e: dict, i: int = 1) -> str:
    """[[...]] 항목 이름: name, 없으면 (등온) sample_T_iso, sample, run{i}."""
    if e.get("name"):
        return str(e["name"])
    sample = str(e.get("sample") or f"run{i}")
    return f"{sample}_{e['T_iso']:g}" if "T_iso" in e else sample


def load_settings(
    path: Path = DEFAULT_CONFIG, need_exps: bool = True, group: str = "exp"
) -> Settings:
    """설정 파일 읽기 (실험 목록은 [[group]]). 경로는 적힌 그대로 (상대경로면 프로젝트 폴더 기준)."""
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
    # 숫자 하나 = 참고 2θ. [시작, 끝] 으로 적은 예전 형식은 가운데를 참고값으로
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
    """설정 파일의 실험 이름 → Experiment (overview x1 처럼 쓸 때).
    'exp2.x8' 처럼 회차를 붙이면 그 회차에서, 아니면 첫 회차부터 찾음.
    이름이 정확히 맞는 게 없으면 sample 로도 찾음 (iso.x1 → [[iso]] 의 sample = "x1" 인 x1_380)."""
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
    """오류 메시지용: 설정 파일의 실험 이름 전부 ('exp.x1, exp2.x1, ..., iso.x1_380')."""
    if not Path(path).exists():
        return ""
    raw = load_toml(path)
    return ", ".join(
        f"{g}.{e.name}" for g in exp_groups(raw) for e in load_settings(path, False, g).exps
    )


def check_paths(exps: list[Experiment]) -> None:
    """계산 시작 전에 없는 경로를 한꺼번에 알려 줌."""
    missing = [f"   {e.name}: {p}" for e in exps for p in (*e.images, *e.recipe) if not p.exists()]
    if missing:
        raise SystemExit(
            "❌ 경로 없음 (experiments.toml 의 images / recipe 확인):\n" + "\n".join(missing)
        )


# ═════════════════════════════ 실험 하나 ═════════════════════════════
@dataclass(slots=True)
class ExpResult:
    name: str
    table: pd.DataFrame  # 프레임별 file, time_kst, T_pv, kind, A_<peak>, X_<peak>
    t50: dict[str, float]
    note: str = ""
    color: str = "#000000"
    ok: bool = True  # False = 전이 전 데이터가 없어 X 를 정규화할 수 없음
    windows: dict[str, tuple[float, float, float, float]] | None = (
        None  # 피크 → (중심, FWHM, 창 시작, 끝)
    )
    trans: dict[str, dict] | None = None  # 피크 → transition_point 결과 (T50 오차 · 폭)


def _clip_window(
    x: np.ndarray, ok: np.ndarray, c: float, lo: float, hi: float
) -> tuple[float, float]:
    """[lo, hi] 를 중심 c 에서 이어지는 ok 구간 안으로 줄임 (갭(NaN) 은 통과, 가장자리 낙하는 막음)."""
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
    """참고 2θ 근처 피크의 (중심, FWHM, 적분 창 반폭 = k×FWHM).
    가우시안+직선 배경 피팅, 실패하면 반값 지점 직접 찾기."""
    fit = fit_peak(x, prof, guess, half_window=search)
    dx = x[1] - x[0]
    if (
        np.isfinite(fit.fwhm)
        and abs(fit.center - guess) <= search
        and 2 * dx < fit.fwhm < 2 * search
    ):
        return fit.center, fit.fwhm, k * fit.fwhm
    # 대안: 범위 양끝을 잇는 직선 배경을 빼고 최댓값의 절반이 되는 좌우 지점
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
    """프레임별 피크 (중심, FWHM): 가우시안 + 직선 배경 피팅. 피크가 실제로 있는 프레임
    (면적 ≥ 최종 면적의 min_frac) 만, 나머지는 NaN. 피팅 범위 = c ± max(1.5·FWHM, 0.4°)."""
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
    """피크별 (중심, FWHM, 창 시작, 끝) · 프레임별 면적 · (widths=True 면) 프레임별 (중심, FWHM).
    → (windows, areas, shapes)
    base 프레임(전이 전, 비정질) 평균으로 나눔: R = I / I_amorphous − 1
      → 디텍터 가장자리 세기 감소(곱셈 효과)와 비정질 배경이 상쇄되어 결정 피크만 깔끔한 봉우리로 남음.
    창은 final 프레임(결정화가 끝난 상태) 평균 R 에서 [peaks] 참고값 근처 피크를 찾아 정함."""
    pre = time_mean(Z[base])
    cover = np.isfinite(pre) & (pre > 0.3 * np.nanmedian(pre))  # 디텍터가 거의 안 닿는 2θ 제외
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
    use: np.ndarray  # 프레임별: 계산에 쓴 프레임 (승온 · 온도 범위 · 빔 정상)
    kind: np.ndarray  # 프레임별 구간 종류 heat/hold/cool
    ok: bool  # False = 전이 전 데이터가 없어 정규화 불가
    msg: str
    t50: dict[str, float]
    windows: dict[str, tuple[float, float, float, float]]  # 피크 → (중심, FWHM, 창 시작, 끝)
    A: dict[str, np.ndarray]
    X: dict[str, np.ndarray]
    center: dict[str, np.ndarray]  # 프레임별 피크 중심 [deg] (피크 없는 프레임 NaN)
    fwhm: dict[str, np.ndarray]  # 프레임별 반치폭 [deg]
    trans: dict[str, dict]  # 피크 → transition_point 결과 (T50, err, width, window ...)


def analyze(
    x: np.ndarray,
    Z: np.ndarray,
    t: np.ndarray,
    T: np.ndarray,
    log: TempLog,
    st: Settings,
    verbose: bool = True,
) -> Analysis:
    """프레임 프로파일 Z (시간순) + 프레임 온도 T(PV) → 피크별 X, T50. compare · overview 공용."""
    say = print if verbose else (lambda *a, **k: None)
    has_T = np.isfinite(T)
    # 구간 종류 (승온/유지/하온)
    kind = np.full(len(t), "", dtype=object)
    heat_T0 = np.nan
    if has_T.any():
        tt = t[has_T]
        # 로그 전체로 감지 (프레임 범위로 자르면 승온 시작 온도를 잘못 앎)
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
    use &= beam_ok(Z, use)  # 빔 꺼진 프레임 제외 (사용 프레임 세기 중앙값 기준)

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
    for pk, (
        c,
        _,
        lo,
        hi,
    ) in windows.items():  # 보고용 FWHM = 프레임별 피팅값 (창을 정한 마지막 프레임들)
        tail = shapes[pk][1][iu_all[-st.n_norm :]]
        if np.isfinite(tail).any():
            windows[pk] = (c, float(np.nanmedian(tail)), lo, hi)
    trans = {}
    for pk, A in As.items():
        X = np.full(len(A), np.nan)
        iu = np.flatnonzero(use & np.isfinite(A))
        n = min(st.n_norm, max(1, len(iu) // 4))
        # 전이점: 전이 폭에 비례하는 창만 씀 (tracks.transition_point) — 범위 끝·먼 아웃라이어·드리프트에 둔감
        tp = (
            transition_point(T[iu], A[iu], st.level, n_end=n) if len(iu) >= 4 * n else {"ok": False}
        )
        if tp["ok"]:  # 전이 창의 전·후 배경 직선으로 → X 곡선이 T50 에서 level
            X[iu] = normalize_step(T[iu], A[iu], tp)
        elif len(iu) >= 2:  # 창을 못 잡으면(전이 전후 데이터 부족) 예전 방식: 양 끝 중앙값
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
    Z: np.ndarray  # 세기 [frame, 2θ], 시간순
    t: np.ndarray  # 촬영시각 Unix [frame]
    files: np.ndarray
    T: np.ndarray  # 프레임 PV 온도 (로그 밖 NaN)
    log: TempLog | None  # 온도 로그가 없으면 None
    note: str = ""


def load_exp(exp: Experiment, st: Settings, geo: Geometry, workers: int | None) -> ExpData:
    """실험의 이미지 폴더(들) 적분(캐시) → 시간순으로 합치고 start/end 로 자르고 온도 매칭."""
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

    # 시간 범위 제한 (선택)
    keep = np.ones(len(t), bool)
    day = fmt_kst(t[0])[:10]
    if exp.start:
        keep &= t >= parse_time(exp.start, day)
    if exp.end:
        keep &= t <= parse_time(exp.end, day)
    Z, t, files = Z[keep], t[keep], files[keep]

    # 온도: 로그가 실제로 있는 시각만 (바깥은 NaN)
    if not exp.recipe:  # 온도 로그 없음 → 온도 NaN (avrami 는 그래도 n·k 계산)
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


# ═════════════════════════════ 그림 ═════════════════════════════
def _lab(r: ExpResult, pk: str) -> str:
    t50 = r.t50[pk]
    if not np.isfinite(t50):
        return f"{r.name}  –"
    tp = (r.trans or {}).get(pk, {})
    width = f" (Δ{tp['width']:.0f})" if tp.get("ok") else ""
    return f"{r.name}  {t50:.0f} °C{width}"  # 범례 제목: T50 (ΔT25–75)


NO_BASELINE = "no pre-transition images"


def smooth(y: np.ndarray, size: int = 7) -> np.ndarray:
    """NaN 을 건너뛴 이동 중앙값 (FWHM 처럼 프레임마다 흔들리는 값의 추세선)."""
    out = np.full(len(y), np.nan)
    ok = np.flatnonzero(np.isfinite(y))
    if len(ok) >= size:
        out[ok] = median_filter(y[ok], size=size, mode="nearest")
    return out


def _fwhm_rows(r: ExpResult, pk: str) -> pd.DataFrame:
    """FWHM vs T 에 그릴 프레임: 사용 구간(승온 · 온도 범위) 중 피크가 있는 프레임."""
    d = r.table[r.table["used"]]
    return d[np.isfinite(d[f"FWHM_{pk}"])]


LABEL_ROWS = (1.13, 1.22, 1.31)  # T50 숫자를 놓는 높이 (X 축 위쪽 여백, 가까우면 다음 줄)


def _t50_marks(results: list[ExpResult], pk: str, span: float) -> list[tuple]:
    """그릴 전이점 [(결과, T50, T25, T75, 라벨 높이)] — T50 순, 라벨이 겹치지 않게 줄 배정."""
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
            for r in results:  # FWHM 은 전이 전 기준이 없어도 그릴 수 있음
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
            # 전이점: 폭(25–75 %) 음영 · 세로 점선(아래 FWHM 칸까지) · ◆ · 위쪽에 온도 숫자
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
        for a in legend_axes:  # 그래프 안 빈 곳에 범례 (곡선 · ◆ · T50 숫자 안 가리게)
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
        for r in results:  # FWHM vs T (아래 줄)
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
        # 전이점: 폭(25–75 %) 음영 · 세로 점선 · ◆ (마우스 올리면 T50 ± 오차 · 폭) · 위쪽 온도 숫자
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
    # 창이 커도 늘어나지 않는 크기 (피크 하나당 ~540 px) → 스샷해도 글씨 비율 일정
    return write_fit_html(fig, path, 150 + 540 * len(peaks), 800)


# ═════════════════════════════ 실행 ═════════════════════════════
def run(
    config: Path = DEFAULT_CONFIG,
    geo: Geometry = Geometry(),
    workers: int | None = None,
    show: bool = True,
    groups: list[str] | None = None,
) -> dict[str, list[ExpResult]]:
    """실험 회차([[exp]], [[exp2]], ..., [[iso]])마다 차례로 비교 → 회차별 결과 파일 · 브라우저 창."""
    raw = load_toml(config)
    groups = groups or exp_groups(raw)
    picks = [g for g in groups if g not in raw]  # 회차가 아닌 것 = 실험 이름 (iso3.x1, x8_300 ...)
    if picks:
        return {"pick": _run_picks(picks, config, geo, workers, show, raw)}
    unknown = [g for g in groups if g not in raw]
    if not groups or unknown:
        raise SystemExit(
            f"❌ {config}: 회차 {', '.join(unknown) or '[[exp]]'} 없음 "
            f"(있는 회차: {', '.join(exp_groups(raw)) or '없음'})"
        )
    # 같은 이름(예: x8)은 모든 회차에서 같은 색
    names = list(
        dict.fromkeys(
            entry_name(e, i) for g in exp_groups(raw) for i, e in enumerate(raw[g], start=1)
        )
    )
    colors = {n: EXP_COLORS[i % len(EXP_COLORS)] for i, n in enumerate(names)}
    for g in groups:  # 경로 확인은 계산 전에 모든 회차 한꺼번에
        check_paths(load_settings(config, group=g).exps)
    out = {}
    for g in groups:
        st = load_settings(config, group=g)
        if g.startswith("iso"):  # 등온 회차: 온도 대신 시간 (등온 시작부터) 으로 비교
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
    """실험 이름 여러 개 (회차 섞어도 됨) → 한 장. 전부 등온 회차면 시간 비교, 아니면 온도 비교."""
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
    if show:  # 회차마다 새 브라우저 창
        webbrowser.open_new(page.resolve().as_uri())
    return results
