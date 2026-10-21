"""폴더(들) + 온도 로그 CSV → [구간 라벨 / 온도 / 2θ 히트맵 (가로축=시간) | 구간 경계 프로파일] 한 장 요약.
승온/유지/하온 구간을 자동 감지해 점선으로 나누고, 각 경계 시점의 프로파일·온도를 오른쪽에 쌓아 보여줌.
폴더를 여러 개 주면 시간순으로 이어 붙여 전체 온도 프로파일을 한 장에 (이미지 없는 시간은 회색)."""

from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LogNorm, Normalize
from matplotlib.ticker import AutoMinorLocator, FuncFormatter, Locator, NullLocator

from . import segments as sg
from . import tracks as tk
from .config import out_dir, saving
from .heatmap import (
    Config,
    HeatmapResult,
    beam_ok,
    color_range,
    compute,
    intensity_label,
    regrid,
    time_mean,
)
from .style import KIND_LS, ORIGIN_RC, SEG_COLOR, TRACK_COLORS
from .temperature import TempLog, find_log


def tth_range(x: np.ndarray, Z: np.ndarray) -> tuple[float, float]:
    """대부분 프레임에서 데이터가 있는 2θ 범위 (빈 칸 가장자리까지). 디텍터 끝 빈 줄 제외용."""
    ok = np.flatnonzero(np.isfinite(Z).mean(axis=0) >= 0.5)  # 프레임 절반 이상에서 값이 있는 2θ
    if not len(ok):
        return float(x[0]), float(x[-1])
    h = 0.5 * (x[1] - x[0]) if len(x) > 1 else 0.0
    return float(x[ok[0]] - h), float(x[ok[-1]] + h)


def _gap_threshold(t: np.ndarray) -> float:
    """프레임 간격이 이보다 크면 '이미지 없음' 으로 봄 [s]."""
    dt = np.diff(t)
    return max(5 * float(np.median(dt)), 60.0) if len(dt) else 60.0


def gap_indices(t: np.ndarray) -> np.ndarray:
    """프레임 i 와 i+1 사이에 이미지 공백이 있는 i 들."""
    return np.flatnonzero(np.diff(t) > _gap_threshold(t))


def _boundary_profiles(
    res: HeatmapResult,
    segs: list[sg.Segment],
    det: sg.Detection | None,
    scale: float,
    n_avg: int = 1,
):
    """구간 시작·경계·끝 시각 → (상대시간, 온도, 추정여부, 프로파일).
    가장 가까운 정상 프레임 ±n_avg 평균. 근처에 프레임이 없는 경계(이미지 공백)는 건너뜀."""
    fr, Z, t_unix = res.frames, res.intensity, res.t_unix
    marks = [segs[0].t0] + [s.t1 for s in segs] if segs else [t_unix[0], t_unix[-1]]
    # 빔 꺼짐 등으로 거의 0 인 프레임은 대표 프로파일에서 제외 (폴더마다 세기 수준이 다를 수 있어 폴더별로)
    good = np.flatnonzero(beam_ok(Z, fr["folder"].to_numpy() if "folder" in fr else None))
    if not len(good):
        good = np.arange(len(Z))
    tol = _gap_threshold(t_unix)
    out, seen = [], set()
    for tm in marks:
        i = int(good[np.argmin(np.abs(t_unix[good] - tm))])
        if abs(t_unix[i] - tm) > tol or i in seen:
            continue
        seen.add(i)
        k = int(np.searchsorted(good, i))
        prof = time_mean(Z[good[max(0, k - n_avg) : k + n_avg + 1]])
        if det is not None:
            T, e = (v[0] for v in det.at(np.array([t_unix[i]])))
        else:
            T, e = (fr["temp_pv"].iloc[i] if "temp_pv" in fr else np.nan), False
        out.append(((t_unix[i] - t_unix[0]) / scale, T, bool(e), prof))
    return out


def _spread(ys: list[float], hs: list[float], lo: float, hi: float) -> list[float]:
    """중심 ys, 크기 hs 인 라벨들이 [lo, hi] 안에서 겹치지 않게 최소 이동 (1차원)."""
    if not ys:
        return []
    order = np.argsort(ys)
    y = np.array(ys, float)[order]
    h = np.array(hs, float)[order]
    y = np.clip(y, lo + h / 2, hi - h / 2)
    for i in range(1, len(y)):  # 아래에서 위로 밀기
        y[i] = max(y[i], y[i - 1] + (h[i - 1] + h[i]) / 2)
    if y[-1] + h[-1] / 2 > hi:  # 위가 넘치면 위에서 아래로 되밀기
        y[-1] = hi - h[-1] / 2
        for i in range(len(y) - 2, -1, -1):
            y[i] = min(y[i], y[i + 1] - (h[i] + h[i + 1]) / 2)
    out = np.empty_like(y)
    out[order] = y
    return list(out)


def _with_gaps(t: np.ndarray, y: np.ndarray, Z: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """프레임 공백에 NaN 행을 넣어 pcolormesh 가 마지막 프레임을 공백 전체로 늘리지 않게."""
    gi = gap_indices(t)
    if len(t) < 2 or not len(gi):
        return y, Z
    half = 0.5 * float(np.median(np.diff(t))) * (y[-1] - y[0]) / (t[-1] - t[0])
    # 공백마다 (앞 프레임 + half, 뒤 프레임 − half) 두 줄을 NaN 으로 끼워 넣음
    ins = np.repeat(gi + 1, 2)
    y_new = np.ravel(np.column_stack([y[gi] + half, y[gi + 1] - half]))
    return np.insert(y, ins, y_new), np.insert(Z, ins, np.nan, axis=0)


TIME_STEPS = {
    "min": [0.05, 0.1, 0.2, 0.25, 0.5, 1, 2, 2.5, 5, 10, 15, 20, 30, 60],
    "s": [1, 2, 5, 10, 15, 20, 30, 60, 120, 150, 300, 600, 900, 1200, 1800, 3600],
}


def nice_ticks(lo: float, hi: float, unit: str, max_n: int = 20) -> np.ndarray:
    """[lo, hi] 안에서 라벨이 max_n 개 이하가 되는 가장 촘촘한 '보기 좋은' 시간 간격의 눈금."""
    steps = TIME_STEPS[unit]
    step = next((st for st in steps if (hi - lo) / st <= max_n), steps[-1])
    return np.arange(np.ceil(lo / step - 1e-9) * step, hi + 1e-9, step)


def temp_lookup(det: sg.Detection | None, fr: pd.DataFrame, t0: float, scale: float):
    """상대시간 → (온도, 추정여부) 함수. 온도 정보가 없으면 None."""
    if det is not None and np.isfinite(det.T).any():

        def f(rel):
            t_abs = t0 + np.atleast_1d(np.asarray(rel, float)) * scale
            T, est = det.at(t_abs)
            return np.where((t_abs >= det.t[0]) & (t_abs <= det.t[-1]), T, np.nan), est

        return f
    if "temp_pv" in fr and fr["temp_pv"].notna().any():

        def f(rel):
            t_abs = t0 + np.atleast_1d(np.asarray(rel, float)) * scale
            T = np.interp(t_abs, fr["time_unix"], fr["temp_pv"])
            return T, np.zeros(len(T), bool)

        return f
    return None


def fmt_temp(T: float, est: bool) -> str:
    return "" if not np.isfinite(T) else f"{'~' if est else ''}{T:.0f}"


class _TempTickLocator(Locator):
    """확대/이동할 때마다 보이는 구간에 맞춰 시간 간격을 다시 고름 (확대할수록 촘촘)."""

    def __init__(self, parent, unit: str, max_n: int):
        self.parent, self.unit, self.max_n = parent, unit, max_n

    def __call__(self):
        lo, hi = sorted(self.parent.get_xlim())
        return nice_ticks(lo, hi, self.unit, self.max_n)


def _temp_ticks(
    ax,
    trel: np.ndarray,
    unit: str,
    t0: float,
    scale: float,
    det: sg.Detection | None,
    fr: pd.DataFrame,
    max_n: int = 20,
    label_ax=None,
) -> None:
    """일정 시간 간격의 온도 값을 눈금 라벨로 (온도 곡선 위 점 표시 포함). 확대하면 간격이 촘촘해짐.
    label_ax 를 주면 숫자 줄을 그 축의 윗변에 (히트맵 윗변 → 히트맵만 잘라도 온도가 보임),
    아니면 ax(온도 그래프) 아랫변에."""
    look = temp_lookup(det, fr, t0, scale)
    if look is None:
        return
    loc = _TempTickLocator(ax, unit, max_n)
    (dots,) = ax.plot([], [], "o", ms=3.5, color="black", zorder=6)

    def label(v, pos=None):
        T, e = look(v)
        s = fmt_temp(T[0], bool(e[0]))
        # 히트맵 윗변 줄은 촘촘해서 두 줄로 엇갈려 (홀수 번째는 위 줄)
        if label_ax is not None and pos is not None and pos % 2:
            s += "\n"
        elif label_ax is not None:
            s = "\n" + s
        return s

    def update(_ax=None):
        ticks = loc()
        dots.set_data(ticks, look(ticks)[0])

    host = label_ax if label_ax is not None else ax
    top = label_ax is not None
    sec = host.secondary_xaxis("top" if top else "bottom")
    sec.xaxis.set_major_locator(loc)
    sec.xaxis.set_major_formatter(FuncFormatter(label))
    sec.xaxis.set_minor_locator(NullLocator())
    sec.tick_params(labelsize=10.5 if top else 9, length=4, width=1.0, direction="in", pad=3)
    update()
    ax.callbacks.connect("xlim_changed", update)
    host.text(
        -0.045,
        1.06 if top else -0.035,
        "T (°C)",
        transform=host.transAxes,
        ha="right",
        va="bottom" if top else "top",
        fontsize=11,
        fontweight="bold",
    )


def frame_breaks(fr: pd.DataFrame) -> np.ndarray:
    """프레임 앞에 이미지 공백 또는 폴더 경계가 있으면 True."""
    t = fr["time_unix"].to_numpy()
    br = np.zeros(len(t), bool)
    br[gap_indices(t) + 1] = True
    if "folder" in fr:
        f = fr["folder"].to_numpy()
        br[1:] |= f[1:] != f[:-1]
    return br


def frame_temps(fr: pd.DataFrame, det: sg.Detection | None) -> np.ndarray:
    if det is not None and np.isfinite(det.T).any():
        return det.at(fr["time_unix"].to_numpy())[0]
    return fr["temp_pv"].to_numpy(float) if "temp_pv" in fr else np.full(len(fr), np.nan)


def frame_kinds(fr: pd.DataFrame, segs: list[sg.Segment]) -> np.ndarray:
    return sg.kinds_at(fr["time_unix"].to_numpy(), segs)


def _runs(mask_vals: np.ndarray):
    """같은 값이 이어지는 [a, b) 구간들."""
    edges = np.flatnonzero(mask_vals[1:] != mask_vals[:-1]) + 1
    return zip(np.r_[0, edges], np.r_[edges, len(mask_vals)])


def _peak_tags(ax_h, tracks: list[tk.Track]) -> None:
    """히트맵 오른쪽 바깥에 피크 이름 (그 2θ 높이, 서로 겹치면 위아래로 벌림) + 짧은 눈금선."""
    if not tracks:
        return
    lo, hi = ax_h.get_ylim()
    fig = ax_h.figure
    h_in = ax_h.get_position().height * fig.get_figheight()
    gap = 10 * 1.3 / 72 / h_in * (hi - lo)  # 글자 한 줄 높이 (데이터 단위)
    ys = _spread([tr.center for tr in tracks], [gap] * len(tracks), lo, hi)
    tr_ = ax_h.get_yaxis_transform()
    for i, (tr, y) in enumerate(zip(tracks, ys)):
        c = TRACK_COLORS[i % len(TRACK_COLORS)]
        ax_h.annotate(
            tr.name,
            xy=(1.0, tr.center),
            xycoords=tr_,
            xytext=(1.035, y),
            textcoords=tr_,
            ha="left",
            va="center",
            fontsize=12.5,
            fontweight="bold",
            color=c,
            annotation_clip=False,
            arrowprops=dict(arrowstyle="-", color=c, lw=1.5, shrinkA=0, shrinkB=0),
        )


def _plot_tracks(
    ax_h,
    ax_k,
    ax_v,
    trel,
    tracks: list[tk.Track],
    temp,
    kinds,
    breaks,
    unit,
    ax_leg=None,
    ax_sty=None,
):
    """히트맵 위 피크 위치 표시 / 피크 면적 vs 시간 / 피크 면적 vs 온도 (승온 실선, 하온 점선).
    범례마다 피크 이름 + 전이 온도 (↑ 나타남 / ↓ 사라짐)."""
    from matplotlib.lines import Line2D

    handles = []
    for i, tr in enumerate(tracks):
        c = TRACK_COLORS[i % len(TRACK_COLORS)]
        area = tk.norm_area(tr.area)
        # 면적 vs 시간 (이미지 공백에서 선이 이어지지 않게 끊음)
        lab = tk.label(tr, temp)
        ax_k.plot(trel, np.where(breaks, np.nan, area), color=c, lw=1.6, label=lab)
        evs = {ev for ev, _ in tr.events}
        handles.append(
            Line2D(
                [],
                [],
                color=c,
                lw=1.8,
                label=lab,
                marker="^" if "appears" in evs else "v" if evs else None,
                ms=8,
                mec="black",
                mew=0.8,
            )
        )
        # 나타남/사라짐 표시
        for ev, k in tr.events:
            mk = "^" if ev == "appears" else "v"
            ax_k.plot(trel[k], area[k], mk, color=c, ms=9, mec="black", mew=0.8, zorder=6)
            for ax in (ax_h, ax_k):
                ax.axvline(trel[k], color=c, lw=1.0, ls=":", alpha=0.9)
        # 면적 vs 온도: 구간 종류(승온/하온/유지)별로 선 모양
        for a, b in _runs(kinds):
            seg = slice(a, b)
            ax_v.plot(temp[seg], area[seg], color=c, lw=1.5, ls=KIND_LS.get(kinds[a], "-"))
        for ev, k in tr.events:
            ax_v.plot(
                temp[k],
                area[k],
                "^" if ev == "appears" else "v",
                color=c,
                ms=9,
                mec="black",
                mew=0.8,
                zorder=6,
            )
            ax_v.axvline(temp[k], color=c, lw=1.0, ls=":")
    _peak_tags(ax_h, tracks)
    ax_k.set_ylabel("Peak area\n(norm.)")
    ax_k.set_xlabel(f"Time ({unit})")
    ax_k.axhline(0, color="0.6", lw=0.8)
    level = tracks[0].level if tracks else 0.5
    for ax in (ax_k, ax_v):  # 전이 기준선 (정규화 면적 = 0.5)
        ax.axhline(level, color="0.25", lw=1.0, ls=(0, (5, 3)), zorder=1)
    ax_k.text(
        1.005,
        level,
        f"{level:g}",
        transform=ax_k.get_yaxis_transform(),
        va="center",
        ha="left",
        fontsize=11,
        color="0.25",
    )
    leg_kw = dict(fontsize=11, frameon=True, framealpha=0.88, edgecolor="0.6", fancybox=False)
    if tracks:
        kw = dict(
            handles=handles,
            handlelength=1.8,
            title=f"peaks  (▲ area crosses {level:g} up, ▼ down)",
            title_fontproperties={"weight": "bold", "size": 12},
        )
        if ax_leg is not None:  # 아래 칸 (전이 온도 상자 오른쪽) 에 꽉 차게
            ax_leg.set_axis_off()
            ax_leg.legend(
                loc="center left",
                bbox_to_anchor=(0.04, 0.5),
                ncol=1,
                borderpad=0.9,
                labelspacing=0.7,
                **kw,
                **{**leg_kw, "fontsize": 10},
            )
        else:
            ax_k.legend(
                loc="upper center",
                bbox_to_anchor=(0.5, -0.3),
                ncol=min(len(handles), 3),
                **kw,
                **leg_kw,
            )
    ax_v.set_xlabel("T (°C)")
    ax_v.set_ylabel("Peak area (norm.)")
    ax_v.axhline(0, color="0.6", lw=0.8)
    ax_v.set_title("Peak area vs temperature", fontsize=15, fontweight="bold", loc="left")
    styles = [
        Line2D([], [], color="0.3", ls=ls, label=lab)
        for lab, ls in (("heating", "-"), ("cooling", "--"), ("hold", ":"))
    ]
    if ax_sty is not None:  # 아래 칸: 선 모양 뜻만 (피크 색·전이 온도는 왼쪽 아래 범례에 있음)
        ax_sty.set_axis_off()
        ax_sty.legend(
            handles=styles,
            loc="upper left",
            ncol=3,
            handlelength=2.4,
            title="Peak area vs T: line style",
            title_fontproperties={"weight": "bold", "size": 12},
            **{**leg_kw, "fontsize": 10},
        )
    else:  # 그래프 오른쪽 바깥: 피크(전이 온도) + 선 모양 뜻
        ax_v.legend(
            handles=handles + [Line2D([], [], ls="none", label="")] + styles,
            loc="upper left",
            bbox_to_anchor=(1.02, 1.0),
            handlelength=2.2,
            title="peak · transition T",
            title_fontproperties={"weight": "bold", "size": 12},
            **leg_kw,
        )


def fwhm_series(
    summary: dict | None, tracks: list[tk.Track] | None
) -> list[tuple[str, str, np.ndarray]]:
    """[(피크 이름, 색, 프레임별 FWHM)] — experiments.toml [peaks] 의 피크. 색은 같은 이름의 추적 피크와 같게."""
    if not summary or not summary.get("fwhm"):
        return []
    names = [tr.name for tr in tracks or []]
    out = []
    for j, (pk, w) in enumerate(summary["fwhm"].items()):
        if not np.isfinite(w).any():
            continue
        i = names.index(pk) if pk in names else len(names) + j
        out.append((pk, TRACK_COLORS[i % len(TRACK_COLORS)], w))
    return out


def _plot_fwhm(ax_w, ax_wt, trel, series, temp, kinds, breaks, unit) -> None:
    """FWHM vs 시간 (점 + 이동 중앙값) / FWHM vs 온도 (승온 실선 · 하온 점선)."""
    from .compare import smooth

    for pk, c, w in series:
        s = smooth(w)
        last = np.nanmedian(w[np.isfinite(w)][-10:])
        ax_w.plot(trel, w, "o", color=c, ms=2.5, alpha=0.3, mew=0)
        ax_w.plot(
            trel, np.where(breaks, np.nan, s), color=c, lw=1.8, label=f"{pk}  {last:.2f}° (end)"
        )
        if ax_wt is not None:
            for a, b in _runs(kinds):
                ax_wt.plot(temp[a:b], s[a:b], color=c, lw=1.5, ls=KIND_LS.get(kinds[a], "-"))
    ax_w.set_ylabel("FWHM (°)")
    ax_w.set_xlabel(f"Time ({unit})", fontsize=14, labelpad=1)
    ax_w.legend(
        loc="upper right",
        fontsize=11,
        frameon=True,
        framealpha=0.85,
        edgecolor="0.6",
        title="per-frame Gaussian fit",
        title_fontsize=11,
    )
    if ax_wt is not None:
        ax_wt.set_xlabel("T (°C)")
        ax_wt.set_ylabel("FWHM (°)")
        ax_wt.set_title("FWHM vs temperature", fontsize=15, fontweight="bold", loc="left")


def _draw_transition_box(ax, summary: dict | None, header: str = "") -> None:
    """전이 온도를 크게 (로그북 스크린샷용): 시료·시작 시각 / 피크 이름 · T50 · 중심/FWHM."""
    ax.set_axis_off()
    if not summary:
        return
    from matplotlib.patches import FancyBboxPatch

    ax.add_patch(
        FancyBboxPatch(
            (0.0, 0.0),
            1.0,
            1.0,
            boxstyle="round,pad=0.0,rounding_size=0.05",
            transform=ax.transAxes,
            fc="#fff8e1",
            ec="#b8860b",
            lw=1.8,
            clip_on=False,
        )
    )
    if header:
        ax.text(
            0.04,
            0.92,
            header,
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=14.5,
            fontweight="bold",
            color="0.1",
        )
    ax.text(
        0.04,
        0.92 - (0.17 if header else 0),
        summary["title"],
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=12,
        color="0.35",
    )
    rows = summary["rows"]
    top = 0.52 if header else 0.62
    if not summary["ok"]:
        ax.text(
            0.5,
            top - 0.15,
            summary["msg"],
            transform=ax.transAxes,
            ha="center",
            va="center",
            fontsize=13,
            color="#b00020",
            wrap=True,
        )
        return
    ys = np.linspace(top, 0.16, len(rows)) if len(rows) > 1 else [(top + 0.16) / 2]
    iso = summary.get("unit") == "min"  # 등온 모드: t50 [min] + n · k
    sides = summary.get("sides") or [f"{c:.2f}°\nFWHM {fwhm:.2f}°" for _, _, c, fwhm in rows]
    for (pk, t50, c, fwhm), side, y in zip(rows, sides, ys):
        ax.text(
            0.04,
            y,
            pk,
            transform=ax.transAxes,
            ha="left",
            va="center",
            fontsize=21,
            fontweight="bold",
            color="0.1",
        )
        val = (f"{t50:.1f} min" if iso else f"{t50:.0f} °C") if np.isfinite(t50) else "–"
        ax.text(
            0.30,
            y,
            val,
            transform=ax.transAxes,
            ha="left",
            va="center",
            fontsize=27.5,
            fontweight="bold",
            color="#b00020",
        )
        ax.text(
            0.97,
            y,
            side,
            transform=ax.transAxes,
            ha="right",
            va="center",
            fontsize=11,
            color="0.35",
            linespacing=1.2,
        )


def plot_overview(
    res: HeatmapResult,
    cfg: Config,
    det: sg.Detection | None = None,
    scaled: bool = False,
    tracks: list[tk.Track] | None = None,
    split: bool = False,
    summary: dict | None = None,
):
    """가로축 = 시간. 위: 구간 라벨 / 온도 vs 시간, 가운데: 2θ 히트맵, 아래: 피크 면적 vs 시간 (시간축 공유).
    오른쪽: 경계 프로파일 / 피크 면적 vs 온도.
    split=True 면 (시간축 그림, 2θ·온도축 그림) 두 장으로 나눠 반환 (창 두 개로 띄울 때)."""
    x, Z, trel, fr = res.tth, res.intensity, res.trel, res.frames
    segs = det.segments if det is not None else []
    scale, t_unix = res.scale, res.t_unix
    gaps = gap_indices(t_unix)

    def rel(t):
        return (np.asarray(t) - t_unix[0]) / scale

    with plt.rc_context(ORIGIN_RC):
        has_tr = tracks is not None
        fw = fwhm_series(summary, tracks)
        has_w = bool(fw)
        has_foot = has_tr or bool(summary)
        # 행: [구간 라벨 | 온도 | (온도 숫자 줄) | 히트맵 | 여백 | 피크 면적 | 여백 | FWHM | 여백 | 아래 칸],
        #     아래 칸 = [전이 온도 상자 | 피크 범례]  ← 왼쪽 아래만 잘라 써도 다 보이게
        # 열: [본 그래프 | 피크 이름 | 컬러바 | 여백 | 경계 프로파일 / 면적 vs 온도 / FWHM vs 온도]
        # FWHM 은 맨 위 (왼쪽 아래 = 피크 면적 · 전이 온도 상자 스크린샷 구역을 가리지 않게)
        o = 2 if has_w else 0  # FWHM 줄 + 여백만큼 아래로 밀림
        hr = (
            ([0.5, 0.2] if has_w else [])
            + [0.42, 0.62, 0.17, 1]
            + ([0.3, 0.62] if has_tr else [])
            + ([0.24, 0.62] if has_foot else [])
        )
        r_w = 0  # FWHM 줄
        r_k = o + 5  # 피크 면적 줄
        r_f = len(hr) - 1  # 아래 칸 줄
        extra_h = (2.2 if has_foot else 0) + (2.0 if has_w else 0)
        if split:
            fig = plt.figure(figsize=(12, (11.5 if has_tr else 8) + extra_h))
            gs = fig.add_gridspec(
                len(hr), 3, height_ratios=hr, width_ratios=[1, 0.12, 0.022], hspace=0.0, wspace=0.0
            )
            r2 = [1] + ([0.28, 0.75] if has_tr else []) + ([0.28, 0.6] if has_w else [])
            fig2 = plt.figure(figsize=(7.5, 7 + (3 if has_tr else 0) + (2.4 if has_w else 0)))
            gs2 = fig2.add_gridspec(len(r2), 1, height_ratios=r2, hspace=0.0)
            ax_p = fig2.add_subplot(gs2[0, 0])
            ax_v = fig2.add_subplot(gs2[2, 0]) if has_tr else None
            ax_wt = fig2.add_subplot(gs2[len(r2) - 1, 0]) if has_w else None
        else:
            fig = fig2 = plt.figure(figsize=(15.5, (13.5 if has_tr else 9) + extra_h))
            gs = fig.add_gridspec(
                len(hr),
                5,
                height_ratios=hr,
                width_ratios=[1, 0.12, 0.022, 0.24, 0.5],
                hspace=0.0,
                wspace=0.0,
            )
            ax_p = fig.add_subplot(gs[o + 1 : o + 4, 4])
            ax_v = fig.add_subplot(gs[r_k, 4]) if has_tr else None
            ax_wt = fig.add_subplot(gs[r_w, 4]) if has_w else None
        ax_h = fig.add_subplot(gs[o + 3, 0])
        ax_t = fig.add_subplot(gs[o + 1, 0], sharex=ax_h)
        ax_l = fig.add_subplot(gs[o + 0, 0], sharex=ax_h)
        cax = fig.add_subplot(gs[o + 3, 2])
        ax_k = fig.add_subplot(gs[r_k, 0], sharex=ax_h) if has_tr else None
        ax_w = fig.add_subplot(gs[r_w, 0], sharex=ax_h) if has_w else None
        ax_l.set_axis_off()
        ax_leg = None
        if has_foot:
            foot = gs[r_f, 0].subgridspec(1, 2, width_ratios=[0.52, 0.48], wspace=0.05)
            ax_s = fig.add_subplot(foot[0, 0])
            ax_leg = fig.add_subplot(foot[0, 1]) if has_tr else None
            _draw_transition_box(
                ax_s, summary, header=f"{cfg.folder.name}   ·   {fr['time_kst'].iloc[0][:16]} KST"
            )
        ax_sty = fig.add_subplot(gs[r_f, 4]) if (has_foot and has_tr and not split) else None

        # 온도 vs 시간: 로그 전체(1 s)로 그림. 로그가 끊긴 곳은 외삽 추정값을 점선으로
        if det is not None and np.isfinite(det.T).any():
            m = (det.t >= t_unix[0]) & (det.t <= t_unix[-1])
            tx = rel(det.t[m])
            ax_t.plot(tx, det.pv[m], color="black", label="PV")
            ax_t.plot(tx, det.sv[m], color="red", ls="--", lw=1.5, label="SV")
            if det.estimated[m].any():
                ax_t.plot(
                    tx,
                    np.where(det.estimated[m], det.T[m], np.nan),
                    color="0.4",
                    ls=":",
                    lw=2.2,
                    label="est.",
                )
            vals = np.concatenate([det.T[m], det.pv[m], det.sv[m]])
            lo, hi = np.nanmin(vals), np.nanmax(vals)
            pad = 0.1 * (hi - lo or 1)
            ax_t.set_ylim(lo - pad, hi + pad)
            ax_t.legend(
                loc="upper left",
                bbox_to_anchor=(1.01, 1.0),
                fontsize=12,
                frameon=False,
                handlelength=1.8,
            )
        elif "temp_pv" in fr and fr["temp_pv"].notna().any():
            ax_t.plot(trel, fr["temp_pv"], color="black", label="PV")
            ax_t.plot(trel, fr["temp_sv"], color="red", ls="--", lw=1.5, label="SV")
            ax_t.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), fontsize=12, frameon=False)
        else:
            ax_t.text(
                0.5,
                0.5,
                "no temperature log in range",
                transform=ax_t.transAxes,
                ha="center",
                va="center",
                fontsize=14,
                color="0.4",
            )
        ax_t.set_ylabel("T (°C)")
        ax_t.tick_params(labelbottom=False)
        ax_t.set_xlim(trel[0], trel[-1])
        _temp_ticks(ax_t, trel, res.time_unit, t_unix[0], scale, det, fr, max_n=36, label_ax=ax_h)

        # 2θ 히트맵: 가로 = 시간, 세로 = 2θ (이미지 없는 시간은 회색)
        vlo, vhi = color_range(Z, cfg.plot.clip)
        kw = (
            {"norm": LogNorm(max(vlo, 1e-3), vhi)}
            if cfg.plot.log_color
            else {"vmin": vlo, "vmax": vhi}
        )
        tg, Zg = _with_gaps(t_unix, trel, Z)
        ax_h.set_facecolor("0.82")
        mesh = ax_h.pcolormesh(
            tg,
            x,
            np.ma.masked_invalid(Zg).T,
            shading="nearest",
            cmap=cfg.plot.cmap,
            rasterized=True,
            **kw,
        )
        ax_h.set_ylim(*tth_range(x, Z))  # 디텍터 밖(모든 프레임 NaN) 가장자리 빈 줄 제외
        ax_h.set_ylabel("2θ (°)")
        ax_h.set_xlabel(f"Time ({res.time_unit}) from {fr['time_kst'].iloc[0][11:19]} KST")
        for i in gaps:
            ax_h.text(
                rel((t_unix[i] + t_unix[i + 1]) / 2),
                0.5,
                "no images",
                transform=ax_h.get_xaxis_transform(),
                ha="center",
                va="center",
                fontsize=14,
                color="0.35",
                style="italic",
            )

        cb = fig.colorbar(mesh, cax=cax)
        cb.set_label(intensity_label(cfg, scaled), fontweight="bold")
        cb.outline.set_linewidth(1.5)
        cax.tick_params(direction="in", width=1.5, which="both")

        for ax in (ax_t, ax_h, ax_k, ax_v, ax_w, ax_wt):
            if ax is not None:
                ax.xaxis.set_minor_locator(AutoMinorLocator(2))
                ax.yaxis.set_minor_locator(AutoMinorLocator(2))
        temps, kinds, breaks = frame_temps(fr, det), frame_kinds(fr, segs), frame_breaks(fr)
        if has_tr:
            _plot_tracks(
                ax_h,
                ax_k,
                ax_v,
                trel,
                tracks,
                temps,
                kinds,
                breaks,
                res.time_unit,
                ax_leg=ax_leg,
                ax_sty=ax_sty,
            )
        if has_w:
            _plot_fwhm(ax_w, ax_wt, trel, fw, temps, kinds, breaks, res.time_unit)
        for ax in (ax_k, ax_w):
            if ax is not None:
                for i in gaps:
                    ax.axvspan(trel[i], trel[i + 1], color="0.88", lw=0, zorder=0)

        # ── 구간: 온도 패널 음영 + 경계 점선(세로), 라벨은 온도 그래프 위 바깥 ──
        labels = []
        for k, s in enumerate(segs):
            x0, x1 = rel(s.t0), rel(s.t1)
            ax_t.axvspan(
                x0,
                x1,
                color=SEG_COLOR[s.kind],
                alpha=0.12 if s.kind != "gap" else 0.5,
                lw=0,
                zorder=0,
                hatch="//" if s.kind == "gap" else None,
            )
            if k:  # 경계
                for ax, c in ((ax_t, "0.2"), (ax_h, "white"), (ax_k, "0.4"), (ax_w, "0.4")):
                    if ax is not None:
                        ax.axvline(x0, color=c, ls="--", lw=1.3, zorder=5)
            lines = s.describe()
            lines[0] = f"{k + 1}. {lines[0]}"
            labels.append((s, x0, x1, "\n".join(lines), max(len(ln) for ln in lines)))

        # 등온 모드: 등온 시작 / 끝 세로선 (온도 · 히트맵 · 면적 · FWHM)
        for xm, lab in (summary or {}).get("marks", []):
            for ax, c in (
                (ax_t, "#b00020"),
                (ax_h, "#ff5c5c"),
                (ax_k, "#b00020"),
                (ax_w, "#b00020"),
            ):
                if ax is not None:
                    ax.axvline(xm, color=c, ls="-.", lw=1.8, zorder=6)
            ax_t.text(
                xm,
                0.97,
                f" {lab}",
                transform=ax_t.get_xaxis_transform(),
                color="#b00020",
                fontsize=12,
                fontweight="bold",
                va="top",
                ha="left",
                zorder=7,
            )

        # 라벨이 서로 겹치지 않게 가로 위치 조정 (구간 가운데에서 최소한만 이동)
        fs = 9.0
        lab_w_in = ax_l.get_position().width * fig.get_figwidth()
        per_char = fs * 0.6 / 72 / lab_w_in * (trel[-1] - trel[0])
        widths = [(n + 2) * per_char for *_, n in labels]
        xs = _spread([(x0 + x1) / 2 for _, x0, x1, *_ in labels], widths, trel[0], trel[-1])
        tr = ax_l.get_xaxis_transform()
        for (s, x0, x1, txt, _), xc in zip(labels, xs):
            c = SEG_COLOR[s.kind]
            ax_l.plot(
                [x0, x1],
                [0.03, 0.03],
                color=c,
                lw=5,
                solid_capstyle="butt",
                transform=tr,
                clip_on=False,
            )
            ax_l.annotate(
                txt,
                xy=((x0 + x1) / 2, 0.03),
                xycoords=tr,
                xytext=(xc, 0.2),
                textcoords=tr,
                ha="center",
                va="bottom",
                fontsize=fs,
                color="white",
                fontweight="bold",
                linespacing=1.15,
                annotation_clip=False,
                bbox=dict(boxstyle="round,pad=0.3", fc=c, ec="none", alpha=0.92),
                arrowprops=dict(arrowstyle="-", color=c, lw=1.5, shrinkA=0, shrinkB=0),
            )
        (ax_w if has_w else ax_l).set_title(
            f"{cfg.folder.name}   start {fr['time_kst'].iloc[0]} KST",
            fontsize=16,
            fontweight="bold",
            loc="left",
            pad=4,
        )

        # ── 경계 시점 프로파일 (아래=처음, 위=나중; 색=온도) ──
        bp = _boundary_profiles(res, list(segs), det, scale)
        Ts = np.array([b[1] for b in bp], float)
        cnorm = (
            Normalize(np.nanmin(Ts), np.nanmax(Ts)) if np.isfinite(Ts).any() else Normalize(0, 1)
        )
        rng = [np.nanpercentile(b[3], 99.5) - np.nanpercentile(b[3], 0.5) for b in bp]
        step = 1.08 * (np.nanmax(rng) if np.isfinite(rng).any() else 1.0)
        for k, (tr_, T, est, prof) in enumerate(bp):
            base = np.nanpercentile(prof, 0.5)
            c = plt.cm.plasma(0.85 * cnorm(T)) if np.isfinite(T) else "0.3"
            ax_p.plot(x, prof - base + k * step, color=c, lw=1.3)
            lab = f"{tr_:.1f} {res.time_unit}"
            if np.isfinite(T):
                lab += f"\n{'~' if est else ''}{T:.0f} °C"
            ax_p.text(
                1.02,
                k * step + 0.1 * step,
                lab,
                transform=ax_p.get_yaxis_transform(),
                fontsize=12,
                color=c,
                va="bottom",
                ha="left",
                fontweight="bold",
            )
        ax_p.set_xlabel("2θ (°)")
        ax_p.set_ylabel("Intensity (offset)")
        ax_p.set_xlim(x[0], x[-1])
        ax_p.yaxis.set_major_locator(NullLocator())
        ax_p.yaxis.set_minor_locator(NullLocator())
        ax_p.xaxis.set_minor_locator(AutoMinorLocator(2))
        ax_p.set_title("Profiles at segment boundaries", fontsize=15, fontweight="bold", loc="left")
    return (fig, fig2) if split else fig


def fit_in_window(fig: plt.Figure, pad: float = 0.12) -> None:
    """바깥 범례·라벨까지 그림 안에 들어오게 축 위치를 옮기고 그림 크기를 맞춤.
    (저장할 때의 bbox_inches='tight' 를 창에서도 적용)"""
    fig.canvas.draw()
    bb = fig.get_tightbbox(fig.canvas.get_renderer())
    W, H = fig.get_size_inches()
    x0, y0, x1, y1 = bb.x0 - pad, bb.y0 - pad, bb.x1 + pad, bb.y1 + pad
    W2, H2 = x1 - x0, y1 - y0
    for ax in fig.axes:
        q = ax.get_position()
        ax.set_position(
            [(q.x0 * W - x0) / W2, (q.y0 * H - y0) / H2, q.width * W / W2, q.height * H / H2]
        )
    fig.set_size_inches(W2, H2, forward=False)


def show_windows(figs: Sequence[plt.Figure]) -> None:
    """창 여러 개를 화면에 맞는 크기로 나란히 띄움 (안 들어가면 각자 화면에 맞춤)."""
    for f in figs:
        fit_in_window(f)
    try:
        win = figs[0].canvas.manager.window
        sw, sh = win.winfo_screenwidth(), win.winfo_screenheight()
    except Exception:  # Tk 가 아닌 백엔드
        sw, sh = 1920, 1080
    sizes = [f.get_size_inches() for f in figs]
    dpi = min(100.0, 0.97 * sw / sum(w for w, _ in sizes), 0.86 * sh / max(h for _, h in sizes))
    side = dpi >= 60  # 너무 작아지면 나란히 말고 각자 화면 크기에 맞춤
    x = 0
    for f, (w, h) in zip(figs, sizes):
        d = dpi if side else min(100.0, 0.95 * sw / w, 0.86 * sh / h)
        f.set_dpi(d)
        mgr = f.canvas.manager
        try:
            mgr.resize(int(w * d), int(h * d))
            mgr.window.geometry(f"+{x}+0")
        except Exception:
            pass
        x += int(w * d) + 8 if side else 40
    plt.show()


def find_segments(
    res: HeatmapResult, cfg: Config, opt: sg.SegmentOptions = sg.SegmentOptions()
) -> sg.Detection | None:
    if not cfg.temp_log:
        return None
    t = res.t_unix
    return sg.detect(TempLog.load(cfg.temp_log), t[0], t[-1], opt)


def _level(Z: np.ndarray) -> float:
    """빔 정상 프레임들의 세기 수준 (프레임 중앙값의 중앙값)."""
    with np.errstate(all="ignore"):
        return float(np.nanmedian(np.nanmedian(Z[beam_ok(Z)], axis=1)))


def combine(
    results: Sequence[HeatmapResult], names: Sequence[str], match: bool = True
) -> HeatmapResult:
    """여러 폴더 결과를 시간순으로 이어 붙임 (2θ 축이 다르면 첫 폴더 축에 보간).
    match=True 면 폴더마다 세기 수준(정상 프레임 median)을 마지막 폴더에 맞춤 (노출 차이 보정)."""
    order = np.argsort([r.frames["time_unix"].iloc[0] for r in results])
    results = [results[i] for i in order]
    names = [names[i] for i in order]
    x = results[0].tth
    Zs = [regrid(x, r.tth, r.intensity) for r in results]
    if match:
        ref = _level(Zs[-1])
        for i, n in enumerate(names):
            k = ref / _level(Zs[i])
            Zs[i] = Zs[i] * k
            print(f"   세기 맞춤 {n}: ×{k:.3g}")
    fr = pd.concat([r.frames.assign(folder=n) for r, n in zip(results, names)], ignore_index=True)
    res = HeatmapResult(x, np.vstack(Zs), fr, results[0].time_unit)
    return res.subset(np.ones(len(fr), bool))  # 상대시간을 첫 폴더 기준으로


def transition_summary(
    res: HeatmapResult, cfg: Config, tracks: list[tk.Track] | None, peaks_cfg: Path | None
) -> dict | None:
    """experiments.toml 의 [peaks]·[options] 로 compare 와 같은 방식의 T50 계산.
    맞는 추적 피크에는 hkl 이름과 T50 을 붙임 (범례·히트맵 이름표에 그대로 나옴)."""
    if peaks_cfg is None or not Path(peaks_cfg).exists() or not cfg.temp_log:
        return None
    from . import compare as cmp
    from .temperature import PV

    st = cmp.load_settings(peaks_cfg, need_exps=False)
    log = TempLog.load(cfg.temp_log)
    t = res.t_unix
    T = log.at(t, PV)
    print(
        f"\n🌡️  전이 온도 — {Path(peaks_cfg).name} 의 피크, T50 = 승온 중 X 가 {st.level:g} 을 넘는 온도"
    )
    an = cmp.analyze(res.tth, res.intensity, t, T, log, st)
    rows = [(pk, an.t50.get(pk, np.nan), an.windows[pk][0], an.windows[pk][1]) for pk in st.peaks]
    if tracks:
        taken: set[int] = set()
        for pk, t50, c, _ in rows:
            near = [
                (abs(tr.center - c), i)
                for i, tr in enumerate(tracks)
                if i not in taken and abs(tr.center - c) <= st.search
            ]
            if not near:
                continue
            i = min(near)[1]
            taken.add(i)
            tr = tracks[i]
            tr.name, tr.t50 = pk, t50
            if np.isfinite(t50):  # ▲ 표시도 같은 기준(X 가 기준을 넘은 첫 프레임)으로
                Xk = an.X[pk]
                k = np.flatnonzero(an.use & (np.nan_to_num(Xk, nan=-1) >= st.level))
                tr.events = [("appears", int(k[0]))] if len(k) else []
    return {
        "title": f"Transition T  (T50: X = {st.level:g}, heating)",
        "rows": rows,
        "ok": an.ok,
        "msg": an.msg or "",
        "level": st.level,
        "fwhm": an.fwhm,
        "center": an.center,
    }  # 프레임별 (피크 없는 프레임 NaN)


def iso_summary(
    res: HeatmapResult,
    cfg: Config,
    tracks: list[tk.Track] | None,
    peaks_cfg: Path | None,
    t0: str | None = None,
) -> dict | None:
    """등온 모드: avrami 와 같은 분석 (승온 → 등온 → 하온, t0 = 등온 시작) 으로 피크별 t50 · n · k.
    transition_summary 와 같은 모양 (전이 상자 · FWHM 줄에 그대로 씀) + 등온 시작/끝 표시."""
    if peaks_cfg is None or not Path(peaks_cfg).exists() or not cfg.temp_log:
        return None
    from . import avrami as av
    from . import compare as cmp
    from .temperature import PV

    st = cmp.load_settings(peaks_cfg, need_exps=False)
    t = res.t_unix
    T = TempLog.load(cfg.temp_log).at(t, PV)
    files = res.frames["file"].to_numpy(str)
    run = av.IsoRun(cmp.Experiment(name=cfg.folder.name, images=[], recipe=[]), cfg.folder.name, t0)
    r = av.analyze_iso(run, res.tth, res.intensity, t, T, files, st)
    print(
        f"\n🌡️  등온 분석 — t0 = {r.t0_src or '–'}, T_iso = {r.T_iso:.1f} °C"
        if np.isfinite(r.T_iso)
        else "\n🌡️  등온 분석"
    )
    idx = {f: i for i, f in enumerate(files)}  # 분석에 쓴 프레임 → 전체 프레임 위치
    pos = np.array([idx[f] for f in r.files]) if len(r.files) else np.empty(0, int)

    def full(a: np.ndarray) -> np.ndarray:
        out = np.full(len(files), np.nan)
        if len(pos) and len(a) == len(pos):
            out[pos] = a
        return out

    rows, details, sides = [], [], []
    for pk in st.peaks:
        f = r.fits.get(pk, {})
        c, w = f.get("center_end", np.nan), f.get("fwhm_end", np.nan)
        rows.append((pk, f.get("t50", np.nan), c, w))
        sides.append(f"n {f.get('n', np.nan):.2f}  k {f.get('k', np.nan):.3g}/min\nFWHM {w:.2f}°")
        details.append(f"{pk} n {f.get('n', np.nan):.2f} · k {f.get('k', np.nan):.3g}/min")
        print(
            f"   {pk:>6}: t50 {f.get('t50', np.nan):.2f} min  n {f.get('n', np.nan):.2f}  "
            f"k {f.get('k', np.nan):.3g}/min  τ {f.get('tau', np.nan):.2f} min"
        )
    if tracks:  # 맞는 추적 피크에 hkl 이름만
        taken: set[int] = set()
        for pk, _, c, _ in rows:
            near = [
                (abs(tr.center - c), i)
                for i, tr in enumerate(tracks)
                if i not in taken and np.isfinite(c) and abs(tr.center - c) <= st.search
            ]
            if near:
                i = min(near)[1]
                taken.add(i)
                tracks[i].name = pk
    marks = []
    hold = av.hold_range(T)
    if np.isfinite(r.t0):
        marks.append(
            ((r.t0 - t[0]) / res.scale, "isothermal start" if r.t0_src == "등온 시작" else "t₀")
        )
    if hold is not None and hold[1] < len(t) - 1:
        marks.append(((t[hold[1]] - t[0]) / res.scale, "isothermal end"))
    T_txt = f"{r.T_iso:.0f} °C" if np.isfinite(r.T_iso) else "T –"
    return {
        "title": f"Isothermal {T_txt}  ·  t50 from isothermal start (X = 0.5)",
        "rows": rows,
        "ok": r.ok,
        "msg": r.fail,
        "level": 0.5,
        "unit": "min",
        "details": details,
        "sides": sides,
        "marks": marks,
        "fwhm": {pk: full(w) for pk, w in r.width.items()},
        "center": {pk: full(c) for pk, c in r.center.items()},
    }


def drop_orphans(res: HeatmapResult) -> HeatmapResult:
    """이미지 공백으로 나뉜 블록 중 온도가 한 프레임도 안 맞는 블록(덮어쓰기 후 남은 예전 측정 등)을 뺌.
    온도 매칭은 절대시각(EPICS) 기준이라, 로그 시간대 밖의 블록은 이 실험이 아님."""
    fr = res.frames
    if "temp_pv" not in fr or not fr["temp_pv"].notna().any():
        return res
    block = np.cumsum(frame_breaks(fr))
    has_T = fr["temp_pv"].notna().to_numpy()
    keep = np.isin(block, np.unique(block[has_T]))
    if keep.all():
        return res
    for b in np.unique(block[~keep]):
        m = block == b
        print(
            f"⚠️  온도 로그와 시간이 전혀 안 겹치는 이미지 블록 제외: {fr['time_kst'][m].iloc[0]} ~ "
            f"{fr['time_kst'][m].iloc[-1][11:]} ({m.sum()} 프레임, 예: {fr['file'][m].iloc[0]})"
            "  — 덮어쓰기 후 남은 다른 측정으로 보임 (--keep-orphans 로 유지)"
        )
    return res.subset(keep)


def _pick_log(res: HeatmapResult, cfg: Config) -> Config:
    """온도 로그를 안 줬으면 프레임 절대시각(EPICS)과 가장 많이 겹치는 data/temperature 아래 로그를 고름."""
    log = find_log(res.t_unix)
    if log is None:
        print("⚠️  프레임 시간대와 겹치는 온도 로그가 data/temperature 에 없음 → 온도 없이 그림")
        return cfg
    print(f"🌡️  온도 로그 자동 선택 (절대시각이 가장 많이 겹침): {log}")
    res.add_temperature(TempLog.load(log))
    return replace(cfg, temp_log=log)


def run(
    cfg: Config,
    seg_opt: sg.SegmentOptions | None = sg.SegmentOptions(),
    extra: Sequence[Path] = (),
    match: bool = True,
    html: bool = True,
    browser: bool = False,
    track_opt: tk.TrackOptions | None = tk.TrackOptions(),
    mpl: bool = False,
    peaks_cfg: Path | None = None,
    keep_orphans: bool = False,
    auto_log: bool = True,
    iso: bool = False,
    iso_t0: str | None = None,
) -> HeatmapResult:
    """cfg.folder (+ extra 폴더들) → overview. 여러 폴더면 결과 이름은 'a+b'.
    cfg.temp_log 가 없고 auto_log 면 시간이 맞는 온도 로그를 자동 선택.
    결과 저장 후 cfg.plot.show 면: 기본은 인터랙티브 HTML 을 브라우저로,
    mpl=True 면 matplotlib 창 두 개 (시간축 그림 / 2θ·온도축 그림)."""
    folders: list[Path] = []
    for f in (cfg.folder, *extra):  # 같은 폴더 중복 지정 시 프레임이 두 번 들어가지 않게
        if f.resolve() in [g.resolve() for g in folders]:
            print(f"⚠️  중복 폴더 무시: {f}")
        else:
            folders.append(f)
    results = [compute(replace(cfg, folder=f)) for f in folders]
    if len(results) > 1:
        res = combine(results, [f.name for f in folders], match)
        cfg = replace(cfg, folder=Path("+".join(res.frames["folder"].unique())))
    else:
        res = results[0]
    if cfg.temp_log is None and auto_log:
        cfg = _pick_log(res, cfg)
    if not keep_orphans:
        res = drop_orphans(res)
    name = cfg.folder.name
    d = out_dir(name)
    det = find_segments(res, cfg, seg_opt) if seg_opt else None
    if det is not None and det.segments:
        segs = det.segments
        t = res.t_unix
        tab = sg.to_frame(segs, t[0], res.scale)
        res.frames["segment"] = sg.assign(t, segs)
        res.frames["temp_profile"], res.frames["temp_estimated"] = det.at(t)
        print(
            "\n📐 온도 구간\n"
            + tab[
                [
                    "segment",
                    "label",
                    "start_kst",
                    "end_kst",
                    "duration_min",
                    "T_start",
                    "T_end",
                    "estimated",
                ]
            ]
            .round(1)
            .to_string(index=False)
        )
        if det.estimated.any():
            print("   (estimated=True: 로그가 끊긴 시간의 온도를 이웃 램프 외삽으로 추정)")
        if saving():
            seg_csv = d / f"{name}_segments.csv"
            tab.to_csv(seg_csv, index=False, encoding="utf-8-sig")
            print(f"💾 {seg_csv}")
    tracks = None
    if track_opt is not None:
        tracks = tk.track(res.tth, res.intensity, frame_breaks(res.frames), track_opt)
    summary = (
        iso_summary(res, cfg, tracks, peaks_cfg, iso_t0)
        if iso  # 등온: t50 · n · k
        else transition_summary(res, cfg, tracks, peaks_cfg)
    )  # 맞는 트랙에 hkl 이름·T50 붙임
    if tracks is not None:
        ev = tk.events_table(tracks, res.frames, frame_temps(res.frames, det))
        print("\n🔎 피크 추적 (↑ 나타남 / ↓ 사라짐)\n" + ev.round(1).to_string(index=False))
        if saving():
            pk_csv = d / f"{name}_peaks.csv"
            tk.to_frames(tracks, res.frames).to_csv(pk_csv, index=False, encoding="utf-8-sig")
            ev.to_csv(d / f"{name}_peak_events.csv", index=False, encoding="utf-8-sig")
            print(f"💾 {pk_csv}")
    if summary:
        for pk in summary["fwhm"]:
            res.frames[f"center_{pk}"] = summary["center"][pk]
            res.frames[f"FWHM_{pk}"] = summary["fwhm"][pk]
        if saving():
            pd.DataFrame(
                summary["rows"],
                columns=["peak", "t50_min" if iso else "T50_C", "center_deg", "FWHM_deg"],
            ).to_csv(d / f"{name}_transition.csv", index=False, encoding="utf-8-sig")
    scaled = len(results) > 1 and match
    if saving():
        fig = plot_overview(res, cfg, det, scaled=scaled, tracks=tracks, summary=summary)
        png = d / f"{name}_overview.png"
        with plt.rc_context(ORIGIN_RC):
            fig.savefig(png)
        plt.close(fig)
        frames_csv = d / f"{name}_frames.csv"
        res.frames.to_csv(frames_csv, index=False, encoding="utf-8-sig")
        print(f"\n💾 {png}\n💾 {frames_csv}")
    show_mpl = cfg.plot.show and mpl
    browser = browser or (cfg.plot.show and not mpl)
    if (html and saving()) or browser:
        from . import interactive

        page = interactive.write_html(
            res,
            cfg,
            det,
            d / f"{name}_overview.html",
            scaled=scaled,
            tracks=tracks,
            summary=summary,
        )
        if saving():
            print(f"💾 {page}  (인터랙티브: 브라우저로 열기)")
        if browser:
            import webbrowser

            webbrowser.open(page.resolve().as_uri())
    if show_mpl:
        with plt.rc_context(ORIGIN_RC):
            figs = plot_overview(
                res, cfg, det, scaled=scaled, tracks=tracks, split=True, summary=summary
            )
            show_windows(figs)
    return res
