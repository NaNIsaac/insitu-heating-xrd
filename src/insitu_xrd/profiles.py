"""Overlay the 2θ profiles of several scans at the same temperature (state).

    uv run main.py profiles exp2.x1 exp2.x2 exp.x8      experiment names from experiments.toml
    uv run main.py profiles x1 x8 -T 350 -T 450 --no-end      temperatures given directly

Rows are states (each temperature ± tol while heating; 'end' is the last frames after cooling).
Columns are [intensity I | R = I / I_amorphous - 1, the same quantity as compare, leaving only
the crystalline peaks]. Lines are scans; reference 2θ from [peaks] are dotted.
Output: out/profiles/<names>.png / .html
"""

from __future__ import annotations

import webbrowser
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from . import segments as sg
from .compare import DEFAULT_CONFIG, find_exp, load_exp, load_settings
from .config import out_dir, saving
from .geometry import Geometry
from .heatmap import beam_ok, regrid, time_mean
from .style import EXP_COLORS, ORIGIN_RC, origin_plotly


@dataclass(slots=True)
class Scan:
    name: str
    x: np.ndarray
    I: dict[str, np.ndarray]  # state -> mean profile
    R: dict[str, np.ndarray]
    T: dict[str, float]  # state -> actual mean temperature
    color: str


def _state_frames(
    T: np.ndarray, kind: np.ndarray, ok: np.ndarray, target: float, tol: float, n: int
) -> np.ndarray:
    """The n frames closest to target within ± tol during heating or hold (before cooling)."""
    m = ok & np.isin(kind, ["heat", "hold"]) & (np.abs(T - target) <= tol)
    idx = np.flatnonzero(m)
    return idx[np.argsort(np.abs(T[idx] - target))[:n]] if len(idx) else idx


def load_scan(
    name: str,
    temps: list[float],
    end: bool,
    config: Path,
    geo: Geometry,
    workers: int | None,
    color: str,
    tol: float = 3.0,
    n: int = 5,
) -> Scan:
    e = find_exp(name, config)
    if e is None:
        raise SystemExit(f"❌ {Path(config).name} 에 실험 '{name}' 없음 (예: x1, exp2.x8)")
    st = load_settings(config, need_exps=False)
    d = load_exp(e, st, geo, workers)
    if np.isfinite(
        d.T
    ).any():  # frames outside the log time range (older measurements left after overwriting) are not this run
        keep = np.isfinite(d.T)
        d.Z, d.t, d.T = d.Z[keep], d.t[keep], d.T[keep]
    ok = beam_ok(d.Z)
    kind = np.full(len(d.t), "", dtype=object)
    if d.log is not None and np.isfinite(d.T).any():
        det = sg.detect(d.log, d.t[0], d.t[-1])
        kind = sg.kinds_at(d.t, det.segments)
    # amorphous reference: the first n_norm good frames (start of heating)
    base = np.flatnonzero(ok)[: st.n_norm]
    pre = time_mean(d.Z[base])
    cover = np.isfinite(pre) & (pre > 0.3 * np.nanmedian(pre))  # skip 2θ the detector barely covers
    with np.errstate(all="ignore"):
        R = np.where(cover, d.Z / pre - 1.0, np.nan)
    out = Scan(name, d.x, {}, {}, {}, color)
    states = [(f"{t:g} °C", _state_frames(d.T, kind, ok, t, tol, n)) for t in temps]
    if end:
        states.append(("end", np.flatnonzero(ok)[-10:]))
    for lab, idx in states:
        if not len(idx):
            print(f"   ⚠️  {name}: {lab} ± {tol:g} °C 프레임 없음")
            continue
        out.I[lab], out.R[lab] = time_mean(d.Z[idx]), time_mean(R[idx])
        out.T[lab] = float(np.nanmean(d.T[idx])) if np.isfinite(d.T[idx]).any() else np.nan
    return out


def _states(scans: list[Scan]) -> list[str]:
    return list(dict.fromkeys(s for sc in scans for s in sc.I))


def _label(sc: Scan, state: str) -> str:
    T = sc.T.get(state, np.nan)
    return f"{sc.name}" + (f"  ({T:.0f} °C)" if state == "end" and np.isfinite(T) else "")


def plot(scans: list[Scan], peaks: dict[str, float]) -> plt.Figure:
    states = _states(scans)
    x = scans[0].x
    with plt.rc_context(ORIGIN_RC):
        fig, axes = plt.subplots(
            len(states), 2, figsize=(14, 2.9 * len(states) + 0.6), squeeze=False, sharex=True
        )
        for row, state in zip(axes, states):
            for sc in scans:
                if state not in sc.I:
                    continue
                for ax, y in zip(row, (sc.I[state], sc.R[state])):
                    ax.plot(
                        x,
                        regrid(x, sc.x, y[None])[0],
                        color=sc.color,
                        lw=1.6,
                        label=_label(sc, state),
                    )
            for ax in row:
                for pk, v in peaks.items():
                    ax.axvline(v, color="0.55", ls=":", lw=1.0, zorder=0)
                ax.text(
                    0.01,
                    0.95,
                    state,
                    transform=ax.transAxes,
                    ha="left",
                    va="top",
                    fontsize=15,
                    fontweight="bold",
                )
            row[0].set_ylabel("Intensity (a.u.)")
            row[1].set_ylabel("R = I / I$_{amorph}$ − 1")
            row[1].axhline(0, color="0.6", lw=0.8)
            row[1].legend(loc="upper right", fontsize=12)
        for ax in axes[-1]:
            ax.set_xlabel("2θ (°)")
        top = axes[0]
        for pk, v in peaks.items():
            for ax in top:
                ax.text(
                    v,
                    1.02,
                    pk,
                    transform=ax.get_xaxis_transform(),
                    ha="center",
                    va="bottom",
                    fontsize=11,
                    color="0.4",
                )
        axes[0][0].set_title("Intensity", fontsize=16, fontweight="bold", pad=18)
        axes[0][1].set_title("amorphous-normalised  R", fontsize=16, fontweight="bold", pad=18)
        fig.tight_layout()
    return fig


def write_html(scans: list[Scan], peaks: dict[str, float], path: Path) -> Path:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    states = _states(scans)
    x = scans[0].x
    fig = make_subplots(
        rows=len(states),
        cols=2,
        shared_xaxes=True,
        vertical_spacing=0.04,
        horizontal_spacing=0.08,
        subplot_titles=[f"{s} — {k}" for s in states for k in ("I", "R = I/I_amorph − 1")],
    )
    for i, state in enumerate(states, start=1):
        for sc in scans:
            if state not in sc.I:
                continue
            for j, y in enumerate((sc.I[state], sc.R[state]), start=1):
                fig.add_trace(
                    go.Scatter(
                        x=x,
                        y=regrid(x, sc.x, y[None])[0],
                        mode="lines",
                        name=sc.name,
                        legendgroup=sc.name,
                        showlegend=(i == 1 and j == 1),
                        line=dict(color=sc.color, width=1.8),
                        hovertemplate=f"<b>{_label(sc, state)}</b> · {state}<br>2θ %{{x:.3f}}° · %{{y:.3f}}<extra></extra>",
                    ),
                    i,
                    j,
                )
        for j in (1, 2):
            for pk, v in peaks.items():
                fig.add_vline(x=v, line=dict(color="#999", dash="dot", width=1), row=i, col=j)
    fig.update_layout(
        template="simple_white",
        height=280 * len(states) + 120,
        font=dict(family="Arial", size=16),
        title=dict(
            text="<b>" + " · ".join(sc.name for sc in scans) + "</b>  — 2θ profiles", x=0.01
        ),
        legend=dict(orientation="h", x=0, y=1.0, yanchor="bottom", xanchor="left"),
        margin=dict(t=110, l=70, r=30, b=60),
        hovermode="x unified",
    )
    origin_plotly(fig)
    fig.update_xaxes(title="2θ (°)", row=len(states))
    fig.write_html(
        path,
        include_plotlyjs=True,
        config={"displaylogo": False, "scrollZoom": True, "toImageButtonOptions": {"scale": 3}},
    )
    return path


def run(
    names: list[str],
    temps: list[float] = (300, 400, 500, 700),
    end: bool = True,
    config: Path = DEFAULT_CONFIG,
    geo: Geometry = Geometry(),
    workers: int | None = None,
    tol: float = 3.0,
    show: bool = True,
) -> list[Scan]:
    print(
        f"📋 프로파일 겹쳐 보기: {', '.join(names)}  —  "
        f"{', '.join(f'{t:g} °C' for t in temps)}{' + end' if end else ''} (± {tol:g} °C)"
    )
    scans = [
        load_scan(nm, list(temps), end, config, geo, workers, EXP_COLORS[i % len(EXP_COLORS)], tol)
        for i, nm in enumerate(names)
    ]
    peaks = load_settings(config, need_exps=False).peaks
    d = out_dir("profiles")
    stem = "+".join(n.replace(".", "_") for n in names)
    page = write_html(scans, peaks, d / f"{stem}.html")
    if saving():
        fig = plot(scans, peaks)
        png = d / f"{stem}.png"
        with plt.rc_context(ORIGIN_RC):
            fig.savefig(png)
        plt.close(fig)
        print(f"\n💾 {png}\n💾 {page}")
    if show:
        webbrowser.open_new(page.resolve().as_uri())
    return scans
