"""Overlay the transitions of several scans (any run) in one figure: the left of overview
(temperature, peak area vs time) plus X vs temperature.

    uv run main.py overlay exp.x1 exp2.x1      reproducibility (same sample, different runs)
    uv run main.py overlay x1 x2 x4 x8 exp2.x8

Time axis is minutes from the start of heating (curves overlap when the recipe is the same).
Rows: [temperature vs time] / [X vs time per peak] / [X vs temperature per peak], with each scan's
transition point (T50) as a diamond, dashed line and number.
X and T50 are computed as in compare (tracks.transition_point). Output: out/overlay/<names>.png / .html
"""

from __future__ import annotations

import webbrowser
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .compare import DEFAULT_CONFIG, ExpResult, find_exp, load_settings, run_exp
from .config import out_dir, saving
from .geometry import Geometry
from .style import EXP_COLORS, ORIGIN_RC, origin_plotly, place_legend, write_fit_html
from .tracks import normalize_step


@dataclass(slots=True)
class Scan:
    r: ExpResult
    tmin: np.ndarray  # minutes from the start of heating
    T: np.ndarray
    X: dict[str, np.ndarray]  # peak -> X of every frame (heat, hold, cool)
    t50_time: dict[str, float]  # peak -> time T50 is reached [min]


def _scan(name: str, i: int, config: Path, geo: Geometry, workers: int | None) -> Scan:
    e = find_exp(name, config)
    if e is None:
        raise SystemExit(f"❌ {Path(config).name} 에 실험 '{name}' 없음 (예: x1, exp2.x8)")
    st = load_settings(config, need_exps=False)
    r = run_exp(e, st, geo, workers, EXP_COLORS[i % len(EXP_COLORS)])
    r.name = name  # the legend shows the run too (exp2.x1)
    tab = r.table
    t = pd.to_datetime(tab["time_kst"]).to_numpy("datetime64[s]").astype(float)
    heat = np.flatnonzero(tab["kind"].to_numpy() == "heat")
    t0 = t[heat[0]] if len(heat) else t[0]
    T = tab["T_pv"].to_numpy(float)
    X, t50_time = {}, {}
    for pk in st.peaks:
        A = tab[f"A_{pk}"].to_numpy(float)
        tp = (r.trans or {}).get(pk, {})
        X[pk] = normalize_step(T, A, tp) if tp.get("ok") else tab[f"X_{pk}"].to_numpy(float)
        X[pk][~np.isfinite(T)] = np.nan
        # time T crosses T50 while heating, interpolated over the used frames
        u = tab["used"].to_numpy(bool)
        t50 = r.t50.get(pk, np.nan)
        t50_time[pk] = (
            float(np.interp(t50, T[u], (t[u] - t0) / 60))
            if np.isfinite(t50) and u.sum() > 1
            else np.nan
        )
    return Scan(r, (t - t0) / 60, T, X, t50_time)


def _span(scans, pk, key) -> float:
    """Visible range of the axis, used for label spacing: time = whole measurement, temperature = T50 ± 40 °C."""
    if key == "t":
        return float(max(np.nanmax(s.tmin) for s in scans) - min(np.nanmin(s.tmin) for s in scans))
    t50s = [s.r.t50[pk] for s in scans if np.isfinite(s.r.t50.get(pk, np.nan))]
    return (max(t50s) - min(t50s) + 80) if t50s else 300.0


def _rows(scans, pk, key) -> list[tuple]:
    """[(scan, value, label height)] with labels spread over rows; closer than 7 % of the axis range goes to the next row."""
    vals = [(s, s.r.t50.get(pk, np.nan) if key == "T" else s.t50_time[pk]) for s in scans]
    vals = sorted([v for v in vals if np.isfinite(v[1])], key=lambda v: v[1])
    gap = 0.07 * _span(scans, pk, key)
    out, last = [], {}
    for s, v in vals:
        row = next((k for k in range(3) if v - last.get(k, -np.inf) > gap), len(out) % 3)
        last[row] = v
        out.append((s, v, (1.13, 1.22, 1.31)[row]))
    return out


def plot(scans: list[Scan], peaks: list[str], level: float) -> plt.Figure:
    with plt.rc_context(ORIGIN_RC):
        fig = plt.figure(figsize=(7.2 * len(peaks), 15))
        gs = fig.add_gridspec(3, len(peaks), height_ratios=[0.6, 1, 1], hspace=0.45, wspace=0.2)
        ax_T = fig.add_subplot(gs[0, :])
        for s in scans:
            ax_T.plot(s.tmin, s.T, color=s.r.color, lw=1.6, label=s.r.name)
        ax_T.set(xlabel="Time from heating start (min)", ylabel="T (°C)")
        legend_axes = [ax_T]
        for j, pk in enumerate(peaks):
            ax_t = fig.add_subplot(gs[1, j], sharex=ax_T)
            ax_x = fig.add_subplot(gs[2, j])
            for s in scans:
                t50, tt = s.r.t50.get(pk, np.nan), s.t50_time[pk]
                ax_t.plot(
                    s.tmin,
                    s.X[pk],
                    color=s.r.color,
                    lw=1.3,
                    label=f"{s.r.name}  {tt:.1f} min" if np.isfinite(tt) else s.r.name,
                )
                d = s.r.table[s.r.table["used"]]
                ax_x.plot(
                    d["T_pv"],
                    d[f"X_{pk}"],
                    "-o",
                    color=s.r.color,
                    ms=2.5,
                    lw=1.3,
                    label=f"{s.r.name}  {t50:.0f} °C" if np.isfinite(t50) else s.r.name,
                )
            for ax, key in ((ax_t, "t"), (ax_x, "T")):
                for s, v, ly in _rows(scans, pk, key):
                    ax.axvline(v, color=s.r.color, ls=(0, (4, 2)), lw=1.2, zorder=1)
                    ax.plot(v, level, "D", ms=9, mfc=s.r.color, mec="black", mew=1.1, zorder=6)
                    ax.text(
                        v,
                        ly,
                        f"{v:.0f}" if key == "T" else f"{v:.1f}",
                        ha="center",
                        va="center",
                        fontsize=12.5,
                        fontweight="bold",
                        color=s.r.color,
                        zorder=7,
                        bbox=dict(boxstyle="round,pad=0.2", fc="white", ec=s.r.color, lw=1.1),
                    )
                    if key == "t":
                        ax_T.axvline(v, color=s.r.color, ls=(0, (4, 2)), lw=1.0, zorder=0)
                ax.axhline(level, color="0.6", lw=0.9, ls=(0, (5, 3)), zorder=0)
                ax.set_ylim(-0.1, 1.38)
                ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
            legend_axes += [ax_t, ax_x]
            ax_t.set(xlabel="Time from heating start (min)", ylabel="X (normalised area)")
            ax_t.set_title(f"{pk}  — X vs time  (number = min to T50)", fontsize=15, loc="left")
            ax_x.set(xlabel="PV (°C)", ylabel="X (normalised area)")
            ax_x.set_title(f"{pk}  — X vs T  (number = T50 °C)", fontsize=15, loc="left")
            t50s = [s.r.t50[pk] for s in scans if np.isfinite(s.r.t50.get(pk, np.nan))]
            if t50s:  # only near the transition (± 40 °C)
                ax_x.set_xlim(min(t50s) - 40, max(t50s) + 40)
        fig.suptitle(
            "  vs  ".join(s.r.name for s in scans),
            x=0.01,
            ha="left",
            fontsize=17.5,
            fontweight="bold",
        )
        for i, ax in enumerate(
            legend_axes
        ):  # a legend inside each graph so a screenshot of one graph still shows it
            title = (
                None if i == 0 else ("time to T50" if i % 2 == 1 else "T50")
            )  # order: [T, (X-t, X-T) per peak]
            place_legend(
                ax,
                fontsize=12.5,
                frameon=True,
                framealpha=0.9,
                edgecolor="0.6",
                handlelength=1.4,
                title=title,
                title_fontsize=12.5,
            )
    return fig


# HTML figure size (kept in a larger window, shrunk in a smaller one); also used to size legends
HTML_W, HTML_H = 1250, 1050
HTML_MARGIN = dict(t=110, l=70, r=30, b=60)


def _legend_spot(
    pts: list[tuple[np.ndarray, np.ndarray, float]],
    xr,
    yr,
    labels: list[str],
    dom_x,
    dom_y,
    grid: int = 13,
) -> tuple[float, float]:
    """Free spot for a legend inside one plot (center, paper coordinates). pts = [(x, y, weight)];
    lines are densely interpolated before counting.
    Same idea as place_legend for PNG: the position that covers the fewest points (diamonds and
    number labels weigh heavily)."""
    pw = HTML_W - HTML_MARGIN["l"] - HTML_MARGIN["r"]
    ph = HTML_H - HTML_MARGIN["t"] - HTML_MARGIN["b"]
    bw = (max(len(s) for s in labels) * 8.6 + 60) / (
        pw * (dom_x[1] - dom_x[0])
    )  # as a fraction of the axis (15 px text)
    bh = (len(labels) * 26 + 14) / (ph * (dom_y[1] - dom_y[0]))
    P, W = [], []
    for x, y, w in pts:
        u = (np.asarray(x, float) - xr[0]) / (xr[1] - xr[0])
        v = (np.asarray(y, float) - yr[0]) / (yr[1] - yr[0])
        ok = np.isfinite(u) & np.isfinite(v)
        u, v = u[ok], v[ok]
        # line: interpolate at 0.004 spacing so a mere crossing is counted
        if len(u) > 1 and w <= 1:
            seg = np.hypot(np.diff(u), np.diff(v))
            k = np.minimum(np.maximum(1, (seg / 0.004).astype(int)), 100)
            s = np.repeat(np.arange(len(seg)), k)
            f = np.concatenate([np.arange(n) / n for n in k])
            u, v = u[s] + np.diff(u)[s] * f, v[s] + np.diff(v)[s] * f
        P.append(np.column_stack([u, v]))
        W.append(np.full(len(u), w))
    P = np.vstack(P) if P else np.empty((0, 2))
    W = np.concatenate(W) if W else np.empty(0)
    best = None
    xs = np.linspace(bw / 2 + 0.01, 1 - bw / 2 - 0.01, grid) if bw < 0.98 else [0.5]
    ys = np.linspace(bh / 2 + 0.01, 1 - bh / 2 - 0.01, grid) if bh < 0.98 else [0.5]
    for fy in ys[::-1]:  # on equal scores prefer upper right
        for fx in xs[::-1]:
            inside = (np.abs(P[:, 0] - fx) <= bw / 2) & (np.abs(P[:, 1] - fy) <= bh / 2)
            score = float(W[inside].sum())
            if best is None or score < best[0]:
                best = (score, fx, fy)
    _, fx, fy = best
    return dom_x[0] + fx * (dom_x[1] - dom_x[0]), dom_y[0] + fy * (dom_y[1] - dom_y[0])


def write_html(scans: list[Scan], peaks: list[str], level: float, path: Path) -> Path:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    n = len(peaks)
    fig = make_subplots(
        rows=3,
        cols=n,
        row_heights=[0.26, 0.37, 0.37],
        vertical_spacing=0.115,
        horizontal_spacing=0.07,
        specs=[[{"colspan": n}] + [None] * (n - 1)] + [[{}] * n] * 2,
        subplot_titles=["Temperature"]
        + [f"{pk} — X vs time" for pk in peaks]
        + [f"{pk} — X vs T" for pk in peaks],
    )
    # one legend per plot: legend (temperature), legend{2j} (X vs time), legend{2j+1} (X vs T)
    lg = {
        (1, 1): "legend",
        **{(2, j): f"legend{2 * j}" for j in range(1, n + 1)},
        **{(3, j): f"legend{2 * j + 1}" for j in range(1, n + 1)},
    }
    panel = {k: dict(pts=[], labels=[]) for k in lg}
    t_all = np.concatenate([s.tmin for s in scans])
    T_all = np.concatenate([s.T for s in scans])
    xr_t = (float(np.nanmin(t_all)), float(np.nanmax(t_all)))
    pad = 0.05 * (np.nanmax(T_all) - np.nanmin(T_all))
    panel[(1, 1)].update(xr=xr_t, yr=(float(np.nanmin(T_all) - pad), float(np.nanmax(T_all) + pad)))
    for j, pk in enumerate(peaks, start=1):
        t50s = [s.r.t50[pk] for s in scans if np.isfinite(s.r.t50.get(pk, np.nan))]
        xr_T = (min(t50s) - 40, max(t50s) + 40) if t50s else (250.0, 520.0)
        panel[(2, j)].update(xr=xr_t, yr=(-0.1, 1.38))
        panel[(3, j)].update(xr=xr_T, yr=(-0.1, 1.38))
        fig.update_xaxes(range=list(xr_T), row=3, col=j)

    def add(trace, row, col, label):
        fig.add_trace(trace.update(name=label, legend=lg[(row, col)], showlegend=True), row, col)
        panel[(row, col)]["labels"].append(label)
        panel[(row, col)]["pts"].append((trace.x, trace.y, 1.0))

    for s in scans:
        c, nm = s.r.color, s.r.name
        add(
            go.Scatter(
                x=s.tmin,
                y=s.T,
                mode="lines",
                legendgroup=nm,
                line=dict(color=c, width=2),
                hovertemplate=f"<b>{nm}</b><br>%{{x:.1f}} min · %{{y:.1f}} °C<extra></extra>",
            ),
            1,
            1,
            nm,
        )
        for j, pk in enumerate(peaks, start=1):
            t50, tt = s.r.t50.get(pk, np.nan), s.t50_time[pk]
            add(
                go.Scatter(
                    x=s.tmin,
                    y=s.X[pk],
                    mode="lines",
                    legendgroup=nm,
                    line=dict(color=c, width=1.6),
                    hovertemplate=f"<b>{nm} {pk}</b><br>%{{x:.1f}} min · X %{{y:.3f}}<extra></extra>",
                ),
                2,
                j,
                f"{nm}  {tt:.1f} min" if np.isfinite(tt) else nm,
            )
            d = s.r.table[s.r.table["used"]]
            add(
                go.Scatter(
                    x=d["T_pv"],
                    y=d[f"X_{pk}"],
                    mode="lines+markers",
                    legendgroup=nm,
                    marker=dict(size=4, color=c),
                    line=dict(color=c, width=1.5),
                    hovertemplate=f"<b>{nm} {pk}</b><br>%{{x:.1f}} °C · X %{{y:.3f}}<extra></extra>",
                ),
                3,
                j,
                f"{nm}  T50 {t50:.0f} °C" if np.isfinite(t50) else nm,
            )
    for j, pk in enumerate(peaks, start=1):
        for row, key in ((2, "t"), (3, "T")):
            for s, v, ly in _rows(scans, pk, key):
                panel[(row, j)]["pts"] += [
                    ([v], [level], 60.0),
                    ([v], [ly], 1000.0),
                ]  # keep the legend off the diamond and number label
                c = s.r.color
                rows = (1, row) if key == "t" else (row,)
                for rr in rows:
                    fig.add_vline(
                        x=v,
                        line=dict(color=c, dash="dash", width=1.3),
                        row=rr,
                        col=1 if rr == 1 else j,
                    )
                txt = f"{v:.0f} °C" if key == "T" else f"{v:.1f} min"
                fig.add_trace(
                    go.Scatter(
                        x=[v],
                        y=[level],
                        mode="markers",
                        legendgroup=s.r.name,
                        showlegend=False,
                        marker=dict(
                            symbol="diamond", size=13, color=c, line=dict(color="black", width=1.2)
                        ),
                        hovertemplate=f"<b>{s.r.name} {pk}  T50</b><br>{txt}<extra></extra>",
                    ),
                    row,
                    j,
                )
                fig.add_annotation(
                    x=v,
                    y=ly,
                    text=f"<b>{v:.0f}</b>" if key == "T" else f"<b>{v:.1f}</b>",
                    showarrow=False,
                    row=row,
                    col=j,
                    font=dict(color=c, size=15),
                    bgcolor="white",
                    bordercolor=c,
                    borderwidth=1.2,
                    borderpad=2,
                )
            fig.add_hline(y=level, line=dict(color="#999", dash="dash", width=1), row=row, col=j)
            fig.update_yaxes(
                range=[-0.1, 1.38], tickvals=[0, 0.2, 0.4, 0.6, 0.8, 1.0], row=row, col=j
            )
    legends = {}
    for (row, col), p in panel.items():
        ax = fig.get_subplot(row, col)
        x, y = _legend_spot(
            p["pts"], p["xr"], p["yr"], p["labels"], ax.xaxis.domain, ax.yaxis.domain
        )
        legends[lg[(row, col)]] = dict(
            x=x,
            y=y,
            xanchor="center",
            yanchor="middle",
            font=dict(size=15),
            bgcolor="rgba(255,255,255,0.88)",
            bordercolor="#999",
            borderwidth=1,
        )
    fig.update_layout(
        template="simple_white",
        font=dict(family="Arial", size=16),
        title=dict(text="<b>" + "  vs  ".join(s.r.name for s in scans) + "</b>", x=0.01),
        margin=HTML_MARGIN,
        **legends,
    )
    origin_plotly(fig)
    fig.update_xaxes(title="Time from heating start (min)", row=1)
    fig.update_xaxes(title="Time from heating start (min)", matches="x", row=2)
    fig.update_xaxes(title="PV (°C)", row=3)
    fig.update_yaxes(title="T (°C)", row=1, col=1)
    fig.update_yaxes(title="X", col=1, row=2)
    fig.update_yaxes(title="X", col=1, row=3)
    return write_fit_html(fig, path, HTML_W, HTML_H)


def run(
    names: list[str],
    config: Path = DEFAULT_CONFIG,
    geo: Geometry = Geometry(),
    workers: int | None = None,
    show: bool = True,
) -> list[Scan]:
    st = load_settings(config, need_exps=False)
    scans = [_scan(nm, i, config, geo, workers) for i, nm in enumerate(names)]
    peaks = list(st.peaks)
    print("\n📊 T50 (°C) / 승온 시작부터 T50 까지 (min)")
    for s in scans:
        print(
            f"   {s.r.name:>10}: "
            + "   ".join(
                f"{pk} {s.r.t50.get(pk, np.nan):6.1f} °C · {s.t50_time[pk]:5.1f} min"
                for pk in peaks
            )
        )
    d = out_dir("overlay")
    stem = "+".join(n.replace(".", "_") for n in names)
    page = write_html(scans, peaks, st.level, d / f"{stem}.html")
    if saving():
        fig = plot(scans, peaks, st.level)
        png = d / f"{stem}.png"
        with plt.rc_context(ORIGIN_RC):
            fig.savefig(png)
        plt.close(fig)
        print(f"\n💾 {png}\n💾 {page}")
    if show:
        webbrowser.open_new(page.resolve().as_uri())
    return scans
