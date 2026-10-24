"""One folder: 2θ heatmap, peak intensity and FWHM over time, without temperature (for checking during a measurement).

    uv run main.py heatmap <folder>             everything recorded so far
    uv run main.py heatmap <folder> --watch 30  pick up new files every 30 s (ends when the window closes)

Peak intensity is the mean of the raw intensity I within peak center ± max(0.15°, FWHM/4),
as seen in the heatmap. The value box shows the current intensity (initial intensity), the change
over the last N minutes (% of the total rise), FWHM and center. A recent change near 0 % means
saturation (the phase transition is over). The integration cache appends only new files, so
reruns are fast.
"""

from __future__ import annotations

import time
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from .compare import DEFAULT_CONFIG, load_settings, peak_areas, smooth
from .config import out_dir, saving
from .geometry import Geometry
from .heatmap import Config, beam_ok, color_range, integrate_folder
from .integrate import ProfileOptions
from .style import ORIGIN_RC, TRACK_COLORS
from .temperature import fmt_kst


def compute(folder: Path, geo: Geometry, workers: int | None, config: Path = DEFAULT_CONFIG):
    st = load_settings(config, need_exps=False)
    x, Z, t, names = integrate_folder(
        Config(
            folder=folder, geometry=geo, workers=workers, profile=ProfileOptions(tth_bin=st.tth_bin)
        )
    )
    ok = beam_ok(Z)
    idx = np.flatnonzero(ok)
    n = max(1, min(st.n_norm, len(idx) // 4))
    win, areas, shapes = peak_areas(x, Z, idx[:n], idx[-n:], st, say=lambda *a: None, widths=True)
    # raw I rather than area: the area uses R (divided by the amorphous), so it can fall while the
    # peak grows if the amorphous halo shrinks; the max of a wide window can pick up the halo at its edge
    inten, band = {}, {}
    for pk, (c, fwhm, _, _) in win.items():
        hw = max(0.15, fwhm / 4 if np.isfinite(fwhm) else 0.15)
        m = np.abs(x - c) <= hw
        band[pk] = (c - hw, c + hw)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            I = np.nanmean(Z[:, m], axis=1) if m.any() else np.full(len(Z), np.nan)
        I[~ok] = np.nan
        inten[pk] = I
    return dict(
        x=x, Z=Z, t=t, names=names, ok=ok, win=win, band=band, inten=inten, shapes=shapes, n_norm=n
    )


def _recent(tm: np.ndarray, A: np.ndarray, minutes: float, a0: float) -> tuple[float, float]:
    """(current value, change over the last minutes / total rise [%]); change is the line-fit slope times minutes."""
    m = np.isfinite(A)
    if m.sum() < 3:
        return np.nan, np.nan
    now = float(np.median(A[m][-5:]))
    r = m & (tm >= tm[m][-1] - minutes)
    if r.sum() < 3 or not np.isfinite(a0) or now == a0:
        return now, np.nan
    slope = np.polyfit(tm[r], A[r], 1)[0]
    return now, float(100 * slope * minutes / abs(now - a0))


def draw(
    fig: plt.Figure,
    d: dict,
    folder: Path,
    recent: float = 5.0,
    clip: tuple[float, float] = (25.0, 99.7),
    cmap: str = "inferno",
) -> None:
    fig.clf()
    x, Z, t, ok = d["x"], d["Z"], d["t"], d["ok"]
    tm = (t - t[0]) / 60.0
    gs = fig.add_gridspec(
        3, 2, height_ratios=[1, 1.9, 1.5], width_ratios=[1, 0.022], hspace=0.08, wspace=0.03
    )
    ax_w = fig.add_subplot(gs[0, 0])
    ax_h = fig.add_subplot(gs[1, 0], sharex=ax_w)
    ax_a = fig.add_subplot(gs[2, 0], sharex=ax_w)
    cax = fig.add_subplot(gs[1, 1])

    Zs = np.where(ok[:, None], Z, np.nan)
    lo, hi = color_range(Zs, clip)
    dt = np.median(np.diff(tm)) if len(tm) > 1 else 1.0
    gap = np.flatnonzero(np.diff(tm) > 3 * dt)
    tg = np.insert(tm, gap + 1, tm[gap] + dt)  # one empty frame at each acquisition gap
    Zg = np.insert(Zs, gap + 1, np.nan, axis=0)
    edges = np.r_[tg[0] - dt / 2, (tg[1:] + tg[:-1]) / 2, tg[-1] + dt / 2]
    xe = np.r_[x[0] - (x[1] - x[0]) / 2, (x[1:] + x[:-1]) / 2, x[-1] + (x[1] - x[0]) / 2]
    m = ax_h.pcolormesh(
        edges, xe, Zg.T, cmap=cmap, vmin=lo, vmax=hi, shading="flat", rasterized=True
    )
    fig.colorbar(m, cax=cax).set_label("I")
    ax_h.set_ylabel("2θ (deg)")

    lines = []
    for i, (pk, A) in enumerate(d["inten"].items()):
        c = TRACK_COLORS[i % len(TRACK_COLORS)]
        cen, (lo_w, hi_w) = d["win"][pk][0], d["band"][pk]  # band over which intensity is measured
        for v in (lo_w, hi_w):
            ax_h.axhline(v, color=c, lw=1.2, ls="--", alpha=0.9)
        ax_h.text(
            1.0,
            (hi_w - x[0]) / (x[-1] - x[0]),
            f" {pk}",
            transform=ax_h.transAxes,
            color=c,
            fontsize=13,
            va="bottom",
            ha="right",
            fontweight="bold",
        )
        ax_a.plot(tm, A, "o", color=c, ms=2.5, alpha=0.35, mec="none")
        ax_a.plot(tm, smooth(A, 5), color=c, lw=2.2, label=pk)
        center, width = d["shapes"][pk]
        ax_w.plot(tm, width, "o", color=c, ms=2.5, alpha=0.35, mec="none")
        ax_w.plot(tm, smooth(width, 5), color=c, lw=2.0)
        a0 = float(np.nanmedian(A[np.flatnonzero(ok)[: d["n_norm"]]]))
        now, rate = _recent(tm, A, recent, a0)
        wm, cm = np.isfinite(width), np.isfinite(center)
        w_now = np.median(width[wm][-5:]) if wm.any() else np.nan
        c_now = np.median(center[cm][-5:]) if cm.any() else cen
        lines.append(
            (
                c,
                f"{pk}  I {now:.1f}  (start {a0:.1f})  |  last {recent:g} min {rate:+.1f} %  |  "
                f"FWHM {w_now:.3f}°  |  2θ {c_now:.3f}°",
            )
        )
    ax_a.set_ylabel("peak intensity\n(I at center)")
    ax_a.set_xlabel("time from first frame (min)")
    ax_w.set_ylabel("FWHM (deg)")
    for a in (ax_w, ax_h):
        a.tick_params(labelbottom=False)
    ax_a.set_xlim(tm[0] - dt / 2, tm[-1] + dt / 2)
    for j, (c, s) in enumerate(lines):
        fig.text(0.07, 0.95 - 0.04 * j, s, color=c, fontsize=15, fontweight="bold", va="top")
    fig.suptitle(
        f"{folder.name}   {len(t)} frames   last {fmt_kst(t[-1])[11:19]}   ({tm[-1]:.1f} min)",
        x=0.07,
        ha="left",
        y=0.995,
        fontsize=17,
        fontweight="bold",
    )
    fig.subplots_adjust(left=0.07, right=0.93, bottom=0.07, top=0.93 - 0.04 * len(lines))


def peak_stats(d: dict, recent: float) -> list[dict]:
    """Current value per peak: intensity (initial), change over the last recent minutes [%], FWHM, 2θ."""
    tm = (d["t"] - d["t"][0]) / 60.0
    out = []
    for i, (pk, A) in enumerate(d["inten"].items()):
        center, width = d["shapes"][pk]
        a0 = float(np.nanmedian(A[np.flatnonzero(d["ok"])[: d["n_norm"]]]))
        now, rate = _recent(tm, A, recent, a0)
        wm, cm = np.isfinite(width), np.isfinite(center)
        out.append(
            dict(
                pk=pk,
                color=TRACK_COLORS[i % len(TRACK_COLORS)],
                I=now,
                I0=a0,
                rate=rate,
                fwhm=np.median(width[wm][-5:]) if wm.any() else np.nan,
                center=np.median(center[cm][-5:]) if cm.any() else d["win"][pk][0],
            )
        )
    return out


def write_html(
    d: dict,
    folder: Path,
    path: Path,
    recent: float = 5.0,
    clip: tuple[float, float] = (25.0, 99.7),
    cmap: str = "inferno",
    refresh: float = 0,
) -> Path:
    """HTML page: per-peak current values on top, then FWHM(t), the 2θ heatmap and peak intensity(t)."""
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    from .style import LEGEND, header, origin_plotly, write_fit_html

    x, Z, t, ok = d["x"], d["Z"], d["t"], d["ok"]
    tm = (t - t[0]) / 60.0
    stats = peak_stats(d, recent)
    fig = make_subplots(
        rows=3, cols=1, shared_xaxes=True, row_heights=[0.24, 0.44, 0.32], vertical_spacing=0.035
    )
    Zs = np.where(ok[:, None], Z, np.nan)
    lo, hi = color_range(Zs, clip)
    dt = np.median(np.diff(tm)) if len(tm) > 1 else 1.0
    gap = np.flatnonzero(np.diff(tm) > 3 * dt)
    tg = np.insert(tm, gap + 1, tm[gap] + dt)  # empty frame at each acquisition gap
    Zg = np.insert(Zs, gap + 1, np.nan, axis=0)
    names = np.insert(d["names"].astype(object), gap + 1, "")
    fig.add_trace(
        go.Heatmap(
            x=tg,
            y=x,
            z=Zg.T,
            colorscale=cmap.capitalize() if cmap == "inferno" else cmap,
            zmin=lo,
            zmax=hi,
            customdata=np.broadcast_to(names, Zg.T.shape),
            colorbar=dict(
                title=dict(text="Intensity", side="right"),
                len=0.42,
                y=0.53,
                thickness=18,
                outlinewidth=1.5,
                tickfont=dict(size=15),
            ),
            hovertemplate="%{x:.2f} min · 2θ %{y:.3f}° · I %{z:.1f}<br>%{customdata}<extra></extra>",
        ),
        2,
        1,
    )
    for s_ in stats:
        pk, c = s_["pk"], s_["color"]
        A = d["inten"][pk]
        center, width = d["shapes"][pk]
        for v in d["band"][pk]:
            fig.add_hline(y=v, line=dict(color=c, dash="dash", width=1.5), row=2, col=1)
        fig.add_annotation(
            x=1.0,
            y=d["band"][pk][1],
            xref="x2 domain",
            yref="y2",
            text=f"<b>{pk}</b>",
            showarrow=False,
            xanchor="right",
            yanchor="bottom",
            font=dict(color=c, size=15),
        )
        fig.add_trace(
            go.Scatter(
                x=tm,
                y=width,
                mode="markers",
                marker=dict(color=c, size=4, opacity=0.3),
                legendgroup=pk,
                showlegend=False,
                hoverinfo="skip",
            ),
            1,
            1,
        )
        fig.add_trace(
            go.Scatter(
                x=tm,
                y=smooth(width, 5),
                mode="lines",
                line=dict(color=c, width=2.4),
                legendgroup=pk,
                showlegend=False,
                hovertemplate=f"<b>{pk}</b> %{{x:.2f}} min · FWHM %{{y:.3f}}°<extra></extra>",
            ),
            1,
            1,
        )
        fig.add_trace(
            go.Scatter(
                x=tm,
                y=A,
                mode="markers",
                marker=dict(color=c, size=4, opacity=0.35),
                legendgroup=pk,
                showlegend=False,
                hoverinfo="skip",
            ),
            3,
            1,
        )
        fig.add_trace(
            go.Scatter(
                x=tm,
                y=smooth(A, 5),
                mode="lines",
                line=dict(color=c, width=2.6),
                name=pk,
                legendgroup=pk,
                hovertemplate=f"<b>{pk}</b> %{{x:.2f}} min · I %{{y:.2f}}<extra></extra>",
            ),
            3,
            1,
        )
    lines = [
        f"<span style='color:{s_['color']}'><b>{s_['pk']}</b>   I <b>{s_['I']:.1f}</b> (start {s_['I0']:.1f})"
        f"   ·   last {recent:g} min <b>{s_['rate']:+.1f} %</b>   ·   FWHM {s_['fwhm']:.3f}°"
        f"   ·   2θ {s_['center']:.3f}°</span>"
        for s_ in stats
    ]
    fig.add_annotation(
        x=0,
        y=1.0,
        xref="paper",
        yref="paper",
        xanchor="left",
        yanchor="bottom",
        align="left",
        showarrow=False,
        text="<br>".join(lines),
        font=dict(size=18),
        yshift=12,
    )
    header(
        fig,
        folder.name,
        f"{len(t)} frames · last {fmt_kst(t[-1])[11:19]} KST · {tm[-1]:.1f} min"
        + (f" · auto refresh {refresh:g} s" if refresh else ""),
    )
    origin_plotly(fig)
    fig.update_yaxes(title="FWHM (°)", row=1, col=1)
    fig.update_yaxes(title="2θ (°)", row=2, col=1, minor=dict(ticks=""))
    fig.update_xaxes(minor=dict(ticks=""), row=2, col=1)
    fig.update_yaxes(title="peak I (center)", row=3, col=1)
    fig.update_xaxes(
        title="time from first frame (min)", range=[tg[0] - dt / 2, tg[-1] + dt / 2], row=3, col=1
    )
    top = 70 + 30 * len(lines)
    fig.update_layout(
        margin=dict(t=top + 20, l=95, r=40, b=70),
        legend=dict(x=0.995, xanchor="right", y=0.005, yanchor="bottom", **LEGEND),
    )
    return write_fit_html(fig, path, 1300, 860 + 30 * len(lines), refresh=refresh)


def run(
    folder: Path,
    geo: Geometry = Geometry(),
    workers: int | None = None,
    watch: float = 0,
    recent: float = 5.0,
    show: bool = True,
    config: Path = DEFAULT_CONFIG,
    clip: tuple[float, float] = (25.0, 99.7),
    cmap: str = "inferno",
) -> None:
    import webbrowser

    d = compute(folder, geo, workers, config)
    for s_ in peak_stats(d, recent):
        print(
            f"   {s_['pk']:>6}: I {s_['I']:.1f} (start {s_['I0']:.1f})  last {recent:g} min {s_['rate']:+.1f} %"
            f"  FWHM {s_['fwhm']:.3f}°  2θ {s_['center']:.3f}°"
        )
    page = write_html(
        d,
        folder,
        out_dir(folder.name) / f"{folder.name}_heatmap.html",
        recent,
        clip,
        cmap,
        refresh=watch,
    )
    if saving():
        with plt.rc_context(ORIGIN_RC):
            fig = plt.figure(figsize=(15, 10.5))
            draw(fig, d, folder, recent, clip, cmap)
            png = out_dir(folder.name) / f"{folder.name}_heatmap.png"
            fig.savefig(png, dpi=150)
            plt.close(fig)
        print(f"💾 {png}\n💾 {page}")
    if show:
        webbrowser.open_new(page.resolve().as_uri())
    if watch <= 0:
        return
    print(f"⏱️  {watch:g} 초마다 새 파일 반영 — 브라우저가 알아서 새로 읽음 (끝내려면 Ctrl+C)")
    try:
        while True:
            time.sleep(watch)
            n_old = len(d["t"])
            d = compute(folder, geo, workers, config)
            if len(d["t"]) != n_old:
                write_html(d, folder, page, recent, clip, cmap, refresh=watch)
    except KeyboardInterrupt:
        print("⏹️  끝")
