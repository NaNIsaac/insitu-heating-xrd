"""단일 이미지 → 2θ 프로파일 (중앙 스트립 / 전체 링 평균) + 피크 피팅.
브라우저 HTML (다른 명령과 같은 스타일), PNG 는 --save 일 때."""

import webbrowser
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from .config import out_dir, saving
from .geometry import Geometry
from .integrate import ProfileOptions, RadialIntegrator, strip_bounds
from .io import load_image
from .peaks import detect_peaks, fit_peak
from .style import ORIGIN_RC


def analyze(
    img: np.ndarray,
    geo: Geometry,
    strip_width: int,
    tth_bin: float,
    smooth: int,
    prominence: float | None,
    min_distance: float,
) -> dict:
    """{영역 이름: (2θ, 세기, 다듬은 세기, 피크 인덱스, [피팅 결과 dict])}."""
    tth = geo.tth_map(img.shape)
    x0, x1 = strip_bounds(img.shape[1], strip_width)
    out = {}
    for name, region in ((f"strip x={x0}–{x1 - 1}", "strip"), ("full image (ring avg)", "full")):
        opt = ProfileOptions(tth_bin=tth_bin, region=region, strip_width=strip_width, min_fill=0)
        integ = RadialIntegrator.build(geo, opt, img.shape, tth)
        x, y = integ.centers, integ(img)
        ys, pk = detect_peaks(x, y, prominence, min_distance, smooth)
        fits = []
        for p in pk:
            f = fit_peak(x, y, x[p])
            c = f.center if np.isfinite(f.center) else x[p]
            fits.append(
                dict(
                    max=x[p],
                    fit=f.center,
                    err=f.center_err,
                    fwhm=f.fwhm,
                    I=ys[p],
                    q=geo.tth_to_q(c),
                    d=geo.d_spacing(c),
                )
            )
        out[name] = (x, y, ys, pk, fits)
    return dict(tth=tth, strip=(x0, x1), regions=out)


def write_html(img: np.ndarray, r: dict, file: Path, sub: str, path: Path) -> Path:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    from .style import LEGEND, header, origin_plotly, write_fit_html

    names = list(r["regions"])
    fig = make_subplots(
        rows=1,
        cols=3,
        column_widths=[0.26, 0.37, 0.37],
        horizontal_spacing=0.07,
        subplot_titles=["<b>detector</b>  (red: strip)"]
        + [f"<b>{n}</b>  ·  {len(r['regions'][n][3])} peaks" for n in names],
    )
    k = 2
    h, w = (img.shape[0] // k) * k, (img.shape[1] // k) * k
    with np.errstate(all="ignore"):
        d = np.nanmean(img[:h, :w].reshape(h // k, k, w // k, k), axis=(1, 3)).astype(np.float32)
        t = np.nanmean(r["tth"][:h, :w].reshape(h // k, k, w // k, k), axis=(1, 3)).astype(
            np.float32
        )
    xs, ys_ = np.arange(w // k) * k + 0.5, np.arange(h // k) * k + 0.5
    fig.add_trace(
        go.Heatmap(
            x=xs,
            y=ys_,
            z=d,
            customdata=t,
            colorscale="Inferno",
            showscale=False,
            zmin=float(np.nanpercentile(img, 1)),
            zmax=float(np.nanpercentile(img, 99.5)),
            hovertemplate="x %{x:.0f} · y %{y:.0f}<br>2θ %{customdata:.3f}° · I %{z:.1f}<extra></extra>",
        ),
        1,
        1,
    )
    x0, x1 = r["strip"]
    fig.add_vrect(x0=x0, x1=x1, fillcolor="red", opacity=0.35, line_width=0, row=1, col=1)
    for j, n in enumerate(names, start=2):
        x, y, ys, pk, fits = r["regions"][n]
        fig.add_trace(
            go.Scatter(
                x=x,
                y=y,
                mode="lines",
                name="raw",
                line=dict(color="#aaaaaa", width=1.2),
                legendgroup="raw",
                showlegend=j == 2,
                hovertemplate="2θ %{x:.3f}° · I %{y:.2f}<extra></extra>",
            ),
            1,
            j,
        )
        fig.add_trace(
            go.Scatter(
                x=x,
                y=ys,
                mode="lines",
                name="smoothed",
                line=dict(color="black", width=2.2),
                legendgroup="sm",
                showlegend=j == 2,
                hoverinfo="skip",
            ),
            1,
            j,
        )
        txt = [f"{f['fit']:.3f}°" if np.isfinite(f["fit"]) else f"{f['max']:.2f}°" for f in fits]
        fig.add_trace(
            go.Scatter(
                x=x[pk],
                y=ys[pk],
                mode="markers+text",
                text=txt,
                textposition="top center",
                textfont=dict(color="#b00020", size=14),
                showlegend=False,
                marker=dict(symbol="x", size=11, color="#b00020", line=dict(width=2)),
                customdata=[[f["fwhm"], f["d"], f["q"]] for f in fits],
                hovertemplate="2θ %{x:.3f}°<br>FWHM %{customdata[0]:.3f}° · d %{customdata[1]:.4f} Å"
                " · q %{customdata[2]:.4f} Å⁻¹<extra></extra>",
            ),
            1,
            j,
        )
        table = "<br>".join(
            f"<b>{f['fit']:.3f}°</b>  FWHM {f['fwhm']:.3f}°  d {f['d']:.4f} Å"
            for f in fits
            if np.isfinite(f["fit"])
        )
        if table:
            fig.add_annotation(
                x=0.98,
                y=0.97,
                xref=f"x{j} domain",
                yref=f"y{j} domain",
                text=table,
                align="right",
                showarrow=False,
                xanchor="right",
                yanchor="top",
                font=dict(size=14),
                bgcolor="rgba(255,255,255,0.85)",
                bordercolor="#999",
                borderwidth=1,
            )
        fig.update_xaxes(title="2θ (°)", row=1, col=j)
        fig.update_yaxes(title="mean I", row=1, col=j)
    header(fig, f"{file.parent.name} / {file.name}", sub)
    origin_plotly(fig)
    fig.update_yaxes(range=[h - 0.5, -0.5], title="y (px)", minor=dict(ticks=""), row=1, col=1)
    fig.update_xaxes(range=[-0.5, w - 0.5], title="x (px)", minor=dict(ticks=""), row=1, col=1)
    fig.update_layout(
        margin=dict(t=95, l=85, r=30, b=75),
        legend=dict(x=0.63, xanchor="right", y=0.02, yanchor="bottom", **LEGEND),
    )
    return write_fit_html(fig, path, 1550, 620)


def run(
    file: Path,
    geo: Geometry = Geometry(),
    strip_width: int = 10,
    tth_bin: float = 0.02,
    smooth: int = 7,
    prominence: float | None = None,
    min_distance: float = 0.15,
    save: bool | None = None,
    show: bool = True,
    sub: str = "",
) -> Path:
    img = load_image(file)
    r = analyze(img, geo, strip_width, tth_bin, smooth, prominence, min_distance)
    for name, (x, y, ys, pk, fits) in r["regions"].items():
        print(f"\n[{name}]")
        for f in fits:
            print(
                f"  2θ(max) = {f['max']:8.3f}°  2θ(fit) = {f['fit']:8.4f} ± {f['err']:.4f}°"
                f"  FWHM = {f['fwhm']:.3f}°  I = {f['I']:.2f}  q = {f['q']:.4f} Å⁻¹  d = {f['d']:.4f} Å"
            )
    page = write_html(
        img, r, file, sub, out_dir(file.parent.name) / f"{file.stem}_tth_profile.html"
    )
    if saving() if save is None else save:
        with plt.rc_context(ORIGIN_RC):
            fig, axes = plt.subplots(
                1, 3, figsize=(17, 5.5), gridspec_kw={"width_ratios": [1, 1.4, 1.4]}
            )
            axes[0].imshow(
                img, vmin=0, vmax=np.nanpercentile(img, 99.5), aspect="auto", cmap="inferno"
            )
            axes[0].axvspan(*r["strip"], color="r", alpha=0.3)
            axes[0].set_title(file.name, fontsize=13)
            for ax, (name, (x, y, ys, pk, fits)) in zip(axes[1:], r["regions"].items()):
                ax.plot(x, y, color="0.65", lw=0.8, label="raw")
                ax.plot(x, ys, color="k", lw=1.6, label="smoothed")
                ax.plot(x[pk], ys[pk], "x", color="#b00020", ms=9, mew=2)
                for p in pk:
                    ax.annotate(
                        f"{x[p]:.2f}",
                        (x[p], ys[p]),
                        xytext=(0, 6),
                        textcoords="offset points",
                        ha="center",
                        fontsize=11,
                        color="#b00020",
                    )
                ax.set(xlabel="2θ (°)", ylabel="mean I")
                ax.set_title(f"{name}: {len(pk)} peaks", fontsize=13)
                ax.legend(fontsize=11)
            fig.tight_layout()
            png = out_dir(file.parent.name) / f"{file.stem}_tth_profile.png"
            fig.savefig(png, dpi=150)
            plt.close(fig)
        print(f"\n💾 {png}\n💾 {page}")
    if show:
        webbrowser.open_new(page.resolve().as_uri())
    return page
