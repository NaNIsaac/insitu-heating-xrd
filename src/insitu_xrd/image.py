"""View one h5 image: raw detector image (left) | image unwrapped to 2θ–χ plus the χ-averaged 2θ profile (right).

Unwrapped image: pixels rebinned by (2θ, χ) and averaged per bin. A ring becomes a vertical line;
if intensity along the ring is uneven (texture, large-grain spots) the line breaks or turns to dots.
χ is the azimuth measured in the plane perpendicular to the beam, 0 at the vertical center line of
the detector (the scattering plane).
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm, Normalize

from .config import SETTINGS_FILE, out_dir, saving
from .geometry import FArray, Geometry
from .io import list_files, read_frame
from .style import ORIGIN_RC
from .temperature import TempLog, find_log, fmt_kst


def chi_map(geo: Geometry, shape: tuple[int, int]) -> FArray:
    """Azimuth χ of each pixel [deg], in the same lab frame as tth_map (beam along z)."""
    y, x = np.mgrid[: shape[0], : shape[1]]
    u = (x - geo.xc) * geo.pixel
    v = (geo.yc - y) * geo.pixel
    a = np.deg2rad(geo.alpha_deg)
    py = geo.sdd * np.sin(a) + v * np.cos(a)
    return np.rad2deg(np.arctan2(u, py))


def cake(img: FArray, tth: FArray, chi: FArray, tth_bin: float, chi_bin: float):
    """Image rebinned on a (2θ, χ) grid and the χ-averaged 2θ profile; bins with no pixel are NaN.
    The profile is weighted by pixel count; 2θ columns with under half the median count (across
    a gap) are dropped so they do not produce false spikes."""
    ok = np.isfinite(img)
    te = np.arange(np.nanmin(tth), np.nanmax(tth) + tth_bin, tth_bin)
    ce = np.arange(np.nanmin(chi), np.nanmax(chi) + chi_bin, chi_bin)
    s, _, _ = np.histogram2d(chi[ok], tth[ok], bins=(ce, te), weights=img[ok])
    n, _, _ = np.histogram2d(chi[ok], tth[ok], bins=(ce, te))
    m = n.sum(axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        prof = s.sum(axis=0) / m
        prof[m < 0.5 * np.median(m[m > 0])] = np.nan
        return te, ce, s / n, prof


def resolve(target: str, index: int) -> tuple[Path, int, int, list[Path]]:
    """Folder, .h5 file or experiments.toml experiment name to (file, index, total, temperature logs)."""
    p = Path(target)
    if p.suffix.lower() == ".h5" and p.is_file():
        files = sorted(p.parent.glob("*.h5"))
        return p, files.index(p) if p in files else 0, len(files), []
    if p.is_dir():
        files, logs = list_files(p), []
    else:
        from .compare import exp_names, find_exp

        e = find_exp(target)
        if e is None:
            raise SystemExit(
                f"❌ 없음: {target}  (.h5 파일 · 이미지 폴더 · {SETTINGS_FILE.name} 의 실험 이름)\n"
                f"   있는 이름: {exp_names()}"
            )
        # with several folders the index runs across all of them
        files = [f for d in e.images for f in list_files(d)]
        logs = e.recipe
    i = index if index >= 0 else len(files) + index
    if not 0 <= i < len(files):
        raise SystemExit(f"❌ 번호 {index} 범위 밖 (0 ~ {len(files) - 1}, 음수는 뒤에서부터)")
    return files[i], i, len(files), logs


def plot(
    img: FArray,
    tth: FArray,
    chi: FArray,
    te: FArray,
    ce: FArray,
    Z: FArray,
    prof: FArray,
    title: str,
    log: bool,
    clip: tuple[float, float],
    peaks: dict[str, float],
    cmap: str,
) -> plt.Figure:
    lo, hi = np.nanpercentile(img, clip)
    norm = LogNorm(max(lo, 1e-3), max(hi, 1e-2)) if log else Normalize(lo, hi)
    fig = plt.figure(figsize=(17, 9.2))
    gs = fig.add_gridspec(
        2, 2, width_ratios=[1, 1.5], height_ratios=[2.2, 1], wspace=0.22, hspace=0.06
    )
    ax0 = fig.add_subplot(gs[:, 0])
    ax0.imshow(img, cmap=cmap, norm=norm, origin="upper", interpolation="nearest")
    cs = ax0.contour(
        tth,
        levels=np.arange(np.ceil(np.nanmin(tth)), np.nanmax(tth), 2.0),
        colors="w",
        linewidths=0.8,
        alpha=0.7,
    )
    ax0.clabel(cs, fmt="%g°", fontsize=11)
    ax0.set_xlabel("x (px)")
    ax0.set_ylabel("y (px)")
    ax0.set_title("detector (2θ contours)")
    ax0.minorticks_off()

    def coord(x, y):
        r, c = int(round(y)), int(round(x))
        if 0 <= r < img.shape[0] and 0 <= c < img.shape[1]:
            return f"x={c} y={r}  2θ={tth[r, c]:.3f}°  χ={chi[r, c]:.2f}°  I={img[r, c]:.1f}"
        return ""

    ax0.format_coord = coord

    ax1 = fig.add_subplot(gs[0, 1])
    m = ax1.imshow(
        Z,
        cmap=cmap,
        norm=norm,
        origin="lower",
        aspect="auto",
        interpolation="nearest",
        extent=(te[0], te[-1], ce[0], ce[-1]),
    )
    ax1.set_ylabel("χ (deg)")
    ax1.set_title("unwrapped (2θ–χ)")
    ax1.tick_params(labelbottom=False)
    cb = fig.colorbar(m, ax=ax1, pad=0.015, fraction=0.04)
    cb.set_label("intensity")

    ax2 = fig.add_subplot(gs[1, 1], sharex=ax1)
    x = (te[1:] + te[:-1]) / 2
    ax2.plot(x, prof, color="k", lw=1.6)
    ax2.set_xlabel("2θ (deg)")
    ax2.set_ylabel("χ-avg I")
    if log:
        ax2.set_yscale("log")
    for k, (name, c) in enumerate(peaks.items()):  # [peaks] positions from experiments.toml
        if te[0] < c < te[-1]:
            for a in (ax1, ax2):
                a.axvline(c, color="#1f77b4", ls="--", lw=1.4, alpha=0.8)
            ax2.annotate(
                name,
                (c, 1),
                xycoords=("data", "axes fraction"),
                xytext=(3, -4),
                textcoords="offset points",
                va="top",
                fontsize=13,
                color="#1f77b4",
            )
    ax2.set_xlim(te[0], te[-1])
    # align the 2θ axis with the upper plot, which the colorbar narrowed
    p1, p2 = ax1.get_position(), ax2.get_position()
    ax2.set_position([p1.x0, p2.y0, p1.width, p2.height])
    fig.suptitle(title, fontsize=17, fontweight="bold")
    return fig


def write_html(
    img: FArray,
    tth: FArray,
    chi: FArray,
    te: FArray,
    ce: FArray,
    Z: FArray,
    prof: FArray,
    title: str,
    sub: str,
    log: bool,
    clip: tuple[float, float],
    peaks: dict[str, float],
    cmap: str,
    path: Path,
) -> Path:
    """HTML page: raw detector (hover shows 2θ and χ) | unwrapped 2θ–χ with the χ-averaged profile."""
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    from .style import header, origin_plotly, write_fit_html

    k = 2  # raw image is averaged over 2x2 pixels to keep the HTML small
    h, w = (img.shape[0] // k) * k, (img.shape[1] // k) * k

    def down(a):
        with np.errstate(all="ignore"), __import__("warnings").catch_warnings():
            __import__("warnings").simplefilter("ignore", RuntimeWarning)
            return np.nanmean(a[:h, :w].reshape(h // k, k, w // k, k), axis=(1, 3)).astype(
                np.float32
            )

    lo, hi = np.nanpercentile(img, clip)
    f = (lambda v: np.log10(np.clip(v, max(lo, 1e-2), None))) if log else (lambda v: v)
    zmin, zmax = (f(max(lo, 1e-2)), f(hi)) if log else (lo, hi)
    cs = cmap.capitalize() if cmap in ("inferno", "viridis", "magma", "plasma", "cividis") else cmap
    fig = make_subplots(
        rows=2,
        cols=2,
        specs=[[{"rowspan": 2}, {}], [None, {}]],
        column_widths=[0.42, 0.58],
        row_heights=[0.66, 0.34],
        horizontal_spacing=0.09,
        vertical_spacing=0.04,
        subplot_titles=["<b>detector</b>  (lines: 2θ)", "<b>unwrapped</b>  2θ–χ", ""],
    )
    ys, xs = np.arange(h // k) * k + (k - 1) / 2, np.arange(w // k) * k + (k - 1) / 2
    td, cd = down(tth), down(chi)
    fig.add_trace(
        go.Heatmap(
            x=xs,
            y=ys,
            z=f(down(img)),
            customdata=np.dstack([td, cd]),
            colorscale=cs,
            zmin=zmin,
            zmax=zmax,
            showscale=False,
            hovertemplate="x %{x:.0f} · y %{y:.0f}<br>2θ %{customdata[0]:.3f}° · "
            "χ %{customdata[1]:.2f}°<br>I %{z:.1f}<extra></extra>",
        ),
        1,
        1,
    )
    fig.add_trace(
        go.Contour(
            x=xs,
            y=ys,
            z=td,
            showscale=False,
            hoverinfo="skip",
            ncontours=12,
            contours=dict(coloring="none", showlabels=True, labelfont=dict(color="white", size=13)),
            line=dict(color="white", width=1),
        ),
        1,
        1,
    )
    xc, yc = (te[1:] + te[:-1]) / 2, (ce[1:] + ce[:-1]) / 2
    fig.add_trace(
        go.Heatmap(
            x=xc,
            y=yc,
            z=f(Z).astype(np.float32),
            colorscale=cs,
            zmin=zmin,
            zmax=zmax,
            colorbar=dict(
                title=dict(text="log I" if log else "Intensity", side="right"),
                len=0.62,
                y=0.68,
                thickness=18,
                outlinewidth=1.5,
            ),
            hovertemplate="2θ %{x:.3f}° · χ %{y:.2f}°<br>I %{z:.1f}<extra></extra>",
        ),
        1,
        2,
    )
    fig.add_trace(
        go.Scatter(
            x=xc,
            y=prof,
            mode="lines",
            line=dict(color="black", width=2),
            showlegend=False,
            hovertemplate="2θ %{x:.3f}° · I %{y:.2f}<extra></extra>",
        ),
        2,
        2,
    )
    for name, c in peaks.items():  # [peaks] positions from experiments.toml
        if te[0] < c < te[-1]:
            for row in (1, 2):
                fig.add_vline(
                    x=c, line=dict(color="#1f77b4", dash="dash", width=1.6), row=row, col=2
                )
            fig.add_annotation(
                x=c,
                y=1,
                xref="x3",
                yref="y3 domain",
                text=f"<b>{name}</b>",
                showarrow=False,
                xanchor="left",
                yanchor="top",
                font=dict(color="#1f77b4", size=15),
            )
    header(fig, title, sub)
    origin_plotly(fig)
    fig.update_yaxes(
        range=[h - 0.5, -0.5], scaleanchor="x", constrain="domain", title="y (px)", row=1, col=1
    )
    fig.update_xaxes(title="x (px)", range=[-0.5, w - 0.5], constrain="domain", row=1, col=1)
    fig.update_xaxes(matches="x3", showticklabels=False, row=1, col=2)
    fig.update_yaxes(title="χ (°)", row=1, col=2)
    fig.update_xaxes(title="2θ (°)", range=[te[0], te[-1]], row=2, col=2)
    fig.update_yaxes(title="χ-avg I", type="log" if log else "linear", row=2, col=2)
    for r_, c_ in ((1, 1), (1, 2)):
        fig.update_xaxes(minor=dict(ticks=""), row=r_, col=c_)
        fig.update_yaxes(minor=dict(ticks=""), row=r_, col=c_)
    fig.update_layout(margin=dict(t=95, l=90, r=30, b=70), showlegend=False)
    return write_fit_html(fig, path, 1450, 820)


def run(
    target: str,
    index: int = 0,
    geo: Geometry = Geometry(),
    tth_bin: float = 0.01,
    chi_bin: float = 0.1,
    log: bool = False,
    clip: tuple[float, float] = (1.0, 99.7),
    cmap: str = "inferno",
    show: bool = True,
) -> Path:
    f, i, n, logs = resolve(target, index)
    img, t = read_frame(f)
    tth, chi = geo.tth_map(img.shape), chi_map(geo, img.shape)
    te, ce, Z, prof = cake(img, tth, chi, tth_bin, chi_bin)

    T = np.nan
    log_path = logs or find_log([t])
    if log_path:
        T = float(TempLog.load(log_path).at([t])[0])
    title = f"{f.parent.name} / {f.name}   [{i} / {n - 1}]   {fmt_kst(t)}" + (
        f"   {T:.1f} °C" if np.isfinite(T) else ""
    )
    print(f"🖼️  {f}\n   {title}\n   2θ {te[0]:.2f} ~ {te[-1]:.2f}°,  χ {ce[0]:.1f} ~ {ce[-1]:.1f}°")

    peaks = {}
    if SETTINGS_FILE.exists():
        from .compare import load_settings

        peaks = load_settings(SETTINGS_FILE, need_exps=False).peaks
    page = write_html(
        img,
        tth,
        chi,
        te,
        ce,
        Z,
        prof,
        f"{f.parent.name} / {f.name}",
        f"[{i} / {n - 1}]  ·  {fmt_kst(t)} KST" + (f"  ·  {T:.1f} °C" if np.isfinite(T) else ""),
        log,
        clip,
        peaks,
        cmap,
        out_dir(f.parent.name) / f"{f.stem}_image.html",
    )
    if saving():
        with plt.rc_context(ORIGIN_RC):
            fig = plot(img, tth, chi, te, ce, Z, prof, title, log, clip, peaks, cmap)
            png = out_dir(f.parent.name) / f"{f.stem}_image.png"
            fig.savefig(png, dpi=150)
            plt.close(fig)
        print(f"💾 {png}\n💾 {page}")
    if show:
        import webbrowser

        webbrowser.open_new(page.resolve().as_uri())
    return page
