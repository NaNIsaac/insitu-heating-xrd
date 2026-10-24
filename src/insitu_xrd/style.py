"""Shared figure style (matplotlib and Plotly) and automatic legend placement."""

import numpy as np

# Origin-like graph: thick frame on all sides, inward major/minor ticks, large text, no grid, Arial
ORIGIN_RC = {
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 16,
    "mathtext.default": "regular",
    "axes.linewidth": 2.2,
    "axes.labelsize": 18,
    "axes.labelweight": "bold",
    "axes.titlesize": 17,
    "axes.grid": False,
    "axes.unicode_minus": False,
    "xtick.labelsize": 16,
    "ytick.labelsize": 16,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.top": True,
    "ytick.right": True,
    "xtick.major.size": 8,
    "ytick.major.size": 8,
    "xtick.minor.size": 4,
    "ytick.minor.size": 4,
    "xtick.major.width": 2.2,
    "ytick.major.width": 2.2,
    "xtick.minor.width": 1.5,
    "ytick.minor.width": 1.5,
    "xtick.major.pad": 6,
    "ytick.major.pad": 6,
    "xtick.minor.visible": True,
    "ytick.minor.visible": True,
    "legend.frameon": False,
    "legend.fontsize": 14,
    "lines.linewidth": 2.2,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
}

# Plotly (HTML) style shared by all commands, same Origin / paper look.
# Sizes (px): title 22, subtitle 15, graph title 18, axis title 20, ticks 17, legend 15, body 16
FONT_FAMILY = "Arial, Helvetica, sans-serif"
PLOTLY_FONT = dict(family=FONT_FAMILY, size=16, color="black")
PLOTLY_AXIS = dict(
    showline=True,
    linewidth=2.5,
    linecolor="black",
    mirror=True,
    ticks="inside",
    ticklen=8,
    tickwidth=2.2,
    tickcolor="black",
    tickfont=dict(size=17),
    title=dict(font=dict(size=20)),
    minor=dict(ticks="inside", ticklen=4, tickwidth=1.5),
    showgrid=False,
    zeroline=False,
)
LEGEND = dict(
    font=dict(size=15), bgcolor="rgba(255,255,255,0.88)", bordercolor="#999", borderwidth=1
)
MARGIN = dict(t=80, l=90, r=30, b=70)


def origin_plotly(fig) -> None:
    """Apply the Origin style (thick frame, large text) to a Plotly figure; call just before write_html."""
    fig.update_layout(
        template="simple_white",
        font=PLOTLY_FONT,
        hoverlabel=dict(font_size=15),
        plot_bgcolor="white",
        paper_bgcolor="white",
    )
    fig.update_xaxes(**PLOTLY_AXIS)
    fig.update_yaxes(**PLOTLY_AXIS)
    for a in fig.layout.annotations:  # subplot titles from make_subplots too
        if a.font.size is None:
            a.font.size = 18


def header(fig, title: str, sub: str = "") -> None:
    """Top-left title of every HTML page: bold name plus a gray subtitle (sample, time, conditions)."""
    text = f"<b>{title}</b>" + (
        f"   <span style='font-size:15px;color:#555'>{sub}</span>" if sub else ""
    )
    fig.update_layout(
        title=dict(
            text=text,
            x=0.01,
            xanchor="left",
            y=0.995,
            yanchor="top",
            font=dict(size=22, family=FONT_FAMILY),
        )
    )


_FIT_JS = """
(function () {
  const d = document.getElementById('{id}'), W = {w}, H = {h}, MIN = {min}, FITH = {fith};
  document.body.style.margin = '0';
  function fit() {   // 창이 크면 설계 크기 그대로 (가운데), 작으면 창에 맞게 줄임
    const s = Math.max(MIN, Math.min(1, (window.innerWidth - 16) / W, FITH ? (window.innerHeight - 16) / H : 1));
    const w = Math.round(W * s), h = Math.round(H * s);
    d.style.width = w + 'px';
    d.style.height = h + 'px';
    d.style.margin = '8px auto';
    Plotly.relayout(d, {width: w, height: h});
  }
  window.addEventListener('resize', fit);
  fit();
})();
"""


def write_fit_html(
    fig,
    path,
    width: int,
    height: int,
    min_scale: float = 0.6,
    div_id: str = "fig",
    fit_height: bool = True,
    extra_js: str = "",
    refresh: float = 0,
):
    """Write the figure at a fixed size (width x height px). A larger window does not stretch it, so the
    text ratio stays constant (a screenshot pasted into a logbook keeps readable text); a smaller
    window shrinks it to fit. fit_height=False does not fit the height (tall figures scroll).
    extra_js is a script appended after the figure; refresh > 0 reloads the page every that many
    seconds (--watch during a measurement)."""
    fig.update_layout(autosize=True, width=None, height=None)
    js = (
        _FIT_JS.replace("{id}", div_id)
        .replace("{w}", str(width))
        .replace("{h}", str(height))
        .replace("{min}", str(min_scale))
        .replace("{fith}", "true" if fit_height else "false")
    )
    if refresh > 0:
        js += f"\nsetTimeout(function () {{ location.reload(); }}, {int(refresh * 1000)});\n"
    fig.write_html(
        path,
        include_plotlyjs=True,
        full_html=True,
        div_id=div_id,
        post_script=js + extra_js,
        config={
            "displaylogo": False,
            "scrollZoom": True,
            "toImageButtonOptions": {"format": "png", "scale": 3},
        },
    )
    return path


SEG_COLOR = {"heat": "#d62728", "hold": "#8c8c8c", "cool": "#1f77b4", "gap": "#d9d9d9"}

EXP_COLORS = [
    "#d62728",
    "#1f77b4",
    "#2ca02c",
    "#9467bd",
    "#ff7f0e",
    "#17becf",
    "#e377c2",
    "#8c564b",
]
TRACK_COLORS = ["#2ca02c", "#9467bd", "#ff7f0e", "#17becf", "#e377c2", "#8c564b", "#bcbd22"]

KIND_LS = {"heat": "-", "cool": "--", "hold": ":"}
KIND_DASH = {"heat": "solid", "cool": "dash", "hold": "dot"}


# Legends go in a free spot inside the axes so a screenshot of a single graph still shows them.
LEGEND_LOCS = [
    "upper left",
    "upper right",
    "lower left",
    "lower right",
    "center left",
    "center right",
    "lower center",
    "upper center",
    "center",
]


def _densify(p: np.ndarray, step: float = 3.0) -> np.ndarray:
    """Resample a polyline in screen coordinates every step px, so a line merely crossing a box is counted."""
    p = p[np.all(np.isfinite(p), axis=1)]
    if len(p) < 2:
        return p
    seg = np.diff(p, axis=0)
    n = np.maximum(1, (np.hypot(seg[:, 0], seg[:, 1]) / step).astype(int))
    n = np.minimum(n, 200)
    parts = [p[i] + seg[i] * np.linspace(0, 1, k, endpoint=False)[:, None] for i, k in enumerate(n)]
    return np.vstack(parts + [p[-1:]])


def place_legend(
    ax, marker_weight: float = 60.0, text_weight: float = 1e4, grid: int = 19, **kw
) -> None:
    """Put the legend where it hides the least. Candidates are the 9 standard locations plus an inner
    grid x grid of positions (only those that fit inside the axes).
    Score = line points in the box (3 px spacing) + big markers such as diamonds x marker_weight
    + text_weight when it overlaps a numeric label."""
    fig = ax.figure
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    pts, w = [], []
    for ln in ax.get_lines():
        xy = ln.get_xydata()
        if not len(xy) or ln.get_transform() != ax.transData:
            continue  # reference lines (axhline, axvline) in axes coordinates may be covered
        p = ln.get_transform().transform(xy)
        big = ln.get_marker() not in (None, "None", "", " ", "o", ".") and ln.get_markersize() >= 6
        p = p if big else _densify(p)
        pts.append(p)
        w.append(np.full(len(p), marker_weight if big else 1.0))
    pts = np.vstack(pts) if pts else np.empty((0, 2))
    w = np.concatenate(w) if w else np.empty(0)
    texts = [t.get_window_extent(r) for t in ax.texts if t.get_visible() and t.get_text()]
    axbb = ax.get_window_extent(r)
    cands = [dict(loc=loc) for loc in LEGEND_LOCS] + [
        dict(loc="center", bbox_to_anchor=(fx, fy))
        for fx in np.linspace(0.05, 0.95, grid)
        for fy in np.linspace(0.05, 0.95, grid)
    ]
    best = None
    for i, c in enumerate(cands):
        bb = ax.legend(**c, **kw).get_window_extent(r)
        if bb.x0 < axbb.x0 - 1 or bb.x1 > axbb.x1 + 1 or bb.y0 < axbb.y0 - 1 or bb.y1 > axbb.y1 + 1:
            continue  # sticks out of the axes
        inside = (
            (pts[:, 0] >= bb.x0)
            & (pts[:, 0] <= bb.x1)
            & (pts[:, 1] >= bb.y0)
            & (pts[:, 1] <= bb.y1)
        )
        score = (
            float(np.nansum(w[inside]))
            + text_weight * sum(bb.overlaps(t) for t in texts)
            + (0 if i < len(LEGEND_LOCS) else 0.5)
        )  # on a tie, prefer a standard (corner) location
        if best is None or score < best[0]:
            best = (score, c)
    ax.legend(**(best[1] if best else dict(loc="best")), **kw)
