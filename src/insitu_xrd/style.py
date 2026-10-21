"""그림 공통 스타일 (matplotlib · Plotly) 과 범례 자동 배치."""

import numpy as np

# Origin 기본 그래프 느낌: 사방 굵은 테두리, 안쪽 눈금(주/보조), 큰 글씨, 격자 없음, Arial
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

# ─────────── Plotly (HTML) 공통 스타일 — 모든 명령이 같은 모양 ───────────
#   Origin / 논문 그림: Arial, 사방 굵은 검은 테두리, 안쪽 주·보조 눈금, 격자 없음, 큰 글씨
#   크기 규칙 (px): 제목 22 · 부제 15 · 그래프 제목 18 · 축 제목 20 · 눈금 17 · 범례 15 · 본문 16
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
    """Plotly 그림 전체에 Origin 스타일 (굵은 테두리 · 큰 글씨) 적용. write_html 직전에 호출."""
    fig.update_layout(
        template="simple_white",
        font=PLOTLY_FONT,
        hoverlabel=dict(font_size=15),
        plot_bgcolor="white",
        paper_bgcolor="white",
    )
    fig.update_xaxes(**PLOTLY_AXIS)
    fig.update_yaxes(**PLOTLY_AXIS)
    for a in fig.layout.annotations:  # 서브플롯 제목 (make_subplots) 도 크게
        if a.font.size is None:
            a.font.size = 18


def header(fig, title: str, sub: str = "") -> None:
    """모든 HTML 의 맨 위 왼쪽 제목: 굵은 이름 + 회색 부제 (시료 · 시각 · 조건)."""
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
    """그림을 정해진 크기(width×height px)로: 창이 더 커도 늘어나지 않아 빈 곳 없이 글씨 비율 일정
    (스샷 → 로그북에 붙여도 글씨가 작아지지 않음), 창이 작으면 창에 맞게 줄어듦.
    fit_height=False 면 세로는 맞추지 않음 (긴 그림은 스크롤). extra_js = 그림 뒤에 붙일 스크립트,
    refresh > 0 이면 그 초마다 페이지 새로 읽기 (측정 중 --watch)."""
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


# 온도 구간 색 (matplotlib · Plotly 공용)
SEG_COLOR = {"heat": "#d62728", "hold": "#8c8c8c", "cool": "#1f77b4", "gap": "#d9d9d9"}

# 실험 / 피크 트랙 선 색
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

# 면적 vs 온도: 구간 종류별 선 모양
KIND_LS = {"heat": "-", "cool": "--", "hold": ":"}
KIND_DASH = {"heat": "solid", "cool": "dash", "hold": "dot"}


# ─────────── matplotlib 범례: 그래프 안 빈 곳 자동 (그래프 하나만 잘라 캡처해도 보이게) ───────────
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
    """화면 좌표 꺾은선을 step px 마다 점으로 (선이 상자를 가로지르기만 해도 세어지게)."""
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
    """범례를 가리는 게 가장 적은 곳에. 후보 = 표준 9 위치 + 안쪽 grid×grid 위치 (축 안에 다 들어가는 것만).
    점수 = 상자 안 선(3 px 간격 점) 수 + ◆ 같은 큰 표식 × marker_weight + 숫자 라벨과 겹치면 text_weight."""
    fig = ax.figure
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    pts, w = [], []
    for ln in ax.get_lines():
        xy = ln.get_xydata()
        if not len(xy) or ln.get_transform() != ax.transData:
            continue  # axhline · axvline 같은 기준선(축 좌표)은 가려도 됨
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
            continue  # 축 밖으로 삐져나가면 안 씀
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
        )  # 같으면 표준 위치(모서리) 우선
        if best is None or score < best[0]:
            best = (score, c)
    ax.legend(**(best[1] if best else dict(loc="best")), **kw)
