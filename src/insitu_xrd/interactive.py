"""overview 의 인터랙티브 HTML 판 (Plotly). 브라우저에서 확대/이동, 마우스 올리면 값 표시.

- 위: 온도 vs 시간 (구간 음영·라벨), 아래: 2θ 히트맵 — 시간축 연동 확대
- 온도 그래프 아래 눈금 = 그 시각의 온도. 확대하면 간격이 자동으로 촘촘해짐
- 히트맵 위 마우스: 시간 / KST / 온도 / 2θ / 세기
- 히트맵 클릭: 그 시점 프로파일을 오른쪽 패널에 겹쳐 그림 (Shift+클릭은 누적)
- 히트맵 Alt+클릭: 그 2θ 의 세기 vs 시간을 아래 피크 패널에 추가 (상변이로 생기는 픽 확인용)
- 피크 추적: 히트맵 위 피크 위치, 피크 면적 vs 시간 / vs 온도 (승온 실선, 하온 점선, 유지 점)
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
from matplotlib import colormaps
from matplotlib.colors import Normalize, to_hex
from plotly.subplots import make_subplots

from . import segments as sg
from . import tracks as tk
from .heatmap import Config, HeatmapResult, color_range, intensity_label
from .overview import (
    TIME_STEPS,
    _boundary_profiles,
    _runs,
    _spread,
    _with_gaps,
    fmt_temp,
    frame_breaks,
    frame_kinds,
    frame_temps,
    fwhm_series,
    gap_indices,
    temp_lookup,
    tth_range,
)
from .style import KIND_DASH, PLOTLY_AXIS, SEG_COLOR, TRACK_COLORS, header, write_fit_html
from .temperature import fmt_kst

AXIS = PLOTLY_AXIS  # Origin 스타일 (style.py)

# 확대/이동할 때마다 온도 눈금을 다시 계산 + 히트맵 클릭 → 프로파일
_JS = """
const gd = document.getElementById('{plot_id}');
const D = {data};
function niceTicks(lo, hi) {
  let step = D.steps[D.steps.length - 1];
  for (const s of D.steps) { if ((hi - lo) / s <= D.maxN) { step = s; break; } }
  const out = [];
  for (let v = Math.ceil(lo / step - 1e-9) * step; v <= hi + 1e-9; v += step) out.push(+v.toFixed(6));
  return out;
}
function tempAt(v) {
  const t = D.t, n = t.length;
  if (v < t[0] || v > t[n - 1]) return null;
  let lo = 0, hi = n - 1;
  while (hi - lo > 1) { const m = (lo + hi) >> 1; if (t[m] <= v) lo = m; else hi = m; }
  const w = (v - t[lo]) / ((t[hi] - t[lo]) || 1);
  const T = D.T[lo] + w * (D.T[hi] - D.T[lo]);
  return Number.isFinite(T) ? (D.est[lo] ? '~' : '') + Math.round(T) : null;
}
let busy = false;
function updateTicks() {
  if (busy || !D.t.length || D.dotTrace === null) return;
  const r = gd.layout.xaxis.range;
  const ticks = niceTicks(Math.min(r[0], r[1]), Math.max(r[0], r[1]));
  const vals = [], text = [];
  for (const v of ticks) { const s = tempAt(v); if (s !== null) { vals.push(v); text.push(s); } }
  // 보이는 시간 구간의 온도 범위로 세로축 맞춤 (확대하면 온도 변화도 크게 보이게)
  const lo = Math.min(r[0], r[1]), hi = Math.max(r[0], r[1]);
  let tmin = Infinity, tmax = -Infinity;
  for (let i = 0; i < D.t.length; i++) {
    const T = D.T[i];
    if (D.t[i] >= lo && D.t[i] <= hi && T !== null) { tmin = Math.min(tmin, T); tmax = Math.max(tmax, T); }
  }
  const lay = {'xaxis.tickvals': vals, 'xaxis.ticktext': text,
               'xaxis9.tickvals': vals, 'xaxis9.ticktext': text};   // 히트맵 윗변도 같이
  if (Number.isFinite(tmin)) {
    const pad = Math.max(0.1 * (tmax - tmin), 2);
    lay['yaxis.range'] = [tmin - pad, tmax + pad];
  }
  busy = true;
  Plotly.restyle(gd, {x: [vals], y: [vals.map(v => D.T[nearest(v)])]}, [D.dotTrace]);
  Plotly.relayout(gd, lay).then(() => { busy = false; });
}
function nearest(v) {
  const t = D.t; let lo = 0, hi = t.length - 1;
  while (hi - lo > 1) { const m = (lo + hi) >> 1; if (t[m] <= v) lo = m; else hi = m; }
  return (v - t[lo] < t[hi] - v) ? lo : hi;
}
gd.on('plotly_relayout', ev => {
  if (busy) return;
  if (Object.keys(ev).some(k => k.includes('range') || k === 'autosize')) updateTicks();
});
updateTicks();

const palette = ['#000000', '#2ca02c', '#9467bd', '#8c564b', '#e377c2', '#17becf', '#bcbd22'];
let nSel = 0, nCut = 0;
gd.on('plotly_click', ev => {
  const p = ev.points.find(q => q.curveNumber === D.heatTrace);
  if (!p) return;
  const j = Array.isArray(p.pointNumber) ? p.pointNumber[1] : p.pointIndex[1];
  const z = gd.calcdata[D.heatTrace][0].z;   // gd.data 의 z 는 base64 압축 형태라 계산된 값을 씀
  if (ev.event && ev.event.altKey && D.hasTracks) {
    const i = Array.isArray(p.pointNumber) ? p.pointNumber[0] : p.pointIndex[0];
    const row = Array.from(z[i]);
    const fin = row.filter(v => v !== null && Number.isFinite(v));
    const lo = Math.min(...fin), hi = Math.max(...fin);
    Plotly.addTraces(gd, {x: D.tg, y: row.map(v => (v === null || !Number.isFinite(v)) ? null : (v - lo) / ((hi - lo) || 1)),
      type: 'scatter', mode: 'lines', name: `2θ = ${p.y.toFixed(3)}° (cut)`, legendgroup: 'cut',
      line: {color: palette[(nCut++) % palette.length], width: 1.5, dash: 'dashdot'}, xaxis: 'x4', yaxis: 'y4',
      legend: 'legend3', connectgaps: false});
    return;
  }
  const col = Array.from({length: z.length}, (_, i) => z[i][j]);
  if (col.every(v => v === null || Number.isNaN(v))) return;
  const name = `${p.x.toFixed(2)} ${D.unit} · ${p.customdata[1]} · ${p.customdata[0]} °C`;
  const add = ev.event && ev.event.shiftKey;
  const color = palette[nSel % palette.length];
  const tr = {x: D.tth, y: col, type: 'scatter', mode: 'lines', name: name,
              line: {color: color, width: 2.5}, xaxis: 'x2', yaxis: 'y2', legendgroup: 'sel'};
  if (!add) {
    const old = gd.data.map((d, i) => d.legendgroup === 'sel' ? i : -1).filter(i => i >= 0);
    if (old.length) Plotly.deleteTraces(gd, old);
    nSel = 0;
  }
  tr.line.color = palette[nSel++ % palette.length];
  Plotly.addTraces(gd, tr);
});
"""


def write_html(
    res: HeatmapResult,
    cfg: Config,
    det: sg.Detection | None,
    path: Path,
    scaled: bool = False,
    max_n: int = 20,
    tracks=None,
    summary: dict | None = None,
) -> Path:
    x, Z, trel, fr = res.tth, res.intensity, res.trel, res.frames
    unit, scale, t_unix = res.time_unit, res.scale, res.t_unix
    t0 = t_unix[0]
    rel = lambda t: (np.asarray(t) - t0) / scale  # noqa: E731
    segs = det.segments if det is not None else []
    look = temp_lookup(det, fr, t0, scale)

    has_tr = bool(tracks)
    fw = fwhm_series(summary, tracks)
    has_w = bool(fw)
    heights = {
        (True, True): [0.24, 0.40, 0.18, 0.18],
        (True, False): [0.27, 0.45, 0.28],
        (False, True): [0.30, 0.50, 0.20],
        (False, False): [0.36, 0.64],
    }[(has_tr, has_w)]
    fig = make_subplots(
        rows=len(heights),
        cols=2,
        column_widths=[0.68, 0.32],
        row_heights=heights,
        specs=[[{}, {"rowspan": 2}], [{}, None]] + [[{}, {}]] * (len(heights) - 2),
        horizontal_spacing=0.15,
        vertical_spacing=0.085 if len(heights) == 3 else 0.07,
    )
    # 축 이름: (1,1)=x/y 온도, (1,2)=x2/y2 프로파일, (2,1)=x3/y3 히트맵,
    #         (3,1)=x4/y4 피크 면적 vs 시간, (3,2)=x5/y5 피크 면적 vs 온도,
    #         FWHM 줄 r_w: x{2r_w−2} vs 시간, x{2r_w−1} vs 온도.  히트맵 윗변 온도 눈금 = x9
    r_w = 4 if has_tr else 3
    if has_w:
        # FWHM 줄을 맨 위로 (왼쪽 아래 = 피크 면적 · 전이 온도 상자 스크린샷 구역을 가리지 않게).
        # 축 번호는 그대로 두고 세로 위치(domain)만 바꿈. FWHM 과 온도 사이 여백 = 구간 라벨 자리
        parts = [("w", 0.13), (None, 0.15), ("t", 0.18), (None, 0.06), ("h", 0.30)] + (
            [(None, 0.07), ("k", 0.16)] if has_tr else []
        )
        scale_ = 1 / sum(h for _, h in parts)
        dom, top = {}, 1.0
        for key, h in parts:
            if key:
                dom[key] = [top - h * scale_, top]
            top -= h * scale_
        wl, wr = 2 * r_w - 2, 2 * r_w - 1
        fig.update_layout(
            {
                f"yaxis{wl}": dict(domain=dom["w"]),
                f"yaxis{wr}": dict(domain=dom["w"]),
                "yaxis": dict(domain=dom["t"]),
                "yaxis3": dict(domain=dom["h"]),
                "yaxis2": dict(domain=[dom["h"][0], dom["t"][1]]),
            }
        )
        if has_tr:
            fig.update_layout(yaxis4=dict(domain=dom["k"]), yaxis5=dict(domain=dom["k"]))
    dx, dy = fig.layout.xaxis.domain, fig.layout.yaxis.domain
    dx2, dy2 = fig.layout.xaxis2.domain, fig.layout.yaxis2.domain
    dy3 = fig.layout.yaxis3.domain

    # ── 온도 ──
    dot_trace = None
    T_line = np.full(0, np.nan)
    if det is not None and np.isfinite(det.T).any():
        m = (det.t >= t_unix[0]) & (det.t <= t_unix[-1])
        tx, kst = rel(det.t[m]), [fmt_kst(v)[11:19] for v in det.t[m]]
        hov = (
            "%{x:.2f} " + unit + " · %{customdata}<br>%{fullData.name}: %{y:.1f} °C<extra></extra>"
        )
        fig.add_trace(
            go.Scatter(
                x=tx,
                y=det.pv[m],
                name="PV",
                line=dict(color="black", width=2),
                customdata=kst,
                hovertemplate=hov,
                legend="legend2",
            ),
            1,
            1,
        )
        fig.add_trace(
            go.Scatter(
                x=tx,
                y=det.sv[m],
                name="SV",
                line=dict(color="red", width=1.5, dash="dash"),
                customdata=kst,
                hovertemplate=hov,
                legend="legend2",
            ),
            1,
            1,
        )
        if det.estimated[m].any():
            fig.add_trace(
                go.Scatter(
                    x=tx,
                    y=np.where(det.estimated[m], det.T[m], np.nan),
                    name="est. (no log)",
                    line=dict(color="gray", width=2.5, dash="dot"),
                    customdata=kst,
                    hovertemplate=hov,
                    legend="legend2",
                ),
                1,
                1,
            )
        T_line, est_line, t_line = det.T[m], det.estimated[m], tx
    elif look is not None:
        fig.add_trace(
            go.Scatter(
                x=trel, y=fr["temp_pv"], name="PV", line=dict(color="black"), legend="legend2"
            ),
            1,
            1,
        )
        t_line = trel
        T_line, est_line = look(trel)
    if look is not None:
        dot_trace = len(fig.data)
        fig.add_trace(
            go.Scatter(
                x=[],
                y=[],
                mode="markers",
                marker=dict(color="black", size=6),
                showlegend=False,
                hoverinfo="skip",
            ),
            1,
            1,
        )

    # ── 히트맵 (가로 = 시간, 세로 = 2θ) ──
    tg, Zg = _with_gaps(t_unix, trel, Z)
    vlo, vhi = color_range(Z, cfg.plot.clip)
    if look is not None:
        Tg, eg = look(tg)
        t_lab = [fmt_temp(a, b) or "–" for a, b in zip(Tg, eg)]
    else:
        t_lab = ["–"] * len(tg)
    k_lab = [fmt_kst(t0 + v * scale)[11:19] for v in tg]
    col_cd = np.array([t_lab, k_lab], dtype=object).T  # (time, 2)
    # 대부분 프레임이 NaN 인 디텍터 끝 2θ 줄은 빼고 넘김 (축이 히트맵에 딱 맞게)
    hlo, hhi = tth_range(x, Z)
    yk = (x >= hlo) & (x <= hhi)
    cd = np.broadcast_to(col_cd[None, :, :], (int(yk.sum()), len(tg), 2))
    heat_trace = len(fig.data)
    cb_title = intensity_label(cfg, scaled)
    fig.add_trace(
        go.Heatmap(
            x=tg,
            y=x[yk],
            z=Zg.T[yk],
            zmin=vlo,
            zmax=vhi,
            colorscale="Inferno",
            customdata=cd.tolist(),
            hovertemplate=(
                "%{x:.2f} " + unit + " · %{customdata[1]} KST<br>T = %{customdata[0]} °C"
                "<br>2θ = %{y:.3f}°<br>I = %{z:.2f}<extra></extra>"
            ),
            colorbar=dict(
                title=dict(text=cb_title.replace(" (", "<br>("), side="top", font=dict(size=14)),
                x=dx[1] + 0.035,
                xanchor="left",
                y=dy3[0],
                yanchor="bottom",
                len=dy3[1] - dy3[0],
                thickness=14,
                outlinewidth=1.5,
                ticks="inside",
            ),
            hoverongaps=False,
            name="heatmap",
        ),
        2,
        1,
    )

    if has_tr:
        _add_tracks(
            fig,
            tracks,
            trel,
            frame_temps(fr, det),
            frame_kinds(fr, segs),
            frame_breaks(fr),
            fr,
            unit,
        )
    if has_w:
        _add_fwhm(
            fig,
            fw,
            r_w,
            trel,
            frame_temps(fr, det),
            frame_kinds(fr, segs),
            frame_breaks(fr),
            fr,
            unit,
        )

    # 히트맵 윗변 온도 눈금: x3 와 같은 범위로 묶인 x9 축 (보이지 않는 점 하나로 축을 살림)
    fig.add_trace(
        go.Scatter(
            x=[trel[0]],
            y=[x[0]],
            xaxis="x9",
            yaxis="y3",
            mode="markers",
            marker=dict(opacity=0),
            hoverinfo="skip",
            showlegend=False,
        )
    )
    fig.update_layout(
        xaxis9=dict(
            overlaying="x3",
            matches="x3",
            side="top",
            anchor="y3",
            showgrid=False,
            ticks="inside",
            ticklen=5,
            showticklabels=False,
            showline=False,
            zeroline=False,
            title=dict(text=""),
        )
    )  # 뜻은 위 온도 그래프 축 제목에

    # 이미지 없는 시간 = 회색 배경
    for i in gap_indices(t_unix):
        fig.add_vrect(
            x0=trel[i],
            x1=trel[i + 1],
            fillcolor="#d0d0d0",
            line_width=0,
            layer="below",
            row=2,
            col=1,
        )
        for row in ([3] if has_tr else []) + ([r_w] if has_w else []):
            fig.add_vrect(
                x0=trel[i],
                x1=trel[i + 1],
                fillcolor="#e6e6e6",
                line_width=0,
                layer="below",
                row=row,
                col=1,
            )
        fig.add_annotation(
            x=(trel[i] + trel[i + 1]) / 2,
            y=0.5,
            xref="x3",
            yref="y3 domain",
            text="<i>no images</i>",
            showarrow=False,
            font=dict(color="#555", size=16),
        )

    # ── 구간: 음영 · 경계 점선 · 위쪽 라벨 ──
    span = trel[-1] - trel[0]
    for k, s in enumerate(segs):
        a, b = float(rel(s.t0)), float(rel(s.t1))
        fig.add_vrect(
            x0=a,
            x1=b,
            fillcolor=SEG_COLOR[s.kind],
            opacity=0.12,
            line_width=0,
            layer="below",
            row=1,
            col=1,
        )
        if k:
            fig.add_vline(x=a, line=dict(color="#333", dash="dash", width=1.3), row=1, col=1)
            fig.add_vline(x=a, line=dict(color="white", dash="dash", width=1.3), row=2, col=1)
            for row in ([3] if has_tr else []) + ([r_w] if has_w else []):
                fig.add_vline(x=a, line=dict(color="#666", dash="dash", width=1.2), row=row, col=1)
    for xm, lab in (summary or {}).get("marks", []):  # 등온 모드: 등온 시작 / 끝
        for row in [1, 2] + ([3] if has_tr else []) + ([r_w] if has_w else []):
            fig.add_vline(x=xm, line=dict(color="#b00020", dash="dashdot", width=2), row=row, col=1)
        fig.add_annotation(
            x=xm,
            y=0.97,
            xref="x",
            yref="y domain",
            text=f"<b>{lab}</b>",
            showarrow=False,
            xanchor="left",
            yanchor="top",
            font=dict(color="#b00020", size=15),
            bgcolor="rgba(255,255,255,0.8)",
        )
    lines = [s.describe() for s in segs]
    px_per = 1000 / (span or 1)  # 그래프 폭 ~1000 px 가정 (라벨 겹침 계산용)
    widths = [(max(len(l) for l in ln) + 3) * 8.6 / px_per for ln in lines]
    mids = [float(rel((s.t0 + s.t1) / 2)) for s in segs]
    xs = _spread(mids, widths, trel[0], trel[-1])
    for k, (s, ln, mid, xc) in enumerate(zip(segs, lines, mids, xs)):
        text = "<br>".join([f"<b>{k + 1}. {ln[0]}</b>", *ln[1:]])
        fig.add_annotation(
            x=mid,
            y=1.0,
            xref="x",
            yref="y domain",
            text=text,
            showarrow=True,
            arrowhead=0,
            arrowcolor=SEG_COLOR[s.kind],
            arrowwidth=1.5,
            ax=(xc - mid) * px_per,
            ay=-45,
            yanchor="bottom",
            bgcolor=SEG_COLOR[s.kind],
            opacity=0.95,
            font=dict(color="white", size=14),
            borderpad=4,
        )

    # ── 오른쪽: 구간 경계 프로파일 (겹쳐 그림, 범례 클릭으로 켜고 끔) + 클릭한 시점 ──
    bp = _boundary_profiles(res, list(segs), det, scale)
    Ts = np.array([b[1] for b in bp], float)
    cnorm = Normalize(np.nanmin(Ts), np.nanmax(Ts)) if np.isfinite(Ts).any() else Normalize(0, 1)
    cmap = colormaps["plasma"]
    for tr_, T, est, prof in bp:
        c = to_hex(cmap(0.85 * cnorm(T))) if np.isfinite(T) else "#555555"
        nm = f"{tr_:.1f} {unit}" + (f" · {'~' if est else ''}{T:.0f} °C" if np.isfinite(T) else "")
        fig.add_trace(
            go.Scatter(
                x=x,
                y=prof,
                name=nm,
                line=dict(color=c, width=1.6),
                legendgroup="bnd",
                legendgrouptitle_text="segment boundaries",
                hovertemplate="2θ %{x:.3f}° · I %{y:.2f}<extra>" + nm + "</extra>",
            ),
            1,
            2,
        )

    # ── 축 / 레이아웃 ──
    title = f"{cfg.folder.name}   start {fr['time_kst'].iloc[0]} KST"
    fig.update_layout(
        template="simple_white",
        height=(1350 if has_tr else 950) + (70 if summary else 0) + (330 if has_w else 0),
        margin=dict(t=90 if has_w else 190, l=80, r=40, b=300 if (has_tr or summary) else 70),
        hovermode="closest",
        legend=dict(
            x=dx2[1] - 0.005,
            xanchor="right",
            y=dy2[1] - 0.005,
            yanchor="top",
            font=dict(size=14),
            groupclick="toggleitem",
            bgcolor="rgba(255,255,255,0.8)",
        ),
        legend2=dict(
            x=dx[1] + 0.005,
            xanchor="left",
            y=dy[1],
            yanchor="top",
            font=dict(size=14),
            bgcolor="rgba(255,255,255,0)",
        ),
        **(
            {
                "legend3": dict(
                    title=dict(
                        text=f"<b>peak · transition T = area crosses {tracks[0].level:g}</b>  (▲ up ▼ down)",
                        side="top",
                    ),
                    # 전이 온도 상자 오른쪽에 나란히 (상자 없으면 왼쪽 끝)
                    x=dx[0] + (0.3 if summary else 0),
                    xanchor="left",
                    y=-0.055,
                    yanchor="top",
                    font=dict(size=16),
                    bgcolor="rgba(255,255,255,0)",
                    bordercolor="#999",
                    borderwidth=1,
                ),
                # 면적 vs 온도 패널은 작아서 범례를 그래프 아래 바깥에
                "legend4": dict(
                    title=dict(text="<b>Peak area vs T: line style</b>", side="top"),
                    orientation="h",
                    x=fig.layout.xaxis5.domain[0],
                    xanchor="left",
                    y=-0.07,
                    yanchor="top",
                    font=dict(size=14),
                    bgcolor="rgba(255,255,255,0)",
                ),
            }
            if has_tr
            else {}
        ),
        font=dict(family="Arial", size=16),
    )
    spikes = dict(
        showspikes=True,
        spikemode="across",
        spikesnap="cursor",
        spikethickness=1,
        spikecolor="#666",
        spikedash="dot",
    )
    fig.update_xaxes(AXIS)
    fig.update_yaxes(AXIS)
    fig.update_xaxes(
        matches="x3",
        range=[trel[0], trel[-1]],
        tickfont=dict(size=14),
        title=dict(text="T (°C) at time ↓", font=dict(size=14), standoff=4),
        **spikes,
        row=1,
        col=1,
    )
    fig.update_yaxes(title="T (°C)", row=1, col=1)
    fig.update_xaxes(
        title=f"Time ({unit}) from {fr['time_kst'].iloc[0][11:19]} KST",
        **spikes,
        range=[trel[0], trel[-1]],
        row=2,
        col=1,
    )
    # 히트맵 셀 경계에 범위를 딱 맞추면 픽셀 반올림으로 1 px 배경이 보여서 반 칸 안쪽으로
    inset = 0.5 * (x[1] - x[0])
    fig.update_yaxes(title="2θ (°)", range=[hlo + inset, hhi - inset], row=2, col=1)
    if has_tr:
        fig.update_xaxes(
            matches="x3", range=[trel[0], trel[-1]], title=f"Time ({unit})", **spikes, row=3, col=1
        )
        fig.update_yaxes(title="Peak area (norm.)", row=3, col=1)
        lv = tracks[0].level
        for col in (1, 2):  # 전이 기준선
            fig.add_hline(
                y=lv,
                line=dict(color="#444", width=1, dash="dash"),
                row=3,
                col=col,
                annotation_text=f"{lv:g}",
                annotation_position="right",
            )
        fig.update_xaxes(title="T (°C)", row=3, col=2)
        fig.update_yaxes(title="Peak area (norm.)", row=3, col=2)
        fig.add_annotation(
            x=0.0,
            y=1.0,
            xref="x5 domain",
            yref="y5 domain",
            yanchor="bottom",
            showarrow=False,
            text="<b>Peak area vs T</b> — solid heating · dash cooling · dot hold",
        )
        fig.add_annotation(
            x=0.0,
            y=1.0,
            xref="x4 domain",
            yref="y4 domain",
            yanchor="bottom",
            showarrow=False,
            text="<b>Peaks</b> ▲ appears ▼ vanishes — Alt+click heatmap: add 2θ cut",
        )
    if has_w:
        wl, wr = 2 * r_w - 2, 2 * r_w - 1
        fig.update_xaxes(matches="x3", range=[trel[0], trel[-1]], **spikes, row=r_w, col=1)
        fig.update_yaxes(title="FWHM (°)", row=r_w, col=1)
        fig.update_xaxes(title="T (°C)", row=r_w, col=2)
        fig.update_yaxes(title="FWHM (°)", row=r_w, col=2)
        fig.add_annotation(
            x=0.0,
            y=1.0,
            xref=f"x{wl} domain",
            yref=f"y{wl} domain",
            yanchor="bottom",
            showarrow=False,
            text="<b>FWHM</b> — per-frame Gaussian fit (line: running median)",
        )
        fig.add_annotation(
            x=0.0,
            y=1.0,
            xref=f"x{wr} domain",
            yref=f"y{wr} domain",
            yanchor="bottom",
            showarrow=False,
            text="<b>FWHM vs T</b> — solid heating · dash cooling",
        )
        fig.update_layout(
            legend5=dict(
                x=fig.layout[f"xaxis{wl}"].domain[1] - 0.005,
                xanchor="right",
                y=fig.layout[f"yaxis{wl}"].domain[1] - 0.005,
                yanchor="top",
                font=dict(size=14),
                bgcolor="rgba(255,255,255,0.8)",
            )
        )
    fig.update_xaxes(title="2θ (°)", range=[x[0], x[-1]], row=1, col=2)
    fig.update_yaxes(title="Intensity", row=1, col=2)
    fig.add_annotation(
        x=0.0,
        y=1.0,
        xref="x2 domain",
        yref="y2 domain",
        yanchor="bottom",
        showarrow=False,
        align="left",
        text="<b>Profiles</b> — click heatmap to add (Shift+click to keep)",
    )

    if summary:  # 전이 온도 상자: 왼쪽 맨 아래 (피크 범례 왼쪽) — 왼쪽 아래만 캡처해도 보이게
        if summary["ok"]:
            vunit = " min" if summary.get("unit") == "min" else " °C"
            nd = 1 if vunit == " min" else 0
            body = "<br>".join(
                f"<b>{pk}</b>   <b><span style='color:#b00020'>{t50:.{nd}f}{vunit}</span></b>"
                if np.isfinite(t50)
                else f"<b>{pk}</b>   –"
                for pk, t50, c, fw in summary["rows"]
            )
            detail = (
                "   ".join(summary["details"])
                if summary.get("details")
                else "   ".join(
                    f"{pk} {c:.2f}° · FWHM {fw:.2f}°" for pk, t50, c, fw in summary["rows"]
                )
            )
        else:
            body, detail = "no T50", summary["msg"]
        head = f"{cfg.folder.name}  ·  {fr['time_kst'].iloc[0][:16]} KST"
        fig.add_annotation(
            x=dx[0],
            y=-0.055,
            xref="paper",
            yref="paper",
            xanchor="left",
            yanchor="top",
            align="left",
            showarrow=False,
            text=f"<span style='font-size:19px'><b>{head}</b></span><br>"
            f"<span style='font-size:15px;color:#555'>{summary['title']}</span><br>"
            f"{body}<br><span style='font-size:14px;color:#555'>{detail}</span>",
            font=dict(size=32.5, family="Arial"),
            bgcolor="#fff8e1",
            bordercolor="#b8860b",
            borderwidth=2,
            borderpad=12,
        )

    data = {
        "t": np.round(np.asarray(t_line, float), 5).tolist() if look is not None else [],
        "T": [None if not np.isfinite(v) else round(float(v), 2) for v in T_line],
        "est": [bool(v) for v in (est_line if look is not None else [])],
        "steps": TIME_STEPS[unit],
        "maxN": 20,
        "unit": unit,
        "dotTrace": dot_trace,
        "heatTrace": heat_trace,
        "tth": np.round(x, 4).tolist(),
        "tg": [None if not np.isfinite(v) else round(float(v), 4) for v in tg],
        "hasTracks": has_tr,
    }
    plot_id = "overview"
    js = _JS.replace("{plot_id}", plot_id).replace("{data}", json.dumps(data))
    # 다른 명령과 같은 머리말 · 같은 크기 규칙 (정해진 폭, 창이 작으면 줄어듦, 세로는 스크롤)
    header(
        fig,
        cfg.folder.name,
        f"start {fr['time_kst'].iloc[0]} KST"
        + ("  ·  isothermal mode" if summary and summary.get("unit") == "min" else ""),
    )
    H = fig.layout.height
    return write_fit_html(fig, path, 1500, H, div_id=plot_id, fit_height=False, extra_js=js)


def _add_fwhm(fig, series, row: int, trel, temp, kinds, breaks, fr, unit: str) -> None:
    """FWHM vs 시간 (점 + 이동 중앙값, 범례 클릭으로 켜고 끔) / FWHM vs 온도 (구간 종류별 선 모양)."""
    from .compare import smooth

    kst = fr["time_kst"].str[11:19].to_numpy()
    for pk, c, w in series:
        s = smooth(w)
        last = np.nanmedian(w[np.isfinite(w)][-10:])
        cd = np.stack([kst, np.round(temp, 1)], axis=1)
        fig.add_trace(
            go.Scatter(
                x=trel,
                y=w,
                mode="markers",
                marker=dict(size=4, color=c, opacity=0.3),
                legendgroup=f"w{pk}",
                showlegend=False,
                customdata=cd,
                legend="legend5",
                hovertemplate=(
                    "%{x:.2f} " + unit + " · %{customdata[0]}<br>T = %{customdata[1]} °C"
                    f"<br>FWHM = %{{y:.3f}}°<extra>{pk}</extra>"
                ),
            ),
            row,
            1,
        )
        fig.add_trace(
            go.Scatter(
                x=trel,
                y=np.where(breaks, np.nan, s),
                mode="lines",
                connectgaps=False,
                line=dict(color=c, width=2),
                name=f"{pk}  {last:.2f}° (end)",
                legendgroup=f"w{pk}",
                legend="legend5",
                hoverinfo="skip",
            ),
            row,
            1,
        )
        for a, b in _runs(kinds):
            fig.add_trace(
                go.Scatter(
                    x=temp[a:b],
                    y=s[a:b],
                    mode="lines",
                    connectgaps=False,
                    legendgroup=f"w{pk}",
                    showlegend=False,
                    line=dict(color=c, width=1.6, dash=KIND_DASH.get(kinds[a], "solid")),
                    hovertemplate=f"T = %{{x:.1f}} °C<br>FWHM = %{{y:.3f}}°<extra>{pk} ({kinds[a] or '-'})</extra>",
                ),
                row,
                2,
            )


def _add_tracks(fig, tracks: list[tk.Track], trel, temp, kinds, breaks, fr, unit: str) -> None:
    kst = fr["time_kst"].str[11:19].to_numpy()
    for i, tr in enumerate(tracks):
        c = TRACK_COLORS[i % len(TRACK_COLORS)]
        area = tk.norm_area(tr.area)
        nm = tk.label(tr, temp)
        # 히트맵 오른쪽 바깥에 피크 이름 (그 2θ 높이) — 데이터는 가리지 않음
        fig.add_annotation(
            x=1.0,
            y=tr.center,
            xref="x3 domain",
            yref="y3",
            xanchor="left",
            text=f"<b>◀ {tr.name}</b>",
            showarrow=False,
            font=dict(color=c, size=15),
            hovertext=nm,
        )
        cd = np.stack([kst, np.round(temp, 1), np.round(tr.pos, 3)], axis=1)
        y = np.where(breaks, np.nan, area)
        fig.add_trace(
            go.Scatter(
                x=trel,
                y=y,
                mode="lines",
                line=dict(color=c, width=1.8),
                name=nm,
                legendgroup=tr.name,
                legend="legend3",
                customdata=cd,
                connectgaps=False,
                hovertemplate=(
                    "%{x:.2f} " + unit + " · %{customdata[0]}<br>T = %{customdata[1]} °C"
                    "<br>2θ = %{customdata[2]}°<br>area = %{y:.3f}<extra>" + tr.name + "</extra>"
                ),
            ),
            3,
            1,
        )
        for ev, k in tr.events:
            sym = "triangle-up" if ev == "appears" else "triangle-down"
            txt = f"{tr.name} {'↑' if ev == 'appears' else '↓'} {temp[k]:.0f} °C"
            for row, col, xv in ((3, 1, trel[k]), (3, 2, temp[k])):
                fig.add_trace(
                    go.Scatter(
                        x=[xv],
                        y=[area[k]],
                        mode="markers",
                        marker=dict(
                            symbol=sym, size=13, color=c, line=dict(color="black", width=1)
                        ),
                        legendgroup=tr.name,
                        showlegend=False,
                        hovertemplate=txt + f" · {kst[k]}<extra></extra>",
                    ),
                    row,
                    col,
                )
            fig.add_vline(x=trel[k], line=dict(color=c, width=1, dash="dot"), row=2, col=1)
            fig.add_vline(x=trel[k], line=dict(color=c, width=1, dash="dot"), row=3, col=1)
            fig.add_vline(x=temp[k], line=dict(color=c, width=1, dash="dot"), row=3, col=2)
        # 면적 vs 온도: 승온/하온/유지 구간별 선 모양
        for a, b in _runs(kinds):
            seg = slice(a, b)
            fig.add_trace(
                go.Scatter(
                    x=temp[seg],
                    y=area[seg],
                    mode="lines",
                    connectgaps=False,
                    line=dict(color=c, width=1.6, dash=KIND_DASH.get(kinds[a], "solid")),
                    name=nm,
                    legendgroup=tr.name,
                    showlegend=False,
                    legend="legend4",
                    customdata=cd[seg],
                    hovertemplate=(
                        "T = %{x:.1f} °C · %{customdata[0]}<br>area = %{y:.3f}"
                        f"<extra>{tr.name} ({kinds[a] or '-'})</extra>"
                    ),
                ),
                3,
                2,
            )
    for lab, dash in (("heating", "solid"), ("cooling", "dash"), ("hold", "dot")):
        fig.add_trace(
            go.Scatter(
                x=[None],
                y=[None],
                mode="lines",
                name=lab,
                legend="legend4",
                line=dict(color="#444", dash=dash, width=2),
            ),
            3,
            2,
        )
