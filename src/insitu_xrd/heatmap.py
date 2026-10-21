"""폴더 내 HDF5 디텍터 이미지 → 링 평균 2θ 프로파일 → 시간 히트맵 (+ 온도 로그).

적분 결과는 out/cache/ 에 저장해 두고, 폴더의 파일(개수·수정시각)과 기하·적분 옵션이 그대로면
다시 계산하지 않음 (overview · compare · heatmap 공용).
"""

from __future__ import annotations

import hashlib
import json
import os
import warnings
from collections.abc import Sequence
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LogNorm
from tqdm.auto import tqdm

from .config import ATTR, CACHE_DIR, CACHE_MODE, CACHE_MODES, DEFAULT_FOLDER, DSET, out_dir, saving
from .geometry import FArray, Geometry
from .integrate import ProfileOptions, RadialIntegrator
from .io import list_files, load_image, read_frame
from .temperature import PV, SV, TempLog, fmt_kst, parse_time


# ═════════════════════════════ 설정 ═════════════════════════════
@dataclass(frozen=True, slots=True)
class PlotOptions:
    log_color: bool = False
    time_unit: Literal["s", "min"] = "min"
    cmap: str = "inferno"
    show: bool = True
    clip: tuple[float, float] = (
        25.0,
        99.7,
    )  # overview 히트맵 색 범위 (세기 백분위). 낮출수록 대비 약함


@dataclass(frozen=True, slots=True)
class Config:
    folder: Path = DEFAULT_FOLDER
    pattern: str = "*.h5"
    dset: str = DSET
    attr: str = ATTR
    workers: int | None = None  # None = min(8, CPU 수)
    geometry: Geometry = Geometry()
    profile: ProfileOptions = ProfileOptions()
    plot: PlotOptions = PlotOptions()
    temp_log: Path | Sequence[Path] | None = (
        None  # 온도 로그 CSV (여러 개면 합침, 없으면 온도 생략)
    )
    at: tuple[
        float | str, ...
    ] = ()  # 표시할 시각: 숫자=첫 프레임 기준 상대시간(time_unit), 'HH:MM:SS'=KST 시각
    cache: str | None = None  # use / refresh / off (None = 전역: --cache · 환경변수 · toml)


# ═════════════════════════════ 프레임 로딩 (병렬) ═════════════════════════════
# 워커 프로세스마다 한 번만 만드는 전역 상태 (Windows spawn 대응)
_worker: tuple[Config, RadialIntegrator] | None = None


def _init_worker(cfg: Config, shape: tuple[int, int]) -> None:
    global _worker
    _worker = (cfg, RadialIntegrator.build(cfg.geometry, cfg.profile, shape))


def _process(path: Path) -> tuple[float, FArray] | str:
    """(촬영시각, 프로파일). 깨진 파일은 오류 문자열 (하나 때문에 전체가 멈추지 않게)."""
    assert _worker is not None
    cfg, integ = _worker
    try:
        img, t = read_frame(path, cfg.dset, cfg.attr)
    except Exception as e:
        return f"{path.name}: {e}"
    prof = integ(img)
    if cfg.profile.normalize:
        prof = prof / np.nanmean(prof)
    return t, prof


def _integrate(files: Sequence[Path], cfg: Config, shape: tuple[int, int]):
    """파일들 → (시각[n], 프로파일[n, 2θ], 파일명[n]), 시간순. 몇 개뿐이면 프로세스 없이 바로."""
    workers = cfg.workers or min(8, os.cpu_count() or 1)
    times, profs, names = [], [], []

    def collect(it):
        for p, r in tqdm(
            zip(files, it),
            total=len(files),
            desc="📡 적분",
            unit="frame",
            colour="cyan",
            dynamic_ncols=True,
            disable=len(files) < 16,
        ):
            if isinstance(r, str):
                tqdm.write(f"⚠️  {r}")
                continue
            times.append(r[0])
            profs.append(r[1])
            names.append(p.name)

    if len(files) < 16:
        _init_worker(cfg, shape)
        collect(map(_process, files))
    else:
        with ProcessPoolExecutor(workers, initializer=_init_worker, initargs=(cfg, shape)) as pool:
            collect(
                pool.map(_process, files, chunksize=max(1, min(16, len(files) // (4 * workers))))
            )
    if not times:
        return np.empty(0), np.empty((0, 0)), np.empty(0, str)
    order = np.argsort(times, kind="stable")
    return np.asarray(times)[order], np.vstack(profs)[order], np.asarray(names)[order]


def _cache_file(cfg: Config) -> Path:
    """폴더 · 기하 · 적분 옵션 → 캐시 파일 이름. 파일 목록 · 수정시각은 캐시 안에 저장해 두고 비교
    (새 파일만 늘었으면 그것만 적분해서 이어 붙임 → 측정 중에 다시 돌려도 빠름)."""
    g = cfg.geometry
    sig = json.dumps(
        [
            str(cfg.folder),
            cfg.pattern,
            [g.pixel, g.sdd, g.xc, g.yc, g.alpha_deg, g.energy_kev],
            asdict(cfg.profile),
            cfg.dset,
            cfg.attr,
        ],
        default=str,
    )
    return CACHE_DIR / f"{cfg.folder.name}_{hashlib.md5(sig.encode()).hexdigest()[:10]}.npz"


def _mtimes(cfg: Config, files: Sequence[Path]) -> np.ndarray:
    mt = {e.name: e.stat().st_mtime for e in os.scandir(cfg.folder)}  # Windows 는 stat 이 공짜
    return np.array([mt.get(f.name, 0.0) for f in files])


_cache_mode = CACHE_MODE  # 명령의 --cache 로 바꿈 (set_cache_mode)


def set_cache_mode(mode: str | None) -> None:
    """이번 실행 전체의 캐시 방식: use / refresh / off (None 이면 그대로)."""
    global _cache_mode
    if mode is None:
        return
    if mode not in CACHE_MODES:
        raise SystemExit(f"❌ --cache {mode}: {' / '.join(CACHE_MODES)} 중 하나")
    _cache_mode = mode
    if mode != "use":
        tqdm.write(
            f"🗂️  캐시: {mode} ({'다시 적분해서 덮어씀' if mode == 'refresh' else '읽지도 쓰지도 않음'})"
        )


def integrate_folder(cfg: Config) -> tuple[FArray, FArray, FArray, np.ndarray]:
    """폴더 → (2θ, 세기[frame, 2θ], 촬영시각 Unix[frame], 파일명[frame]). 캐시 방식은 cfg.cache 또는 전역."""
    mode = cfg.cache or _cache_mode
    files = list_files(cfg.folder, cfg.pattern)
    cache = _cache_file(cfg)
    mt = _mtimes(cfg, files)
    if mode == "use" and cache.exists():
        d = dict(np.load(cache, allow_pickle=False))
        old = dict(zip(d["all_files"], d["all_mtimes"])) if "all_files" in d else {}
        now = dict(zip((f.name for f in files), mt))
        if old and all(now.get(k) == v for k, v in old.items()):  # 지운 · 덮어쓴 파일 없음
            new = [f for f in files if f.name not in old]  # 새 파일 + 지난번 읽기 실패
            if not new:
                tqdm.write(f"📂 {cfg.folder}  ({len(files)} files)  ↺ 캐시 {cache.name}")
                return d["tth"], d["intensity"], d["time_unix"], d["files"]
            tqdm.write(
                f"📂 {cfg.folder}  ({len(files)} files)  ↺ 캐시 + 새 파일 {len(new)} 개 적분"
            )
            t1, Z1, n1 = _integrate(new, cfg, load_image(new[0], cfg.dset).shape)
            t = np.concatenate([d["time_unix"], t1])
            order = np.argsort(t, kind="stable")
            Z = np.vstack([d["intensity"], Z1]) if len(t1) else d["intensity"]
            res = (d["tth"], Z[order], t[order], np.concatenate([d["files"], n1])[order])
            np.savez_compressed(
                cache,
                tth=res[0],
                intensity=res[1],
                time_unix=res[2],
                files=res[3],
                all_files=res[3],
                all_mtimes=np.array([now[k] for k in res[3]]),
            )
            return res
    shape = load_image(files[0], cfg.dset).shape
    workers = cfg.workers or min(8, os.cpu_count() or 1)
    x = RadialIntegrator.build(cfg.geometry, cfg.profile, shape).centers
    tqdm.write(
        f"📂 {cfg.folder}  ({len(files)} files, {shape[0]}×{shape[1]}, "
        f"2θ {x[0]:.2f}–{x[-1]:.2f}°, workers={workers})"
    )
    t, Z, names = _integrate(files, cfg, shape)
    if mode != "off":
        cache.parent.mkdir(parents=True, exist_ok=True)
        now = dict(zip((f.name for f in files), mt))  # 읽기 실패한 파일(쓰는 중 등)은 다음에 다시
        np.savez_compressed(
            cache,
            tth=x,
            intensity=Z,
            time_unix=t,
            files=names,
            all_files=names,
            all_mtimes=np.array([now[k] for k in names]),
        )
    return x, Z, t, names


def cache_files(match: str | None = None) -> list[Path]:
    """out/cache 의 캐시 파일 (match 가 있으면 폴더 이름에 그 글자가 들어간 것만)."""
    fs = sorted(CACHE_DIR.glob("*.npz")) if CACHE_DIR.exists() else []
    return [f for f in fs if match is None or match.lower() in f.stem.rsplit("_", 1)[0].lower()]


# ═════════════════════════════ 공용 도우미 ═════════════════════════════
def time_mean(Z: FArray) -> FArray:
    """갭(전 프레임 NaN) 빈 경고 없이 시간 평균."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.nanmean(Z, axis=0)


def beam_ok(Z: FArray, groups: np.ndarray | None = None, frac: float = 0.2) -> np.ndarray:
    """빔 꺼짐 등 거의 0 인 프레임 = False. 그룹(폴더·블록)마다 프레임 세기 중앙값의 frac 미만."""
    with np.errstate(all="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        level = np.nanmedian(Z, axis=1)
        if groups is None:
            return level > frac * np.nanmedian(level)
        ok = np.zeros(len(Z), bool)
        for g in np.unique(groups):
            m = groups == g
            ok[m] = level[m] > frac * np.nanmedian(level[m])
    return ok


def regrid(x: FArray, tth: FArray, Z: FArray) -> FArray:
    """다른 2θ 축(tth)의 프로파일들을 x 축으로 보간 (같은 축이면 그대로)."""
    if len(tth) == len(x) and np.allclose(tth, x):
        return Z
    return np.array([np.interp(x, tth, z, left=np.nan, right=np.nan) for z in Z])


def color_range(Z: FArray, clip: tuple[float, float]) -> tuple[float, float]:
    lo, hi = np.percentile(Z[np.isfinite(Z)], clip)
    return float(lo), float(hi)


def intensity_label(cfg: Config, scaled: bool = False) -> str:
    return (
        "Normalized intensity"
        if cfg.profile.normalize
        else "Intensity (a.u., per-folder scaled)"
        if scaled
        else "Intensity (a.u.)"
    )


# ═════════════════════════════ 결과 ═════════════════════════════
@dataclass(slots=True)
class HeatmapResult:
    tth: FArray
    intensity: FArray  # (frame, 2θ)
    frames: pd.DataFrame  # file, time_kst, time_unix, time_rel, [temp_pv, temp_sv]
    time_unit: str

    @property
    def scale(self) -> float:
        """상대시간 단위의 초 수."""
        return 60.0 if self.time_unit == "min" else 1.0

    @property
    def t_unix(self) -> FArray:
        return self.frames["time_unix"].to_numpy()

    @property
    def trel(self) -> FArray:
        return self.frames["time_rel"].to_numpy()

    def subset(self, keep: np.ndarray) -> HeatmapResult:
        """일부 프레임만 (상대시간은 남은 첫 프레임 기준으로 다시)."""
        fr = self.frames[keep].reset_index(drop=True)
        fr["time_rel"] = (fr["time_unix"] - fr["time_unix"].iloc[0]) / self.scale
        return HeatmapResult(self.tth, self.intensity[keep], fr, self.time_unit)

    def add_temperature(self, log: TempLog) -> None:
        t = self.t_unix
        self.frames["temp_pv"] = log.at(t, PV)
        self.frames["temp_sv"] = log.at(t, SV)

    def nearest(self, at: float | str) -> int:
        """at (상대시간 또는 'HH:MM:SS' KST) 에 가장 가까운 프레임 인덱스."""
        if isinstance(at, str):
            return int(
                np.argmin(
                    np.abs(self.t_unix - parse_time(at, self.frames["time_kst"].iloc[0][:10]))
                )
            )
        return int(np.argmin(np.abs(self.trel - at)))


# ═════════════════════════════ 그림 ═════════════════════════════
def plot(res: HeatmapResult, cfg: Config) -> plt.Figure:
    geo, po = cfg.geometry, cfg.plot
    x, Z, trel, fr = res.tth, res.intensity, res.trel, res.frames
    has_temp = "temp_pv" in fr
    ratios = [2.2, 0.7, 1.2] if has_temp else [2.2, 1.2]
    fig, axes = plt.subplots(
        1, len(ratios), figsize=(5 * sum(ratios), 7), gridspec_kw={"width_ratios": ratios}
    )
    ax_map, ax_prof = axes[0], axes[-1]
    at_idx = [res.nearest(t) for t in cfg.at]
    colors = plt.cm.tab10(np.arange(len(at_idx)) % 10)

    # 히트맵
    lo, hi = color_range(Z, (1, 99.5))
    kw = {"norm": LogNorm(max(lo, 1e-3), hi)} if po.log_color else {"vmin": lo, "vmax": hi}
    mesh = ax_map.pcolormesh(x, trel, Z, shading="nearest", cmap=po.cmap, **kw)
    fig.colorbar(
        mesh,
        ax=ax_map,
        pad=0.02,
        fraction=0.04,
        label="normalized intensity" if cfg.profile.normalize else "mean intensity",
    )
    ax_map.set(
        xlabel="2θ (deg)",
        ylabel=f"time ({po.time_unit}) from {fr['time_kst'].iloc[0]} KST",
        title=f"{cfg.folder.name}  [{cfg.profile.region}]",
    )
    ax_map.secondary_xaxis("top", functions=(geo.tth_to_q, geo.q_to_tth)).set_xlabel("q (Å⁻¹)")

    # 온도 vs 시간 (히트맵과 시간축 공유)
    if has_temp:
        ax = axes[1]
        ax.sharey(ax_map)
        ax.plot(fr["temp_pv"], trel, lw=1.2, label="PV")
        ax.plot(fr["temp_sv"], trel, lw=0.8, ls="--", label="SV")
        ax.set(xlabel="T (°C)", title="temperature")
        ax.tick_params(labelleft=False)
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3)

    # 시간 평균 + 선택 시각 프로파일
    ax_prof.plot(x, time_mean(Z), lw=1, color="0.6", label="time average")
    for i, c in zip(at_idx, colors):
        label = f"{trel[i]:.2f} {po.time_unit}"
        if has_temp:
            label += f", {fr['temp_pv'].iloc[i]:.0f} °C"
        ax_prof.plot(x, Z[i], lw=1, color=c, label=label)
        for ax in axes[:-1]:
            ax.axhline(trel[i], color=c, lw=0.8, ls=":")
    ax_prof.set(xlabel="2θ (deg)", ylabel="mean intensity", title="profiles")
    ax_prof.legend(fontsize=9)
    ax_prof.grid(alpha=0.3)

    fig.tight_layout()
    return fig


# ═════════════════════════════ 실행 ═════════════════════════════
def compute(cfg: Config) -> HeatmapResult:
    x, Z, t, names = integrate_folder(cfg)
    unit = cfg.plot.time_unit
    fr = pd.DataFrame(
        {
            "file": names,
            "time_kst": [fmt_kst(v) for v in t],
            "time_unix": t,
            "time_rel": (t - t[0]) / (60.0 if unit == "min" else 1.0),
        }
    )
    res = HeatmapResult(x, Z, fr, unit)
    tqdm.write(
        f"⏱️  {len(fr)} frames, {fr['time_kst'].iloc[0]} ~ {fr['time_kst'].iloc[-1]} "
        f"(총 {res.trel[-1]:.2f} {unit})"
    )
    if cfg.temp_log:
        log = TempLog.load(cfg.temp_log)
        res.add_temperature(log)
        n = fr["temp_pv"].notna().sum()
        tqdm.write(
            f"🌡️  {log.path.name} ({log.span[0]} ~ {log.span[1]}): 온도 매칭 {n}/{len(fr)} frames"
        )
        if n == 0:
            tqdm.write("   ⚠️  이 실험 시간대에 온도 기록이 없음")
    return res


def run(cfg: Config = Config()) -> HeatmapResult:
    res = compute(cfg)
    if cfg.at:
        cols = [
            c for c in ("file", "time_kst", "time_rel", "temp_pv", "temp_sv") if c in res.frames
        ]
        sel = res.frames.iloc[[res.nearest(t) for t in cfg.at]][cols]
        tqdm.write("\n📍 선택 시각\n" + sel.round(3).to_string(index=False))

    fig = plot(res, cfg)
    if cfg.plot.show and not saving():
        plt.show()
        return res
    stem = out_dir(cfg.folder.name) / f"{cfg.folder.name}_tth_heatmap"
    fig.savefig(stem.with_suffix(".png"), dpi=150)
    np.savez(
        stem.with_suffix(".npz"),
        tth=res.tth,
        q=cfg.geometry.tth_to_q(res.tth),
        time_rel=res.trel,
        time_unit=res.time_unit,
        intensity=res.intensity,
        files=res.frames["file"].to_numpy(str),
        **({"temp_pv": res.frames["temp_pv"].to_numpy()} if "temp_pv" in res.frames else {}),
    )
    frames_csv = stem.with_name(f"{cfg.folder.name}_frames.csv")
    res.frames.to_csv(frames_csv, index=False, encoding="utf-8-sig")
    tqdm.write(f"\n💾 {stem.with_suffix('.png')}\n💾 {stem.with_suffix('.npz')}\n💾 {frames_csv}")
    if cfg.plot.show:
        plt.show()
    return res
