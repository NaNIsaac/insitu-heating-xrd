"""단일 이미지의 가로 중앙 스트립 → 세로(픽셀) 프로파일 + 피크."""

from pathlib import Path
from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import find_peaks, peak_widths

from .config import out_dir, saving
from .geometry import FArray
from .integrate import strip_bounds
from .io import load_image


def center_strip_profile(img: FArray, width: int, reduce: Literal["mean", "sum"] = "mean"):
    """가로 정중앙 기준 width 폭을 잘라 세로(높이) 방향 프로파일 반환."""
    x0, x1 = strip_bounds(img.shape[1], width)
    strip = img[:, x0:x1]
    prof = np.nansum(strip, axis=1) if reduce == "sum" else np.nanmean(strip, axis=1)
    return np.nan_to_num(prof, nan=0.0), (x0, x1)


def detect_pixel_peaks(prof: FArray, prominence: float | None = None, distance: int = 5):
    """prominence=None 이면 프로파일 표준편차의 3배."""
    if prominence is None:
        prominence = 3 * np.std(prof)
    return find_peaks(prof, prominence=prominence, distance=distance)


def run(
    file: Path,
    strip_width: int = 100,
    reduce: Literal["mean", "sum"] = "mean",
    prominence: float | None = 1,
    min_distance: int = 10,
    save: bool | None = None,
    show: bool = True,
) -> plt.Figure:
    img = load_image(file)
    prof, (x0, x1) = center_strip_profile(img, strip_width, reduce)
    peaks, props = detect_pixel_peaks(prof, prominence, min_distance)
    y = np.arange(prof.size)

    print(f"파일: {file.name}  shape={img.shape}  strip x=[{x0}, {x1})")
    print(f"피크 {len(peaks)}개")
    print(f"{'row(y)':>8} {'intensity':>12} {'prominence':>12} {'FWHM(px)':>10}")
    widths = peak_widths(prof, peaks, rel_height=0.5)[0] if len(peaks) else []
    for p, pr, wd in zip(peaks, props["prominences"], widths):
        print(f"{p:>8d} {prof[p]:>12.2f} {pr:>12.2f} {wd:>10.2f}")

    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(13, 6), gridspec_kw={"width_ratios": [1, 1.6]})
    # 왼쪽: 이미지 + 스트립 위치
    vmax = np.nanpercentile(img, 99.5)
    ax0.imshow(img, cmap="viridis", vmin=0, vmax=vmax, aspect="auto")
    ax0.axvspan(x0, x1, color="r", alpha=0.3)
    ax0.set_title(f"{file.name}")
    ax0.set_xlabel("x (px)")
    ax0.set_ylabel("y (px)")

    # 오른쪽: 높이 방향 프로파일 + 피크
    ax1.plot(y, prof, lw=1)
    ax1.plot(peaks, prof[peaks], "rx", ms=8)
    for p in peaks:
        ax1.annotate(
            str(p),
            (p, prof[p]),
            textcoords="offset points",
            xytext=(0, 6),
            ha="center",
            fontsize=10,
            color="r",
        )
    ax1.set_xlabel("y (row, px)")
    ax1.set_ylabel(f"{reduce} intensity (x={x0}–{x1 - 1})")
    ax1.set_title(f"Center strip profile (width={strip_width}px), {len(peaks)} peaks")
    ax1.grid(alpha=0.3)

    fig.tight_layout()
    if saving() if save is None else save:
        save_file = out_dir(file.parent.name) / f"{file.stem}_center_profile.png"
        fig.savefig(save_file, dpi=150)
        print(f"그림 저장: {save_file}")
    if show:
        plt.show()
    return fig
