"""이미지 → 2θ 1D 프로파일 적분."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import NDArray

from .geometry import FArray, Geometry


@dataclass(frozen=True, slots=True)
class ProfileOptions:
    tth_bin: float = 0.02  # 2θ 빈 크기 [deg]
    region: Literal["full", "strip"] = "full"
    strip_width: int = 10  # region="strip" 일 때 가로 중앙 폭 [px]
    normalize: bool = False
    min_fill: float = 0.2  # 빈 픽셀 수가 중앙값의 이 비율 미만이면 NaN (0 이면 끔)


def strip_bounds(img_width: int, width: int) -> tuple[int, int]:
    """가로 정중앙 기준 width 폭의 [x0, x1) 범위."""
    x0 = img_width // 2 - width // 2
    return x0, x0 + width


@dataclass(slots=True)
class RadialIntegrator:
    """픽셀 → 2θ 빈 인덱스를 한 번만 계산해 두고 bincount 로 빠르게 적분."""

    selection: NDArray[np.bool_]
    bin_index: NDArray[np.intp]
    centers: FArray
    min_fill: float

    @classmethod
    def build(
        cls, geo: Geometry, opt: ProfileOptions, shape: tuple[int, int], tth: FArray | None = None
    ) -> RadialIntegrator:
        tth = geo.tth_map(shape) if tth is None else tth
        sel = np.ones(shape, dtype=bool)
        if opt.region == "strip":
            sel[:] = False
            x0, x1 = strip_bounds(shape[1], opt.strip_width)
            sel[:, x0:x1] = True
        t = tth[sel]
        edges = np.arange(t.min(), t.max() + opt.tth_bin, opt.tth_bin)
        centers = (edges[1:] + edges[:-1]) / 2
        idx = np.clip(np.digitize(t, edges) - 1, 0, centers.size - 1)
        return cls(sel, idx, centers, opt.min_fill)

    def __call__(self, img: FArray) -> FArray:
        values = img[self.selection]
        ok = np.isfinite(values)
        n_bins = self.centers.size
        total = np.bincount(self.bin_index[ok], weights=values[ok], minlength=n_bins)
        count = np.bincount(self.bin_index[ok], minlength=n_bins)
        with np.errstate(invalid="ignore", divide="ignore"):
            prof = total / count  # 데이터 없는 빈(갭)은 NaN
        if self.min_fill > 0:
            prof[count < self.min_fill * np.median(count[count > 0])] = np.nan
        return prof
