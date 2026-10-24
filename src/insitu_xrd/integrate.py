"""Integrate an image into a 1D 2θ profile."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import NDArray

from .geometry import FArray, Geometry


@dataclass(frozen=True, slots=True)
class ProfileOptions:
    tth_bin: float = 0.02  # 2θ bin size [deg]
    region: Literal["full", "strip"] = "full"
    strip_width: int = 10  # width of the horizontal center strip for region="strip" [px]
    normalize: bool = False
    # bins with fewer pixels than this fraction of the median count are NaN (0 disables)
    min_fill: float = 0.2


def strip_bounds(img_width: int, width: int) -> tuple[int, int]:
    """[x0, x1) of a strip of the given width centered horizontally."""
    x0 = img_width // 2 - width // 2
    return x0, x0 + width


@dataclass(slots=True)
class RadialIntegrator:
    """Computes the pixel to 2θ bin index once, then integrates quickly with bincount."""

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
            prof = total / count  # empty bins (gaps) are NaN
        if self.min_fill > 0:
            prof[count < self.min_fill * np.median(count[count > 0])] = np.nan
        return prof
