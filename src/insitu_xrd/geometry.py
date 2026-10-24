"""Detector geometry (calibration values) and 2θ / q / d conversion."""

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from qorageom import geometry as qg

from . import config as C

FArray = NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class Geometry:
    """Defaults come from the calibration block in config.py (distances in m internally)."""

    pixel: float = C.PIXEL_UM * 1e-6  # pixel size [m]
    sdd: float = C.SDD_MM * 1e-3  # sample to PONI distance [m]
    xc: float = C.PONI_PX[0]  # PONI x [px]
    yc: float = C.PONI_PX[1]  # PONI y [px]
    alpha_deg: float = C.ALPHA_DEG  # 2θ motor value of the detector arm [deg]
    energy_kev: float = C.ENERGY_KEV

    @property
    def wavelength(self) -> float:  # [Å]
        return float(qg.wavelength(self.energy_kev))

    def tth_map(self, shape: tuple[int, int]) -> FArray:
        """True 2θ of each pixel [deg]. The top of the image is +2θ; for α=0, 2θ = atan(r / SDD)."""
        return qg.tth_map(shape, self.pixel, self.sdd, self.xc, self.yc, self.alpha_deg)

    def tth_to_q(self, tth: FArray | float) -> FArray:
        return qg.tth_to_q(tth, self.wavelength)

    def q_to_tth(self, q: FArray | float) -> FArray:
        return qg.q_to_tth(q, self.wavelength)

    def d_spacing(self, tth: float) -> float:
        return qg.tth_to_d(tth, self.wavelength)
