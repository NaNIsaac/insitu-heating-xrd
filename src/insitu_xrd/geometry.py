"""디텍터 기하 (캘리브레이션 값) 와 2θ / q / d 변환."""
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from qorageom import geometry as qg

from . import config as C

FArray = NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class Geometry:
    """기본값은 config.py 의 캘리브레이션 블록 (내부 단위: 거리 m)."""
    pixel: float = C.PIXEL_UM * 1e-6     # 픽셀 크기 [m]
    sdd: float = C.SDD_MM * 1e-3         # 시료 → PONI 거리 [m]
    xc: float = C.PONI_PX[0]             # PONI x [px]
    yc: float = C.PONI_PX[1]             # PONI y [px]
    alpha_deg: float = C.ALPHA_DEG       # 디텍터 암의 2θ 모터 값 [deg]
    energy_kev: float = C.ENERGY_KEV

    @property
    def wavelength(self) -> float:  # [Å]
        return float(qg.wavelength(self.energy_kev))

    def tth_map(self, shape: tuple[int, int]) -> FArray:
        """각 픽셀의 true 2θ [deg]. 이미지 위쪽이 +2θ, α=0 이면 2θ = atan(r / SDD)."""
        return qg.tth_map(shape, self.pixel, self.sdd, self.xc, self.yc, self.alpha_deg)

    def tth_to_q(self, tth: FArray | float) -> FArray:
        return qg.tth_to_q(tth, self.wavelength)

    def q_to_tth(self, q: FArray | float) -> FArray:
        return qg.q_to_tth(q, self.wavelength)

    def d_spacing(self, tth: float) -> float:
        return qg.tth_to_d(tth, self.wavelength)
