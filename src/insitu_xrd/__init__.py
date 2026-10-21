"""승온 in-situ 2D XRD (Eiger HDF5 + 온도 컨트롤러 로그) 분석 도구 — 261004 빔타임.

각 기능은 모듈의 run() 으로 파이썬에서도 바로 호출 가능:
    from insitu_xrd import heatmap, tth_profile
    heatmap.run(heatmap.Config(folder=Path(...)))
"""

from .geometry import Geometry

__all__ = ["Geometry"]
