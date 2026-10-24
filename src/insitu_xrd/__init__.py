"""Analysis tools for heating in-situ 2D XRD (Eiger HDF5 + temperature controller log), beamtime 261004.

Each feature can also be called from Python through its module's run():
    from insitu_xrd import heatmap, tth_profile
    heatmap.run(heatmap.Config(folder=Path(...)))
"""

from .geometry import Geometry

__all__ = ["Geometry"]
