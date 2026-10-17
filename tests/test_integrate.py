import numpy as np
import pytest

from insitu_xrd.geometry import Geometry
from insitu_xrd.image import cake, chi_map
from insitu_xrd.integrate import ProfileOptions, RadialIntegrator, strip_bounds
from insitu_xrd.peaks import fit_peak

GEO = Geometry(pixel=1e-4, sdd=0.05, xc=100.0, yc=100.0, alpha_deg=0.0, energy_kev=12.0)
SHAPE = (201, 201)
RING_TTH, RING_SIG = 8.0, 0.15


def ring_image(base: float = 1.0) -> np.ndarray:
    """Gaussian ring at RING_TTH, built from the pixel radius (independent of Geometry.tth_map)."""
    y, x = np.mgrid[: SHAPE[0], : SHAPE[1]]
    r = np.hypot(x - GEO.xc, y - GEO.yc) * GEO.pixel
    tth = np.rad2deg(np.arctan(r / GEO.sdd))
    return base + 100 * np.exp(-0.5 * ((tth - RING_TTH) / RING_SIG) ** 2)


def test_strip_bounds():
    assert strip_bounds(100, 10) == (45, 55)
    assert strip_bounds(1030, 11) == (510, 521)


@pytest.mark.parametrize("region", ["full", "strip"])
def test_ring_profile_peak(region):
    opt = ProfileOptions(tth_bin=0.05, region=region, strip_width=20, min_fill=0)
    integ = RadialIntegrator.build(GEO, opt, SHAPE)
    prof = integ(ring_image())
    x = integ.centers
    assert np.nanmax(prof) == pytest.approx(101, rel=0.05)
    assert abs(x[np.nanargmax(prof)] - RING_TTH) <= opt.tth_bin
    f = fit_peak(x, prof, x[np.nanargmax(prof)], half_window=0.6)
    assert f.center == pytest.approx(RING_TTH, abs=0.01)
    assert f.fwhm == pytest.approx(2.3548 * RING_SIG, rel=0.1)


def test_constant_image_gives_constant_profile():
    integ = RadialIntegrator.build(GEO, ProfileOptions(min_fill=0), SHAPE)
    prof = integ(np.full(SHAPE, 3.0))
    np.testing.assert_allclose(prof[np.isfinite(prof)], 3.0)
    assert np.isfinite(prof).mean() > 0.9


def test_nan_pixels_are_ignored():
    img = ring_image()
    img[:, 95:105] = np.nan                         # detector gap
    integ = RadialIntegrator.build(GEO, ProfileOptions(tth_bin=0.05, min_fill=0), SHAPE)
    prof = integ(img)
    assert abs(integ.centers[np.nanargmax(prof)] - RING_TTH) <= 0.05


def test_min_fill_drops_sparse_bins():
    integ = RadialIntegrator.build(GEO, ProfileOptions(tth_bin=0.05, min_fill=0.2), SHAPE)
    prof = integ(np.ones(SHAPE))
    # first bins hold only a handful of pixels around the PONI
    assert np.isnan(prof[0])
    assert np.isfinite(prof[len(prof) // 2])


def test_cake_profile_peak():
    tth, chi = GEO.tth_map(SHAPE), chi_map(GEO, SHAPE)
    te, ce, Z, prof = cake(ring_image(), tth, chi, 0.05, 5.0)
    assert Z.shape == (len(ce) - 1, len(te) - 1)
    centers = (te[1:] + te[:-1]) / 2
    assert abs(centers[np.nanargmax(prof)] - RING_TTH) <= 0.05
