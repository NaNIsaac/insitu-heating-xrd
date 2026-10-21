import numpy as np
import pytest

from insitu_xrd.geometry import Geometry
from insitu_xrd.image import chi_map

GEO = Geometry(pixel=1e-4, sdd=0.05, xc=50.0, yc=50.0, alpha_deg=0.0, energy_kev=12.398420)


def test_wavelength():
    assert GEO.wavelength == pytest.approx(1.0, rel=1e-6)
    assert Geometry(energy_kev=6.19921).wavelength == pytest.approx(2.0, rel=1e-6)


def test_tth_q_d_known_values():
    lam = GEO.wavelength
    # 2θ = 60° → sinθ = 1/2 → q = 2π/λ, d = λ
    assert GEO.tth_to_q(60.0) == pytest.approx(2 * np.pi / lam)
    assert GEO.d_spacing(60.0) == pytest.approx(lam)
    assert GEO.q_to_tth(2 * np.pi / lam) == pytest.approx(60.0)


def test_tth_q_roundtrip_and_d_relation():
    tth = np.linspace(1.0, 120.0, 50)
    q = GEO.tth_to_q(tth)
    np.testing.assert_allclose(GEO.q_to_tth(q), tth, rtol=1e-12)
    np.testing.assert_allclose([GEO.d_spacing(t) for t in tth], 2 * np.pi / q, rtol=1e-12)


def test_q_beyond_limit_clips_to_180():
    assert GEO.q_to_tth(1e3) == pytest.approx(180.0)


def test_tth_map_flat_detector():
    tth = GEO.tth_map((101, 101))
    assert tth.shape == (101, 101)
    assert tth[50, 50] == pytest.approx(0.0, abs=1e-9)
    # 30 px off the PONI in any direction → atan(r / SDD)
    expect = np.rad2deg(np.arctan(30 * GEO.pixel / GEO.sdd))
    for y, x in ((50, 80), (50, 20), (20, 50), (80, 50)):
        assert tth[y, x] == pytest.approx(expect)
    assert tth[30, 90] == pytest.approx(
        np.rad2deg(np.arctan(np.hypot(40, 20) * GEO.pixel / GEO.sdd))
    )


def test_tth_map_tilted_arm():
    geo = Geometry(pixel=1e-4, sdd=0.05, xc=50.0, yc=50.0, alpha_deg=20.0)
    tth = geo.tth_map((101, 101))
    assert tth[50, 50] == pytest.approx(20.0)
    # up the image = +2θ along the scattering plane
    assert tth[20, 50] == pytest.approx(20.0 + np.rad2deg(np.arctan(30 * geo.pixel / geo.sdd)))
    assert tth[80, 50] == pytest.approx(20.0 - np.rad2deg(np.arctan(30 * geo.pixel / geo.sdd)))
    assert tth[50, 20] == pytest.approx(tth[50, 80])


def test_chi_map_directions():
    chi = chi_map(GEO, (101, 101))
    assert chi[20, 50] == pytest.approx(0.0)  # straight up = scattering plane
    assert chi[50, 80] == pytest.approx(90.0)
    assert chi[50, 20] == pytest.approx(-90.0)
