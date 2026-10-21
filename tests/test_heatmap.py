import warnings

import numpy as np
import pandas as pd
import pytest

from insitu_xrd.heatmap import HeatmapResult, beam_ok, regrid, time_mean


def test_time_mean_ignores_gaps_silently():
    Z = np.array([[1.0, np.nan, 3.0], [3.0, np.nan, np.nan]])
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        m = time_mean(Z)
    np.testing.assert_allclose(m, [2.0, np.nan, 3.0])


def test_beam_ok_flags_dark_frames():
    Z = np.ones((10, 50))
    Z[3] = 0.01
    ok = beam_ok(Z)
    assert ok.sum() == 9 and not ok[3]
    # per-group: a folder 10x dimmer is still fine against its own median
    Z[5:] *= 0.1
    ok = beam_ok(Z, groups=np.repeat([0, 1], 5))
    assert ok.sum() == 9 and not ok[3]


def test_regrid():
    x = np.linspace(10, 20, 11)
    Z = np.vstack([x, 2 * x])
    assert regrid(x, x, Z) is Z
    out = regrid(np.array([9.0, 12.5, 15.0]), x, Z)
    np.testing.assert_allclose(out, [[np.nan, 12.5, 15.0], [np.nan, 25.0, 30.0]])


def make_result() -> HeatmapResult:
    t = 1_700_000_000 + np.arange(6) * 30.0
    fr = pd.DataFrame(
        dict(
            file=[f"f{i}.h5" for i in range(6)],
            time_kst=["2026-10-04 10:00:00"] * 6,
            time_unix=t,
            time_rel=(t - t[0]) / 60,
        )
    )
    return HeatmapResult(np.linspace(10, 20, 4), np.arange(24.0).reshape(6, 4), fr, "min")


def test_heatmap_result_subset_and_nearest():
    res = make_result()
    assert res.scale == 60.0
    assert res.nearest(1.4) == 3  # frames every 0.5 min
    sub = res.subset(np.array([False, False, True, True, False, True]))
    np.testing.assert_allclose(sub.trel, [0.0, 0.5, 1.5])
    np.testing.assert_array_equal(sub.intensity[:, 0], [8, 12, 20])
    assert res.trel[2] == pytest.approx(1.0)  # original untouched
