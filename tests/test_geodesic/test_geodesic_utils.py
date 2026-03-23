import numpy as np
import pytest
from numpy.testing import assert_allclose

from einsteinpy.geodesic.utils import (
    _P,
    _r_from_kerrschild_cartesian,
    _sch,
    _kerr,
    _kerr_ks,
    _kerrnewman,
)


@pytest.mark.parametrize(
    "g, g_prms, q, p, time_like, expected",
    [
        (
            _sch,
            (),
            [0., 2.5, np.pi / 6, np.pi / 2],
            [0.1, 2., 2.],
            True,
            [-0.9167333309092672, 0.1, 2., 2.]
        ),
        (
            _kerr,
            (0.9,),
            [0., 25, np.pi / 2, 0.],
            [0., 0, 2.427],
            False,
            [-0.09333038545829885, 0., 0, 2.427]
        ),
        (
            _kerrnewman,
            (0.5, 0.1,),
            [0., 5.5, np.pi / 4, 0.],
            [0.1, -0.2, -4.],
            True,
            [-1.1220908695019767, 0.1, -0.2, -4.]
        ),
    ],
)
def test_P(g, g_prms, q, p, time_like, expected):
    P = _P(g, g_prms, q, p, time_like)

    assert_allclose(P, expected, atol=1e-8, rtol=1e-8)


@pytest.mark.parametrize(
    "x",
    [
        (
            [0., 2.5, np.pi / 6, np.pi / 2],
        ),
        (
            [0., 25, np.pi / 2, 0.],
        ),
        (
            [0., 5.5, np.pi / 4, 0.],
        ),
    ],
)
def test_metrics(x):
    x = x[0]

    # a = Q = 0.
    s = _sch(x).astype(float)
    k = _kerr(x, 0.).astype(float)
    kn = _kerrnewman(x, 0., 0.).astype(float)
    assert_allclose(s, k, atol=1e-8, rtol=1e-8)
    assert_allclose(k, kn, atol=1e-8, rtol=1e-8)
    assert_allclose(kn, s, atol=1e-8, rtol=1e-8)

    # Non-zero Spin
    k = _kerr(x, 0.4).astype(float)
    kn = _kerrnewman(x, 0.4, 0.).astype(float)
    assert_allclose(k, kn, atol=1e-8, rtol=1e-8)


def test_r_from_kerrschild_cartesian():
    """r from Kerr-Schild Cartesian matches oblate spheroidal definition."""
    a = 0.9
    # At r=6, theta=pi/2, phi=0: x = sqrt(r^2+a^2), y=0, z=0
    r_bl = 6.0
    xa = np.sqrt(r_bl**2 + a**2)
    x, y, z = xa, 0.0, 0.0
    r = _r_from_kerrschild_cartesian(x, y, z, a)
    assert_allclose(r, r_bl, atol=1e-10, rtol=1e-10)


def test_kerr_ks_metric_shape():
    """Kerr-Schild metric returns 4x4 array."""
    a = 0.9
    x_vec = np.array([0.0, 6.0, 0.0, 0.0])  # t, x, y, z with y=z=0
    g = _kerr_ks(x_vec, a)
    assert g.shape == (4, 4)
    # Should be symmetric
    g_vals = np.array([[float(g[i, j]) for j in range(4)] for i in range(4)])
    assert_allclose(g_vals, g_vals.T, atol=1e-10)


def test_kerr_ks_regular_near_horizon():
    """Kerr-Schild metric stays finite near horizon (r ~ 1.5 * r_horizon)."""
    a = 0.9
    r_horizon = 1.0 + np.sqrt(1.0 - a**2)
    r = 1.5 * r_horizon  # Close to horizon
    th, ph = np.pi / 2, 0.0
    xa = np.sqrt(r**2 + a**2)
    x, y, z = xa * np.cos(ph), xa * np.sin(ph), r * np.cos(th)
    q_ks = np.array([0.0, x, y, z])
    g = _kerr_ks(q_ks, a).astype(float)
    # All components should be finite (no overflow from 1/Delta)
    assert np.all(np.isfinite(g))
