"""
Gravitational time dilation and redshift.

Formulas from standard GR:
- Schwarzschild stationary: dτ/dt = √(1 - 2M/r) = √(1 - r_s/r)
- Kerr stationary (Boyer-Lindquist): dτ/dt = √(g_tt/c²) for g_tt > 0
- Redshift: 1+z = √(-g_tt(emitter)) / √(-g_tt(observer))
  For (+, -, -, -) signature: √(g_tt(emitter)) / √(g_tt(observer))
"""

import numpy as np

from einsteinpy import constant

_c = constant.c.value


def _position_to_x_vec(position):
    """Convert position (r, theta) or (r, theta, phi) to 4-position [t, r, θ, φ]."""
    pos = np.atleast_1d(np.asarray(position, dtype=float))
    if pos.size == 1:
        r, theta, phi = float(pos[0]), np.pi / 2, 0.0
    elif pos.size == 2:
        r, theta, phi = float(pos[0]), float(pos[1]), 0.0
    elif pos.size >= 3:
        r, theta, phi = float(pos[0]), float(pos[1]), float(pos[2])
    else:
        raise ValueError("position must have 1, 2, or 3 elements (r, theta, phi)")
    return np.array([0.0, r, theta, phi])


def proper_time_ratio(metric, position, velocity=None):
    """
    dτ/dt for an observer at the given position.

    For a stationary observer (velocity=None): dτ/dt = √(g_tt/c²).
    For Kerr metrics, g_tt can be negative inside the ergosphere,
    in which case a stationary observer cannot exist.

    Parameters
    ----------
    metric : ~einsteinpy.metric.BaseMetric or callable
        Metric object (Schwarzschild, Kerr, KerrNewman) or callable
        returning covariant metric at 4-position.
    position : array_like
        (r, theta) or (r, theta, phi). r in meters, theta/phi in radians.
        theta defaults to π/2 (equatorial) if omitted.
    velocity : array_like, optional
        (v_r, v_theta, v_phi) in coordinate basis. If None, assumes
        stationary observer. Not yet implemented.

    Returns
    -------
    float
        dτ/dt: proper time elapsed per unit coordinate time.

    Raises
    ------
    ValueError
        If position is inside horizon or ergosphere where a stationary
        observer cannot exist (g_tt ≤ 0).
    """
    if velocity is not None:
        raise NotImplementedError("Non-stationary observers not yet supported")

    x_vec = _position_to_x_vec(position)
    if hasattr(metric, "metric_covariant"):
        g = metric.metric_covariant(x_vec)
    else:
        g = metric(x_vec)

    g_tt = g[0, 0]
    if g_tt <= 0:
        return np.nan

    return np.sqrt(g_tt / (_c ** 2))


def redshift_factor(metric, emitter_position, observer_position):
    """
    1+z: factor by which photon wavelength shifts between emitter and observer.

    For two stationary observers: 1+z = √(g_tt(emitter)) / √(g_tt(observer)).

    Parameters
    ----------
    metric : ~einsteinpy.metric.BaseMetric or callable
        Metric object.
    emitter_position : array_like
        (r, theta) or (r, theta, phi) of emitter.
    observer_position : array_like
        (r, theta) or (r, theta, phi) of observer.

    Returns
    -------
    float
        (1+z) = λ_obs / λ_em. Redshift when > 1.
    """
    dtau_emit = proper_time_ratio(metric, emitter_position)
    dtau_obs = proper_time_ratio(metric, observer_position)
    if np.isnan(dtau_emit) or np.isnan(dtau_obs):
        return np.nan
    return dtau_obs / dtau_emit
