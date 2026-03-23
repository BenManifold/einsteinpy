"""
Gravitational time dilation and redshift.

Formulas from standard GR:
- Schwarzschild stationary: dτ/dt = √(1 - 2M/r) = √(1 - r_s/r)
- Kerr stationary (Boyer-Lindquist): dτ/dt = √(g_tt/c²) for g_tt > 0
- Moving observer: dτ/dt = 1/u^0 where u^0 = dt/dτ from 4-velocity (via v0).
- Redshift: 1+z = √(-g_tt(emitter)) / √(-g_tt(observer))
  For (+, -, -, -) signature: √(g_tt(emitter)) / √(g_tt(observer))
"""

import numpy as np

from einsteinpy import constant
from einsteinpy.coordinates.utils import v0

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
        (v_r, v_theta, v_phi) in coordinate basis: v_r in m/s, v_theta and
        v_phi in rad/s. If None, assumes stationary observer.

    Returns
    -------
    float
        dτ/dt: proper time elapsed per unit coordinate time.

    Raises
    ------
    ValueError
        If position is inside horizon or ergosphere where a stationary
        observer cannot exist (g_tt ≤ 0).

    Notes
    -----
    For moving observers, uses :func:`einsteinpy.coordinates.utils.v0` to compute
    the 4-velocity time component (dt/dτ), then returns dτ/dt = 1/(dt/dτ).
    """
    x_vec = _position_to_x_vec(position)
    if hasattr(metric, "metric_covariant"):
        g = metric.metric_covariant(x_vec)
    else:
        g = metric(x_vec)

    if velocity is not None:
        vel = np.atleast_1d(np.asarray(velocity, dtype=float))
        if vel.size != 3:
            raise ValueError("velocity must have 3 elements (v_r, v_theta, v_phi)")
        v_r, v_th, v_phi = float(vel[0]), float(vel[1]), float(vel[2])
        dt_dtau = v0(g, v_r, v_th, v_phi)
        if dt_dtau <= 0 or not np.isfinite(dt_dtau):
            return np.nan
        return 1.0 / dt_dtau

    g_tt = g[0, 0]
    if g_tt <= 0:
        return np.nan

    return np.sqrt(g_tt / (_c ** 2))


def redshift_factor(
    metric,
    emitter_position,
    observer_position,
    emitter_velocity=None,
    observer_velocity=None,
):
    """
    1+z: factor by which photon wavelength shifts between emitter and observer.

    For two stationary observers: 1+z = √(g_tt(emitter)) / √(g_tt(observer)).
    For moving observers, uses the ratio of proper_time_ratio at each location.

    Parameters
    ----------
    metric : ~einsteinpy.metric.BaseMetric or callable
        Metric object.
    emitter_position : array_like
        (r, theta) or (r, theta, phi) of emitter.
    observer_position : array_like
        (r, theta) or (r, theta, phi) of observer.
    emitter_velocity : array_like, optional
        (v_r, v_theta, v_phi) of emitter. If None, emitter is stationary.
    observer_velocity : array_like, optional
        (v_r, v_theta, v_phi) of observer. If None, observer is stationary.

    Returns
    -------
    float
        (1+z) = λ_obs / λ_em. Redshift when > 1.
    """
    dtau_emit = proper_time_ratio(metric, emitter_position, velocity=emitter_velocity)
    dtau_obs = proper_time_ratio(metric, observer_position, velocity=observer_velocity)
    if np.isnan(dtau_emit) or np.isnan(dtau_obs):
        return np.nan
    return dtau_obs / dtau_emit
