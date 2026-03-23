"""
Geodesic orbit simulation with time dilation.

Bridges einsteinpy geodesic (M-units) and metric (SI) APIs.
"""

import logging
import numpy as np

log = logging.getLogger("einsteinpy.orbit_sim")
from astropy import units as u

from einsteinpy import constant
from einsteinpy.coordinates import BoyerLindquistDifferential, SphericalDifferential
from einsteinpy.coordinates.utils import kerrschild_to_bl_cartesian
from einsteinpy.geodesic import Timelike
from einsteinpy.metric import Kerr, Schwarzschild
from einsteinpy.utils import proper_time_ratio

# Mass for which r_s = 1 m. Geodesic r in GM/c²; r_m = r_geod * (r_s/2).
_DEFAULT_MASS = constant.Mass_Schwarzschild_Radius_1m

# Mass for which 1 geometrized time unit = 1 year (GM/c³ = 1 yr).
_MASS_1YEAR = (
    (constant.c ** 3 / constant.G) * (1.0 * u.yr).to(u.s)
).to(u.kg)

# Time dilation limit: The Kerr metric in Boyer-Lindquist coordinates is singular
# at the horizon (Δ = r² - 2r + a² → 0). Terms ∝ 1/Δ overflow before we reach
# Interstellar-level dilation (1 hr = 7 yr). We stay at r ≳ 3–4 × horizon.


def _kerr_horizon_radius(a):
    """Outer horizon radius in geometrized units: r_+ = 1 + sqrt(1 - a^2)."""
    a = float(a)
    if a >= 1:
        return 1.0
    return 1.0 + np.sqrt(1.0 - a**2)


def schwarzschild_circular_L(r):
    """
    Angular momentum for a stable circular orbit in Schwarzschild at radius r.

    Requires r > 6 (ISCO). In geometrized units: L = r / sqrt(r - 3).
    """
    r = float(r)
    if r <= 3:
        return np.nan
    return r / np.sqrt(r - 3)


def kerr_circular_L(r, a):
    """
    Angular momentum for a prograde circular orbit in Kerr at radius r.

    From Bardeen et al. (1972), equatorial circular geodesics:
    L = (a² - 2a√r + r²) / √(2a·r^(3/2) + (r - 3)r²)

    Requires r > r_ISCO (e.g. r > 3 for a=0, r > ~2.3 for a=1).
    Returns np.nan if orbit is unstable (denominator <= 0 or complex).
    """
    r, a = float(r), float(a)
    sqrt_r = np.sqrt(r)
    denom = 2 * a * r**1.5 + (r - 3) * r**2
    if denom <= 0:
        return np.nan
    L = (a**2 - 2 * a * sqrt_r + r**2) / np.sqrt(denom)
    return L if np.isfinite(L) else np.nan


def circular_L(r, spin):
    """
    Angular momentum for stable circular orbit at radius r.
    Uses Schwarzschild or Kerr formula depending on spin.
    """
    if abs(spin) < 1e-10:
        return schwarzschild_circular_L(r)
    return kerr_circular_L(r, spin)


def validate_params(r_init, L, spin, metric, use_kerrschild=False):
    """
    Check for parameter regimes that cause numerical issues or long runtimes.

    Parameters
    ----------
    r_init, L, spin, metric
        Orbit parameters.
    use_kerrschild : bool
        If True and metric is Kerr, use KerrSchild coordinates (horizon-penetrating).
        Allows r_init closer to horizon (r_s + 0.01).

    Returns
    -------
    ok : bool
        True if params are reasonable.
    message : str
        Empty if ok; otherwise a warning or error message.
    """
    r_init = float(r_init)
    spin = float(spin)
    use_kerr = metric == "Kerr" and abs(spin) > 1e-10
    r_s = _kerr_horizon_radius(spin) if use_kerr else 2.0

    min_r = r_s + 0.01 if (use_kerr and use_kerrschild) else r_s + 0.1
    if r_init <= min_r:
        log.debug(
            "validate_params: r_init=%.2f <= min_r=%.2f", r_init, min_r
        )
        return False, "Too close to horizon"
    if r_init < 6 and not use_kerr:
        return True, "Close orbit — may be slow"
    if abs(spin) > 0.98:
        return True, "High spin — may be unstable"
    return True, ""


def _geod_to_meters(r_geod, r_s_m):
    """Convert geodesic r (in GM/c²) to meters. r_s = 2GM/c², so GM/c² = r_s/2."""
    return np.asarray(r_geod, dtype=float) * float(r_s_m) / 2.0


def compute_trajectory(
    r_init=25.0,
    angular_momentum=3.5,
    metric="Schwarzschild",
    spin=0.0,
    steps=6000,
    delta=0.3,
    mass=_MASS_1YEAR,
    omega=None,
    use_kerrschild=False,
):
    """
    Compute timelike geodesic orbit and time dilation along path.

    Parameters
    ----------
    r_init : float
        Initial radial position in geometrized units (r in GM/c²).
    angular_momentum : float
        Conserved angular momentum L (dimensionless in M-units).
    metric : str
        "Schwarzschild" or "Kerr".
    spin : float
        Kerr spin parameter a (0 <= a <= 1). Ignored for Schwarzschild.
    steps : int
        Number of integration steps.
    delta : float
        Integration step size.
    mass : ~astropy.units.Quantity
        Black hole mass (for time dilation in SI).
    use_kerrschild : bool
        If True and metric is Kerr, use Kerr-Schild coordinates for integration.
        Enables stable integration closer to the horizon (r ≳ r_s + 0.01).
        Defaults to False.

    Returns
    -------
    dict
        Contains:
        - t, x, y, z : 1D arrays (geometrized units)
        - r, theta, phi : 1D arrays (spherical, r in geometrized)
        - r_m : r in meters
        - dtau_dt : proper_time_ratio at each point
        - tau : accumulated proper time (τ) along trajectory
        - metric : metric instance
        - r_s_geod, r_s_m : horizon radius for viz
    """
    if angular_momentum is None:
        angular_momentum = schwarzschild_circular_L(r_init) if metric != "Kerr" or abs(spin) < 1e-10 else 4.0
    angular_momentum = float(angular_momentum)

    # Use smaller omega for plunging / near-horizon orbits. 0.01 can cause
    # non-integrability ("jacked" orbits); 0.05 balances stability vs integrability.
    if omega is None:
        omega = 0.05 if r_init < 30 or (angular_momentum or 4) < 4.5 else 1.0

    log.debug(
        "compute_trajectory: r_init=%.2f L=%.2f metric=%s spin=%.2f steps=%d delta=%.2f omega=%.2f",
        r_init, angular_momentum, metric, spin, steps, delta, omega,
    )

    position = [float(r_init), np.pi / 2, 0.0]
    momentum = [0.0, 0.0, angular_momentum]
    use_kerr = metric == "Kerr" and abs(spin) > 1e-10
    metric_params = (float(spin),) if use_kerr else (0.0,)
    metric_name = "Kerr" if use_kerr else "Schwarzschild"

    coords = "KerrSchild" if (use_kerrschild and use_kerr) else "BoyerLindquist"
    geod = Timelike(
        metric=metric_name,
        metric_params=metric_params,
        position=position,
        momentum=momentum,
        steps=steps,
        delta=delta,
        omega=omega,
        return_cartesian=True,
        coords=coords,
        suppress_warnings=True,
    )

    steps_arr, results = geod.trajectory
    log.debug("compute_trajectory: geodesic integration done, %d steps", len(results))

    t = results[:, 0]
    x, y, z = results[:, 1], results[:, 2], results[:, 3]
    # Reconstruct r, theta, phi. For KerrSchild, Cartesian is oblate; use BL transform.
    if use_kerrschild and use_kerr:
        _, r_geod, theta, phi = kerrschild_to_bl_cartesian(t, x, y, z, spin)
    else:
        r_geod = np.sqrt(x**2 + y**2 + z**2)
        theta = np.arccos(np.clip(z / (r_geod + 1e-12), -1, 1))
        phi = np.arctan2(y, x + 1e-12)

    # Metric for time dilation (SI)
    if use_kerr:
        bl = BoyerLindquistDifferential(
            t=0 * u.s,
            r=100 * u.m,
            theta=np.pi / 2 * u.rad,
            phi=0 * u.rad,
            v_r=0 * u.m / u.s,
            v_th=0 * u.rad / u.s,
            v_p=0 * u.rad / u.s,
        )
        phys_metric = Kerr(coords=bl, M=mass, a=spin * u.one)
    else:
        sph = SphericalDifferential(
            t=0 * u.s,
            r=100 * u.m,
            theta=np.pi / 2 * u.rad,
            phi=0 * u.rad,
            v_r=0 * u.m / u.s,
            v_th=0 * u.rad / u.s,
            v_p=0 * u.rad / u.s,
        )
        phys_metric = Schwarzschild(coords=sph, M=mass)
    sr = phys_metric.sch_rad
    r_s_m = float(sr.to(u.m).value) if hasattr(sr, "to") else float(sr)

    r_m = _geod_to_meters(r_geod, r_s_m)
    dtau_dt = np.zeros_like(r_m)
    for i in range(len(r_m)):
        if r_m[i] > r_s_m * 1.001:
            dtau_dt[i] = proper_time_ratio(
                phys_metric, (r_m[i], theta[i], phi[i])
            )
        else:
            dtau_dt[i] = np.nan

    # Accumulated proper time: dτ = (dτ/dt) * dt. dt between steps from geodesic.
    dt = np.diff(t, prepend=t[0])
    tau = np.zeros_like(t)
    for i in range(1, len(t)):
        tau[i] = tau[i - 1] + dtau_dt[i - 1] * dt[i]

    r_s_geod = _kerr_horizon_radius(spin) if use_kerr else 2.0

    # Truncate at event horizon crossing
    inside = r_geod <= r_s_geod
    hit_horizon = np.any(inside)
    if hit_horizon:
        # Keep points strictly outside; last valid index is before first inside
        first_inside = int(np.argmax(inside))
        last_valid = max(1, first_inside)  # keep at least 1 point for display
        log.debug("compute_trajectory: hit horizon at step %d, truncated to %d points", first_inside, last_valid)
        t, x, y, z = t[:last_valid], x[:last_valid], y[:last_valid], z[:last_valid]
        r_geod = r_geod[:last_valid]
        theta, phi = theta[:last_valid], phi[:last_valid]
        r_m = r_m[:last_valid]
        dtau_dt = dtau_dt[:last_valid]
        tau = tau[:last_valid]

    return {
        "t": t,
        "x": x,
        "y": y,
        "z": z,
        "r": r_geod,
        "theta": theta,
        "phi": phi,
        "r_m": r_m,
        "dtau_dt": dtau_dt,
        "tau": tau,
        "metric": phys_metric,
        "r_s_geod": r_s_geod,
        "r_s_m": r_s_m,
        "hit_horizon": hit_horizon,
    }
