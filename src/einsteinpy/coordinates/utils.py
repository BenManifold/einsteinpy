import numpy as np

from einsteinpy import constant
from einsteinpy.ijit import jit

_c = constant.c.value


def cartesian_to_spherical_fast(
    t, x, y, z, v_x=None, v_y=None, v_z=None, velocities_provided=False
):
    if velocities_provided:
        return cartesian_to_spherical(t, x, y, z, v_x, v_y, v_z)
    return cartesian_to_spherical_novel(t, x, y, z)


@jit
def cartesian_to_spherical(t, x, y, z, v_x, v_y, v_z):
    """
    Utility function (jitted) to convert cartesian to spherical.
    This function should eventually result in Coordinate Transformation Graph!

    """
    hxy = np.hypot(x, y)
    r = np.hypot(hxy, z)
    theta = np.arctan2(hxy, z)
    phi = np.arctan2(y, x)
    n1 = x**2 + y**2
    n2 = n1 + z**2
    v_r = (x * v_x + y * v_y + z * v_z) / np.sqrt(n2)
    v_th = (z * (x * v_x + y * v_y) - n1 * v_z) / (n2 * np.sqrt(n1))
    v_p = -1 * (v_x * y - x * v_y) / n1

    return t, r, theta, phi, v_r, v_th, v_p


@jit
def cartesian_to_spherical_novel(t, x, y, z):
    """
    Utility function (jitted) to convert cartesian to spherical.
    This function should eventually result in Coordinate Transformation Graph!

    """
    hxy = np.hypot(x, y)
    r = np.hypot(hxy, z)
    theta = np.arctan2(hxy, z)
    phi = np.arctan2(y, x)

    return t, r, theta, phi


def cartesian_to_bl_fast(
    t, x, y, z, alpha, v_x=None, v_y=None, v_z=None, velocities_provided=False
):
    if velocities_provided:
        return cartesian_to_bl(t, x, y, z, alpha, v_x, v_y, v_z)
    return cartesian_to_bl_novel(t, x, y, z, alpha)


@jit
def cartesian_to_bl(t, x, y, z, alpha, v_x, v_y, v_z):
    """
    Utility function (jitted) to convert cartesian to boyer lindquist.
    This function should eventually result in Coordinate Transformation Graph!

    """
    w = (x**2 + y**2 + z**2) - (alpha**2)
    r = np.sqrt(0.5 * (w + np.sqrt((w**2) + (4 * (alpha**2) * (z**2)))))
    theta = np.arccos(z / r)
    phi = np.arctan2(y, x)
    dw_dt = 2 * (x * v_x + y * v_y + z * v_z)
    v_r = (1 / (2 * r)) * (
        (dw_dt / 2)
        + (
            (w * dw_dt + 4 * (alpha**2) * z * v_z)
            / (2 * np.sqrt((w**2) + (4 * (alpha**2) * (z**2))))
        )
    )
    v_th = (-1 / np.sqrt(1 - np.square(z / r))) * ((v_z * r - v_r * z) / (r**2))
    v_p = (1 / (1 + np.square(y / x))) * ((v_y * x - v_x * y) / (x**2))

    return t, r, theta, phi, v_r, v_th, v_p


@jit
def cartesian_to_bl_novel(t, x, y, z, alpha):
    """
    Utility function (jitted) to convert cartesian to boyer lindquist.
    This function should eventually result in Coordinate Transformation Graph!

    """
    w = (x**2 + y**2 + z**2) - (alpha**2)
    r = np.sqrt(0.5 * (w + np.sqrt((w**2) + (4 * (alpha**2) * (z**2)))))
    theta = np.arccos(z / r)
    phi = np.arctan2(y, x)

    return t, r, theta, phi


def spherical_to_cartesian_fast(
    t, r, th, p, v_r=None, v_th=None, v_p=None, velocities_provided=False
):
    if velocities_provided:
        return spherical_to_cartesian(t, r, th, p, v_r, v_th, v_p)
    return spherical_to_cartesian_novel(t, r, th, p)


@jit
def spherical_to_cartesian(t, r, th, p, v_r, v_th, v_p):
    """
    Utility function (jitted) to convert spherical to cartesian.
    This function should eventually result in Coordinate Transformation Graph!

    """
    x = r * np.cos(p) * np.sin(th)
    y = r * np.sin(p) * np.sin(th)
    z = r * np.cos(th)
    v_x = (
        np.sin(th) * np.cos(p) * v_r
        - r * np.sin(th) * np.sin(p) * v_p
        + r * np.cos(th) * np.cos(p) * v_th
    )
    v_y = (
        np.sin(th) * np.sin(p) * v_r
        + r * np.cos(th) * np.sin(p) * v_th
        + r * np.sin(th) * np.cos(p) * v_p
    )
    v_z = np.cos(th) * v_r - r * np.sin(th) * v_th

    return t, x, y, z, v_x, v_y, v_z


@jit
def spherical_to_cartesian_novel(t, r, th, p):
    """
    Utility function (jitted) to convert spherical to cartesian.
    This function should eventually result in Coordinate Transformation Graph!

    """
    x = r * np.cos(p) * np.sin(th)
    y = r * np.sin(p) * np.sin(th)
    z = r * np.cos(th)

    return t, x, y, z


def bl_to_kerrschild_cartesian(t, r, th, p, alpha):
    """
    Convert Boyer-Lindquist (t, r, θ, φ) to Kerr-Schild Cartesian (t, x, y, z).

    Spatial coordinates use the oblate spheroidal mapping. Time is unchanged
    (same asymptotic time). Uses geometric units (M-units).

    Parameters
    ----------
    t : float or array
        Time coordinate
    r, th, p : float or array
        BL radial, polar, azimuthal coordinates
    alpha : float
        Spin parameter a (in geometric units)

    Returns
    -------
    t, x, y, z : same type as inputs
        Kerr-Schild Cartesian coordinates
    """
    xa = np.sqrt(r**2 + alpha**2)
    sin_norm = xa * np.sin(th)
    x = sin_norm * np.cos(p)
    y = sin_norm * np.sin(p)
    z = r * np.cos(th)
    return t, x, y, z


def bl_to_kerrschild_cartesian_with_momentum(
    t, r, th, p, alpha, p_t, p_r, p_th, p_phi
):
    """
    Convert BL position and 4-momentum to Kerr-Schild Cartesian.

    The spatial momentum (p^x, p^y, p^z) is obtained via the Jacobian of
    (x, y, z) w.r.t. (r, θ, φ). p^t is unchanged.

    Parameters
    ----------
    t, r, th, p : float
        BL position
    alpha : float
        Spin parameter
    p_t, p_r, p_th, p_phi : float
        Contravariant 4-momentum in BL

    Returns
    -------
    t, x, y, z : float
        Position in Kerr-Schild Cartesian
    p_t, p_x, p_y, p_z : float
        4-momentum in Kerr-Schild Cartesian
    """
    xa = np.sqrt(r**2 + alpha**2)
    sin_th = np.sin(th)
    cos_th = np.cos(th)
    sin_p = np.sin(p)
    cos_p = np.cos(p)
    x = xa * sin_th * cos_p
    y = xa * sin_th * sin_p
    z = r * cos_th
    # Jacobian rows: (dx, dy, dz) = J @ (dr, dth, dphi)
    Jxr = r / xa * sin_th * cos_p
    Jxth = xa * cos_th * cos_p
    Jxphi = -xa * sin_th * sin_p
    Jyr = r / xa * sin_th * sin_p
    Jyth = xa * cos_th * sin_p
    Jyphi = xa * sin_th * cos_p
    Jzr = cos_th
    Jzth = -r * sin_th
    Jzphi = 0.0
    p_x = Jxr * p_r + Jxth * p_th + Jxphi * p_phi
    p_y = Jyr * p_r + Jyth * p_th + Jyphi * p_phi
    p_z = Jzr * p_r + Jzth * p_th + Jzphi * p_phi
    return t, x, y, z, p_t, p_x, p_y, p_z


def kerrschild_to_bl_cartesian(t, x, y, z, alpha):
    """
    Convert Kerr-Schild Cartesian (t, x, y, z) to Boyer-Lindquist (t, r, θ, φ).

    Inverse of bl_to_kerrschild_cartesian. Uses cartesian_to_bl logic.

    Parameters
    ----------
    t, x, y, z : float or array
        Kerr-Schild Cartesian coordinates
    alpha : float
        Spin parameter

    Returns
    -------
    t, r, th, p : same type as inputs
        Boyer-Lindquist coordinates
    """
    w = (x**2 + y**2 + z**2) - (alpha**2)
    r = np.sqrt(0.5 * (w + np.sqrt((w**2) + (4 * (alpha**2) * (z**2)))))
    th = np.arccos(z / np.maximum(r, 1e-30))
    p = np.arctan2(y, x)
    return t, r, th, p


def bl_to_cartesian_fast(
    t, r, th, p, alpha, v_r=None, v_th=None, v_p=None, velocities_provided=False
):
    if velocities_provided:
        return bl_to_cartesian(t, r, th, p, alpha, v_r, v_th, v_p)
    return bl_to_cartesian_novel(t, r, th, p, alpha)


@jit
def bl_to_cartesian(t, r, th, p, alpha, v_r, v_th, v_p):
    """
    Utility function (jitted) to convert bl to cartesian.
    This function should eventually result in Coordinate Transformation Graph!

    """
    xa = np.sqrt(r**2 + alpha**2)
    sin_norm = xa * np.sin(th)
    x = sin_norm * np.cos(p)
    y = sin_norm * np.sin(p)
    z = r * np.cos(th)
    v_x = (
        (r * v_r * np.sin(th) * np.cos(p) / xa)
        + (xa * np.cos(th) * np.cos(p) * v_th)
        - (xa * np.sin(th) * np.sin(p) * v_p)
    )
    v_y = (
        (r * v_r * np.sin(th) * np.sin(p) / xa)
        + (xa * np.cos(th) * np.sin(p) * v_th)
        + (xa * np.sin(th) * np.cos(p) * v_p)
    )
    v_z = (v_r * np.cos(th)) - (r * np.sin(th) * v_th)

    return t, x, y, z, v_x, v_y, v_z


@jit
def bl_to_cartesian_novel(t, r, th, p, alpha):
    """
    Utility function (jitted) to convert bl to cartesian.
    This function should eventually result in Coordinate Transformation Graph!

    """
    xa = np.sqrt(r**2 + alpha**2)
    sin_norm = xa * np.sin(th)
    x = sin_norm * np.cos(p)
    y = sin_norm * np.sin(p)
    z = r * np.cos(th)

    return t, x, y, z


def lorentz_factor(v1, v2, v3):
    """
    Returns the Lorentz Factor, ``gamma``

    Parameters
    ----------
    v1 : float
        First component of 3-Velocity
    v2 : float
        Second component of 3-Velocity
    v3 : float
        Third component of 3-Velocity

    Returns
    -------
    gamma : float
        Lorentz Factor

    """
    v_vec = np.array([v1, v2, v3])
    v_norm2 = v_vec.dot(v_vec)
    gamma = 1 / np.sqrt(1 - v_norm2 / _c**2)

    return gamma


@jit
def v0(g_cov_mat, v1, v2, v3):
    """
    Utility function to return Timelike component (v0) of 4-Velocity
    Assumes a (+, -, -, -) Metric Signature

    Parameters
    ----------
    g_cov_mat : ~numpy.ndarray
        Matrix, containing Covariant Metric \
        Tensor values, in same coordinates as ``v_vec``
        Numpy array of shape (4,4)
    v1 : float
        First component of 3-Velocity
    v2 : float
        Second component of 3-Velocity
    v3 : float
        Third component of 3-Velocity
    Returns
    -------
    float
        Timelike component of 4-Velocity

    """
    g = g_cov_mat
    # Factor to add to coefficient, C
    fac = -1 * _c**2
    # Defining coefficients for quadratic equation
    A = g[0, 0]
    B = 2 * (g[0, 1] * v1 + g[0, 2] * v2 + g[0, 3] * v3)
    C = (
        (g[1, 1] * v1**2 + g[2, 2] * v2**2 + g[3, 3] * v3**2)
        + 2 * v1 * (g[1, 2] * v2 + g[1, 3] * v3)
        + 2 * v2 * g[2, 3] * v3
        + fac
    )
    D = (B**2) - (4 * A * C)

    v_t = (-B + np.sqrt(D)) / (2 * A)

    return v_t
