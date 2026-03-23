Near-Horizon Coordinates and Handling Divergent Metric Regions
==============================================================

This document summarizes best practices in numerical relativity and relativistic physics simulation for handling coordinate singularities (e.g., denominators that vanish at horizons) and explores implementation options for einsteinpy.

The Problem
-----------

In Boyer-Lindquist coordinates, the Kerr metric has components involving :math:`\Delta = r^2 - 2r + a^2`. At the event horizon, :math:`\Delta \to 0`, so terms :math:`\propto 1/\Delta` diverge. This causes overflow when integrating geodesics or evaluating the metric near the horizon. The singularity is **coordinate-only**—spacetime is smooth there; BL coordinates are simply ill-suited.

Established Practices
---------------------

1. **Horizon-Penetrating Coordinates** (Preferred)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Use coordinate systems where the metric components remain regular at and across the horizon:

+---------------------------+------------------------------------------------+------------------------------------------+
| Coordinate System         | Horizon Behavior                              | Use Case                                 |
+===========================+================================================+==========================================+
| **Kerr-Schild**           | Metric is g = η + H k⊗k; regular everywhere   | Geodesics, accretion, excision           |
+---------------------------+------------------------------------------------+------------------------------------------+
| **Ingoing Kerr-Schild**   | Null coordinates; regular at horizon          | Perturbations, wave propagation          |
+---------------------------+------------------------------------------------+------------------------------------------+
| **Harmonic coordinates**  | Constructed from Kerr-Schild; regular         | Numerical evolution                      |
+---------------------------+------------------------------------------------+------------------------------------------+
| **Kruskal–Szekeres**      | Covers full manifold (Schwarzschild)          | Pedagogy, maximal extension              |
+---------------------------+------------------------------------------------+------------------------------------------+

**Kerr-Schild form** for Kerr:

.. math::

   g_{ab} = \eta_{ab} + 2H \ell_a \ell_b

where :math:`H` is a scalar and :math:`\ell` is null. The metric is linear in the "perturbation" and has no :math:`1/\Delta` terms. Geodesic equations simplify and remain well-behaved at the horizon.

**Implementation path**: Add a Kerr-Schild metric (or :py:class:`~einsteinpy.metric.Kerr` with a ``coords=`` option) that implements the Kerr-Schild line element. The geodesic integrator accepts any callable metric, so a new ``_kerr_ks`` function in :py:mod:`einsteinpy.geodesic.utils` could be registered alongside ``_kerr``.

2. **Coordinate Switching / Patched Coordinates**
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Run integration in BL when far from the horizon; switch to Kerr-Schild (or another regular chart) when :math:`r` drops below a threshold (e.g., :math:`r < 2 \times r_{\mathrm{horizon}}`). Transform state (position, 4-momentum) between charts at the patch boundary.

* **Pros**: Reuses existing BL implementation where it works.
* **Cons**: Careful bookkeeping; possible discontinuities at the patch boundary if the transformation is not smooth.

3. **Regularization via Gauge / Formulation**
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

In full numerical relativity (BSSN, etc.):

* Use **covariant formulations** and **cell-centered grids** so that singular tensor components are handled in a regular basis.
* **Partially implicit Runge-Kutta (PIRK)** time integration can avoid explicit regularization at coordinate singularities.
* **Analytically regularized tensor components** (SENR/NRPy+ style): change to a basis where components are regular by construction.

For geodesics: these techniques are more relevant to evolution codes than to particle motion, but the "change of basis" idea translates to choosing coordinates where metric components are regular.

4. **Clipping / Guarding** (Pragmatic, Not Ideal)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Clamp :math:`r` to stay above :math:`r_{\mathrm{horizon}} + \varepsilon`, or replace :math:`1/\Delta` with :math:`1/\max(\Delta, \varepsilon)`. This prevents overflow but is **ad hoc** and can produce unphysical behavior. Use only as a temporary workaround, not as a substitute for proper coordinates.

5. **Extended Precision**
~~~~~~~~~~~~~~~~~~~~~~~~~

Use ``float128`` or arbitrary-precision arithmetic to push the overflow threshold. This delays the problem but does not remove the underlying coordinate singularity. Not standard in production numerical relativity.

Implementation in einsteinpy
---------------------------

EinsteinPy provides Kerr-Schild coordinates for the Kerr metric:

* **Geodesic API**: Use :py:class:`~einsteinpy.geodesic.Timelike` or :py:class:`~einsteinpy.geodesic.Nulllike` with ``coords="KerrSchild"`` when integrating near the horizon. Position and momentum are still specified in Boyer-Lindquist spherical (r, θ, φ) and (p_r, p_θ, p_φ); conversion is automatic.

  .. code-block:: python

      from einsteinpy.geodesic import Timelike
      import numpy as np

      a = 0.9
      r, theta, phi = 2.5, np.pi / 2, 0.0   # Close to horizon
      L = 2.4   # Angular momentum for circular orbit

      geod = Timelike(
          metric="Kerr",
          metric_params=(a,),
          position=[r, theta, phi],
          momentum=[0.0, 0.0, L],
          coords="KerrSchild",
          steps=2000,
          delta=0.2,
          suppress_warnings=True,
      )

* **Orbit sim**: Enable the "Use Kerr-Schild (near-horizon)" checkbox to allow orbits closer to the event horizon. Alternatively, pass ``use_kerrschild=True`` to :py:func:`~einsteinpy.orbit_sim.simulation.compute_trajectory`.

* **Coordinate conversion**: ``bl_to_kerrschild_cartesian`` and ``kerrschild_to_bl_cartesian`` in :py:mod:`einsteinpy.coordinates.utils` transform between Boyer-Lindquist and Kerr-Schild Cartesian.

Recommendations
---------------

1. **Document coordinate limitations** — For each metric, document in which coordinates it is defined and where it becomes singular. Provide warnings when users request evaluation in singular regions.

2. **Conversion utilities** — The :py:mod:`einsteinpy.coordinates.utils` module provides BL ↔ Kerr-Schild transformations for interpretation and visualization.

References
----------

* Bardeen et al. (1972), "The Four Laws of Black Hole Mechanics"
* Pretorius (2005), "Evolution of binary black-hole spacetimes"
* Ruchlin et al. (2018), "SENR/NRPy+: Numerical Relativity in Singular Curvilinear Coordinate Systems"
* gr-qc/0010034: "Perturbations of the Kerr spacetime in horizon penetrating coordinates"
