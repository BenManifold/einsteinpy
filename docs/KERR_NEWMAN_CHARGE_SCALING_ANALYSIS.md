# Kerr-Newman Charge Scaling: Analysis of Issue #144

## Summary

The mysterious factor **11604461683.91822052001953125** that must be applied to charge values for correct Kerr-Newman trajectory calculations in einsteinpy appears to stem from a **unit conversion inconsistency** between:

1. **Geometrized / natural units** (used in the GR literature, with \(G = c = M = 1\) and often \(4\pi\epsilon_0 = 1\) or \(k_e = 1\))
2. **SI units** (used throughout einsteinpy for user-facing inputs: \(M\) in kg, \(Q\) in Coulombs, positions in meters, etc.)

The metric and electromagnetic tensor in einsteinpy are implemented correctly according to the standard Kerr-Newman formulae. The anomaly arises when combining these with the Lorentz force term and the specific affine parameterization used in the geodesic integration.

---

## Theoretical Background

### Kerr-Newman Metric and Charge Length Scale

From the [Kerr-Newman metric](https://en.wikipedia.org/wiki/Kerr%E2%80%93Newman_metric), the charge enters through the length scale

\[
r_Q^2 = \frac{Q^2 G}{4\pi\epsilon_0\,c^4}
\]

einsteinpy implements this correctly in `base_metric.py` and `kerrnewman.py`:

```python
r_Q2 = (Q**2) * _G * _Cc / _c**4   # base_metric.delta, kerrnewman.em_tensor_covariant
```

with `_Cc = 1/(4\pi\epsilon_0)` (Coulomb's constant \(k_e\)). This matches the standard formula.

### Equations of Motion for Charged Test Particles

The equation of motion is

\[
\frac{d^2 x^\mu}{d\lambda^2} + \Gamma^\mu_{\alpha\beta} \frac{dx^\alpha}{d\lambda}\frac{dx^\beta}{d\lambda} = \frac{q}{m} F^\mu{}_\nu u^\nu
\]

where \(q\) is the test particle charge, \(m\) its mass, and \(u^\nu = dx^\nu/d\lambda\) the 4-velocity. The Lorentz 4-force is

\[
f^\mu = \frac{q}{m} F^\mu{}_\nu u^\nu \quad \text{with} \quad F^\mu{}_\nu = g^{\mu\alpha} F_{\alpha\nu}
\]

In many GR references (e.g., Reissner–Nordström, Hackmann & Xu on Kerr-Newman orbits), quantities are given in **geometrized units** where \(G = M = c = K = 1\), with \(K = 1/(4\pi\epsilon_0)\). In these units, \(Q\) and \(q\) are dimensionless and the scaling factors are absorbed into the definitions.

---

## Unit Systems in einsteinpy

### 1. Metric Module (KerrNewman, BaseMetric)

- Uses **SI units** consistently.
- \(M\) in kg, \(Q\) in C, \(q\) in C/kg (charge-to-mass ratio).
- Positions in meters (\(x^0 = ct\), \(r\), \(\theta\), \(\phi\)).
- Metric components include explicit factors of \(c\), \(G\), and \(\epsilon_0\) (via `_Cc = 1/(4\pi\epsilon_0)`).
- The charge length scale is implemented as \(r_Q^2 = Q^2 G/(4\pi\epsilon_0 c^4)\).

### 2. Geodesic Module (GeodesicIntegrator, `_kerrnewman` in utils)

- `geodesic/utils.py` documents: **“Uses natural units, with \(c = G = M = k_e = 1\)”**.
- The low-level `_kerrnewman` metric uses \(\Delta = r^2 - 2r + a^2 + Q^2\) (i.e. \(r_s = 2\), \(r_Q^2 = Q^2\)).
- This implies **geometrized charge** for this code path.

### 3. Two Integration Paths

- **Path A (metric object + f_vec):** Uses `KerrNewman.f_vec`, which includes the Lorentz force and operates in SI. This is the path used when passing a metric instance (e.g. `Timelike(metric=mkn, coords=bl, ...)`).
- **Path B (string metric + GeodesicIntegrator):** Uses `_kerrnewman` from utils, which is in geometrized units and **does not include** the Lorentz force. Used when passing `metric="KerrNewman"` and `metric_params=(a, Q)`.

The problematic scaling appears when Path A is used with SI charges, and the Lorentz term or its normalization does not match the rest of the ODE system.

---

## Origin of the Factor 11604461683.918…

### Conversion Between Charge Units

In geometrized units (\(G = c = 1\), \(4\pi\epsilon_0 = 1\)):

\[
r_Q^2 = Q_{\text{geom}}^2
\]

In SI:

\[
r_Q^2 = \frac{Q_{\text{SI}}^2\,G}{4\pi\epsilon_0\,c^4}
\]

So the conversion is

\[
Q_{\text{geom}} = Q_{\text{SI}} \sqrt{\frac{G}{4\pi\epsilon_0\,c^4}} = Q_{\text{SI}} \sqrt{\frac{G\,k_e}{c^4}}
\]

with \(k_e = 1/(4\pi\epsilon_0) \approx 8.98755\times10^9\,\mathrm{N}\cdot\mathrm{m}^2/\mathrm{C}^2\). The numerical factor \(\sqrt{G\,k_e/c^4}\) is on the order of \(10^{-21}\), so \(Q_{\text{geom}}\) is tiny for typical SI charges. The inverse factor \(Q_{\text{SI}}/Q_{\text{geom}} \sim \sqrt{c^4/(G\,k_e)}\) is on the order of \(10^{21}\), i.e. far larger than \(1.16\times10^{10}\).

### Interpretation of the Empirical Factor

The empirical factor \(1.16\times10^{10}\) does **not** match a full SI ↔ geometrized charge conversion. It likely reflects one or more of:

1. **Incomplete or inconsistent SI normalization** in the Lorentz term compared with the metric and Christoffel terms.
2. **Mixing conventions:** Some parts of the metric/EM tensor normalized as in one convention (e.g. with \(c\) in the time coordinate) while the Lorentz term uses a different convention.
3. **Missing or redundant factors of \(c\) or \(4\pi\epsilon_0\)** in the Lorentz force implementation.

The test in `bin/geodesics_units/old_test_geodesics.py` sets:

```python
q = _G * M.value / _Cc * u.C / u.kg   # For GM = k_e Q q balance
Q = 11604461683.91822052001953125 * u.C
```

For equilibrium between gravity and Coulomb force:

\[
\frac{GM}{r^2} = \frac{k_e Q q}{r^2} \implies q/m = \frac{GM}{k_e Q}
\]

So \(Q = GM/(k_e (q/m))\). With the chosen \(q\), the “correct” \(Q\) for balance would be 1 C; the large factor on \(Q\) is used to compensate for the EM term in the code being effectively too weak by that factor.

---

## Potential Bugs Identified

### 1. Lorentz Force Index Structure (Most Likely Root Cause)

In `kerrnewman.py` line 408:

```python
vals[4:] -= self.q.value * (F_contra @ vec[4:] @ g_cov)
```

**Index analysis:**

- `F_contra` is \(F^{\mu\nu}\) (contravariant EM tensor).
- `vec[4:]` is \(u^\nu\) (contravariant 4-velocity).
- `F_contra @ vec[4:]` → \(F^{\mu\nu} u^\nu\) → contravariant Lorentz 4-force \(f^\mu\) ✓
- The extra `@ g_cov` contracts: \((F^{\mu\nu} u^\nu)\, g_{\mu\sigma}\) → a **covariant** vector \(f_\sigma\).

The geodesic equation gives **contravariant** acceleration:

\[
a^\mu = -\Gamma^\mu_{\alpha\beta} u^\alpha u^\beta + \frac{q}{m} F^\mu{}_\nu u^\nu
\]

So `vals[4:]` stores \(a^\mu\) (contravariant). The code subtracts a **covariant** vector from it, which is a tensor type error.

**Proposed fix:** Use only the contravariant Lorentz term:

```python
vals[4:] += self.q.value * (F_contra @ vec[4:])  # Note: + not -, and no @ g_cov
```

(Whether it is `+` or `−` depends on the sign convention of \(F_{\mu\nu}\); the Lorentz force should oppose or add to the geodesic acceleration consistently with the chosen convention.)

The erroneous `@ g_cov` effectively multiplies the Lorentz term by metric components. For a metric with \(|g_{00}| \sim c^2\) and \(g_{ii} \sim 1\), this can introduce factors of order \(c^2 \approx 9\times10^{16}\) or \(1/c^2 \approx 10^{-17}\). Combined with other conventions (e.g., \(x^0 = ct\)), such mismatches can produce correction factors of the order of the observed \(10^{10}\).

### 2. 4-Velocity and Affine Parameter

The position 4-vector uses \(x^0 = c\,t\) (length). The affine parameter \(\lambda\) and the 4-velocity \(u^\mu = dx^\mu/d\lambda\) must have compatible units. If \(\lambda\) or the ODE normalization implicitly assumes a certain unit convention, mismatches can introduce scaling factors similar to powers of \(c\).

### 3. CHANGELOG Note

CHANGELOG records: *“Fixed #508, Removed a stray scaling factor in metric.KerrNewman class”*. That fix may have altered the normalization without fully resolving the SI ↔ geometrized charge consistency for the Lorentz term.

---

## Recommendations

1. **Clarify unit conventions**  
   Document explicitly for all Kerr-Newman–related code:
   - Affine parameter units.
   - 4-position and 4-velocity units.
   - Whether \(Q\) and \(q\) are intended as SI or geometrized.

2. **Check the Lorentz force contraction**  
   - Ensure \(a^\mu = (q/m) F^\mu{}_\nu u^\nu\) is used in contravariant form.
   - Remove the extra `@ g_cov` if it is incorrect.

3. **Validate against a known solution**  
   - Compare with analytical or published results (e.g., Hackmann & Xu) in a well-defined unit system, then add explicit conversions for SI.
   - **Implemented**: `test_conserved_energy_along_integrated_trajectory` integrates a charged trajectory and verifies that the conserved energy \(E = -\pi_t = -c(g_{00} u^0 + g_{03} u^3 + q A_0)\) stays constant (from Hackmann & Xu). `test_f_vec_matches_kerr_when_Q_and_q_zero` verifies that KerrNewman reduces to Kerr when \(Q = q = 0\).

4. **Add unit tests**  
   - Include a test with gravitational and electromagnetic forces balanced in a clearly specified unit system, so future changes do not reintroduce scaling errors.

---

## References

- [Kerr–Newman metric (Wikipedia)](https://en.wikipedia.org/wiki/Kerr%E2%80%93Newman_metric)
- [Reissner–Nordström metric (Wikipedia)](https://en.wikipedia.org/wiki/Reissner%E2%80%93Nordstr%C3%B6m_metric)
- Hackmann, E. & Xu, H., “Charged particle motion in Kerr–Newman space–times,” Phys. Rev. D **87**, 124030 (2013), [arXiv:1304.2142](https://arxiv.org/abs/1304.2142)
- Lynden-Bell, D., “Electromagnetic magic: The relativistically rotating disk,” Phys. Rev. D **70**, 105017 (2004)
- [EinsteinPy Kerr-Newman implementation](https://github.com/einsteinpy/einsteinpy)
- [GitHub Issue #144](https://github.com/einsteinpy/einsteinpy/issues/144)
