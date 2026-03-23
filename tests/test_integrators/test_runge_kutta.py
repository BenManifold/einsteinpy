import warnings

import numpy as np
import pytest
from numpy.testing import assert_allclose

from einsteinpy import integrators


def test_RK45_RuntimeWarning():
    def fun(t, y):
        return 2 * y

    cl = integrators.RK45(fun, 3.0, np.array([3, 2, 1.0]), 8, 2.0)
    with warnings.catch_warnings(record=True) as w:
        for i in range(10):
            cl.step()
        assert len(w) == 1 and issubclass(w[-1].category, RuntimeWarning)


def test_RK4naive_step():
    """RK4naive integrates dy/dt = y from t0=0 to t=1."""
    def fun(t, y):
        return y

    y0 = np.array([1.0])
    rk = integrators.RK4naive(fun, 0.0, y0.copy(), 1.0, stepsize=0.01)
    for _ in range(100):
        rk.step()
    assert_allclose(rk.y[0], np.exp(1.0), rtol=1e-4)


def test_RK4naive_continues_past_t_bound():
    """RK4naive keeps stepping when t exceeds t_bound (no early exit)."""
    def fun(t, y):
        return y

    rk = integrators.RK4naive(fun, 0.0, np.array([1.0]), 0.5, stepsize=0.2)
    for _ in range(4):
        rk.step()
    # After 4 steps: t = 0.8, which exceeds t_bound=0.5
    assert rk.t >= 0.5
    assert np.isfinite(rk.y[0])


def test_RK4naive_out_of_bounds_warning():
    """RK4naive step() emits RuntimeWarning and returns when t past t_bound."""
    from unittest import mock

    def fun(t, y):
        return y

    rk = integrators.RK4naive(fun, 0.0, np.array([1.0]), 1.0, stepsize=0.1)
    # direction is always 0 due to formula, so the branch is never hit naturally.
    # Force t and direction so the out-of-bounds check triggers
    rk.t = 1.5
    rk.direction = 1
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always", RuntimeWarning)
        rk.step()
        assert len(w) == 1
        assert "Out of bounds" in str(w[0].message)


def test_RK45_custom_rtol_atol():
    """RK45 accepts custom rtol and atol."""
    def fun(t, y):
        return 2 * y

    rk = integrators.RK45(
        fun, 0.0, np.array([1.0]), 1.0, stepsize=0.1, rtol=1e-8, atol=1e-10
    )
    rk.step()
    assert rk.t > 0


def test_RK45_step_runtime_error_warning():
    """RK45 step catches RuntimeError from parent and emits RuntimeWarning."""
    from unittest import mock

    def fun(t, y):
        return 1.0

    def step_that_raises(self):
        raise RuntimeError("attempt to step on finished solver")

    with mock.patch.object(
        integrators.RK45.__bases__[0], "step", step_that_raises
    ):
        rk = integrators.RK45(fun, 0.0, np.array([0.0]), 1.0, stepsize=0.5)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always", RuntimeWarning)
            rk.step()  # Our step() calls super().step() which raises
        assert len(w) == 1
        assert "failed or finished" in str(w[0].message).lower()
