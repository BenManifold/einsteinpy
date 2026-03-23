"""Tests for einsteinpy.plotting.geodesic.core."""
from unittest import mock

import pytest

from einsteinpy.plotting.geodesic import InteractiveGeodesicPlotter, StaticGeodesicPlotter
from einsteinpy.plotting.geodesic.core import GeodesicPlotter, in_ipynb


def test_geodesic_plotter_is_static_or_interactive():
    """GeodesicPlotter is a subclass of Static or Interactive plotter."""
    assert issubclass(GeodesicPlotter, StaticGeodesicPlotter) or issubclass(
        GeodesicPlotter, InteractiveGeodesicPlotter
    )


def test_in_ipynb_name_error_returns_static():
    """in_ipynb returns StaticGeodesicPlotter when get_ipython not available."""
    result = in_ipynb()
    assert result is StaticGeodesicPlotter


def test_in_ipynb_zmq_returns_interactive():
    """in_ipynb returns InteractiveGeodesicPlotter for ZMQ (Jupyter) shell."""
    import einsteinpy.plotting.geodesic.core as core_mod

    mock_shell = type("ZMQInteractiveShell", (), {})()
    with mock.patch.object(core_mod, "get_ipython", lambda: mock_shell, create=True):
        result = core_mod.in_ipynb()
    assert result is InteractiveGeodesicPlotter


def test_in_ipynb_terminal_returns_static():
    """in_ipynb returns StaticGeodesicPlotter for TerminalInteractiveShell."""
    import einsteinpy.plotting.geodesic.core as core_mod

    mock_shell = type("TerminalInteractiveShell", (), {})()
    with mock.patch.object(core_mod, "get_ipython", lambda: mock_shell, create=True):
        result = core_mod.in_ipynb()
    assert result is StaticGeodesicPlotter


def test_in_ipynb_other_shell_returns_static():
    """in_ipynb returns StaticGeodesicPlotter for unknown shell."""
    import einsteinpy.plotting.geodesic.core as core_mod

    mock_shell = type("OtherShell", (), {})()
    with mock.patch.object(core_mod, "get_ipython", lambda: mock_shell, create=True):
        result = core_mod.in_ipynb()
    assert result is StaticGeodesicPlotter
