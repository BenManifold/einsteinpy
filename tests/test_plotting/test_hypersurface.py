from unittest import mock

import numpy as np
import pytest
from astropy import units as u

from einsteinpy.hypersurface import SchwarzschildEmbedding
from einsteinpy.plotting import HypersurfacePlotter


@pytest.fixture()
def dummy_data():
    surface_obj = SchwarzschildEmbedding(5.927e23 * u.kg)
    return surface_obj


@mock.patch("einsteinpy.plotting.hypersurface.core.plt.show")
def test_plot_calls_plt_show(mock_show, dummy_data):
    surface = dummy_data
    cl = HypersurfacePlotter(surface)
    cl.plot()
    cl.show()
    mock_show.assert_called_with()
    assert cl.alpha == 100
    assert cl.plot_type == "wireframe"


@mock.patch("einsteinpy.plotting.hypersurface.core.plt.show")
def test_plot_surface_type(mock_show, dummy_data):
    """HypersurfacePlotter with plot_type='surface' uses plot_surface path."""
    surface = dummy_data
    cl = HypersurfacePlotter(surface, plot_type="surface")
    X = np.array([[0, 1], [0, 1]])
    Y = np.array([[0, 0], [1, 1]])
    Z = np.array([[0, 0.5], [0.5, 1]])
    with mock.patch.object(
        cl.embedding, "get_values_surface", return_value=(X, Y, Z)
    ):
        with mock.patch(
            "einsteinpy.plotting.hypersurface.core.plt.axes"
        ) as mock_axes:
            mock_ax = mock.MagicMock()
            mock_axes.return_value = mock_ax
            cl.plot()
    mock_ax.plot_surface.assert_called_once()
    assert cl.plot_type == "surface"


@mock.patch("einsteinpy.plotting.hypersurface.core.HypersurfacePlotter.show")
@mock.patch("einsteinpy.plotting.hypersurface.core.HypersurfacePlotter.plot")
def test_plot_works_with_different_plot_type(mock_plot, mock_show, dummy_data):
    surface = dummy_data
    cl = HypersurfacePlotter(surface, plot_type="surface")
    cl.plot()
    cl.show()
    assert cl.plot_type == "surface"
