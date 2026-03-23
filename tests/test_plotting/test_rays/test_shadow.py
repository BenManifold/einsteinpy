from unittest import mock

import numpy as np
import pytest
from astropy import units as u
from matplotlib import pyplot as plt

from einsteinpy.plotting import ShadowPlotter
from einsteinpy.rays import Shadow


@pytest.fixture()
def dummy_data():
    mass = 1 * u.kg
    fov = 30 * u.km
    shadow = Shadow(mass=mass, fov=fov, n_rays=1000)
    return shadow


def test_plotter_has_correct_attributes(dummy_data):
    shadow = dummy_data
    cl = ShadowPlotter(shadow=shadow)
    assert isinstance(cl.shadow, Shadow)
    assert cl.is_intensity_plot


def test_plotter_contour_mode(dummy_data):
    """ShadowPlotter with is_line_plot=False uses contour/heatmap path."""
    shadow = dummy_data
    cl = ShadowPlotter(shadow=shadow, is_line_plot=False)
    assert not cl.is_intensity_plot
    cl.plot()
    assert hasattr(cl, "r1") and hasattr(cl, "theta1")
    assert hasattr(cl, "values1") and hasattr(cl, "values2")


@mock.patch("einsteinpy.plotting.rays.shadow.plt.plot")
def test_plot_calls_plt_plot(mock_patch, dummy_data):
    shadow = dummy_data
    cl = ShadowPlotter(shadow=shadow)
    cl.plot()
    expected = [
        mock.call(cl.shadow.fb1, cl.shadow.intensity, "r"),
        mock.call(cl.shadow.fb2, cl.shadow.intensity, "r"),
    ]
    assert mock_patch.call_args_list == expected


@mock.patch("einsteinpy.plotting.rays.shadow.plt.gca")
@mock.patch("einsteinpy.plotting.rays.shadow.plt.pcolormesh")
@mock.patch("einsteinpy.plotting.rays.shadow.plt.figure")
def test_show_contour_mode(mock_figure, mock_pcolormesh, mock_gca, dummy_data):
    """show() with is_line_plot=False uses pcolormesh path."""
    shadow = dummy_data
    cl = ShadowPlotter(shadow=shadow, is_line_plot=False)
    cl.plot()
    cl.show()
    mock_pcolormesh.assert_called_once()


@mock.patch("einsteinpy.plotting.rays.shadow.plt.show")
def test_show_line_plot_mode(mock_show, dummy_data):
    """show() with is_line_plot=True calls plt.show()."""
    shadow = dummy_data
    cl = ShadowPlotter(shadow=shadow)
    cl.plot()
    cl.show()
    mock_show.assert_called_once()


def test_shadow_smoothen():
    """Shadow.smoothen() interpolates intensity with cubic spline."""
    mass = 1 * u.kg
    fov = 30 * u.km
    shadow = Shadow(mass=mass, fov=fov, n_rays=15)
    orig_len = len(shadow.fb1)
    shadow.smoothen(points=50)
    assert len(shadow.fb1) == 50
    assert len(shadow.intensity) == 50
    assert np.allclose(shadow.fb2, -shadow.fb1)
