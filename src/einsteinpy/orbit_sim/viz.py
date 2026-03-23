"""
PyQtGraph 3D visualization for orbit simulation.
"""

import numpy as np
from PySide6 import QtCore, QtWidgets
import pyqtgraph as pg
import pyqtgraph.opengl as gl


def make_horizon_mesh(r_s_geod, rows=20, cols=24):
    """Create GLMeshItem for event horizon sphere (radius r_s in geometrized units)."""
    mesh = gl.MeshData.sphere(rows=rows, cols=cols, radius=r_s_geod, offset=False)
    item = gl.GLMeshItem(
        meshdata=mesh,
        color=(0.1, 0.1, 0.1, 0.85),
        smooth=True,
    )
    return item


def create_orbit_view(data):
    """
    Create GLViewWidget with orbit path and current position.

    Parameters
    ----------
    data : dict
        From simulation.compute_trajectory: x, y, z, r_s_geod.

    Returns
    -------
    view : gl.GLViewWidget
    orbit_line : gl.GLLinePlotItem
    particle : gl.GLScatterPlotItem
    """
    view = gl.GLViewWidget()
    view.setBackgroundColor(72, 75, 80, 255)  # Darker grey for orbiter visibility
    view.opts["distance"] = 80  # Zoomed in
    view.opts["elevation"] = 20
    view.opts["azimuth"] = 45

    x, y, z = data["x"], data["y"], data["z"]
    r_s = data["r_s_geod"]

    # Orbit path
    pos = np.column_stack([x, y, z])
    orbit_line = gl.GLLinePlotItem(
        pos=pos,
        color=(0.2, 0.6, 1.0, 0.9),
        width=2,
        antialias=True,
        mode="line_strip",
    )
    view.addItem(orbit_line)

    # Current particle position (orbiting twin - bright to stand out)
    particle = gl.GLScatterPlotItem(
        pos=np.array([[x[0], y[0], z[0]]]),
        color=(1.0, 0.95, 0.3, 1.0),
        size=1.6,
        pxMode=False,
    )
    view.addItem(particle)

    # Event horizon sphere
    horizon = make_horizon_mesh(r_s)
    view.addItem(horizon)

    return view, orbit_line, particle


def create_clock_panel():
    """Create a panel showing twin clocks and time dilation."""
    widget = QtWidgets.QGroupBox("Twin Paradox — Time Dilation")
    layout = QtWidgets.QFormLayout(widget)

    t_label = QtWidgets.QLabel("0.0")
    t_label.setObjectName("t_label")
    layout.addRow("Coordinate time t:", t_label)

    tau_far_label = QtWidgets.QLabel("0.0")
    tau_far_label.setObjectName("tau_far_label")
    layout.addRow("Far twin τ (≈ t):", tau_far_label)

    tau_orb_label = QtWidgets.QLabel("0.0")
    tau_orb_label.setObjectName("tau_orb_label")
    layout.addRow("Orbiting twin τ:", tau_orb_label)

    ratio_label = QtWidgets.QLabel("1.0")
    ratio_label.setObjectName("ratio_label")
    layout.addRow("dτ/dt (orbiter):", ratio_label)

    status_label = QtWidgets.QLabel("")
    status_label.setObjectName("status_label")
    status_label.setStyleSheet("color: #c44; font-weight: bold;")
    status_label.setWordWrap(True)
    status_label.setAlignment(QtCore.Qt.AlignLeft | QtCore.Qt.AlignTop)
    layout.addRow("", status_label)

    return widget


def create_control_panel(
    on_r_change,
    on_L_change,
    on_spin_change,
    on_play_pause,
    on_speed_change,
    on_reset=None,
    on_circular=None,
    on_kerrschild_change=None,
):
    """Create sliders and buttons for r_init, L, spin, play/pause, speed."""
    widget = QtWidgets.QGroupBox("Controls")
    layout = QtWidgets.QVBoxLayout(widget)

    # r_init
    r_layout = QtWidgets.QHBoxLayout()
    r_label = QtWidgets.QLabel("Initial radius r:")
    r_spin = QtWidgets.QDoubleSpinBox()
    r_spin.setRange(6, 80)
    r_spin.setValue(4)
    r_spin.setDecimals(1)
    r_spin.setButtonSymbols(QtWidgets.QAbstractSpinBox.NoButtons)
    r_spin.valueChanged.connect(on_r_change)
    r_layout.addWidget(r_label)
    r_layout.addWidget(r_spin)
    layout.addLayout(r_layout)

    # L (1.0 - 8.0 for circular orbits at various r)
    L_layout = QtWidgets.QHBoxLayout()
    L_label = QtWidgets.QLabel("Angular momentum L:")
    L_spin = QtWidgets.QDoubleSpinBox()
    L_spin.setRange(2, 8)
    L_spin.setValue(2.4)  # ≈ kerr_circular_L(4, 0.9)
    L_spin.setDecimals(2)
    L_spin.setButtonSymbols(QtWidgets.QAbstractSpinBox.NoButtons)
    L_spin.valueChanged.connect(on_L_change)
    L_layout.addWidget(L_label)
    L_layout.addWidget(L_spin)
    layout.addLayout(L_layout)

    # Kerr-Schild (near-horizon)
    ks_check = QtWidgets.QCheckBox("Use Kerr-Schild (near-horizon)")
    ks_check.setToolTip(
        "Enable horizon-penetrating coordinates; allows orbits closer to the horizon"
    )
    ks_check.setChecked(False)
    if on_kerrschild_change is not None:
        ks_check.stateChanged.connect(on_kerrschild_change)
    layout.addWidget(ks_check)

    # Spin (Kerr)
    spin_layout = QtWidgets.QHBoxLayout()
    spin_label = QtWidgets.QLabel("Black hole spin a:")
    spin_spin = QtWidgets.QDoubleSpinBox()
    spin_spin.setRange(0, 0.99)
    spin_spin.setValue(0.9)
    spin_spin.setDecimals(2)
    spin_spin.setButtonSymbols(QtWidgets.QAbstractSpinBox.NoButtons)
    spin_spin.valueChanged.connect(on_spin_change)
    spin_layout.addWidget(spin_label)
    spin_layout.addWidget(spin_spin)
    layout.addLayout(spin_layout)

    # Play / Pause
    play_btn = QtWidgets.QPushButton("Play")
    play_btn.setCheckable(True)
    play_btn.toggled.connect(on_play_pause)
    layout.addWidget(play_btn)

    # Reset
    reset_btn = QtWidgets.QPushButton("Reset")
    if on_reset is not None:
        reset_btn.clicked.connect(on_reset)
    layout.addWidget(reset_btn)

    # Circular orbit (stable at current r, Schwarzschild)
    circular_btn = QtWidgets.QPushButton("Circular orbit")
    if on_circular is not None:
        circular_btn.clicked.connect(on_circular)
    layout.addWidget(circular_btn)

    # Speed
    speed_layout = QtWidgets.QHBoxLayout()
    speed_label = QtWidgets.QLabel("Speed:")
    speed_spin = QtWidgets.QSpinBox()
    speed_spin.setMinimum(1)
    speed_spin.setMaximum(100)
    speed_spin.setValue(1)
    speed_spin.valueChanged.connect(on_speed_change)
    speed_layout.addWidget(speed_label)
    speed_layout.addWidget(speed_spin)
    layout.addLayout(speed_layout)

    return widget, {
        "r_spin": r_spin,
        "L_spin": L_spin,
        "spin_spin": spin_spin,
        "ks_check": ks_check,
        "play_btn": play_btn,
        "reset_btn": reset_btn,
        "circular_btn": circular_btn,
        "speed_spin": speed_spin,
    }
