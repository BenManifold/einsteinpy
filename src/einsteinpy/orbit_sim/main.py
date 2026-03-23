"""
Main Qt application for orbit simulation.
"""

import json
import logging
import queue
import threading
import time

import numpy as np

log = logging.getLogger("einsteinpy.orbit_sim")

def _dbg(evt, hyp, data=None):
    try:
        with open("/Users/benjaminmanifold/einsteinpy/.cursor/debug-08691c.log", "a") as f:
            f.write(json.dumps({"sessionId": "08691c", "hypothesisId": hyp, "location": "main.py", "message": evt, "data": data or {}, "timestamp": int(time.time() * 1000)}) + "\n")
    except Exception:
        pass

import pyqtgraph as pg
from PySide6 import QtCore, QtWidgets

from . import simulation
from . import viz  # imports pyqtgraph.opengl - can block on OpenGL init


def _main_module_loaded():
    try:
        with open("/Users/benjaminmanifold/einsteinpy/.cursor/debug-08691c.log", "a") as f:
            import json as _j, time as _t
            f.write(_j.dumps({"sessionId": "08691c", "hypothesisId": "A", "location": "main.py", "message": "main module loaded (viz import done)", "data": {}, "timestamp": int(_t.time() * 1000)}) + "\n")
    except Exception:
        pass


_main_module_loaded()


class ComputeWorker(QtCore.QObject):
    """Holds finished signal. Actual compute runs in a Python thread."""
    finished = QtCore.Signal(dict)

    def __init__(self, work_queue, stop_event):
        super().__init__()
        self._work_queue = work_queue
        self._stop_event = stop_event

    def run(self):
        """Thread entry: wait for params on queue, compute, emit finished."""
        _dbg("ComputeWorker.run: thread started", "C")
        while not self._stop_event.is_set():
            try:
                params = self._work_queue.get(timeout=0.5)
            except queue.Empty:
                continue
            if params is None:
                break
            _dbg("ComputeWorker.run: got params", "C", {"keys": list(params.keys())})
            try:
                data = simulation.compute_trajectory(**params)
                n = len(data.get("x", []))
                log.debug("ComputeWorker: trajectory ready, %d points", n)
                self.finished.emit(data)
            except Exception as e:
                log.exception("ComputeWorker: trajectory computation failed")
                self.finished.emit({"error": str(e)})


class OrbitSimWindow(QtWidgets.QMainWindow):
    def __init__(self):
        _dbg("OrbitSimWindow.__init__ start", "B")
        super().__init__()
        self.setWindowTitle("Black Hole Orbit Simulation")
        self.resize(1100, 700)

        self._data = None
        self._idx = 0
        self._playing = False
        self._speed = 1
        # Interstellar-style: Kerr black hole, stable circular orbit.
        # r=4 gives ~2× dilation; closer overflows (BL coords singular at horizon).
        self._r_init = 4.0
        self._spin = 0.9
        self._L = float(simulation.kerr_circular_L(4.0, 0.9))
        self._computing = False
        self._compute_request_id = 0
        self._use_kerrschild = False

        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        layout = QtWidgets.QHBoxLayout(central)

        # 3D view with centered computing overlay
        self._view_stack = QtWidgets.QStackedWidget()
        self._view_page = QtWidgets.QWidget()
        self._view_layout = QtWidgets.QVBoxLayout(self._view_page)
        self._view_layout.setContentsMargins(0, 0, 0, 0)
        self._view_stack.addWidget(self._view_page)

        self._computing_page = QtWidgets.QWidget()
        computing_layout = QtWidgets.QVBoxLayout(self._computing_page)
        computing_layout.setAlignment(QtCore.Qt.AlignCenter)
        self._computing_label = QtWidgets.QLabel("Computing...")
        self._computing_label.setStyleSheet("font-size: 18px; color: #888;")
        computing_layout.addWidget(self._computing_label, 0, QtCore.Qt.AlignCenter)
        self._view_stack.addWidget(self._computing_page)

        self._view_stack.setCurrentWidget(self._computing_page)
        layout.addWidget(self._view_stack, stretch=1)

        # Right panel (create before _recompute so _update_display can use it)
        right = QtWidgets.QWidget()
        right_layout = QtWidgets.QVBoxLayout(right)
        right_layout.setContentsMargins(8, 8, 8, 8)

        self._clock_panel = viz.create_clock_panel()
        _dbg("after create_clock_panel", "E")
        right_layout.addWidget(self._clock_panel)

        ctrl, ctrl_widgets = viz.create_control_panel(
            on_r_change=self._on_r_change,
            on_L_change=self._on_L_change,
            on_spin_change=self._on_spin_change,
            on_play_pause=self._on_play_pause,
            on_speed_change=self._on_speed_change,
            on_reset=self._on_reset,
            on_circular=self._on_circular,
            on_kerrschild_change=self._on_kerrschild_change,
        )
        self._ctrl_widgets = ctrl_widgets
        right_layout.addWidget(ctrl)
        _dbg("after create_control_panel", "E")

        right_layout.addStretch()
        right.setMaximumWidth(220)
        layout.addWidget(right)

        self._view = None
        self._orbit_line = None
        self._particle = None

        # Background computation using Python threading (avoids PySide6
        # QThread/moveToThread signal delivery issues on macOS).
        self._compute_queue = queue.Queue()
        self._compute_stop = threading.Event()
        self._compute_worker = ComputeWorker(
            self._compute_queue, self._compute_stop
        )
        self._compute_worker.finished.connect(self._on_trajectory_ready)
        self._compute_thread = threading.Thread(
            target=self._compute_worker.run, daemon=True
        )
        self._compute_thread.start()
        _dbg("after compute_thread.start", "C")

        # Timer for animation
        self._timer = QtCore.QTimer(self)
        self._timer.timeout.connect(self._tick)
        self._tick_interval_ms = 80  # Slower, more cinematic

        # Debounce timer: recompute only after user pauses sliders
        self._recompute_timer = QtCore.QTimer(self)
        self._recompute_timer.setSingleShot(True)
        self._recompute_timer.timeout.connect(self._recompute_and_display)

        # Initial compute: run synchronously in main thread (bypasses PySide6
        # threading issues). UI will show overlay and block for a few seconds.
        QtCore.QTimer.singleShot(50, self._run_initial_compute_sync)
        _dbg("scheduled _run_initial_compute_sync via QTimer.singleShot(50)", "C")

    def closeEvent(self, event):
        """Stop the worker thread on close."""
        self._compute_stop.set()
        try:
            self._compute_queue.put(None, timeout=0.5)  # poison pill
        except queue.Full:
            pass
        self._compute_thread.join(timeout=2.0)
        super().closeEvent(event)

    def _run_initial_compute_sync(self):
        """Run first trajectory compute in main thread (workaround for threading)."""
        _dbg("_run_initial_compute_sync called", "C")
        metric = "Kerr" if abs(self._spin) > 1e-6 else "Schwarzschild"
        ok, _ = simulation.validate_params(
            self._r_init, self._L, self._spin, metric,
            use_kerrschild=self._use_kerrschild,
        )
        if not ok:
            return
        self._computing = True
        self._set_controls_enabled(False)
        self._computing_label.setText("Computing...")
        self._view_stack.setCurrentWidget(self._computing_page)
        QtWidgets.QApplication.processEvents()  # Paint overlay before blocking
        try:
            data = simulation.compute_trajectory(
                r_init=self._r_init,
                angular_momentum=self._L,
                metric=metric,
                spin=self._spin,
                steps=6000,
                delta=0.3,
                mass=simulation._MASS_1YEAR,
                use_kerrschild=self._use_kerrschild,
            )
            self._on_trajectory_ready(data)
        except Exception as e:
            log.exception("Initial compute failed")
            self._on_trajectory_ready({"error": str(e)})
        # Set up for future recomputes via thread
        self._computing = False

    def _recompute_and_display(self):
        """Start trajectory computation in background thread."""
        _dbg("_recompute_and_display called (QTimer fired)", "C")
        metric = "Kerr" if abs(self._spin) > 1e-6 else "Schwarzschild"
        log.debug(
            "_recompute: r_init=%.2f L=%.2f spin=%.2f metric=%s",
            self._r_init, self._L, self._spin, metric,
        )
        ok, msg = simulation.validate_params(
            self._r_init, self._L, self._spin, metric,
            use_kerrschild=self._use_kerrschild,
        )
        if not ok:
            log.warning("_recompute: validation failed: %s", msg)
            status_label = self._clock_panel.findChild(
                QtWidgets.QLabel, "status_label"
            )
            if status_label:
                status_label.setText(msg)
            self._set_controls_enabled(True)
            return

        self._computing = True
        self._compute_request_id += 1
        request_id = self._compute_request_id
        self._set_controls_enabled(False)
        self._computing_label.setText("Computing...")
        self._view_stack.setCurrentWidget(self._computing_page)

        params = {
            "r_init": self._r_init,
            "angular_momentum": self._L,
            "metric": metric,
            "spin": self._spin,
            "steps": 6000,
            "delta": 0.3,
            "mass": simulation._MASS_1YEAR,
            "use_kerrschild": self._use_kerrschild,
        }
        _dbg("putting params on queue", "C", {"queue_size_before": self._compute_queue.qsize()})
        self._compute_queue.put(params)
        _dbg("params put on queue", "C", {"queue_size_after": self._compute_queue.qsize()})

    def _on_trajectory_ready(self, data):
        """Handle trajectory result from worker (runs on main thread)."""
        self._computing = False
        self._set_controls_enabled(True)

        if "error" in data:
            log.error("_on_trajectory_ready: %s", data["error"])
            err_msg = f"Error: {data['error']}"
            status_label = self._clock_panel.findChild(
                QtWidgets.QLabel, "status_label"
            )
            if status_label:
                status_label.setText(err_msg)
            # Show error on computing overlay when no prior view (first-run failure)
            self._computing_label.setText(err_msg)
            self._computing_label.setWordWrap(True)
            if self._view is not None:
                self._view_stack.setCurrentWidget(self._view_page)
            return

        n = len(data.get("x", []))
        log.debug("_on_trajectory_ready: success, %d points, hit_horizon=%s", n, data.get("hit_horizon"))
        self._data = data
        self._idx = 0

        # Replace view widget
        if self._view is not None:
            self._view_layout.removeWidget(self._view)
            self._view.deleteLater()

        self._view, self._orbit_line, self._particle = viz.create_orbit_view(
            self._data
        )
        self._view_layout.addWidget(self._view)
        self._view_stack.setCurrentWidget(self._view_page)
        status_label = self._clock_panel.findChild(QtWidgets.QLabel, "status_label")
        if status_label:
            status_label.setText(
                "Captured by horizon!" if self._data.get("hit_horizon") else ""
            )
        self._update_display()

    def _set_controls_enabled(self, enabled):
        """Enable or disable control widgets."""
        self._ctrl_widgets["r_spin"].setEnabled(enabled)
        self._ctrl_widgets["L_spin"].setEnabled(enabled)
        self._ctrl_widgets["spin_spin"].setEnabled(enabled)
        self._ctrl_widgets["play_btn"].setEnabled(enabled)
        self._ctrl_widgets["reset_btn"].setEnabled(enabled)
        self._ctrl_widgets["circular_btn"].setEnabled(enabled)
        self._ctrl_widgets["speed_spin"].setEnabled(enabled)

    def _update_display(self):
        """Update particle position and clock labels from current index."""
        if self._data is None:
            return
        idx = min(self._idx, len(self._data["x"]) - 1)
        x, y, z = self._data["x"][idx], self._data["y"][idx], self._data["z"][idx]
        self._particle.setData(pos=np.array([[x, y, z]]))

        t = self._data["t"][idx]
        tau = self._data["tau"][idx]
        ratio = self._data["dtau_dt"][idx]
        if not np.isfinite(ratio):
            ratio = 0.0

        t_label = self._clock_panel.findChild(QtWidgets.QLabel, "t_label")
        tau_far = self._clock_panel.findChild(QtWidgets.QLabel, "tau_far_label")
        tau_orb = self._clock_panel.findChild(QtWidgets.QLabel, "tau_orb_label")
        ratio_label = self._clock_panel.findChild(QtWidgets.QLabel, "ratio_label")
        # Display: years for far twin; months for orbiter when small (Interstellar feel)
        if t_label:
            t_label.setText(f"{t:.0f} yr")
        if tau_far:
            tau_far.setText(f"{t:.0f} yr")  # Far twin: τ ≈ t
        if tau_orb:
            if tau < 1.0:
                months = tau * 12
                tau_orb.setText(f"{months:.0f} mo")
            else:
                tau_orb.setText(f"{tau:.1f} yr")
        if ratio_label:
            ratio_label.setText(f"{ratio:.4f}")

    def _tick(self):
        """Advance simulation by _speed steps."""
        if self._data is None:
            return
        self._idx = min(
            self._idx + self._speed,
            len(self._data["t"]) - 1,
        )
        self._update_display()
        if self._idx >= len(self._data["t"]) - 1:
            self._playing = False
            self._ctrl_widgets["play_btn"].setChecked(False)
            self._timer.stop()

    def _on_r_change(self, value):
        self._r_init = float(value)
        self._recompute_timer.stop()
        self._recompute_timer.start(250)

    def _on_L_change(self, value):
        self._L = float(value)
        self._recompute_timer.stop()
        self._recompute_timer.start(250)

    def _on_spin_change(self, value):
        self._spin = float(value)
        self._recompute_timer.stop()
        self._recompute_timer.start(250)

    def _on_kerrschild_change(self, state):
        self._use_kerrschild = state == QtCore.Qt.Checked
        r_spin = self._ctrl_widgets["r_spin"]
        if self._use_kerrschild and abs(self._spin) > 1e-6:
            r_min = simulation._kerr_horizon_radius(self._spin) + 0.1
            r_spin.setRange(max(2.0, r_min), 80)
        else:
            r_spin.setRange(6, 80)
        self._recompute_timer.stop()
        self._recompute_and_display()

    def _on_reset(self):
        """Restart animation from the beginning."""
        self._playing = False
        self._timer.stop()
        self._ctrl_widgets["play_btn"].setChecked(False)
        self._ctrl_widgets["play_btn"].setText("Play")
        self._idx = 0
        self._update_display()

    def _on_circular(self):
        """Set L to stable circular orbit at current r_init."""
        L = simulation.circular_L(self._r_init, self._spin)
        if not np.isfinite(L):
            return
        self._L = float(L)
        self._ctrl_widgets["L_spin"].setValue(self._L)
        self._recompute_timer.stop()
        self._recompute_and_display()

    def _on_play_pause(self, checked):
        self._playing = bool(checked)
        if self._playing:
            self._timer.start(self._tick_interval_ms)
            self._ctrl_widgets["play_btn"].setText("Pause")
        else:
            self._timer.stop()
            self._ctrl_widgets["play_btn"].setText("Play")

    def _on_speed_change(self, value):
        self._speed = int(value)


def run():
    """Launch the orbit simulation application."""
    _dbg("main.run() start", "B")
    app = QtWidgets.QApplication.instance()
    if app is None:
        app = QtWidgets.QApplication([])
    _dbg("after QApplication", "B")

    pg.setConfigOptions(antialias=True)
    win = OrbitSimWindow()
    _dbg("after OrbitSimWindow", "B")
    win.show()
    _dbg("after win.show, before app.exec", "B")
    return app.exec()


if __name__ == "__main__":
    run()
