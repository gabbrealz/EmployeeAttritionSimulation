"""
PyQt6 front-end for the employee attrition simulation.

    python app.py

Layout: scenario controls + Run button on the left; the predictor graph, the
attrition graph and the numeric outputs on the right. This file contains no
statistics: all maths lives in simulation.py, all drawing in plots.py.
"""
import sys

import numpy as np

# Import PyQt6 BEFORE the matplotlib Qt backend so matplotlib picks this binding.
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QFontDatabase
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QSplitter,
    QGroupBox, QFormLayout, QComboBox, QCheckBox, QDoubleSpinBox, QSpinBox,
    QLabel, QPushButton, QMessageBox,
)
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg, NavigationToolbar2QT
from matplotlib.figure import Figure

from plots import plot_attrition, plot_predictors
from simulation import (
    CSV_PATH, MC_VARS, N_STEPS, SCENARIOS, SEED, load_and_fit, run_simulation,
)

CUSTOM = "Custom"

LABELS = {  # variable -> (display name, spin-box suffix)
    "overtime_hours_per_week": ("Overtime Hours", " h/wk"),
    "burnout_score": ("Burnout Score", " pts"),
    "engagement_score": ("Engagement Score", " pts"),
    "manager_support_score": ("Manager Support Score", " pts"),
}

SHIFT_TOOLTIP = (
    "Ticked: this variable is shifted by exactly this amount.\n"
    "Unticked: it follows the shifts of the ticked variables through the "
    "dataset's correlations."
)


# --- value formatters for the outputs panel ---------------------------------
def _f2(v):  return f"{v:.2f}"
def _f3(v):  return f"{v:.3f}"
def _pct(v): return f"{v:.1%}"
def _int(v): return f"{int(v)}"


def _signed(v):
    """'+3.00' / '-0.62'; a value that rounds to zero is always '+0.00' (never '-0.00')."""
    s = f"{v:+.2f}"
    return "+0.00" if s == "-0.00" else s


# (row label, key in result.summary, formatter)
PREDICTOR_OUTPUTS = [
    ("Overtime Hours (h/week)", "avg_hours_overtime", _f2),
    ("Burnout Score", "avg_employee_burnout", _f2),
    ("Engagement Score", "avg_employee_engagement", _f2),
    ("Manager support Score", "avg_manager_support", _f2),
    ("Years Since Promotion", "avg_years_since_promotion", _f2),
]
ATTRITION_OUTPUTS = [
    ("Attrition", "employee_attrition", _int),
    ("Turnover rate", "turnover_rate", _pct),
    ("Avg Attrition Probability", "avg_attrition_probability", _f3),
    ("Years simulated", "time_elapsed", _int),
]


class MplCanvas(FigureCanvasQTAgg):
    """A matplotlib canvas that owns a plain Figure (no pyplot global state)."""

    def __init__(self):
        self.fig = Figure(layout="constrained")   # keeps axis labels from overlapping
        super().__init__(self.fig)


def make_plot_panel(title, canvas, parent):
    """Titled group box holding matplotlib's toolbar (zoom, pan, save PNG) + the canvas."""
    box = QGroupBox(title)
    lay = QVBoxLayout(box)
    lay.addWidget(NavigationToolbar2QT(canvas, parent))
    lay.addWidget(canvas)
    return box


class ShiftRow:
    """One scenario row: [x] name  [spin]. Ticked = pinned to that shift."""

    def __init__(self, name, suffix, span, on_edit):
        self.check = QCheckBox(name)
        self.check.setToolTip(SHIFT_TOOLTIP)

        self.spin = QDoubleSpinBox()
        self.spin.setRange(-span, span)
        self.spin.setSingleStep(0.5)
        self.spin.setDecimals(2)
        self.spin.setSuffix(suffix)
        self.spin.setEnabled(False)               # disabled until ticked
        self.spin.setToolTip(SHIFT_TOOLTIP)

        self.check.toggled.connect(self.spin.setEnabled)
        self.check.toggled.connect(on_edit)
        self.spin.valueChanged.connect(on_edit)

    def set_state(self, ticked, value):
        """Set the row programmatically WITHOUT firing the user-edit handlers.
        Signals are blocked, so the enabled state must be set explicitly."""
        widgets = (self.check, self.spin)
        for w in widgets:
            w.blockSignals(True)
        try:
            self.check.setChecked(ticked)
            self.spin.setValue(value)
            self.spin.setEnabled(ticked)
        finally:
            for w in widgets:
                w.blockSignals(False)


class MainWindow(QMainWindow):
    def __init__(self, df, corr, model):
        super().__init__()
        self.df, self.corr, self.model = df, corr, model
        self.setWindowTitle("Employee attrition simulation")

        self.rows = {}            # variable -> ShiftRow
        self.value_labels = {}    # summary key -> (QLabel, formatter)
        self.pred_canvas = MplCanvas()
        self.attr_canvas = MplCanvas()

        central = QWidget()
        self.setCentralWidget(central)
        root = QHBoxLayout(central)
        root.addWidget(self._build_left_panel())
        root.addWidget(self._build_right_panel(), stretch=1)

        # Connect once, here. (Signals are connected after all widgets exist.)
        self.preset_combo.currentTextChanged.connect(self.on_preset_changed)
        self.run_btn.clicked.connect(self.on_run)
        self.statusBar().showMessage("Ready")

    # ------------------------------------------------------------------ layout
    def _build_left_panel(self):
        panel = QWidget()
        panel.setFixedWidth(320)
        lay = QVBoxLayout(panel)
        lay.setContentsMargins(0, 0, 0, 0)

        # Preset selector
        preset_row = QHBoxLayout()
        preset_row.addWidget(QLabel("Preset:"))
        self.preset_combo = QComboBox()
        self.preset_combo.addItems(list(SCENARIOS) + [CUSTOM])
        preset_row.addWidget(self.preset_combo, stretch=1)
        lay.addLayout(preset_row)

        # Scenario: one row per Monte Carlo variable
        scenario_box = QGroupBox("Scenario (shift per variable)")
        form = QFormLayout(scenario_box)
        for var in MC_VARS:
            name, suffix = LABELS[var]
            # Range comes from the dataset, not hard-coded.
            span = float(self.df[var].max() - self.df[var].min())
            row = ShiftRow(name, suffix, span, self.on_row_edited)
            self.rows[var] = row
            form.addRow(row.check, row.spin)
        lay.addWidget(scenario_box)

        # Run settings (extension)
        run_box = QGroupBox("Run settings")
        run_form = QFormLayout(run_box)
        self.steps_spin = QSpinBox()
        self.steps_spin.setRange(10, 100000)
        self.steps_spin.setValue(N_STEPS)
        self.steps_spin.setSuffix(" years")
        self.seed_spin = QSpinBox()
        self.seed_spin.setRange(0, 2147483647)    # largest value a QSpinBox holds
        self.seed_spin.setValue(SEED)
        run_form.addRow("Years", self.steps_spin)
        run_form.addRow("Seed", self.seed_spin)
        lay.addWidget(run_box)

        # Effective shifts (after correlation propagation)
        shifts_box = QGroupBox("Effective shifts")
        shifts_lay = QVBoxLayout(shifts_box)
        self.shifts_label = QLabel("Run a simulation to see the shifts applied.")
        self.shifts_label.setFont(QFontDatabase.systemFont(QFontDatabase.SystemFont.FixedFont))
        self.shifts_label.setWordWrap(True)
        self.shifts_label.setToolTip(
            "set  = pinned by you (ticked row)\n"
            "auto = propagated from the pinned variables via the dataset's correlations"
        )
        shifts_lay.addWidget(self.shifts_label)
        lay.addWidget(shifts_box)

        lay.addStretch()
        self.run_btn = QPushButton("Run Simulation")
        lay.addWidget(self.run_btn)
        return panel

    def _build_right_panel(self):
        # Top: the two graphs side by side
        plots = QSplitter(Qt.Orientation.Horizontal)
        plots.addWidget(make_plot_panel("Predictors", self.pred_canvas, self))
        plots.addWidget(make_plot_panel("Attrition", self.attr_canvas, self))
        plots.setChildrenCollapsible(False)

        # Bottom: the two numeric output boxes
        outputs = QWidget()
        out_lay = QHBoxLayout(outputs)
        out_lay.setContentsMargins(0, 0, 0, 0)
        out_lay.addWidget(self._build_output_box("Predictor averages", PREDICTOR_OUTPUTS))
        out_lay.addWidget(self._build_output_box("Attrition", ATTRITION_OUTPUTS))

        split = QSplitter(Qt.Orientation.Vertical)
        split.addWidget(plots)
        split.addWidget(outputs)
        split.setChildrenCollapsible(False)
        split.setStretchFactor(0, 3)              # graphs get ~3x the height of the outputs
        split.setStretchFactor(1, 1)
        split.setSizes([600, 200])
        return split

    def _build_output_box(self, title, fields):
        box = QGroupBox(title)
        form = QFormLayout(box)
        for text, key, fmt in fields:
            value = QLabel("–")
            value.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
            value.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
            form.addRow(text, value)
            self.value_labels[key] = (value, fmt)
        return box

    # ---------------------------------------------------------------- scenario
    def collect_shifts(self):
        """Only TICKED rows go into the dict: those variables are pinned.
        Unticked ones are left out so they follow the correlations."""
        return {var: row.spin.value()
                for var, row in self.rows.items() if row.check.isChecked()}

    def apply_preset(self, shifts):
        """Ticked + preset value if the variable is in the preset, else unticked at 0."""
        for var, row in self.rows.items():
            row.set_state(var in shifts, shifts.get(var, 0.0))

    def on_preset_changed(self, name):
        if name != CUSTOM:
            self.apply_preset(SCENARIOS[name])

    def on_row_edited(self, *_):
        """User edited a row by hand -> the combo no longer describes the scenario."""
        if self.preset_combo.currentText() != CUSTOM:
            self.preset_combo.blockSignals(True)   # else this would re-trigger on_preset_changed
            self.preset_combo.setCurrentText(CUSTOM)
            self.preset_combo.blockSignals(False)

    # ----------------------------------------------------------------- outputs
    def show_summary(self, result):
        for key, (label, fmt) in self.value_labels.items():
            label.setText(fmt(result.summary[key]))

    def show_effective_shifts(self, applied, requested):
        """One line per variable: value + origin ('set' = pinned, 'auto' = propagated)."""
        lines = []
        for var in MC_VARS:
            origin = "set" if var in requested else "auto"
            lines.append(f"{LABELS[var][0]:<16}{_signed(applied[var])} ({origin})")
        self.shifts_label.setText("\n".join(lines))

    # --------------------------------------------------------------------- run
    def on_run(self):
        shifts = self.collect_shifts()
        try:
            result = run_simulation(self.df, self.corr, self.model, shifts,
                                    n_steps=self.steps_spin.value(),
                                    seed=self.seed_spin.value())
        except (ValueError, np.linalg.LinAlgError) as exc:
            # e.g. a singular covariance sub-matrix when ticked variables are perfectly correlated
            QMessageBox.warning(self, "Simulation error", str(exc))
            return

        plot_predictors(self.pred_canvas.fig, result.history)
        plot_attrition(self.attr_canvas.fig, result.history)
        self.pred_canvas.draw_idle()
        self.attr_canvas.draw_idle()

        self.show_summary(result)
        self.show_effective_shifts(result.applied_shifts, shifts)
        self.statusBar().showMessage(
            f"Ran {len(result.history)} years (seed {self.seed_spin.value()})")


def main():
    app = QApplication(sys.argv)
    try:
        # The dataset never changes during a session: fit once, reuse on every run.
        df, corr, model = load_and_fit()
    except Exception as exc:        # missing file, missing columns, fit failure
        QMessageBox.critical(None, "Cannot load dataset", f"{CSV_PATH}\n\n{exc}")
        sys.exit(1)

    win = MainWindow(df, corr, model)
    win.resize(1300, 800)
    win.show()
    QTimer.singleShot(0, win.on_run)    # run the baseline once so the window isn't empty
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
