"""
PyQt6 front-end for the employee attrition simulation: multi-scenario manager.

    python app.py

Main window: a scenario list (add / duplicate / remove), an editor for the
selected scenario, shared run settings (months, seed) and a results table with
one column per scenario. "Run All" simulates every scenario with the same
months and seed; the graphs open in two separate windows (predictors,
attrition), each overlaying all scenarios.

This file contains no statistics: all maths lives in simulation.py, all
drawing in plots.py.
"""
import sys

# Import PyQt6 BEFORE the matplotlib Qt backend so matplotlib picks this binding.
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QSplitter,
    QGroupBox, QFormLayout, QCheckBox, QDoubleSpinBox, QSpinBox, QLabel, QComboBox,
    QPushButton, QMessageBox, QListWidget, QListWidgetItem, QLineEdit,
    QTableWidget, QTableWidgetItem, QHeaderView, QAbstractItemView,
)
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg, NavigationToolbar2QT
from matplotlib.figure import Figure

from plots import plot_attrition, plot_predictors, plot_scatter
from simulation import (
    CSV_PATH, MC_VARS, N_STEPS, SEED, Scenario, default_scenarios, fresh_seed,
    load_and_fit, run_scenarios,
)

LABELS = {  # variable -> (display name, spin-box suffix)
    "overtime_hours_per_week": ("Overtime hours", " h/wk"),
    "burnout_score": ("Burnout score", " pts"),
    "engagement_score": ("Engagement score", " pts"),
    "manager_support_score": ("Manager support score", " pts"),
}

SHIFT_TOOLTIP = (
    "Ticked: this variable is shifted by exactly this amount.\n"
    "Unticked: it follows the shifts of the ticked variables through the "
    "dataset's correlations."
)

STALE_TEXT = "Out of date: scenarios or settings changed since the last run. Press Run All."


# --- value formatters --------------------------------------------------------
def _f2(v):  return f"{v:.2f}"
def _f3(v):  return f"{v:.3f}"
def _pct(v): return f"{v:.1%}"
def _int(v): return f"{int(v)}"


def _signed(v):
    """'+3.00' / '-0.62'; a value that rounds to zero is always '+0.00' (never '-0.00')."""
    s = f"{v:+.2f}"
    return "+0.00" if s == "-0.00" else s


# --- results table rows ------------------------------------------------------
# Each row = (label, function (result, pinned) -> text). `pinned` is the
# scenario's shifts dict as it was when the run started (for the set/auto tag).
def _summary_cell(key, fmt):
    return lambda r, pinned: fmt(r.summary[key])


def _shift_cell(var):
    def fmt(r, pinned):
        tag = "set" if var in pinned else "auto"      # set = pinned, auto = via correlations
        return f"{_signed(r.applied_shifts[var])} ({tag})"
    return fmt


METRICS = [
    # attrition
    ("Resignations",                _summary_cell("employee_attrition", _int)),
    ("Turnover rate",               _summary_cell("turnover_rate", _pct)),
    ("Avg attrition probability",   _summary_cell("avg_attrition_probability", _f3)),
    ("Months simulated",            _summary_cell("time_elapsed", _int)),
    # predictor averages
    ("Overtime hours per week",     _summary_cell("avg_hours_overtime", _f2)),
    ("Burnout score",               _summary_cell("avg_employee_burnout", _f2)),
    ("Engagement score",            _summary_cell("avg_employee_engagement", _f2)),
    ("Manager support score",       _summary_cell("avg_manager_support", _f2)),
    ("Years since promotion",       _summary_cell("avg_years_since_promotion", _f2)),
] + [
    # effective shifts (after correlation propagation)
    (f"Shift: {LABELS[var][0].lower()}", _shift_cell(var)) for var in MC_VARS
]


def unique_name(base, existing):
    """base, or 'base (2)', 'base (3)', ... so that it is not in `existing`."""
    if base not in existing:
        return base
    i = 2
    while f"{base} ({i})" in existing:
        i += 1
    return f"{base} ({i})"


# --- widgets -----------------------------------------------------------------
class MplCanvas(FigureCanvasQTAgg):
    """A matplotlib canvas that owns a plain Figure (no pyplot global state)."""

    def __init__(self):
        self.fig = Figure(layout="constrained")   # keeps axis labels from overlapping
        super().__init__(self.fig)


class PlotWindow(QWidget):
    """A standalone top-level window holding one canvas and its toolbar.
    It owns its plot function and exposes update_plot(histories), so the main
    window treats every figure window the same way. Closing it only hides it:
    the object lives on, so it can be redrawn and shown again at any time."""

    def __init__(self, title, plot_fn, parent=None):
        # Qt.WindowType.Window makes this a separate top-level window, while
        # `parent` keeps it owned by (and closed with) the main window.
        super().__init__(parent, Qt.WindowType.Window)
        self.setWindowTitle(title)
        self.plot_fn = plot_fn
        self.canvas = MplCanvas()
        lay = QVBoxLayout(self)
        lay.addWidget(NavigationToolbar2QT(self.canvas, self))
        lay.addWidget(self.canvas)
        self.resize(900, 700)

    def update_plot(self, histories):
        self.plot_fn(self.canvas.fig, histories)
        self.canvas.draw_idle()


class ScatterWindow(PlotWindow):
    """Four scatterplots of ONE scenario at a time, chosen in a combo box.
    (Color marks resigned vs stayed here, so scenarios cannot also be color-coded.)
    Switching scenario redraws from the stored histories; it does not re-run."""

    def __init__(self, parent=None):
        super().__init__("Attrition scatterplots", plot_scatter, parent)
        self.histories = {}
        self.combo = QComboBox()
        bar = QHBoxLayout()
        bar.addWidget(QLabel("Scenario:"))
        bar.addWidget(self.combo, 1)
        self.layout().insertLayout(0, bar)           # above the matplotlib toolbar
        self.combo.currentTextChanged.connect(self._draw)
        self.resize(1000, 800)

    def update_plot(self, histories):                # called after every Run All
        self.histories = histories
        previous = self.combo.currentText()
        self.combo.blockSignals(True)                # clear()/addItems() must not trigger redraws
        self.combo.clear()
        self.combo.addItems(list(histories))
        if previous in histories:
            self.combo.setCurrentText(previous)      # keep the user's choice across runs
        self.combo.blockSignals(False)
        self._draw()

    def _draw(self, *_):
        name = self.combo.currentText()
        if name not in self.histories:
            return
        self.plot_fn(self.canvas.fig, name, self.histories)
        self.canvas.draw_idle()


class ShiftRow:
    """One editor row: [x] name  [spin]. Ticked = pinned to that shift."""

    def __init__(self, name, suffix, span, on_edit):
        self.check = QCheckBox(name)
        self.check.setToolTip(SHIFT_TOOLTIP)

        self.spin = QDoubleSpinBox()
        self.spin.setRange(-span, span)
        self.spin.setSingleStep(0.5)
        self.spin.setDecimals(2)
        self.spin.setSuffix(suffix)
        self.spin.setMaximumWidth(140)
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
        self.setWindowTitle("Employee attrition simulation - scenarios")

        self.rows = {}               # variable -> ShiftRow (the editor)
        self.last_results = None     # name -> SimulationResult of the last run
        self.last_shifts = None      # name -> pinned shifts as they were at that run
        self._loading = False        # True while the editor is being filled programmatically
        self._plots_placed = False

        # The three graph windows: created now, hidden until the first run. Parented
        # to this window (so they are kept alive and closed with it).
        self.plot_windows = [
            PlotWindow("Predictor variables", plot_predictors, self),
            PlotWindow("Attrition", plot_attrition, self),
            ScatterWindow(self),
        ]

        central = QWidget()
        self.setCentralWidget(central)
        outer = QVBoxLayout(central)
        split = QSplitter(Qt.Orientation.Vertical)
        split.addWidget(self._build_top())
        split.addWidget(self._build_results())
        split.setChildrenCollapsible(False)
        split.setStretchFactor(0, 1)
        split.setStretchFactor(1, 2)
        split.setSizes([320, 430])
        outer.addWidget(split)

        # Connect once, here, after every widget exists.
        self.list.currentItemChanged.connect(self.on_selection_changed)
        self.name_edit.editingFinished.connect(self.on_name_edited)
        self.add_btn.clicked.connect(self.on_add)
        self.dup_btn.clicked.connect(self.on_duplicate)
        self.remove_btn.clicked.connect(self.on_remove)
        self.steps_spin.valueChanged.connect(self.mark_stale)
        self.seed_spin.valueChanged.connect(self.mark_stale)
        self.seed_check.toggled.connect(self.seed_spin.setEnabled)   # box only active when ticked
        self.seed_check.toggled.connect(self.mark_stale)             # a run setting changed
        self.run_btn.clicked.connect(self.on_run_all)
        self.show_graphs_btn.clicked.connect(self.on_show_graphs)

        self.update_controls()
        self.set_status("No results yet. Press Run All.")

    # ------------------------------------------------------------------ layout
    def _build_top(self):
        top = QWidget()
        lay = QHBoxLayout(top)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.addWidget(self._build_left_column())
        lay.addWidget(self._build_editor(), stretch=1)
        return top

    def _build_left_column(self):
        col = QWidget()
        col.setFixedWidth(300)
        lay = QVBoxLayout(col)
        lay.setContentsMargins(0, 0, 0, 0)

        # Scenario list + Add / Duplicate / Remove
        scen_box = QGroupBox("Scenarios")
        scen_lay = QVBoxLayout(scen_box)
        self.list = QListWidget()
        self.list.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        scen_lay.addWidget(self.list)
        btn_row = QHBoxLayout()
        self.add_btn = QPushButton("Add")
        self.dup_btn = QPushButton("Duplicate")
        self.remove_btn = QPushButton("Remove")
        for b in (self.add_btn, self.dup_btn, self.remove_btn):
            btn_row.addWidget(b)
        scen_lay.addLayout(btn_row)
        lay.addWidget(scen_box, stretch=1)

        # Run settings: global to every scenario
        run_box = QGroupBox("Run settings (all scenarios)")
        run_form = QFormLayout(run_box)
        self.steps_spin = QSpinBox()
        self.steps_spin.setRange(10, 100000)
        self.steps_spin.setValue(N_STEPS)
        self.steps_spin.setSuffix(" months")
        self.seed_spin = QSpinBox()
        self.seed_spin.setRange(0, 2**31 - 1)     # QSpinBox defaults to 0..99; 2**31-1 is its maximum
        self.seed_spin.setValue(SEED)
        self.seed_check = QCheckBox("Fixed seed")
        self.seed_check.setChecked(True)          # default = reproducible, as before
        self.seed_spin.setEnabled(self.seed_check.isChecked())
        self.seed_check.setToolTip(
            "Ticked: every run uses this seed, so results are reproducible.\n"
            "Unticked: a new random seed is drawn on each Run All (shown in the status line).")
        run_form.addRow("Months", self.steps_spin)
        run_form.addRow(self.seed_check, self.seed_spin)
        lay.addWidget(run_box)

        # Run All + Show graphs
        action_row = QHBoxLayout()
        self.run_btn = QPushButton("Run All")
        self.show_graphs_btn = QPushButton("Show graphs")
        self.show_graphs_btn.setEnabled(False)    # nothing to show until the first run
        action_row.addWidget(self.run_btn)
        action_row.addWidget(self.show_graphs_btn)
        lay.addLayout(action_row)
        return col

    def _build_editor(self):
        self.editor_box = QGroupBox("Scenario editor")
        lay = QVBoxLayout(self.editor_box)

        form = QFormLayout()
        self.name_edit = QLineEdit()
        form.addRow("Name:", self.name_edit)

        for var in MC_VARS:
            name, suffix = LABELS[var]
            span = 100 if var == "overtime_hours_per_week" else 10

            row = ShiftRow(name, suffix, span, self.on_editor_changed)
            self.rows[var] = row
            form.addRow(row.check, row.spin)

        lay.addLayout(form)

        note = QLabel("Ticked = pinned to exactly this shift.\n"
                      "Unticked = follows the correlations (see the 'Shift' rows "
                      "of the results table).")
        note.setWordWrap(True)
        lay.addWidget(note)
        lay.addStretch()
        return self.editor_box

    def _build_results(self):
        box = QGroupBox("Results")
        lay = QVBoxLayout(box)
        self.status_label = QLabel()
        lay.addWidget(self.status_label)

        # Rows are fixed (METRICS); columns change with every run.
        t = self.results_table = QTableWidget()
        t.setRowCount(len(METRICS))
        t.setVerticalHeaderLabels([label for label, _ in METRICS])
        t.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        t.setAlternatingRowColors(True)
        t.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        t.horizontalHeader().setMinimumSectionSize(120)   # many scenarios -> scroll, not squash
        lay.addWidget(t)
        return box

    # ------------------------------------------------------------------ status
    def set_status(self, text, stale=False):
        self.status_label.setText(text)
        self.status_label.setStyleSheet("color: #b45f06;" if stale else "")

    def mark_stale(self, *_):
        """Results on screen no longer match the scenarios/settings on screen."""
        if self.last_results is not None:         # only meaningful after a first run
            self.set_status(STALE_TEXT, stale=True)

    def update_controls(self):
        """Enable/disable controls from the list's state. Safe to call at any time."""
        has = self.list.currentItem() is not None
        self.editor_box.setEnabled(has)
        self.dup_btn.setEnabled(has)
        self.remove_btn.setEnabled(has)
        self.run_btn.setEnabled(self.list.count() > 0)

    # --------------------------------------------------------- scenario list
    # Each Scenario object lives INSIDE its list item, so the list and the data
    # cannot drift out of sync.
    def add_scenario(self, sc):
        item = QListWidgetItem(sc.name)
        item.setData(Qt.ItemDataRole.UserRole, sc)
        self.list.addItem(item)
        return item

    def current_scenario(self):
        item = self.list.currentItem()
        return item.data(Qt.ItemDataRole.UserRole) if item else None

    def scenarios(self):
        """In list order = column order in the results table."""
        return [self.list.item(i).data(Qt.ItemDataRole.UserRole)
                for i in range(self.list.count())]

    def on_add(self):
        names = [s.name for s in self.scenarios()]
        item = self.add_scenario(Scenario(unique_name("New scenario", names)))
        self.list.setCurrentItem(item)
        self.name_edit.setFocus()                 # let the user type a name right away
        self.name_edit.selectAll()
        self.update_controls()
        self.mark_stale()

    def on_duplicate(self):
        sc = self.current_scenario()
        if sc is None:
            return
        names = [s.name for s in self.scenarios()]
        # dict(...): a COPY. Sharing the dict would make the two scenarios edit each other.
        copy = Scenario(unique_name(f"{sc.name} copy", names), dict(sc.shifts))
        self.list.setCurrentItem(self.add_scenario(copy))
        self.update_controls()
        self.mark_stale()

    def on_remove(self):
        row = self.list.currentRow()
        if row < 0:
            return
        self.list.takeItem(row)                   # Qt then selects a neighbour (or nothing)
        # Re-evaluate AFTER the row is gone: the selection-changed signal fires
        # while the removed row still counts in list.count().
        self.update_controls()
        self.mark_stale()

    # ------------------------------------------------- list <-> editor binding
    def on_selection_changed(self, current, previous):
        self.update_controls()
        sc = self.current_scenario()
        if sc is not None:
            self.load_editor(sc)

    def load_editor(self, sc):
        """Fill the editor from a scenario. Programmatic changes must not write back."""
        self._loading = True
        try:
            self.name_edit.setText(sc.name)
            for var, row in self.rows.items():
                pinned = var in sc.shifts
                row.set_state(pinned, sc.shifts.get(var, 0.0))
        finally:
            self._loading = False

    def collect_shifts(self):
        """Only TICKED rows go into the dict: those variables are pinned.
        Unticked ones are left out so they follow the correlations."""
        return {var: row.spin.value()
                for var, row in self.rows.items() if row.check.isChecked()}

    def on_editor_changed(self, *_):
        """User edited a shift row -> write it into the selected scenario."""
        if self._loading:
            return
        sc = self.current_scenario()
        if sc is None:
            return
        sc.shifts = self.collect_shifts()         # a NEW dict each time
        self.mark_stale()

    def on_name_edited(self):
        """Rename on editingFinished (Enter / focus out), not on every keystroke:
        mid-typing the text would constantly collide with other names."""
        if self._loading:
            return
        sc, item = self.current_scenario(), self.list.currentItem()
        if sc is None:
            return
        others = [s.name for s in self.scenarios() if s is not sc]
        new = unique_name(self.name_edit.text().strip() or "Scenario", others)
        self.name_edit.setText(new)               # show the corrected name
        if new == sc.name:
            return                                # nothing changed: results are not stale
        sc.name = new
        item.setText(new)
        self.mark_stale()

    # ------------------------------------------------------------ results table
    def fill_results_table(self, results, pinned_by_name):
        t = self.results_table
        names = list(results)
        t.setColumnCount(0)                       # drop old columns (do NOT call t.clear():
        t.setColumnCount(len(names))              #   it also erases the vertical header labels)
        t.setHorizontalHeaderLabels(names)
        for col, name in enumerate(names):
            for row, (_, fmt) in enumerate(METRICS):
                item = QTableWidgetItem(fmt(results[name], pinned_by_name[name]))
                item.setTextAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
                t.setItem(row, col, item)

    # ------------------------------------------------------------------- graphs
    def _place_plot_windows(self):
        """Cascade the two windows once, so they don't open on top of each other."""
        if self._plots_placed:
            return
        self._plots_placed = True
        for i, win in enumerate(self.plot_windows):
            win.move(self.x() + 60 + 40 * i, self.y() + 60 + 40 * i)

    def draw_graphs(self, histories):
        self._place_plot_windows()
        for win in self.plot_windows:
            win.update_plot(histories)
            if not win.isVisible():
                win.show()
                win.raise_()

    def on_show_graphs(self):
        """Reopen windows the user closed, without re-running."""
        for win in self.plot_windows:
            win.show()
            win.raise_()

    # ---------------------------------------------------------------------- run
    def resolve_seed(self):
        """Return (seed, text for the status line) for this run. Unseeded = ONE fresh
        seed shared by all scenarios (so they still see the same random draws)."""
        if self.seed_check.isChecked():
            seed = self.seed_spin.value()
            return seed, f"seed {seed} (fixed)"
        seed = fresh_seed()
        return seed, f"seed {seed} (random)"

    def on_run_all(self):
        scenarios = self.scenarios()
        if not scenarios:
            return
        pinned = {sc.name: dict(sc.shifts) for sc in scenarios}   # snapshot taken at run time
        n_steps = self.steps_spin.value()
        seed, seed_text = self.resolve_seed()
        try:
            results, errors = run_scenarios(self.df, self.corr, self.model, scenarios,
                                            n_steps=n_steps, seed=seed)
        except Exception as exc:                  # slots must not raise (it would kill the app)
            QMessageBox.critical(self, "Run failed", str(exc))
            return

        if errors:
            QMessageBox.warning(self, "Some scenarios failed",
                                "\n".join(f"{name}: {msg}" for name, msg in errors.items()))
        if not results:
            return

        self.last_results, self.last_shifts = results, pinned
        self.fill_results_table(results, pinned)
        self.draw_graphs({name: r.history for name, r in results.items()})
        self.show_graphs_btn.setEnabled(True)
        failed = f", {len(errors)} failed" if errors else ""
        if not self.seed_check.isChecked():
            # Show the random seed in the (disabled) box: ticking "Fixed seed" afterwards
            # reproduces this exact run. Signals blocked, else valueChanged would mark the
            # results we just produced as out of date.
            self.seed_spin.blockSignals(True)
            self.seed_spin.setValue(seed)
            self.seed_spin.blockSignals(False)
        self.set_status(f"Up to date: {len(results)} scenario(s){failed}, "
                        f"{n_steps} months, {seed_text}")

    # ----------------------------------------------------------------- shutdown
    def closeEvent(self, event):
        for win in self.plot_windows:
            win.close()
        super().closeEvent(event)


def install_excepthook():
    """An exception raised inside a Qt slot would otherwise abort the whole app.
    Show it in a dialog instead (the traceback is still printed to the terminal)."""
    def hook(exc_type, exc, tb):
        sys.__excepthook__(exc_type, exc, tb)
        if not issubclass(exc_type, KeyboardInterrupt):
            QMessageBox.critical(None, "Unexpected error", f"{exc_type.__name__}: {exc}")
    sys.excepthook = hook


def main():
    app = QApplication(sys.argv)
    install_excepthook()
    try:
        # The dataset never changes during a session: fit once, reuse on every run.
        df, corr, model = load_and_fit()
    except Exception as exc:        # missing file, missing columns, fit failure
        QMessageBox.critical(None, "Cannot load dataset", f"{CSV_PATH}\n\n{exc}")
        sys.exit(1)

    win = MainWindow(df, corr, model)
    for sc in default_scenarios():            # baseline, heavy overtime, weak management
        win.add_scenario(sc)
    win.list.setCurrentRow(0)
    win.resize(1100, 750)
    win.show()
    QTimer.singleShot(0, win.on_run_all)      # run once so nothing is empty at launch
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
