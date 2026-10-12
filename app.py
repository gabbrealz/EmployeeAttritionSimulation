import sys

# Import PyQt6 BEFORE the matplotlib Qt backend so matplotlib picks this binding.
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QGroupBox, QFormLayout, QCheckBox, QDoubleSpinBox, QSpinBox, QLabel, QComboBox,
    QPushButton, QMessageBox, QListWidget, QListWidgetItem, QLineEdit,
    QTableWidget, QTableWidgetItem, QHeaderView, QAbstractItemView, QGridLayout,
    QTabWidget, QStackedWidget, QSizePolicy,
)
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg, NavigationToolbar2QT
from matplotlib.figure import Figure

from plots import plot_attrition, plot_predictors, plot_scatter
from simulation import (
    CSV_PATH, MC_VARS, N_STEPS, SEED, TEST_SIZE, Scenario, default_scenarios, fresh_seed,
    load_and_fit, run_replications, run_scenarios,
)
from PyQt6.QtGui import QColor

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
    ("Employee attrition",               _summary_cell("employee_attrition", _int)),
    ("Actual turnover rate",             _summary_cell("turnover_rate", _pct)),
    ("Avg. attrition probability",       _summary_cell("avg_attrition_probability", _f3)),
    # predictor averages
    ("Avg. overtime hours per week",     _summary_cell("avg_hours_overtime", _f2)),
    ("Avg. burnout score",               _summary_cell("avg_employee_burnout", _f2)),
    ("Avg. engagement score",            _summary_cell("avg_employee_engagement", _f2)),
    ("Avg. manager support score",       _summary_cell("avg_manager_support", _f2)),
    ("Avg. years since promotion",       _summary_cell("avg_years_since_promotion", _f2)),
    ("Avg. years at resignation",        _summary_cell("avg_years_at_resignation", _f2)),
    ("Time elapsed (months)",            _summary_cell("time_elapsed", _int)),
] + [
    # effective shifts (after correlation propagation)
    (f"Shift: {LABELS[var][0].lower()}", _shift_cell(var)) for var in MC_VARS
]

# --- AI evaluation metrics (the attrition model's hold-out scores) -----------
# (key in the dict returned by load_and_fit, caption, tooltip), in display order.
MODEL_METRICS = [
    ("accuracy", "Accuracy"),
    ("precision", "Precision"),
    ("recall", "Recall"),
    ("f1", "F1 score"),
    ("roc_auc", "ROC-AUC"),
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


class PlotPanel(QWidget):
    """One graph page inside the Visualizations tab. It owns its
    plot function and exposes update_plot(histories), so the main window treats
    every graph the same way."""

    def __init__(self, title, plot_fn, parent=None):
        super().__init__(parent)                 # a plain child widget, NOT a top-level window
        self.title = title                       # shown in the graph drop-down
        self.plot_fn = plot_fn
        self.canvas = MplCanvas()
        lay = QVBoxLayout(self)
        lay.addWidget(NavigationToolbar2QT(self.canvas, self))
        lay.addWidget(self.canvas)

    def update_plot(self, histories):
        self.plot_fn(self.canvas.fig, histories)
        self.canvas.draw_idle()


class ScatterPanel(PlotPanel):
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


class ReplicationWindow(QWidget):
    """Standalone window: the n-run comparison, one row per scenario."""
    COLUMNS = ["Scenario", "Avg. resignations", "± 95% CI", "Avg. attrition prob.",
               "Avg. years at resignation", "Reduction vs baseline", "Meets target"]

    def __init__(self, parent=None):
        super().__init__(parent, Qt.WindowType.Window)
        self.setWindowTitle("Replication results")
        self.note = QLabel()
        self.note.setWordWrap(True)
        self.table = QTableWidget()
        self.table.setColumnCount(len(self.COLUMNS))
        self.table.setHorizontalHeaderLabels(self.COLUMNS)
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.setAlternatingRowColors(True)
        self.table.verticalHeader().setVisible(False)
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        lay = QVBoxLayout(self)
        lay.addWidget(self.note)
        lay.addWidget(self.table)
        self.resize(900, 360)

    def show_results(self, table, baseline, n_runs, n_steps, target, seed):
        self.note.setText(
            f"{n_runs} runs x {n_steps} months per scenario (seeds {seed} to {seed + n_runs - 1}). "
            f"Baseline = {baseline}. Reduction = fewer resignations than the baseline "
            f"(positive = better). Target: at least {target:g}%.")
        self.table.setRowCount(len(table.index))
        for r, name in enumerate(table.index):
            is_base = name == baseline
            red = table.loc[name, ("employee_attrition", "reduction_%")]
            meets = (not is_base) and red >= target
            cells = [
                name,
                f"{table.loc[name, ('employee_attrition', 'mean')]:.1f}",
                f"±{table.loc[name, ('employee_attrition', 'ci95')]:.1f}",
                f"{table.loc[name, ('avg_attrition_probability', 'mean')]:.4f}",
                "-" if is_base else f"{red:+.1f}%",
                "-" if is_base else ("Yes" if meets else "No"),
                f"{table.loc[name, ('avg_years_at_resignation', 'mean')]:.2f}"
            ]
            for c, text in enumerate(cells):
                item = QTableWidgetItem(text)
                item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                if meets:
                    item.setBackground(QColor("#d9ead3"))   # light green = target met
                    item.setForeground(QColor("#000000"))   # dark text: readable in light and dark mode
                self.table.setItem(r, c, item)


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
    def __init__(self, df, corr, model, metrics):
        super().__init__()
        self.df, self.corr, self.model = df, corr, model
        self.metrics = metrics       # the model's hold-out scores (fixed at fit time)
        self.setWindowTitle("Employee attrition simulation - scenarios")

        self.rows = {}               # variable -> ShiftRow (the editor)
        self.last_results = None     # name -> SimulationResult of the last run
        self.last_shifts = None      # name -> pinned shifts as they were at that run
        self._loading = False        # True while the editor is being filled programmatically

        self.rep_window = ReplicationWindow(self)

        self.tabs = QTabWidget()                      # its tab bar = the navigation bar
        self.tabs.setDocumentMode(True)
        self.setCentralWidget(self.tabs)
        self.tabs.addTab(self._build_home(), "Home")
        self.tabs.addTab(self._build_visualizations(), "Visualizations")

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
        self.rep_btn.clicked.connect(self.on_run_replications)
        # Graph navigation: the combo is the single source of truth; the stack follows it.
        self.viz_combo.currentIndexChanged.connect(self.viz_stack.setCurrentIndex)
        self.prev_btn.clicked.connect(self.on_prev_graph)
        self.next_btn.clicked.connect(self.on_next_graph)

        self.update_controls()
        self.set_graphs_available(False)          # show the hint until the first run
        self.set_status("No results yet. Press Run All.")

    # ------------------------------------------------------------------ layout
    def _build_home(self):
        """The Home tab, top to bottom: settings / scenarios / editor, the AI
        evaluation metrics, then the results. The first two are held at the height
        of their content, so the results table gets all the spare window height."""
        page = QWidget()
        lay = QVBoxLayout(page)
        lay.addWidget(self._build_top())
        lay.addWidget(self._build_model_metrics())
        lay.addWidget(self._build_results(), 1)
        return page

    def _build_visualizations(self):
        """The Visualizations tab: a navigation row, then ONE graph at a time."""
        page = QWidget()
        lay = QVBoxLayout(page)

        # Navigation row: Graph drop-down, Previous, Next, (spacer), status
        nav = QHBoxLayout()
        nav.addWidget(QLabel("Graph:"))
        self.viz_combo = QComboBox()
        self.viz_combo.setMinimumWidth(220)
        nav.addWidget(self.viz_combo)
        self.prev_btn = QPushButton("Previous")
        self.next_btn = QPushButton("Next")
        nav.addWidget(self.prev_btn)
        nav.addWidget(self.next_btn)
        nav.addStretch(1)
        self.viz_status = QLabel()                # mirrors the Home status line
        nav.addWidget(self.viz_status)
        lay.addLayout(nav)

        # Shown until the first successful run
        self.viz_hint = QLabel("No graphs yet. Press Run All on the Home tab.")
        self.viz_hint.setAlignment(Qt.AlignmentFlag.AlignCenter)
        lay.addWidget(self.viz_hint, 1)

        # One page per graph; a QStackedWidget shows exactly one at a time
        self.viz_stack = QStackedWidget()
        self.panels = [
            PlotPanel("Predictor variables", plot_predictors),
            PlotPanel("Attrition", plot_attrition),
            ScatterPanel(),
        ]
        for panel in self.panels:
            self.viz_stack.addWidget(panel)       # also reparents the panel into the stack
            self.viz_combo.addItem(panel.title)   # same order => combo index == stack index
        lay.addWidget(self.viz_stack, 1)
        return page

    def _build_top(self):
        top = QWidget()
        # Never taller than its content. The grid row is as tall as its tallest
        # column (the run column), and the other two columns stretch to match it.
        top.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Maximum)
        grid = QGridLayout(top)
        grid.setContentsMargins(0, 0, 0, 0)
        grid.addWidget(self._build_run_column(), 0, 0)
        grid.addWidget(self._build_scenarios(),  0, 1)
        grid.addWidget(self._build_editor(),     0, 2)
        for col, (stretch, min_w) in enumerate([(1, 250), (1, 240), (2, 340)]):
            grid.setColumnStretch(col, stretch)
            grid.setColumnMinimumWidth(col, min_w)
        return top

    def _build_run_column(self):
        """Column 1: two group boxes, each with its own Run button INSIDE it.
        The boxes take only the height their content needs (no stretch), so each
        border sits snug against its fields and button. This column therefore sets
        the height of the whole top row; the Scenarios and Scenario editor boxes
        beside it stretch to match (see _build_scenarios). The addStretch() at the
        bottom of each box only matters if the column is ever given extra height:
        it keeps the contents packed at the top."""
        col = QWidget()
        lay = QVBoxLayout(col)
        lay.setContentsMargins(0, 0, 0, 0)

        # --- Simulation settings: Months, Fixed seed, Run All ----------------
        sim_box = QGroupBox("Simulation settings")
        sim_lay = QVBoxLayout(sim_box)
        sim_form = QFormLayout()
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
        sim_form.addRow("Months", self.steps_spin)
        sim_form.addRow(self.seed_check, self.seed_spin)
        sim_lay.addLayout(sim_form)

        self.run_btn = QPushButton("Run All")
        sim_lay.addWidget(self.run_btn)
        sim_lay.addStretch()                      # fields + button stay packed at the top of the box
        lay.addWidget(sim_box)                    # no stretch: the box wraps its content

        # --- Replication Settings: Replications, Target reduction, Run replications
        rep_box = QGroupBox("Replication Settings")
        rep_lay = QVBoxLayout(rep_box)
        rep_form = QFormLayout()
        self.runs_spin = QSpinBox()
        self.runs_spin.setRange(2, 10000)
        self.runs_spin.setValue(100)
        self.runs_spin.setSuffix(" runs")
        self.target_spin = QDoubleSpinBox()
        self.target_spin.setRange(0, 100)
        self.target_spin.setValue(10.0)
        self.target_spin.setSuffix(" %")
        rep_form.addRow("Replications", self.runs_spin)
        rep_form.addRow("Target reduction", self.target_spin)
        rep_lay.addLayout(rep_form)

        self.rep_btn = QPushButton("Run replications")
        self.rep_btn.setToolTip("Runs every scenario many times and compares each with the "
                                "FIRST scenario in the list (the baseline).")
        rep_lay.addWidget(self.rep_btn)
        rep_lay.addStretch()                      # fields + button stay packed at the top of the box
        lay.addWidget(rep_box)                    # no stretch: the box wraps its content

        return col

    def _build_scenarios(self):
        scen_box = QGroupBox("Scenarios")
        scen_lay = QVBoxLayout(scen_box)
        self.list = QListWidget()
        self.list.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.list.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Ignored)
        scen_lay.addWidget(self.list, 1)
        btn_row = QHBoxLayout()
        self.add_btn = QPushButton("Add")
        self.dup_btn = QPushButton("Duplicate")
        self.remove_btn = QPushButton("Remove")
        for b in (self.add_btn, self.dup_btn, self.remove_btn):
            btn_row.addWidget(b)
        scen_lay.addLayout(btn_row)
        return scen_box

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
        lay.addStretch()
        return self.editor_box

    def _build_model_metrics(self):
        """'AI Evaluation Metrics': how well the logistic regression predicts attrition
        on its hold-out split, one caption + value per metric in a single row. The
        scores are fixed when the model is fitted, so they are filled once here and
        do not change on Run All."""
        box = QGroupBox("AI Evaluation Metrics")
        lay = QHBoxLayout(box)
        for key, caption in MODEL_METRICS:
            cell = QVBoxLayout()
            cell.setSpacing(0)
            name = QLabel(caption)
            value = QLabel(_f3(self.metrics[key]))
            value.setStyleSheet("font-size: 13pt; font-weight: bold;")
            for label in (name, value):
                label.setAlignment(Qt.AlignmentFlag.AlignCenter)
                cell.addWidget(label)
            lay.addLayout(cell, 1)                # equal width for every metric
        # Exactly as tall as its content, like the top grid above it.
        box.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Maximum)
        return box

    def _build_results(self):
        box = QGroupBox("Simulation Outputs")
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
        """Show the status on the Home tab AND on the Visualizations tab: the graphs
        go stale exactly when the results table does."""
        style = "color: #b45f06;" if stale else ""
        for label in (self.status_label, self.viz_status):
            label.setText(text)
            label.setStyleSheet(style)

    # ------------------------------------------------------- graph navigation
    def on_prev_graph(self):
        n = self.viz_combo.count()
        self.viz_combo.setCurrentIndex((self.viz_combo.currentIndex() - 1) % n)   # wraps around

    def on_next_graph(self):
        n = self.viz_combo.count()
        self.viz_combo.setCurrentIndex((self.viz_combo.currentIndex() + 1) % n)

    def set_graphs_available(self, available):
        """Before the first run there is nothing to show: disable the graph controls
        and show the hint instead of empty canvases."""
        for w in (self.viz_combo, self.prev_btn, self.next_btn):
            w.setEnabled(available)
        self.viz_stack.setVisible(available)
        self.viz_hint.setVisible(not available)

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
        self.rep_btn.setEnabled(self.list.count() > 0)

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
    def draw_graphs(self, histories):
        """Redraw every graph page in place. Nothing is shown or raised: the graphs live
        in the Visualizations tab, which the user opens when they want to look."""
        for panel in self.panels:
            panel.update_plot(histories)
        self.set_graphs_available(True)

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
        
    def on_run_replications(self):
        scenarios = self.scenarios()
        if not scenarios:
            return
        n_runs, n_steps = self.runs_spin.value(), self.steps_spin.value()
        target = self.target_spin.value()
        seed, _ = self.resolve_seed()
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        try:
            _, table = run_replications(self.df, self.corr, self.model, scenarios,
                                        n_runs=n_runs, n_steps=n_steps,
                                        base_seed=seed, target=target)
        except Exception as exc:
            QMessageBox.critical(self, "Replications failed", str(exc))
            return
        finally:
            QApplication.restoreOverrideCursor()
        self.rep_window.show_results(table, scenarios[0].name, n_runs, n_steps, target, seed)
        self.rep_window.show()
        self.rep_window.raise_()

    # ----------------------------------------------------------------- shutdown
    def closeEvent(self, event):
        self.rep_window.close()                   # the only remaining separate window
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
        df, corr, model, metrics = load_and_fit()
    except Exception as exc:        # missing file, missing columns, fit failure
        QMessageBox.critical(None, "Cannot load dataset", f"{CSV_PATH}\n\n{exc}")
        sys.exit(1)

    win = MainWindow(df, corr, model, metrics)
    for sc in default_scenarios():            # baseline, heavy overtime, weak management
        win.add_scenario(sc)
    win.list.setCurrentRow(0)
    win.resize(1200, 750)
    win.show()
    QTimer.singleShot(0, win.on_run_all)      # run once so nothing is empty at launch
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
