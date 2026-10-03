"""
Plotting for the attrition simulation. No Qt imports, and no matplotlib.pyplot:
every function receives a matplotlib Figure, clears it and redraws on it.

Both functions take the `history` DataFrame returned by
simulation.run_simulation (index = year, columns = the 5 predictors,
p_attrition, resigned). If `baseline` (another history) is given, its smoothed
series are drawn as a gray dashed line for comparison.
"""
from simulation import BOUNDS

PREDICTORS = [  # (column, axis label)
    ("overtime_hours_per_week", "Overtime\n(h/week)"),
    ("burnout_score", "Burnout"),
    ("engagement_score", "Engagement"),
    ("manager_support_score", "Manager\nsupport"),
    ("years_since_promotion", "Years since\npromotion"),
]

# Y labels are rotated and the 5 stacked axes are short, so keep them small and
# (where long) two-line; otherwise neighbouring labels overlap in a ~450 px panel.
LABEL_SIZE = 8

RAW_STYLE = dict(lw=0.6, alpha=0.35)       # thin, faint: the per-year noise
SMOOTH_STYLE = dict(lw=1.8)                # thick: the trend
BASE_STYLE = dict(color="0.45", ls="--", lw=1.4, label="baseline")


def _smooth(series, n_steps):
    """Rolling mean over max(1, n_steps // 20) years."""
    return series.rolling(max(1, n_steps // 20), min_periods=1).mean()


def plot_predictors(fig, history, baseline=None):
    """5 stacked axes (shared x): the predictor variables over the simulated years."""
    fig.clear()
    axes = fig.subplots(len(PREDICTORS), 1, sharex=True)
    n = len(history)

    for i, (ax, (col, label)) in enumerate(zip(axes, PREDICTORS)):
        color = f"C{i}"                             # same hue for raw + smoothed line
        ax.plot(history.index, history[col], color=color, **RAW_STYLE)

        # years_since_promotion: raw only. The sawtooth (+1/year, reset after a
        # resignation) is the signal; smoothing it would hide that.
        if col != "years_since_promotion":
            ax.plot(history.index, _smooth(history[col], n), color=color, **SMOOTH_STYLE)

        if baseline is not None:
            ax.plot(baseline.index, _smooth(baseline[col], len(baseline)), **BASE_STYLE)

        # Fixed y-limits (from BOUNDS) so two scenarios are visually comparable;
        # None = leave that side on autoscale (overtime top, years top).
        lo, hi = BOUNDS.get(col, (0, None))
        ax.set_ylim(bottom=lo, top=hi)
        ax.set_ylabel(label, fontsize=LABEL_SIZE)
        ax.grid(True, alpha=0.25)

    if baseline is not None:
        axes[0].legend(loc="upper right", fontsize="small", frameon=False)
    axes[-1].set_xlabel("Year")
    fig.suptitle("Predictor variables")


def plot_attrition(fig, history, baseline=None):
    """2 stacked axes (shared x): cumulative resignations and yearly P(attrition)."""
    fig.clear()
    ax_cum, ax_p = fig.subplots(2, 1, sharex=True)
    n = len(history)

    # Top: cumulative resignations. The slope is the turnover rate.
    ax_cum.step(history.index, history["resigned"].cumsum(), where="post", lw=1.6,
                label="scenario")
    if baseline is not None:
        ax_cum.step(baseline.index, baseline["resigned"].cumsum(), where="post",
                    **BASE_STYLE)
        ax_cum.legend(loc="upper left", fontsize="small", frameon=False)
    ax_cum.set_ylim(bottom=0)
    ax_cum.set_ylabel("Cumulative\nresignations", fontsize=LABEL_SIZE)
    ax_cum.grid(True, alpha=0.25)

    # Bottom: P(attrition) per year (faint), its rolling mean (thick), and a
    # red tick on the x-axis at every year in which the employee resigned.
    color = "C1"
    ax_p.plot(history.index, history["p_attrition"], color=color, **RAW_STYLE)
    ax_p.plot(history.index, _smooth(history["p_attrition"], n), color=color, **SMOOTH_STYLE)
    if baseline is not None:
        ax_p.plot(baseline.index, _smooth(baseline["p_attrition"], len(baseline)),
                  **BASE_STYLE)

    left = history.index[history["resigned"] == 1]
    ax_p.plot(left, [0] * len(left), ls="none", marker="|", ms=9, color="red",
              transform=ax_p.get_xaxis_transform(), clip_on=False)
    ax_p.set_ylim(bottom=0)
    ax_p.set_ylabel("P(attrition)", fontsize=LABEL_SIZE)
    ax_p.set_xlabel("Year")
    ax_p.grid(True, alpha=0.25)

    fig.suptitle("Attrition")
