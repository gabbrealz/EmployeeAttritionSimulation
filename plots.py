import textwrap

from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator

from simulation import BOUNDS

PREDICTORS = [  # (column, axis label)
    ("overtime_hours_per_week", "Overtime hours\nper week"),
    ("burnout_score", "Burnout\nscore"),
    ("engagement_score", "Engagement\nscore"),
    ("manager_support_score", "Manager support\nscore"),
    ("years_since_promotion", "Years since\npromotion"),
]

# Y labels are rotated and the 5 stacked axes are short, so keep them small and
# (where long) two-line; otherwise neighbouring labels overlap.
LABEL_SIZE = 8

SCATTER_PAIRS = [  # (x column, y column), in subplot order: row by row
    ("overtime_hours_per_week", "burnout_score"),
    ("manager_support_score",   "engagement_score"),
    ("burnout_score",           "engagement_score"),
    ("overtime_hours_per_week", "manager_support_score"),
]
# column -> one-line axis label (the predictor plot's labels are two-line; scatter axes are roomy)
LABEL = {col: label.replace("\n", " ") for col, label in PREDICTORS}
STAYED_COLOR, RESIGNED_COLOR = "0.55", "crimson"   # gray vs red: avoids the first scenario colors
MAX_STAYED = 5000                   # display cap on "stayed" points (resignations are never thinned)

RAW_STYLE = dict(lw=0.6, alpha=0.35)       # thin, faint: the per-month noise
SMOOTH_STYLE = dict(lw=1)                # thick: the trend


def _colors(histories):
    """name -> matplotlib color. Same scenario, same color, in every figure."""
    return {name: f"C{i % 10}" for i, name in enumerate(histories)}   # repeats after 10


def _smooth(series, n_steps):
    """Rolling mean over max(1, n_steps // 20) months."""
    return series.rolling(max(1, n_steps // 20), min_periods=1).mean()


def _legend(ax, histories, **kwargs):
    opts = dict(ncol=min(len(histories), 4), fontsize="small", frameon=False)
    opts.update(kwargs)
    ax.legend(**opts)


def plot_predictors(fig, histories):
    """5 stacked axes (shared x): the predictor variables over the simulated months."""
    fig.clear()
    if not histories:
        return
    colors = _colors(histories)
    single = len(histories) == 1
    axes = fig.subplots(len(PREDICTORS), 1, sharex=True)

    for ax, (col, label) in zip(axes, PREDICTORS):
        for name, h in histories.items():
            if col == "years_since_promotion":
                # Always plot raw line only (never smoothed)
                ax.plot(h.index, h[col], color=colors[name], label=name, **SMOOTH_STYLE)
            elif single:
                # If there's one scenario only, plot the raw line
                ax.plot(h.index, h[col], color=colors[name], **RAW_STYLE)
            else:
                ax.plot(h.index, _smooth(h[col], len(h)), color=colors[name], label=name, **SMOOTH_STYLE)

        # Fixed y-limits (from BOUNDS) so runs stay visually comparable;
        # None = leave that side on autoscale (overtime top, years top).
        lo, hi = BOUNDS.get(col, (0, None))
        ax.set_ylim(bottom=lo, top=hi)
        ax.set_ylabel(label, fontsize=LABEL_SIZE)
        ax.grid(True, alpha=0.25)

    _legend(axes[0], histories)
    axes[-1].set_xlabel("Month")
    fig.suptitle(f"Predictor variables ({len(histories)} scenario/s)")


def _bar_panel(ax, histories, colors, values, title, ylabel, fmt, integer=False):
    """One bar per scenario, in the scenario's color, with the value written on top.
    `values` maps scenario name -> number, in the same order as `histories`."""
    names = list(histories)
    bars = ax.bar(range(len(names)), [values[n] for n in names], width=0.6,
                  color=[colors[n] for n in names])
    ax.bar_label(bars, labels=[fmt.format(values[n]) for n in names],
                 padding=2, fontsize="x-small")

    # Long names ("stronger management") are wrapped onto two lines; with many
    # scenarios they are rotated as well, so neighbouring labels never overlap.
    many = len(names) > 6
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels([textwrap.fill(n, 12) for n in names], fontsize="x-small",
                       rotation=30 if many else 0, ha="right" if many else "center")
    ax.set_xlim(-0.6, len(names) - 0.4)       # symmetric; a lone bar is not stretched wide
    ax.set_ylim(0, max(values.values()) * 1.15 or 1.0)   # headroom for the value labels
    if integer:
        ax.yaxis.set_major_locator(MaxNLocator(integer=True))   # counts: no 2.5 resignations
    ax.set_title(title, fontsize="medium")
    ax.set_ylabel(ylabel, fontsize=LABEL_SIZE)
    ax.grid(True, axis="y", alpha=0.25)
    ax.set_axisbelow(True)                    # grid lines behind the bars


def plot_attrition(fig, histories):
    """2x2 grid.
    Left column (shared x): cumulative resignations and monthly P(attrition) over time.
    Right column: total resignations per scenario and mean P(attrition) per scenario."""
    fig.clear()
    if not histories:
        return
    colors = _colors(histories)
    single = len(histories) == 1

    gs = fig.add_gridspec(2, 2, width_ratios=[3, 2])
    ax_cum = fig.add_subplot(gs[0, 0])
    ax_p = fig.add_subplot(gs[1, 0], sharex=ax_cum)
    ax_total = fig.add_subplot(gs[0, 1])
    ax_avg = fig.add_subplot(gs[1, 1])

    for name, h in histories.items():
        c = colors[name]
        # Top: cumulative resignations. The slope is the turnover rate, so the
        # slopes of different scenarios compare turnover directly.
        ax_cum.plot(h.index, h["resigned"].cumsum(), drawstyle="steps-post",
                    lw=1.6, color=c, label=name)

        # Bottom: rolling mean of P(attrition); one scenario also gets the raw
        # monthly probability (faint) and a red tick at every resignation month.
        if single:
            ax_p.plot(h.index, h["p_attrition"], color=c, **RAW_STYLE)
        ax_p.plot(h.index, _smooth(h["p_attrition"], len(h)), color=c,
                  label=name, **SMOOTH_STYLE)
        if single:
            left = h.index[h["resigned"] == 1]
            ax_p.plot(left, [0] * len(left), ls="none", marker="|", ms=9, color="red",
                      transform=ax_p.get_xaxis_transform(), clip_on=False)

    ax_cum.set_ylim(0, max(1, ax_cum.get_ylim()[1]))     # a run with no resignations still gets a 0-1 axis
    ax_cum.yaxis.set_major_locator(MaxNLocator(integer=True))
    ax_cum.set_ylabel("Cumulative attrition", fontsize=LABEL_SIZE)
    ax_cum.grid(True, alpha=0.25)
    _legend(ax_cum, histories, ncol=1 if len(histories) <= 5 else 2)

    ax_p.set_ylim(bottom=0)
    ax_p.set_ylabel("Attrition probability", fontsize=LABEL_SIZE)
    ax_p.set_xlabel("Month")
    ax_p.grid(True, alpha=0.25)

    # Right column: one number per scenario, over the WHOLE run. These are the same
    # figures as "Employee attrition" and "Avg. attrition probability" in the results table.
    total = {n: int(h["resigned"].sum()) for n, h in histories.items()}
    avg_p = {n: float(h["p_attrition"].mean()) for n, h in histories.items()}
    _bar_panel(ax_total, histories, colors, total,
               "Total attrition per scenario", "Attrition", "{:d}", integer=True)
    _bar_panel(ax_avg, histories, colors, avg_p,
               "Average attrition probability per scenario", "Avg. attrition probability", "{:.3f}")

    fig.suptitle(f"Attrition ({len(histories)} scenario/s)")


def _axis_limits(col, histories):
    """Same limits for every scenario, so switching scenarios shows the cloud MOVING
    instead of the axes rescaling: BOUNDS where defined, else the data range across ALL
    scenarios. A small pad keeps points clipped at a bound visible."""
    lo, hi = BOUNDS.get(col, (None, None))
    if lo is None:
        lo = min(h[col].min() for h in histories.values())
    if hi is None:
        hi = max(h[col].max() for h in histories.values())
    pad = 0.03 * ((hi - lo) or 1.0)
    return lo - pad, hi + pad


def plot_scatter(fig, name, histories):
    """2x2 scatterplots of the simulated states of ONE scenario. Each point = one month.
    Stayed = small, faint gray; resigned = larger, opaque red, drawn last (on top)."""
    fig.clear()
    if name not in histories:
        return
    h = histories[name]
    resigned = h["resigned"] == 1
    left, stayed = h[resigned], h[~resigned]
    n_left, n_stayed = len(left), len(stayed)                # true counts, for the legend
    if n_stayed > MAX_STAYED:
        stayed = stayed.sample(MAX_STAYED, random_state=0)   # display only; all resignations kept

    axes = fig.subplots(2, 2).ravel()                        # row-major = order of SCATTER_PAIRS
    for ax, (xc, yc) in zip(axes, SCATTER_PAIRS):
        ax.scatter(stayed[xc], stayed[yc], s=10, color=STAYED_COLOR, alpha=0.6, linewidths=0)
        ax.scatter(left[xc], left[yc], s=20, color=RESIGNED_COLOR, alpha=0.85, linewidths=0)  # last = on top
        ax.set_xlabel(LABEL[xc])
        ax.set_ylabel(LABEL[yc])
        ax.set_xlim(*_axis_limits(xc, histories))
        ax.set_ylim(*_axis_limits(yc, histories))
        ax.grid(True, alpha=0.25)

    # Proxy handles, so the "stayed" legend dot isn't invisibly faint
    handles = [
        Line2D([], [], marker="o", ls="", color=STAYED_COLOR, alpha=0.6, label=f"Stayed ({n_stayed})"),
        Line2D([], [], marker="o", ls="", color=RESIGNED_COLOR, label=f"Resigned ({n_left})"),
    ]
    try:
        fig.legend(handles=handles, loc="outside lower center", ncol=2)
    except ValueError:                                       # matplotlib < 3.7 has no "outside" locations
        axes[0].legend(handles=handles, loc="best")
    fig.suptitle(f"{name}: {n_left} resignations in {len(h)} months")