"""
Employee attrition risk simulation (single agent, replacement-style).

At startup the program asks for a scenario (an additive shift for each Monte
Carlo variable, blank = baseline) and the number of months to simulate. All
inputs are validated, then the run starts.

Each iteration (= 1 month):
  1. Gaussian-copula Monte Carlo draws overtime, burnout, engagement, manager support
  2. years_since_promotion += 1/12
  3. A logistic regression turns the state into P(attrition)
  4. A Bernoulli draw decides if the employee resigns; if so, count it and
     replace them with a new employee at the initial state
"""
import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.special import expit          # logistic function: 1 / (1 + e^-x)

# ----------------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------------
CSV_PATH = "dataset.csv"   # columns = the state variable names
DEFAULT_N_STEPS = 12*10          # months per run if the input is left blank (10 years)
MAX_N_STEPS = 100_000            # upper limit accepted by the input prompt
STEP_YEARS = 1 / 12              # time added to years_since_promotion per iteration
SEED = 42

SMOOTH_WINDOW = 12                   # months, rolling mean applied to the predictor plots

# Initial state of every new employee. Only years_since_promotion is actually
# needed by the loop: the other four are redrawn by Monte Carlo every step.
INITIAL_STATE = {
    "overtime_hours_per_week": 6.83,
    "burnout_score": 4.03,
    "engagement_score": 6.51,
    "manager_support_score": 6.61,
    "years_since_promotion": 2.0,
}

# copula variables
MC_VARS = ["overtime_hours_per_week", "burnout_score", "engagement_score", "manager_support_score"]
# regression inputs (copula variables + years_since_promotion)
FEATURES = MC_VARS + ["years_since_promotion"]

# Bounds applied AFTER the scenario shift (None = no limit on that side).
BOUNDS = {
    "overtime_hours_per_week": (0, None),
    "burnout_score": (0, 10),
    "engagement_score": (0, 10),
    "manager_support_score": (0, 10),
}

# Panels of the figure: (column in the history table, axis label)
PREDICTOR_PANELS = [
    ("overtime_hours_per_week", "Overtime (hours/week)"),
    ("burnout_score", "Burnout score"),
    ("engagement_score", "Engagement score"),
    ("manager_support_score", "Manager support score"),
    ("years_since_promotion", "Years since promotion"),
]


# ----------------------------------------------------------------------------
# 1. Fit the Gaussian copula (dependence between the 4 Monte Carlo variables)
# ----------------------------------------------------------------------------
def fit_copula_correlation(df):
    """Correlation matrix of the 'normal scores' of each variable's ranks."""
    u = df[MC_VARS].rank() / (len(df) + 1)     # ranks -> uniform values (between 0 and 1)
    z = norm.ppf(u)                            # uniform -> standard normal scores
    return np.corrcoef(z, rowvar=False)        # 4x4 correlation matrix


# ----------------------------------------------------------------------------
# 2. Monte Carlo sampler
# ----------------------------------------------------------------------------
def propagate_shifts(df, corr, shifts):
    """Expand user-specified shifts to all MC variables via Gaussian conditioning.
    Listed variables get exactly their shift; the others shift by their
    expected change given the listed ones, using the copula correlation."""

    if not shifts:
        return {v: 0.0 for v in MC_VARS}
    
    sd = df[MC_VARS].std().to_numpy()

    cov = corr * np.outer(sd, sd)                       # correlation -> covariance
    S = [MC_VARS.index(v) for v in shifts]              # shifted variable indices

    delta = np.array(list(shifts.values()))
    full = cov[:, S] @ np.linalg.solve(cov[np.ix_(S, S)], delta)   # Σ_·s Σ_ss⁻¹ Δ_s

    return dict(zip(MC_VARS, full))


def sample_mc(df, corr, shifts, n_steps, rng):
    """Draw n_steps rows of the 4 Monte Carlo variables, with scenario shifts."""
    # Correlated normals -> uniforms: this is the copula step
    z = rng.multivariate_normal(np.zeros(len(MC_VARS)), corr, size=n_steps)
    u = norm.cdf(z)

    draws = {}
    for i, var in enumerate(MC_VARS):
        data = df[var].to_numpy()
        # Marginal = the dataset's own distribution (inverse empirical CDF).
        # "inverted_cdf" keeps discrete scales (<=15 distinct values) as real
        # observed values instead of interpolating between them.
        method = "inverted_cdf" if df[var].nunique() <= 15 else "linear"
        x = np.quantile(data, u[:, i], method=method)

        x = x + shifts.get(var, 0.0)                          # scenario shift
        lo, hi = BOUNDS.get(var, (data.min(), data.max()))    # keep in valid range
        draws[var] = np.clip(x, lo, hi)
        
    return pd.DataFrame(draws)


# ----------------------------------------------------------------------------
# 3. Predictive attrition model (logistic regression from statsmodels)
# ----------------------------------------------------------------------------
def fit_attrition_model(df):
    X = sm.add_constant(df[FEATURES])          # adds the intercept column "const"
    return sm.Logit(df["attrition"], X).fit(disp=0)


# ----------------------------------------------------------------------------
# 4. The simulation loop
# ----------------------------------------------------------------------------
def run_simulation(df, corr, model, shifts, n_steps, seed=SEED):
    """Returns (summary, history):
       summary = dict of the run's output metrics
       history = DataFrame with one row per month (state seen by the model,
                 attrition probability, resignation flag, cumulative attrition)."""
    unknown = set(shifts) - set(MC_VARS)
    if unknown:
        raise ValueError(f"Shifts given for non-Monte-Carlo variables: {unknown}")

    rng = np.random.default_rng(seed)   # same seed per scenario -> fair comparison

    # Monte Carlo draws don't depend on who resigns, so draw every step up front
    full_shifts = propagate_shifts(df, corr, shifts)
    draws = sample_mc(df, corr, full_shifts, n_steps, rng)
    mc_values = draws.to_numpy()                      # columns follow MC_VARS

    intercept = model.params["const"]
    coefs = model.params[FEATURES].to_numpy()         # same order as FEATURES

    initial_years = INITIAL_STATE["years_since_promotion"]
    years = initial_years
    years_log = np.empty(n_steps)
    prob_log = np.empty(n_steps)
    resigned_log = np.zeros(n_steps, dtype=int)

    for t in range(n_steps):
        years += STEP_YEARS                           # deterministic update (+1 month)
        years_log[t] = years                          # state the model sees

        x = np.append(mc_values[t], years)            # full state, FEATURES order
        p = expit(intercept + coefs @ x)              # P(attrition)
        prob_log[t] = p

        if rng.random() < p:                          # random experiment
            resigned_log[t] = 1
            years = initial_years                     # new employee

    history = draws.copy()
    history["years_since_promotion"] = years_log
    history["attrition_prob"] = prob_log
    history["resigned"] = resigned_log
    history["cum_attrition"] = np.cumsum(resigned_log)
    history.index = pd.RangeIndex(1, n_steps + 1, name="month")

    # Outputs (averages are over all iterations of the run)
    summary = {
        "employee_attrition": int(resigned_log.sum()),          # cumulative
        "avg_attrition_probability": prob_log.mean(),
        "avg_employee_burnout": draws["burnout_score"].mean(),
        "avg_employee_engagement": draws["engagement_score"].mean(),
        "avg_hours_overtime": draws["overtime_hours_per_week"].mean(),
        "avg_years_since_promotion": years_log.mean(),
        "avg_manager_support": draws["manager_support_score"].mean(),
        "time_elapsed": n_steps,                                # months
    }
    return summary, history


# ----------------------------------------------------------------------------
# 5. Plots
# ----------------------------------------------------------------------------
def plot_results(histories):
    """One panel per predictor (rolling mean) + cumulative attrition, one line per scenario."""
    fig, axes = plt.subplots(3, 2, figsize=(12, 10), sharex=True)
    axes = axes.ravel()

    for name, h in histories.items():
        for ax, (col, _) in zip(axes, PREDICTOR_PANELS):
            ax.plot(h.index, h[col].rolling(SMOOTH_WINDOW, min_periods=1).mean(), label=name, linewidth=1.3)

    for ax, (_, label) in zip(axes, PREDICTOR_PANELS):
        ax.set_ylabel(label)
        ax.set_title(f"{label} ({SMOOTH_WINDOW}-month rolling mean)", fontsize=10)

    for ax in axes[-2:]:
        ax.set_xlabel("Month")
    for ax in axes:
        ax.grid(alpha=0.3)

    fig.suptitle("Employee attrition simulation: predictors and attrition by scenario", fontsize=13)
    fig.tight_layout()
    return fig


def plot_regression_output(histories):
    """Separate figure for the regression output: cumulative attrition and
    attrition probability (raw monthly values + rolling mean) across time."""
    fig, (ax_cum, ax_prob) = plt.subplots(2, 1, figsize=(10, 7), sharex=True)

    for name, h in histories.items():
        ax_cum.plot(h.index, h["cum_attrition"], label=name, linewidth=1.5)

        line, = ax_prob.plot(h.index, h["attrition_prob"], alpha=0.25, linewidth=0.8)
        ax_prob.plot(h.index, h["attrition_prob"].rolling(SMOOTH_WINDOW, min_periods=1).mean(),
                     color=line.get_color(), linewidth=1.6, label=f"{name} ({SMOOTH_WINDOW}-month mean)")
        ax_prob.axhline(h["attrition_prob"].mean(), color=line.get_color(), linestyle="--", linewidth=1, alpha=0.7)

    ax_cum.set_ylabel("Cumulative attrition")
    ax_cum.set_title("Cumulative attrition (employees who resigned)", fontsize=10)
    ax_cum.legend(title="Scenario")
    ax_prob.set_ylabel("Attrition probability")
    ax_prob.set_title("Attrition probability from the regression (faint = raw, dashed = run average)", fontsize=10)
    ax_prob.set_xlabel("Month")
    ax_prob.legend()
    for ax in (ax_cum, ax_prob):
        ax.grid(alpha=0.3)

    fig.suptitle("Regression output over time", fontsize=13)
    fig.tight_layout()
    return fig


# ----------------------------------------------------------------------------
# 6. Scenario input (asked at startup)
# ----------------------------------------------------------------------------
def parse_shift(raw, var, df):
    """Validate one answer. Returns (shift, error_message); blank = 0.0 (baseline)."""
    raw = raw.strip()
    if raw == "":
        return 0.0, None

    try:
        shift = float(raw)
    except ValueError:
        return None, f"'{raw}' is not a number"
    if not np.isfinite(shift):
        return None, f"'{raw}' is not a finite number"

    # A shift that moves the whole observed range outside the allowed range
    # would clip every draw to one boundary value.
    lo, hi = BOUNDS[var]
    data = df[var]
    if (hi is not None and data.min() + shift > hi) or (lo is not None and data.max() + shift < lo):
        return None, (f"a shift of {shift:g} puts the whole observed range "
                      f"({data.min():g} to {data.max():g}) outside the allowed range")
    return shift, None


def parse_n_steps(raw):
    """Validate the number of months. Returns (n_steps, error_message); blank = default."""
    raw = raw.strip()
    if raw == "":
        return DEFAULT_N_STEPS, None

    try:
        value = float(raw)
    except ValueError:
        return None, f"'{raw}' is not a number"
    if not np.isfinite(value) or value != int(value):
        return None, f"'{raw}' is not a whole number"
    if not 1 <= int(value) <= MAX_N_STEPS:
        return None, f"must be between 1 and {MAX_N_STEPS:,} months"
    
    return int(value), None


def parse_seed(raw):
    """Validate the number of months. Returns (n_steps, error_message); blank = default."""
    raw = raw.strip()
    if raw == "":
        return SEED, None

    try:
        value = float(raw)
    except ValueError:
        return None, f"'{raw}' is not a number"
    if not np.isfinite(value) or value != int(value):
        return None, f"'{raw}' is not a whole number"
    if int(value) < 0:
        return None, f"must not be below 0"
    
    return int(value), None


def ask_inputs(df):
    """Ask one shift per Monte Carlo variable plus the number of months,
    validate all answers after the batch, and re-ask only the invalid ones.
    Returns ({variable: shift} for all MC variables, n_steps).
    Blank shift = 0.0 (baseline); blank months = DEFAULT_N_STEPS."""

    print("\nScenario setup: enter an additive shift for each variable, in its own units")
    print("(e.g. 3 or -2). Leave blank for baseline (no shift).\n")

    values, pending = {}, list(MC_VARS) + ["n_steps", "seed"]
    while pending:
        answers = {}
        for key in pending:
            if key == "n_steps":
                answers[key] = input(f"  number of months to simulate (blank = {DEFAULT_N_STEPS}): ")
            elif key == "seed":
                answers[key] = input(f"  seed number for RNG (blank = {SEED}): ")
            else:
                lo, hi = BOUNDS[key]
                allowed = f"{'-inf' if lo is None else lo} to {'inf' if hi is None else hi}"
                answers[key] = input(f"  {key} shift (values stay within {allowed}): ")

        errors = {}
        for key, raw in answers.items():
            value, error = parse_n_steps(raw) if key == "n_steps" else parse_seed(raw) if key == "seed" else parse_shift(raw, key, df)
            if error:
                errors[key] = error
            else:
                values[key] = value

        if errors:
            print("\nInvalid input:")
            for key, error in errors.items():
                print(f"  - {'number of months' if key == 'n_steps' else key}: {error}")
            print("Please re-enter these values.\n")

        pending = list(errors)

    return {var: values[var] for var in MC_VARS}, values["n_steps"], values["seed"]


# ----------------------------------------------------------------------------
# Main: fit everything from the dataset, ask for the scenario, run the simulation
# ----------------------------------------------------------------------------
if __name__ == "__main__":
    df = pd.read_csv(CSV_PATH)[FEATURES + ["attrition"]].dropna()

    shifts, n_steps, seed = ask_inputs(df)          # all inputs validated before we continue
    active = [f"{var} {s:+g}" for var, s in shifts.items() if s != 0]
    name = "custom" if active else "baseline"

    corr = fit_copula_correlation(df)
    model = fit_attrition_model(df)
    summary, history = run_simulation(df, corr, model, shifts, n_steps, seed=seed)

    print(f"\nScenario: {', '.join(active) or 'baseline'}  |  {n_steps} months\n")
    # metrics as rows (counts stay integers, everything else rounded)
    rounded = {k: v if isinstance(v, int) else round(float(v), 3) for k, v in summary.items()}
    print(pd.Series(rounded, name=name, dtype=object).to_frame().to_string())

    label = "\n".join(active) or "baseline"
    plot_results({label: history})                # figure 1: predictors
    plot_regression_output({label: history})      # figure 2: regression output
    plt.show()                                    # opens both windows
