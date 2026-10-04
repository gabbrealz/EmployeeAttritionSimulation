"""
Employee attrition risk simulation (single agent, replacement-style).

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
CSV_PATH = "dataset.csv"         # columns = the state variable names
N_STEPS = 1000                   # iterations per run (1 iteration = 1 month)
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

# Scenarios = additive shifts of the variables' location (0 = baseline).
# Shifts do NOT propagate: shift burnout explicitly if overtime should raise it.
SCENARIOS = {
    "baseline": {},
    "heavy overtime": {"overtime_hours_per_week": 3.0},
    "weak management": {"manager_support_score": -2.0},
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
def run_simulation(df, corr, model, shifts, seed=SEED):
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
    draws = sample_mc(df, corr, full_shifts, N_STEPS, rng)
    mc_values = draws.to_numpy()                      # columns follow MC_VARS

    intercept = model.params["const"]
    coefs = model.params[FEATURES].to_numpy()         # same order as FEATURES

    initial_years = INITIAL_STATE["years_since_promotion"]
    years = initial_years
    years_log = np.empty(N_STEPS)
    prob_log = np.empty(N_STEPS)
    resigned_log = np.zeros(N_STEPS, dtype=int)

    for t in range(N_STEPS):
        years += 1/12                                 # deterministic update (+1 month)
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
    history.index = pd.RangeIndex(1, N_STEPS + 1, name="month")

    # Outputs (averages are over all iterations of the run)
    summary = {
        "employee_attrition": int(resigned_log.sum()),          # cumulative
        "avg_attrition_probability": prob_log.mean(),
        "avg_employee_burnout": draws["burnout_score"].mean(),
        "avg_employee_engagement": draws["engagement_score"].mean(),
        "avg_hours_overtime": draws["overtime_hours_per_week"].mean(),
        "avg_years_since_promotion": years_log.mean(),
        "avg_manager_support": draws["manager_support_score"].mean(),
        "time_elapsed": N_STEPS,                                # months
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
            
        axes[5].plot(h.index, h["cum_attrition"], label=name, linewidth=1.5)

    for ax, (_, label) in zip(axes, PREDICTOR_PANELS):
        ax.set_ylabel(label)
        ax.set_title(f"{label} ({SMOOTH_WINDOW}-month rolling mean)", fontsize=10)
        
    axes[5].set_ylabel("Cumulative attrition")
    axes[5].set_title("Cumulative attrition (employees who resigned)", fontsize=10)
    axes[5].legend(title="Scenario")

    for ax in axes[-2:]:
        ax.set_xlabel("Month")
    for ax in axes:
        ax.grid(alpha=0.3)

    fig.suptitle("Employee attrition simulation: predictors and attrition by scenario", fontsize=13)
    fig.tight_layout()
    plt.show()


# ----------------------------------------------------------------------------
# Main: fit everything from the dataset, then run each scenario
# ----------------------------------------------------------------------------
if __name__ == "__main__":
    df = pd.read_csv(CSV_PATH)[FEATURES + ["attrition"]].dropna()

    corr = fit_copula_correlation(df)
    model = fit_attrition_model(df)

    summaries, histories = {}, {}
    for name, shifts in SCENARIOS.items():
        summaries[name], histories[name] = run_simulation(df, corr, model, shifts)

    table = pd.DataFrame.from_dict(summaries, orient="index").round(3).astype(object).T
    print(table.to_string())
    plot_results(histories)
