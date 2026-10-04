"""
Employee attrition risk simulation (single agent, replacement-style).

Each iteration (= 1 year):
  1. Gaussian-copula Monte Carlo draws overtime, burnout, engagement, manager support
  2. years_since_promotion += 1
  3. A logistic regression turns the state into P(attrition)
  4. A Bernoulli draw decides if the employee resigns; if so, count it and
     replace them with a new employee at the initial state

This module has no GUI code: app.py (PyQt6) and plots.py (matplotlib) build on it.
run_scenarios() runs a list of Scenario objects with a shared years/seed.
"""
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import norm
from scipy.special import expit          # logistic function: 1 / (1 + e^-x)

# ----------------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------------
# Resolved relative to this file, so the GUI works from any working directory.
CSV_PATH = Path(__file__).with_name("dataset.csv")   # columns = the state variable names
N_STEPS = 12*10     # 10 years
SEED = 42

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
# Listed variables are PINNED to exactly their shift; unlisted variables are
# free and move by their expected change given the pinned ones, through the
# copula correlation (see propagate_shifts).
SCENARIOS = {
    "baseline": {},
    "heavy overtime": {"overtime_hours_per_week": 3.0},
    "weak management": {"manager_support_score": -2.0},
}


@dataclass
class SimulationResult:
    """What run_simulation returns (and what the UI consumes)."""
    summary: dict               # the 7 spec outputs + turnover_rate, avg_attrition_probability
    history: pd.DataFrame       # one row per year, index named "year" (1..n_steps)
    applied_shifts: dict        # full shifts after propagation (all MC variables)


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


def load_and_fit(csv_path=CSV_PATH):
    """Read the dataset and fit everything that does not change between runs.
    Used by both the CLI (below) and the GUI. Returns (df, corr, model)."""
    df = pd.read_csv(csv_path)[FEATURES + ["attrition"]].dropna()
    return df, fit_copula_correlation(df), fit_attrition_model(df)


# ----------------------------------------------------------------------------
# 4. The simulation loop
# ----------------------------------------------------------------------------
def run_simulation(df, corr, model, shifts, n_steps=N_STEPS, seed=SEED):
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

    years = INITIAL_STATE["years_since_promotion"]
    years_log = np.empty(n_steps)
    p_log = np.empty(n_steps)                         # P(attrition) each year
    resigned_log = np.zeros(n_steps, dtype=int)       # 1 = resigned that year
    employee_attrition = 0

    for t in range(n_steps):
        years += 1/12                                 # deterministic update
        years_log[t] = years                          # state the model sees

        x = np.append(mc_values[t], years)            # full state, FEATURES order
        p = expit(intercept + coefs @ x)              # P(attrition)
        p_log[t] = p

        if rng.random() < p:                          # random experiment
            resigned_log[t] = 1
            employee_attrition += 1
            years = INITIAL_STATE["years_since_promotion"]   # new employee

    # Per-year history for plotting (index = year 1..n_steps)
    history = draws.copy()
    history["years_since_promotion"] = years_log
    history["p_attrition"] = p_log
    history["resigned"] = resigned_log
    history.index = pd.RangeIndex(1, n_steps + 1, name="year")

    # Outputs (averages are over all iterations of the run)
    summary = {
        "employee_attrition": employee_attrition,
        "avg_employee_burnout": draws["burnout_score"].mean(),
        "avg_employee_engagement": draws["engagement_score"].mean(),
        "avg_hours_overtime": draws["overtime_hours_per_week"].mean(),
        "avg_years_since_promotion": years_log.mean(),
        "avg_manager_support": draws["manager_support_score"].mean(),
        "time_elapsed": n_steps,
        # extensions (not in the original spec's output list)
        "turnover_rate": employee_attrition / n_steps,
        "avg_attrition_probability": p_log.mean(),
    }
    return SimulationResult(summary, history, {k: float(v) for k, v in full_shifts.items()})


# ----------------------------------------------------------------------------
# 5. Scenario layer (several scenarios, one shared years/seed)
# ----------------------------------------------------------------------------
@dataclass
class Scenario:
    name: str
    shifts: dict = field(default_factory=dict)   # variable -> shift; only PINNED variables appear


def default_scenarios():
    """Fresh, independent copies of the presets, so the UI can edit them
    without ever modifying the module-level SCENARIOS dict."""
    return [Scenario(name, dict(shifts)) for name, shifts in SCENARIOS.items()]


def run_scenarios(df, corr, model, scenarios, n_steps=N_STEPS, seed=SEED):
    """Run every scenario with the same years and seed.

    Each run_simulation call seeds its own RNG, so all scenarios see the same
    underlying random draws (common random numbers): differences between them
    are due to the scenario, and results do not depend on scenario order.

    Returns (results, errors): name -> SimulationResult, and name -> message
    for scenarios that failed. One failure does not stop the others."""
    names = [s.name for s in scenarios]
    if len(set(names)) != len(names):
        raise ValueError("Scenario names must be unique")
    results, errors = {}, {}
    for sc in scenarios:
        try:
            results[sc.name] = run_simulation(df, corr, model, sc.shifts,
                                              n_steps=n_steps, seed=seed)
        except (ValueError, np.linalg.LinAlgError) as exc:
            errors[sc.name] = str(exc)
    return results, errors


# ----------------------------------------------------------------------------
# Main: fit everything from the dataset, then run each scenario
# ----------------------------------------------------------------------------
if __name__ == "__main__":
    df, corr, model = load_and_fit()
    print(model.summary(), "\n")

    results, errors = run_scenarios(df, corr, model, default_scenarios())
    print(pd.DataFrame({name: r.summary for name, r in results.items()}).T.round(2))
    for name, msg in errors.items():
        print(f"{name}: FAILED ({msg})")
