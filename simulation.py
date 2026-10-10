"""
Employee attrition risk simulation (single agent, replacement-style).

Each iteration (= 1 month):
  1. Gaussian-copula Monte Carlo draws overtime, burnout, engagement, manager support
  2. years_since_promotion += 1/12
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
from scipy.stats import norm
from scipy.special import expit          # logistic function: 1 / (1 + e^-x)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score, roc_auc_score,
)
from sklearn.model_selection import train_test_split

# ----------------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------------
# Resolved relative to this file, so the GUI works from any working directory.
CSV_PATH = Path(__file__).with_name("dataset.csv")   # columns = the state variable names
N_STEPS = 12*10     # 10 years
SEED = 42
TEST_SIZE = 0.2     # share of the dataset held out to evaluate the attrition model

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
    "reduced overtime": {"overtime_hours_per_week": -3.0},
    "stronger management": {"manager_support_score": 2.0},
    "heavy overtime": {"overtime_hours_per_week": 3.0},
    "weak management": {"manager_support_score": -2.0},
}


@dataclass
class SimulationResult:
    """What run_simulation returns (and what the UI consumes)."""
    summary: dict               # the 7 spec outputs + turnover_rate, avg_attrition_probability
    history: pd.DataFrame       # one row per month, index named "month" (1..n_steps)
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
# 3. Predictive attrition model (scikit-learn LogisticRegression)
# ----------------------------------------------------------------------------
def _new_model():
    """An unfitted logistic regression that matches the plain maximum-likelihood fit.

    sklearn's default (C=1, L2 penalty) shrinks the coefficients a little (more on
    small datasets), so the model would no longer be the unpenalised fit this
    simulation was built on. C=1e10 switches the penalty off for practical purposes
    (np.inf would do the same, but sklearn 1.8 prints a spurious warning for it).
    tol=1e-8 makes the solver converge to the exact optimum; the default 1e-4 stops
    short of it on these unscaled features."""
    return LogisticRegression(C=1e10, tol=1e-8, max_iter=1000)


def fit_attrition_model(df, test_size=TEST_SIZE, seed=SEED):
    X, y = df[FEATURES], df["attrition"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=seed, stratify=y)

    held_out = _new_model().fit(X_train, y_train)
    y_pred = held_out.predict(X_test)                        # class labels
    y_prob = held_out.predict_proba(X_test)[:, 1]            # P(attrition = 1)
    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),   # 0 if no one is predicted to resign
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1": f1_score(y_test, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_test, y_prob),
    }
    metrics = {name: float(value) for name, value in metrics.items()}   # plain floats, not numpy scalars

    model = _new_model().fit(X, y)
    return model, metrics


def load_and_fit(csv_path=CSV_PATH):
    """Read the dataset and fit everything that does not change between runs.
    Used by both the CLI (below) and the GUI. Returns (df, corr, model, metrics)."""
    df = pd.read_csv(csv_path)[FEATURES + ["attrition"]].dropna()
    model, metrics = fit_attrition_model(df)
    return df, fit_copula_correlation(df), model, metrics


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

    intercept = model.intercept_[0]
    coefs = model.coef_[0]                            # same order as FEATURES

    years = INITIAL_STATE["years_since_promotion"]
    years_log = np.empty(n_steps)
    p_log = np.empty(n_steps)                         # P(attrition) each month
    resigned_log = np.zeros(n_steps, dtype=int)       # 1 = resigned that month
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

    # Per-month history for plotting (index = month 1..n_steps)
    history = draws.copy()
    history["years_since_promotion"] = years_log
    history["p_attrition"] = p_log
    history["resigned"] = resigned_log
    history.index = pd.RangeIndex(1, n_steps + 1, name="month")

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
# 5. Scenario layer (several scenarios, one shared n_steps/seed)
# ----------------------------------------------------------------------------
@dataclass
class Scenario:
    name: str
    shifts: dict = field(default_factory=dict)   # variable -> shift; only PINNED variables appear


def default_scenarios():
    """Fresh, independent copies of the presets, so the UI can edit them
    without ever modifying the module-level SCENARIOS dict."""
    return [Scenario(name, dict(shifts)) for name, shifts in SCENARIOS.items()]


def fresh_seed():
    """A new random seed that fits in a QSpinBox (31 bits), so it can be shown and typed back."""
    return int(np.random.default_rng().integers(0, 2**31 - 1))


def run_scenarios(df, corr, model, scenarios, n_steps=N_STEPS, seed=SEED):
    """Run every scenario with the same months and seed.

    Each run_simulation call seeds its own RNG, so all scenarios see the same
    underlying random draws (common random numbers): differences between them
    are due to the scenario, and results do not depend on scenario order.

    seed=None draws one fresh random seed for the whole batch, so the shared-draws
    guarantee also holds for unseeded runs.

    Returns (results, errors): name -> SimulationResult, and name -> message
    for scenarios that failed. One failure does not stop the others."""
    names = [s.name for s in scenarios]
    if len(set(names)) != len(names):
        raise ValueError("Scenario names must be unique")
    if seed is None:
        seed = fresh_seed()      # ONE seed for the whole batch, not one per scenario
    results, errors = {}, {}
    for sc in scenarios:
        try:
            results[sc.name] = run_simulation(df, corr, model, sc.shifts,
                                              n_steps=n_steps, seed=seed)
        except (ValueError, np.linalg.LinAlgError) as exc:
            errors[sc.name] = str(exc)
    return results, errors


# ----------------------------------------------------------------------------
# 6. Replications (many runs per scenario, for the 100-run comparison)
# ----------------------------------------------------------------------------
def run_replications(df, corr, model, scenarios, n_runs=100, n_steps=N_STEPS,
                     base_seed=SEED, baseline=None, target=10.0):
    """Run every scenario n_runs times and compare each with the baseline.

    Run r uses seed base_seed + r for ALL scenarios (same random draws within a
    run, different draws between runs), so differences are due to the scenario.
    baseline = name of the reference scenario (default: the first one).

    Returns (runs, table):
      runs  = one row per (run, scenario) with the two attrition measures
      table = one row per scenario: mean, sd, 95% CI of the mean, and the
              % reduction of the mean versus the baseline."""
    names = [s.name for s in scenarios]
    baseline = baseline or names[0]

    rows = []
    for r in range(n_runs):
        results, errors = run_scenarios(df, corr, model, scenarios,
                                        n_steps=n_steps, seed=base_seed + r)
        if errors:
            raise ValueError(f"Run {r} failed: {errors}")
        for name, res in results.items():
            rows.append({
                "run": r,
                "scenario": name,
                "employee_attrition": res.summary["employee_attrition"],
                "avg_attrition_probability": res.summary["avg_attrition_probability"],
            })
    runs = pd.DataFrame(rows)

    table = {}
    for metric in ["employee_attrition", "avg_attrition_probability"]:
        g = runs.groupby("scenario")[metric]
        mean = g.mean().reindex(names)
        sd = g.std().reindex(names)
        table[(metric, "mean")] = mean
        table[(metric, "sd")] = sd
        table[(metric, "ci95")] = 1.96 * sd / np.sqrt(n_runs)
        table[(metric, "reduction_%")] = (1 - mean / mean[baseline]) * 100
    table = pd.DataFrame(table)

    # Does the scenario meet the target on the main measure?
    table[("meets_target", f">={target:g}%")] = (
        table[("employee_attrition", "reduction_%")] >= target)
    return runs, table

# ----------------------------------------------------------------------------
# Main: fit everything from the dataset, then run each scenario
# ----------------------------------------------------------------------------
if __name__ == "__main__":
    df, corr, model, metrics = load_and_fit()
    print(f"Attrition model, hold-out metrics ({1 - TEST_SIZE:.0%} train / {TEST_SIZE:.0%} test):")
    print(pd.Series(metrics).round(3).to_string(), "\n")

    results, errors = run_scenarios(df, corr, model, default_scenarios())
    print(pd.DataFrame({name: r.summary for name, r in results.items()}).T.round(2))
    for name, msg in errors.items():
        print(f"{name}: FAILED ({msg})")

    runs, table = run_replications(df, corr, model, default_scenarios(), n_runs=100)
    print(table.round(3).to_string())
