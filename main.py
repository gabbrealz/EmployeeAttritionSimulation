"""
Employee attrition risk simulation (single agent, replacement-style).

Each iteration (= 1 year):
  1. Gaussian-copula Monte Carlo draws overtime, burnout, engagement, manager support
  2. years_since_promotion += 1
  3. A logistic regression turns the state into P(attrition)
  4. A Bernoulli draw decides if the employee resigns; if so, count it and
     replace them with a new employee at the initial state
"""
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import norm
from scipy.special import expit          # logistic function: 1 / (1 + e^-x)

# ----------------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------------
CSV_PATH = "dataset.csv"   # columns = the state variable names
N_STEPS = 1000                   # iterations per run (1 iteration = 1 year)
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

MC_VARS = ["overtime_hours_per_week", "burnout_score",
           "engagement_score", "manager_support_score"]   # copula variables
FEATURES = MC_VARS + ["years_since_promotion"]            # regression inputs

# Bounds applied AFTER the scenario shift (None = no limit on that side).
# Variables not listed default to the min/max observed in the dataset.
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
    "heavy overtime": {"overtime_hours_per_week": 3.0, "burnout_score": 1.0},
    "weak management": {"manager_support_score": -2.0, "engagement_score": -1.0},
}


# ----------------------------------------------------------------------------
# 1. Fit the Gaussian copula (dependence between the 4 Monte Carlo variables)
# ----------------------------------------------------------------------------
def fit_copula_correlation(df):
    """Correlation matrix of the 'normal scores' of each variable's ranks."""
    u = df[MC_VARS].rank() / (len(df) + 1)     # ranks -> uniform values in (0, 1)
    z = norm.ppf(u)                            # uniform -> standard normal scores
    return np.corrcoef(z, rowvar=False)        # 4x4 correlation matrix


# ----------------------------------------------------------------------------
# 2. Monte Carlo sampler
# ----------------------------------------------------------------------------
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
    unknown = set(shifts) - set(MC_VARS)
    if unknown:
        raise ValueError(f"Shifts given for non-Monte-Carlo variables: {unknown}")

    rng = np.random.default_rng(seed)   # same seed per scenario -> fair comparison

    # Monte Carlo draws don't depend on who resigns, so draw every step up front
    draws = sample_mc(df, corr, shifts, N_STEPS, rng)
    mc_values = draws.to_numpy()                      # columns follow MC_VARS

    intercept = model.params["const"]
    coefs = model.params[FEATURES].to_numpy()         # same order as FEATURES

    years = INITIAL_STATE["years_since_promotion"]
    years_log = np.empty(N_STEPS)
    employee_attrition = 0

    for t in range(N_STEPS):
        years += 1                                    # deterministic update
        years_log[t] = years                          # state the model sees

        x = np.append(mc_values[t], years)            # full state, FEATURES order
        p = expit(intercept + coefs @ x)              # P(attrition)

        if rng.random() < p:                          # random experiment
            employee_attrition += 1
            years = INITIAL_STATE["years_since_promotion"]   # new employee

    # Outputs (averages are over all iterations of the run)
    return {
        "employee_attrition": employee_attrition,
        "avg_employee_burnout": draws["burnout_score"].mean(),
        "avg_employee_engagement": draws["engagement_score"].mean(),
        "avg_hours_overtime": draws["overtime_hours_per_week"].mean(),
        "avg_years_since_promotion": years_log.mean(),
        "avg_manager_support": draws["manager_support_score"].mean(),
        "time_elapsed": N_STEPS,
    }


# ----------------------------------------------------------------------------
# Main: fit everything from the dataset, then run each scenario
# ----------------------------------------------------------------------------
if __name__ == "__main__":
    df = pd.read_csv(CSV_PATH)[FEATURES + ["attrition"]].dropna()

    corr = fit_copula_correlation(df)
    model = fit_attrition_model(df)
    print(model.summary(), "\n")

    results = { name: run_simulation(df, corr, model, shifts) for name, shifts in SCENARIOS.items() }
    print(pd.DataFrame(results).T.round(2))