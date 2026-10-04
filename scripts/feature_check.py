"""
Feature check for the attrition logistic regression (simulation.py).

Answers: which of the 5 features carry real signal, and is a plain
(unpenalized) logistic regression well calibrated?

Run:  python feature_check.py [path/to/data.csv]

Needs only numpy, pandas, scipy, scikit-learn.
Unpenalized fits use C=np.inf (works on old and new scikit-learn versions).
"""
import sys
import warnings

import numpy as np
import pandas as pd
from scipy.stats import chi2, norm
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold
from sklearn.preprocessing import StandardScaler

# scikit-learn >= 1.8 warns that C=np.inf means "no penalty"; that is what we want here.
warnings.filterwarnings("ignore", message="Setting penalty=None")

CSV = sys.argv[1] if len(sys.argv) > 1 else "employee_attrition_hr_2026.csv"
TARGET = "attrition"
FEATURES = [
    "overtime_hours_per_week",
    "burnout_score",
    "engagement_score",
    "manager_support_score",
    "years_since_promotion",   # missing from your Elastic Net script, but used by simulation.py
    "perceived_ai_job_risk"
]

df = pd.read_csv(CSV)[FEATURES + [TARGET]].dropna().reset_index(drop=True)
y = df[TARGET].astype(int).to_numpy()
n, n_events = len(df), int(y.sum())
print(f"Rows: {n}, events: {n_events} ({n_events / n:.1%}), "
      f"events per variable: {n_events / len(FEATURES):.0f}\n")


def fit(cols, idx=None):
    """Unpenalized logistic regression on standardized columns. Returns (scaler, model)."""
    idx = np.arange(n) if idx is None else idx
    sc = StandardScaler().fit(df.loc[idx, cols])
    m = LogisticRegression(C=np.inf, tol=1e-10, max_iter=2000)
    m.fit(sc.transform(df.loc[idx, cols]), y[idx])
    return sc, m


def predict(sc, m, cols, idx):
    return m.predict_proba(sc.transform(df.loc[idx, cols]))[:, 1]


def loglik(cols):
    sc, m = fit(cols)
    return -log_loss(y, predict(sc, m, cols, np.arange(n)), normalize=False)


# ---------------------------------------------------------------------------
# 1. Full model: odds ratios per +1 SD, with Wald confidence intervals
# ---------------------------------------------------------------------------
sc, m = fit(FEATURES)
Z = np.column_stack([np.ones(n), sc.transform(df[FEATURES])])
p = predict(sc, m, FEATURES, np.arange(n))
cov = np.linalg.inv(Z.T @ (Z * (p * (1 - p))[:, None]))     # inverse Fisher information
beta = np.r_[m.intercept_, m.coef_[0]]
se = np.sqrt(np.diag(cov))
zval = beta / se
full = pd.DataFrame({
    "coef_per_SD": beta[1:],
    "std_err": se[1:],
    "p_value": 2 * norm.sf(abs(zval[1:])),
    "OR_per_SD": np.exp(beta[1:]),
    "OR_CI_low": np.exp(beta[1:] - 1.96 * se[1:]),
    "OR_CI_high": np.exp(beta[1:] + 1.96 * se[1:]),
}, index=FEATURES)
print("=== 1. Full model (standardized features) ===")
print(full.round(4).to_string(), "\n")

# ---------------------------------------------------------------------------
# 2. Likelihood-ratio test + AIC/BIC, dropping one feature at a time
# ---------------------------------------------------------------------------
def aic(ll, k):
    return 2 * k - 2 * ll


def bic(ll, k):
    return k * np.log(n) - 2 * ll


ll_full = loglik(FEATURES)
k_full = len(FEATURES) + 1
rows = []
for f in FEATURES:
    cols = [c for c in FEATURES if c != f]
    ll_red = loglik(cols)
    lr = 2 * (ll_full - ll_red)
    rows.append({
        "dropped": f,
        "LR_stat": lr,
        "p_value": chi2.sf(lr, 1),
        "dAIC": aic(ll_red, k_full - 1) - aic(ll_full, k_full),
        "dBIC": bic(ll_red, k_full - 1) - bic(ll_full, k_full),
    })
print("=== 2. Likelihood-ratio tests (dAIC/dBIC > 0 means dropping the feature makes the model worse) ===")
print(pd.DataFrame(rows).set_index("dropped").round(4).to_string(), "\n")

# ---------------------------------------------------------------------------
# 3. Drop-one cross-validation (repeated, stratified) on log loss and AUC
# ---------------------------------------------------------------------------
splits = list(RepeatedStratifiedKFold(n_splits=5, n_repeats=10, random_state=42).split(df, y))


def cv_scores(cols):
    ll, auc = [], []
    for tr, te in splits:
        s, mod = fit(cols, tr)
        pr = predict(s, mod, cols, te)
        ll.append(log_loss(y[te], pr))
        auc.append(roc_auc_score(y[te], pr))
    return np.array(ll), np.array(auc)


ll_full_cv, auc_full_cv = cv_scores(FEATURES)
rows = []
for f in FEATURES:
    ll_r, auc_r = cv_scores([c for c in FEATURES if c != f])
    rows.append({
        "dropped": f,
        "d_logloss": (ll_r - ll_full_cv).mean(),      # > 0: feature improves held-out log loss
        "folds_where_helps": (ll_r > ll_full_cv).mean(),
        "d_AUC": (auc_full_cv - auc_r).mean(),        # > 0: feature improves held-out AUC
    })
print(f"=== 3. Drop-one CV (full model: log loss {ll_full_cv.mean():.4f}, AUC {auc_full_cv.mean():.4f}) ===")
print(pd.DataFrame(rows).set_index("dropped").round(5).to_string(), "\n")

# ---------------------------------------------------------------------------
# 4. Calibration of the full model with out-of-fold predictions (deciles)
# ---------------------------------------------------------------------------
oof = np.zeros(n)
for tr, te in StratifiedKFold(5, shuffle=True, random_state=42).split(df, y):
    s, mod = fit(FEATURES, tr)
    oof[te] = predict(s, mod, FEATURES, te)
cal = (pd.DataFrame({"pred": oof, "obs": y})
       .assign(decile=lambda d: pd.qcut(d["pred"], 10, labels=False, duplicates="drop") + 1)
       .groupby("decile").agg(mean_predicted=("pred", "mean"),
                              observed_rate=("obs", "mean"),
                              n=("obs", "size")))
print("=== 4. Calibration by decile (out-of-fold) ===")
print(cal.round(4).to_string())
print(f"\nOverall: mean predicted {oof.mean():.4f} vs observed {y.mean():.4f}")