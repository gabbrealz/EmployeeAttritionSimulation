"""
Evaluate correlation structure and tail dependence among copula variables.

This script loads dataset.csv and performs three evaluations:
1. Linear vs. Rank Correlations (Pearson vs. Spearman & Kendall's Tau)
2. Empirical Tail Dependence Coefficients (Lower at q=0.10, Upper at q=0.90)
3. Copula Fit & Model Selection (Gaussian Copula vs. Student's t-Copula via Maximum Likelihood)
"""

import numpy as np
import pandas as pd
from itertools import combinations
from scipy.stats import norm, t, multivariate_t
from scipy.optimize import minimize_scalar

CSV_PATH = "dataset.csv"
MC_VARS = [
    "overtime_hours_per_week",
    "burnout_score",
    "engagement_score",
    "manager_support_score",
]


def load_data(filepath=CSV_PATH):
    df = pd.read_csv(filepath)[MC_VARS].dropna()
    return df


def analyze_correlations(df):
    pearson = df.corr(method="pearson")
    spearman = df.corr(method="spearman")
    kendall = df.corr(method="kendall")
    return pearson, spearman, kendall


def empirical_tail_dependence(df, q_lower=0.10, q_upper=0.90):
    u = df[MC_VARS].rank() / (len(df) + 1)
    
    lower_td = pd.DataFrame(np.eye(len(MC_VARS)), index=MC_VARS, columns=MC_VARS)
    upper_td = pd.DataFrame(np.eye(len(MC_VARS)), index=MC_VARS, columns=MC_VARS)
    
    for v1, v2 in combinations(MC_VARS, 2):
        u1, u2 = u[v1].values, u[v2].values
        
        # Lower tail dependence: P(U2 <= q_lower | U1 <= q_lower)
        lower_cond = u1 <= q_lower
        l_lambda = np.mean((u1 <= q_lower) & (u2 <= q_lower)) / np.mean(lower_cond) if np.sum(lower_cond) > 0 else 0.0
        
        # Upper tail dependence: P(U2 >= q_upper | U1 >= q_upper)
        upper_cond = u1 >= q_upper
        u_lambda = np.mean((u1 >= q_upper) & (u2 >= q_upper)) / np.mean(upper_cond) if np.sum(upper_cond) > 0 else 0.0
        
        lower_td.loc[v1, v2] = lower_td.loc[v2, v1] = l_lambda
        upper_td.loc[v1, v2] = upper_td.loc[v2, v1] = u_lambda
        
    return lower_td, upper_td


def fit_gaussian_copula(u):
    z = norm.ppf(u.values)
    R = np.corrcoef(z, rowvar=False)
    inv_R = np.linalg.inv(R)
    _, logdet_R = np.linalg.slogdet(R)
    
    quad_term = np.sum(z @ (inv_R - np.eye(len(MC_VARS))) * z, axis=1)
    log_lik = np.sum(-0.5 * logdet_R - 0.5 * quad_term)
    
    k = len(MC_VARS) * (len(MC_VARS) - 1) // 2  # Number of correlation parameters
    aic = 2 * k - 2 * log_lik
    return {"log_lik": log_lik, "aic": aic, "R": R}


def fit_student_t_copula(u):
    z = norm.ppf(u.values)
    R = np.corrcoef(z, rowvar=False)
    
    def neg_log_lik(nu):
        if nu <= 2.0:
            return 1e10
        x = t.ppf(u.values, df=nu)
        log_mvt = multivariate_t.logpdf(x, shape=R, df=nu)
        log_uvt = np.sum(t.logpdf(x, df=nu), axis=1)
        return -np.sum(log_mvt - log_uvt)
        
    res = minimize_scalar(neg_log_lik, bounds=(2.01, 100.0), method="bounded")
    best_nu = res.x
    log_lik = -res.fun
    
    k = (len(MC_VARS) * (len(MC_VARS) - 1) // 2) + 1  # Correlations + degrees of freedom
    aic = 2 * k - 2 * log_lik
    return {"log_lik": log_lik, "aic": aic, "df": best_nu, "R": R}


def main():
    df = load_data()
    # Pseudo-observations on (0, 1) scale
    u = df[MC_VARS].rank() / (len(df) + 1)
    
    pearson, spearman, kendall = analyze_correlations(df)
    lower_td, upper_td = empirical_tail_dependence(df)
    
    gauss_res = fit_gaussian_copula(u)
    t_res = fit_student_t_copula(u)
    
    print("=" * 80)
    print("1. CORRELATION MATRICES (Linear vs Monotone Rank)")
    print("=" * 80)
    print("\n--- Pearson Correlation (Linear) ---")
    print(pearson.round(3))
    print("\n--- Spearman Rank Correlation (Monotone) ---")
    print(spearman.round(3))
    print("\n--- Kendall's Tau Correlation ---")
    print(kendall.round(3))
    
    print("\n" + "=" * 80)
    print("2. EMPIRICAL TAIL DEPENDENCE")
    print("=" * 80)
    print("\n--- Lower Tail Dependence (q = 0.10) ---")
    print(lower_td.round(3))
    print("\n--- Upper Tail Dependence (q = 0.90) ---")
    print(upper_td.round(3))
    
    print("\n" + "=" * 80)
    print("3. COPULA FIT & MODEL COMPARISON")
    print("=" * 80)
    
    comp_df = pd.DataFrame({
        "Model": ["Gaussian Copula", "Student's t Copula"],
        "Log-Likelihood": [gauss_res["log_lik"], t_res["log_lik"]],
        "AIC": [gauss_res["aic"], t_res["aic"]],
        "Degrees of Freedom (nu)": ["Infinity (Fixed)", f"{t_res['df']:.2f}"]
    })
    print(comp_df.to_string(index=False))
    
    print("\n--- DIAGNOSTIC DECISION ---")
    delta_aic = gauss_res["aic"] - t_res["aic"]
    if delta_aic > 2.0 and t_res["df"] < 30.0:
        print("RESULT: Move to Student's t-Copula.")
        print(f"- Student's t improves AIC by {delta_aic:.2f} points over Gaussian copula.")
        print(f"- Estimated degrees of freedom nu = {t_res['df']:.2f} (< 30), indicating heavy joint tail behavior.")
    else:
        print("RESULT: Keep Gaussian Copula.")
        print(f"- AIC difference is negligible ({delta_aic:.2f}) or estimated degrees of freedom nu = {t_res['df']:.2f} is large.")
        print("- The empirical joint distribution does not exhibit heavy tail dependence.")


if __name__ == "__main__":
    main()