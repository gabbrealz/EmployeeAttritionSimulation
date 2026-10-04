import pandas as pd
import statsmodels.api as sm

df = pd.read_csv("employee_attrition_hr_2026.csv")

# 1. Define predictors (X) and binary target (y)

# X = df[["engagement_score", "manager_support_score", "work_life_balance_score"]]
X = df[["burnout_score", "overtime_hours_per_week"]]

y = df["attrition"].astype(int)

# 2. Add an explicit intercept term (beta_0)
X_with_const = sm.add_constant(X)

# 3. Fit the model using Maximum Likelihood Estimation (MLE)
logit_model = sm.Logit(y, X_with_const)
results = logit_model.fit()

# 4. View detailed summary (p-values, z-scores, confidence intervals)
print(results.summary())