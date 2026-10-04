import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# 1. Load Dataset from CSV
# Replace "employee_attrition_data.csv" with your actual file path
df = pd.read_csv("employee_attrition_hr_2026.csv")

# Define feature columns and target column
# Update column names if they differ in your CSV file
feature_cols = [
    "engagement_score",
    "manager_support_score",
    "burnout_score",
    "overtime_hours_per_week",
]
target_col = "attrition"


# Drop missing values in the relevant columns (or handle imputation beforehand)
df = df.dropna(subset=feature_cols + [target_col])

X = df[feature_cols]
y = df[target_col].astype(int)

# 2. Build Pipeline (Standardization is CRITICAL for Elastic Net penalties)
pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("model", LogisticRegression(
        l1_ratio=0.5,
        solver="saga",      # "saga" solver supports elasticnet penalty
        max_iter=5000,
        random_state=42
    ))
])

# 3. Define Hyperparameter Grid for Tuning
# Note: In scikit-learn, C = 1/alpha (smaller C = stronger regularization)
param_grid = {
    "model__C": [0.01, 0.1, 1.0, 10.0],
    "model__l1_ratio": [0.0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0] # 0 = Ridge, 1 = Lasso
}

# 4. Perform Grid Search with Stratified 5-Fold Cross-Validation
grid_search = GridSearchCV(
    estimator=pipeline,
    param_grid=param_grid,
    cv=5,
    scoring="roc_auc",
    n_jobs=-1
)

grid_search.fit(X, y)

# 5. Extract Best Model and Parameters
best_pipeline = grid_search.best_estimator_
best_params = grid_search.best_params_

print("--- Best Hyperparameters Found ---")
print(f"Optimal C (1/alpha): {best_params["model__C"]}")
print(f"Optimal l1_ratio: {best_params["model__l1_ratio"]}")
print(f"Best CV ROC-AUC Score: {grid_search.best_score_:.4f}\n")

# 6. Extract Coefficients & Calculate Odds Ratios
fitted_model = best_pipeline.named_steps["model"]
scaled_coefs = fitted_model.coef_[0]

# Standardized Odds Ratios (per 1 Std Dev increase)
std_odds_ratios = np.exp(scaled_coefs)

# Unscale coefficients to get raw scale Odds Ratios (per 1 unit increase on original scale)
scaler = best_pipeline.named_steps["scaler"]
unscaled_coefs = scaled_coefs / scaler.scale_
unscaled_odds_ratios = np.exp(unscaled_coefs)

# Summary DataFrame
results_df = pd.DataFrame({
    "Feature": X.columns,
    "Scaled Coef": scaled_coefs,
    "Unscaled Coef": unscaled_coefs,
    "Odds Ratio (per +1 Unit)": unscaled_odds_ratios,
    "Odds Ratio (per +1 StdDev)": std_odds_ratios
})

print("--- Model Interpretation Table ---")
print(results_df.to_string(index=False))