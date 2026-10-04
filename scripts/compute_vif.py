import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor


def compute_vif(df: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    """Computes Variance Inflation Factor (VIF) for selected features.

    Parameters:
    -----------
    df : pd.DataFrame
        Dataset containing predictor variables.
    feature_cols : list of str
        List of numeric feature column names to evaluate.

    Returns:
    --------
    pd.DataFrame
        DataFrame listing each feature alongside its computed VIF value.
    """
    # Prepare matrix and drop missing values
    X = df[feature_cols].dropna().copy()

    # Add intercept/constant column required for proper VIF calculation
    X_with_const = sm.add_constant(X)

    # Compute VIF for each column
    vif_data = pd.DataFrame(
        {
            "Feature": X_with_const.columns,
            "VIF": [
                variance_inflation_factor(X_with_const.values, i)
                for i in range(X_with_const.shape[1])
            ],
        }
    )

    # Remove the constant term from the output report
    return vif_data[vif_data["Feature"] != "const"].reset_index(drop=True)


if __name__ == "__main__":
    # Project specific features
    FEATURES = [
        "overtime_hours_per_week",
        "burnout_score",
        "engagement_score",
        "manager_support_score",
        "years_since_promotion",
        "perceived_ai_job_risk"
    ]

    df = pd.read_csv("employee_attrition_hr_2026.csv")
    vif_df = compute_vif(df, FEATURES)

    print("Variance Inflation Factors (VIF):")
    print(vif_df.round(2).to_string(index=False))