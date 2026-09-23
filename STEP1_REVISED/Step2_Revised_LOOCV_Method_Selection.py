"""
STEP 2 - REVISED LOOCV METHOD SELECTION

Purpose
-------
Compare candidate mathematical prediction/interpolation methods for the
three revised experimental responses:

    hc_hi
    td_to
    Vch

ONLY the 67 Train_Anchors from Step 1 are used.

The 17 Test_Holdout observations are NOT used anywhere in this step.

Candidate methods
-----------------
1. Linear Griddata
2. Cubic Griddata
3. 1st-order RSM
4. 2nd-order RSM
5. 3rd-order RSM

Validation
----------
Leave-One-Out Cross-Validation (LOOCV) over the 67 training anchors.

For each target and each method:
    - remove one training observation
    - fit the method on the remaining training observations
    - predict the removed observation
    - repeat for all 67 anchors where prediction is possible

Griddata may return NaN for some folds when the held-out point lies outside
the convex hull of the remaining coordinates. Such folds are reported and
NOT silently replaced.

Metrics
-------
R2
RMSE
MAE
MSE
MAPE

Method selection
----------------
No method is selected using R2 alone.

A summary table is produced so that model selection can consider:
    - predictive error
    - R2
    - valid-fold coverage
    - numerical stability

The default "Best_Candidate" flag is based primarily on lowest RMSE among
methods with valid predictions on all 67 folds. If no method has full
coverage, the best-coverage / lowest-RMSE candidate is reported as a
candidate only. Final selection should be reviewed before being locked.
"""

from pathlib import Path
import warnings

import numpy as np
import pandas as pd
from scipy.interpolate import griddata
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    r2_score,
    mean_squared_error,
    mean_absolute_error,
)

warnings.filterwarnings("ignore")


# =============================================================================
# 1. PATHS
# =============================================================================

BASE_DIR = Path(__file__).resolve().parent

INPUT_FILE = BASE_DIR / "Step1_Stratified_Split.xlsx"
OUTPUT_FILE = BASE_DIR / "Step2_LOOCV_Results.xlsx"


# =============================================================================
# 2. CONFIGURATION
# =============================================================================

REQUIRED_COLUMNS = [
    "Point_ID",
    "d",
    "Rpm",
    "hc_hi",
    "td_to",
    "Vch",
]

TARGETS = [
    "hc_hi",
    "td_to",
    "Vch",
]

METHODS = [
    "Linear Griddata",
    "Cubic Griddata",
    "1st-order RSM",
    "2nd-order RSM",
    "3rd-order RSM",
]

GRIDDATA_METHODS = {
    "Linear Griddata": "linear",
    "Cubic Griddata": "cubic",
}

RSM_DEGREES = {
    "1st-order RSM": 1,
    "2nd-order RSM": 2,
    "3rd-order RSM": 3,
}


# =============================================================================
# 3. METRIC FUNCTIONS
# =============================================================================

def calculate_metrics(y_true, y_pred):
    """
    Calculate regression metrics on finite predictions only.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    valid = np.isfinite(y_true) & np.isfinite(y_pred)

    y_true = y_true[valid]
    y_pred = y_pred[valid]

    n = len(y_true)

    if n == 0:
        return {
            "N_Valid": 0,
            "R2": np.nan,
            "RMSE": np.nan,
            "MAE": np.nan,
            "MSE": np.nan,
            "MAPE_percent": np.nan,
        }

    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)

    if n >= 2:
        r2 = r2_score(y_true, y_pred)
    else:
        r2 = np.nan

    nonzero = np.abs(y_true) > 1e-12

    if np.any(nonzero):
        mape = np.mean(
            np.abs(
                (y_true[nonzero] - y_pred[nonzero])
                / y_true[nonzero]
            )
        ) * 100.0
    else:
        mape = np.nan

    return {
        "N_Valid": n,
        "R2": float(r2),
        "RMSE": float(rmse),
        "MAE": float(mae),
        "MSE": float(mse),
        "MAPE_percent": float(mape),
    }


# =============================================================================
# 4. MODEL PREDICTION FUNCTIONS
# =============================================================================

def griddata_predict(x_train, y_train, x_test, method):
    """
    Predict one point using scipy.griddata.
    """
    try:
        prediction = griddata(
            x_train,
            y_train,
            x_test,
            method=method,
            fill_value=np.nan,
        )

        prediction = float(np.asarray(prediction).reshape(-1)[0])

        if not np.isfinite(prediction):
            return np.nan

        return prediction

    except Exception:
        return np.nan


def rsm_predict(x_train, y_train, x_test, degree):
    """
    Fit a polynomial response surface of the specified total degree.
    """
    poly = PolynomialFeatures(
        degree=degree,
        include_bias=False,
    )

    X_train = poly.fit_transform(x_train)
    X_test = poly.transform(
        np.asarray(x_test, dtype=float).reshape(1, -1)
    )

    model = LinearRegression()
    model.fit(X_train, y_train)

    prediction = float(model.predict(X_test)[0])

    return prediction


# =============================================================================
# 5. LOAD STEP 1 TRAINING DATA
# =============================================================================

if not INPUT_FILE.exists():
    raise FileNotFoundError(
        f"Input file not found:\n{INPUT_FILE}"
    )

train = pd.read_excel(
    INPUT_FILE,
    sheet_name="Train_Anchors",
)

missing = [
    col for col in REQUIRED_COLUMNS
    if col not in train.columns
]

if missing:
    raise ValueError(
        f"Train_Anchors is missing required columns: {missing}"
    )

for col in REQUIRED_COLUMNS:
    train[col] = pd.to_numeric(
        train[col],
        errors="raise",
    )

if len(train) != 67:
    raise ValueError(
        f"Expected 67 Train_Anchors; found {len(train)}."
    )

# Safety check: the test sheet must exist but must NOT be loaded for fitting.
xls = pd.ExcelFile(INPUT_FILE)

if "Test_Holdout" not in xls.sheet_names:
    raise ValueError(
        "Test_Holdout sheet is missing from Step1_Stratified_Split.xlsx."
    )


# =============================================================================
# 6. LOOCV
# =============================================================================

summary_rows = []
fold_rows = []

X_all = train[["d", "Rpm"]].to_numpy(dtype=float)

for target in TARGETS:

    y_all = train[target].to_numpy(dtype=float)

    for method in METHODS:

        predictions = []
        actuals = []
        prediction_status = []

        for i in range(len(train)):

            mask = np.ones(
                len(train),
                dtype=bool,
            )

            mask[i] = False

            X_train = X_all[mask]
            y_train = y_all[mask]

            X_test = X_all[i].reshape(1, -1)
            y_test = float(y_all[i])

            if method in GRIDDATA_METHODS:

                pred = griddata_predict(
                    X_train,
                    y_train,
                    X_test,
                    GRIDDATA_METHODS[method],
                )

            elif method in RSM_DEGREES:

                pred = rsm_predict(
                    X_train,
                    y_train,
                    X_test,
                    RSM_DEGREES[method],
                )

            else:
                raise ValueError(
                    f"Unknown method: {method}"
                )

            is_valid = np.isfinite(pred)

            predictions.append(pred)
            actuals.append(y_test)

            prediction_status.append(
                "VALID" if is_valid else "INVALID"
            )

            fold_rows.append({
                "Target": target,
                "Method": method,
                "Fold_Index": i + 1,
                "Point_ID": int(train.iloc[i]["Point_ID"]),
                "d": float(train.iloc[i]["d"]),
                "Rpm": float(train.iloc[i]["Rpm"]),
                "Actual": y_test,
                "Prediction": pred,
                "Residual": (
                    pred - y_test
                    if np.isfinite(pred)
                    else np.nan
                ),
                "Absolute_Error": (
                    abs(pred - y_test)
                    if np.isfinite(pred)
                    else np.nan
                ),
                "Prediction_Status": (
                    "VALID" if is_valid else "INVALID"
                ),
            })

        y_true = np.asarray(actuals, dtype=float)
        y_pred = np.asarray(predictions, dtype=float)

        metrics = calculate_metrics(
            y_true,
            y_pred,
        )

        n_total = len(train)
        n_valid = metrics["N_Valid"]

        coverage = (
            100.0 * n_valid / n_total
        )

        summary_rows.append({
            "Target": target,
            "Method": method,
            "N_Total_Folds": n_total,
            "N_Valid_Folds": n_valid,
            "N_Invalid_Folds": n_total - n_valid,
            "Coverage_percent": coverage,
            "R2": metrics["R2"],
            "RMSE": metrics["RMSE"],
            "MAE": metrics["MAE"],
            "MSE": metrics["MSE"],
            "MAPE_percent": metrics["MAPE_percent"],
        })


# =============================================================================
# 7. RESULTS TABLE
# =============================================================================

summary_df = pd.DataFrame(summary_rows)

# Rank methods within each target.
ranking_rows = []

for target in TARGETS:

    sub = summary_df[
        summary_df["Target"] == target
    ].copy()

    # Primary ranking:
    # 1. Full LOOCV coverage preferred
    # 2. Lowest RMSE
    # 3. Lowest MAE
    # 4. Highest R2
    sub["Full_Coverage"] = (
        sub["Coverage_percent"] >= 100.0
    )

    sub = sub.sort_values(
        by=[
            "Full_Coverage",
            "RMSE",
            "MAE",
            "R2",
        ],
        ascending=[
            False,
            True,
            True,
            False,
        ],
    ).reset_index(drop=True)

    for rank, (_, row) in enumerate(
        sub.iterrows(),
        start=1,
    ):
        ranking_rows.append({
            "Target": target,
            "Method": row["Method"],
            "Rank": rank,
            "Full_LOOCV_Coverage": row["Full_Coverage"],
            "Coverage_percent": row["Coverage_percent"],
            "R2": row["R2"],
            "RMSE": row["RMSE"],
            "MAE": row["MAE"],
            "MSE": row["MSE"],
            "MAPE_percent": row["MAPE_percent"],
        })

ranking_df = pd.DataFrame(ranking_rows)


# =============================================================================
# 8. BEST CANDIDATE SUMMARY
# =============================================================================
# This is a candidate ranking, NOT an irreversible lock.
# The user/researcher should inspect the complete metrics and coverage.

best_rows = []

for target in TARGETS:

    sub = ranking_df[
        ranking_df["Target"] == target
    ].sort_values("Rank")

    best = sub.iloc[0]

    best_rows.append({
        "Target": target,
        "Best_Candidate": best["Method"],
        "Full_LOOCV_Coverage": best["Full_LOOCV_Coverage"],
        "Coverage_percent": best["Coverage_percent"],
        "R2": best["R2"],
        "RMSE": best["RMSE"],
        "MAE": best["MAE"],
        "MSE": best["MSE"],
        "MAPE_percent": best["MAPE_percent"],
        "Selection_Basis": (
            "Full coverage preferred; then lowest RMSE, "
            "lowest MAE, highest R2."
        ),
        "Final_Status": "CANDIDATE - REVIEW BEFORE LOCKING",
    })

best_df = pd.DataFrame(best_rows)


# =============================================================================
# 9. COVERAGE SUMMARY
# =============================================================================

coverage_df = (
    summary_df[
        [
            "Target",
            "Method",
            "N_Total_Folds",
            "N_Valid_Folds",
            "N_Invalid_Folds",
            "Coverage_percent",
        ]
    ]
    .sort_values(
        ["Target", "Coverage_percent"],
        ascending=[True, False],
    )
    .reset_index(drop=True)
)


# =============================================================================
# 10. METHOD DEFINITIONS
# =============================================================================

method_definitions = pd.DataFrame([
    {
        "Method": "Linear Griddata",
        "Type": "2-D local interpolation",
        "Description": "Piecewise linear interpolation over (d, RPM).",
        "Can_Extrapolate": "No",
    },
    {
        "Method": "Cubic Griddata",
        "Type": "2-D local interpolation",
        "Description": "Piecewise cubic interpolation over (d, RPM).",
        "Can_Extrapolate": "No",
    },
    {
        "Method": "1st-order RSM",
        "Type": "Global polynomial",
        "Description": "Total polynomial degree 1 in d and RPM.",
        "Can_Extrapolate": "Yes",
    },
    {
        "Method": "2nd-order RSM",
        "Type": "Global polynomial",
        "Description": "Total polynomial degree 2 in d and RPM.",
        "Can_Extrapolate": "Yes",
    },
    {
        "Method": "3rd-order RSM",
        "Type": "Global polynomial",
        "Description": "Total polynomial degree 3 in d and RPM.",
        "Can_Extrapolate": "Yes",
    },
])


# =============================================================================
# 11. VALIDATION LOG
# =============================================================================

validation_log = pd.DataFrame([
    {
        "Check": "Input workbook exists",
        "Status": "PASS",
    },
    {
        "Check": "Train_Anchors sheet exists",
        "Status": "PASS",
    },
    {
        "Check": "67 training observations used",
        "Status": "PASS" if len(train) == 67 else "FAIL",
    },
    {
        "Check": "Test_Holdout loaded for fitting",
        "Status": "NO - PASS",
    },
    {
        "Check": "17 test observations used in LOOCV fitting",
        "Status": "NO - PASS",
    },
    {
        "Check": "Synthetic/interpolated rows used",
        "Status": "NO - PASS",
    },
    {
        "Check": "Targets evaluated",
        "Status": ", ".join(TARGETS),
    },
    {
        "Check": "Methods evaluated",
        "Status": ", ".join(METHODS),
    },
])


# =============================================================================
# 12. WRITE WORKBOOK
# =============================================================================

with pd.ExcelWriter(
    OUTPUT_FILE,
    engine="openpyxl",
) as writer:

    summary_df.to_excel(
        writer,
        sheet_name="LOOCV_Summary",
        index=False,
    )

    ranking_df.to_excel(
        writer,
        sheet_name="Method_Ranking",
        index=False,
    )

    best_df.to_excel(
        writer,
        sheet_name="Best_Candidates",
        index=False,
    )

    coverage_df.to_excel(
        writer,
        sheet_name="Coverage_Check",
        index=False,
    )

    fold_df = pd.DataFrame(fold_rows)

    fold_df.to_excel(
        writer,
        sheet_name="Fold_Predictions",
        index=False,
    )

    method_definitions.to_excel(
        writer,
        sheet_name="Method_Definitions",
        index=False,
    )

    validation_log.to_excel(
        writer,
        sheet_name="Validation_Log",
        index=False,
    )


# =============================================================================
# 13. CONSOLE OUTPUT
# =============================================================================

print("=" * 78)
print("STEP 2 - REVISED LOOCV METHOD SELECTION")
print("=" * 78)
print()

print(f"Input:  {INPUT_FILE}")
print(f"Output: {OUTPUT_FILE}")
print()

print("-" * 78)
print("DATA PROTOCOL")
print("-" * 78)
print("Train_Anchors used : 67")
print("Test_Holdout used  : 0")
print("Synthetic data used: 0")
print()

print("-" * 78)
print("LOOCV SUMMARY")
print("-" * 78)
print(summary_df.to_string(index=False))
print()

print("-" * 78)
print("METHOD RANKING")
print("-" * 78)
print(ranking_df.to_string(index=False))
print()

print("-" * 78)
print("BEST CANDIDATES")
print("-" * 78)
print(best_df.to_string(index=False))
print()

print("-" * 78)
print("IMPORTANT")
print("-" * 78)
print(
    "The Best_Candidates sheet contains candidates for review. "
    "Do not treat the automatic ranking as a final scientific lock "
    "until coverage, metrics, physical validity and the later "
    "interpolation requirements are reviewed."
)
print()

print("=" * 78)
print("COMPLETED")
print("=" * 78)
print("Next step: inspect Step 2 results before designing Step 3.")
