"""
Step 3B - RPM Interpolation Validity Check

Purpose
-------
Validate whether the 5-RPM, 1-D cubic RPM interpolation used in Step 3A is
numerically reliable before any extrapolation is performed.

Validation is performed WITHOUT using the 17 locked test observations.

Method
------
For each response (hc, A, Vmin) and each diameter:

1. Use only the TRAINING ANCHORS.
2. Perform leave-one-anchor-out validation for anchors that remain inside the
   RPM range of the remaining anchors.
3. Fit a 1-D cubic spline to the remaining anchors.
4. Predict the omitted anchor.
5. Calculate:
      R2
      RMSE
      MAE
      MSE
      MAPE
6. Check whether the actual Step 3A synthetic values are finite.
7. Check for negative synthetic values.
8. Check whether each spline passes through its training anchors.
9. Produce plots of measured training anchors and 5-RPM interpolation curves.

The 17 test points are NOT used in fitting or interpolation validation.

Output
------
Step3B_RPM_Interpolation_Validity.xlsx
Step3B_RPM_Interpolation_Validity_Plots/
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline
from sklearn.metrics import (
    r2_score,
    mean_squared_error,
    mean_absolute_error,
)


# ---------------------------------------------------------------------------
# 1. Paths
# ---------------------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent

INPUT_SPLIT = BASE_DIR / "Step1_Stratified_Split.xlsx"
INPUT_EXPANDED = BASE_DIR / "Step3A_RPM_Interpolation.xlsx"

OUTPUT_EXCEL = BASE_DIR / "Step3B_RPM_Interpolation_Validity.xlsx"
PLOT_DIR = BASE_DIR / "Step3B_RPM_Interpolation_Validity_Plots"

PLOT_DIR.mkdir(exist_ok=True)


# ---------------------------------------------------------------------------
# 2. Configuration
# ---------------------------------------------------------------------------

TARGETS = ["hc", "A", "Vmin"]

REQUIRED_COLUMNS = [
    "Point_ID",
    "d",
    "Rpm",
    "hc",
    "A",
    "Vmin",
]


# ---------------------------------------------------------------------------
# 3. Load data
# ---------------------------------------------------------------------------

if not INPUT_SPLIT.exists():
    raise FileNotFoundError(f"Missing: {INPUT_SPLIT}")

if not INPUT_EXPANDED.exists():
    raise FileNotFoundError(f"Missing: {INPUT_EXPANDED}")

train = pd.read_excel(
    INPUT_SPLIT,
    sheet_name="Train_Anchors",
)

expanded = {
    target: pd.read_excel(
        INPUT_EXPANDED,
        sheet_name=f"{target}_Expanded",
    )
    for target in TARGETS
}

for df in [train]:
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Train_Anchors missing columns: {missing}")


# ---------------------------------------------------------------------------
# 4. Helper functions
# ---------------------------------------------------------------------------

def calculate_metrics(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)

    nonzero = np.abs(y_true) > 1e-12
    if nonzero.any():
        mape = np.mean(
            np.abs(
                (y_true[nonzero] - y_pred[nonzero])
                / y_true[nonzero]
            )
        ) * 100
    else:
        mape = np.nan

    return {
        "R2": r2,
        "RMSE": rmse,
        "MAE": mae,
        "MSE": mse,
        "MAPE_percent": mape,
    }


# ---------------------------------------------------------------------------
# 5. LOOCV validation
# ---------------------------------------------------------------------------

cv_rows = []
fold_rows = []

diameters = sorted(train["d"].unique())

for target in TARGETS:

    for d in diameters:

        subset = (
            train[train["d"] == d]
            .sort_values("Rpm")
            .reset_index(drop=True)
        )

        x_all = subset["Rpm"].to_numpy(dtype=float)
        y_all = subset[target].to_numpy(dtype=float)

        predictions = []
        actuals = []
        omitted_rpms = []

        # Leave one anchor out at a time.
        for i in range(len(subset)):

            x_train = np.delete(x_all, i)
            y_train = np.delete(y_all, i)

            x_test = x_all[i]
            y_test = y_all[i]

            # The omitted point must lie inside the remaining RPM range.
            # Boundary points are therefore excluded from interpolation CV
            # because predicting them would become extrapolation.
            if x_test <= x_train.min() or x_test >= x_train.max():
                continue

            if len(np.unique(x_train)) < 4:
                continue

            spline = CubicSpline(
                x_train,
                y_train,
                bc_type="not-a-knot",
                extrapolate=False,
            )

            prediction = float(spline(x_test))

            if not np.isfinite(prediction):
                continue

            actuals.append(y_test)
            predictions.append(prediction)
            omitted_rpms.append(x_test)

            fold_rows.append({
                "Target": target,
                "d": d,
                "Omitted_RPM": x_test,
                "Actual": y_test,
                "Predicted": prediction,
                "Absolute_Error": abs(y_test - prediction),
                "Percentage_Error": (
                    abs((y_test - prediction) / y_test) * 100
                    if abs(y_test) > 1e-12
                    else np.nan
                ),
            })

        if len(actuals) >= 2:
            metrics = calculate_metrics(actuals, predictions)

            cv_rows.append({
                "Target": target,
                "d": d,
                "Valid_LOOCV_Folds": len(actuals),
                **metrics,
            })
        else:
            cv_rows.append({
                "Target": target,
                "d": d,
                "Valid_LOOCV_Folds": len(actuals),
                "R2": np.nan,
                "RMSE": np.nan,
                "MAE": np.nan,
                "MSE": np.nan,
                "MAPE_percent": np.nan,
            })


cv_df = pd.DataFrame(cv_rows)
fold_df = pd.DataFrame(fold_rows)


# ---------------------------------------------------------------------------
# 6. Overall validation metrics
# ---------------------------------------------------------------------------

overall_rows = []

for target in TARGETS:

    subset = fold_df[fold_df["Target"] == target]

    if len(subset) >= 2:
        metrics = calculate_metrics(
            subset["Actual"],
            subset["Predicted"],
        )

        overall_rows.append({
            "Target": target,
            "Valid_LOOCV_Folds": len(subset),
            **metrics,
        })

overall_df = pd.DataFrame(overall_rows)


# ---------------------------------------------------------------------------
# 7. Validate Step 3A generated data
# ---------------------------------------------------------------------------

quality_rows = []

for target in TARGETS:

    df = expanded[target].copy()

    measured = df[df["Data_Type"] == "MEASURED"]
    synthetic = df[df["Data_Type"] == "SYNTHETIC_IN_RANGE"]

    # Check all target values are finite.
    finite_count = np.isfinite(df[target].to_numpy(dtype=float)).sum()

    # Negative values can be physically meaningful for some quantities,
    # but for these geometric responses they should be flagged.
    negative_count = (
        synthetic[target].to_numpy(dtype=float) < 0
    ).sum()

    # Check that synthetic RPM values are actually on 5-RPM spacing.
    fractional_steps = np.mod(
        synthetic["Rpm"].to_numpy(dtype=float),
        5.0,
    )

    non_5rpm_count = (
        np.abs(fractional_steps) > 1e-8
    ).sum()

    # Check interpolation exactly reproduces the training anchors.
    anchor_errors = []

    for _, row in measured.iterrows():

        d = float(row["d"])
        rpm = float(row["Rpm"])
        actual = float(row[target])

        # Refit using all training anchors at this diameter.
        subset = (
            train[train["d"] == d]
            .sort_values("Rpm")
        )

        spline = CubicSpline(
            subset["Rpm"].to_numpy(dtype=float),
            subset[target].to_numpy(dtype=float),
            bc_type="not-a-knot",
            extrapolate=False,
        )

        predicted = float(spline(rpm))
        anchor_errors.append(abs(actual - predicted))

    max_anchor_error = max(anchor_errors) if anchor_errors else np.nan

    quality_rows.append({
        "Target": target,
        "Rows_Total": len(df),
        "Measured_Anchors": len(measured),
        "Synthetic_RPM_Points": len(synthetic),
        "Finite_Target_Values": finite_count,
        "Nonfinite_Target_Values": len(df) - finite_count,
        "Negative_Synthetic_Values": negative_count,
        "Synthetic_Points_Not_On_5RPM_Grid": non_5rpm_count,
        "Max_Anchor_Reproduction_Error": max_anchor_error,
    })


quality_df = pd.DataFrame(quality_rows)


# ---------------------------------------------------------------------------
# 8. Plot interpolation curves
# ---------------------------------------------------------------------------

for target in TARGETS:

    for d in diameters:

        train_d = (
            train[train["d"] == d]
            .sort_values("Rpm")
        )

        expanded_d = (
            expanded[target][expanded[target]["d"] == d]
            .sort_values("Rpm")
        )

        measured = expanded_d[
            expanded_d["Data_Type"] == "MEASURED"
        ]

        synthetic = expanded_d[
            expanded_d["Data_Type"] == "SYNTHETIC_IN_RANGE"
        ]

        plt.figure(figsize=(9, 6))

        plt.plot(
            synthetic["Rpm"],
            synthetic[target],
            linewidth=2,
            label="5-RPM cubic interpolation",
        )

        plt.scatter(
            measured["Rpm"],
            measured[target],
            s=45,
            label="Measured training anchors",
            zorder=3,
        )

        plt.xlabel("RPM")
        plt.ylabel(target)
        plt.title(f"{target} vs RPM — d = {int(d)} mm")
        plt.grid(True, alpha=0.25)
        plt.legend()
        plt.tight_layout()

        plot_path = (
            PLOT_DIR
            / f"{target}_d{int(d)}mm_RPM_Interpolation.png"
        )

        plt.savefig(plot_path, dpi=200)
        plt.close()


# ---------------------------------------------------------------------------
# 9. Save Excel report
# ---------------------------------------------------------------------------

with pd.ExcelWriter(
    OUTPUT_EXCEL,
    engine="openpyxl",
) as writer:

    overall_df.to_excel(
        writer,
        sheet_name="Overall_LOOCV",
        index=False,
    )

    cv_df.to_excel(
        writer,
        sheet_name="Diameter_LOOCV",
        index=False,
    )

    fold_df.to_excel(
        writer,
        sheet_name="LOOCV_Folds",
        index=False,
    )

    quality_df.to_excel(
        writer,
        sheet_name="Interpolation_Quality",
        index=False,
    )


# ---------------------------------------------------------------------------
# 10. Console summary
# ---------------------------------------------------------------------------

print("\n" + "=" * 75)
print("STEP 3B - RPM INTERPOLATION VALIDITY CHECK COMPLETE")
print("=" * 75)

print("\nValidation basis:")
print("  Data used       : 67 TRAINING ANCHORS ONLY")
print("  Test observations used : 0")
print("  Method          : 1-D cubic RPM interpolation")
print("  Validation      : leave-one-anchor-out interpolation CV")
print("  Boundary folds  : excluded because they would be extrapolation")

print("\nOVERALL INTERPOLATION VALIDITY")
print("-" * 75)

if len(overall_df):
    print(
        overall_df[
            [
                "Target",
                "Valid_LOOCV_Folds",
                "R2",
                "RMSE",
                "MAE",
                "MAPE_percent",
            ]
        ].to_string(index=False)
    )

print("\nSTEP 3A DATA QUALITY")
print("-" * 75)

print(
    quality_df.to_string(index=False)
)

print("\nFiles:")
print(f"  Excel report : {OUTPUT_EXCEL}")
print(f"  Plots        : {PLOT_DIR}")

print("\nInterpretation:")
print(
    "The LOOCV metrics quantify how accurately the cubic RPM interpolation "
    "reconstructs known training-anchor values that were temporarily omitted."
)
print(
    "Boundary anchors are not included in interpolation CV because predicting "
    "them after removal would constitute extrapolation."
)
print(
    "If the LOOCV errors are small and the plots show smooth, physically "
    "reasonable curves, Step 3A can be accepted before extrapolation."
)

print("=" * 75)