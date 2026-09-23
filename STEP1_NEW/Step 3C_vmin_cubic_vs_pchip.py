"""
Step 3C - Cubic Spline vs PCHIP for Vmin RPM Interpolation

Purpose
-------
Resolve the three physically impossible negative Vmin values identified in
Step 3B by comparing the existing 1-D cubic spline interpolation with the
shape-preserving PCHIP interpolator.

IMPORTANT:
- Only the 67 TRAINING ANCHORS are used.
- The 17 locked test observations are not used.
- hc and A are not changed in this comparison.
- No extrapolation is performed.
- The comparison is only for Vmin RPM interpolation.

Validation
----------
For each diameter:
    - Leave one interior training RPM anchor out.
    - Fit CubicSpline and PCHIP to the remaining anchors.
    - Predict the omitted anchor.
    - Calculate R2, RMSE, MAE, MSE and MAPE.

Physical check
--------------
Each method is also evaluated on the 5-RPM interpolation grid:
    - number of negative Vmin values
    - minimum generated Vmin
    - number of non-finite values

Decision
--------
The preferred method should have:
    1. no physically impossible negative Vmin values, and
    2. comparable or better LOOCV interpolation accuracy.

Output
------
Step3C_Vmin_Cubic_vs_PCHIP.xlsx
Step3C_Vmin_Cubic_vs_PCHIP_Plots/
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline, PchipInterpolator
from sklearn.metrics import (
    r2_score,
    mean_squared_error,
    mean_absolute_error,
)


# ---------------------------------------------------------------------------
# 1. Paths
# ---------------------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent

INPUT_FILE = BASE_DIR / "Step1_Stratified_Split.xlsx"
OUTPUT_FILE = BASE_DIR / "Step3C_Vmin_Cubic_vs_PCHIP.xlsx"
PLOT_DIR = BASE_DIR / "Step3C_Vmin_Cubic_vs_PCHIP_Plots"

PLOT_DIR.mkdir(exist_ok=True)


# ---------------------------------------------------------------------------
# 2. Configuration
# ---------------------------------------------------------------------------

RPM_STEP = 5
TARGET = "Vmin"

REQUIRED_COLUMNS = [
    "Point_ID",
    "d",
    "Rpm",
    "hc",
    "A",
    "Vmin",
]


# ---------------------------------------------------------------------------
# 3. Load training anchors
# ---------------------------------------------------------------------------

if not INPUT_FILE.exists():
    raise FileNotFoundError(f"Could not find: {INPUT_FILE}")

train = pd.read_excel(
    INPUT_FILE,
    sheet_name="Train_Anchors",
)

missing = [c for c in REQUIRED_COLUMNS if c not in train.columns]
if missing:
    raise ValueError(f"Train_Anchors is missing columns: {missing}")

train["d"] = pd.to_numeric(train["d"])
train["Rpm"] = pd.to_numeric(train["Rpm"])
train[TARGET] = pd.to_numeric(train[TARGET])

if len(train) != 67:
    raise ValueError(
        f"Expected 67 training anchors, found {len(train)}."
    )


# ---------------------------------------------------------------------------
# 4. Metrics
# ---------------------------------------------------------------------------

def metrics(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)

    nz = np.abs(y_true) > 1e-12

    if nz.any():
        mape = (
            np.mean(
                np.abs(
                    (y_true[nz] - y_pred[nz])
                    / y_true[nz]
                )
            )
            * 100
        )
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
# 5. LOOCV comparison
# ---------------------------------------------------------------------------

fold_rows = []

diameters = sorted(train["d"].unique())

for d in diameters:

    subset = (
        train[train["d"] == d]
        .sort_values("Rpm")
        .reset_index(drop=True)
    )

    x_all = subset["Rpm"].to_numpy(dtype=float)
    y_all = subset[TARGET].to_numpy(dtype=float)

    for i in range(len(subset)):

        x_train = np.delete(x_all, i)
        y_train = np.delete(y_all, i)

        x_test = x_all[i]
        y_test = y_all[i]

        # Boundary points would become extrapolation after being removed.
        if x_test <= x_train.min() or x_test >= x_train.max():
            continue

        if len(np.unique(x_train)) < 4:
            continue

        interpolators = {
            "CubicSpline": CubicSpline(
                x_train,
                y_train,
                bc_type="not-a-knot",
                extrapolate=False,
            ),
            "PCHIP": PchipInterpolator(
                x_train,
                y_train,
                extrapolate=False,
            ),
        }

        for method, interpolator in interpolators.items():

            prediction = float(interpolator(x_test))

            if not np.isfinite(prediction):
                continue

            fold_rows.append({
                "Diameter_mm": d,
                "Omitted_RPM": x_test,
                "Method": method,
                "Actual_Vmin": y_test,
                "Predicted_Vmin": prediction,
                "Absolute_Error": abs(y_test - prediction),
                "Percentage_Error": (
                    abs((y_test - prediction) / y_test) * 100
                    if abs(y_test) > 1e-12
                    else np.nan
                ),
            })


fold_df = pd.DataFrame(fold_rows)


# ---------------------------------------------------------------------------
# 6. Overall method comparison
# ---------------------------------------------------------------------------

overall_rows = []

for method in ["CubicSpline", "PCHIP"]:

    subset = fold_df[fold_df["Method"] == method]

    m = metrics(
        subset["Actual_Vmin"],
        subset["Predicted_Vmin"],
    )

    overall_rows.append({
        "Method": method,
        "Valid_LOOCV_Folds": len(subset),
        **m,
    })

overall_df = pd.DataFrame(overall_rows)


# ---------------------------------------------------------------------------
# 7. Diameter-level comparison
# ---------------------------------------------------------------------------

diameter_rows = []

for d in diameters:

    for method in ["CubicSpline", "PCHIP"]:

        subset = fold_df[
            (fold_df["Diameter_mm"] == d)
            & (fold_df["Method"] == method)
        ]

        if len(subset) >= 2:

            m = metrics(
                subset["Actual_Vmin"],
                subset["Predicted_Vmin"],
            )

            diameter_rows.append({
                "Diameter_mm": d,
                "Method": method,
                "Valid_LOOCV_Folds": len(subset),
                **m,
            })

diameter_df = pd.DataFrame(diameter_rows)


# ---------------------------------------------------------------------------
# 8. Generate complete 5-RPM interpolation and physical checks
# ---------------------------------------------------------------------------

physical_rows = []
interpolation_rows = []

for d in diameters:

    subset = (
        train[train["d"] == d]
        .sort_values("Rpm")
    )

    x = subset["Rpm"].to_numpy(dtype=float)
    y = subset[TARGET].to_numpy(dtype=float)

    rpm_grid = np.arange(
        x.min(),
        x.max() + RPM_STEP,
        RPM_STEP,
        dtype=float,
    )

    rpm_grid = rpm_grid[rpm_grid <= x.max() + 1e-9]

    cubic = CubicSpline(
        x,
        y,
        bc_type="not-a-knot",
        extrapolate=False,
    )

    pchip = PchipInterpolator(
        x,
        y,
        extrapolate=False,
    )

    cubic_values = np.asarray(cubic(rpm_grid), dtype=float)
    pchip_values = np.asarray(pchip(rpm_grid), dtype=float)

    for rpm, cubic_value, pchip_value in zip(
        rpm_grid,
        cubic_values,
        pchip_values,
    ):

        interpolation_rows.append({
            "d": d,
            "Rpm": rpm,
            "CubicSpline_Vmin": cubic_value,
            "PCHIP_Vmin": pchip_value,
        })

    for method, values in [
        ("CubicSpline", cubic_values),
        ("PCHIP", pchip_values),
    ]:

        finite = np.isfinite(values)
        finite_values = values[finite]

        physical_rows.append({
            "Diameter_mm": d,
            "Method": method,
            "Grid_Points": len(values),
            "Nonfinite_Values": int((~finite).sum()),
            "Negative_Values": int((values < 0).sum()),
            "Minimum_Vmin": (
                float(np.min(finite_values))
                if len(finite_values)
                else np.nan
            ),
            "Maximum_Vmin": (
                float(np.max(finite_values))
                if len(finite_values)
                else np.nan
            ),
        })


physical_df = pd.DataFrame(physical_rows)
interpolation_df = pd.DataFrame(interpolation_rows)


# ---------------------------------------------------------------------------
# 9. Create method-level physical summary
# ---------------------------------------------------------------------------

physical_summary = (
    physical_df
    .groupby("Method", as_index=False)
    .agg(
        Total_Negative_Values=("Negative_Values", "sum"),
        Worst_Minimum_Vmin=("Minimum_Vmin", "min"),
        Total_Nonfinite_Values=("Nonfinite_Values", "sum"),
    )
)


# ---------------------------------------------------------------------------
# 10. Diagnostic plots
# ---------------------------------------------------------------------------

for d in diameters:

    subset = (
        train[train["d"] == d]
        .sort_values("Rpm")
    )

    plot_data = interpolation_df[
        interpolation_df["d"] == d
    ].sort_values("Rpm")

    plt.figure(figsize=(9, 6))

    plt.plot(
        plot_data["Rpm"],
        plot_data["CubicSpline_Vmin"],
        linewidth=2,
        label="Cubic spline",
    )

    plt.plot(
        plot_data["Rpm"],
        plot_data["PCHIP_Vmin"],
        linewidth=2,
        label="PCHIP",
    )

    plt.scatter(
        subset["Rpm"],
        subset["Vmin"],
        s=45,
        label="Measured training anchors",
        zorder=3,
    )

    plt.axhline(0, linewidth=1)

    plt.xlabel("RPM")
    plt.ylabel("Vmin")
    plt.title(f"Vmin RPM Interpolation Comparison — d = {int(d)} mm")
    plt.grid(True, alpha=0.25)
    plt.legend()
    plt.tight_layout()

    plot_path = (
        PLOT_DIR
        / f"Vmin_Cubic_vs_PCHIP_d{int(d)}mm.png"
    )

    plt.savefig(plot_path, dpi=200)
    plt.close()


# ---------------------------------------------------------------------------
# 11. Save report
# ---------------------------------------------------------------------------

decision_rows = [
    ["CubicSpline", "Existing Step 3A method"],
    ["PCHIP", "Candidate shape-preserving alternative"],
    ["Validation", "67 training anchors only; boundary LOOCV folds excluded"],
    ["Test holdout", "Not used"],
    ["RPM grid", "5 RPM"],
    ["Diameter interpolation", "Not performed"],
    ["RPM extrapolation", "Not performed"],
    ["Diameter extrapolation", "Not performed"],
]

decision_df = pd.DataFrame(
    decision_rows,
    columns=["Item", "Description"],
)

with pd.ExcelWriter(
    OUTPUT_FILE,
    engine="openpyxl",
) as writer:

    overall_df.to_excel(
        writer,
        sheet_name="Overall_Comparison",
        index=False,
    )

    diameter_df.to_excel(
        writer,
        sheet_name="Diameter_Comparison",
        index=False,
    )

    fold_df.to_excel(
        writer,
        sheet_name="LOOCV_Folds",
        index=False,
    )

    physical_df.to_excel(
        writer,
        sheet_name="Physical_Check",
        index=False,
    )

    physical_summary.to_excel(
        writer,
        sheet_name="Physical_Summary",
        index=False,
    )

    interpolation_df.to_excel(
        writer,
        sheet_name="5RPM_Comparison",
        index=False,
    )

    decision_df.to_excel(
        writer,
        sheet_name="Methodology",
        index=False,
    )


# ---------------------------------------------------------------------------
# 12. Console output
# ---------------------------------------------------------------------------

print("\n" + "=" * 78)
print("STEP 3C - VMIN CUBIC SPLINE vs PCHIP COMPARISON COMPLETE")
print("=" * 78)

print("\nValidation basis:")
print("  Training anchors : 67")
print("  Test observations used : 0")
print("  Validation : leave-one-interior-anchor-out")
print("  Boundary folds : excluded")

print("\nOVERALL LOOCV COMPARISON")
print("-" * 78)
print(
    overall_df[
        [
            "Method",
            "Valid_LOOCV_Folds",
            "R2",
            "RMSE",
            "MAE",
            "MAPE_percent",
        ]
    ].to_string(index=False)
)

print("\nPHYSICAL VALIDITY OF 5-RPM GRID")
print("-" * 78)
print(
    physical_summary.to_string(index=False)
)

print("\nOutput:")
print(f"  Excel : {OUTPUT_FILE}")
print(f"  Plots : {PLOT_DIR}")

print("\nDecision rule:")
print("  Prefer a method that has no negative Vmin values while")
print("  retaining comparable or better interpolation accuracy.")

print("=" * 78)