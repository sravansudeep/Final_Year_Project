"""
STEP 5 - FINAL EXPERIMENTAL HOLDOUT VALIDATION

Purpose
-------
Evaluate the final continuous prediction methods against the 17 completely
untouched experimental holdout observations defined in Step 1.

Data protocol
-------------
- Input: Step1_Stratified_Split.xlsx
- Train_Anchors: 67 observations, used for fitting only
- Test_Holdout: 17 observations, used only for final validation
- No synthetic/interpolated rows are used for fitting in this step.

Methods evaluated
-----------------
hc:
    1. Cubic Griddata
    2. Saturating RPM model (fit separately at each diameter)

A:
    1. Cubic Griddata
    2. Saturating RPM model (fit separately at each diameter)

Vmin:
    1. 3rd-order RSM using d and RPM

Important
---------
Cubic Griddata may be unable to predict some holdout points if they lie outside
the convex hull of the 67 training anchors. Such predictions remain NaN and are
reported explicitly; they are never silently replaced.

The holdout set is also used for the physical consistency check:
    Vmin_derived = A_pred * hc_pred

where possible.
"""

from pathlib import Path
import warnings

import numpy as np
import pandas as pd
from scipy.interpolate import griddata
from scipy.optimize import curve_fit
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    r2_score,
    mean_squared_error,
    mean_absolute_error,
)

warnings.filterwarnings("ignore")

BASE_DIR = Path(__file__).resolve().parent

INPUT_FILE = BASE_DIR / "Step1_Stratified_Split.xlsx"
OUTPUT_FILE = BASE_DIR / "Step5_Final_Holdout_Validation.xlsx"

REQUIRED_COLS = ["Point_ID", "d", "Rpm", "hc", "A", "Vmin"]
TARGETS = ["hc", "A", "Vmin"]


# =============================================================================
# MODEL DEFINITIONS
# =============================================================================

def saturating_model(rpm, a, b, c):
    """
    Saturating RPM model:
        y = a + b * rpm / (c + rpm)
    """
    return a + b * rpm / (c + rpm)


def fit_saturating(x, y):
    """
    Fit the saturating model with c constrained to positive values.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    a0 = float(np.min(y))
    b0 = float(np.max(y) - np.min(y))
    c0 = max(float(np.median(x)), 1.0)

    popt, _ = curve_fit(
        saturating_model,
        x,
        y,
        p0=[a0, b0, c0],
        bounds=([-np.inf, -np.inf, 1e-9],
                [np.inf, np.inf, np.inf]),
        maxfev=100000,
    )
    return popt


def predict_saturating_by_diameter(train_df, holdout_df, target):
    """
    Fit a separate saturating RPM model for each fixed diameter.
    """
    predictions = np.full(len(holdout_df), np.nan, dtype=float)
    fit_rows = []

    for d in sorted(train_df["d"].unique()):
        train_sub = train_df[train_df["d"] == d].sort_values("Rpm")
        test_mask = holdout_df["d"].to_numpy() == d

        x = train_sub["Rpm"].to_numpy(dtype=float)
        y = train_sub[target].to_numpy(dtype=float)

        if test_mask.sum() == 0:
            continue

        params = fit_saturating(x, y)

        rpm_test = holdout_df.loc[test_mask, "Rpm"].to_numpy(dtype=float)
        predictions[test_mask] = saturating_model(rpm_test, *params)

        fit_rows.append({
            "Target": target,
            "Diameter_mm": int(d),
            "Model": "Saturating",
            "n_training_anchors": len(train_sub),
            "a": float(params[0]),
            "b": float(params[1]),
            "c": float(params[2]),
            "RPM_fit_min": float(x.min()),
            "RPM_fit_max": float(x.max()),
            "Fit_Status": "Success",
        })

    return predictions, pd.DataFrame(fit_rows)


def cubic_griddata_predict(train_df, holdout_df, target):
    """
    2-D cubic interpolation over (d, RPM).

    griddata returns NaN outside the convex hull of the training coordinates.
    """
    points = train_df[["d", "Rpm"]].to_numpy(dtype=float)
    values = train_df[target].to_numpy(dtype=float)
    xi = holdout_df[["d", "Rpm"]].to_numpy(dtype=float)

    predictions = griddata(
        points,
        values,
        xi,
        method="cubic",
        fill_value=np.nan,
    )

    return np.asarray(predictions, dtype=float)


def rsm3_predict(train_df, holdout_df, target):
    """
    Global third-order RSM over d and RPM using all terms through total degree 3.
    """
    X_train = train_df[["d", "Rpm"]].to_numpy(dtype=float)
    y_train = train_df[target].to_numpy(dtype=float)
    X_test = holdout_df[["d", "Rpm"]].to_numpy(dtype=float)

    poly = PolynomialFeatures(degree=3, include_bias=False)
    X_train_poly = poly.fit_transform(X_train)
    X_test_poly = poly.transform(X_test)

    model = LinearRegression()
    model.fit(X_train_poly, y_train)

    predictions = model.predict(X_test_poly)

    return np.asarray(predictions, dtype=float), model, poly


# =============================================================================
# METRICS
# =============================================================================

def regression_metrics(y_true, y_pred):
    """
    Calculate metrics on finite predictions only.

    MAPE is reported as a supplementary metric. Values with y_true == 0
    are excluded from the MAPE calculation to avoid division by zero.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    valid = np.isfinite(y_true) & np.isfinite(y_pred)
    n_valid = int(np.sum(valid))
    n_total = int(len(y_true))

    result = {
        "N_Total": n_total,
        "N_Valid": n_valid,
        "N_Missing_Predictions": n_total - n_valid,
        "Coverage_percent": (
            100.0 * n_valid / n_total if n_total else np.nan
        ),
        "R2": np.nan,
        "RMSE": np.nan,
        "MAE": np.nan,
        "MSE": np.nan,
        "MAPE_percent": np.nan,
    }

    if n_valid == 0:
        return result

    yt = y_true[valid]
    yp = y_pred[valid]

    result["R2"] = (
        r2_score(yt, yp) if n_valid >= 2 else np.nan
    )
    result["RMSE"] = float(np.sqrt(mean_squared_error(yt, yp)))
    result["MAE"] = float(mean_absolute_error(yt, yp))
    result["MSE"] = float(mean_squared_error(yt, yp))

    nonzero = np.abs(yt) > 1e-12
    if np.any(nonzero):
        result["MAPE_percent"] = float(
            np.mean(
                np.abs(
                    (yt[nonzero] - yp[nonzero]) / yt[nonzero]
                )
            ) * 100.0
        )

    return result


def physical_summary(values, label):
    values = np.asarray(values, dtype=float)
    finite = values[np.isfinite(values)]

    if len(finite) == 0:
        return {
            "Variable": label,
            "N": len(values),
            "Nonfinite": int(np.sum(~np.isfinite(values))),
            "Negative": 0,
            "Minimum": np.nan,
            "Maximum": np.nan,
            "Physically_Nonnegative": False,
        }

    return {
        "Variable": label,
        "N": len(values),
        "Nonfinite": int(np.sum(~np.isfinite(values))),
        "Negative": int(np.sum(finite < 0)),
        "Minimum": float(np.min(finite)),
        "Maximum": float(np.max(finite)),
        "Physically_Nonnegative": bool(np.all(finite >= 0)),
    }


# =============================================================================
# LOAD AND VALIDATE DATA
# =============================================================================

if not INPUT_FILE.exists():
    raise FileNotFoundError(
        f"Input file not found: {INPUT_FILE}"
    )

train_df = pd.read_excel(INPUT_FILE, sheet_name="Train_Anchors")
holdout_df = pd.read_excel(INPUT_FILE, sheet_name="Test_Holdout")

for name, df in [("Train_Anchors", train_df), ("Test_Holdout", holdout_df)]:
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(
            f"{name} is missing required columns: {missing}"
        )

if len(train_df) != 67:
    raise ValueError(
        f"Expected 67 training anchors; found {len(train_df)}"
    )

if len(holdout_df) != 17:
    raise ValueError(
        f"Expected 17 holdout observations; found {len(holdout_df)}"
    )

train_df = train_df.copy()
holdout_df = holdout_df.copy()

for df in [train_df, holdout_df]:
    for col in ["d", "Rpm", "hc", "A", "Vmin"]:
        df[col] = pd.to_numeric(df[col], errors="raise")

# Check no Point_ID overlap.
overlap_ids = set(train_df["Point_ID"]).intersection(
    set(holdout_df["Point_ID"])
)
if overlap_ids:
    raise ValueError(
        f"Training/test Point_ID overlap detected: {sorted(overlap_ids)}"
    )

# Check no coordinate overlap.
train_coords = set(zip(train_df["d"], train_df["Rpm"]))
holdout_coords = set(zip(holdout_df["d"], holdout_df["Rpm"]))
coord_overlap = train_coords.intersection(holdout_coords)
if coord_overlap:
    raise ValueError(
        f"Training/test coordinate overlap detected: {sorted(coord_overlap)}"
    )

# =============================================================================
# BUILD PREDICTIONS
# =============================================================================

prediction_df = holdout_df[
    ["Point_ID", "d", "Rpm", "hc", "A", "Vmin"]
].copy()

prediction_df = prediction_df.rename(
    columns={
        "hc": "hc_actual",
        "A": "A_actual",
        "Vmin": "Vmin_actual",
    }
)

fit_log_rows = []
metric_rows = []
coverage_rows = []

# -------------------------------------------------------------------------
# hc - cubic griddata
# -------------------------------------------------------------------------

hc_cubic = cubic_griddata_predict(train_df, holdout_df, "hc")
prediction_df["hc_pred_Cubic_Griddata"] = hc_cubic

m = regression_metrics(
    prediction_df["hc_actual"],
    hc_cubic,
)
metric_rows.append({
    "Target": "hc",
    "Model": "Cubic Griddata",
    **m,
})

coverage_rows.append({
    "Target": "hc",
    "Model": "Cubic Griddata",
    "Holdout_points": len(holdout_df),
    "Successful_predictions": int(np.isfinite(hc_cubic).sum()),
    "Missing_predictions": int(np.isnan(hc_cubic).sum()),
    "Coverage_percent": float(
        np.isfinite(hc_cubic).sum() / len(hc_cubic) * 100.0
    ),
})

# -------------------------------------------------------------------------
# hc - saturating
# -------------------------------------------------------------------------

hc_sat, hc_fit_log = predict_saturating_by_diameter(
    train_df, holdout_df, "hc"
)
prediction_df["hc_pred_Saturating"] = hc_sat
fit_log_rows.extend(hc_fit_log.to_dict("records"))

m = regression_metrics(
    prediction_df["hc_actual"],
    hc_sat,
)
metric_rows.append({
    "Target": "hc",
    "Model": "Saturating",
    **m,
})

coverage_rows.append({
    "Target": "hc",
    "Model": "Saturating",
    "Holdout_points": len(holdout_df),
    "Successful_predictions": int(np.isfinite(hc_sat).sum()),
    "Missing_predictions": int(np.isnan(hc_sat).sum()),
    "Coverage_percent": float(
        np.isfinite(hc_sat).sum() / len(hc_sat) * 100.0
    ),
})

# -------------------------------------------------------------------------
# A - cubic griddata
# -------------------------------------------------------------------------

A_cubic = cubic_griddata_predict(train_df, holdout_df, "A")
prediction_df["A_pred_Cubic_Griddata"] = A_cubic

m = regression_metrics(
    prediction_df["A_actual"],
    A_cubic,
)
metric_rows.append({
    "Target": "A",
    "Model": "Cubic Griddata",
    **m,
})

coverage_rows.append({
    "Target": "A",
    "Model": "Cubic Griddata",
    "Holdout_points": len(holdout_df),
    "Successful_predictions": int(np.isfinite(A_cubic).sum()),
    "Missing_predictions": int(np.isnan(A_cubic).sum()),
    "Coverage_percent": float(
        np.isfinite(A_cubic).sum() / len(A_cubic) * 100.0
    ),
})

# -------------------------------------------------------------------------
# A - saturating
# -------------------------------------------------------------------------

A_sat, A_fit_log = predict_saturating_by_diameter(
    train_df, holdout_df, "A"
)
prediction_df["A_pred_Saturating"] = A_sat
fit_log_rows.extend(A_fit_log.to_dict("records"))

m = regression_metrics(
    prediction_df["A_actual"],
    A_sat,
)
metric_rows.append({
    "Target": "A",
    "Model": "Saturating",
    **m,
})

coverage_rows.append({
    "Target": "A",
    "Model": "Saturating",
    "Holdout_points": len(holdout_df),
    "Successful_predictions": int(np.isfinite(A_sat).sum()),
    "Missing_predictions": int(np.isnan(A_sat).sum()),
    "Coverage_percent": float(
        np.isfinite(A_sat).sum() / len(A_sat) * 100.0
    ),
})

# -------------------------------------------------------------------------
# Vmin - 3rd-order RSM
# -------------------------------------------------------------------------

Vmin_rsm, Vmin_model, Vmin_poly = rsm3_predict(
    train_df, holdout_df, "Vmin"
)
prediction_df["Vmin_pred_3rd_RSM"] = Vmin_rsm

m = regression_metrics(
    prediction_df["Vmin_actual"],
    Vmin_rsm,
)
metric_rows.append({
    "Target": "Vmin",
    "Model": "3rd-order RSM",
    **m,
})

coverage_rows.append({
    "Target": "Vmin",
    "Model": "3rd-order RSM",
    "Holdout_points": len(holdout_df),
    "Successful_predictions": int(np.isfinite(Vmin_rsm).sum()),
    "Missing_predictions": int(np.isnan(Vmin_rsm).sum()),
    "Coverage_percent": float(
        np.isfinite(Vmin_rsm).sum() / len(Vmin_rsm) * 100.0
    ),
})

# =============================================================================
# VMIN CONSISTENCY CHECKS
# =============================================================================

consistency_df = prediction_df[
    ["Point_ID", "d", "Rpm", "Vmin_actual"]
].copy()

# Direct Vmin prediction
consistency_df["Vmin_direct_pred_3rd_RSM"] = Vmin_rsm

# Derived Vmin from the saturating predictions
consistency_df["Vmin_derived_Saturating_hc_A"] = (
    hc_sat * A_sat
)

# Derived Vmin from combinations involving cubic Griddata where both
# predictions are available.
consistency_df["Vmin_derived_Cubic_hc_A"] = (
    hc_cubic * A_cubic
)

# Errors for derived quantities
consistency_df["Error_direct_RSM"] = (
    consistency_df["Vmin_direct_pred_3rd_RSM"]
    - consistency_df["Vmin_actual"]
)

consistency_df["Error_derived_Saturating"] = (
    consistency_df["Vmin_derived_Saturating_hc_A"]
    - consistency_df["Vmin_actual"]
)

consistency_df["Error_derived_Cubic"] = (
    consistency_df["Vmin_derived_Cubic_hc_A"]
    - consistency_df["Vmin_actual"]
)

# Summary metrics for Vmin consistency.
consistency_metrics = []

for label, pred_col in [
    ("Direct 3rd-order RSM", "Vmin_direct_pred_3rd_RSM"),
    ("A_pred × hc_pred using Saturating models",
     "Vmin_derived_Saturating_hc_A"),
    ("A_pred × hc_pred using Cubic Griddata",
     "Vmin_derived_Cubic_hc_A"),
]:
    m = regression_metrics(
        consistency_df["Vmin_actual"],
        consistency_df[pred_col],
    )
    consistency_metrics.append({
        "Vmin_Consistency_Method": label,
        **m,
    })

consistency_metrics_df = pd.DataFrame(consistency_metrics)

# =============================================================================
# RESIDUALS
# =============================================================================

residual_rows = []

prediction_specs = [
    ("hc", "Cubic Griddata", "hc_actual", "hc_pred_Cubic_Griddata"),
    ("hc", "Saturating", "hc_actual", "hc_pred_Saturating"),
    ("A", "Cubic Griddata", "A_actual", "A_pred_Cubic_Griddata"),
    ("A", "Saturating", "A_actual", "A_pred_Saturating"),
    ("Vmin", "3rd-order RSM", "Vmin_actual", "Vmin_pred_3rd_RSM"),
]

for target, model_name, actual_col, pred_col in prediction_specs:
    for _, row in prediction_df.iterrows():
        actual = float(row[actual_col])
        pred = row[pred_col]

        if pd.isna(pred):
            residual = np.nan
            abs_error = np.nan
            sq_error = np.nan
        else:
            residual = float(pred) - actual
            abs_error = abs(residual)
            sq_error = residual ** 2

        residual_rows.append({
            "Point_ID": row["Point_ID"],
            "d": row["d"],
            "Rpm": row["Rpm"],
            "Target": target,
            "Model": model_name,
            "Actual": actual,
            "Predicted": pred,
            "Residual": residual,
            "Absolute_Error": abs_error,
            "Squared_Error": sq_error,
        })

residuals_df = pd.DataFrame(residual_rows)

# =============================================================================
# PHYSICAL VALIDITY CHECK
# =============================================================================

physical_rows = []

physical_columns = [
    ("hc", "Cubic Griddata", hc_cubic),
    ("hc", "Saturating", hc_sat),
    ("A", "Cubic Griddata", A_cubic),
    ("A", "Saturating", A_sat),
    ("Vmin", "3rd-order RSM", Vmin_rsm),
    ("Vmin", "Derived Saturating hc*A", hc_sat * A_sat),
    ("Vmin", "Derived Cubic hc*A", hc_cubic * A_cubic),
]

for target, model_name, vals in physical_columns:
    s = physical_summary(vals, f"{target} - {model_name}")
    s["Target"] = target
    s["Model"] = model_name
    physical_rows.append(s)

physical_df = pd.DataFrame(physical_rows)[
    [
        "Target",
        "Model",
        "N",
        "Nonfinite",
        "Negative",
        "Minimum",
        "Maximum",
        "Physically_Nonnegative",
        "Variable",
    ]
]

# =============================================================================
# METHOD LOG / VALIDATION LOG
# =============================================================================

method_log = pd.DataFrame([
    {
        "Target": "hc",
        "Model": "Cubic Griddata",
        "Fitting_Data": "67 Train_Anchors",
        "Prediction_Data": "17 Test_Holdout",
        "Method_Description": "2-D cubic interpolation over (d, RPM)",
        "Allowed_Outside_Convex_Hull": "No; NaN reported",
    },
    {
        "Target": "hc",
        "Model": "Saturating",
        "Fitting_Data": "67 Train_Anchors",
        "Prediction_Data": "17 Test_Holdout",
        "Method_Description": "Per-diameter y = a + b*RPM/(c+RPM)",
        "Allowed_Outside_Convex_Hull": "Yes, functional extrapolation",
    },
    {
        "Target": "A",
        "Model": "Cubic Griddata",
        "Fitting_Data": "67 Train_Anchors",
        "Prediction_Data": "17 Test_Holdout",
        "Method_Description": "2-D cubic interpolation over (d, RPM)",
        "Allowed_Outside_Convex_Hull": "No; NaN reported",
    },
    {
        "Target": "A",
        "Model": "Saturating",
        "Fitting_Data": "67 Train_Anchors",
        "Prediction_Data": "17 Test_Holdout",
        "Method_Description": "Per-diameter y = a + b*RPM/(c+RPM)",
        "Allowed_Outside_Convex_Hull": "Yes, functional extrapolation",
    },
    {
        "Target": "Vmin",
        "Model": "3rd-order RSM",
        "Fitting_Data": "67 Train_Anchors",
        "Prediction_Data": "17 Test_Holdout",
        "Method_Description": "Global polynomial response surface through degree 3",
        "Allowed_Outside_Convex_Hull": "Yes",
    },
])

validation_log = pd.DataFrame([
    {
        "Check": "Training observations",
        "Expected": 67,
        "Actual": len(train_df),
        "Status": "PASS" if len(train_df) == 67 else "FAIL",
    },
    {
        "Check": "Holdout observations",
        "Expected": 17,
        "Actual": len(holdout_df),
        "Status": "PASS" if len(holdout_df) == 17 else "FAIL",
    },
    {
        "Check": "Point_ID overlap",
        "Expected": 0,
        "Actual": len(overlap_ids),
        "Status": "PASS" if len(overlap_ids) == 0 else "FAIL",
    },
    {
        "Check": "Coordinate overlap",
        "Expected": 0,
        "Actual": len(coord_overlap),
        "Status": "PASS" if len(coord_overlap) == 0 else "FAIL",
    },
    {
        "Check": "Holdout used for fitting",
        "Expected": "No",
        "Actual": "No",
        "Status": "PASS",
    },
    {
        "Check": "Synthetic interpolation rows used for fitting",
        "Expected": "No",
        "Actual": "No",
        "Status": "PASS",
    },
])

# =============================================================================
# WRITE OUTPUT WORKBOOK
# =============================================================================

metrics_df = pd.DataFrame(metric_rows)
coverage_df = pd.DataFrame(coverage_rows)
fit_log_df = pd.DataFrame(fit_log_rows)

with pd.ExcelWriter(OUTPUT_FILE, engine="openpyxl") as writer:
    prediction_df.to_excel(
        writer, sheet_name="Holdout_Predictions", index=False
    )

    metrics_df.to_excel(
        writer, sheet_name="Metrics", index=False
    )

    coverage_df.to_excel(
        writer, sheet_name="Coverage_Check", index=False
    )

    consistency_df.to_excel(
        writer, sheet_name="Vmin_Consistency", index=False
    )

    consistency_metrics_df.to_excel(
        writer, sheet_name="Vmin_Consistency_Metrics", index=False
    )

    residuals_df.to_excel(
        writer, sheet_name="Residuals", index=False
    )

    physical_df.to_excel(
        writer, sheet_name="Physical_Check", index=False
    )

    method_log.to_excel(
        writer, sheet_name="Method_Log", index=False
    )

    validation_log.to_excel(
        writer, sheet_name="Validation_Log", index=False
    )

    fit_log_df.to_excel(
        writer, sheet_name="Saturating_Fit_Log", index=False
    )

# =============================================================================
# CONSOLE OUTPUT
# =============================================================================

print("=" * 78)
print("STEP 5 - FINAL EXPERIMENTAL HOLDOUT VALIDATION")
print("=" * 78)

print(f"Input:  {INPUT_FILE}")
print(f"Output: {OUTPUT_FILE}")
print()

print(f"Training anchors: {len(train_df)}")
print(f"Final holdout observations: {len(holdout_df)}")
print(f"Point_ID overlap: {len(overlap_ids)}")
print(f"Coordinate overlap: {len(coord_overlap)}")
print()

print("-" * 78)
print("METRICS")
print("-" * 78)
print(metrics_df.to_string(index=False))
print()

print("-" * 78)
print("COVERAGE")
print("-" * 78)
print(coverage_df.to_string(index=False))
print()

print("-" * 78)
print("VMIN CONSISTENCY METRICS")
print("-" * 78)
print(consistency_metrics_df.to_string(index=False))
print()

print("-" * 78)
print("PHYSICAL CHECK")
print("-" * 78)
print(physical_df.to_string(index=False))
print()

print("=" * 78)
print("COMPLETED")
print("=" * 78)
