"""
STEP 4D - FINAL RPM EXTRAPOLATION DATASET GENERATION

Purpose
-------
Generate the final high-RPM extrapolation dataset using the methods selected
through Steps 4B-4C.

Locked extrapolation methods
----------------------------
hc   -> Saturating RPM model
A    -> Saturating RPM model
Vmin -> 3rd-order Response Surface Method (RSM)

Extrapolation grid
------------------
Diameters: 6, 8, 10, 12, 14, 16 mm
RPM:       280, 285, 290, ..., 320 RPM

Important
---------
1. Only the 67 real training anchors from Step1_Stratified_Split.xlsx are used
   for fitting.
2. The 17 final experimental holdout points are never used for fitting.
3. These extrapolated rows are kept separate and explicitly labelled.
4. No interpolated synthetic rows are used for fitting the extrapolation models.
5. The output is an engineering prediction dataset, not a replacement for
   experimental validation.
"""

from pathlib import Path
import warnings

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

warnings.filterwarnings("ignore")

BASE_DIR = Path(__file__).resolve().parent

INPUT_FILE = BASE_DIR / "Step1_Stratified_Split.xlsx"
OUTPUT_FILE = BASE_DIR / "Step4D_Final_RPM_Extrapolation.xlsx"

DIAMETERS = [6, 8, 10, 12, 14, 16]
EXTRAP_RPM = list(range(280, 321, 5))
TARGETS = ["hc", "A", "Vmin"]

# ---------------------------------------------------------------------
# Model definitions
# ---------------------------------------------------------------------

def saturating_model(rpm, a, b, c):
    """
    y = a + b * rpm / (c + rpm)
    """
    return a + b * rpm / (c + rpm)


def fit_saturating(x, y):
    """
    Fit the saturating model robustly with sensible initial values/bounds.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    # Initial values based on the observed response.
    a0 = float(np.min(y))
    b0 = float(np.max(y) - np.min(y))
    c0 = max(float(np.median(x)), 1.0)

    # Broad positive bounds for c. a/b remain practically unconstrained.
    lower = [-np.inf, -np.inf, 1e-9]
    upper = [np.inf, np.inf, np.inf]

    popt, _ = curve_fit(
        saturating_model,
        x,
        y,
        p0=[a0, b0, c0],
        bounds=(lower, upper),
        maxfev=100000,
    )
    return popt


def rsm3_fit_predict(train_df, pred_df, target):
    """
    Global third-order RSM over d and RPM:
        all polynomial terms up to total degree 3.
    """
    x_train = train_df[["d", "Rpm"]].to_numpy(dtype=float)
    y_train = train_df[target].to_numpy(dtype=float)
    x_pred = pred_df[["d", "Rpm"]].to_numpy(dtype=float)

    poly = PolynomialFeatures(degree=3, include_bias=False)
    X_train = poly.fit_transform(x_train)
    X_pred = poly.transform(x_pred)

    model = LinearRegression()
    model.fit(X_train, y_train)
    return model.predict(X_pred), model, poly


# ---------------------------------------------------------------------
# Load and validate Step 1 data
# ---------------------------------------------------------------------

if not INPUT_FILE.exists():
    raise FileNotFoundError(
        f"Input file not found: {INPUT_FILE}\n"
        "Place Step1_Stratified_Split.xlsx in the same folder as this script."
    )

train_df = pd.read_excel(INPUT_FILE, sheet_name="Train_Anchors")

required_cols = ["Point_ID", "d", "Rpm", "hc", "A", "Vmin"]
missing = [c for c in required_cols if c not in train_df.columns]
if missing:
    raise ValueError(f"Missing required columns: {missing}")

if len(train_df) != 67:
    raise ValueError(f"Expected 67 training anchors, found {len(train_df)}")

train_df = train_df.copy()
train_df["d"] = pd.to_numeric(train_df["d"], errors="raise")
train_df["Rpm"] = pd.to_numeric(train_df["Rpm"], errors="raise")

if train_df[["d", "Rpm", "hc", "A", "Vmin"]].isna().any().any():
    raise ValueError("Training data contains missing numeric values.")

if set(train_df["d"].unique()) != set(DIAMETERS):
    raise ValueError(
        f"Unexpected diameter set: {sorted(train_df['d'].unique().tolist())}"
    )

# ---------------------------------------------------------------------
# Build extrapolation coordinate grid
# ---------------------------------------------------------------------

pred_coords = pd.DataFrame(
    [(d, rpm) for d in DIAMETERS for rpm in EXTRAP_RPM],
    columns=["d", "Rpm"],
)

# Confirm none of the extrapolation coordinates are inside training anchors.
train_coords = set(zip(train_df["d"], train_df["Rpm"]))
overlap = [
    (int(d), int(rpm))
    for d, rpm in zip(pred_coords["d"], pred_coords["Rpm"])
    if (d, rpm) in train_coords
]
if overlap:
    raise ValueError(f"Extrapolation grid overlaps training coordinates: {overlap}")

# ---------------------------------------------------------------------
# Fit hc and A with the locked saturating model
#       separately for each fixed diameter
# ---------------------------------------------------------------------

prediction = pred_coords.copy()

fit_log_rows = []

for target in ["hc", "A"]:
    prediction[f"{target}_pred"] = np.nan

    for d in DIAMETERS:
        sub = train_df.loc[train_df["d"] == d].sort_values("Rpm")
        x = sub["Rpm"].to_numpy(dtype=float)
        y = sub[target].to_numpy(dtype=float)

        try:
            params = fit_saturating(x, y)
            pred_mask = prediction["d"] == d
            rpm_pred = prediction.loc[pred_mask, "Rpm"].to_numpy(dtype=float)

            prediction.loc[pred_mask, f"{target}_pred"] = saturating_model(
                rpm_pred, *params
            )

            fit_log_rows.append({
                "Target": target,
                "Diameter_mm": int(d),
                "Model": "Saturating",
                "n_training_anchors": len(sub),
                "a": float(params[0]),
                "b": float(params[1]),
                "c": float(params[2]),
                "RPM_fit_min": float(x.min()),
                "RPM_fit_max": float(x.max()),
                "Extrap_RPM_min": min(EXTRAP_RPM),
                "Extrap_RPM_max": max(EXTRAP_RPM),
                "Fit_Status": "Success",
            })

        except Exception as exc:
            fit_log_rows.append({
                "Target": target,
                "Diameter_mm": int(d),
                "Model": "Saturating",
                "n_training_anchors": len(sub),
                "a": np.nan,
                "b": np.nan,
                "c": np.nan,
                "RPM_fit_min": float(x.min()),
                "RPM_fit_max": float(x.max()),
                "Extrap_RPM_min": min(EXTRAP_RPM),
                "Extrap_RPM_max": max(EXTRAP_RPM),
                "Fit_Status": f"FAILED: {exc}",
            })
            raise

# ---------------------------------------------------------------------
# Fit Vmin with locked third-order RSM
# ---------------------------------------------------------------------

vmin_pred, rsm_model, rsm_poly = rsm3_fit_predict(
    train_df, prediction, "Vmin"
)
prediction["Vmin_pred"] = vmin_pred

# ---------------------------------------------------------------------
# Derived consistency quantity
# ---------------------------------------------------------------------

prediction["A_times_hc_pred"] = (
    prediction["A_pred"] * prediction["hc_pred"]
)

prediction["Source"] = "EXTRAPOLATED"
prediction["Method_hc"] = "Saturating RPM model"
prediction["Method_A"] = "Saturating RPM model"
prediction["Method_Vmin"] = "3rd-order RSM"
prediction["Validation_Status"] = (
    "Engineering extrapolation - not experimentally validated at these RPMs"
)

prediction = prediction[
    [
        "d",
        "Rpm",
        "hc_pred",
        "A_pred",
        "Vmin_pred",
        "A_times_hc_pred",
        "Source",
        "Method_hc",
        "Method_A",
        "Method_Vmin",
        "Validation_Status",
    ]
].sort_values(["d", "Rpm"]).reset_index(drop=True)

# ---------------------------------------------------------------------
# Physical and completeness checks
# ---------------------------------------------------------------------

physical_rows = []

for target in ["hc_pred", "A_pred", "Vmin_pred"]:
    vals = prediction[target].to_numpy(dtype=float)
    physical_rows.append({
        "Target": target,
        "N": len(vals),
        "Nonfinite": int(np.sum(~np.isfinite(vals))),
        "Negative": int(np.sum(vals < 0)),
        "Minimum": float(np.min(vals)),
        "Maximum": float(np.max(vals)),
        "Physically_Nonnegative": bool(np.all(vals >= 0)),
    })

physical_check = pd.DataFrame(physical_rows)

grid_check = pd.DataFrame([{
    "Expected_rows": len(DIAMETERS) * len(EXTRAP_RPM),
    "Actual_rows": len(prediction),
    "Expected_diameters": len(DIAMETERS),
    "Actual_diameters": prediction["d"].nunique(),
    "Expected_RPM_levels_per_diameter": len(EXTRAP_RPM),
    "Actual_RPM_levels_per_diameter": prediction.groupby("d").size().min(),
    "RPM_min": prediction["Rpm"].min(),
    "RPM_max": prediction["Rpm"].max(),
    "Coordinate_overlap_with_training": len(overlap),
    "Complete_grid": (
        len(prediction) == len(DIAMETERS) * len(EXTRAP_RPM)
        and len(overlap) == 0
        and prediction.groupby("d").size().nunique() == 1
    ),
}])

diameter_summary = (
    prediction.groupby("d")
    .agg(
        Rows=("Rpm", "size"),
        RPM_min=("Rpm", "min"),
        RPM_max=("Rpm", "max"),
        hc_min=("hc_pred", "min"),
        hc_max=("hc_pred", "max"),
        A_min=("A_pred", "min"),
        A_max=("A_pred", "max"),
        Vmin_min=("Vmin_pred", "min"),
        Vmin_max=("Vmin_pred", "max"),
        Vmin_derived_min=("A_times_hc_pred", "min"),
        Vmin_derived_max=("A_times_hc_pred", "max"),
    )
    .reset_index()
    .rename(columns={"d": "Diameter_mm"})
)

fit_log = pd.DataFrame(fit_log_rows)

# Add RSM metadata.
rsm_metadata = pd.DataFrame([{
    "Target": "Vmin",
    "Model": "3rd-order RSM",
    "Degree": 3,
    "Inputs": "d, Rpm",
    "Training_observations": len(train_df),
    "Predictions_generated": len(prediction),
    "Polynomial_terms_including_intercept": int(len(rsm_model.coef_) + 1),
}])

# ---------------------------------------------------------------------
# Write output workbook
# ---------------------------------------------------------------------

with pd.ExcelWriter(OUTPUT_FILE, engine="openpyxl") as writer:
    prediction.to_excel(writer, sheet_name="Extrapolated_Data", index=False)
    diameter_summary.to_excel(writer, sheet_name="Diameter_Summary", index=False)
    physical_check.to_excel(writer, sheet_name="Physical_Check", index=False)
    grid_check.to_excel(writer, sheet_name="Grid_Check", index=False)
    fit_log.to_excel(writer, sheet_name="Saturating_Fit_Log", index=False)
    rsm_metadata.to_excel(writer, sheet_name="RSM_Metadata", index=False)

    # Compact notes sheet for traceability.
    notes = pd.DataFrame({
        "Item": [
            "Input workbook",
            "Input sheet",
            "Training observations",
            "Excluded final test observations",
            "Extrapolated diameters (mm)",
            "Extrapolated RPM",
            "hc method",
            "A method",
            "Vmin method",
            "Interpolation data used for fitting",
            "Purpose",
            "Important interpretation",
        ],
        "Value": [
            INPUT_FILE.name,
            "Train_Anchors",
            67,
            17,
            "6, 8, 10, 12, 14, 16",
            "280 to 320 RPM in 5-RPM increments",
            "Saturating RPM model, fitted separately for each diameter",
            "Saturating RPM model, fitted separately for each diameter",
            "Third-order RSM using d and RPM",
            "No",
            "Engineering extrapolation study",
            "Predictions beyond the measured RPM range; experimental holdout validation remains the primary model validation.",
        ],
    })
    notes.to_excel(writer, sheet_name="Method_Notes", index=False)

# ---------------------------------------------------------------------
# Console summary
# ---------------------------------------------------------------------

print("=" * 78)
print("STEP 4D - FINAL RPM EXTRAPOLATION DATASET")
print("=" * 78)
print(f"Input:  {INPUT_FILE}")
print(f"Output: {OUTPUT_FILE}")
print()
print(f"Training anchors used: {len(train_df)}")
print(f"Final test points used: 0")
print(f"Diameters: {DIAMETERS}")
print(f"RPM grid: {EXTRAP_RPM}")
print(f"Extrapolated rows: {len(prediction)}")
print()
print("Locked methods:")
print("  hc   -> Saturating RPM model")
print("  A    -> Saturating RPM model")
print("  Vmin -> 3rd-order RSM")
print()
print("Physical checks:")
print(physical_check.to_string(index=False))
print()
print("Grid check:")
print(grid_check.to_string(index=False))
print()
print("COMPLETED")
print("=" * 78)
