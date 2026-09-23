"""
STEP 5A - VMIN PHYSICAL/DEFINITION CONSISTENCY INVESTIGATION

Purpose
-------
Investigate why the apparent relationship

    Vmin = A * hc

does not numerically hold in the Step 5 prediction check.

This script does NOT change any modelling results and does NOT refit the
predictive models. It is a diagnostic/definition investigation.

It checks:
1. How Vmin relates numerically to A * hc in the REAL measured data.
2. Whether the ratio Vmin / (A*hc) is approximately constant.
3. Whether that ratio varies with diameter or RPM.
4. Whether a simple unit/scaling factor could explain the discrepancy.
5. What happens at RPM = 0 where A*hc may be zero.
6. Whether the relationship is consistent separately in Train_Anchors and
   Test_Holdout.
7. The same relationship for Step 5 predictions, without using the result to
   alter the predictions.

Important:
- This script does NOT infer or invent units.
- It reports the source workbook's columns exactly as they exist.
- A conclusion such as "unit conversion factor" is only justified if the
  measured-data ratio is sufficiently stable.
- If the ratio is not stable, Vmin should be treated as an independently
  measured/defined response rather than mathematically regenerated from A*hc.
"""

from pathlib import Path
import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent

INPUT_SPLIT = BASE_DIR / "Step1_Stratified_Split.xlsx"
INPUT_STEP5 = BASE_DIR / "Step5_Final_Holdout_Validation.xlsx"
OUTPUT_FILE = BASE_DIR / "Step5A_Vmin_Consistency_Investigation.xlsx"

REQUIRED = ["Point_ID", "d", "Rpm", "hc", "A", "Vmin"]


# =============================================================================
# HELPERS
# =============================================================================

def safe_metrics(actual, pred):
    actual = np.asarray(actual, dtype=float)
    pred = np.asarray(pred, dtype=float)

    mask = np.isfinite(actual) & np.isfinite(pred)
    actual = actual[mask]
    pred = pred[mask]

    if len(actual) == 0:
        return {
            "N": 0,
            "MAE": np.nan,
            "RMSE": np.nan,
            "MAPE_percent": np.nan,
            "Mean_Error": np.nan,
            "Median_Error": np.nan,
            "R2": np.nan,
        }

    err = pred - actual
    mae = np.mean(np.abs(err))
    rmse = np.sqrt(np.mean(err ** 2))

    nz = np.abs(actual) > 1e-12
    mape = (
        np.mean(np.abs(err[nz] / actual[nz])) * 100.0
        if np.any(nz)
        else np.nan
    )

    ss_res = np.sum((actual - pred) ** 2)
    ss_tot = np.sum((actual - np.mean(actual)) ** 2)
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 1e-15 else np.nan

    return {
        "N": len(actual),
        "MAE": float(mae),
        "RMSE": float(rmse),
        "MAPE_percent": float(mape),
        "Mean_Error": float(np.mean(err)),
        "Median_Error": float(np.median(err)),
        "R2": float(r2),
    }


def describe_ratio(series):
    s = pd.to_numeric(series, errors="coerce")
    s = s[np.isfinite(s)]

    if len(s) == 0:
        return {
            "N": 0,
            "Mean": np.nan,
            "Median": np.nan,
            "Std": np.nan,
            "Min": np.nan,
            "Max": np.nan,
            "CV_percent": np.nan,
            "IQR": np.nan,
        }

    q1 = float(np.percentile(s, 25))
    q3 = float(np.percentile(s, 75))
    mean = float(np.mean(s))
    std = float(np.std(s, ddof=1)) if len(s) > 1 else 0.0

    return {
        "N": len(s),
        "Mean": mean,
        "Median": float(np.median(s)),
        "Std": std,
        "Min": float(np.min(s)),
        "Max": float(np.max(s)),
        "CV_percent": (
            abs(std / mean) * 100.0
            if abs(mean) > 1e-15
            else np.nan
        ),
        "IQR": q3 - q1,
    }


def add_relationship_columns(df):
    out = df.copy()

    out["A_times_hc"] = out["A"] * out["hc"]

    # Vmin / (A*hc), only where denominator is nonzero.
    denom = out["A_times_hc"].to_numpy(dtype=float)
    vmin = out["Vmin"].to_numpy(dtype=float)

    ratio = np.full(len(out), np.nan, dtype=float)
    nonzero = np.isfinite(denom) & np.isfinite(vmin) & (np.abs(denom) > 1e-12)
    ratio[nonzero] = vmin[nonzero] / denom[nonzero]

    out["Vmin_over_A_hc"] = ratio

    # Absolute difference if direct multiplication is treated literally.
    out["Vmin_minus_A_hc"] = out["Vmin"] - out["A_times_hc"]

    # Relative difference, only when Vmin is nonzero.
    v = out["Vmin"].to_numpy(dtype=float)
    rel = np.full(len(out), np.nan, dtype=float)
    nzv = np.isfinite(v) & (np.abs(v) > 1e-12)
    rel[nzv] = (out.loc[nzv, "A_times_hc"].to_numpy() - v[nzv]) / v[nzv] * 100.0
    out["A_hc_relative_difference_from_Vmin_percent"] = rel

    # Whether denominator is effectively zero.
    out["A_times_hc_zero_or_near_zero"] = (
        ~np.isfinite(denom) | (np.abs(denom) <= 1e-12)
    )

    return out


def ratio_summary_for_group(df, group_name):
    ratio = df["Vmin_over_A_hc"].dropna()
    desc = describe_ratio(ratio)

    result = {"Group": group_name, **desc}

    # Exact/near-exact constant-factor diagnostic.
    if len(ratio) > 1:
        result["Within_5_percent_of_median"] = int(
            np.sum(
                np.abs(ratio - ratio.median())
                <= 0.05 * abs(ratio.median())
            )
        )
        result["Within_10_percent_of_median"] = int(
            np.sum(
                np.abs(ratio - ratio.median())
                <= 0.10 * abs(ratio.median())
            )
        )
    else:
        result["Within_5_percent_of_median"] = np.nan
        result["Within_10_percent_of_median"] = np.nan

    return result


# =============================================================================
# LOAD DATA
# =============================================================================

if not INPUT_SPLIT.exists():
    raise FileNotFoundError(f"Missing input: {INPUT_SPLIT}")

train = pd.read_excel(INPUT_SPLIT, sheet_name="Train_Anchors")
test = pd.read_excel(INPUT_SPLIT, sheet_name="Test_Holdout")

for name, df in [("Train_Anchors", train), ("Test_Holdout", test)]:
    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        raise ValueError(f"{name} missing columns: {missing}")

    for c in ["d", "Rpm", "hc", "A", "Vmin"]:
        df[c] = pd.to_numeric(df[c], errors="raise")

# Full measured dataset for diagnostics only.
all_measured = pd.concat(
    [
        train.assign(Split="Train_Anchors"),
        test.assign(Split="Test_Holdout"),
    ],
    ignore_index=True,
)

all_measured = add_relationship_columns(all_measured)
train_diag = add_relationship_columns(train)
test_diag = add_relationship_columns(test)

# =============================================================================
# MEASURED-DATA RELATIONSHIP DIAGNOSTICS
# =============================================================================

measured_ratio_summary = pd.DataFrame([
    ratio_summary_for_group(all_measured, "All 84 measured"),
    ratio_summary_for_group(train_diag, "67 training anchors"),
    ratio_summary_for_group(test_diag, "17 holdout experiments"),
])

# Per-diameter ratio behaviour.
diameter_rows = []
for d, group in all_measured.groupby("d"):
    row = ratio_summary_for_group(group, f"d={int(d)} mm")
    row["Diameter_mm"] = int(d)
    diameter_rows.append(row)

diameter_ratio_summary = pd.DataFrame(diameter_rows)

# Per-RPM ratio behaviour.
rpm_rows = []
for rpm, group in all_measured.groupby("Rpm"):
    row = ratio_summary_for_group(group, f"RPM={int(rpm)}")
    row["RPM"] = int(rpm)
    rpm_rows.append(row)

rpm_ratio_summary = pd.DataFrame(rpm_rows)

# Zero-RPM and zero-product diagnostics.
zero_product = all_measured[
    all_measured["A_times_hc_zero_or_near_zero"]
].copy()

zero_rpm = all_measured[all_measured["Rpm"] == 0].copy()

# =============================================================================
# FIT A SIMPLE CONSTANT SCALE FACTOR — DIAGNOSTIC ONLY
# =============================================================================
# Least-squares constant k in:
#     Vmin ≈ k * (A*hc)
#
# This is NOT adopted automatically. It is only a test of whether a single
# scale factor could explain the discrepancy.

fit_df = all_measured[
    np.isfinite(all_measured["Vmin"])
    & np.isfinite(all_measured["A_times_hc"])
    & (np.abs(all_measured["A_times_hc"]) > 1e-12)
].copy()

if len(fit_df) > 0:
    x = fit_df["A_times_hc"].to_numpy(dtype=float)
    y = fit_df["Vmin"].to_numpy(dtype=float)

    denom = np.sum(x ** 2)
    k_origin = np.sum(x * y) / denom if denom > 1e-15 else np.nan

    pred_k = k_origin * x if np.isfinite(k_origin) else np.full(len(x), np.nan)
    k_metrics = safe_metrics(y, pred_k)

    constant_factor_summary = pd.DataFrame([{
        "Model": "Vmin = k*(A*hc), through origin",
        "Estimated_k": k_origin,
        **k_metrics,
        "Measured_points_used": len(fit_df),
    }])

    fit_df["Vmin_pred_constant_factor"] = pred_k
    fit_df["Residual_constant_factor"] = y - pred_k
else:
    constant_factor_summary = pd.DataFrame([{
        "Model": "Vmin = k*(A*hc), through origin",
        "Estimated_k": np.nan,
        "N": 0,
        "MAE": np.nan,
        "RMSE": np.nan,
        "MAPE_percent": np.nan,
        "Mean_Error": np.nan,
        "Median_Error": np.nan,
        "R2": np.nan,
        "Measured_points_used": 0,
    }])
    fit_df["Vmin_pred_constant_factor"] = np.nan
    fit_df["Residual_constant_factor"] = np.nan

# =============================================================================
# EXAMINE TRANSFORMED RELATIONSHIPS
# =============================================================================
# These are diagnostics only. They help identify whether the apparent mismatch
# may be multiplicative rather than a true identity.

positive_df = all_measured[
    (all_measured["Vmin"] > 0)
    & (all_measured["A_times_hc"] > 0)
].copy()

positive_df["log_Vmin"] = np.log(positive_df["Vmin"])
positive_df["log_Ahc"] = np.log(positive_df["A_times_hc"])

if len(positive_df) >= 2:
    slope, intercept = np.polyfit(
        positive_df["log_Ahc"],
        positive_df["log_Vmin"],
        1
    )
    positive_df["log_relationship_pred"] = (
        intercept + slope * positive_df["log_Ahc"]
    )
else:
    slope = np.nan
    intercept = np.nan
    positive_df["log_relationship_pred"] = np.nan

log_summary = pd.DataFrame([{
    "N_positive_points": len(positive_df),
    "Log_Log_slope": slope,
    "Log_Log_intercept": intercept,
    "Interpretation": (
        "Slope near 1 would support an approximately proportional relationship; "
        "slope far from 1 indicates that Vmin is not simply a constant multiple "
        "of A*hc."
    ),
}])

# =============================================================================
# STEP 5 PREDICTION CONSISTENCY (IF WORKBOOK EXISTS)
# =============================================================================

step5_prediction = None
step5_consistency_summary = None

if INPUT_STEP5.exists():
    try:
        step5_prediction = pd.read_excel(
            INPUT_STEP5,
            sheet_name="Holdout_Predictions"
        )

        # Add measured relationship for holdout.
        holdout_actual = step5_prediction[
            ["Point_ID", "d", "Rpm", "hc_actual", "A_actual", "Vmin_actual"]
        ].copy()

        holdout_actual["A_actual_times_hc_actual"] = (
            holdout_actual["A_actual"] * holdout_actual["hc_actual"]
        )

        # Predicted pairs.
        pred_relationships = []

        prediction_pairs = [
            (
                "Saturating_hc_A",
                "hc_pred_Saturating",
                "A_pred_Saturating",
            ),
            (
                "Cubic_hc_A",
                "hc_pred_Cubic_Griddata",
                "A_pred_Cubic_Griddata",
            ),
        ]

        for label, hc_col, a_col in prediction_pairs:
            if hc_col not in step5_prediction.columns or a_col not in step5_prediction.columns:
                continue

            step = step5_prediction[
                ["Point_ID", hc_col, a_col]
            ].copy()

            step["Predicted_A_times_hc"] = step[hc_col] * step[a_col]
            step["Prediction_Method"] = label
            pred_relationships.append(step)

        if pred_relationships:
            predicted_pairs = pd.concat(
                pred_relationships,
                ignore_index=True
            )

            step5_consistency = holdout_actual.merge(
                predicted_pairs,
                on="Point_ID",
                how="left"
            )

            step5_consistency["Predicted_vs_Measured_Ahc_Ratio"] = (
                step5_consistency["Predicted_A_times_hc"]
                / step5_consistency["A_actual_times_hc_actual"].replace(
                    0, np.nan
                )
            )

            summary_rows = []

            for method, group in step5_consistency.groupby("Prediction_Method"):
                valid = group[
                    np.isfinite(group["Predicted_A_times_hc"])
                    & np.isfinite(group["Vmin_actual"])
                ]

                metrics = safe_metrics(
                    valid["Vmin_actual"],
                    valid["Predicted_A_times_hc"]
                )

                summary_rows.append({
                    "Prediction_Method": method,
                    **metrics,
                    "Negative_predicted_A_times_hc": int(
                        np.sum(valid["Predicted_A_times_hc"] < 0)
                    ),
                    "Minimum_predicted_A_times_hc": (
                        float(valid["Predicted_A_times_hc"].min())
                        if len(valid) else np.nan
                    ),
                    "Maximum_predicted_A_times_hc": (
                        float(valid["Predicted_A_times_hc"].max())
                        if len(valid) else np.nan
                    ),
                })

            step5_consistency_summary = pd.DataFrame(summary_rows)
        else:
            step5_consistency = None
            step5_consistency_summary = pd.DataFrame()

    except Exception as exc:
        step5_consistency = None
        step5_consistency_summary = pd.DataFrame([{
            "Prediction_Method": "ERROR",
            "Message": str(exc),
        }])
else:
    step5_consistency = None
    step5_consistency_summary = pd.DataFrame([{
        "Prediction_Method": "Step5 workbook not found",
        "Message": f"Expected: {INPUT_STEP5.name}",
    }])

# =============================================================================
# OVERALL DECISION DIAGNOSTIC
# =============================================================================

ratio_all = all_measured["Vmin_over_A_hc"].dropna()

if len(ratio_all) >= 3:
    med = float(ratio_all.median())
    cv = describe_ratio(ratio_all)["CV_percent"]

    within10 = (
        np.mean(np.abs(ratio_all - med) <= 0.10 * abs(med)) * 100.0
        if abs(med) > 1e-15
        else 0.0
    )

    if cv <= 10 and within10 >= 80:
        conclusion = (
            "The measured data are approximately proportional to A*hc. "
            "A constant scaling/unit factor is plausible, but the factor's "
            "physical meaning must be confirmed from the experimental definition "
            "or workbook units before changing any model."
        )
    elif cv <= 25:
        conclusion = (
            "The measured data show some proportionality to A*hc but the scale "
            "factor varies materially. Do not treat Vmin as a strict identity "
            "with A*hc without checking definitions/units."
        )
    else:
        conclusion = (
            "The measured data do not support a stable constant-factor identity "
            "between Vmin and A*hc. Treat Vmin as an independently defined/measured "
            "response unless the source methodology establishes a different formula."
        )
else:
    conclusion = (
        "Insufficient nonzero A*hc observations to determine whether a stable "
        "relationship exists."
    )

decision_df = pd.DataFrame([{
    "Diagnostic": "Measured-data relationship",
    "Conclusion": conclusion,
}])

# =============================================================================
# WRITE WORKBOOK
# =============================================================================

with pd.ExcelWriter(OUTPUT_FILE, engine="openpyxl") as writer:
    all_measured.to_excel(
        writer,
        sheet_name="Measured_Relationships",
        index=False
    )

    measured_ratio_summary.to_excel(
        writer,
        sheet_name="Ratio_Summary",
        index=False
    )

    diameter_ratio_summary.to_excel(
        writer,
        sheet_name="Ratio_By_Diameter",
        index=False
    )

    rpm_ratio_summary.to_excel(
        writer,
        sheet_name="Ratio_By_RPM",
        index=False
    )

    zero_product.to_excel(
        writer,
        sheet_name="Zero_Product_Cases",
        index=False
    )

    zero_rpm.to_excel(
        writer,
        sheet_name="Zero_RPM_Cases",
        index=False
    )

    constant_factor_summary.to_excel(
        writer,
        sheet_name="Constant_Factor_Test",
        index=False
    )

    fit_df.to_excel(
        writer,
        sheet_name="Constant_Factor_Predictions",
        index=False
    )

    log_summary.to_excel(
        writer,
        sheet_name="Log_Log_Diagnostic",
        index=False
    )

    positive_df.to_excel(
        writer,
        sheet_name="Positive_Log_Data",
        index=False
    )

    if step5_consistency is not None:
        step5_consistency.to_excel(
            writer,
            sheet_name="Step5_Prediction_Check",
            index=False
        )

    step5_consistency_summary.to_excel(
        writer,
        sheet_name="Step5_Prediction_Summary",
        index=False
    )

    decision_df.to_excel(
        writer,
        sheet_name="Decision_Diagnostic",
        index=False
    )

    methodology = pd.DataFrame([
        ["Purpose", "Investigate numerical/definition consistency of Vmin versus A*hc"],
        ["Model fitting", "No predictive model refitted in this step"],
        ["Unit assumption", "None; source units are not inferred automatically"],
        ["Measured data", "84 real observations analysed diagnostically"],
        ["Training/test distinction", "67 anchors and 17 holdout retained as separate labels"],
        ["Constant factor test", "Diagnostic only; not adopted automatically"],
        ["Recommended action", "Confirm source definition/units before changing Vmin modelling"],
    ], columns=["Item", "Value"])

    methodology.to_excel(
        writer,
        sheet_name="Methodology",
        index=False
    )

# =============================================================================
# CONSOLE OUTPUT
# =============================================================================

print("=" * 78)
print("STEP 5A - VMIN PHYSICAL/DEFINITION CONSISTENCY INVESTIGATION")
print("=" * 78)
print()
print(f"Input split workbook: {INPUT_SPLIT}")
print(f"Step 5 workbook found: {INPUT_STEP5.exists()}")
print(f"Output: {OUTPUT_FILE}")
print()

print("-" * 78)
print("MEASURED-DATA RATIO SUMMARY: Vmin / (A*hc)")
print("-" * 78)
print(measured_ratio_summary.to_string(index=False))
print()

print("-" * 78)
print("CONSTANT FACTOR TEST")
print("-" * 78)
print(constant_factor_summary.to_string(index=False))
print()

print("-" * 78)
print("LOG-LOG DIAGNOSTIC")
print("-" * 78)
print(log_summary.to_string(index=False))
print()

print("-" * 78)
print("STEP 5 PREDICTION CONSISTENCY SUMMARY")
print("-" * 78)
print(step5_consistency_summary.to_string(index=False))
print()

print("-" * 78)
print("DIAGNOSTIC CONCLUSION")
print("-" * 78)
print(conclusion)
print()

print("=" * 78)
print("COMPLETED")
print("=" * 78)
