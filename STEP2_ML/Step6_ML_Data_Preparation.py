"""
Step 6 — ML Data Preparation & Leakage Audit
Input : Step5B_Final_Analysis_Consolidated.xlsx
Output: Step6_ML_Data_Preparation.xlsx

Builds ML-ready train/holdout matrices from the consolidated Step 5 output.
Train: up to 315 points (measured + interpolated, optionally + extrapolated).
Holdout: 17 locked experimental points, never used for model selection.
"""

from pathlib import Path
import pandas as pd
import numpy as np

BASE_DIR = Path(__file__).resolve().parent

INPUT_FILE  = BASE_DIR / "Step5B_Final_Analysis_Consolidated.xlsx"
OUTPUT_FILE = BASE_DIR / "Step6_ML_Data_Preparation.xlsx"

FEATURES = ["d", "Rpm"]
TARGETS  = ["hc_hi", "td_to", "Vch"]
ID_COL   = "Point_ID"

LOCKED_HOLDOUT_IDS = [71, 79, 83, 59, 64, 70, 43, 51, 53, 29, 36, 42, 18, 23, 27, 4, 8]

PHYSICAL_BOUNDS = {
    "hc_hi": (0.0, 1.0),
    "td_to": (0.0, np.inf),
    "Vch":   (0.0, 1.0),
}

EXPECTED_DIAMETERS = [6, 8, 10, 12, 14, 16]


# Load source sheets

interp_raw = pd.read_excel(INPUT_FILE, sheet_name="RPM_Interpolation_301")
extrap_raw = pd.read_excel(INPUT_FILE, sheet_name="RPM_Extrapolation_Ref")
holdout    = pd.read_excel(INPUT_FILE, sheet_name="ML_Holdout_17")

print(f"RPM_Interpolation_301: {interp_raw.shape[0]} rows")
print(f"RPM_Extrapolation_Ref: {extrap_raw.shape[0]} rows")
print(f"ML_Holdout_17:         {holdout.shape[0]} rows")

results = {}
overall_pass = True

def record(name, status, detail=""):
    global overall_pass
    flag = "PASS" if status else "FAIL"
    if not status:
        overall_pass = False
    results[name] = {"Status": flag, "Detail": str(detail)}
    print(f"  [{flag}] {name}: {detail}")


# Clean interpolation rows

# Drop NaN targets
interp_no_nan = interp_raw.dropna(subset=TARGETS)
nan_dropped = interp_raw.shape[0] - interp_no_nan.shape[0]
record("Interp_NaN_filter", True, f"{nan_dropped} rows removed")

# Drop rows that violate physical bounds
phys_mask = (
    (interp_no_nan["hc_hi"] >= PHYSICAL_BOUNDS["hc_hi"][0]) &
    (interp_no_nan["hc_hi"] <= PHYSICAL_BOUNDS["hc_hi"][1]) &
    (interp_no_nan["td_to"] >= PHYSICAL_BOUNDS["td_to"][0]) &
    (interp_no_nan["Vch"]   >= PHYSICAL_BOUNDS["Vch"][0]) &
    (interp_no_nan["Vch"]   <= PHYSICAL_BOUNDS["Vch"][1])
)
interp_clean = interp_no_nan[phys_mask].copy()
phys_dropped = interp_no_nan.shape[0] - interp_clean.shape[0]
record("Interp_physical_filter", True, f"{phys_dropped} rows removed")

# Remove any (d, Rpm) coordinates shared with the holdout to prevent leakage
holdout_coords = set(zip(holdout["d"], holdout["Rpm"]))
interp_no_holdout = interp_clean[
    ~interp_clean.apply(lambda r: (r["d"], r["Rpm"]) in holdout_coords, axis=1)
]
holdout_removed = interp_clean.shape[0] - interp_no_holdout.shape[0]
record("Interp_holdout_filter", True, f"{holdout_removed} rows removed")
print(f"  Clean interpolation rows: {interp_no_holdout.shape[0]}")

# Tag each row with its provenance
interp_final = interp_no_holdout[FEATURES + TARGETS].copy()
if "hc_hi_Source" in interp_no_holdout.columns:
    interp_final["Source"] = interp_no_holdout["hc_hi_Source"].apply(
        lambda s: "MEASURED" if s == "MEASURED_TRAIN" else "INTERPOLATED"
    ).values
else:
    interp_final["Source"] = "INTERPOLATED"


# Optionally append extrapolation rows

INCLUDE_EXTRAPOLATION = True

if INCLUDE_EXTRAPOLATION:
    extrap_wide = extrap_raw.pivot_table(
        index=["d", "Rpm"], columns="Target", values="Predicted"
    ).reset_index()
    extrap_wide = extrap_wide[FEATURES + TARGETS].copy()
    extrap_wide["Source"] = "EXTRAPOLATED"
    print(f"  Extrapolation rows (pivoted): {extrap_wide.shape[0]}, "
          f"RPM {extrap_wide['Rpm'].min()}-{extrap_wide['Rpm'].max()}")

    extrap_violations = sum(
        int(((extrap_wide[t] < lo) | (extrap_wide[t] > hi)).sum())
        for t, (lo, hi) in PHYSICAL_BOUNDS.items()
    )
    record("Extrap_physical_valid", extrap_violations == 0,
           f"violations={extrap_violations}")
    train = pd.concat([interp_final, extrap_wide], ignore_index=True)
else:
    record("Extrap_excluded", True, "Extrapolation omitted from training")
    train = interp_final.copy()

train.insert(0, ID_COL, range(1, len(train) + 1))

TRAIN_COUNT = len(train)
TRAIN_SHEET = f"ML_Train_{TRAIN_COUNT}"

source_counts = train["Source"].value_counts()
print(f"  Training set: {TRAIN_COUNT} rows  "
      + "  ".join(f"{s}={source_counts.get(s, 0)}" for s in ["MEASURED", "INTERPOLATED", "EXTRAPOLATED"]))


# Row and column sanity checks

record("Train_row_count", TRAIN_COUNT > 0,
       f"{TRAIN_COUNT} rows (MEASURED={source_counts.get('MEASURED',0)}, "
       f"INTERPOLATED={source_counts.get('INTERPOLATED',0)}, "
       f"EXTRAPOLATED={source_counts.get('EXTRAPOLATED',0)})")
record("Holdout_row_count", holdout.shape[0] == 17,
       f"{holdout.shape[0]} (expected 17)")

expected_train_cols = [ID_COL] + FEATURES + TARGETS + ["Source"]
record("Train_columns", list(train.columns) == expected_train_cols,
       f"{list(train.columns)}")
expected_holdout_cols = [ID_COL] + FEATURES + TARGETS
record("Holdout_columns", list(holdout.columns) == expected_holdout_cols,
       f"{list(holdout.columns)}")


# Leakage audit

# Every (d, Rpm) pair in the holdout must be absent from the training set
train_coords = set(zip(train["d"], train["Rpm"]))
holdout_coords_check = set(zip(holdout["d"], holdout["Rpm"]))
coord_overlap_final = train_coords & holdout_coords_check
record("No_coordinate_overlap", len(coord_overlap_final) == 0,
       f"overlap={coord_overlap_final if coord_overlap_final else 'none'}")

holdout_ids = set(holdout[ID_COL])
record("Holdout_IDs_match_locked",
       sorted(holdout_ids) == sorted(LOCKED_HOLDOUT_IDS),
       f"holdout_ids_sorted={sorted(holdout_ids)}")


# Missing / non-finite checks

for label, df in [("Train", train), ("Holdout", holdout)]:
    null_count = df[FEATURES + TARGETS].isnull().sum().sum()
    nonfinite = sum(
        int((~np.isfinite(df[c].astype(float))).sum())
        for c in FEATURES + TARGETS
        if df[c].dtype in [np.float64, np.int64, float, int]
    )
    record(f"{label}_no_missing", null_count == 0, f"null_cells={null_count}")
    record(f"{label}_all_finite", nonfinite == 0, f"nonfinite_cells={nonfinite}")


# Domain and physical-validity checks

train_diameters = sorted(train["d"].unique())
holdout_diameters = sorted(holdout["d"].unique())
record("Train_diameters_valid",
       set(train_diameters).issubset(set(EXPECTED_DIAMETERS)),
       f"found={train_diameters}")
record("Holdout_diameters_coverage",
       set(holdout_diameters).issubset(set(EXPECTED_DIAMETERS)),
       f"found={holdout_diameters}")

train_rpm_range = (int(train["Rpm"].min()), int(train["Rpm"].max()))
record("Train_RPM_range_valid", train_rpm_range[0] >= 0,
       f"RPM range: {train_rpm_range[0]} - {train_rpm_range[1]}")

for label, df in [("Train", train), ("Holdout", holdout)]:
    for t in TARGETS:
        lo, hi = PHYSICAL_BOUNDS[t]
        violations = ((df[t] < lo) | (df[t] > hi)).sum()
        record(f"{label}_{t}_physical", violations == 0,
               f"violations={violations} (bounds [{lo}, {hi}])")


# Feature and target statistics

feature_stats = train[FEATURES].describe().T
feature_stats["range"] = feature_stats["max"] - feature_stats["min"]
feature_stats["iqr"] = train[FEATURES].quantile(0.75) - train[FEATURES].quantile(0.25)
print("\nFeature statistics:")
print(feature_stats.to_string())

target_stats_train = train[TARGETS].describe().T
target_stats_train["range"] = target_stats_train["max"] - target_stats_train["min"]
target_stats_train["skew"] = train[TARGETS].skew()
target_stats_train["kurtosis"] = train[TARGETS].kurtosis()
print("\nTarget statistics - train:")
print(target_stats_train.to_string())

target_stats_holdout = holdout[TARGETS].describe().T
target_stats_holdout["range"] = target_stats_holdout["max"] - target_stats_holdout["min"]
target_stats_holdout["skew"] = holdout[TARGETS].skew()
target_stats_holdout["kurtosis"] = holdout[TARGETS].kurtosis()
print("\nTarget statistics - holdout:")
print(target_stats_holdout.to_string())


# Correlation summaries

feat_corr = train[FEATURES].corr()
print("\nFeature correlation:")
print(feat_corr.to_string())

feat_target_corr = train[FEATURES + TARGETS].corr().loc[FEATURES, TARGETS]
print("\nFeature-target correlation:")
print(feat_target_corr.to_string())


# Diameter-wise coverage

coverage_rows = []
for d_val in EXPECTED_DIAMETERS:
    t_rpms = sorted(train.loc[train["d"] == d_val, "Rpm"].tolist())
    h_rpms = sorted(holdout.loc[holdout["d"] == d_val, "Rpm"].tolist())
    coverage_rows.append({
        "d": d_val,
        "train_count": len(t_rpms),
        "holdout_count": len(h_rpms),
        "total": len(t_rpms) + len(h_rpms),
        "train_RPM_min": min(t_rpms) if t_rpms else None,
        "train_RPM_max": max(t_rpms) if t_rpms else None,
        "holdout_RPMs": str(h_rpms),
    })
coverage_df = pd.DataFrame(coverage_rows)
print("\nDiameter coverage:")
print(coverage_df.to_string(index=False))


# Source distribution

source_summary = train.groupby("Source").agg(
    count=("d", "size"),
    d_unique=("d", "nunique"),
    rpm_min=("Rpm", "min"),
    rpm_max=("Rpm", "max"),
).reset_index()
print("\nSource distribution:")
print(source_summary.to_string(index=False))


# ML-ready matrices

X_train   = train[FEATURES].copy()
y_train   = train[TARGETS].copy()
X_holdout = holdout[FEATURES].copy()
y_holdout = holdout[TARGETS].copy()

print(f"\nX_train   shape: {X_train.shape}")
print(f"y_train   shape: {y_train.shape}")
print(f"X_holdout shape: {X_holdout.shape}")
print(f"y_holdout shape: {y_holdout.shape}")


overall = "PASS" if overall_pass else "FAIL"
print(f"\nStep 6 overall status: {overall}")


# Write output workbook

# Source is exported to its own sheet; the main training sheet is kept clean
train_out = train.drop(columns=["Source"])

n_measured     = int((train["Source"] == "MEASURED").sum())
n_interpolated = int((train["Source"] == "INTERPOLATED").sum())
n_extrapolated = int((train["Source"] == "EXTRAPOLATED").sum())

if INCLUDE_EXTRAPOLATION:
    provenance_note = "measured + interpolated + extrapolated"
else:
    provenance_note = "measured + interpolated (extrapolation excluded)"

with pd.ExcelWriter(OUTPUT_FILE, engine="openpyxl") as writer:

    readme = pd.DataFrame({
        "Item": [
            "Step", "Input", "Output", "Date",
            "Features", "Targets", "Identifier",
            "Train_rows", "Train_sheet_name", "Holdout_rows",
            "Train_MEASURED", "Train_INTERPOLATED", "Train_EXTRAPOLATED",
            "Overall_Status",
            "Provenance",
            "Note_1",
            "Note_2",
        ],
        "Value": [
            "Step 6 - ML Data Preparation & Leakage Audit",
            str(INPUT_FILE.name),
            str(OUTPUT_FILE.name),
            pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
            ", ".join(FEATURES),
            ", ".join(TARGETS),
            f"{ID_COL} (sequential, not a model feature)",
            TRAIN_COUNT, TRAIN_SHEET, 17,
            n_measured, n_interpolated, n_extrapolated,
            overall,
            provenance_note,
            f"{ID_COL} must never be passed to a model",
            "ML_Holdout_17 is locked - not used for selection or tuning",
        ],
    })
    readme.to_excel(writer, sheet_name="README", index=False)

    audit_df = pd.DataFrame([
        {"Check": k, "Status": v["Status"], "Detail": v["Detail"]}
        for k, v in results.items()
    ])
    audit_df.to_excel(writer, sheet_name="Audit_Results", index=False)

    train_out.to_excel(writer, sheet_name=TRAIN_SHEET, index=False)
    holdout.to_excel(writer, sheet_name="ML_Holdout_17", index=False)

    train[[ID_COL, "d", "Rpm", "Source"]].to_excel(
        writer, sheet_name="Train_Source_Provenance", index=False)

    feature_stats.to_excel(writer, sheet_name="Feature_Statistics")
    target_stats_train.to_excel(writer, sheet_name="Target_Statistics_Train")
    target_stats_holdout.to_excel(writer, sheet_name="Target_Statistics_Holdout")
    feat_corr.to_excel(writer, sheet_name="Feature_Correlation")
    feat_target_corr.to_excel(writer, sheet_name="Feature_Target_Correlation")
    coverage_df.to_excel(writer, sheet_name="Diameter_Coverage", index=False)
    source_summary.to_excel(writer, sheet_name="Source_Distribution", index=False)

print(f"Output written to: {OUTPUT_FILE.name}  (sheet: {TRAIN_SHEET})")
print("Step 6 complete.")
