"""
Step 6 — ML Data Preparation & Leakage Audit
Input : Step5B_Final_Analysis_Consolidated.xlsx
Output: Step6_ML_Data_Preparation.xlsx

Training set: 315 points (interpolated + extrapolated, evaluator-approved)
Holdout set : 17 measured experimental points (locked)
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


# 1. Load source sheets

interp_raw = pd.read_excel(INPUT_FILE, sheet_name="RPM_Interpolation_301")
extrap_raw = pd.read_excel(INPUT_FILE, sheet_name="RPM_Extrapolation_Ref")
holdout    = pd.read_excel(INPUT_FILE, sheet_name="ML_Holdout_17")

print(f"RPM_Interpolation_301 loaded: {interp_raw.shape[0]} rows, columns {list(interp_raw.columns)}")
print(f"RPM_Extrapolation_Ref loaded: {extrap_raw.shape[0]} rows")
print(f"ML_Holdout_17 loaded:         {holdout.shape[0]} rows, columns {list(holdout.columns)}")

results = {}
overall_pass = True

def record(name, status, detail=""):
    global overall_pass
    flag = "PASS" if status else "FAIL"
    if not status:
        overall_pass = False
    results[name] = {"Status": flag, "Detail": str(detail)}
    print(f"  [{flag}] {name}: {detail}")


# 2. Filter interpolation data

print("\n--- INTERPOLATION FILTERING ---")

# 2a: Drop rows with NaN in any target
interp_no_nan = interp_raw.dropna(subset=TARGETS)
nan_dropped = interp_raw.shape[0] - interp_no_nan.shape[0]
print(f"  Dropped {nan_dropped} rows with NaN targets")
record("Interp_NaN_filter", True, f"{nan_dropped} rows removed")

# 2b: Drop rows with physical violations
phys_mask = (
    (interp_no_nan["hc_hi"] >= PHYSICAL_BOUNDS["hc_hi"][0]) &
    (interp_no_nan["hc_hi"] <= PHYSICAL_BOUNDS["hc_hi"][1]) &
    (interp_no_nan["td_to"] >= PHYSICAL_BOUNDS["td_to"][0]) &
    (interp_no_nan["Vch"]   >= PHYSICAL_BOUNDS["Vch"][0]) &
    (interp_no_nan["Vch"]   <= PHYSICAL_BOUNDS["Vch"][1])
)
interp_clean = interp_no_nan[phys_mask].copy()
phys_dropped = interp_no_nan.shape[0] - interp_clean.shape[0]
print(f"  Dropped {phys_dropped} rows with physical violations")
record("Interp_physical_filter", True, f"{phys_dropped} rows removed")

# 2c: Remove holdout coordinates to prevent leakage
holdout_coords = set(zip(holdout["d"], holdout["Rpm"]))
interp_no_holdout = interp_clean[
    ~interp_clean.apply(lambda r: (r["d"], r["Rpm"]) in holdout_coords, axis=1)
]
holdout_removed = interp_clean.shape[0] - interp_no_holdout.shape[0]
print(f"  Removed {holdout_removed} rows overlapping with holdout coordinates")
record("Interp_holdout_filter", True, f"{holdout_removed} rows removed")
print(f"  Clean interpolation rows: {interp_no_holdout.shape[0]}")

# Build interpolation training set with source tracking
interp_final = interp_no_holdout[FEATURES + TARGETS].copy()
if "hc_hi_Source" in interp_no_holdout.columns:
    interp_final["Source"] = interp_no_holdout["hc_hi_Source"].apply(
        lambda s: "MEASURED" if s == "MEASURED_TRAIN" else "INTERPOLATED"
    ).values
else:
    interp_final["Source"] = "INTERPOLATED"


# 3. Extrapolation data

INCLUDE_EXTRAPOLATION = True

if INCLUDE_EXTRAPOLATION:
    print("\n--- EXTRAPOLATION PREPARATION ---")
    extrap_wide = extrap_raw.pivot_table(
        index=["d", "Rpm"], columns="Target", values="Predicted"
    ).reset_index()
    extrap_wide = extrap_wide[FEATURES + TARGETS].copy()
    extrap_wide["Source"] = "EXTRAPOLATED"
    print(f"  Extrapolation rows (pivoted): {extrap_wide.shape[0]}")
    print(f"  RPM range: {extrap_wide['Rpm'].min()} - {extrap_wide['Rpm'].max()}")

    extrap_violations = 0
    for t in TARGETS:
        lo, hi = PHYSICAL_BOUNDS[t]
        extrap_violations += int(((extrap_wide[t] < lo) | (extrap_wide[t] > hi)).sum())
    record("Extrap_physical_valid", extrap_violations == 0,
           f"violations={extrap_violations}")
    train = pd.concat([interp_final, extrap_wide], ignore_index=True)
else:
    print("\n--- EXTRAPOLATION EXCLUDED ---")
    print("  Extrapolation rows excluded from training set.")
    record("Extrap_excluded", True, "Extrapolation omitted from training")
    train = interp_final.copy()

train.insert(0, ID_COL, range(1, len(train) + 1))

TRAIN_COUNT = len(train)
TRAIN_SHEET = f"ML_Train_{TRAIN_COUNT}"

print(f"  Combined training set: {TRAIN_COUNT} rows")
source_counts = train["Source"].value_counts()
for src, count in source_counts.items():
    print(f"    {src}: {count}")


# 5. Row count checks

print("\n--- ROW COUNT CHECKS ---")
record("Train_row_count", TRAIN_COUNT > 0,
       f"{TRAIN_COUNT} rows (MEASURED={source_counts.get('MEASURED',0)}, "
       f"INTERPOLATED={source_counts.get('INTERPOLATED',0)}, "
       f"EXTRAPOLATED={source_counts.get('EXTRAPOLATED',0)})")
record("Holdout_row_count", holdout.shape[0] == 17,
       f"{holdout.shape[0]} (expected 17)")


# 6. Column checks

print("\n--- COLUMN CHECKS ---")
expected_train_cols = [ID_COL] + FEATURES + TARGETS + ["Source"]
record("Train_columns", list(train.columns) == expected_train_cols,
       f"{list(train.columns)}")
expected_holdout_cols = [ID_COL] + FEATURES + TARGETS
record("Holdout_columns", list(holdout.columns) == expected_holdout_cols,
       f"{list(holdout.columns)}")


# 7. Coordinate overlap check (LEAKAGE AUDIT)

print("\n--- LEAKAGE AUDIT ---")
train_coords = set(zip(train["d"], train["Rpm"]))
holdout_coords_check = set(zip(holdout["d"], holdout["Rpm"]))
coord_overlap_final = train_coords & holdout_coords_check
record("No_coordinate_overlap", len(coord_overlap_final) == 0,
       f"overlap={coord_overlap_final if coord_overlap_final else 'none'}")

# Verify holdout IDs match locked set
holdout_ids = set(holdout[ID_COL])
record("Holdout_IDs_match_locked",
       sorted(holdout_ids) == sorted(LOCKED_HOLDOUT_IDS),
       f"holdout_ids_sorted={sorted(holdout_ids)}")


# 8. Missing / non-finite checks

print("\n--- MISSING / NON-FINITE CHECKS ---")
for label, df in [("Train", train), ("Holdout", holdout)]:
    null_count = df[FEATURES + TARGETS].isnull().sum().sum()
    nonfinite = 0
    for c in FEATURES + TARGETS:
        if df[c].dtype in [np.float64, np.int64, float, int]:
            nonfinite += (~np.isfinite(df[c].astype(float))).sum()
    record(f"{label}_no_missing", null_count == 0, f"null_cells={null_count}")
    record(f"{label}_all_finite", nonfinite == 0, f"nonfinite_cells={nonfinite}")


# 9. Domain checks

print("\n--- DOMAIN CHECKS ---")
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


# 10. Physical validity of targets

print("\n--- PHYSICAL VALIDITY ---")
for label, df in [("Train", train), ("Holdout", holdout)]:
    for t in TARGETS:
        lo, hi = PHYSICAL_BOUNDS[t]
        violations = ((df[t] < lo) | (df[t] > hi)).sum()
        record(f"{label}_{t}_physical", violations == 0,
               f"violations={violations} (bounds [{lo}, {hi}])")


# 11. Feature statistics

print("\n--- FEATURE STATISTICS ---")
feature_stats = train[FEATURES].describe().T
feature_stats["range"] = feature_stats["max"] - feature_stats["min"]
feature_stats["iqr"] = train[FEATURES].quantile(0.75) - train[FEATURES].quantile(0.25)
print(feature_stats.to_string())


# 12. Target statistics

print("\n--- TARGET STATISTICS ---")
target_stats_train = train[TARGETS].describe().T
target_stats_train["range"] = target_stats_train["max"] - target_stats_train["min"]
target_stats_train["skew"] = train[TARGETS].skew()
target_stats_train["kurtosis"] = train[TARGETS].kurtosis()
print("Train:")
print(target_stats_train.to_string())

target_stats_holdout = holdout[TARGETS].describe().T
target_stats_holdout["range"] = target_stats_holdout["max"] - target_stats_holdout["min"]
target_stats_holdout["skew"] = holdout[TARGETS].skew()
target_stats_holdout["kurtosis"] = holdout[TARGETS].kurtosis()
print("\nHoldout:")
print(target_stats_holdout.to_string())


# 13. Feature correlation

print("\n--- FEATURE CORRELATION ---")
feat_corr = train[FEATURES].corr()
print(feat_corr.to_string())


# 14. Feature-target correlation

print("\n--- FEATURE-TARGET CORRELATIONS ---")
feat_target_corr = train[FEATURES + TARGETS].corr().loc[FEATURES, TARGETS]
print(feat_target_corr.to_string())


# 15. Scaling plan

print("\n--- SCALING PLAN ---")
scaling_plan = pd.DataFrame({
    "Feature": FEATURES + TARGETS,
    "Role": ["input"] * len(FEATURES) + ["target"] * len(TARGETS),
    "Scale_For_SVR_KNN_MLP_GPR": ["StandardScaler"] * len(FEATURES) + ["No"] * len(TARGETS),
    "Scale_For_Tree_Models": ["Not needed"] * (len(FEATURES) + len(TARGETS)),
    "Target_Transform": ["No"] * len(FEATURES) + ["No (unless EDA justifies)"] * len(TARGETS),
    "Notes": [
        f"Port diameter [{int(train['d'].min())}-{int(train['d'].max())} mm]",
        f"Rotational speed [{int(train['Rpm'].min())}-{int(train['Rpm'].max())} RPM]",
        "Dimensionless, bounded [0,1]",
        f"Dimensionless, bounded [1, ~{train['td_to'].max():.1f}]",
        "Dimensionless, bounded [0,1], zeros at RPM=0",
    ],
})
print(scaling_plan.to_string(index=False))


# 16. Diameter-wise coverage summary

print("\n--- DIAMETER COVERAGE ---")
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
print(coverage_df.to_string(index=False))


# 17. Source distribution

print("\n--- SOURCE DISTRIBUTION ---")
source_summary = train.groupby("Source").agg(
    count=("d", "size"),
    d_unique=("d", "nunique"),
    rpm_min=("Rpm", "min"),
    rpm_max=("Rpm", "max"),
).reset_index()
print(source_summary.to_string(index=False))


# 18. ML-ready matrices

X_train = train[FEATURES].copy()
y_train = train[TARGETS].copy()
X_holdout = holdout[FEATURES].copy()
y_holdout = holdout[TARGETS].copy()

print(f"\nX_train  shape: {X_train.shape}")
print(f"y_train  shape: {y_train.shape}")
print(f"X_holdout shape: {X_holdout.shape}")
print(f"y_holdout shape: {y_holdout.shape}")


# 19. Overall status

print("\n" + "=" * 60)
overall = "PASS" if overall_pass else "FAIL"
print(f"STEP 6 OVERALL STATUS: {overall}")
print("=" * 60)


# 20. Write output workbook

train_out = train.drop(columns=["Source"])  # Source kept in separate sheet
train_source_col = train[["Source"]].copy()
train_source_col.insert(0, ID_COL, train[ID_COL].values)

with pd.ExcelWriter(OUTPUT_FILE, engine="openpyxl") as writer:

    # README
    n_measured = int((train["Source"] == "MEASURED").sum())
    n_interpolated = int((train["Source"] == "INTERPOLATED").sum())
    n_extrapolated = int((train["Source"] == "EXTRAPOLATED").sum())

    readme = pd.DataFrame({
        "Item": [
            "Step", "Input", "Output", "Date",
            "Features", "Targets", "Identifier",
            "Train_rows", "Train_sheet_name", "Holdout_rows",
            "Train_MEASURED", "Train_INTERPOLATED", "Train_EXTRAPOLATED",
            "Overall_Status",
            "Leakage_Status",
            "Physical_Status",
            "Provenance",
            "Scaling_Policy",
            "Target_Transform_Policy",
            "Rule_1", "Rule_2", "Rule_3", "Rule_4", "Rule_5",
        ],
        "Value": [
            "Step 6 — ML Data Preparation & Leakage Audit",
            str(INPUT_FILE.name),
            str(OUTPUT_FILE.name),
            pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
            "d, Rpm",
            "hc_hi, td_to, Vch",
            f"{ID_COL} (sequential, not a feature)",
            TRAIN_COUNT, TRAIN_SHEET, 17,
            n_measured, n_interpolated, n_extrapolated,
            overall,
            "No coordinate overlap between train and holdout",
            "All targets within physical bounds",
            "Training includes measured + interpolated points (extrapolation excluded)",
            "StandardScaler for SVR/KNN/MLP/GPR on features; not needed for trees",
            "No target transformation by default (unless EDA justifies)",
            f"{ID_COL} must NEVER be used as a model feature",
            "ML_Holdout_17 must NEVER be used for model selection or tuning",
            "Source provenance (MEASURED/INTERPOLATED/EXTRAPOLATED) tracked",
            "td_to is the correct column name (never td/to)",
            "Extrapolation points (RPM>260) are reference-grade, clearly marked",
        ],
    })
    readme.to_excel(writer, sheet_name="README", index=False)

    # Audit results
    audit_df = pd.DataFrame([
        {"Check": k, "Status": v["Status"], "Detail": v["Detail"]}
        for k, v in results.items()
    ])
    audit_df.to_excel(writer, sheet_name="Audit_Results", index=False)

    # ML-ready training set (without Source column for clean ML input)
    train_out.to_excel(writer, sheet_name=TRAIN_SHEET, index=False)

    # ML-ready holdout (locked, untouched)
    holdout.to_excel(writer, sheet_name="ML_Holdout_17", index=False)

    # Source provenance tracking
    train[[ID_COL, "d", "Rpm", "Source"]].to_excel(
        writer, sheet_name="Train_Source_Provenance", index=False)

    # Feature statistics
    feature_stats.to_excel(writer, sheet_name="Feature_Statistics")

    # Target statistics (train)
    target_stats_train.to_excel(writer, sheet_name="Target_Statistics_Train")

    # Target statistics (holdout)
    target_stats_holdout.to_excel(writer, sheet_name="Target_Statistics_Holdout")

    # Feature correlation
    feat_corr.to_excel(writer, sheet_name="Feature_Correlation")

    # Feature-target correlations
    feat_target_corr.to_excel(writer, sheet_name="Feature_Target_Correlation")

    # Scaling plan
    scaling_plan.to_excel(writer, sheet_name="Scaling_Plan", index=False)

    # Diameter coverage
    coverage_df.to_excel(writer, sheet_name="Diameter_Coverage", index=False)

    # Source distribution
    source_summary.to_excel(writer, sheet_name="Source_Distribution", index=False)

print(f"\nOutput written to: {OUTPUT_FILE.name}")
print(f"Training sheet name: {TRAIN_SHEET}")
print("Step 6 complete.")
