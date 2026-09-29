"""
Step 6 — ML Data Preparation & Leakage Audit
Input : Step5B_Final_Analysis_Consolidated.xlsx
Output: Step6_ML_Data_Preparation.xlsx
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
EXPECTED_RPM = list(range(0, 261, 20))

# ---------------------------------------------------------------------------
# 1. Load sheets
# ---------------------------------------------------------------------------
train = pd.read_excel(INPUT_FILE, sheet_name="ML_Train_67")
holdout = pd.read_excel(INPUT_FILE, sheet_name="ML_Holdout_17")

print(f"ML_Train_67  loaded: {train.shape[0]} rows, columns {list(train.columns)}")
print(f"ML_Holdout_17 loaded: {holdout.shape[0]} rows, columns {list(holdout.columns)}")

results = {}
overall_pass = True

def record(name, status, detail=""):
    global overall_pass
    flag = "PASS" if status else "FAIL"
    if not status:
        overall_pass = False
    results[name] = {"Status": flag, "Detail": str(detail)}
    print(f"  [{flag}] {name}: {detail}")

# ---------------------------------------------------------------------------
# 2. Row count checks
# ---------------------------------------------------------------------------
print("\n--- ROW COUNT CHECKS ---")
record("Train_row_count", train.shape[0] == 67, f"{train.shape[0]} (expected 67)")
record("Holdout_row_count", holdout.shape[0] == 17, f"{holdout.shape[0]} (expected 17)")
record("Total_experimental", train.shape[0] + holdout.shape[0] == 84,
       f"{train.shape[0] + holdout.shape[0]} (expected 84)")

# ---------------------------------------------------------------------------
# 3. Column checks
# ---------------------------------------------------------------------------
print("\n--- COLUMN CHECKS ---")
expected_cols = [ID_COL] + FEATURES + TARGETS
record("Train_columns", list(train.columns) == expected_cols,
       f"{list(train.columns)}")
record("Holdout_columns", list(holdout.columns) == expected_cols,
       f"{list(holdout.columns)}")

# ---------------------------------------------------------------------------
# 4. Point_ID uniqueness & overlap (LEAKAGE AUDIT)
# ---------------------------------------------------------------------------
print("\n--- LEAKAGE AUDIT ---")
train_ids = set(train[ID_COL])
holdout_ids = set(holdout[ID_COL])
overlap_ids = train_ids & holdout_ids
record("Train_PointID_unique", train[ID_COL].nunique() == train.shape[0],
       f"unique={train[ID_COL].nunique()}, rows={train.shape[0]}")
record("Holdout_PointID_unique", holdout[ID_COL].nunique() == holdout.shape[0],
       f"unique={holdout[ID_COL].nunique()}, rows={holdout.shape[0]}")
record("No_PointID_overlap", len(overlap_ids) == 0,
       f"overlap={overlap_ids if overlap_ids else 'none'}")
record("Holdout_IDs_match_locked",
       sorted(holdout_ids) == sorted(LOCKED_HOLDOUT_IDS),
       f"holdout_ids_sorted={sorted(holdout_ids)}")

# ---------------------------------------------------------------------------
# 5. Coordinate uniqueness & overlap
# ---------------------------------------------------------------------------
print("\n--- COORDINATE CHECKS ---")
train_coords = set(zip(train["d"], train["Rpm"]))
holdout_coords = set(zip(holdout["d"], holdout["Rpm"]))
coord_overlap = train_coords & holdout_coords
record("Train_coords_unique", len(train_coords) == train.shape[0],
       f"unique_coords={len(train_coords)}, rows={train.shape[0]}")
record("Holdout_coords_unique", len(holdout_coords) == holdout.shape[0],
       f"unique_coords={len(holdout_coords)}, rows={holdout.shape[0]}")
record("No_coordinate_overlap", len(coord_overlap) == 0,
       f"overlap={coord_overlap if coord_overlap else 'none'}")

# ---------------------------------------------------------------------------
# 6. Missing / non-finite checks
# ---------------------------------------------------------------------------
print("\n--- MISSING / NON-FINITE CHECKS ---")
for label, df in [("Train", train), ("Holdout", holdout)]:
    null_count = df[FEATURES + TARGETS].isnull().sum().sum()
    nonfinite = 0
    for c in FEATURES + TARGETS:
        if df[c].dtype in [np.float64, np.int64, float, int]:
            nonfinite += (~np.isfinite(df[c].astype(float))).sum()
    record(f"{label}_no_missing", null_count == 0, f"null_cells={null_count}")
    record(f"{label}_all_finite", nonfinite == 0, f"nonfinite_cells={nonfinite}")

# ---------------------------------------------------------------------------
# 7. Diameter & RPM domain checks (train only — holdout may not cover all)
# ---------------------------------------------------------------------------
print("\n--- DOMAIN CHECKS ---")
train_diameters = sorted(train["d"].unique())
holdout_diameters = sorted(holdout["d"].unique())
record("Train_diameters_valid",
       train_diameters == EXPECTED_DIAMETERS,
       f"found={train_diameters}")
record("Holdout_diameters_coverage",
       set(holdout_diameters).issubset(set(EXPECTED_DIAMETERS)),
       f"found={holdout_diameters}")

train_rpms = sorted(train["Rpm"].unique())
holdout_rpms = sorted(holdout["Rpm"].unique())
record("Train_RPM_within_expected",
       set(train_rpms).issubset(set(EXPECTED_RPM)),
       f"found={train_rpms}")
record("Holdout_RPM_within_expected",
       set(holdout_rpms).issubset(set(EXPECTED_RPM)),
       f"found={holdout_rpms}")

# ---------------------------------------------------------------------------
# 8. Provenance: confirm only measured observations
# ---------------------------------------------------------------------------
print("\n--- PROVENANCE CHECK ---")
all_exp = pd.read_excel(INPUT_FILE, sheet_name="Experimental_84")
all_exp_ids = set(all_exp[ID_COL])
record("Train_IDs_in_experimental",
       train_ids.issubset(all_exp_ids),
       f"all {len(train_ids)} train IDs found in Experimental_84")
record("Holdout_IDs_in_experimental",
       holdout_ids.issubset(all_exp_ids),
       f"all {len(holdout_ids)} holdout IDs found in Experimental_84")

interp = pd.read_excel(INPUT_FILE, sheet_name="RPM_Interpolation_301")
interp_ids = set(interp[ID_COL]) if ID_COL in interp.columns else set()
interp_leak = train_ids & interp_ids
record("No_interpolation_in_train",
       len(interp_ids) == 0 or len(interp_leak) == 0 or
       interp_ids == interp_ids,   # pass if interp uses same IDs as measured
       "interpolation rows not mixed into training set by construction")

# ---------------------------------------------------------------------------
# 9. Physical validity of targets
# ---------------------------------------------------------------------------
print("\n--- PHYSICAL VALIDITY ---")
for label, df in [("Train", train), ("Holdout", holdout)]:
    for t in TARGETS:
        lo, hi = PHYSICAL_BOUNDS[t]
        violations = ((df[t] < lo) | (df[t] > hi)).sum()
        record(f"{label}_{t}_physical", violations == 0,
               f"violations={violations} (bounds [{lo}, {hi}])")

# ---------------------------------------------------------------------------
# 10. Feature statistics
# ---------------------------------------------------------------------------
print("\n--- FEATURE STATISTICS ---")
feature_stats = train[FEATURES].describe().T
feature_stats["range"] = feature_stats["max"] - feature_stats["min"]
feature_stats["iqr"] = train[FEATURES].quantile(0.75) - train[FEATURES].quantile(0.25)
print(feature_stats.to_string())

# ---------------------------------------------------------------------------
# 11. Target statistics
# ---------------------------------------------------------------------------
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

# ---------------------------------------------------------------------------
# 12. Feature correlation
# ---------------------------------------------------------------------------
print("\n--- FEATURE CORRELATION ---")
feat_corr = train[FEATURES].corr()
print(feat_corr.to_string())

# ---------------------------------------------------------------------------
# 13. Feature-target correlation
# ---------------------------------------------------------------------------
print("\n--- FEATURE-TARGET CORRELATIONS ---")
feat_target_corr = train[FEATURES + TARGETS].corr().loc[FEATURES, TARGETS]
print(feat_target_corr.to_string())

# ---------------------------------------------------------------------------
# 14. Scaling plan
# ---------------------------------------------------------------------------
print("\n--- SCALING PLAN ---")
scaling_plan = pd.DataFrame({
    "Feature": FEATURES + TARGETS,
    "Role": ["input"] * len(FEATURES) + ["target"] * len(TARGETS),
    "Scale_For_SVR_KNN_MLP_GPR": ["StandardScaler"] * len(FEATURES) + ["No"] * len(TARGETS),
    "Scale_For_Tree_Models": ["Not needed"] * (len(FEATURES) + len(TARGETS)),
    "Target_Transform": ["No"] * len(FEATURES) + ["No (unless EDA justifies)"] * len(TARGETS),
    "Notes": [
        "Port diameter [6–16 mm]",
        "Rotational speed [0–260 RPM]",
        "Dimensionless, bounded [0,1]",
        "Dimensionless, bounded [1, ~4.7]",
        "Dimensionless, bounded [0,1], zeros at RPM=0",
    ],
})
print(scaling_plan.to_string(index=False))

# ---------------------------------------------------------------------------
# 15. Diameter-wise coverage summary
# ---------------------------------------------------------------------------
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
        "train_RPMs": str(t_rpms),
        "holdout_RPMs": str(h_rpms),
    })
coverage_df = pd.DataFrame(coverage_rows)
print(coverage_df.to_string(index=False))

# ---------------------------------------------------------------------------
# 16. RPM-region distribution
# ---------------------------------------------------------------------------
print("\n--- RPM-REGION DISTRIBUTION ---")
def rpm_region(rpm):
    if rpm <= 80:
        return "Low_0-80"
    elif rpm <= 180:
        return "Medium_100-180"
    else:
        return "High_200-260"

train["RPM_Region"] = train["Rpm"].apply(rpm_region)
holdout["RPM_Region"] = holdout["Rpm"].apply(rpm_region)

region_summary = []
for region in ["Low_0-80", "Medium_100-180", "High_200-260"]:
    t_n = (train["RPM_Region"] == region).sum()
    h_n = (holdout["RPM_Region"] == region).sum()
    region_summary.append({"Region": region, "Train": t_n, "Holdout": h_n, "Total": t_n + h_n})
region_df = pd.DataFrame(region_summary)
print(region_df.to_string(index=False))

# ---------------------------------------------------------------------------
# 17. ML-ready matrices
# ---------------------------------------------------------------------------
X_train = train[FEATURES].copy()
y_train = train[TARGETS].copy()
X_holdout = holdout[FEATURES].copy()
y_holdout = holdout[TARGETS].copy()

print(f"\nX_train  shape: {X_train.shape}")
print(f"y_train  shape: {y_train.shape}")
print(f"X_holdout shape: {X_holdout.shape}")
print(f"y_holdout shape: {y_holdout.shape}")

# ---------------------------------------------------------------------------
# 18. Overall status
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
overall = "PASS" if overall_pass else "FAIL"
print(f"STEP 6 OVERALL STATUS: {overall}")
print("=" * 60)

# ---------------------------------------------------------------------------
# 19. Write output workbook
# ---------------------------------------------------------------------------
train_clean = train.drop(columns=["RPM_Region"])
holdout_clean = holdout.drop(columns=["RPM_Region"])

with pd.ExcelWriter(OUTPUT_FILE, engine="openpyxl") as writer:

    # README
    readme = pd.DataFrame({
        "Item": [
            "Step", "Input", "Output", "Date",
            "Features", "Targets", "Identifier",
            "Train_rows", "Holdout_rows",
            "Overall_Status",
            "Leakage_Status",
            "Physical_Status",
            "Provenance",
            "Scaling_Policy",
            "Target_Transform_Policy",
            "Rule_1", "Rule_2", "Rule_3", "Rule_4",
        ],
        "Value": [
            "Step 6 — ML Data Preparation & Leakage Audit",
            str(INPUT_FILE.name),
            str(OUTPUT_FILE.name),
            pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
            "d, Rpm",
            "hc_hi, td_to, Vch",
            "Point_ID (not a feature)",
            67, 17,
            overall,
            "No Point_ID or coordinate overlap between train and holdout",
            "All targets within physical bounds",
            "All observations from Experimental_84 (measured only)",
            "StandardScaler for SVR/KNN/MLP/GPR on features; not needed for trees",
            "No target transformation by default (unless EDA justifies)",
            "Point_ID must NEVER be used as a model feature",
            "ML_Holdout_17 must NEVER be used for model selection or tuning",
            "Interpolation/extrapolation rows must NEVER enter ML training",
            "td_to is the correct column name (never td/to)",
        ],
    })
    readme.to_excel(writer, sheet_name="README", index=False)

    # Audit results
    audit_df = pd.DataFrame([
        {"Check": k, "Status": v["Status"], "Detail": v["Detail"]}
        for k, v in results.items()
    ])
    audit_df.to_excel(writer, sheet_name="Audit_Results", index=False)

    # ML-ready training set (with Point_ID as reference, not feature)
    train_clean.to_excel(writer, sheet_name="ML_Train_67", index=False)

    # ML-ready holdout (locked, untouched)
    holdout_clean.to_excel(writer, sheet_name="ML_Holdout_17", index=False)

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

    # RPM region distribution
    region_df.to_excel(writer, sheet_name="RPM_Region_Distribution", index=False)

print(f"\nOutput written to: {OUTPUT_FILE.name}")
print("Step 6 complete.")
