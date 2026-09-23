"""
STEP 1 - REVISED LEAKAGE-SAFE EXPERIMENTAL TRAIN/TEST SPLIT

Purpose
-------
Create the locked 67/17 train-test split for the revised response variables:

Inputs:
    d
    Rpm

Responses:
    hc_hi
    td_to
    Vch

Important
---------
The split is response-independent. The same experimental coordinates used in
the previous project branch are retained so that the new response formulation
can be compared consistently with the earlier analysis.

Locked test Point_IDs:
    71, 79, 83,
    59, 64, 70,
    43, 51, 53,
    29, 36, 42,
    18, 23, 27,
    4, 8

These 17 observations must not be used for:
    - interpolation construction
    - extrapolation model fitting
    - ML training
    - hyperparameter tuning
    - model selection using test performance

Expected:
    67 Train_Anchors
    17 Test_Holdout
"""

from pathlib import Path
import numpy as np
import pandas as pd


# =============================================================================
# 1. PATHS
# =============================================================================

BASE_DIR = Path(__file__).resolve().parent

INPUT_FILE = BASE_DIR / "Original.xlsx"
INPUT_SHEET = "Working_Data"
OUTPUT_FILE = BASE_DIR / "Step1_Stratified_Split.xlsx"


# =============================================================================
# 2. DATASET DEFINITION
# =============================================================================

REQUIRED_COLUMNS = [
    "Point_ID",
    "d",
    "Rpm",
    "hc_hi",
    "td_to",
    "Vch",
]

TARGET_COLUMNS = [
    "hc_hi",
    "td_to",
    "Vch",
]

EXPECTED_TRAIN = 67
EXPECTED_TEST = 17

LOCKED_TEST_POINT_IDS = [
    71, 79, 83,
    59, 64, 70,
    43, 51, 53,
    29, 36, 42,
    18, 23, 27,
    4, 8,
]

EXPECTED_DIAMETERS = [6, 8, 10, 12, 14, 16]


# =============================================================================
# 3. LOAD DATA
# =============================================================================

if not INPUT_FILE.exists():
    raise FileNotFoundError(
        f"Input file not found:\n{INPUT_FILE}"
    )

xls = pd.ExcelFile(INPUT_FILE)

if INPUT_SHEET not in xls.sheet_names:
    raise ValueError(
        f"Sheet '{INPUT_SHEET}' not found.\n"
        f"Available sheets: {xls.sheet_names}"
    )

df = pd.read_excel(
    INPUT_FILE,
    sheet_name=INPUT_SHEET
)

missing = [
    col for col in REQUIRED_COLUMNS
    if col not in df.columns
]

if missing:
    raise ValueError(
        f"Missing required columns: {missing}"
    )

df = df[REQUIRED_COLUMNS].copy()


# =============================================================================
# 4. NUMERIC VALIDATION
# =============================================================================

for col in REQUIRED_COLUMNS:
    df[col] = pd.to_numeric(df[col], errors="raise")

if df[REQUIRED_COLUMNS].isna().any().any():
    raise ValueError(
        "Missing values detected in the canonical Working_Data table."
    )

if len(df) != 84:
    raise ValueError(
        f"Expected 84 observations, found {len(df)}."
    )

if df["Point_ID"].duplicated().any():
    raise ValueError(
        "Duplicate Point_ID values detected."
    )

if df[["d", "Rpm"]].duplicated().any():
    raise ValueError(
        "Duplicate (d, RPM) coordinates detected."
    )

if sorted(df["d"].unique().tolist()) != EXPECTED_DIAMETERS:
    raise ValueError(
        f"Unexpected diameter values: {sorted(df['d'].unique().tolist())}"
    )


# =============================================================================
# 5. CHECK LOCKED TEST IDS
# =============================================================================

available_ids = set(df["Point_ID"].astype(int).tolist())
locked_ids = set(LOCKED_TEST_POINT_IDS)

missing_locked_ids = sorted(locked_ids - available_ids)

if missing_locked_ids:
    raise ValueError(
        f"Locked test Point_IDs are missing from Working_Data: "
        f"{missing_locked_ids}"
    )

if len(LOCKED_TEST_POINT_IDS) != EXPECTED_TEST:
    raise ValueError(
        "The locked test ID list does not contain exactly 17 IDs."
    )


# =============================================================================
# 6. CREATE SPLIT
# =============================================================================

test_df = (
    df[df["Point_ID"].isin(LOCKED_TEST_POINT_IDS)]
    .copy()
    .sort_values(["d", "Rpm", "Point_ID"])
    .reset_index(drop=True)
)

train_df = (
    df[~df["Point_ID"].isin(LOCKED_TEST_POINT_IDS)]
    .copy()
    .sort_values(["d", "Rpm", "Point_ID"])
    .reset_index(drop=True)
)

if len(train_df) != EXPECTED_TRAIN:
    raise ValueError(
        f"Expected {EXPECTED_TRAIN} training points, found {len(train_df)}."
    )

if len(test_df) != EXPECTED_TEST:
    raise ValueError(
        f"Expected {EXPECTED_TEST} test points, found {len(test_df)}."
    )


# =============================================================================
# 7. LEAKAGE CHECKS
# =============================================================================

train_ids = set(train_df["Point_ID"].astype(int))
test_ids = set(test_df["Point_ID"].astype(int))

id_overlap = train_ids.intersection(test_ids)

train_coords = set(zip(train_df["d"], train_df["Rpm"]))
test_coords = set(zip(test_df["d"], test_df["Rpm"]))

coordinate_overlap = train_coords.intersection(test_coords)

if id_overlap:
    raise RuntimeError(
        f"Point_ID leakage detected: {sorted(id_overlap)}"
    )

if coordinate_overlap:
    raise RuntimeError(
        f"Coordinate leakage detected: {sorted(coordinate_overlap)}"
    )


# =============================================================================
# 8. DIAMETER REPRESENTATION
# =============================================================================

diameter_check_rows = []

for d in EXPECTED_DIAMETERS:
    tr = train_df[train_df["d"] == d]
    te = test_df[test_df["d"] == d]

    diameter_check_rows.append({
        "d": d,
        "Train_Count": len(tr),
        "Test_Count": len(te),
        "Total_Count": len(tr) + len(te),
        "Train_RPM_Min": tr["Rpm"].min() if len(tr) else np.nan,
        "Train_RPM_Max": tr["Rpm"].max() if len(tr) else np.nan,
        "Test_RPM_Min": te["Rpm"].min() if len(te) else np.nan,
        "Test_RPM_Max": te["Rpm"].max() if len(te) else np.nan,
        "Diameter_Represented_In_Train": len(tr) > 0,
        "Diameter_Represented_In_Test": len(te) > 0,
    })

diameter_check = pd.DataFrame(diameter_check_rows)


# =============================================================================
# 9. RPM REGION REPRESENTATION
# =============================================================================

def rpm_region(rpm):
    if rpm <= 80:
        return "Low"
    if rpm <= 180:
        return "Medium"
    return "High"


all_region = df["Rpm"].apply(rpm_region)
train_region = train_df["Rpm"].apply(rpm_region)
test_region = test_df["Rpm"].apply(rpm_region)

rpm_check_rows = []

for region in ["Low", "Medium", "High"]:
    rpm_check_rows.append({
        "RPM_Region": region,
        "All_Count": int((all_region == region).sum()),
        "Train_Count": int((train_region == region).sum()),
        "Test_Count": int((test_region == region).sum()),
    })

rpm_check = pd.DataFrame(rpm_check_rows)


# =============================================================================
# 10. TEST POINT AUDIT
# =============================================================================

test_point_ids = (
    test_df[
        ["Point_ID", "d", "Rpm", "hc_hi", "td_to", "Vch"]
    ]
    .sort_values(["d", "Rpm", "Point_ID"])
    .reset_index(drop=True)
)

test_point_ids["RPM_Region"] = test_point_ids["Rpm"].apply(rpm_region)


# =============================================================================
# 11. TARGET SUMMARY
# =============================================================================

target_summary_rows = []

for target in TARGET_COLUMNS:
    target_summary_rows.append({
        "Target": target,
        "Train_N": len(train_df),
        "Test_N": len(test_df),
        "Train_Min": float(train_df[target].min()),
        "Train_Max": float(train_df[target].max()),
        "Test_Min": float(test_df[target].min()),
        "Test_Max": float(test_df[target].max()),
        "Overall_Min": float(df[target].min()),
        "Overall_Max": float(df[target].max()),
    })

target_summary = pd.DataFrame(target_summary_rows)


# =============================================================================
# 12. 0-RPM AUDIT
# =============================================================================

zero_rpm_train = train_df[train_df["Rpm"] == 0].copy()
zero_rpm_test = test_df[test_df["Rpm"] == 0].copy()

zero_rpm_audit = pd.DataFrame([
    {
        "Dataset": "Train_Anchors",
        "Rows_at_RPM_0": len(zero_rpm_train),
        "td_to_Min": zero_rpm_train["td_to"].min(),
        "td_to_Max": zero_rpm_train["td_to"].max(),
        "Vch_Min": zero_rpm_train["Vch"].min(),
        "Vch_Max": zero_rpm_train["Vch"].max(),
        "hc_hi_Min": zero_rpm_train["hc_hi"].min(),
        "hc_hi_Max": zero_rpm_train["hc_hi"].max(),
    },
    {
        "Dataset": "Test_Holdout",
        "Rows_at_RPM_0": len(zero_rpm_test),
        "td_to_Min": zero_rpm_test["td_to"].min(),
        "td_to_Max": zero_rpm_test["td_to"].max(),
        "Vch_Min": zero_rpm_test["Vch"].min(),
        "Vch_Max": zero_rpm_test["Vch"].max(),
        "hc_hi_Min": zero_rpm_test["hc_hi"].min(),
        "hc_hi_Max": zero_rpm_test["hc_hi"].max(),
    },
])


# =============================================================================
# 13. SPLIT LOG
# =============================================================================

split_log = pd.DataFrame([
    ["Input observations", len(df)],
    ["Training anchors", len(train_df)],
    ["Experimental holdout", len(test_df)],
    ["Holdout fraction_percent", len(test_df) / len(df) * 100.0],
    ["Split basis", "Fixed experimental (d, RPM) coordinates"],
    ["Random seed", "Not used; locked Point_ID list retained"],
    ["Test Point_IDs", ", ".join(map(str, LOCKED_TEST_POINT_IDS))],
    ["Target variables", ", ".join(TARGET_COLUMNS)],
    ["Synthetic interpolation used", "No"],
    ["Extrapolation used", "No"],
    ["Test-coordinate leakage", len(coordinate_overlap)],
    ["Point_ID leakage", len(id_overlap)],
    ["Status", "LOCKED" if not id_overlap and not coordinate_overlap else "FAILED"],
], columns=["Item", "Value"])


# =============================================================================
# 14. VALIDATION STATUS
# =============================================================================

overall_status = pd.DataFrame([{
    "Check": "Revised Step 1 split",
    "Status": (
        "PASS"
        if (
            len(train_df) == 67
            and len(test_df) == 17
            and len(id_overlap) == 0
            and len(coordinate_overlap) == 0
            and len(diameter_check[~diameter_check["Diameter_Represented_In_Train"]]) == 0
        )
        else "FAIL"
    ),
    "Train_Anchors": len(train_df),
    "Test_Holdout": len(test_df),
    "PointID_Overlap": len(id_overlap),
    "Coordinate_Overlap": len(coordinate_overlap),
}])


# =============================================================================
# 15. WRITE WORKBOOK
# =============================================================================

with pd.ExcelWriter(
    OUTPUT_FILE,
    engine="openpyxl"
) as writer:

    df.to_excel(
        writer,
        sheet_name="Working_Data",
        index=False
    )

    train_df.to_excel(
        writer,
        sheet_name="Train_Anchors",
        index=False
    )

    test_df.to_excel(
        writer,
        sheet_name="Test_Holdout",
        index=False
    )

    test_point_ids.to_excel(
        writer,
        sheet_name="Test_Point_IDs",
        index=False
    )

    split_log.to_excel(
        writer,
        sheet_name="Split_Log",
        index=False
    )

    diameter_check.to_excel(
        writer,
        sheet_name="Diameter_Check",
        index=False
    )

    rpm_check.to_excel(
        writer,
        sheet_name="RPM_Check",
        index=False
    )

    target_summary.to_excel(
        writer,
        sheet_name="Target_Summary",
        index=False
    )

    zero_rpm_audit.to_excel(
        writer,
        sheet_name="RPM0_Audit",
        index=False
    )

    overall_status.to_excel(
        writer,
        sheet_name="Overall_Status",
        index=False
    )


# =============================================================================
# 16. CONSOLE SUMMARY
# =============================================================================

print("=" * 78)
print("STEP 1 - REVISED LEAKAGE-SAFE EXPERIMENTAL TRAIN/TEST SPLIT")
print("=" * 78)
print()

print(f"Input:  {INPUT_FILE}")
print(f"Output: {OUTPUT_FILE}")
print()

print("-" * 78)
print("SPLIT")
print("-" * 78)
print(f"Total real observations : {len(df)}")
print(f"Train_Anchors           : {len(train_df)}")
print(f"Test_Holdout            : {len(test_df)}")
print(f"Holdout fraction        : {len(test_df) / len(df) * 100.0:.2f}%")
print()

print("Locked Test Point_IDs:")
print(LOCKED_TEST_POINT_IDS)
print()

print("-" * 78)
print("DIAMETER CHECK")
print("-" * 78)
print(diameter_check.to_string(index=False))
print()

print("-" * 78)
print("RPM REGION CHECK")
print("-" * 78)
print(rpm_check.to_string(index=False))
print()

print("-" * 78)
print("TARGET SUMMARY")
print("-" * 78)
print(target_summary.to_string(index=False))
print()

print("-" * 78)
print("LEAKAGE CHECK")
print("-" * 78)
print(f"Point_ID overlap   : {len(id_overlap)}")
print(f"Coordinate overlap : {len(coordinate_overlap)}")
print()

print("-" * 78)
print("OVERALL STATUS")
print("-" * 78)
print(overall_status.to_string(index=False))
print()

if overall_status.loc[0, "Status"] == "PASS":
    print("STEP 1 PASSED AND SPLIT IS LOCKED.")
    print("Next step: Step 2 - revised LOOCV method selection.")
else:
    print("STEP 1 FAILED. Do not proceed until the checks are resolved.")

print("=" * 78)
