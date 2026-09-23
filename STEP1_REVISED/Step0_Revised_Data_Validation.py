"""
STEP 0 - REVISED DATA VALIDATION

Purpose
-------
Validate the revised experimental dataset before rebuilding the entire
modelling pipeline.

Revised inputs:
    d
    Rpm

Revised responses:
    hc_hi
    td_to
    Vch

Expected experimental design:
    6 diameters x 14 RPM levels = 84 real observations

This step DOES NOT:
    - split the data
    - interpolate
    - extrapolate
    - fit any model
    - modify experimental values

It only validates the canonical Working_Data table and creates an audit
workbook for Step 1.

Expected input:
    Original.xlsx
    sheet: Working_Data

Expected columns:
    Point_ID | d | Rpm | hc_hi | td/to | Vch
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
OUTPUT_FILE = BASE_DIR / "Step0_Revised_Data_Validation.xlsx"


# =============================================================================
# 2. EXPECTED DATASET DEFINITION
# =============================================================================

REQUIRED_COLUMNS = [
    "Point_ID",
    "d",
    "Rpm",
    "hc_hi",
    "td_to",
    "Vch",
]

EXPECTED_DIAMETERS = [6, 8, 10, 12, 14, 16]
EXPECTED_RPM = list(range(0, 261, 20))

EXPECTED_ROWS = len(EXPECTED_DIAMETERS) * len(EXPECTED_RPM)

TARGET_COLUMNS = [
    "hc_hi",
    "td_to",
    "Vch",
]


# =============================================================================
# 3. HELPER
# =============================================================================

def check(condition, passed_message, failed_message):
    return {
        "Status": "PASS" if condition else "FAIL",
        "Check": passed_message if condition else failed_message,
    }


# =============================================================================
# 4. LOAD DATA
# =============================================================================

if not INPUT_FILE.exists():
    raise FileNotFoundError(
        f"Input file not found:\n{INPUT_FILE}"
    )

xls = pd.ExcelFile(INPUT_FILE)

if INPUT_SHEET not in xls.sheet_names:
    raise ValueError(
        f"Required sheet '{INPUT_SHEET}' not found in {INPUT_FILE.name}.\n"
        f"Available sheets: {xls.sheet_names}"
    )

df = pd.read_excel(
    INPUT_FILE,
    sheet_name=INPUT_SHEET
)

print("=" * 78)
print("STEP 0 - REVISED DATA VALIDATION")
print("=" * 78)
print(f"Input:  {INPUT_FILE}")
print(f"Sheet:  {INPUT_SHEET}")
print()


# =============================================================================
# 5. COLUMN CHECK
# =============================================================================

missing_columns = [
    col for col in REQUIRED_COLUMNS
    if col not in df.columns
]

if missing_columns:
    raise ValueError(
        "Required columns are missing:\n"
        + "\n".join(f"  - {col}" for col in missing_columns)
    )

# Keep only canonical columns for the modelling table.
working = df[REQUIRED_COLUMNS].copy()

# =============================================================================
# 6. HANDLE DIAMETER BLANKS SAFELY
# =============================================================================
# Some source spreadsheets store d only on the first row of a diameter block.
# We allow this representation and forward-fill d ONLY for blank cells.
# Existing nonblank d values are never changed.

d_before = int(working["d"].isna().sum())

if d_before > 0:
    working["d"] = working["d"].ffill()

d_after = int(working["d"].isna().sum())

if d_after > 0:
    raise ValueError(
        "Diameter column still contains blank values after forward-fill."
    )


# =============================================================================
# 7. NUMERIC CONVERSION
# =============================================================================

numeric_columns = [
    "Point_ID",
    "d",
    "Rpm",
    "hc_hi",
    "td_to",
    "Vch",
]

conversion_errors = []

for col in numeric_columns:
    original = working[col].copy()
    converted = pd.to_numeric(original, errors="coerce")

    invalid = converted.isna() & original.notna()

    if invalid.any():
        conversion_errors.append(
            {
                "Column": col,
                "Invalid_Numeric_Values": int(invalid.sum()),
            }
        )

    working[col] = converted

if conversion_errors:
    conversion_error_df = pd.DataFrame(conversion_errors)
    raise ValueError(
        "Non-numeric values were detected in required numeric columns:\n"
        + conversion_error_df.to_string(index=False)
    )


# =============================================================================
# 8. BASIC SIZE CHECKS
# =============================================================================

basic_checks = []

basic_checks.append({
    "Check": "Expected row count",
    "Expected": EXPECTED_ROWS,
    "Actual": len(working),
    "Status": "PASS" if len(working) == EXPECTED_ROWS else "FAIL",
})

basic_checks.append({
    "Check": "Expected number of diameters",
    "Expected": len(EXPECTED_DIAMETERS),
    "Actual": working["d"].nunique(),
    "Status": (
        "PASS"
        if working["d"].nunique() == len(EXPECTED_DIAMETERS)
        else "FAIL"
    ),
})

basic_checks.append({
    "Check": "Expected number of RPM levels",
    "Expected": len(EXPECTED_RPM),
    "Actual": working["Rpm"].nunique(),
    "Status": (
        "PASS"
        if working["Rpm"].nunique() == len(EXPECTED_RPM)
        else "FAIL"
    ),
})


# =============================================================================
# 9. UNIQUE VALUES
# =============================================================================

actual_diameters = sorted(
    working["d"].dropna().unique().tolist()
)

actual_rpm = sorted(
    working["Rpm"].dropna().unique().tolist()
)

unique_values = pd.DataFrame([
    {
        "Variable": "d",
        "Expected": ", ".join(map(str, EXPECTED_DIAMETERS)),
        "Actual": ", ".join(map(str, actual_diameters)),
        "Exact_Match": actual_diameters == EXPECTED_DIAMETERS,
    },
    {
        "Variable": "Rpm",
        "Expected": ", ".join(map(str, EXPECTED_RPM)),
        "Actual": ", ".join(map(str, actual_rpm)),
        "Exact_Match": actual_rpm == EXPECTED_RPM,
    },
])


# =============================================================================
# 10. MISSING-VALUE CHECK
# =============================================================================

missing_rows = []

for col in REQUIRED_COLUMNS:
    missing_rows.append({
        "Column": col,
        "Missing_Count": int(working[col].isna().sum()),
        "Missing_Percent": (
            float(working[col].isna().mean() * 100.0)
        ),
    })

missing_df = pd.DataFrame(missing_rows)


# =============================================================================
# 11. POINT_ID DUPLICATE CHECK
# =============================================================================

duplicate_point_id = working[
    working["Point_ID"].duplicated(keep=False)
].sort_values("Point_ID")

point_id_check = pd.DataFrame([{
    "Expected_Duplicates": 0,
    "Duplicate_Point_ID_Rows": len(duplicate_point_id),
    "Status": "PASS" if len(duplicate_point_id) == 0 else "FAIL",
}])


# =============================================================================
# 12. (d, RPM) DUPLICATE CHECK
# =============================================================================

duplicate_coordinates = working[
    working.duplicated(["d", "Rpm"], keep=False)
].sort_values(["d", "Rpm"])

coordinate_duplicate_check = pd.DataFrame([{
    "Expected_Duplicates": 0,
    "Duplicate_Coordinate_Rows": len(duplicate_coordinates),
    "Status": (
        "PASS"
        if len(duplicate_coordinates) == 0
        else "FAIL"
    ),
}])


# =============================================================================
# 13. COMPLETE FACTORIAL GRID CHECK
# =============================================================================

expected_coordinates = pd.MultiIndex.from_product(
    [EXPECTED_DIAMETERS, EXPECTED_RPM],
    names=["d", "Rpm"]
)

actual_coordinates = pd.MultiIndex.from_frame(
    working[["d", "Rpm"]]
)

missing_coordinates = expected_coordinates.difference(
    actual_coordinates
)

unexpected_coordinates = actual_coordinates.difference(
    expected_coordinates
)

grid_check = pd.DataFrame([{
    "Expected_Coordinates": len(expected_coordinates),
    "Actual_Unique_Coordinates": working[["d", "Rpm"]].drop_duplicates().shape[0],
    "Missing_Expected_Coordinates": len(missing_coordinates),
    "Unexpected_Coordinates": len(unexpected_coordinates),
    "Complete_6x14_Grid": (
        len(missing_coordinates) == 0
        and len(unexpected_coordinates) == 0
        and working[["d", "Rpm"]].drop_duplicates().shape[0] == EXPECTED_ROWS
    ),
}])


missing_coordinates_df = pd.DataFrame(
    list(missing_coordinates),
    columns=["d", "Rpm"]
)

unexpected_coordinates_df = pd.DataFrame(
    list(unexpected_coordinates),
    columns=["d", "Rpm"]
)


# =============================================================================
# 14. TARGET FINITENESS / PHYSICAL CHECK
# =============================================================================

target_rows = []

for target in TARGET_COLUMNS:
    values = working[target].to_numpy(dtype=float)

    finite = np.isfinite(values)

    target_rows.append({
        "Target": target,
        "N": len(values),
        "Nonfinite": int(np.sum(~finite)),
        "Negative": int(np.sum(values[finite] < 0)),
        "Minimum": (
            float(np.min(values[finite]))
            if np.any(finite)
            else np.nan
        ),
        "Maximum": (
            float(np.max(values[finite]))
            if np.any(finite)
            else np.nan
        ),
        "All_Finite": bool(np.all(finite)),
        "All_Nonnegative": bool(
            np.all(values[finite] >= 0)
            if np.any(finite)
            else False
        ),
    })

target_physical_check = pd.DataFrame(target_rows)


# =============================================================================
# 15. RPM = 0 SANITY CHECK
# =============================================================================

zero_rpm = working[working["Rpm"] == 0].copy()

if len(zero_rpm) > 0:
    td_to_zero = zero_rpm["td_to"].to_numpy(dtype=float)
    vch_zero = zero_rpm["Vch"].to_numpy(dtype=float)
    hc_hi_zero = zero_rpm["hc_hi"].to_numpy(dtype=float)

    # td/to is expected to be approximately 1.
    td_to_close_to_one = np.isclose(
        td_to_zero,
        1.0,
        atol=1e-8,
        rtol=1e-8,
    )

    # Vch is expected to be 0 at no rotation.
    vch_close_to_zero = np.isclose(
        vch_zero,
        0.0,
        atol=1e-12,
        rtol=0.0,
    )

    zero_rpm_summary = pd.DataFrame([{
        "Zero_RPM_Rows": len(zero_rpm),
        "td_to_All_Approximately_1": bool(
            np.all(td_to_close_to_one)
        ),
        "td_to_Min": float(np.min(td_to_zero)),
        "td_to_Max": float(np.max(td_to_zero)),
        "Vch_All_Approximately_0": bool(
            np.all(vch_close_to_zero)
        ),
        "Vch_Min": float(np.min(vch_zero)),
        "Vch_Max": float(np.max(vch_zero)),
        "hc_hi_Min": float(np.min(hc_hi_zero)),
        "hc_hi_Max": float(np.max(hc_hi_zero)),
        "hc_hi_Zero_or_NearZero_Count": int(
            np.sum(np.isclose(hc_hi_zero, 0.0, atol=1e-12))
        ),
        "hc_hi_Nonzero_Count": int(
            np.sum(~np.isclose(hc_hi_zero, 0.0, atol=1e-12))
        ),
    }])
else:
    zero_rpm_summary = pd.DataFrame([{
        "Zero_RPM_Rows": 0,
        "td_to_All_Approximately_1": False,
        "td_to_Min": np.nan,
        "td_to_Max": np.nan,
        "Vch_All_Approximately_0": False,
        "Vch_Min": np.nan,
        "Vch_Max": np.nan,
        "hc_hi_Min": np.nan,
        "hc_hi_Max": np.nan,
        "hc_hi_Zero_or_NearZero_Count": 0,
        "hc_hi_Nonzero_Count": 0,
    }])


# =============================================================================
# 16. DIAMETER/RPM REPRESENTATION
# =============================================================================

diameter_summary = (
    working.groupby("d")
    .agg(
        Rows=("Point_ID", "size"),
        RPM_Min=("Rpm", "min"),
        RPM_Max=("Rpm", "max"),
        RPM_Levels=("Rpm", "nunique"),
    )
    .reset_index()
)

rpm_summary = (
    working.groupby("Rpm")
    .agg(
        Rows=("Point_ID", "size"),
        Diameter_Min=("d", "min"),
        Diameter_Max=("d", "max"),
        Diameter_Levels=("d", "nunique"),
    )
    .reset_index()
)


# =============================================================================
# 17. OVERALL STATUS
# =============================================================================

all_pass = (
    len(working) == EXPECTED_ROWS
    and actual_diameters == EXPECTED_DIAMETERS
    and actual_rpm == EXPECTED_RPM
    and missing_df["Missing_Count"].sum() == 0
    and len(duplicate_point_id) == 0
    and len(duplicate_coordinates) == 0
    and len(missing_coordinates) == 0
    and len(unexpected_coordinates) == 0
    and target_physical_check["All_Finite"].all()
    and target_physical_check["All_Nonnegative"].all()
    and bool(zero_rpm_summary.loc[
        0, "td_to_All_Approximately_1"
    ])
    and bool(zero_rpm_summary.loc[
        0, "Vch_All_Approximately_0"
    ])
)

overall_status = pd.DataFrame([{
    "Step": "Step 0 - Revised Data Validation",
    "Overall_Status": "PASS" if all_pass else "REVIEW_REQUIRED",
    "Expected_Rows": EXPECTED_ROWS,
    "Actual_Rows": len(working),
    "Expected_Diameters": len(EXPECTED_DIAMETERS),
    "Actual_Diameters": len(actual_diameters),
    "Expected_RPM_Levels": len(EXPECTED_RPM),
    "Actual_RPM_Levels": len(actual_rpm),
    "Point_ID_Duplicates": len(duplicate_point_id),
    "Coordinate_Duplicates": len(duplicate_coordinates),
    "Missing_Grid_Coordinates": len(missing_coordinates),
    "Unexpected_Coordinates": len(unexpected_coordinates),
    "Target_Nonfinite_Count": int(
        target_physical_check["Nonfinite"].sum()
    ),
    "Target_Negative_Count": int(
        target_physical_check["Negative"].sum()
    ),
    "RPM0_td_to_Check": bool(
        zero_rpm_summary.loc[0, "td_to_All_Approximately_1"]
    ),
    "RPM0_Vch_Check": bool(
        zero_rpm_summary.loc[0, "Vch_All_Approximately_0"]
    ),
    "RPM0_hc_hi_Nonzero_Count": int(
        zero_rpm_summary.loc[0, "hc_hi_Nonzero_Count"]
    ),
}])


# =============================================================================
# 18. METHOD / TRACEABILITY NOTES
# =============================================================================

notes = pd.DataFrame([
    [
        "Input workbook",
        INPUT_FILE.name,
    ],
    [
        "Input sheet",
        INPUT_SHEET,
    ],
    [
        "Canonical inputs",
        "d, Rpm",
    ],
    [
        "Canonical responses",
        "hc_hi, td/to, Vch",
    ],
    [
        "Expected design",
        "6 diameters × 14 RPM = 84 real observations",
    ],
    [
        "Interpolation/extrapolation",
        "Not performed in Step 0",
    ],
    [
        "Model fitting",
        "Not performed in Step 0",
    ],
    [
        "Train/test split",
        "Not performed in Step 0",
    ],
    [
        "Diameter blanks",
        (
            "Blank d cells are forward-filled only if present in the source "
            "Working_Data sheet; existing values are not changed."
        ),
    ],
    [
        "hc_hi at RPM=0",
        (
            "Reported as observed; no automatic correction is applied."
        ),
    ],
    [
        "td/to at RPM=0",
        "Expected approximately 1.0",
    ],
    [
        "Vch at RPM=0",
        "Expected approximately 0.0",
    ],
    [
        "Next step",
        "Step 1 - locked 67/17 experimental split",
    ],
])

notes.columns = ["Item", "Value"]


# =============================================================================
# 19. WRITE OUTPUT WORKBOOK
# =============================================================================

with pd.ExcelWriter(
    OUTPUT_FILE,
    engine="openpyxl"
) as writer:

    working.to_excel(
        writer,
        sheet_name="Working_Data",
        index=False
    )

    overall_status.to_excel(
        writer,
        sheet_name="Overall_Status",
        index=False
    )

    pd.DataFrame(basic_checks).to_excel(
        writer,
        sheet_name="Basic_Checks",
        index=False
    )

    unique_values.to_excel(
        writer,
        sheet_name="Unique_Values",
        index=False
    )

    missing_df.to_excel(
        writer,
        sheet_name="Missing_Check",
        index=False
    )

    point_id_check.to_excel(
        writer,
        sheet_name="PointID_Check",
        index=False
    )

    duplicate_point_id.to_excel(
        writer,
        sheet_name="Duplicate_PointIDs",
        index=False
    )

    coordinate_duplicate_check.to_excel(
        writer,
        sheet_name="Coordinate_Check",
        index=False
    )

    duplicate_coordinates.to_excel(
        writer,
        sheet_name="Duplicate_Coordinates",
        index=False
    )

    grid_check.to_excel(
        writer,
        sheet_name="Grid_Check",
        index=False
    )

    missing_coordinates_df.to_excel(
        writer,
        sheet_name="Missing_Coordinates",
        index=False
    )

    unexpected_coordinates_df.to_excel(
        writer,
        sheet_name="Unexpected_Coordinates",
        index=False
    )

    target_physical_check.to_excel(
        writer,
        sheet_name="Target_Physical_Check",
        index=False
    )

    zero_rpm_summary.to_excel(
        writer,
        sheet_name="RPM0_Sanity_Check",
        index=False
    )

    zero_rpm.to_excel(
        writer,
        sheet_name="RPM0_Data",
        index=False
    )

    diameter_summary.to_excel(
        writer,
        sheet_name="Diameter_Summary",
        index=False
    )

    rpm_summary.to_excel(
        writer,
        sheet_name="RPM_Summary",
        index=False
    )

    notes.to_excel(
        writer,
        sheet_name="Method_Notes",
        index=False
    )


# =============================================================================
# 20. CONSOLE SUMMARY
# =============================================================================

print("-" * 78)
print("DATASET")
print("-" * 78)
print(f"Rows:          {len(working)}")
print(f"Diameters:     {actual_diameters}")
print(f"RPM levels:    {actual_rpm}")
print(f"Unique coords: {working[['d', 'Rpm']].drop_duplicates().shape[0]}")
print()

print("-" * 78)
print("TARGETS")
print("-" * 78)
print(target_physical_check.to_string(index=False))
print()

print("-" * 78)
print("RPM = 0 SANITY CHECK")
print("-" * 78)
print(zero_rpm_summary.to_string(index=False))
print()

print("-" * 78)
print("GRID CHECK")
print("-" * 78)
print(grid_check.to_string(index=False))
print()

print("-" * 78)
print("OVERALL STATUS")
print("-" * 78)
print(overall_status.to_string(index=False))
print()

print(f"Output: {OUTPUT_FILE}")

if all_pass:
    print()
    print("STEP 0 PASSED.")
    print("Next step: Step 1 - create the locked 67/17 experimental split.")
else:
    print()
    print("STEP 0 REQUIRES REVIEW.")
    print("Do NOT proceed to Step 1 until the failed checks are resolved.")

print("=" * 78)
