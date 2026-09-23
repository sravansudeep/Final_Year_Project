"""
Step 3A - RPM-only interpolation at 5 RPM intervals

Purpose
-------
Create a leakage-controlled synthetic training dataset by interpolating ONLY
along RPM, while keeping diameter fixed at the six experimentally tested
diameters:

    d = 6, 8, 10, 12, 14, 16 mm

RPM is refined to 5 RPM intervals wherever interpolation is supported by the
TRAINING ANCHORS only.

Important:
- The 17 locked Test_Holdout observations are NEVER used to fit interpolation.
- Exact Test_Holdout (d, RPM) coordinates are excluded from the synthetic data.
- No diameter interpolation is performed in this step.
- No RPM extrapolation is performed in this step.
- RPM extrapolation and diameter extrapolation are handled later as separate
  steps.

Input
-----
Step1_Stratified_Split.xlsx
  Sheet: Train_Anchors
  Sheet: Test_Holdout

Output
------
Step3A_RPM_Interpolation.xlsx
  - hc_Expanded
  - A_Expanded
  - Vmin_Expanded
  - Expansion_Log
  - RPM_Coverage

Interpolation
-------------
For each fixed diameter:
    response = f(RPM)

A 1-D cubic interpolator is fitted using ONLY the available training anchors
at that diameter.

This is deliberately a 1-D RPM interpolation rather than 2-D griddata because
diameter is being held fixed in this stage.

For a diameter, interpolation is generated only between the minimum and
maximum RPM available among its TRAINING anchors. Thus this step never
extrapolates.

Data_Type:
    MEASURED
    SYNTHETIC_IN_RANGE

Generation_Method:
    "Measured training anchor"
    "1D cubic RPM interpolation"

The 17 test coordinates remain completely untouched.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.interpolate import CubicSpline


# ---------------------------------------------------------------------------
# 1. File paths
# ---------------------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent

INPUT_FILE = BASE_DIR / "Step1_Stratified_Split.xlsx"
OUTPUT_FILE = BASE_DIR / "Step3A_RPM_Interpolation.xlsx"


# ---------------------------------------------------------------------------
# 2. Configuration
# ---------------------------------------------------------------------------

RPM_STEP = 5

REQUIRED_COLUMNS = [
    "Point_ID",
    "d",
    "Rpm",
    "hc",
    "A",
    "Vmin",
]

TARGETS = ["hc", "A", "Vmin"]


# ---------------------------------------------------------------------------
# 3. Load locked train/test split
# ---------------------------------------------------------------------------

if not INPUT_FILE.exists():
    raise FileNotFoundError(
        f"Could not find:\n{INPUT_FILE}\n\n"
        "Place this script in the same folder as Step1_Stratified_Split.xlsx."
    )

train = pd.read_excel(INPUT_FILE, sheet_name="Train_Anchors")
test = pd.read_excel(INPUT_FILE, sheet_name="Test_Holdout")

missing_train = [c for c in REQUIRED_COLUMNS if c not in train.columns]
missing_test = [c for c in REQUIRED_COLUMNS if c not in test.columns]

if missing_train:
    raise ValueError(f"Train_Anchors is missing columns: {missing_train}")

if missing_test:
    raise ValueError(f"Test_Holdout is missing columns: {missing_test}")


# Standardise numeric columns
for df in [train, test]:
    df["d"] = pd.to_numeric(df["d"])
    df["Rpm"] = pd.to_numeric(df["Rpm"])
    for target in TARGETS:
        df[target] = pd.to_numeric(df[target])


# ---------------------------------------------------------------------------
# 4. Lock test coordinates
# ---------------------------------------------------------------------------

test_coordinates = set(
    zip(
        test["d"].astype(float),
        test["Rpm"].astype(float),
    )
)


# ---------------------------------------------------------------------------
# 5. Validation checks
# ---------------------------------------------------------------------------

if len(train) != 67:
    raise ValueError(
        f"Expected 67 training anchors, found {len(train)}."
    )

if len(test) != 17:
    raise ValueError(
        f"Expected 17 test observations, found {len(test)}."
    )

if train.duplicated(subset=["d", "Rpm"]).any():
    raise ValueError("Duplicate (d, RPM) combinations found in Train_Anchors.")

if test.duplicated(subset=["d", "Rpm"]).any():
    raise ValueError("Duplicate (d, RPM) combinations found in Test_Holdout.")


# ---------------------------------------------------------------------------
# 6. Generate 5-RPM interpolation for each fixed diameter
# ---------------------------------------------------------------------------

expanded = {}
logs = []
coverage_rows = []

diameters = sorted(train["d"].unique())

for target in TARGETS:

    rows = []

    for d in diameters:

        subset = (
            train[train["d"] == d]
            .sort_values("Rpm")
            .copy()
        )

        x = subset["Rpm"].to_numpy(dtype=float)
        y = subset[target].to_numpy(dtype=float)

        # Cubic interpolation needs at least 4 distinct points.
        if len(x) < 4:
            raise ValueError(
                f"Diameter {d} has only {len(x)} training anchors for {target}. "
                "At least 4 are required for cubic interpolation."
            )

        if len(np.unique(x)) != len(x):
            raise ValueError(
                f"Duplicate RPM values found for diameter {d}, target {target}."
            )

        # Add the measured training anchors first.
        for _, row in subset.iterrows():
            rows.append({
                "Point_ID": row["Point_ID"],
                "d": float(row["d"]),
                "Rpm": float(row["Rpm"]),
                target: float(row[target]),
                "Data_Type": "MEASURED",
                "Generation_Method": "Measured training anchor",
            })

        # Fit ONLY to training anchors.
        spline = CubicSpline(
            x,
            y,
            bc_type="not-a-knot",
            extrapolate=False,
        )

        # Only interpolate inside the training-anchor RPM range.
        rpm_grid = np.arange(
            x.min(),
            x.max() + RPM_STEP,
            RPM_STEP,
            dtype=float,
        )

        # Avoid floating-point overshoot.
        rpm_grid = rpm_grid[rpm_grid <= x.max() + 1e-9]

        measured_rpms = set(np.round(x, 10))

        generated_count = 0
        skipped_test_count = 0

        for rpm in rpm_grid:

            # Do not create a synthetic row where a measured training anchor
            # already exists.
            if round(float(rpm), 10) in measured_rpms:
                continue

            # The locked test coordinates are never allowed into the training
            # expansion, even though their X values are known.
            if (float(d), float(rpm)) in test_coordinates:
                skipped_test_count += 1
                continue

            value = float(spline(rpm))

            if not np.isfinite(value):
                continue

            rows.append({
                "Point_ID": f"SYN_d{int(d)}_rpm{int(rpm)}_{target}",
                "d": float(d),
                "Rpm": float(rpm),
                target: value,
                "Data_Type": "SYNTHETIC_IN_RANGE",
                "Generation_Method": "1D cubic RPM interpolation",
            })

            generated_count += 1

        coverage_rows.append({
            "Target": target,
            "d": float(d),
            "Training_Min_RPM": float(x.min()),
            "Training_Max_RPM": float(x.max()),
            "Training_Anchor_Count": len(x),
            "Generated_5RPM_Points": generated_count,
            "Skipped_Test_Coordinates": skipped_test_count,
        })

    result = pd.DataFrame(rows)

    # Sort for easy inspection.
    result = result.sort_values(
        by=["d", "Rpm", "Data_Type"],
        ascending=[True, True, True],
    ).reset_index(drop=True)

    # Final safety check: no test coordinate may occur in expanded training data.
    result_coordinates = set(
        zip(
            result["d"].astype(float),
            result["Rpm"].astype(float),
        )
    )

    leaked_coordinates = result_coordinates.intersection(test_coordinates)

    if leaked_coordinates:
        raise RuntimeError(
            f"TEST COORDINATE LEAK DETECTED for {target}: "
            f"{sorted(leaked_coordinates)}"
        )

    expanded[target] = result


# ---------------------------------------------------------------------------
# 7. Create expansion log
# ---------------------------------------------------------------------------

log_rows = [
    ["Input file", INPUT_FILE.name],
    ["Training anchors", len(train)],
    ["Locked test observations", len(test)],
    ["RPM interpolation step", f"{RPM_STEP} RPM"],
    ["Diameter interpolation", "NOT PERFORMED"],
    ["RPM extrapolation", "NOT PERFORMED"],
    ["Diameter extrapolation", "NOT PERFORMED"],
    ["Test coordinates excluded", len(test_coordinates)],
    ["Interpolation basis", "TRAINING ANCHORS ONLY"],
    ["Interpolation type", "1-D cubic interpolation at fixed diameter"],
]

log_df = pd.DataFrame(
    log_rows,
    columns=["Item", "Value"],
)


# ---------------------------------------------------------------------------
# 8. RPM coverage table
# ---------------------------------------------------------------------------

coverage_df = pd.DataFrame(coverage_rows)


# ---------------------------------------------------------------------------
# 9. Write Excel workbook
# ---------------------------------------------------------------------------

with pd.ExcelWriter(OUTPUT_FILE, engine="openpyxl") as writer:

    expanded["hc"].to_excel(
        writer,
        sheet_name="hc_Expanded",
        index=False,
    )

    expanded["A"].to_excel(
        writer,
        sheet_name="A_Expanded",
        index=False,
    )

    expanded["Vmin"].to_excel(
        writer,
        sheet_name="Vmin_Expanded",
        index=False,
    )

    log_df.to_excel(
        writer,
        sheet_name="Expansion_Log",
        index=False,
    )

    coverage_df.to_excel(
        writer,
        sheet_name="RPM_Coverage",
        index=False,
    )


# ---------------------------------------------------------------------------
# 10. Console summary
# ---------------------------------------------------------------------------

print("\n" + "=" * 70)
print("STEP 3A - RPM-ONLY INTERPOLATION COMPLETE")
print("=" * 70)

print(f"\nInput : {INPUT_FILE}")
print(f"Output: {OUTPUT_FILE}")

print("\nLocked split:")
print(f"  Training anchors : {len(train)}")
print(f"  Test holdout     : {len(test)}")

print(f"\nRPM interpolation step: {RPM_STEP} RPM")
print("Diameter interpolation: NOT PERFORMED")
print("RPM extrapolation     : NOT PERFORMED")
print("Diameter extrapolation: NOT PERFORMED")

for target in TARGETS:
    df = expanded[target]

    measured_count = (df["Data_Type"] == "MEASURED").sum()
    synthetic_count = (df["Data_Type"] == "SYNTHETIC_IN_RANGE").sum()

    print(f"\n{target}:")
    print(f"  Measured training anchors : {measured_count}")
    print(f"  Synthetic RPM points      : {synthetic_count}")
    print(f"  Total rows                : {len(df)}")

print("\nTest coordinates were explicitly excluded from all expanded datasets.")

print("\nNext step:")
print("  Review RPM interpolation before performing RPM/diameter extrapolation.")
print("=" * 70)