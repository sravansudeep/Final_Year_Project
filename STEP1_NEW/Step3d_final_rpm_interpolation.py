"""
Step 3D - Final RPM Interpolation Dataset

Creates the locked, final RPM-interpolated training dataset after Step 3C.

Methods:
    hc    -> 1-D Cubic Spline
    A     -> 1-D Cubic Spline
    Vmin  -> 1-D PCHIP

Basis:
    67 Train_Anchors only.

Rules:
    - RPM grid uses 5-RPM spacing.
    - Diameter remains only at measured levels: 6, 8, 10, 12, 14, 16 mm.
    - No RPM extrapolation.
    - No diameter interpolation.
    - No diameter extrapolation.
    - The 17 Test_Holdout coordinates are excluded.
    - Original Step 3A cubic Vmin dataset is not overwritten.

Output:
    Step3_Final_RPM_Interpolation.xlsx
"""

from pathlib import Path
import numpy as np
import pandas as pd
from scipy.interpolate import CubicSpline, PchipInterpolator

BASE_DIR = Path(__file__).resolve().parent

INPUT_FILE = BASE_DIR / "Step1_Stratified_Split.xlsx"
OUTPUT_FILE = BASE_DIR / "Step3_Final_RPM_Interpolation.xlsx"

RPM_STEP = 5
TARGETS = ["hc", "A", "Vmin"]

METHODS = {
    "hc": "1-D Cubic Spline",
    "A": "1-D Cubic Spline",
    "Vmin": "1-D PCHIP",
}

train = pd.read_excel(INPUT_FILE, sheet_name="Train_Anchors")
test = pd.read_excel(INPUT_FILE, sheet_name="Test_Holdout")

if len(train) != 67:
    raise ValueError(f"Expected 67 training anchors, found {len(train)}.")
if len(test) != 17:
    raise ValueError(f"Expected 17 test observations, found {len(test)}.")

for df in [train, test]:
    df["d"] = pd.to_numeric(df["d"])
    df["Rpm"] = pd.to_numeric(df["Rpm"])
    for target in TARGETS:
        df[target] = pd.to_numeric(df[target])

test_coordinates = set(
    zip(test["d"].astype(float), test["Rpm"].astype(float))
)

all_outputs = {}
log_rows = []

for target in TARGETS:

    rows = []

    for d in sorted(train["d"].unique()):

        subset = (
            train[train["d"] == d]
            .sort_values("Rpm")
            .reset_index(drop=True)
        )

        x = subset["Rpm"].to_numpy(float)
        y = subset[target].to_numpy(float)

        if len(np.unique(x)) < 4:
            raise ValueError(
                f"Not enough unique RPM anchors for cubic interpolation at d={d}."
            )

        method = METHODS[target]

        if method == "1-D Cubic Spline":
            interpolator = CubicSpline(
                x, y,
                bc_type="not-a-knot",
                extrapolate=False,
            )
        else:
            interpolator = PchipInterpolator(
                x, y,
                extrapolate=False,
            )

        # Add the measured training anchors.
        for _, row in subset.iterrows():
            rows.append({
                "Point_ID": row["Point_ID"],
                "d": float(row["d"]),
                "Rpm": float(row["Rpm"]),
                target: float(row[target]),
                "Data_Type": "MEASURED",
                "Generation_Method": "Measured training anchor",
            })

        rpm_grid = np.arange(
            x.min(),
            x.max() + RPM_STEP,
            RPM_STEP,
            dtype=float,
        )
        rpm_grid = rpm_grid[rpm_grid <= x.max() + 1e-9]

        measured_rpms = set(np.round(x, 10))

        generated = 0
        skipped_test = 0

        for rpm in rpm_grid:

            if round(float(rpm), 10) in measured_rpms:
                continue

            if (float(d), float(rpm)) in test_coordinates:
                skipped_test += 1
                continue

            value = float(interpolator(rpm))

            if not np.isfinite(value):
                raise ValueError(
                    f"Non-finite interpolated {target} at d={d}, RPM={rpm}."
                )

            rows.append({
                "Point_ID": f"SYN_d{int(d)}_rpm{int(rpm)}_{target}",
                "d": float(d),
                "Rpm": float(rpm),
                target: value,
                "Data_Type": "SYNTHETIC_IN_RANGE",
                "Generation_Method": method,
            })

            generated += 1

        log_rows.append({
            "Target": target,
            "Diameter_mm": d,
            "Method": method,
            "Training_Anchors": len(subset),
            "Generated_Synthetic_Points": generated,
            "Skipped_Test_Coordinates": skipped_test,
            "Minimum_Value": float(
                pd.DataFrame(rows)
                .query("d == @d")[target]
                .min()
            ),
            "Maximum_Value": float(
                pd.DataFrame(rows)
                .query("d == @d")[target]
                .max()
            ),
        })

    result = (
        pd.DataFrame(rows)
        .sort_values(["d", "Rpm", "Data_Type"])
        .reset_index(drop=True)
    )

    # Final safety checks.
    coords = set(zip(result["d"], result["Rpm"]))
    leaked = coords.intersection(test_coordinates)

    if leaked:
        raise RuntimeError(
            f"Test coordinate leakage detected for {target}: {sorted(leaked)}"
        )

    if not np.isfinite(result[target]).all():
        raise RuntimeError(f"Non-finite values detected in {target}.")

    if target == "Vmin":
        negative = result[result[target] < 0]
        if len(negative) > 0:
            raise RuntimeError(
                "Negative Vmin values remain after PCHIP interpolation:\n"
                + negative.to_string(index=False)
            )

    all_outputs[target] = result


log_df = pd.DataFrame(log_rows)

summary_df = pd.DataFrame([
    {
        "Target": target,
        "Interpolation_Method": METHODS[target],
        "Measured_Training_Anchors": int(
            (all_outputs[target]["Data_Type"] == "MEASURED").sum()
        ),
        "Synthetic_In_Range_Points": int(
            (all_outputs[target]["Data_Type"] == "SYNTHETIC_IN_RANGE").sum()
        ),
        "Total_Rows": len(all_outputs[target]),
        "Negative_Values": int(
            (all_outputs[target][target] < 0).sum()
        ),
    }
    for target in TARGETS
])

methodology_df = pd.DataFrame([
    ["Training data", "67 Train_Anchors only"],
    ["Test data", "17 Test_Holdout observations kept untouched"],
    ["RPM interpolation", "Every 5 RPM"],
    ["Diameter levels", "6, 8, 10, 12, 14, 16 mm only"],
    ["hc method", "1-D Cubic Spline"],
    ["A method", "1-D Cubic Spline"],
    ["Vmin method", "1-D PCHIP"],
    ["RPM extrapolation", "Not performed"],
    ["Diameter interpolation", "Not performed"],
    ["Diameter extrapolation", "Not performed"],
    ["Test-coordinate exclusion", "Exact held-out (d, RPM) coordinates excluded"],
])

with pd.ExcelWriter(OUTPUT_FILE, engine="openpyxl") as writer:

    all_outputs["hc"].to_excel(
        writer, sheet_name="hc_Expanded", index=False
    )
    all_outputs["A"].to_excel(
        writer, sheet_name="A_Expanded", index=False
    )
    all_outputs["Vmin"].to_excel(
        writer, sheet_name="Vmin_Expanded", index=False
    )
    summary_df.to_excel(
        writer, sheet_name="Summary", index=False
    )
    log_df.to_excel(
        writer, sheet_name="Diameter_Log", index=False
    )
    methodology_df.to_excel(
        writer, sheet_name="Methodology", index=False
    )

print("\n" + "=" * 75)
print("STEP 3D - FINAL RPM INTERPOLATION DATASET CREATED")
print("=" * 75)

print("\nFinal methods:")
print("  hc    -> 1-D Cubic Spline")
print("  A     -> 1-D Cubic Spline")
print("  Vmin  -> 1-D PCHIP")

print("\nDataset summary:")
print(summary_df.to_string(index=False))

print("\nValidation:")
print("  Training anchors used : 67")
print("  Test observations used: 0")
print("  RPM interval          : 5 RPM")
print("  Test-coordinate leak  : 0")
print("  Vmin negative values  : 0")

print(f"\nOutput: {OUTPUT_FILE}")
print("\nThis file is now the locked RPM-interpolated training dataset.")
print("Next step: RPM extrapolation beyond 260 RPM.")
print("=" * 75)
