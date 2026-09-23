"""
Step 4A - RPM Extrapolation Model Validation (Pseudo-Extrapolation)

Purpose
-------
Before generating 280, 300 and 320 RPM predictions, validate whether a
continuous regression model can reasonably extrapolate beyond the observed
RPM range.

This step uses ONLY the 67 training anchors from Step 1.

For each response (hc, A, Vmin), compare:
    1st-order RSM
    2nd-order RSM
    3rd-order RSM

Pseudo-extrapolation protocol
-----------------------------
For each diameter, temporarily withhold the highest available TRAINING RPM
anchor(s). These points are treated as pseudo-future observations.

The RSM is fitted using the remaining training anchors and predicts the
withheld high-RPM observations.

The final locked 17 Test_Holdout observations are NEVER used.

Important:
- This is a model-selection/diagnostic step, not the final test.
- No 280/300/320 RPM values are generated yet.
- No diameter extrapolation is performed.
- The purpose is to choose a defensible continuous model for RPM extrapolation.

Output
------
Step4A_RPM_Extrapolation_Validation.xlsx
Step4A_RPM_Extrapolation_Validation_Plots/
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    r2_score,
    mean_squared_error,
    mean_absolute_error,
)


# ---------------------------------------------------------------------------
# 1. Paths
# ---------------------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent

INPUT_FILE = BASE_DIR / "Step1_Stratified_Split.xlsx"

OUTPUT_FILE = (
    BASE_DIR / "Step4A_RPM_Extrapolation_Validation.xlsx"
)

PLOT_DIR = (
    BASE_DIR / "Step4A_RPM_Extrapolation_Validation_Plots"
)

PLOT_DIR.mkdir(exist_ok=True)


# ---------------------------------------------------------------------------
# 2. Configuration
# ---------------------------------------------------------------------------

TARGETS = ["hc", "A", "Vmin"]
DEGREES = [1, 2, 3]

# Number of high-RPM training anchors withheld per diameter.
# Three levels gives a meaningful pseudo-extrapolation test while retaining
# enough data to fit the RSM.
N_HELD_OUT_PER_DIAMETER = 3

REQUIRED_COLUMNS = [
    "Point_ID",
    "d",
    "Rpm",
    "hc",
    "A",
    "Vmin",
]


# ---------------------------------------------------------------------------
# 3. Load data
# ---------------------------------------------------------------------------

if not INPUT_FILE.exists():
    raise FileNotFoundError(f"Could not find: {INPUT_FILE}")

train = pd.read_excel(
    INPUT_FILE,
    sheet_name="Train_Anchors",
)

missing = [c for c in REQUIRED_COLUMNS if c not in train.columns]
if missing:
    raise ValueError(f"Train_Anchors is missing columns: {missing}")

for col in ["d", "Rpm"] + TARGETS:
    train[col] = pd.to_numeric(train[col])

if len(train) != 67:
    raise ValueError(
        f"Expected 67 training anchors, found {len(train)}."
    )


# ---------------------------------------------------------------------------
# 4. RSM fitting helper
# ---------------------------------------------------------------------------

def fit_rsm(df, target, degree):
    """
    Fit polynomial response surface:
        target = f(d, RPM)

    Degree 1, 2, or 3.
    """
    X = df[["d", "Rpm"]].to_numpy(dtype=float)
    y = df[target].to_numpy(dtype=float)

    poly = PolynomialFeatures(
        degree=degree,
        include_bias=False,
    )

    X_poly = poly.fit_transform(X)

    model = LinearRegression()
    model.fit(X_poly, y)

    return poly, model


def predict_rsm(poly, model, df):
    X = df[["d", "Rpm"]].to_numpy(dtype=float)
    return model.predict(poly.transform(X))


def calc_metrics(y_true, y_pred):

    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    mse = mean_squared_error(y_true, y_pred)

    return {
        "R2": r2_score(y_true, y_pred),
        "RMSE": np.sqrt(mse),
        "MAE": mean_absolute_error(y_true, y_pred),
        "MSE": mse,
        "MAPE_percent": (
            np.mean(
                np.abs(
                    (y_true - y_pred)
                    / y_true
                )
            ) * 100
            if np.all(np.abs(y_true) > 1e-12)
            else np.nan
        ),
    }


# ---------------------------------------------------------------------------
# 5. Create pseudo-extrapolation set
# ---------------------------------------------------------------------------

pseudo_test_parts = []
pseudo_train_parts = []

for d in sorted(train["d"].unique()):

    subset = (
        train[train["d"] == d]
        .sort_values("Rpm")
        .reset_index(drop=True)
    )

    if len(subset) <= N_HELD_OUT_PER_DIAMETER + 3:
        raise ValueError(
            f"Too few training anchors at d={d} for pseudo-extrapolation."
        )

    held = subset.tail(N_HELD_OUT_PER_DIAMETER)
    remaining = subset.iloc[:-N_HELD_OUT_PER_DIAMETER]

    pseudo_test_parts.append(held)
    pseudo_train_parts.append(remaining)

pseudo_test = pd.concat(
    pseudo_test_parts,
    ignore_index=True,
)

pseudo_train = pd.concat(
    pseudo_train_parts,
    ignore_index=True,
)


# ---------------------------------------------------------------------------
# 6. Validate 1st/2nd/3rd order RSM
# ---------------------------------------------------------------------------

summary_rows = []
fold_rows = []

for target in TARGETS:

    for degree in DEGREES:

        poly, model = fit_rsm(
            pseudo_train,
            target,
            degree,
        )

        predictions = predict_rsm(
            poly,
            model,
            pseudo_test,
        )

        actual = pseudo_test[target].to_numpy(float)

        metric = calc_metrics(actual, predictions)

        summary_rows.append({
            "Target": target,
            "RSM_Degree": degree,
            "Pseudo_Train_Points": len(pseudo_train),
            "Pseudo_Extrapolation_Points": len(pseudo_test),
            **metric,
        })

        for i, row in pseudo_test.iterrows():

            fold_rows.append({
                "Target": target,
                "RSM_Degree": degree,
                "Point_ID": row["Point_ID"],
                "d": row["d"],
                "Rpm": row["Rpm"],
                "Actual": row[target],
                "Predicted": predictions[i],
                "Absolute_Error": abs(
                    row[target] - predictions[i]
                ),
                "Percentage_Error": (
                    abs(
                        (row[target] - predictions[i])
                        / row[target]
                    ) * 100
                    if abs(row[target]) > 1e-12
                    else np.nan
                ),
            })


summary_df = pd.DataFrame(summary_rows)
fold_df = pd.DataFrame(fold_rows)


# ---------------------------------------------------------------------------
# 7. Diameter-level results
# ---------------------------------------------------------------------------

diameter_rows = []

for target in TARGETS:

    for degree in DEGREES:

        for d in sorted(pseudo_test["d"].unique()):

            subset = fold_df[
                (fold_df["Target"] == target)
                & (fold_df["RSM_Degree"] == degree)
                & (fold_df["d"] == d)
            ]

            if len(subset) >= 2:

                m = calc_metrics(
                    subset["Actual"],
                    subset["Predicted"],
                )

                diameter_rows.append({
                    "Target": target,
                    "RSM_Degree": degree,
                    "d": d,
                    "N": len(subset),
                    **m,
                })

diameter_df = pd.DataFrame(diameter_rows)


# ---------------------------------------------------------------------------
# 8. Physical validity
# ---------------------------------------------------------------------------

physical_rows = []

for target in TARGETS:

    for degree in DEGREES:

        subset = summary_df[
            (summary_df["Target"] == target)
            & (summary_df["RSM_Degree"] == degree)
        ]

        predictions = fold_df[
            (fold_df["Target"] == target)
            & (fold_df["RSM_Degree"] == degree)
        ]["Predicted"].to_numpy(float)

        physical_rows.append({
            "Target": target,
            "RSM_Degree": degree,
            "Negative_Predictions": int(
                np.sum(predictions < 0)
            ),
            "Nonfinite_Predictions": int(
                np.sum(~np.isfinite(predictions))
            ),
            "Minimum_Prediction": float(
                np.min(predictions)
            ),
            "Maximum_Prediction": float(
                np.max(predictions)
            ),
        })

physical_df = pd.DataFrame(physical_rows)


# ---------------------------------------------------------------------------
# 9. Generate diagnostic plots
# ---------------------------------------------------------------------------

for target in TARGETS:

    for degree in DEGREES:

        subset = fold_df[
            (fold_df["Target"] == target)
            & (fold_df["RSM_Degree"] == degree)
        ].sort_values(["d", "Rpm"])

        plt.figure(figsize=(9, 6))

        plt.scatter(
            subset["Actual"],
            subset["Predicted"],
            s=45,
            label=f"3-point high-RPM pseudo-extrapolation",
        )

        min_val = min(
            subset["Actual"].min(),
            subset["Predicted"].min(),
        )
        max_val = max(
            subset["Actual"].max(),
            subset["Predicted"].max(),
        )

        plt.plot(
            [min_val, max_val],
            [min_val, max_val],
            linewidth=1.5,
            label="Ideal prediction",
        )

        plt.xlabel("Actual")
        plt.ylabel("Predicted")
        plt.title(
            f"{target} — {degree}rd-order RSM "
            "Pseudo-Extrapolation Validation"
        )
        plt.grid(True, alpha=0.25)
        plt.legend()
        plt.tight_layout()

        plot_path = (
            PLOT_DIR
            / f"{target}_RSM_degree{degree}_pseudo_extrapolation.png"
        )

        plt.savefig(plot_path, dpi=200)
        plt.close()


# ---------------------------------------------------------------------------
# 10. Save workbook
# ---------------------------------------------------------------------------

protocol_df = pd.DataFrame([
    ["Training source", "Step1_Stratified_Split.xlsx / Train_Anchors"],
    ["Training anchors", 67],
    ["Final test points used", 0],
    ["Pseudo-extrapolation design",
     "Highest 3 available training RPM anchors withheld at each diameter"],
    ["Purpose",
     "Assess high-RPM extrapolation capability before generating 280/300/320 RPM"],
    ["Candidate models", "1st-, 2nd-, and 3rd-order RSM"],
    ["RPM extrapolation generated", "No"],
    ["Diameter extrapolation generated", "No"],
])

with pd.ExcelWriter(
    OUTPUT_FILE,
    engine="openpyxl",
) as writer:

    summary_df.to_excel(
        writer,
        sheet_name="Overall_Validation",
        index=False,
    )

    diameter_df.to_excel(
        writer,
        sheet_name="Diameter_Validation",
        index=False,
    )

    fold_df.to_excel(
        writer,
        sheet_name="Pseudo_Extrapolation_Folds",
        index=False,
    )

    physical_df.to_excel(
        writer,
        sheet_name="Physical_Check",
        index=False,
    )

    pseudo_train.to_excel(
        writer,
        sheet_name="Pseudo_Training_Set",
        index=False,
    )

    pseudo_test.to_excel(
        writer,
        sheet_name="Pseudo_Test_Set",
        index=False,
    )

    protocol_df.to_excel(
        writer,
        sheet_name="Protocol",
        index=False,
    )


# ---------------------------------------------------------------------------
# 11. Console output
# ---------------------------------------------------------------------------

print("\n" + "=" * 78)
print("STEP 4A - RPM EXTRAPOLATION MODEL VALIDATION COMPLETE")
print("=" * 78)

print("\nProtocol:")
print("  Original training anchors       : 67")
print("  Final test observations used    : 0")
print("  High-RPM anchors withheld       : 3 per diameter")
print(f"  Pseudo-extrapolation observations: {len(pseudo_test)}")
print("  Candidate models                : 1st, 2nd, 3rd order RSM")
print("  Actual 280/300/320 RPM generated: NO")

print("\nOVERALL PSEUDO-EXTRAPOLATION RESULTS")
print("-" * 78)

print(
    summary_df[
        [
            "Target",
            "RSM_Degree",
            "Pseudo_Extrapolation_Points",
            "R2",
            "RMSE",
            "MAE",
            "MAPE_percent",
        ]
    ].to_string(index=False)
)

print("\nPHYSICAL VALIDITY")
print("-" * 78)

print(
    physical_df.to_string(index=False)
)

print("\nOutput:")
print(f"  Excel : {OUTPUT_FILE}")
print(f"  Plots : {PLOT_DIR}")

print("\nNext decision:")
print(
    "Use these pseudo-extrapolation results to select the RSM order "
    "for actual RPM extrapolation to 280, 300 and 320 RPM."
)

print("=" * 78)
