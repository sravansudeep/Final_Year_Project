"""
Step 4B - Progressive RPM Extrapolation Validation

Uses ONLY the 67 Train_Anchors from Step1_Stratified_Split.xlsx.

Progressive pseudo-extrapolation:
  Scenario 1: withhold highest 1 available RPM anchor per diameter
  Scenario 2: withhold highest 2 available RPM anchors per diameter
  Scenario 3: withhold highest 3 available RPM anchors per diameter

For each scenario, compare 1st-, 2nd-, and 3rd-order RSMs for:
  hc, A, Vmin

Important:
Some high-RPM observations were reserved for the final experimental test set.
Therefore the script withholds the highest AVAILABLE training RPMs rather than
assuming every diameter has every nominal RPM level.

No final Test_Holdout observations are used.
"""

from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error


# ============================================================
# SETTINGS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

INPUT_FILE = BASE_DIR / "Step1_Stratified_Split.xlsx"
OUTPUT_FILE = BASE_DIR / "Step4B_Progressive_RPM_Extrapolation_Validation.xlsx"

TARGETS = ["hc", "A", "Vmin"]
RSM_ORDERS = [1, 2, 3]

SCENARIOS = {
    "Scenario_1_Highest_1": 1,
    "Scenario_2_Highest_2": 2,
    "Scenario_3_Highest_3": 3,
}


# ============================================================
# FUNCTIONS
# ============================================================

def fit_rsm(train_df, target, order):
    X = train_df[["d", "Rpm"]].to_numpy(dtype=float)
    y = train_df[target].to_numpy(dtype=float)

    poly = PolynomialFeatures(degree=order, include_bias=True)
    X_poly = poly.fit_transform(X)

    model = LinearRegression(fit_intercept=False)
    model.fit(X_poly, y)

    return poly, model


def predict_rsm(poly, model, df):
    X = df[["d", "Rpm"]].to_numpy(dtype=float)
    return model.predict(poly.transform(X))


def calculate_metrics(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)

    nonzero = np.abs(y_true) > 1e-12
    if np.any(nonzero):
        mape = (
            np.mean(
                np.abs(
                    (y_true[nonzero] - y_pred[nonzero])
                    / y_true[nonzero]
                )
            )
            * 100
        )
    else:
        mape = np.nan

    return r2, rmse, mae, mse, mape


def make_split(train_anchors, n_withhold):
    """
    For each diameter, withhold the highest n available training RPMs.
    """
    fit_parts = []
    hold_parts = []

    for d in sorted(train_anchors["d"].unique()):
        group = (
            train_anchors[train_anchors["d"] == d]
            .sort_values("Rpm")
            .copy()
        )

        if len(group) <= n_withhold:
            raise ValueError(
                f"Diameter {d}: insufficient anchors for withholding "
                f"{n_withhold} points."
            )

        hold = group.tail(n_withhold)
        fit = group.iloc[:-n_withhold]

        fit_parts.append(fit)
        hold_parts.append(hold)

    fit_df = pd.concat(fit_parts, ignore_index=True)
    hold_df = pd.concat(hold_parts, ignore_index=True)

    return fit_df, hold_df


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 75)
print("STEP 4B - PROGRESSIVE RPM EXTRAPOLATION VALIDATION")
print("=" * 75)

if not INPUT_FILE.exists():
    raise FileNotFoundError(
        f"\nCould not find:\n{INPUT_FILE}\n"
        "Place this script in the same folder as Step1_Stratified_Split.xlsx."
    )

train_anchors = pd.read_excel(
    INPUT_FILE,
    sheet_name="Train_Anchors"
)

required_columns = [
    "Point_ID", "d", "Rpm", "hc", "A", "Vmin"
]

missing = [
    col for col in required_columns
    if col not in train_anchors.columns
]

if missing:
    raise ValueError(f"Missing required columns: {missing}")

train_anchors = train_anchors[required_columns].copy()

print(f"\nTraining anchors: {len(train_anchors)}")
print(f"Diameters: {sorted(train_anchors['d'].unique())}")
print(
    f"Overall training RPM range: "
    f"{train_anchors['Rpm'].min()} to {train_anchors['Rpm'].max()}"
)

# ============================================================
# VALIDATION
# ============================================================

overall_rows = []
best_rows = []
diameter_rows = []
prediction_rows = []
physical_rows = []
scenario_rows = []

for scenario_name, n_withhold in SCENARIOS.items():

    fit_df, hold_df = make_split(
        train_anchors,
        n_withhold
    )

    print("\n" + "-" * 75)
    print(scenario_name)
    print(
        f"Withholding highest {n_withhold} available RPM "
        f"anchor(s) per diameter"
    )
    print(f"Fit observations: {len(fit_df)}")
    print(f"Held-out observations: {len(hold_df)}")

    # Record exact RPMs used in each pseudo-extrapolation
    for d in sorted(hold_df["d"].unique()):

        held = sorted(
            hold_df.loc[hold_df["d"] == d, "Rpm"].tolist()
        )

        fitted = sorted(
            fit_df.loc[fit_df["d"] == d, "Rpm"].tolist()
        )

        scenario_rows.append({
            "Scenario": scenario_name,
            "Diameter_d": d,
            "Withheld_RPM": ", ".join(map(str, held)),
            "Fit_RPM_Min": min(fitted),
            "Fit_RPM_Max": max(fitted),
            "N_Fit": len(fitted),
            "N_Heldout": len(held),
        })

    for target in TARGETS:

        for order in RSM_ORDERS:

            poly, model = fit_rsm(
                fit_df,
                target,
                order
            )

            y_true = hold_df[target].to_numpy(dtype=float)
            y_pred = predict_rsm(
                poly,
                model,
                hold_df
            )

            r2, rmse, mae, mse, mape = calculate_metrics(
                y_true,
                y_pred
            )

            overall_rows.append({
                "Scenario": scenario_name,
                "Withheld_Highest_N": n_withhold,
                "Target": target,
                "RSM_Order": order,
                "N_Fit": len(fit_df),
                "N_Heldout": len(hold_df),
                "R2": r2,
                "RMSE": rmse,
                "MAE": mae,
                "MSE": mse,
                "MAPE_percent": mape,
            })

            # Physical validity
            negative_count = int(np.sum(y_pred < 0))
            nonfinite_count = int(
                np.sum(~np.isfinite(y_pred))
            )

            physical_rows.append({
                "Scenario": scenario_name,
                "Target": target,
                "RSM_Order": order,
                "N_Predictions": len(y_pred),
                "Negative_Predictions": negative_count,
                "Nonfinite_Predictions": nonfinite_count,
                "Minimum_Prediction": float(np.min(y_pred)),
                "Maximum_Prediction": float(np.max(y_pred)),
                "Physically_Valid": (
                    negative_count == 0
                    and nonfinite_count == 0
                ),
            })

            # Individual predictions
            for i, (_, row) in enumerate(
                hold_df.iterrows()
            ):

                actual = float(row[target])
                predicted = float(y_pred[i])

                if abs(actual) > 1e-12:
                    percent_error = (
                        abs(actual - predicted)
                        / abs(actual)
                        * 100
                    )
                else:
                    percent_error = np.nan

                prediction_rows.append({
                    "Scenario": scenario_name,
                    "Target": target,
                    "RSM_Order": order,
                    "Point_ID": row["Point_ID"],
                    "d": row["d"],
                    "Rpm": row["Rpm"],
                    "Actual": actual,
                    "Predicted": predicted,
                    "Residual": actual - predicted,
                    "Absolute_Error": abs(actual - predicted),
                    "Percent_Error": percent_error,
                })

            # Diameter-level metrics
            for d in sorted(hold_df["d"].unique()):

                mask = (
                    hold_df["d"].to_numpy(dtype=float) == float(d)
                )

                y_d_true = y_true[mask]
                y_d_pred = y_pred[mask]

                if (
                    len(y_d_true) >= 2
                    and np.std(y_d_true) > 1e-12
                ):
                    d_r2 = r2_score(
                        y_d_true,
                        y_d_pred
                    )
                else:
                    d_r2 = np.nan

                d_mse = mean_squared_error(
                    y_d_true,
                    y_d_pred
                )
                d_rmse = np.sqrt(d_mse)
                d_mae = mean_absolute_error(
                    y_d_true,
                    y_d_pred
                )

                nz = np.abs(y_d_true) > 1e-12

                if np.any(nz):
                    d_mape = (
                        np.mean(
                            np.abs(
                                (
                                    y_d_true[nz]
                                    - y_d_pred[nz]
                                )
                                / y_d_true[nz]
                            )
                        )
                        * 100
                    )
                else:
                    d_mape = np.nan

                diameter_rows.append({
                    "Scenario": scenario_name,
                    "Target": target,
                    "RSM_Order": order,
                    "Diameter_d": d,
                    "N_Heldout": int(mask.sum()),
                    "R2": d_r2,
                    "RMSE": d_rmse,
                    "MAE": d_mae,
                    "MSE": d_mse,
                    "MAPE_percent": d_mape,
                })


# ============================================================
# RESULTS TABLES
# ============================================================

overall_df = pd.DataFrame(overall_rows)
diameter_df = pd.DataFrame(diameter_rows)
prediction_df = pd.DataFrame(prediction_rows)
physical_df = pd.DataFrame(physical_rows)
scenario_df = pd.DataFrame(scenario_rows)

overall_df["RMSE_Rank"] = (
    overall_df
    .groupby(["Scenario", "Target"])["RMSE"]
    .rank(method="min")
)

best_df = (
    overall_df
    .sort_values(
        ["Scenario", "Target", "RMSE", "MAE"]
    )
    .groupby(
        ["Scenario", "Target"],
        as_index=False
    )
    .first()
)


# ============================================================
# PRINT RESULTS
# ============================================================

print("\n" + "=" * 75)
print("OVERALL RESULTS")
print("=" * 75)

print(
    overall_df[
        [
            "Scenario",
            "Target",
            "RSM_Order",
            "R2",
            "RMSE",
            "MAE",
            "MAPE_percent",
        ]
    ]
    .sort_values(
        ["Scenario", "Target", "RSM_Order"]
    )
    .to_string(
        index=False,
        float_format=lambda x: f"{x:.6f}"
    )
)

print("\n" + "=" * 75)
print("BEST RSM ORDER BY SCENARIO AND TARGET")
print("=" * 75)

print(
    best_df[
        [
            "Scenario",
            "Target",
            "RSM_Order",
            "R2",
            "RMSE",
            "MAE",
            "MAPE_percent",
        ]
    ]
    .to_string(
        index=False,
        float_format=lambda x: f"{x:.6f}"
    )
)

print("\n" + "=" * 75)
print("PHYSICAL VALIDITY")
print("=" * 75)

print(
    physical_df.to_string(index=False)
)


# ============================================================
# SAVE EXCEL
# ============================================================

with pd.ExcelWriter(
    OUTPUT_FILE,
    engine="openpyxl"
) as writer:

    train_anchors.to_excel(
        writer,
        sheet_name="Train_Anchors",
        index=False
    )

    overall_df.to_excel(
        writer,
        sheet_name="Overall_Results",
        index=False
    )

    best_df.to_excel(
        writer,
        sheet_name="Best_Methods",
        index=False
    )

    diameter_df.to_excel(
        writer,
        sheet_name="Diameter_Results",
        index=False
    )

    prediction_df.to_excel(
        writer,
        sheet_name="Predictions",
        index=False
    )

    physical_df.to_excel(
        writer,
        sheet_name="Physical_Check",
        index=False
    )

    scenario_df.to_excel(
        writer,
        sheet_name="Scenario_Info",
        index=False
    )

print("\n" + "=" * 75)
print("COMPLETED")
print(f"Output: {OUTPUT_FILE}")
print("=" * 75)
