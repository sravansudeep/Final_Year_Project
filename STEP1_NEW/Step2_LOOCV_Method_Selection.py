# ============================================================
# STEP 2 — LOOCV METHOD SELECTION
# ============================================================
#
# Purpose:
#   Compare candidate interpolation / response-surface methods
#   using ONLY the 67 real training anchor points.
#
# Targets:
#   1. hc
#   2. A
#   3. Vmin
#
# Methods:
#   1. Linear Griddata
#   2. Cubic Griddata
#   3. 1st-order RSM
#   4. 2nd-order RSM
#   5. 3rd-order RSM
#
# IMPORTANT:
#   Test_Holdout is NOT used anywhere in this script.
#
# ============================================================


import pandas as pd
import numpy as np

from pathlib import Path
from scipy.interpolate import griddata
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression


# ============================================================
# 1. SETTINGS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

INPUT_FILE = (
    BASE_DIR / "Step1_Stratified_Split.xlsx"
)

INPUT_SHEET = "Train_Anchors"

OUTPUT_FILE = (
    BASE_DIR / "Step2_LOOCV_Results.xlsx"
)


# ============================================================
# 2. LOAD TRAINING ANCHORS
# ============================================================

print("\n" + "=" * 70)
print("STEP 2 — LOOCV METHOD SELECTION")
print("=" * 70)

print("\nLoading training anchors...")

df = pd.read_excel(
    INPUT_FILE,
    sheet_name=INPUT_SHEET
)

print(f"Training observations loaded: {len(df)}")


# ------------------------------------------------------------
# Verify that only the 67 anchor points are being used
# ------------------------------------------------------------

if len(df) != 67:
    raise ValueError(
        f"Expected 67 training anchors, "
        f"but found {len(df)}."
    )


# ============================================================
# 3. REQUIRED COLUMNS
# ============================================================

required_columns = [
    "Point_ID",
    "d",
    "Rpm",
    "hc",
    "A",
    "Vmin"
]

missing = [
    col for col in required_columns
    if col not in df.columns
]

if missing:
    raise ValueError(
        f"Missing columns: {missing}"
    )


# ============================================================
# 4. PREPARE INPUTS AND TARGETS
# ============================================================

X = df[
    ["d", "Rpm"]
].to_numpy(dtype=float)

targets = {
    "hc": df["hc"].to_numpy(dtype=float),
    "A": df["A"].to_numpy(dtype=float),
    "Vmin": df["Vmin"].to_numpy(dtype=float)
}


# ============================================================
# 5. METRIC FUNCTIONS
# ============================================================

def calculate_metrics(
    actual,
    predicted
):

    actual = np.asarray(actual)
    predicted = np.asarray(predicted)

    error = actual - predicted

    mse = np.mean(
        error ** 2
    )

    rmse = np.sqrt(mse)

    mae = np.mean(
        np.abs(error)
    )

    # MAPE
    nonzero = actual != 0

    if np.any(nonzero):

        mape = np.mean(
            np.abs(
                error[nonzero] /
                actual[nonzero]
            )
        ) * 100

    else:

        mape = np.nan

    # R²
    ss_res = np.sum(
        error ** 2
    )

    ss_tot = np.sum(
        (actual - np.mean(actual)) ** 2
    )

    if ss_tot == 0:

        r2 = np.nan

    else:

        r2 = 1 - (
            ss_res / ss_tot
        )

    return {
        "R2": r2,
        "RMSE": rmse,
        "MAE": mae,
        "MSE": mse,
        "MAPE": mape
    }


# ============================================================
# 6. RSM PREDICTION FUNCTION
# ============================================================

def rsm_predict(
    X_train,
    y_train,
    X_test,
    degree
):

    poly = PolynomialFeatures(
        degree=degree,
        include_bias=True
    )

    X_train_poly = poly.fit_transform(
        X_train
    )

    X_test_poly = poly.transform(
        X_test
    )

    model = LinearRegression(
        fit_intercept=False
    )

    model.fit(
        X_train_poly,
        y_train
    )

    prediction = model.predict(
        X_test_poly
    )

    return prediction


# ============================================================
# 7. LOOCV FUNCTION
# ============================================================

def run_loocv(
    X,
    y,
    target_name
):

    methods = [
        "Linear Griddata",
        "Cubic Griddata",
        "1st-order RSM",
        "2nd-order RSM",
        "3rd-order RSM"
    ]

    predictions = {
        method: []
        for method in methods
    }

    actual_values = {
        method: []
        for method in methods
    }

    point_ids = {
        method: []
        for method in methods
    }

    print("\n" + "-" * 70)
    print(f"TARGET: {target_name}")
    print("-" * 70)

    for i in range(len(X)):

        # ----------------------------------------------------
        # Leave one observation out
        # ----------------------------------------------------

        X_train = np.delete(
            X,
            i,
            axis=0
        )

        y_train = np.delete(
            y,
            i
        )

        X_test = X[
            i:i + 1
        ]

        y_test = y[i]

        point_id = df.iloc[
            i
        ]["Point_ID"]


        # ----------------------------------------------------
        # Linear Griddata
        # ----------------------------------------------------

        linear_prediction = griddata(
            X_train,
            y_train,
            X_test,
            method="linear"
        )

        if (
            linear_prediction is not None
            and np.isfinite(
                linear_prediction[0]
            )
        ):

            predictions[
                "Linear Griddata"
            ].append(
                linear_prediction[0]
            )

            actual_values[
                "Linear Griddata"
            ].append(y_test)

            point_ids[
                "Linear Griddata"
            ].append(point_id)


        # ----------------------------------------------------
        # Cubic Griddata
        # ----------------------------------------------------

        try:

            cubic_prediction = griddata(
                X_train,
                y_train,
                X_test,
                method="cubic"
            )

            if (
                cubic_prediction is not None
                and np.isfinite(
                    cubic_prediction[0]
                )
            ):

                predictions[
                    "Cubic Griddata"
                ].append(
                    cubic_prediction[0]
                )

                actual_values[
                    "Cubic Griddata"
                ].append(y_test)

                point_ids[
                    "Cubic Griddata"
                ].append(point_id)

        except Exception:

            pass


        # ----------------------------------------------------
        # 1st-order RSM
        # ----------------------------------------------------

        pred = rsm_predict(
            X_train,
            y_train,
            X_test,
            degree=1
        )

        predictions[
            "1st-order RSM"
        ].append(pred[0])

        actual_values[
            "1st-order RSM"
        ].append(y_test)

        point_ids[
            "1st-order RSM"
        ].append(point_id)


        # ----------------------------------------------------
        # 2nd-order RSM
        # ----------------------------------------------------

        pred = rsm_predict(
            X_train,
            y_train,
            X_test,
            degree=2
        )

        predictions[
            "2nd-order RSM"
        ].append(pred[0])

        actual_values[
            "2nd-order RSM"
        ].append(y_test)

        point_ids[
            "2nd-order RSM"
        ].append(point_id)


        # ----------------------------------------------------
        # 3rd-order RSM
        # ----------------------------------------------------

        pred = rsm_predict(
            X_train,
            y_train,
            X_test,
            degree=3
        )

        predictions[
            "3rd-order RSM"
        ].append(pred[0])

        actual_values[
            "3rd-order RSM"
        ].append(y_test)

        point_ids[
            "3rd-order RSM"
        ].append(point_id)


    # ========================================================
    # 8. CALCULATE FINAL METRICS
    # ========================================================

    summary = []

    fold_results = []

    for method in methods:

        actual = np.array(
            actual_values[method]
        )

        predicted = np.array(
            predictions[method]
        )

        metrics = calculate_metrics(
            actual,
            predicted
        )

        summary.append({

            "Target": target_name,

            "Method": method,

            "Valid LOOCV Points": len(
                actual
            ),

            "R2": metrics["R2"],

            "RMSE": metrics["RMSE"],

            "MAE": metrics["MAE"],

            "MSE": metrics["MSE"],

            "MAPE (%)": metrics["MAPE"]

        })


        # Individual fold predictions
        for j in range(len(actual)):

            fold_results.append({

                "Target": target_name,

                "Method": method,

                "Point_ID": point_ids[
                    method
                ][j],

                "Actual": actual[j],

                "Predicted": predicted[j],

                "Error":
                    actual[j] -
                    predicted[j],

                "Absolute_Error":
                    abs(
                        actual[j] -
                        predicted[j]
                    )

            })


    return (
        pd.DataFrame(summary),
        pd.DataFrame(fold_results)
    )


# ============================================================
# 9. RUN LOOCV FOR ALL THREE TARGETS
# ============================================================

all_summary = []
all_folds = []


for target_name, y in targets.items():

    summary, folds = run_loocv(
        X,
        y,
        target_name
    )

    all_summary.append(summary)
    all_folds.append(folds)


summary_df = pd.concat(
    all_summary,
    ignore_index=True
)

fold_df = pd.concat(
    all_folds,
    ignore_index=True
)


# ============================================================
# 10. DETERMINE BEST METHOD FOR EACH TARGET
# ============================================================

# Primary criterion:
# Highest R²
#
# Secondary criteria:
# Lowest RMSE and MAE

best_methods = []

for target in targets.keys():

    target_results = summary_df[
        summary_df["Target"] == target
    ].copy()

    target_results = target_results.sort_values(
        by=[
            "R2",
            "RMSE",
            "MAE"
        ],
        ascending=[
            False,
            True,
            True
        ]
    )

    best = target_results.iloc[0]

    best_methods.append({

        "Target": target,

        "Best Method": best["Method"],

        "R2": best["R2"],

        "RMSE": best["RMSE"],

        "MAE": best["MAE"],

        "MAPE (%)": best["MAPE (%)"],

        "Valid LOOCV Points":
            best["Valid LOOCV Points"]

    })


best_df = pd.DataFrame(
    best_methods
)


# ============================================================
# 11. PRINT RESULTS
# ============================================================

print("\n" + "=" * 70)
print("LOOCV SUMMARY")
print("=" * 70)

print(
    summary_df.to_string(
        index=False
    )
)

print("\n" + "=" * 70)
print("BEST METHOD FOR EACH TARGET")
print("=" * 70)

print(
    best_df.to_string(
        index=False
    )
)


# ============================================================
# 12. SAVE RESULTS
# ============================================================

print("\nSaving results...")

with pd.ExcelWriter(
    OUTPUT_FILE,
    engine="openpyxl"
) as writer:

    summary_df.to_excel(
        writer,
        sheet_name="LOOCV_Summary",
        index=False
    )

    best_df.to_excel(
        writer,
        sheet_name="Best_Methods",
        index=False
    )

    fold_df.to_excel(
        writer,
        sheet_name="Fold_Predictions",
        index=False
    )


# ============================================================
# 13. FINISHED
# ============================================================

print("\n" + "=" * 70)
print("STEP 2 COMPLETED")
print("=" * 70)

print("\nOutput:")
print(OUTPUT_FILE)

print(
    "\nThe 17 Test_Holdout observations were NOT used."
)

print(
    "\nNext step: inspect LOOCV results before generating "
    "the expanded training datasets."
)

print("=" * 70)