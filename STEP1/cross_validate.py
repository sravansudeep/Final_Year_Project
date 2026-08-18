import numpy as np
import pandas as pd

from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    r2_score,
    mean_absolute_error,
    mean_squared_error
)

from scipy.interpolate import griddata


# ============================================================
# SETTINGS
# ============================================================

file_path = "edited_excel.xlsx"
sheet_name = "Selected_columns"


# ============================================================
# 1. READ EXCEL DATA
# ============================================================

df = pd.read_excel(
    file_path,
    sheet_name=sheet_name
)

df = df[["d", "rpm", "Vmin"]].copy()

df["d"] = pd.to_numeric(
    df["d"],
    errors="coerce"
)

df["rpm"] = pd.to_numeric(
    df["rpm"],
    errors="coerce"
)

df["Vmin"] = pd.to_numeric(
    df["Vmin"],
    errors="coerce"
)

df = df.dropna()

print("=" * 90)
print("DATA")
print("=" * 90)

print("Total raw observations:", len(df))

print("\nUnique d values:")
print(sorted(df["d"].unique()))

print("\nUnique rpm values:")
print(sorted(df["rpm"].unique()))


# ============================================================
# 2. ORIGINAL 84-POINT DATA
# ============================================================

X_raw = df[["d", "rpm"]].values.astype(float)
y_raw = df["Vmin"].values.astype(float)

print("\nOriginal observations:", len(y_raw))


# ============================================================
# 3. CREATE UNIQUE GRID FOR GRIDDATA
# ============================================================
#
# d = 12 has 3 repeated measurements at each RPM.
# Griddata needs one Vmin value for each (d, rpm)
# coordinate, so we average the replicates.
#

grid_df = (
    df.groupby(
        ["d", "rpm"],
        as_index=False
    )["Vmin"]
    .mean()
)

X_grid = grid_df[
    ["d", "rpm"]
].values.astype(float)

y_grid = grid_df[
    "Vmin"
].values.astype(float)

print("\nUnique (d, rpm) locations:", len(y_grid))


# ============================================================
# 4. POLYNOMIAL LOOCV
# ============================================================

def polynomial_loocv(X, y, degree):

    preds = np.zeros(len(y))

    for i in range(len(y)):

        mask = np.ones(
            len(y),
            dtype=bool
        )

        mask[i] = False

        poly = PolynomialFeatures(
            degree=degree
        )

        X_train = poly.fit_transform(
            X[mask]
        )

        X_test = poly.transform(
            X[i].reshape(1, -1)
        )

        model = LinearRegression()

        model.fit(
            X_train,
            y[mask]
        )

        preds[i] = model.predict(
            X_test
        )[0]

    return preds


# ============================================================
# 5. GRIDDATA LOOCV
# ============================================================

def griddata_loocv(X, y, method):

    preds = np.full(
        len(y),
        np.nan
    )

    for i in range(len(y)):

        mask = np.ones(
            len(y),
            dtype=bool
        )

        mask[i] = False

        X_train = X[mask]
        y_train = y[mask]

        try:

            prediction = griddata(
                X_train,
                y_train,
                X[i],
                method=method
            )

            if prediction is not None:

                prediction = float(
                    np.asarray(prediction).ravel()[0]
                )

                if not np.isnan(prediction):

                    preds[i] = prediction

        except Exception:

            pass

    return preds


# ============================================================
# 6. RUN ALL FIVE MODELS
# ============================================================

print("\n" + "=" * 90)
print("RUNNING MODELS")
print("=" * 90)

print("\nPolynomial Degree 1...")
pred_poly1 = polynomial_loocv(
    X_grid,
    y_grid,
    degree=1
)

print("Polynomial Degree 2...")
pred_poly2 = polynomial_loocv(
    X_grid,
    y_grid,
    degree=2
)

print("Polynomial Degree 3...")
pred_poly3 = polynomial_loocv(
    X_grid,
    y_grid,
    degree=3
)

print("Griddata Linear...")
pred_grid_linear = griddata_loocv(
    X_grid,
    y_grid,
    method="linear"
)

print("Griddata Cubic...")
pred_grid_cubic = griddata_loocv(
    X_grid,
    y_grid,
    method="cubic"
)


# ============================================================
# 7. FIND COMMON VALIDATION POINTS
# ============================================================
#
# Griddata cannot predict 4 boundary points when those
# points are removed during LOOCV.
#
# We therefore compare ALL FIVE MODELS on the same
# points where both griddata methods produce predictions.
#

valid_linear = ~np.isnan(
    pred_grid_linear
)

valid_cubic = ~np.isnan(
    pred_grid_cubic
)

common_valid = (
    valid_linear &
    valid_cubic
)

print("\n" + "=" * 90)
print("VALIDATION POINTS")
print("=" * 90)

print(
    "Total unique locations:",
    len(y_grid)
)

print(
    "Griddata linear valid:",
    np.sum(valid_linear)
)

print(
    "Griddata cubic valid:",
    np.sum(valid_cubic)
)

print(
    "Common valid points:",
    np.sum(common_valid)
)


# ============================================================
# 8. METRIC FUNCTION
# ============================================================

def calculate_metrics(y, preds):

    r2 = r2_score(
        y,
        preds
    )

    rmse = np.sqrt(
        mean_squared_error(
            y,
            preds
        )
    )

    mae = mean_absolute_error(
        y,
        preds
    )

    mse = mean_squared_error(
        y,
        preds
    )

    nonzero = y != 0

    if np.sum(nonzero) > 0:

        mape = np.mean(
            np.abs(
                (
                    y[nonzero] -
                    preds[nonzero]
                )
                /
                y[nonzero]
            )
        ) * 100

    else:

        mape = np.nan

    return (
        r2,
        rmse,
        mae,
        mse,
        mape
    )


# ============================================================
# 9. COMMON-SET METRICS
# ============================================================

y_common = y_grid[common_valid]

pred_poly1_common = (
    pred_poly1[common_valid]
)

pred_poly2_common = (
    pred_poly2[common_valid]
)

pred_poly3_common = (
    pred_poly3[common_valid]
)

pred_grid_linear_common = (
    pred_grid_linear[common_valid]
)

pred_grid_cubic_common = (
    pred_grid_cubic[common_valid]
)


m1 = calculate_metrics(
    y_common,
    pred_poly1_common
)

m2 = calculate_metrics(
    y_common,
    pred_poly2_common
)

m3 = calculate_metrics(
    y_common,
    pred_poly3_common
)

m4 = calculate_metrics(
    y_common,
    pred_grid_linear_common
)

m5 = calculate_metrics(
    y_common,
    pred_grid_cubic_common
)


# ============================================================
# 10. FINAL MODEL COMPARISON TABLE
# ============================================================

results = pd.DataFrame(
    [
        [
            "Polynomial Degree 1",
            *m1,
            np.sum(common_valid),
            len(y_grid)
        ],

        [
            "Polynomial Degree 2",
            *m2,
            np.sum(common_valid),
            len(y_grid)
        ],

        [
            "Polynomial Degree 3",
            *m3,
            np.sum(common_valid),
            len(y_grid)
        ],

        [
            "Griddata Linear",
            *m4,
            np.sum(common_valid),
            len(y_grid)
        ],

        [
            "Griddata Cubic",
            *m5,
            np.sum(common_valid),
            len(y_grid)
        ]

    ],

    columns=[
        "Method",
        "R2",
        "RMSE",
        "MAE",
        "MSE",
        "MAPE (%)",
        "Valid Predictions",
        "Total Locations"
    ]
)


# ============================================================
# 11. DISPLAY RESULTS
# ============================================================

print("\n" + "=" * 100)
print("FINAL MODEL COMPARISON")
print("=" * 100)

print(
    results.to_string(
        index=False,
        float_format=lambda x: f"{x:.6f}"
    )
)


# ============================================================
# 12. BEST MODEL FOR EACH METRIC
# ============================================================

print("\n" + "=" * 100)
print("BEST MODELS")
print("=" * 100)

best_r2 = results.loc[
    results["R2"].idxmax(),
    "Method"
]

best_rmse = results.loc[
    results["RMSE"].idxmin(),
    "Method"
]

best_mae = results.loc[
    results["MAE"].idxmin(),
    "Method"
]

best_mse = results.loc[
    results["MSE"].idxmin(),
    "Method"
]

best_mape = results.loc[
    results["MAPE (%)"].idxmin(),
    "Method"
]

print("Best R2   :", best_r2)
print("Best RMSE :", best_rmse)
print("Best MAE  :", best_mae)
print("Best MSE  :", best_mse)
print("Best MAPE :", best_mape)


# ============================================================
# 13. SAVE RESULTS
# ============================================================

results.to_excel(
    "model_comparison_results_fixed.xlsx",
    index=False
)


# ============================================================
# 14. SAVE PREDICTIONS
# ============================================================

prediction_table = pd.DataFrame({
    "d": X_grid[:, 0],
    "rpm": X_grid[:, 1],
    "Actual Vmin": y_grid,
    "Poly Degree 1": pred_poly1,
    "Poly Degree 2": pred_poly2,
    "Poly Degree 3": pred_poly3,
    "Griddata Linear": pred_grid_linear,
    "Griddata Cubic": pred_grid_cubic
})

prediction_table.to_excel(
    "model_predictions.xlsx",
    index=False
)


print("\n" + "=" * 100)
print("FILES SAVED")
print("=" * 100)

print("model_comparison_results.xlsx")
print("model_predictions.xlsx")