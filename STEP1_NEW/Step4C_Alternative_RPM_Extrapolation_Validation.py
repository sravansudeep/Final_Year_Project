"""
STEP 4C - ALTERNATIVE RPM EXTRAPOLATION VALIDATION

Purpose
-------
Investigate whether hc and A can be extrapolated beyond the experimentally
available RPM range more reliably than with the polynomial RSMs tested in
Steps 4A and 4B.

IMPORTANT:
- Uses ONLY the 67 Train_Anchors from Step1_Stratified_Split.xlsx.
- The 17 final experimental holdout observations are NEVER used.
- The 219 synthetic interpolation points are NEVER used.
- No 280/300/320 RPM predictions are generated in this step.
- Validation is progressive pseudo-extrapolation.

Why a separate 1-D RPM model?
------------------------------
The immediate extrapolation problem is specifically along RPM while port
diameter is held at one of the six experimentally studied diameters. Therefore
this step evaluates RPM functional forms separately at each fixed diameter.
This avoids asking a global 2-D polynomial to extrapolate in RPM and lets us
test whether the observed RPM trajectory has a better-behaved functional form.

Candidate models:
1. Linear
2. Quadratic
3. Cubic
4. Logarithmic: a + b*log(1 + RPM)
5. Square-root: a + b*sqrt(RPM)
6. Saturating: a + b*RPM/(c + RPM)

The nonlinear models are fitted with scipy.curve_fit. Failed fits are recorded
rather than silently removed.

Progressive validation:
Scenario 1: withhold highest 1 available training RPM per diameter
Scenario 2: withhold highest 2 available training RPMs per diameter
Scenario 3: withhold highest 3 available training RPMs per diameter

Primary decision criterion:
- Progressive extrapolation performance
- RMSE / MAE
- R2 where meaningful
- MAPE
- Physical validity
- Stability as extrapolation depth increases

The model is NOT selected merely because it has the best in-domain fit.
"""

from pathlib import Path
import warnings
import numpy as np
import pandas as pd

from scipy.optimize import curve_fit
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import (
    r2_score,
    mean_squared_error,
    mean_absolute_error,
)


# ============================================================
# SETTINGS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

INPUT_FILE = BASE_DIR / "Step1_Stratified_Split.xlsx"
OUTPUT_FILE = BASE_DIR / "Step4C_Alternative_RPM_Extrapolation_Validation.xlsx"

TARGETS = ["hc", "A"]

SCENARIOS = {
    "Scenario_1_Highest_1": 1,
    "Scenario_2_Highest_2": 2,
    "Scenario_3_Highest_3": 3,
}

MODEL_NAMES = [
    "Linear",
    "Quadratic",
    "Cubic",
    "Logarithmic",
    "Square_Root",
    "Saturating",
]


# ============================================================
# MODEL DEFINITIONS
# ============================================================

def linear_func(x, a, b):
    return a + b * x


def log_func(x, a, b):
    return a + b * np.log1p(x)


def sqrt_func(x, a, b):
    return a + b * np.sqrt(x)


def saturating_func(x, a, b, c):
    return a + b * x / (c + x)


NONLINEAR_FUNCTIONS = {
    "Logarithmic": log_func,
    "Square_Root": sqrt_func,
    "Saturating": saturating_func,
}


# ============================================================
# FIT / PREDICT FUNCTIONS
# ============================================================

def fit_model(x, y, model_name):
    """
    Fit one RPM-only functional model.

    Returns a model package suitable for prediction.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    if model_name == "Linear":
        model = LinearRegression()
        model.fit(x.reshape(-1, 1), y)
        return ("sklearn", model)

    if model_name == "Quadratic":
        poly = PolynomialFeatures(
            degree=2,
            include_bias=False
        )
        X = poly.fit_transform(x.reshape(-1, 1))

        model = LinearRegression()
        model.fit(X, y)

        return ("poly", poly, model)

    if model_name == "Cubic":
        poly = PolynomialFeatures(
            degree=3,
            include_bias=False
        )
        X = poly.fit_transform(x.reshape(-1, 1))

        model = LinearRegression()
        model.fit(X, y)

        return ("poly", poly, model)

    func = NONLINEAR_FUNCTIONS[model_name]

    # Conservative initial guesses.
    a0 = float(y[0])
    b0 = float((y[-1] - y[0]) / max(x[-1] - x[0], 1.0))

    if model_name == "Logarithmic":
        p0 = [a0, b0]
        params, _ = curve_fit(
            func,
            x,
            y,
            p0=p0,
            maxfev=50000
        )
        return ("curve", func, params)

    if model_name == "Square_Root":
        p0 = [a0, b0]
        params, _ = curve_fit(
            func,
            x,
            y,
            p0=p0,
            maxfev=50000
        )
        return ("curve", func, params)

    if model_name == "Saturating":
        # c must remain positive to avoid a singularity at positive RPM.
        p0 = [a0, max(y[-1] - y[0], 1e-6), 50.0]

        lower = [-np.inf, -np.inf, 1e-6]
        upper = [np.inf, np.inf, np.inf]

        params, _ = curve_fit(
            func,
            x,
            y,
            p0=p0,
            bounds=(lower, upper),
            maxfev=100000
        )

        return ("curve", func, params)

    raise ValueError(f"Unknown model: {model_name}")


def predict_model(package, x):
    x = np.asarray(x, dtype=float)

    kind = package[0]

    if kind == "sklearn":
        model = package[1]
        return model.predict(x.reshape(-1, 1))

    if kind == "poly":
        poly = package[1]
        model = package[2]
        return model.predict(
            poly.transform(x.reshape(-1, 1))
        )

    if kind == "curve":
        func = package[1]
        params = package[2]
        return func(x, *params)

    raise ValueError("Invalid model package.")


# ============================================================
# METRICS
# ============================================================

def calculate_metrics(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    finite = (
        np.isfinite(y_true)
        & np.isfinite(y_pred)
    )

    if finite.sum() == 0:
        return {
            "R2": np.nan,
            "RMSE": np.nan,
            "MAE": np.nan,
            "MSE": np.nan,
            "MAPE_percent": np.nan,
        }

    yt = y_true[finite]
    yp = y_pred[finite]

    mse = mean_squared_error(yt, yp)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(yt, yp)

    if len(yt) >= 2 and np.std(yt) > 1e-12:
        r2 = r2_score(yt, yp)
    else:
        r2 = np.nan

    nonzero = np.abs(yt) > 1e-12

    if np.any(nonzero):
        mape = (
            np.mean(
                np.abs(
                    (yt[nonzero] - yp[nonzero])
                    / yt[nonzero]
                )
            )
            * 100
        )
    else:
        mape = np.nan

    return {
        "R2": r2,
        "RMSE": rmse,
        "MAE": mae,
        "MSE": mse,
        "MAPE_percent": mape,
    }


# ============================================================
# PROGRESSIVE SPLIT
# ============================================================

def make_split(train_anchors, n_withhold):
    """
    Withhold the highest n AVAILABLE RPM anchors at each diameter.
    """
    fit_parts = []
    hold_parts = []

    for d in sorted(train_anchors["d"].unique()):

        group = (
            train_anchors[
                train_anchors["d"] == d
            ]
            .sort_values("Rpm")
            .copy()
        )

        if len(group) <= n_withhold:
            raise ValueError(
                f"Diameter {d} has only {len(group)} anchors."
            )

        hold = group.tail(n_withhold)
        fit = group.iloc[:-n_withhold]

        fit_parts.append(fit)
        hold_parts.append(hold)

    return (
        pd.concat(fit_parts, ignore_index=True),
        pd.concat(hold_parts, ignore_index=True),
    )


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 78)
print("STEP 4C - ALTERNATIVE RPM EXTRAPOLATION VALIDATION")
print("=" * 78)

if not INPUT_FILE.exists():
    raise FileNotFoundError(
        f"\nInput file not found:\n{INPUT_FILE}\n"
        "Place this script beside Step1_Stratified_Split.xlsx."
    )

train_anchors = pd.read_excel(
    INPUT_FILE,
    sheet_name="Train_Anchors"
)

required = [
    "Point_ID",
    "d",
    "Rpm",
    "hc",
    "A",
    "Vmin",
]

missing = [
    c for c in required
    if c not in train_anchors.columns
]

if missing:
    raise ValueError(
        f"Missing columns: {missing}"
    )

train_anchors = train_anchors[required].copy()

print(f"\nTraining anchors: {len(train_anchors)}")
print(
    f"Diameters: "
    f"{sorted(train_anchors['d'].unique())}"
)


# ============================================================
# VALIDATION
# ============================================================

overall_rows = []
diameter_rows = []
prediction_rows = []
physical_rows = []
fit_rows = []

for scenario_name, n_withhold in SCENARIOS.items():

    fit_df, hold_df = make_split(
        train_anchors,
        n_withhold
    )

    print("\n" + "-" * 78)
    print(scenario_name)
    print(
        f"Withholding highest {n_withhold} "
        f"available RPM anchor(s) per diameter"
    )
    print(
        f"Fit observations: {len(fit_df)}"
    )
    print(
        f"Held-out observations: {len(hold_df)}"
    )

    for target in TARGETS:

        for model_name in MODEL_NAMES:

            all_actual = []
            all_predicted = []

            successful_diameters = 0
            failed_diameters = []

            # Fit independently at each fixed diameter.
            for d in sorted(
                train_anchors["d"].unique()
            ):

                fit_d = (
                    fit_df[
                        fit_df["d"] == d
                    ]
                    .sort_values("Rpm")
                )

                hold_d = (
                    hold_df[
                        hold_df["d"] == d
                    ]
                    .sort_values("Rpm")
                )

                x_fit = fit_d["Rpm"].to_numpy(
                    dtype=float
                )
                y_fit = fit_d[target].to_numpy(
                    dtype=float
                )

                x_hold = hold_d["Rpm"].to_numpy(
                    dtype=float
                )
                y_hold = hold_d[target].to_numpy(
                    dtype=float
                )

                try:
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")

                        package = fit_model(
                            x_fit,
                            y_fit,
                            model_name
                        )

                    y_pred = predict_model(
                        package,
                        x_hold
                    )

                    if not np.all(
                        np.isfinite(y_pred)
                    ):
                        raise ValueError(
                            "Non-finite prediction."
                        )

                    successful_diameters += 1

                    all_actual.extend(
                        y_hold.tolist()
                    )
                    all_predicted.extend(
                        y_pred.tolist()
                    )

                    # Record fitted parameters in a readable form.
                    fit_rows.append({
                        "Scenario": scenario_name,
                        "Target": target,
                        "Model": model_name,
                        "Diameter_d": d,
                        "Fit_RPM_Min": x_fit.min(),
                        "Fit_RPM_Max": x_fit.max(),
                        "Withheld_RPM": ", ".join(
                            map(
                                str,
                                x_hold.astype(int)
                            )
                        ),
                        "Fit_Success": True,
                        "Parameters": repr(
                            package[-1]
                            if package[0] == "curve"
                            else "sklearn/polynomial model"
                        ),
                    })

                    # Individual predictions.
                    for i in range(len(x_hold)):

                        actual = float(
                            y_hold[i]
                        )
                        predicted = float(
                            y_pred[i]
                        )

                        if abs(actual) > 1e-12:
                            pe = (
                                abs(actual - predicted)
                                / abs(actual)
                                * 100
                            )
                        else:
                            pe = np.nan

                        prediction_rows.append({
                            "Scenario": scenario_name,
                            "Target": target,
                            "Model": model_name,
                            "Diameter_d": d,
                            "Rpm": x_hold[i],
                            "Actual": actual,
                            "Predicted": predicted,
                            "Residual": (
                                actual - predicted
                            ),
                            "Absolute_Error": (
                                abs(
                                    actual
                                    - predicted
                                )
                            ),
                            "Percent_Error": pe,
                        })

                except Exception as exc:

                    failed_diameters.append(
                        f"d={d}: {type(exc).__name__}: {exc}"
                    )

                    fit_rows.append({
                        "Scenario": scenario_name,
                        "Target": target,
                        "Model": model_name,
                        "Diameter_d": d,
                        "Fit_RPM_Min": (
                            x_fit.min()
                            if len(x_fit)
                            else np.nan
                        ),
                        "Fit_RPM_Max": (
                            x_fit.max()
                            if len(x_fit)
                            else np.nan
                        ),
                        "Withheld_RPM": ", ".join(
                            map(
                                str,
                                x_hold.astype(int)
                            )
                        ),
                        "Fit_Success": False,
                        "Parameters": str(exc),
                    })

            # Overall metrics across all successful diameter fits.
            y_true = np.asarray(
                all_actual,
                dtype=float
            )
            y_pred = np.asarray(
                all_predicted,
                dtype=float
            )

            if len(y_true) > 0:
                m = calculate_metrics(
                    y_true,
                    y_pred
                )
            else:
                m = {
                    "R2": np.nan,
                    "RMSE": np.nan,
                    "MAE": np.nan,
                    "MSE": np.nan,
                    "MAPE_percent": np.nan,
                }

            overall_rows.append({
                "Scenario": scenario_name,
                "Withheld_Highest_N": n_withhold,
                "Target": target,
                "Model": model_name,
                "N_Fit": len(fit_df),
                "N_Heldout": len(hold_df),
                "Successful_Diameters": successful_diameters,
                "Failed_Diameters": len(
                    failed_diameters
                ),
                "R2": m["R2"],
                "RMSE": m["RMSE"],
                "MAE": m["MAE"],
                "MSE": m["MSE"],
                "MAPE_percent": m["MAPE_percent"],
                "Failure_Details": " | ".join(
                    failed_diameters
                ),
            })

            # Physical validity across successful predictions.
            if len(y_pred) > 0:
                negative_count = int(
                    np.sum(y_pred < 0)
                )
                nonfinite_count = int(
                    np.sum(~np.isfinite(y_pred))
                )
                minimum = float(
                    np.min(y_pred)
                )
                maximum = float(
                    np.max(y_pred)
                )
            else:
                negative_count = np.nan
                nonfinite_count = np.nan
                minimum = np.nan
                maximum = np.nan

            physical_rows.append({
                "Scenario": scenario_name,
                "Target": target,
                "Model": model_name,
                "N_Predictions": len(y_pred),
                "Negative_Predictions": negative_count,
                "Nonfinite_Predictions": nonfinite_count,
                "Minimum_Prediction": minimum,
                "Maximum_Prediction": maximum,
                "Physically_Valid": (
                    False
                    if len(y_pred) == 0
                    else (
                        negative_count == 0
                        and nonfinite_count == 0
                    )
                ),
            })


# ============================================================
# MODEL RANKING
# ============================================================

overall_df = pd.DataFrame(overall_rows)
diameter_df = pd.DataFrame(diameter_rows)
prediction_df = pd.DataFrame(prediction_rows)
physical_df = pd.DataFrame(physical_rows)
fit_df_out = pd.DataFrame(fit_rows)

overall_df["RMSE_Rank"] = (
    overall_df
    .groupby(
        ["Scenario", "Target"]
    )["RMSE"]
    .rank(
        method="min",
        na_option="bottom"
    )
)

best_df = (
    overall_df
    .sort_values(
        [
            "Scenario",
            "Target",
            "RMSE",
            "MAE"
        ]
    )
    .groupby(
        ["Scenario", "Target"],
        as_index=False
    )
    .first()
)

# ------------------------------------------------------------
# Progressive stability table
# ------------------------------------------------------------

stability_rows = []

for target in TARGETS:
    for model in MODEL_NAMES:

        rows = overall_df[
            (overall_df["Target"] == target)
            & (overall_df["Model"] == model)
        ].sort_values("Withheld_Highest_N")

        if len(rows) == 3:
            rmse_values = rows["RMSE"].to_numpy(
                dtype=float
            )

            # A stable model should not show explosive
            # deterioration with increasing extrapolation depth.
            if np.all(np.isfinite(rmse_values)):
                rmse_growth_1_to_2 = (
                    rmse_values[1] / rmse_values[0]
                    if rmse_values[0] > 0
                    else np.nan
                )
                rmse_growth_2_to_3 = (
                    rmse_values[2] / rmse_values[1]
                    if rmse_values[1] > 0
                    else np.nan
                )
                rmse_growth_1_to_3 = (
                    rmse_values[2] / rmse_values[0]
                    if rmse_values[0] > 0
                    else np.nan
                )
            else:
                rmse_growth_1_to_2 = np.nan
                rmse_growth_2_to_3 = np.nan
                rmse_growth_1_to_3 = np.nan

            stability_rows.append({
                "Target": target,
                "Model": model,
                "Scenario_1_RMSE": rmse_values[0],
                "Scenario_2_RMSE": rmse_values[1],
                "Scenario_3_RMSE": rmse_values[2],
                "RMSE_Growth_1_to_2": rmse_growth_1_to_2,
                "RMSE_Growth_2_to_3": rmse_growth_2_to_3,
                "RMSE_Growth_1_to_3": rmse_growth_1_to_3,
            })

stability_df = pd.DataFrame(
    stability_rows
)


# ============================================================
# PRINT RESULTS
# ============================================================

print("\n" + "=" * 78)
print("OVERALL RESULTS")
print("=" * 78)

print(
    overall_df[
        [
            "Scenario",
            "Target",
            "Model",
            "R2",
            "RMSE",
            "MAE",
            "MAPE_percent",
        ]
    ]
    .sort_values(
        [
            "Scenario",
            "Target",
            "RMSE"
        ]
    )
    .to_string(
        index=False,
        float_format=lambda x: f"{x:.6f}"
    )
)

print("\n" + "=" * 78)
print("BEST MODEL BY SCENARIO AND TARGET")
print("=" * 78)

print(
    best_df[
        [
            "Scenario",
            "Target",
            "Model",
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

print("\n" + "=" * 78)
print("PROGRESSIVE RMSE STABILITY")
print("=" * 78)

print(
    stability_df.to_string(
        index=False,
        float_format=lambda x: f"{x:.6f}"
    )
)

print("\n" + "=" * 78)
print("PHYSICAL VALIDITY")
print("=" * 78)

print(
    physical_df.to_string(
        index=False
    )
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

    stability_df.to_excel(
        writer,
        sheet_name="Progressive_Stability",
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

    fit_df_out.to_excel(
        writer,
        sheet_name="Fit_Log",
        index=False
    )

print("\n" + "=" * 78)
print("COMPLETED")
print(f"Output: {OUTPUT_FILE}")
print("=" * 78)
