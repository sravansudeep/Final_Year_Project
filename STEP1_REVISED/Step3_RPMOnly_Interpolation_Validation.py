from pathlib import Path
import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
INPUT_FILE = BASE_DIR / "Step1_Stratified_Split.xlsx"
OUTPUT_FILE = BASE_DIR / "Step3_RPMOnly_Interpolation_Validation.xlsx"

TARGETS = ["hc_hi", "td_to", "Vch"]
REQUIRED_COLUMNS = ["Point_ID", "d", "Rpm"] + TARGETS

METHODS = {
    "Linear Griddata": "linear_griddata",
    "Cubic Griddata": "cubic_griddata",
    "1st-order RSM": "rsm1",
    "2nd-order RSM": "rsm2",
    "3rd-order RSM": "rsm3",
}

MEASURED_DIAMETERS = [6, 8, 10, 12, 14, 16]
MEASURED_RPM = list(range(0, 261, 20))
GRID_RPM = list(range(0, 261, 5))

def build_terms(degree):
    terms = []
    for i in range(degree + 1):
        for j in range(degree + 1 - i):
            terms.append((i, j))
    return terms

def fit_rsm(df, target, degree):
    d = df["d"].to_numpy(dtype=float)
    rpm = df["Rpm"].to_numpy(dtype=float)
    y = df[target].to_numpy(dtype=float)

    dc = float(np.mean(d))
    rc = float(np.mean(rpm))
    ds = float(np.std(d))
    rs = float(np.std(rpm))

    if ds == 0:
        ds = 1.0
    if rs == 0:
        rs = 1.0

    x1 = (d - dc) / ds
    x2 = (rpm - rc) / rs

    terms = build_terms(degree)
    X = np.column_stack([(x1 ** i) * (x2 ** j) for i, j in terms])
    beta, _, rank, _ = np.linalg.lstsq(X, y, rcond=None)

    return {
        "beta": beta,
        "terms": terms,
        "dc": dc,
        "rc": rc,
        "ds": ds,
        "rs": rs,
        "rank": int(rank),
    }

def predict_rsm(model, d, rpm):
    d = np.asarray(d, dtype=float)
    rpm = np.asarray(rpm, dtype=float)

    x1 = (d - model["dc"]) / model["ds"]
    x2 = (rpm - model["rc"]) / model["rs"]

    X = np.column_stack([
        (x1 ** i) * (x2 ** j)
        for i, j in model["terms"]
    ])
    return X @ model["beta"]

def make_griddata_predictor(df, target, method):
    from scipy.interpolate import griddata

    points = df[["d", "Rpm"]].to_numpy(dtype=float)
    values = df[target].to_numpy(dtype=float)

    def predict(d, rpm):
        query = np.column_stack([
            np.asarray(d, dtype=float),
            np.asarray(rpm, dtype=float)
        ])
        return griddata(points, values, query, method=method)

    return predict

def make_rsm_predictor(df, target, degree):
    model = fit_rsm(df, target, degree)
    return lambda d, rpm: predict_rsm(model, d, rpm)

def get_predictor(df, target, method):
    if method == "Linear Griddata":
        return make_griddata_predictor(df, target, "linear")
    if method == "Cubic Griddata":
        return make_griddata_predictor(df, target, "cubic")
    if method == "1st-order RSM":
        return make_rsm_predictor(df, target, 1)
    if method == "2nd-order RSM":
        return make_rsm_predictor(df, target, 2)
    if method == "3rd-order RSM":
        return make_rsm_predictor(df, target, 3)
    raise ValueError(f"Unknown method: {method}")

def metrics(actual, predicted):
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)

    mask = np.isfinite(actual) & np.isfinite(predicted)
    actual = actual[mask]
    predicted = predicted[mask]

    n = len(actual)
    if n == 0:
        return {
            "N_Valid": 0,
            "R2": np.nan,
            "RMSE": np.nan,
            "MAE": np.nan,
            "MSE": np.nan,
            "MAPE_percent": np.nan,
        }

    err = predicted - actual
    mse = float(np.mean(err ** 2))
    rmse = float(np.sqrt(mse))
    mae = float(np.mean(np.abs(err)))

    ss_res = float(np.sum(err ** 2))
    ss_tot = float(np.sum((actual - np.mean(actual)) ** 2))
    r2 = np.nan if ss_tot == 0 else 1.0 - ss_res / ss_tot

    nonzero = np.abs(actual) > 1e-12
    if np.any(nonzero):
        mape = float(
            np.mean(
                np.abs((actual[nonzero] - predicted[nonzero]) / actual[nonzero])
            ) * 100.0
        )
    else:
        mape = np.nan

    return {
        "N_Valid": n,
        "R2": r2,
        "RMSE": rmse,
        "MAE": mae,
        "MSE": mse,
        "MAPE_percent": mape,
    }

def physical_check(target, predicted):
    predicted = np.asarray(predicted, dtype=float)

    if target in {"hc_hi", "Vch"}:
        invalid = (~np.isfinite(predicted)) | (predicted < 0) | (predicted > 1)
    elif target == "td_to":
        invalid = (~np.isfinite(predicted)) | (predicted <= 0)
    else:
        invalid = ~np.isfinite(predicted)

    n = int(np.sum(invalid))
    pct = 100.0 * n / len(predicted) if len(predicted) else np.nan
    return n, pct

def load_data():
    xlsx = pd.ExcelFile(INPUT_FILE)

    if "Train_Anchors" not in xlsx.sheet_names:
        raise ValueError("Train_Anchors sheet missing.")
    if "Test_Holdout" not in xlsx.sheet_names:
        raise ValueError("Test_Holdout sheet missing.")

    train = pd.read_excel(INPUT_FILE, sheet_name="Train_Anchors")
    test = pd.read_excel(INPUT_FILE, sheet_name="Test_Holdout")

    missing = [c for c in REQUIRED_COLUMNS if c not in train.columns]
    if missing:
        raise ValueError(f"Missing columns in Train_Anchors: {missing}")

    train["d"] = pd.to_numeric(train["d"], errors="raise")
    train["Rpm"] = pd.to_numeric(train["Rpm"], errors="raise")

    return train.copy(), test.copy()

def eligible_interpolation_points(train):
    train_keys = set(zip(train["d"].astype(float), train["Rpm"].astype(float)))

    eligible = []

    for _, row in train.iterrows():
        d = float(row["d"])
        rpm = int(row["Rpm"])

        lower = (d, rpm - 20)
        upper = (d, rpm + 20)

        if lower in train_keys and upper in train_keys:
            eligible.append(row)

    eligible_df = pd.DataFrame(eligible).reset_index(drop=True)
    return eligible_df

def run_interpolation_cv(train, target):
    eligible = eligible_interpolation_points(train)
    summary_rows = []
    prediction_rows = []

    for method in METHODS:
        actual_all = []
        pred_all = []

        for _, row in eligible.iterrows():
            fit_df = train[train["Point_ID"] != row["Point_ID"]].copy()

            predictor = get_predictor(fit_df, target, method)
            pred = float(
                np.asarray(
                    predictor(
                        np.array([row["d"]], dtype=float),
                        np.array([row["Rpm"]], dtype=float)
                    )
                )[0]
            )

            prediction_rows.append({
                "Target": target,
                "Method": method,
                "Point_ID": row["Point_ID"],
                "d": row["d"],
                "Rpm": row["Rpm"],
                "Actual": row[target],
                "Predicted": pred,
                "Residual": pred - row[target],
            })

            actual_all.append(row[target])
            pred_all.append(pred)

        actual_all = np.asarray(actual_all, dtype=float)
        pred_all = np.asarray(pred_all, dtype=float)

        met = metrics(actual_all, pred_all)
        n_phys, pct_phys = physical_check(target, pred_all)

        summary_rows.append({
            "Target": target,
            "Method": method,
            "N_Eligible_Folds": len(eligible),
            "N_Valid_Folds": met["N_Valid"],
            "N_Invalid_Folds": len(eligible) - met["N_Valid"],
            "Coverage_percent": 100.0 * met["N_Valid"] / len(eligible) if len(eligible) else np.nan,
            "Physical_Violations": n_phys,
            "Physical_Violation_percent": pct_phys,
            "R2": met["R2"],
            "RMSE": met["RMSE"],
            "MAE": met["MAE"],
            "MSE": met["MSE"],
            "MAPE_percent": met["MAPE_percent"],
        })

    return pd.DataFrame(summary_rows), pd.DataFrame(prediction_rows), eligible

def rank_methods(summary):
    ranked = []

    for target, group in summary.groupby("Target"):
        g = group.copy()

        g["Full_Coverage"] = g["Coverage_percent"] >= 99.999999
        g = g.sort_values(
            [
                "Full_Coverage",
                "Physical_Violation_percent",
                "RMSE",
                "MAE",
                "R2",
            ],
            ascending=[False, True, True, True, False]
        ).reset_index(drop=True)

        g["Rank"] = np.arange(1, len(g) + 1)
        g["Status"] = np.where(
            g["Rank"] == 1,
            "PROVISIONAL - REVIEW BEFORE LOCKING",
            "COMPARATOR"
        )

        ranked.append(g)

    return pd.concat(ranked, ignore_index=True)

def build_rpm_only_grid(train, test, ranked):
    locked = ranked[ranked["Rank"] == 1].copy()

    test_coords = set(
        zip(
            test["d"].astype(float),
            test["Rpm"].astype(float)
        )
    )

    rows = []

    for _, selection in locked.iterrows():
        target = selection["Target"]
        method = selection["Method"]

        predictor = get_predictor(train, target, method)

        for d in MEASURED_DIAMETERS:
            for rpm in GRID_RPM:
                coord = (float(d), float(rpm))

                if coord in test_coords:
                    continue

                is_measured_train = (
                    ((train["d"].astype(float) == d) &
                     (train["Rpm"].astype(float) == rpm)).any()
                )

                pred = float(
                    np.asarray(
                        predictor(
                            np.array([d], dtype=float),
                            np.array([rpm], dtype=float)
                        )
                    )[0]
                )

                if not np.isfinite(pred):
                    status = "INVALID_OR_UNAVAILABLE"
                    synthetic = False
                elif is_measured_train:
                    status = "MEASURED_TRAIN_POINT"
                    synthetic = False
                else:
                    status = "SYNTHETIC_IN_RANGE_RPM_ONLY"
                    synthetic = True

                n_phys, pct_phys = physical_check(target, np.array([pred]))

                rows.append({
                    "Target": target,
                    "Method": method,
                    "d": d,
                    "Rpm": rpm,
                    "Predicted": pred,
                    "Status": status,
                    "Synthetic_In_Range": synthetic,
                    "Physical_Violation": n_phys > 0,
                })

    return pd.DataFrame(rows)

def grid_coverage(grid_df, test):
    test_coords = set(
        zip(
            test["d"].astype(float),
            test["Rpm"].astype(float)
        )
    )

    requested = len(MEASURED_DIAMETERS) * len(GRID_RPM)
    excluded = len([
        (d, r)
        for d, r in test_coords
        if d in MEASURED_DIAMETERS and r in GRID_RPM
    ])

    rows = []

    for (target, method), group in grid_df.groupby(["Target", "Method"]):
        valid = np.isfinite(group["Predicted"].to_numpy(dtype=float))
        synthetic = group["Synthetic_In_Range"].astype(bool)

        rows.append({
            "Target": target,
            "Method": method,
            "Requested_Grid_Points": requested,
            "Test_Coordinates_Excluded": excluded,
            "Non_Test_Grid_Points": requested - excluded,
            "Measured_Train_Points": int(np.sum(group["Status"] == "MEASURED_TRAIN_POINT")),
            "Synthetic_In_Range_Points": int(np.sum(synthetic)),
            "Valid_Output_Points": int(np.sum(valid)),
            "Invalid_or_Unavailable_Points": int(np.sum(~valid)),
            "Coverage_percent_of_non_test_grid": (
                100.0 * np.sum(valid) / (requested - excluded)
                if requested - excluded else np.nan
            ),
        })

    return pd.DataFrame(rows)

def main():
    print("=" * 78)
    print("STEP 3 - REVISED RPM-ONLY INTERPOLATION VALIDATION")
    print("=" * 78)
    print(f"Input:  {INPUT_FILE}")
    print(f"Output: {OUTPUT_FILE}")
    print("-" * 78)
    print("INTERPOLATION DOMAIN")
    print("-" * 78)
    print(f"Diameters used for interpolation grid : {MEASURED_DIAMETERS}")
    print(f"RPM measured levels                   : {MEASURED_RPM}")
    print(f"RPM interpolation grid                : {GRID_RPM[0]}-{GRID_RPM[-1]} step 5")
    print("No intermediate diameters are generated.")
    print()

    train, test = load_data()

    if len(train) != 67:
        raise ValueError(f"Expected 67 Train_Anchors rows, found {len(train)}")
    if len(test) != 17:
        raise ValueError(f"Expected 17 Test_Holdout rows, found {len(test)}")

    all_summary = []
    all_predictions = []
    eligible_reference = None

    for target in TARGETS:
        summary, predictions, eligible = run_interpolation_cv(train, target)
        all_summary.append(summary)
        all_predictions.append(predictions)

        if eligible_reference is None:
            eligible_reference = eligible.copy()

    summary_df = pd.concat(all_summary, ignore_index=True)
    predictions_df = pd.concat(all_predictions, ignore_index=True)

    ranking_df = rank_methods(summary_df)
    locked_candidates = ranking_df[ranking_df["Rank"] == 1].copy()

    grid_df = build_rpm_only_grid(train, test, ranking_df)
    coverage_df = grid_coverage(grid_df, test)

    candidate_df = locked_candidates[
        [
            "Target",
            "Method",
            "Full_Coverage",
            "Coverage_percent",
            "RMSE",
            "MAE",
            "R2",
            "Physical_Violation_percent",
            "Status",
        ]
    ].copy()

    protocol_df = pd.DataFrame({
        "Item": [
            "Train_Anchors",
            "Test_Holdout used",
            "Synthetic data used for interpolation CV",
            "Eligible interpolation folds",
            "Diameter domain",
            "RPM measured domain",
            "RPM interpolation spacing",
            "Intermediate diameters generated",
            "Target names",
            "Test coordinates excluded from grid",
            "Selection rule",
        ],
        "Value": [
            len(train),
            0,
            0,
            len(eligible_reference),
            str(MEASURED_DIAMETERS),
            "0-260 RPM at 20 RPM increments",
            "5 RPM",
            "NO",
            str(TARGETS),
            "YES",
            "Coverage, physical validity, RMSE, MAE, then R2; provisional until final holdout validation",
        ]
    })

    method_notes = pd.DataFrame({
        "Method": list(METHODS.keys()),
        "Role": [
            "In-range interpolation comparator",
            "In-range interpolation comparator",
            "Global response-surface comparator",
            "Global response-surface comparator",
            "Global response-surface comparator",
        ],
        "Note": [
            "May become unavailable outside the training convex hull.",
            "May become unavailable outside the training convex hull.",
            "Polynomial model used in input-space; can support later extrapolation.",
            "Polynomial model used in input-space; can support later extrapolation.",
            "Polynomial model used in input-space; can support later extrapolation.",
        ]
    })

    validation_log = pd.DataFrame({
        "Check": [
            "Train rows",
            "Test rows",
            "Eligible folds",
            "Intermediate diameter generation",
            "Synthetic points used during CV",
            "RPM-only grid total points",
            "Holdout coordinates inserted into grid",
        ],
        "Result": [
            len(train),
            len(test),
            len(eligible_reference),
            "NO",
            0,
            len(MEASURED_DIAMETERS) * len(GRID_RPM),
            "NO",
        ]
    })

    with pd.ExcelWriter(OUTPUT_FILE, engine="openpyxl") as writer:
        summary_df.to_excel(writer, sheet_name="Interpolation_CV_Summary", index=False)
        ranking_df.to_excel(writer, sheet_name="Method_Ranking", index=False)
        candidate_df.to_excel(writer, sheet_name="Provisional_Candidates", index=False)
        eligible_reference.to_excel(writer, sheet_name="Eligible_Folds", index=False)
        predictions_df.to_excel(writer, sheet_name="Fold_Predictions", index=False)
        coverage_df.to_excel(writer, sheet_name="RPMOnly_Grid_Coverage", index=False)
        grid_df.to_excel(writer, sheet_name="RPMOnly_Grid", index=False)
        protocol_df.to_excel(writer, sheet_name="Protocol", index=False)
        method_notes.to_excel(writer, sheet_name="Method_Notes", index=False)
        validation_log.to_excel(writer, sheet_name="Validation_Log", index=False)

    print("-" * 78)
    print("DATA PROTOCOL")
    print("-" * 78)
    print(f"Train_Anchors used : {len(train)}")
    print("Test_Holdout used  : 0")
    print("Synthetic data used: 0")
    print(f"Eligible folds     : {len(eligible_reference)}")

    print("-" * 78)
    print("INTERPOLATION CV SUMMARY")
    print("-" * 78)
    print(summary_df.to_string(index=False))

    print("-" * 78)
    print("PROVISIONAL CANDIDATES")
    print("-" * 78)
    print(candidate_df.to_string(index=False))

    print("-" * 78)
    print("RPM-ONLY GRID COVERAGE")
    print("-" * 78)
    print(coverage_df.to_string(index=False))

    print("-" * 78)
    print("COMPLETED")
    print("-" * 78)
    print("Only measured experimental diameters are included in the interpolation grid.")
    print("No d=7, 9, 11, 13, or 15 mm synthetic points are generated.")
    print("The 17 Test_Holdout coordinates remain excluded.")
    print("Next step: review this RPM-only Step 3 before proceeding to Step 4.")

if __name__ == "__main__":
    main()
