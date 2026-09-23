from pathlib import Path
import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
INPUT_FILE = BASE_DIR / "Step1_Stratified_Split.xlsx"
STEP3_FILE = BASE_DIR / "Step3_Interpolation_Validation.xlsx"
OUTPUT_FILE = BASE_DIR / "Step4_Progressive_Extrapolation_Validation.xlsx"

TARGETS = ["hc_hi", "td_to", "Vch"]
REQUIRED_COLUMNS = ["Point_ID", "d", "Rpm"] + TARGETS

METHODS = {
    "1st-order RSM": 1,
    "2nd-order RSM": 2,
    "3rd-order RSM": 3
}

ONE_STEP_CUTOFFS = [180, 200, 220, 240]
MULTI_STEP_CUTOFFS = [160, 180, 200]
RPM_FINAL = [280, 300, 320]
DIAMETERS_FINAL = [4, 18]
DIAMETERS_RPM_ONLY = [6, 8, 10, 12, 14, 16]

def build_terms(degree):
    terms = []
    for i in range(degree + 1):
        for j in range(degree + 1 - i):
            terms.append((i, j))
    return terms

def fit_rsm(df, target, degree):
    d = df["d"].to_numpy(dtype=float)
    r = df["Rpm"].to_numpy(dtype=float)
    y = df[target].to_numpy(dtype=float)

    d_center = float(np.mean(d))
    r_center = float(np.mean(r))
    d_scale = float(np.std(d))
    r_scale = float(np.std(r))

    if d_scale == 0:
        d_scale = 1.0
    if r_scale == 0:
        r_scale = 1.0

    x1 = (d - d_center) / d_scale
    x2 = (r - r_center) / r_scale

    terms = build_terms(degree)
    X = np.column_stack([(x1 ** i) * (x2 ** j) for i, j in terms])
    beta, _, rank, _ = np.linalg.lstsq(X, y, rcond=None)

    return {
        "beta": beta,
        "terms": terms,
        "d_center": d_center,
        "r_center": r_center,
        "d_scale": d_scale,
        "r_scale": r_scale,
        "rank": int(rank)
    }

def predict_rsm(model, d, rpm):
    d = np.asarray(d, dtype=float)
    rpm = np.asarray(rpm, dtype=float)

    x1 = (d - model["d_center"]) / model["d_scale"]
    x2 = (rpm - model["r_center"]) / model["r_scale"]

    X = np.column_stack([
        (x1 ** i) * (x2 ** j)
        for i, j in model["terms"]
    ])
    return X @ model["beta"]

def metrics(actual, predicted):
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)

    mask = np.isfinite(actual) & np.isfinite(predicted)
    a = actual[mask]
    p = predicted[mask]

    n = len(a)
    if n == 0:
        return {
            "N_Valid": 0,
            "R2": np.nan,
            "RMSE": np.nan,
            "MAE": np.nan,
            "MSE": np.nan,
            "MAPE_percent": np.nan
        }

    err = p - a
    mse = float(np.mean(err ** 2))
    rmse = float(np.sqrt(mse))
    mae = float(np.mean(np.abs(err)))

    ss_res = float(np.sum(err ** 2))
    ss_tot = float(np.sum((a - np.mean(a)) ** 2))
    r2 = np.nan if ss_tot == 0 else 1.0 - ss_res / ss_tot

    nonzero = np.abs(a) > 1e-12
    if np.any(nonzero):
        mape = float(np.mean(np.abs((a[nonzero] - p[nonzero]) / a[nonzero])) * 100.0)
    else:
        mape = np.nan

    return {
        "N_Valid": n,
        "R2": r2,
        "RMSE": rmse,
        "MAE": mae,
        "MSE": mse,
        "MAPE_percent": mape
    }

def physical_limits(target):
    if target == "hc_hi":
        return 0.0, 1.0
    if target == "Vch":
        return 0.0, 1.0
    return 0.0, np.inf

def physical_check(target, predictions):
    predictions = np.asarray(predictions, dtype=float)
    low, high = physical_limits(target)

    finite = np.isfinite(predictions)
    invalid = ~finite
    if low is not None:
        invalid = invalid | (predictions < low)
    if np.isfinite(high):
        invalid = invalid | (predictions > high)

    n_total = len(predictions)
    n_violations = int(np.sum(invalid))
    pct = np.nan if n_total == 0 else 100.0 * n_violations / n_total

    return n_violations, pct

def load_data():
    xlsx = pd.ExcelFile(INPUT_FILE)
    if "Train_Anchors" not in xlsx.sheet_names:
        raise ValueError("Train_Anchors sheet not found in Step1_Stratified_Split.xlsx")
    if "Test_Holdout" not in xlsx.sheet_names:
        raise ValueError("Test_Holdout sheet not found in Step1_Stratified_Split.xlsx")

    train = pd.read_excel(INPUT_FILE, sheet_name="Train_Anchors")
    test = pd.read_excel(INPUT_FILE, sheet_name="Test_Holdout")

    missing = [c for c in REQUIRED_COLUMNS if c not in train.columns]
    if missing:
        raise ValueError(f"Missing required columns in Train_Anchors: {missing}")

    missing_test = [c for c in ["Point_ID", "d", "Rpm"] if c not in test.columns]
    if missing_test:
        raise ValueError(f"Missing required columns in Test_Holdout: {missing_test}")

    train["Rpm"] = pd.to_numeric(train["Rpm"], errors="raise")
    train["d"] = pd.to_numeric(train["d"], errors="raise")

    return train.copy(), test.copy()

def run_one_step(train, target):
    rows = []
    predictions = []

    for cutoff in ONE_STEP_CUTOFFS:
        validation_rpm = cutoff + 20
        fit_df = train[train["Rpm"] <= cutoff].copy()
        val_df = train[train["Rpm"] == validation_rpm].copy()

        for method, degree in METHODS.items():
            if len(fit_df) < len(build_terms(degree)):
                continue
            if len(val_df) == 0:
                continue

            model = fit_rsm(fit_df, target, degree)
            pred = predict_rsm(
                model,
                val_df["d"].to_numpy(dtype=float),
                val_df["Rpm"].to_numpy(dtype=float)
            )

            met = metrics(val_df[target], pred)
            n_phys, pct_phys = physical_check(target, pred)

            for idx, (_, row) in enumerate(val_df.iterrows()):
                predictions.append({
                    "Validation_Mode": "One-Step",
                    "Target": target,
                    "Method": method,
                    "Train_Max_RPM": cutoff,
                    "Validation_RPM": validation_rpm,
                    "Point_ID": row["Point_ID"],
                    "d": row["d"],
                    "Rpm": row["Rpm"],
                    "Actual": row[target],
                    "Predicted": pred[idx],
                    "Residual": pred[idx] - row[target]
                })

            rows.append({
                "Validation_Mode": "One-Step",
                "Target": target,
                "Method": method,
                "Train_Max_RPM": cutoff,
                "Validation_RPM": validation_rpm,
                "N_Train": len(fit_df),
                "N_Validation": len(val_df),
                "N_Valid": met["N_Valid"],
                "R2": met["R2"],
                "RMSE": met["RMSE"],
                "MAE": met["MAE"],
                "MSE": met["MSE"],
                "MAPE_percent": met["MAPE_percent"],
                "Physical_Violations": n_phys,
                "Physical_Violation_percent": pct_phys,
                "Model_Rank": model["rank"]
            })

    fold_df = pd.DataFrame(rows)
    pred_df = pd.DataFrame(predictions)

    summary_rows = []
    for (target_name, method_name), group in fold_df.groupby(["Target", "Method"]):
        target_pred = pred_df[
            (pred_df["Target"] == target_name) &
            (pred_df["Method"] == method_name)
        ]

        pooled = metrics(target_pred["Actual"], target_pred["Predicted"])
        n_phys, pct_phys = physical_check(target_name, target_pred["Predicted"].to_numpy(dtype=float))

        summary_rows.append({
            "Validation_Mode": "One-Step Pooled",
            "Target": target_name,
            "Method": method_name,
            "N_Folds": group["Validation_RPM"].nunique(),
            "N_Valid_Folds": int(np.sum(group["N_Valid"] > 0)),
            "N_Validation_Points": len(target_pred),
            "N_Valid_Points": pooled["N_Valid"],
            "R2": pooled["R2"],
            "RMSE": pooled["RMSE"],
            "MAE": pooled["MAE"],
            "MSE": pooled["MSE"],
            "MAPE_percent": pooled["MAPE_percent"],
            "Physical_Violations": n_phys,
            "Physical_Violation_percent": pct_phys
        })

    return fold_df, pred_df, pd.DataFrame(summary_rows)

def run_multi_step(train, target):
    fold_rows = []
    prediction_rows = []

    for cutoff in MULTI_STEP_CUTOFFS:
        future_rpms = [rpm for rpm in range(cutoff + 20, 261, 20)]
        fit_df = train[train["Rpm"] <= cutoff].copy()

        for method, degree in METHODS.items():
            if len(fit_df) < len(build_terms(degree)):
                continue

            model = fit_rsm(fit_df, target, degree)

            for validation_rpm in future_rpms:
                val_df = train[train["Rpm"] == validation_rpm].copy()
                if len(val_df) == 0:
                    continue

                pred = predict_rsm(
                    model,
                    val_df["d"].to_numpy(dtype=float),
                    val_df["Rpm"].to_numpy(dtype=float)
                )

                met = metrics(val_df[target], pred)
                n_phys, pct_phys = physical_check(target, pred)

                fold_rows.append({
                    "Validation_Mode": "Multi-Step",
                    "Target": target,
                    "Method": method,
                    "Train_Max_RPM": cutoff,
                    "Validation_RPM": validation_rpm,
                    "N_Train": len(fit_df),
                    "N_Validation": len(val_df),
                    "N_Valid": met["N_Valid"],
                    "R2": met["R2"],
                    "RMSE": met["RMSE"],
                    "MAE": met["MAE"],
                    "MSE": met["MSE"],
                    "MAPE_percent": met["MAPE_percent"],
                    "Physical_Violations": n_phys,
                    "Physical_Violation_percent": pct_phys
                })

                for idx, (_, row) in enumerate(val_df.iterrows()):
                    prediction_rows.append({
                        "Validation_Mode": "Multi-Step",
                        "Target": target,
                        "Method": method,
                        "Train_Max_RPM": cutoff,
                        "Validation_RPM": validation_rpm,
                        "Point_ID": row["Point_ID"],
                        "d": row["d"],
                        "Rpm": row["Rpm"],
                        "Actual": row[target],
                        "Predicted": pred[idx],
                        "Residual": pred[idx] - row[target]
                    })

    fold_df = pd.DataFrame(fold_rows)
    pred_df = pd.DataFrame(prediction_rows)

    summary_rows = []
    for (target_name, method_name), group in fold_df.groupby(["Target", "Method"]):
        target_pred = pred_df[
            (pred_df["Target"] == target_name) &
            (pred_df["Method"] == method_name)
        ]

        pooled = metrics(target_pred["Actual"], target_pred["Predicted"])
        n_phys, pct_phys = physical_check(target_name, target_pred["Predicted"].to_numpy(dtype=float))

        summary_rows.append({
            "Validation_Mode": "Multi-Step Pooled",
            "Target": target_name,
            "Method": method_name,
            "N_Cutoffs": group["Train_Max_RPM"].nunique(),
            "N_Validation_Points": len(target_pred),
            "N_Valid_Points": pooled["N_Valid"],
            "R2": pooled["R2"],
            "RMSE": pooled["RMSE"],
            "MAE": pooled["MAE"],
            "MSE": pooled["MSE"],
            "MAPE_percent": pooled["MAPE_percent"],
            "Physical_Violations": n_phys,
            "Physical_Violation_percent": pct_phys
        })

    return fold_df, pred_df, pd.DataFrame(summary_rows)

def choose_extrapolation_methods(one_summary, multi_summary):
    rows = []

    for target in TARGETS:
        one = one_summary[one_summary["Target"] == target].copy()
        multi = multi_summary[multi_summary["Target"] == target].copy()

        merged = one.merge(
            multi[["Target", "Method", "R2", "RMSE", "MAE", "Physical_Violations", "Physical_Violation_percent"]],
            on=["Target", "Method"],
            suffixes=("_OneStep", "_MultiStep")
        )

        merged = merged.sort_values(
            [
                "Physical_Violations_OneStep",
                "Physical_Violations_MultiStep",
                "RMSE_OneStep",
                "RMSE_MultiStep",
                "MAE_OneStep",
                "MAE_MultiStep"
            ],
            ascending=[True, True, True, True, True, True]
        ).reset_index(drop=True)

        for rank, (_, row) in enumerate(merged.iterrows(), start=1):
            rows.append({
                "Target": target,
                "Method": row["Method"],
                "Rank": rank,
                "OneStep_R2": row["R2_OneStep"],
                "OneStep_RMSE": row["RMSE_OneStep"],
                "OneStep_MAE": row["MAE_OneStep"],
                "MultiStep_R2": row["R2_MultiStep"],
                "MultiStep_RMSE": row["RMSE_MultiStep"],
                "MultiStep_MAE": row["MAE_MultiStep"],
                "OneStep_Physical_Violations": int(row["Physical_Violations_OneStep"]),
                "MultiStep_Physical_Violations": int(row["Physical_Violations_MultiStep"]),
                "OneStep_Physical_Violation_percent": row["Physical_Violation_percent_OneStep"],
                "MultiStep_Physical_Violation_percent": row["Physical_Violation_percent_MultiStep"],
                "Status": "PROVISIONAL - REVIEW BEFORE LOCKING" if rank == 1 else "COMPARATOR"
            })

    return pd.DataFrame(rows)

def final_extrapolation(train, selection_df):
    selected = selection_df[selection_df["Rank"] == 1].copy()

    rows = []

    for _, sel in selected.iterrows():
        target = sel["Target"]
        method = sel["Method"]
        degree = METHODS[method]

        model = fit_rsm(train, target, degree)

        for d_value in DIAMETERS_FINAL:
            for rpm_value in RPM_FINAL:
                pred = float(predict_rsm(model, np.array([d_value]), np.array([rpm_value]))[0])

                in_d = 6 <= d_value <= 16
                in_rpm = 0 <= rpm_value <= 260

                if in_d and in_rpm:
                    flag = "IN_RANGE"
                elif (not in_d) and (not in_rpm):
                    flag = "OUTSIDE_DIAMETER_AND_RPM"
                elif not in_d:
                    flag = "OUTSIDE_DIAMETER"
                else:
                    flag = "OUTSIDE_RPM"

                rows.append({
                    "Target": target,
                    "Method": method,
                    "d": d_value,
                    "Rpm": rpm_value,
                    "Predicted": pred,
                    "Diameter_Extrapolation_mm": d_value - 6 if d_value < 6 else d_value - 16 if d_value > 16 else 0,
                    "RPM_Extrapolation": rpm_value - 260 if rpm_value > 260 else 0,
                    "Applicability_Flag": flag,
                    "Physical_Violation": bool(physical_check(target, np.array([pred]))[0] > 0),
                    "Status": "PROVISIONAL ENGINEERING EXTRAPOLATION"
                })

    return pd.DataFrame(rows)

def rpm_only_extrapolation(train, selection_df):
    selected = selection_df[selection_df["Rank"] == 1].copy()

    rows = []

    for _, sel in selected.iterrows():
        target = sel["Target"]
        method = sel["Method"]
        degree = METHODS[method]
        model = fit_rsm(train, target, degree)

        for d_value in DIAMETERS_RPM_ONLY:
            for rpm_value in RPM_FINAL:
                pred = float(predict_rsm(model, np.array([d_value]), np.array([rpm_value]))[0])
                rows.append({
                    "Target": target,
                    "Method": method,
                    "d": d_value,
                    "Rpm": rpm_value,
                    "Predicted": pred,
                    "Applicability_Flag": "OUTSIDE_RPM_ONLY",
                    "Physical_Violation": bool(physical_check(target, np.array([pred]))[0] > 0),
                    "Status": "PROVISIONAL REFERENCE - NOT FINAL DATASET"
                })

    return pd.DataFrame(rows)

def stability_checks(train, selection_df, final_df):
    rows = []

    for _, sel in selection_df[selection_df["Rank"] == 1].iterrows():
        target = sel["Target"]
        method = sel["Method"]
        degree = METHODS[method]
        model = fit_rsm(train, target, degree)

        eval_points = []
        for d_value in DIAMETERS_FINAL:
            for rpm_value in RPM_FINAL:
                eval_points.append((d_value, rpm_value))

        preds = np.array([
            float(predict_rsm(model, np.array([d]), np.array([r]))[0])
            for d, r in eval_points
        ])

        rows.append({
            "Target": target,
            "Method": method,
            "Prediction_Min": float(np.min(preds)),
            "Prediction_Max": float(np.max(preds)),
            "Prediction_Mean": float(np.mean(preds)),
            "Prediction_Std": float(np.std(preds)),
            "Max_Absolute_Prediction": float(np.max(np.abs(preds))),
            "Physical_Violations": int(np.sum([
                physical_check(target, np.array([p]))[0] for p in preds
            ]))
        })

    return pd.DataFrame(rows)

def main():
    print("=" * 78)
    print("STEP 4 - PROGRESSIVE HIGH-RPM EXTRAPOLATION VALIDATION")
    print("=" * 78)
    print(f"Input:  {INPUT_FILE}")
    print(f"Step 3: {STEP3_FILE}")
    print(f"Output: {OUTPUT_FILE}")

    train, test = load_data()

    if len(train) != 67:
        raise ValueError(f"Expected 67 Train_Anchors rows, found {len(train)}")

    if len(test) != 17:
        raise ValueError(f"Expected 17 Test_Holdout rows, found {len(test)}")

    if STEP3_FILE.exists():
        step3_xlsx = pd.ExcelFile(STEP3_FILE)
        step3_sheets = step3_xlsx.sheet_names
    else:
        step3_sheets = []

    one_fold_all = []
    one_pred_all = []
    one_summary_all = []

    multi_fold_all = []
    multi_pred_all = []
    multi_summary_all = []

    for target in TARGETS:
        one_fold, one_pred, one_summary = run_one_step(train, target)
        multi_fold, multi_pred, multi_summary = run_multi_step(train, target)

        one_fold_all.append(one_fold)
        one_pred_all.append(one_pred)
        one_summary_all.append(one_summary)

        multi_fold_all.append(multi_fold)
        multi_pred_all.append(multi_pred)
        multi_summary_all.append(multi_summary)

    one_fold_df = pd.concat(one_fold_all, ignore_index=True)
    one_pred_df = pd.concat(one_pred_all, ignore_index=True)
    one_summary_df = pd.concat(one_summary_all, ignore_index=True)

    multi_fold_df = pd.concat(multi_fold_all, ignore_index=True)
    multi_pred_df = pd.concat(multi_pred_all, ignore_index=True)
    multi_summary_df = pd.concat(multi_summary_all, ignore_index=True)

    ranking_df = choose_extrapolation_methods(one_summary_df, multi_summary_df)
    final_df = final_extrapolation(train, ranking_df)
    rpm_only_df = rpm_only_extrapolation(train, ranking_df)
    stability_df = stability_checks(train, ranking_df, final_df)

    step3_context = pd.DataFrame({
        "Item": [
            "Step3 workbook exists",
            "Step3 candidate sheet",
            "Interpolation target column for drainage time",
            "Step4 uses synthetic interpolation points"
        ],
        "Value": [
            STEP3_FILE.exists(),
            "Provisional_Candidates" if "Provisional_Candidates" in step3_sheets else "NOT FOUND",
            "td_to",
            "No"
        ]
    })

    method_definitions = pd.DataFrame({
        "Method": list(METHODS.keys()),
        "Degree": list(METHODS.values()),
        "Definition": [
            "Total-degree polynomial in scaled d and Rpm",
            "Total-degree polynomial in scaled d and Rpm",
            "Total-degree polynomial in scaled d and Rpm"
        ],
        "Purpose": [
            "Low-complexity extrapolation baseline",
            "Moderate-complexity extrapolation candidate",
            "Higher-complexity extrapolation candidate"
        ]
    })

    validation_log = pd.DataFrame({
        "Check": [
            "Train rows",
            "Test rows",
            "Synthetic points used for fitting",
            "One-step cutoffs",
            "One-step validation RPMs",
            "Multi-step cutoffs",
            "Final requested RPMs",
            "Final requested diameters",
            "Test_Holdout used for model fitting",
            "Target column names"
        ],
        "Result": [
            len(train),
            len(test),
            0,
            str(ONE_STEP_CUTOFFS),
            str([x + 20 for x in ONE_STEP_CUTOFFS]),
            str(MULTI_STEP_CUTOFFS),
            str(RPM_FINAL),
            str(DIAMETERS_FINAL),
            "NO",
            str(TARGETS)
        ]
    })

    ranking_summary = ranking_df.copy()

    with pd.ExcelWriter(OUTPUT_FILE, engine="openpyxl") as writer:
        one_summary_df.to_excel(writer, sheet_name="OneStep_Summary", index=False)
        multi_summary_df.to_excel(writer, sheet_name="MultiStep_Summary", index=False)
        ranking_summary.to_excel(writer, sheet_name="Method_Ranking", index=False)
        one_fold_df.to_excel(writer, sheet_name="OneStep_Folds", index=False)
        multi_fold_df.to_excel(writer, sheet_name="MultiStep_Folds", index=False)
        one_pred_df.to_excel(writer, sheet_name="OneStep_Predictions", index=False)
        multi_pred_df.to_excel(writer, sheet_name="MultiStep_Predictions", index=False)
        stability_df.to_excel(writer, sheet_name="Stability_Checks", index=False)
        final_df.to_excel(writer, sheet_name="Provisional_Extrapolation", index=False)
        rpm_only_df.to_excel(writer, sheet_name="RPM_Only_Extrapolation", index=False)
        step3_context.to_excel(writer, sheet_name="Step3_Context", index=False)
        method_definitions.to_excel(writer, sheet_name="Method_Definitions", index=False)
        validation_log.to_excel(writer, sheet_name="Validation_Log", index=False)

    print("-" * 78)
    print("DATA PROTOCOL")
    print("-" * 78)
    print(f"Train_Anchors used : {len(train)}")
    print("Test_Holdout used  : 0")
    print("Synthetic data used for fitting : 0")
    print(f"Target columns     : {TARGETS}")

    print("-" * 78)
    print("ONE-STEP PROGRESSIVE EXTRAPOLATION")
    print("-" * 78)
    print(one_summary_df.to_string(index=False))

    print("-" * 78)
    print("MULTI-STEP PROGRESSIVE EXTRAPOLATION")
    print("-" * 78)
    print(multi_summary_df.to_string(index=False))

    print("-" * 78)
    print("PROVISIONAL EXTRAPOLATION METHOD RANKING")
    print("-" * 78)
    print(ranking_df.to_string(index=False))

    print("-" * 78)
    print("PROVISIONAL 280-320 RPM / d=4,18 EXTRAPOLATION")
    print("-" * 78)
    print(final_df.to_string(index=False))

    print("-" * 78)
    print("COMPLETED")
    print("-" * 78)
    print("Step 4 uses Train_Anchors only for model fitting.")
    print("The 17 Test_Holdout observations remain untouched.")
    print("Final extrapolation is PROVISIONAL until Step 5 independent holdout validation.")
    print("Next step: review extrapolation stability and lock the continuous-model candidates.")

if __name__ == "__main__":
    main()
