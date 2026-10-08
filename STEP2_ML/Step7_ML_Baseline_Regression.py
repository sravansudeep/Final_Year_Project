"""
Step 7 — ML Baseline Regression
Input : Step6_ML_Data_Preparation.xlsx  (ML_Train_315, ML_Holdout_17)
Output: Step7_ML_Baseline_Regression.xlsx
"""

from pathlib import Path
import time
import warnings
import numpy as np
import pandas as pd
from sklearn.model_selection import RepeatedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel, ConstantKernel
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from xgboost import XGBRegressor

warnings.filterwarnings("ignore")

BASE_DIR = Path(__file__).resolve().parent

INPUT_FILE  = BASE_DIR / "Step6_ML_Data_Preparation.xlsx"
OUTPUT_FILE = BASE_DIR / "Step7_ML_Baseline_Regression.xlsx"

FEATURES = ["d", "Rpm"]
TARGETS  = ["hc_hi", "td_to", "Vch"]
ID_COL   = "Point_ID"

PHYSICAL_BOUNDS = {
    "hc_hi": (0.0, 1.0),
    "td_to": (0.0, np.inf),
    "Vch":   (0.0, 1.0),
}

SEED = 42
N_SPLITS = 5
N_REPEATS = 3


xl_in = pd.ExcelFile(INPUT_FILE)
train_sheet_name = [s for s in xl_in.sheet_names if s.startswith("ML_Train_")][0]
train = pd.read_excel(INPUT_FILE, sheet_name=train_sheet_name)
holdout = pd.read_excel(INPUT_FILE, sheet_name="ML_Holdout_17")

X_train = train[FEATURES].values
y_train = {t: train[t].values for t in TARGETS}

X_holdout = holdout[FEATURES].values
y_holdout = {t: holdout[t].values for t in TARGETS}

print(f"Training set : {X_train.shape[0]} rows, {X_train.shape[1]} features")
print(f"Holdout set  : {X_holdout.shape[0]} rows (locked - not used for selection)")


# 2. Define candidate models

def get_models(seed):
    tree_models = {
        "RandomForest": RandomForestRegressor(
            n_estimators=200, max_depth=None, min_samples_leaf=2,
            random_state=seed, n_jobs=-1,
        ),
        "ExtraTrees": ExtraTreesRegressor(
            n_estimators=200, max_depth=None, min_samples_leaf=2,
            random_state=seed, n_jobs=-1,
        ),
        "XGBoost": XGBRegressor(
            n_estimators=200, max_depth=4, learning_rate=0.1,
            min_child_weight=2, subsample=0.8, colsample_bytree=1.0,
            random_state=seed, verbosity=0, n_jobs=-1,
        ),
    }

    scaled_models = {
        "SVR": Pipeline([
            ("scaler", StandardScaler()),
            ("svr", SVR(kernel="rbf", C=10.0, epsilon=0.01)),
        ]),
        "KNN": Pipeline([
            ("scaler", StandardScaler()),
            ("knn", KNeighborsRegressor(n_neighbors=5, weights="distance")),
        ]),
        "MLP": Pipeline([
            ("scaler", StandardScaler()),
            ("mlp", MLPRegressor(
                hidden_layer_sizes=(64, 32), activation="relu",
                solver="adam", max_iter=2000, early_stopping=True,
                validation_fraction=0.15, random_state=seed,
            )),
        ]),
        "GPR": Pipeline([
            ("scaler", StandardScaler()),
            ("gpr", GaussianProcessRegressor(
                kernel=ConstantKernel() * Matern(nu=2.5) + WhiteKernel(),
                n_restarts_optimizer=5, random_state=seed, normalize_y=True,
            )),
        ]),
    }

    return {**tree_models, **scaled_models}


def safe_mape(y_true, y_pred, epsilon=1e-10):
    mask = np.abs(y_true) > epsilon
    if mask.sum() == 0:
        return np.nan
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100


def count_physical_violations(y_pred, target):
    lo, hi = PHYSICAL_BOUNDS[target]
    return int(((y_pred < lo) | (y_pred > hi)).sum())



# 3. Run repeated k-fold CV for each target × model

rkf = RepeatedKFold(n_splits=N_SPLITS, n_repeats=N_REPEATS, random_state=SEED)

all_cv_results = []
fold_details = []

for target in TARGETS:
    y = y_train[target]
    print(f"\n{'='*60}")
    print(f"TARGET: {target}")
    print(f"{'='*60}")

    models = get_models(SEED)

    for model_name, model in models.items():
        print(f"\n  {model_name}...", end=" ", flush=True)
        t0 = time.time()

        fold_r2, fold_rmse, fold_mae, fold_mape = [], [], [], []
        fold_violations = []
        fold_preds_sum = np.zeros(len(y))
        fold_counts = np.zeros(len(y), dtype=int)

        for fold_idx, (train_idx, val_idx) in enumerate(rkf.split(X_train)):
            X_tr, X_val = X_train[train_idx], X_train[val_idx]
            y_tr, y_val = y[train_idx], y[val_idx]

            model_clone = get_models(SEED)[model_name]
            model_clone.fit(X_tr, y_tr)
            y_pred = model_clone.predict(X_val)

            r2 = r2_score(y_val, y_pred)
            rmse = np.sqrt(mean_squared_error(y_val, y_pred))
            mae = mean_absolute_error(y_val, y_pred)
            mape = safe_mape(y_val, y_pred)
            violations = count_physical_violations(y_pred, target)

            fold_r2.append(r2)
            fold_rmse.append(rmse)
            fold_mae.append(mae)
            fold_mape.append(mape)
            fold_violations.append(violations)

            for i, vi in enumerate(val_idx):
                fold_preds_sum[vi] += y_pred[i]
                fold_counts[vi] += 1

            fold_details.append({
                "Target": target, "Model": model_name,
                "Repeat": fold_idx // N_SPLITS + 1,
                "Fold": fold_idx % N_SPLITS + 1,
                "R2": r2, "RMSE": rmse, "MAE": mae, "MAPE": mape,
                "Physical_Violations": violations,
                "Val_Size": len(val_idx),
            })

        elapsed = time.time() - t0

        mean_r2 = np.mean(fold_r2)
        std_r2 = np.std(fold_r2)
        mean_rmse = np.mean(fold_rmse)
        std_rmse = np.std(fold_rmse)
        mean_mae = np.mean(fold_mae)
        std_mae = np.std(fold_mae)
        valid_mapes = [m for m in fold_mape if not np.isnan(m)]
        mean_mape = np.mean(valid_mapes) if valid_mapes else np.nan
        std_mape = np.std(valid_mapes) if valid_mapes else np.nan
        total_violations = sum(fold_violations)

        oof_mask = fold_counts > 0
        oof_preds = fold_preds_sum.copy()
        oof_preds[oof_mask] /= fold_counts[oof_mask]
        oof_r2 = r2_score(y[oof_mask], oof_preds[oof_mask])
        oof_rmse = np.sqrt(mean_squared_error(y[oof_mask], oof_preds[oof_mask]))
        oof_mae = mean_absolute_error(y[oof_mask], oof_preds[oof_mask])
        oof_mape = safe_mape(y[oof_mask], oof_preds[oof_mask])
        oof_violations = count_physical_violations(oof_preds[oof_mask], target)

        print(f"R2={mean_r2:.4f}+/-{std_r2:.4f}  RMSE={mean_rmse:.6f}+/-{std_rmse:.6f}  "
              f"MAE={mean_mae:.6f}  violations={total_violations}  ({elapsed:.1f}s)")

        all_cv_results.append({
            "Target": target,
            "Model": model_name,
            "CV_R2_mean": mean_r2,
            "CV_R2_std": std_r2,
            "CV_RMSE_mean": mean_rmse,
            "CV_RMSE_std": std_rmse,
            "CV_MAE_mean": mean_mae,
            "CV_MAE_std": std_mae,
            "CV_MAPE_mean": mean_mape,
            "CV_MAPE_std": std_mape,
            "CV_Physical_Violations_Total": total_violations,
            "OOF_R2": oof_r2,
            "OOF_RMSE": oof_rmse,
            "OOF_MAE": oof_mae,
            "OOF_MAPE": oof_mape,
            "OOF_Physical_Violations": oof_violations,
            "Training_Time_s": round(elapsed, 2),
        })

cv_df = pd.DataFrame(all_cv_results)
fold_df = pd.DataFrame(fold_details)


# 4. Rank models per target

rankings = []
for target in TARGETS:
    sub = cv_df[cv_df["Target"] == target].copy()
    sub = sub.sort_values(
        by=["CV_Physical_Violations_Total", "CV_RMSE_mean", "CV_MAE_mean"],
        ascending=[True, True, True],
    ).reset_index(drop=True)
    sub["Rank"] = range(1, len(sub) + 1)
    rankings.append(sub)

ranking_df = pd.concat(rankings, ignore_index=True)

print(f"\n{'='*60}")
print("CV RANKING SUMMARY (by violations -> RMSE -> MAE)")
print(f"{'='*60}")
for target in TARGETS:
    sub = ranking_df[ranking_df["Target"] == target]
    print(f"\n  {target}:")
    for _, row in sub.iterrows():
        print(f"    #{int(row['Rank'])}: {row['Model']:15s}  "
              f"R2={row['CV_R2_mean']:.4f}+/-{row['CV_R2_std']:.4f}  "
              f"RMSE={row['CV_RMSE_mean']:.6f}  violations={int(row['CV_Physical_Violations_Total'])}")


# 5. Retrain best model per target on full training set & predict holdout

print(f"\n{'='*60}")
print("HOLDOUT EVALUATION (retrained on full ML_Train_67)")
print(f"{'='*60}")

holdout_results = []
for target in TARGETS:
    best_row = ranking_df[ranking_df["Target"] == target].iloc[0]
    best_name = best_row["Model"]
    y = y_train[target]

    model = get_models(SEED)[best_name]
    model.fit(X_train, y)

    y_ho_pred = model.predict(X_holdout)
    y_ho_true = y_holdout[target]

    ho_r2 = r2_score(y_ho_true, y_ho_pred)
    ho_rmse = np.sqrt(mean_squared_error(y_ho_true, y_ho_pred))
    ho_mae = mean_absolute_error(y_ho_true, y_ho_pred)
    ho_mape = safe_mape(y_ho_true, y_ho_pred)
    ho_violations = count_physical_violations(y_ho_pred, target)

    print(f"\n  {target} -> {best_name}")
    print(f"    Holdout R2   = {ho_r2:.6f}")
    print(f"    Holdout RMSE = {ho_rmse:.6f}")
    print(f"    Holdout MAE  = {ho_mae:.6f}")
    print(f"    Holdout MAPE = {ho_mape:.2f}%")
    print(f"    Physical violations = {ho_violations}")

    holdout_results.append({
        "Target": target,
        "Best_CV_Model": best_name,
        "CV_R2_mean": best_row["CV_R2_mean"],
        "CV_R2_std": best_row["CV_R2_std"],
        "CV_RMSE_mean": best_row["CV_RMSE_mean"],
        "Holdout_R2": ho_r2,
        "Holdout_RMSE": ho_rmse,
        "Holdout_MAE": ho_mae,
        "Holdout_MAPE": ho_mape,
        "Holdout_Physical_Violations": ho_violations,
    })

    pred_col = f"{target}_pred"
    holdout[pred_col] = y_ho_pred
    holdout[f"{target}_residual"] = y_ho_true - y_ho_pred
    holdout[f"{target}_model"] = best_name

holdout_summary_df = pd.DataFrame(holdout_results)


# 6. Full training set predictions

train_preds = train[[ID_COL, "d", "Rpm"]].copy()
for target in TARGETS:
    best_name = ranking_df[ranking_df["Target"] == target].iloc[0]["Model"]
    model = get_models(SEED)[best_name]
    model.fit(X_train, y_train[target])
    y_pred_full = model.predict(X_train)
    train_preds[f"{target}_actual"] = y_train[target]
    train_preds[f"{target}_pred"] = y_pred_full
    train_preds[f"{target}_residual"] = y_train[target] - y_pred_full
    train_preds[f"{target}_model"] = best_name


# 7. Write output workbook

with pd.ExcelWriter(OUTPUT_FILE, engine="openpyxl") as writer:

    readme = pd.DataFrame({
        "Item": [
            "Step", "Input", "Output", "Date",
            "CV_Strategy", "N_Splits", "N_Repeats", "Random_Seed",
            "Features", "Targets",
            "Train_rows", "Holdout_rows",
            "Models_Tested",
            "Ranking_Criteria",
            "Note_1", "Note_2", "Note_3",
        ],
        "Value": [
            "Step 7 — ML Baseline Regression",
            str(INPUT_FILE.name),
            str(OUTPUT_FILE.name),
            pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
            "RepeatedKFold",
            N_SPLITS,
            N_REPEATS,
            SEED,
            "d, Rpm",
            "hc_hi, td_to, Vch",
            315, 17,
            "RandomForest, ExtraTrees, XGBoost, SVR, KNN, MLP, GPR",
            "Physical violations (asc) → RMSE (asc) → MAE (asc)",
            "Holdout used only for benchmark evaluation, NOT for model selection",
            "Different targets may use different best models",
            "No target transformations applied; no silent prediction clipping",
        ],
    })
    readme.to_excel(writer, sheet_name="README", index=False)

    ranking_df.to_excel(writer, sheet_name="CV_Rankings", index=False)
    cv_df.to_excel(writer, sheet_name="CV_Full_Results", index=False)
    fold_df.to_excel(writer, sheet_name="CV_Fold_Details", index=False)
    holdout_summary_df.to_excel(writer, sheet_name="Holdout_Summary", index=False)
    holdout.to_excel(writer, sheet_name="Holdout_Predictions", index=False)
    train_preds.to_excel(writer, sheet_name="Train_Predictions", index=False)

print(f"\nOutput written to: {OUTPUT_FILE.name}")
print("Step 7 complete.")
