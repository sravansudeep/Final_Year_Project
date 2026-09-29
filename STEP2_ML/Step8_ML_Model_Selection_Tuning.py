"""
Step 8 — ML Cross-Validation & Model Selection / Tuning
Input : Step6_ML_Data_Preparation.xlsx  (ML_Train_67, ML_Holdout_17)
Output: Step8_ML_Model_Selection_Tuning.xlsx
"""

from pathlib import Path
import time
import warnings
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import RepeatedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel, ConstantKernel, RBF, RationalQuadratic
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from xgboost import XGBRegressor

warnings.filterwarnings("ignore")

BASE_DIR = Path(__file__).resolve().parent

INPUT_FILE  = BASE_DIR / "Step6_ML_Data_Preparation.xlsx"
OUTPUT_FILE = BASE_DIR / "Step8_ML_Model_Selection_Tuning.xlsx"
STEP7_FILE  = BASE_DIR / "Step7_ML_Baseline_Regression.xlsx"

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

# ---------------------------------------------------------------------------
# 1. Load data
# ---------------------------------------------------------------------------
train = pd.read_excel(INPUT_FILE, sheet_name="ML_Train_67")
holdout = pd.read_excel(INPUT_FILE, sheet_name="ML_Holdout_17")

X_train = train[FEATURES].values
y_train = {t: train[t].values for t in TARGETS}

X_holdout = holdout[FEATURES].values
y_holdout = {t: holdout[t].values for t in TARGETS}

print(f"Training set : {X_train.shape[0]} rows, {X_train.shape[1]} features")
print(f"Holdout set  : {X_holdout.shape[0]} rows (locked - not used for selection/tuning)")


def safe_mape(y_true, y_pred, epsilon=1e-10):
    mask = np.abs(y_true) > epsilon
    if mask.sum() == 0:
        return np.nan
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)


def count_physical_violations(y_pred, target):
    lo, hi = PHYSICAL_BOUNDS[target]
    return int(((y_pred < lo) | (y_pred > hi)).sum())


# ---------------------------------------------------------------------------
# 2. Define candidate model tuning space
# ---------------------------------------------------------------------------
def get_candidate_configurations(seed):
    configs = []

    # --- Random Forest (12 configs) ---
    for n_est in [100, 200]:
        for depth in [None, 4, 6]:
            for leaf in [1, 2]:
                name = f"RF_n{n_est}_d{depth}_l{leaf}"
                model = RandomForestRegressor(
                    n_estimators=n_est, max_depth=depth, min_samples_leaf=leaf,
                    random_state=seed, n_jobs=-1
                )
                configs.append(("RandomForest", name, model, {
                    "n_estimators": n_est, "max_depth": str(depth), "min_samples_leaf": leaf
                }))

    # --- Extra Trees (12 configs) ---
    for n_est in [100, 200]:
        for depth in [None, 4, 6]:
            for leaf in [1, 2]:
                name = f"ET_n{n_est}_d{depth}_l{leaf}"
                model = ExtraTreesRegressor(
                    n_estimators=n_est, max_depth=depth, min_samples_leaf=leaf,
                    random_state=seed, n_jobs=-1
                )
                configs.append(("ExtraTrees", name, model, {
                    "n_estimators": n_est, "max_depth": str(depth), "min_samples_leaf": leaf
                }))

    # --- XGBoost (12 configs) ---
    for n_est in [100, 200]:
        for depth in [2, 3, 4]:
            for lr in [0.05, 0.1]:
                name = f"XGB_n{n_est}_d{depth}_lr{lr}"
                model = XGBRegressor(
                    n_estimators=n_est, max_depth=depth, learning_rate=lr,
                    subsample=0.85, min_child_weight=2,
                    random_state=seed, verbosity=0, n_jobs=-1
                )
                configs.append(("XGBoost", name, model, {
                    "n_estimators": n_est, "max_depth": depth, "learning_rate": lr, "subsample": 0.85
                }))

    # --- SVR (StandardScaler + SVR: 12 configs) ---
    for C_val in [5.0, 20.0, 50.0]:
        for eps in [0.005, 0.02]:
            for gamma_val in ["scale", 0.5]:
                name = f"SVR_C{C_val}_eps{eps}_g{gamma_val}"
                pipe = Pipeline([
                    ("scaler", StandardScaler()),
                    ("svr", SVR(kernel="rbf", C=C_val, epsilon=eps, gamma=gamma_val))
                ])
                configs.append(("SVR", name, pipe, {
                    "C": C_val, "epsilon": eps, "gamma": str(gamma_val), "kernel": "rbf"
                }))

    # --- KNN (StandardScaler + KNN: 12 configs) ---
    for k in [3, 5, 7]:
        for weights in ["uniform", "distance"]:
            for p in [1, 2]:
                metric_name = "manhattan" if p == 1 else "euclidean"
                name = f"KNN_k{k}_{weights}_{metric_name}"
                pipe = Pipeline([
                    ("scaler", StandardScaler()),
                    ("knn", KNeighborsRegressor(n_neighbors=k, weights=weights, p=p))
                ])
                configs.append(("KNN", name, pipe, {
                    "n_neighbors": k, "weights": weights, "p": p
                }))

    # --- GPR (StandardScaler + GPR: 8 configs) ---
    gpr_kernels = [
        ("Matern25", ConstantKernel(1.0, (1e-2, 1e3)) * Matern(length_scale=1.0, nu=2.5) + WhiteKernel(noise_level=1e-3, noise_level_bounds=(1e-6, 1e-1))),
        ("Matern15", ConstantKernel(1.0, (1e-2, 1e3)) * Matern(length_scale=1.0, nu=1.5) + WhiteKernel(noise_level=1e-3, noise_level_bounds=(1e-6, 1e-1))),
        ("RBF", ConstantKernel(1.0, (1e-2, 1e3)) * RBF(length_scale=1.0) + WhiteKernel(noise_level=1e-3, noise_level_bounds=(1e-6, 1e-1))),
        ("RQ", ConstantKernel(1.0, (1e-2, 1e3)) * RationalQuadratic(length_scale=1.0, alpha=1.0) + WhiteKernel(noise_level=1e-3, noise_level_bounds=(1e-6, 1e-1))),
    ]
    for kname, kernel in gpr_kernels:
        for alpha_val in [1e-10, 1e-5]:
            name = f"GPR_{kname}_a{alpha_val}"
            pipe = Pipeline([
                ("scaler", StandardScaler()),
                ("gpr", GaussianProcessRegressor(
                    kernel=kernel, alpha=alpha_val,
                    n_restarts_optimizer=3, random_state=seed, normalize_y=True
                ))
            ])
            configs.append(("GPR", name, pipe, {
                "kernel": kname, "alpha": alpha_val
            }))

    # --- MLP (StandardScaler + MLP: 8 configs) ---
    for arch in [(64,), (64, 32)]:
        for act in ["relu", "tanh"]:
            for alpha_val in [0.001, 0.01]:
                arch_str = "-".join(map(str, arch))
                name = f"MLP_h{arch_str}_{act}_a{alpha_val}"
                pipe = Pipeline([
                    ("scaler", StandardScaler()),
                    ("mlp", MLPRegressor(
                        hidden_layer_sizes=arch, activation=act, alpha=alpha_val,
                        solver="adam", max_iter=1500, early_stopping=True,
                        validation_fraction=0.15, random_state=seed
                    ))
                ])
                configs.append(("MLP", name, pipe, {
                    "hidden_layer_sizes": str(arch), "activation": act, "alpha": alpha_val
                }))

    return configs


# ---------------------------------------------------------------------------
# 3. Repeated 5-fold CV evaluation
# ---------------------------------------------------------------------------
rkf = RepeatedKFold(n_splits=N_SPLITS, n_repeats=N_REPEATS, random_state=SEED)

all_configs = get_candidate_configurations(SEED)
print(f"Total candidate configurations generated: {len(all_configs)}")

all_cv_results = []

for target in TARGETS:
    y = y_train[target]
    print(f"\n{'='*70}")
    print(f"TUNING TARGET: {target}")
    print(f"{'='*70}")

    for family, config_name, model_template, param_dict in all_configs:
        t0 = time.time()
        fold_r2, fold_rmse, fold_mae, fold_mape = [], [], [], []
        fold_violations = []
        fold_preds_sum = np.zeros(len(y))
        fold_counts = np.zeros(len(y), dtype=int)

        for fold_idx, (tr_idx, val_idx) in enumerate(rkf.split(X_train)):
            X_tr, X_val = X_train[tr_idx], X_train[val_idx]
            y_tr, y_val = y[tr_idx], y[val_idx]

            m = clone(model_template)
            m.fit(X_tr, y_tr)
            y_pred = m.predict(X_val)

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

        elapsed = time.time() - t0

        mean_r2 = float(np.mean(fold_r2))
        std_r2 = float(np.std(fold_r2))
        mean_rmse = float(np.mean(fold_rmse))
        std_rmse = float(np.std(fold_rmse))
        mean_mae = float(np.mean(fold_mae))
        std_mae = float(np.std(fold_mae))
        valid_mapes = [m_val for m_val in fold_mape if not np.isnan(m_val)]
        mean_mape = float(np.mean(valid_mapes)) if valid_mapes else np.nan
        std_mape = float(np.std(valid_mapes)) if valid_mapes else np.nan
        total_violations = int(sum(fold_violations))

        oof_mask = fold_counts > 0
        oof_preds = fold_preds_sum.copy()
        oof_preds[oof_mask] /= fold_counts[oof_mask]
        oof_r2 = float(r2_score(y[oof_mask], oof_preds[oof_mask]))
        oof_rmse = float(np.sqrt(mean_squared_error(y[oof_mask], oof_preds[oof_mask])))
        oof_mae = float(mean_absolute_error(y[oof_mask], oof_preds[oof_mask]))
        oof_mape = float(safe_mape(y[oof_mask], oof_preds[oof_mask]))
        oof_violations = int(count_physical_violations(oof_preds[oof_mask], target))

        all_cv_results.append({
            "Target": target,
            "Family": family,
            "Config_Name": config_name,
            "Parameters": str(param_dict),
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
            "Time_s": round(elapsed, 2),
        })

    sub_df = pd.DataFrame([r for r in all_cv_results if r["Target"] == target])
    sub_df = sub_df.sort_values(
        by=["CV_Physical_Violations_Total", "CV_RMSE_mean", "CV_MAE_mean"],
        ascending=[True, True, True]
    )
    best_row = sub_df.iloc[0]
    print(f"  Top configuration for {target}:")
    print(f"    Config    : {best_row['Config_Name']} ({best_row['Family']})")
    print(f"    CV R2     : {best_row['CV_R2_mean']:.4f} +/- {best_row['CV_R2_std']:.4f}")
    print(f"    CV RMSE   : {best_row['CV_RMSE_mean']:.6f} +/- {best_row['CV_RMSE_std']:.6f}")
    print(f"    CV MAE    : {best_row['CV_MAE_mean']:.6f}")
    print(f"    Violations: {int(best_row['CV_Physical_Violations_Total'])}")

cv_df = pd.DataFrame(all_cv_results)

# ---------------------------------------------------------------------------
# 4. Rank configurations per target
# ---------------------------------------------------------------------------
rankings = []
for target in TARGETS:
    sub = cv_df[cv_df["Target"] == target].copy()
    sub = sub.sort_values(
        by=["CV_Physical_Violations_Total", "CV_RMSE_mean", "CV_MAE_mean", "CV_R2_mean"],
        ascending=[True, True, True, False]
    )
    sub["Rank"] = range(1, len(sub) + 1)
    rankings.append(sub)

ranking_df = pd.concat(rankings, ignore_index=True)

# ---------------------------------------------------------------------------
# 5. Extract Best-Per-Family
# ---------------------------------------------------------------------------
best_per_family_list = []
for target in TARGETS:
    for fam in ["RandomForest", "ExtraTrees", "XGBoost", "SVR", "KNN", "GPR", "MLP"]:
        fam_sub = ranking_df[(ranking_df["Target"] == target) & (ranking_df["Family"] == fam)]
        if not fam_sub.empty:
            fam_best = fam_sub.sort_values(
                by=["CV_Physical_Violations_Total", "CV_RMSE_mean", "CV_MAE_mean"],
                ascending=[True, True, True]
            ).iloc[0]
            best_per_family_list.append(fam_best)

best_per_family_df = pd.DataFrame(best_per_family_list)

# ---------------------------------------------------------------------------
# 6. Baseline (Step 7) vs Tuned (Step 8) comparison
# ---------------------------------------------------------------------------
step7_cv = pd.read_excel(STEP7_FILE, sheet_name="CV_Rankings")

baseline_vs_tuned = []
for target in TARGETS:
    base_sub = step7_cv[step7_cv["Target"] == target]
    tuned_sub = best_per_family_df[best_per_family_df["Target"] == target]

    for fam in ["RandomForest", "ExtraTrees", "XGBoost", "SVR", "KNN", "GPR", "MLP"]:
        b_row = base_sub[base_sub["Model"] == fam]
        t_row = tuned_sub[tuned_sub["Family"] == fam]

        if not b_row.empty and not t_row.empty:
            b = b_row.iloc[0]
            t = t_row.iloc[0]

            rmse_diff = t["CV_RMSE_mean"] - b["CV_RMSE_mean"]
            rmse_pct_change = (rmse_diff / b["CV_RMSE_mean"]) * 100
            r2_diff = t["CV_R2_mean"] - b["CV_R2_mean"]
            viol_diff = t["CV_Physical_Violations_Total"] - b["CV_Physical_Violations_Total"]

            baseline_vs_tuned.append({
                "Target": target,
                "Family": fam,
                "Tuned_Config": t["Config_Name"],
                "Baseline_CV_RMSE": b["CV_RMSE_mean"],
                "Tuned_CV_RMSE": t["CV_RMSE_mean"],
                "RMSE_Change": rmse_diff,
                "RMSE_Pct_Change": rmse_pct_change,
                "Baseline_CV_R2": b["CV_R2_mean"],
                "Tuned_CV_R2": t["CV_R2_mean"],
                "R2_Change": r2_diff,
                "Baseline_Violations": int(b["CV_Physical_Violations_Total"]),
                "Tuned_Violations": int(t["CV_Physical_Violations_Total"]),
                "Violations_Change": int(viol_diff),
            })

baseline_vs_tuned_df = pd.DataFrame(baseline_vs_tuned)

# ---------------------------------------------------------------------------
# 7. Retrain Final Selected Models & Evaluate on Locked Holdout
# ---------------------------------------------------------------------------
print(f"\n{'='*70}")
print("FINAL SELECTED TUNED MODELS & HOLDOUT EVALUATION")
print(f"{'='*70}")

config_dict = {cfg[1]: cfg[2] for cfg in all_configs}

final_selected = []
holdout_predictions = holdout[[ID_COL, "d", "Rpm"]].copy()
train_predictions = train[[ID_COL, "d", "Rpm"]].copy()

for target in TARGETS:
    best_row = ranking_df[ranking_df["Target"] == target].iloc[0]
    config_name = best_row["Config_Name"]
    family = best_row["Family"]
    model = clone(config_dict[config_name])

    # Retrain on full training set
    model.fit(X_train, y_train[target])

    # Predict Holdout
    y_ho_pred = model.predict(X_holdout)
    y_ho_true = y_holdout[target]

    ho_r2 = float(r2_score(y_ho_true, y_ho_pred))
    ho_rmse = float(np.sqrt(mean_squared_error(y_ho_true, y_ho_pred)))
    ho_mae = float(mean_absolute_error(y_ho_true, y_ho_pred))
    ho_mape = float(safe_mape(y_ho_true, y_ho_pred))
    ho_violations = int(count_physical_violations(y_ho_pred, target))

    # Predict Train
    y_tr_pred = model.predict(X_train)
    y_tr_true = y_train[target]
    tr_r2 = float(r2_score(y_tr_true, y_tr_pred))
    tr_rmse = float(np.sqrt(mean_squared_error(y_tr_true, y_tr_pred)))
    tr_mae = float(mean_absolute_error(y_tr_true, y_tr_pred))

    print(f"\nTarget: {target} -> {config_name} ({family})")
    print(f"  CV:      R2 = {best_row['CV_R2_mean']:.4f} +/- {best_row['CV_R2_std']:.4f}  RMSE = {best_row['CV_RMSE_mean']:.6f}  Violations = {int(best_row['CV_Physical_Violations_Total'])}")
    print(f"  Holdout: R2 = {ho_r2:.6f}  RMSE = {ho_rmse:.6f}  MAE = {ho_mae:.6f}  MAPE = {ho_mape:.2f}%  Violations = {ho_violations}")

    final_selected.append({
        "Target": target,
        "Selected_Family": family,
        "Selected_Config": config_name,
        "Parameters": best_row["Parameters"],
        "CV_Rank": 1,
        "CV_R2_mean": best_row["CV_R2_mean"],
        "CV_R2_std": best_row["CV_R2_std"],
        "CV_RMSE_mean": best_row["CV_RMSE_mean"],
        "CV_RMSE_std": best_row["CV_RMSE_std"],
        "CV_MAE_mean": best_row["CV_MAE_mean"],
        "CV_Physical_Violations_Total": int(best_row["CV_Physical_Violations_Total"]),
        "Train_R2_full": tr_r2,
        "Train_RMSE_full": tr_rmse,
        "Train_MAE_full": tr_mae,
        "Holdout_R2": ho_r2,
        "Holdout_RMSE": ho_rmse,
        "Holdout_MAE": ho_mae,
        "Holdout_MAPE": ho_mape,
        "Holdout_Physical_Violations": ho_violations,
    })

    holdout_predictions[f"{target}_actual"] = y_ho_true
    holdout_predictions[f"{target}_pred"] = y_ho_pred
    holdout_predictions[f"{target}_residual"] = y_ho_true - y_ho_pred
    holdout_predictions[f"{target}_model"] = config_name

    train_predictions[f"{target}_actual"] = y_tr_true
    train_predictions[f"{target}_pred"] = y_tr_pred
    train_predictions[f"{target}_residual"] = y_tr_true - y_tr_pred
    train_predictions[f"{target}_model"] = config_name

final_selected_df = pd.DataFrame(final_selected)

# ---------------------------------------------------------------------------
# 8. Write Excel Workbook
# ---------------------------------------------------------------------------
with pd.ExcelWriter(OUTPUT_FILE, engine="openpyxl") as writer:

    readme = pd.DataFrame({
        "Item": [
            "Step", "Input", "Output", "Date",
            "CV_Strategy", "N_Splits", "N_Repeats", "Random_Seed",
            "Features", "Targets",
            "Train_rows", "Holdout_rows",
            "Total_Configurations_Tested",
            "Selection_Criteria_Hierarchy",
            "Note_1", "Note_2", "Note_3", "Note_4"
        ],
        "Value": [
            "Step 8 - ML Model Selection & Hyperparameter Tuning",
            str(INPUT_FILE.name),
            str(OUTPUT_FILE.name),
            pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
            "RepeatedKFold",
            N_SPLITS,
            N_REPEATS,
            SEED,
            "d, Rpm",
            "hc_hi, td_to, Vch",
            67, 17,
            len(all_configs),
            "1. Physical violations (asc) -> 2. CV RMSE (asc) -> 3. CV MAE (asc) -> 4. CV R2 (desc)",
            "Holdout strictly isolated from hyperparameter tuning and model selection",
            "Different targets legitimately select different optimal models",
            "Tuned models strictly evaluated for physical boundary compliance",
            "Zero silent clipping of predictions",
        ]
    })
    readme.to_excel(writer, sheet_name="README", index=False)
    final_selected_df.to_excel(writer, sheet_name="Final_Selected_Models", index=False)
    baseline_vs_tuned_df.to_excel(writer, sheet_name="Baseline_vs_Tuned", index=False)
    best_per_family_df.to_excel(writer, sheet_name="CV_Best_Per_Family", index=False)
    ranking_df.to_excel(writer, sheet_name="CV_Rankings_All", index=False)
    holdout_predictions.to_excel(writer, sheet_name="Holdout_Predictions", index=False)
    train_predictions.to_excel(writer, sheet_name="Train_Predictions", index=False)

print(f"\nOutput written to: {OUTPUT_FILE.name}")
print("Step 8 complete.")
