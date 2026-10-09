"""
Step 8 — ML Model Selection & Hyperparameter Tuning
Input : Step6_ML_Data_Preparation.xlsx, Step7_ML_Baseline_Regression.xlsx
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
from sklearn.gaussian_process.kernels import Matern, WhiteKernel, ConstantKernel, RBF
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
N_SPLITS  = 5
N_REPEATS = 3

# A config is "stable" if its RMSE coefficient of variation across folds < this
STABILITY_THRESHOLD = 30.0
# Tree-based models with train R2 above this are considered to be memorizing
MEMORIZATION_R2_THRESHOLD = 0.995
TREE_FAMILIES = {"RandomForest", "ExtraTrees", "XGBoost"}
MAX_MEMORIZATION_CANDIDATES = 50


# Load data

xl_in = pd.ExcelFile(INPUT_FILE)
train_sheet_name = [s for s in xl_in.sheet_names if s.startswith("ML_Train_")][0]
train   = pd.read_excel(INPUT_FILE, sheet_name=train_sheet_name)
holdout = pd.read_excel(INPUT_FILE, sheet_name="ML_Holdout_17")

X_train = train[FEATURES].values
y_train = {t: train[t].values for t in TARGETS}

X_holdout = holdout[FEATURES].values
y_holdout = {t: holdout[t].values for t in TARGETS}

print(f"Train:   {X_train.shape[0]} rows, {X_train.shape[1]} features  ({train_sheet_name})")
print(f"Holdout: {X_holdout.shape[0]} rows (locked)")


def safe_mape(y_true, y_pred, epsilon=1e-10):
    mask = np.abs(y_true) > epsilon
    if mask.sum() == 0:
        return np.nan
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)


def count_physical_violations(y_pred, target):
    lo, hi = PHYSICAL_BOUNDS[target]
    return int(((y_pred < lo) | (y_pred > hi)).sum())


# Candidate configurations

def get_candidate_configurations(seed):
    configs = []

    for n_est in [100, 200]:
        for depth in [None, 4, 6]:
            for leaf in [1, 2]:
                name = f"RF_n{n_est}_d{depth}_l{leaf}"
                configs.append(("RandomForest", name, RandomForestRegressor(
                    n_estimators=n_est, max_depth=depth, min_samples_leaf=leaf,
                    random_state=seed, n_jobs=-1,
                ), {"n_estimators": n_est, "max_depth": str(depth), "min_samples_leaf": leaf}))

    for n_est in [100, 200]:
        for depth in [None, 4, 5, 6, 8]:
            for leaf in [1, 2, 3, 5, 8]:
                name = f"ET_n{n_est}_d{depth}_l{leaf}"
                configs.append(("ExtraTrees", name, ExtraTreesRegressor(
                    n_estimators=n_est, max_depth=depth, min_samples_leaf=leaf,
                    random_state=seed, n_jobs=-1,
                ), {"n_estimators": n_est, "max_depth": str(depth), "min_samples_leaf": leaf}))

    for n_est in [100, 200]:
        for depth in [2, 3, 4]:
            for lr in [0.05, 0.1]:
                name = f"XGB_n{n_est}_d{depth}_lr{lr}"
                configs.append(("XGBoost", name, XGBRegressor(
                    n_estimators=n_est, max_depth=depth, learning_rate=lr,
                    subsample=0.85, min_child_weight=2,
                    random_state=seed, verbosity=0, n_jobs=-1,
                ), {"n_estimators": n_est, "max_depth": depth, "learning_rate": lr, "subsample": 0.85}))

    for C_val in [5.0, 20.0, 50.0]:
        for eps in [0.005, 0.02]:
            for gamma_val in ["scale", 0.5]:
                name = f"SVR_C{C_val}_eps{eps}_g{gamma_val}"
                configs.append(("SVR", name, Pipeline([
                    ("scaler", StandardScaler()),
                    ("svr", SVR(kernel="rbf", C=C_val, epsilon=eps, gamma=gamma_val)),
                ]), {"C": C_val, "epsilon": eps, "gamma": str(gamma_val), "kernel": "rbf"}))

    for k in [3, 5, 7]:
        for weights in ["uniform", "distance"]:
            for p in [1, 2]:
                metric_name = "manhattan" if p == 1 else "euclidean"
                name = f"KNN_k{k}_{weights}_{metric_name}"
                configs.append(("KNN", name, Pipeline([
                    ("scaler", StandardScaler()),
                    ("knn", KNeighborsRegressor(n_neighbors=k, weights=weights, p=p)),
                ]), {"n_neighbors": k, "weights": weights, "p": p}))

    gpr_kernels = [
        ("Matern25", ConstantKernel(1.0, (1e-2, 1e3)) * Matern(length_scale=1.0, nu=2.5) + WhiteKernel(noise_level=1e-3, noise_level_bounds=(1e-6, 1e-1))),
        ("Matern15", ConstantKernel(1.0, (1e-2, 1e3)) * Matern(length_scale=1.0, nu=1.5) + WhiteKernel(noise_level=1e-3, noise_level_bounds=(1e-6, 1e-1))),
        ("RBF",      ConstantKernel(1.0, (1e-2, 1e3)) * RBF(length_scale=1.0)            + WhiteKernel(noise_level=1e-3, noise_level_bounds=(1e-6, 1e-1))),
    ]
    for kname, kernel in gpr_kernels:
        for alpha_val in [1e-10, 1e-3]:
            name = f"GPR_{kname}_a{alpha_val}"
            configs.append(("GPR", name, Pipeline([
                ("scaler", StandardScaler()),
                ("gpr", GaussianProcessRegressor(
                    kernel=kernel, alpha=alpha_val,
                    n_restarts_optimizer=0, random_state=seed, normalize_y=True,
                )),
            ]), {"kernel": kname, "alpha": alpha_val}))

    for arch in [(64,), (64, 32)]:
        for act in ["relu", "tanh"]:
            for alpha_val in [0.001, 0.01]:
                arch_str = "-".join(map(str, arch))
                name = f"MLP_h{arch_str}_{act}_a{alpha_val}"
                configs.append(("MLP", name, Pipeline([
                    ("scaler", StandardScaler()),
                    ("mlp", MLPRegressor(
                        hidden_layer_sizes=arch, activation=act, alpha=alpha_val,
                        solver="adam", max_iter=1500, early_stopping=True,
                        validation_fraction=0.15, random_state=seed,
                    )),
                ]), {"hidden_layer_sizes": str(arch), "activation": act, "alpha": alpha_val}))

    return configs


# Repeated k-fold CV across all configurations

rkf = RepeatedKFold(n_splits=N_SPLITS, n_repeats=N_REPEATS, random_state=SEED)

all_configs = get_candidate_configurations(SEED)
print(f"Configurations to evaluate: {len(all_configs)}")

all_cv_results = []

for target in TARGETS:
    y = y_train[target]
    print(f"\n{target}")

    for family, config_name, model_template, param_dict in all_configs:
        t0 = time.time()
        fold_r2, fold_rmse, fold_mae, fold_mape = [], [], [], []
        fold_violations = []
        fold_preds_sum = np.zeros(len(y))
        fold_counts    = np.zeros(len(y), dtype=int)

        for fold_idx, (tr_idx, val_idx) in enumerate(rkf.split(X_train)):
            X_tr, X_val = X_train[tr_idx], X_train[val_idx]
            y_tr, y_val = y[tr_idx], y[val_idx]

            m = clone(model_template)
            m.fit(X_tr, y_tr)
            y_pred = m.predict(X_val)

            fold_r2.append(r2_score(y_val, y_pred))
            fold_rmse.append(np.sqrt(mean_squared_error(y_val, y_pred)))
            fold_mae.append(mean_absolute_error(y_val, y_pred))
            fold_mape.append(safe_mape(y_val, y_pred))
            fold_violations.append(count_physical_violations(y_pred, target))

            for i, vi in enumerate(val_idx):
                fold_preds_sum[vi] += y_pred[i]
                fold_counts[vi]    += 1

        elapsed = time.time() - t0

        mean_r2   = float(np.mean(fold_r2))
        std_r2    = float(np.std(fold_r2))
        mean_rmse = float(np.mean(fold_rmse))
        std_rmse  = float(np.std(fold_rmse))
        mean_mae  = float(np.mean(fold_mae))
        std_mae   = float(np.std(fold_mae))
        valid_mapes = [v for v in fold_mape if not np.isnan(v)]
        mean_mape = float(np.mean(valid_mapes)) if valid_mapes else np.nan
        std_mape  = float(np.std(valid_mapes))  if valid_mapes else np.nan
        total_violations = int(sum(fold_violations))

        # RMSE coefficient of variation across folds (stability measure)
        cv_coeffvar = (float(np.std(fold_rmse, ddof=1)) / mean_rmse * 100) if mean_rmse > 1e-15 else np.nan
        stability_rank = 0 if (not np.isnan(cv_coeffvar) and cv_coeffvar < STABILITY_THRESHOLD) else 1

        # OOF: average each sample's predictions across the folds it appeared in
        oof_mask  = fold_counts > 0
        oof_preds = fold_preds_sum.copy()
        oof_preds[oof_mask] /= fold_counts[oof_mask]
        oof_r2         = float(r2_score(y[oof_mask], oof_preds[oof_mask]))
        oof_rmse       = float(np.sqrt(mean_squared_error(y[oof_mask], oof_preds[oof_mask])))
        oof_mae        = float(mean_absolute_error(y[oof_mask], oof_preds[oof_mask]))
        oof_mape       = float(safe_mape(y[oof_mask], oof_preds[oof_mask]))
        oof_violations = int(count_physical_violations(oof_preds[oof_mask], target))

        all_cv_results.append({
            "Target": target,
            "Family": family,
            "Config_Name": config_name,
            "Parameters": str(param_dict),
            "CV_R2_mean": mean_r2,
            "CV_R2_std":  std_r2,
            "CV_RMSE_mean": mean_rmse,
            "CV_RMSE_std":  std_rmse,
            "CV_RMSE_CoeffVar": cv_coeffvar,
            "Stability_Rank": stability_rank,
            "CV_MAE_mean": mean_mae,
            "CV_MAE_std":  std_mae,
            "CV_MAPE_mean": mean_mape,
            "CV_MAPE_std":  std_mape,
            "CV_Physical_Violations_Total": total_violations,
            "OOF_R2":   oof_r2,
            "OOF_RMSE": oof_rmse,
            "OOF_MAE":  oof_mae,
            "OOF_MAPE": oof_mape,
            "OOF_Physical_Violations": oof_violations,
            "Time_s": round(elapsed, 2),
        })

    best = min(
        (r for r in all_cv_results if r["Target"] == target),
        key=lambda r: (r["CV_Physical_Violations_Total"], r["Stability_Rank"], r["CV_RMSE_mean"]),
    )
    stab = "stable" if best["Stability_Rank"] == 0 else "unstable"
    print(f"  best: {best['Config_Name']}  R2={best['CV_R2_mean']:.4f}  "
          f"RMSE={best['CV_RMSE_mean']:.6f}  CoeffVar={best['CV_RMSE_CoeffVar']:.1f}% [{stab}]  "
          f"violations={best['CV_Physical_Violations_Total']}")

cv_df = pd.DataFrame(all_cv_results)


# Rank all configurations per target

rankings = []
for target in TARGETS:
    sub = cv_df[cv_df["Target"] == target].copy()
    sub = sub.sort_values(
        by=["CV_Physical_Violations_Total", "Stability_Rank", "CV_RMSE_mean", "CV_MAE_mean", "CV_R2_mean"],
        ascending=[True, True, True, True, False],
    )
    sub["Rank"] = range(1, len(sub) + 1)
    rankings.append(sub)

ranking_df = pd.concat(rankings, ignore_index=True)

for target in TARGETS:
    sub = ranking_df[ranking_df["Target"] == target]
    n_stable = int((sub["Stability_Rank"] == 0).sum())
    top = sub.iloc[0]
    print(f"  {target}: {n_stable}/{len(sub)} configs stable  "
          f"rank-1: {top['Config_Name']} ({'stable' if top['Stability_Rank']==0 else 'unstable'})")


# Select final model per target: skip memorizing tree configs, retrain, evaluate on holdout

config_dict = {cfg[1]: cfg[2] for cfg in all_configs}

final_selected      = []
holdout_predictions = holdout[[ID_COL, "d", "Rpm"]].copy()
train_predictions   = train[[ID_COL, "d", "Rpm"]].copy()

print("\nFinal selection (violations -> stability -> RMSE, memorization check for trees):")
for target in TARGETS:
    candidates = ranking_df[ranking_df["Target"] == target].copy()

    selected_row   = None
    selected_model = None
    selected_tr_r2 = None
    memorization_rejected = []

    for check_idx, (_, cand_row) in enumerate(candidates.iterrows()):
        if check_idx >= MAX_MEMORIZATION_CANDIDATES:
            break

        config_name = cand_row["Config_Name"]
        family      = cand_row["Family"]

        model = clone(config_dict[config_name])
        model.fit(X_train, y_train[target])

        y_tr_pred = model.predict(X_train)
        tr_r2  = float(r2_score(y_train[target], y_tr_pred))
        tr_rmse = float(np.sqrt(mean_squared_error(y_train[target], y_tr_pred)))

        # Reject tree-based models that perfectly fit the training set
        if tr_r2 > MEMORIZATION_R2_THRESHOLD and family in TREE_FAMILIES:
            memorization_rejected.append({
                "Target": target,
                "Rejected_Rank": int(cand_row["Rank"]),
                "Config": config_name,
                "Family": family,
                "Train_R2": tr_r2,
                "Rejection_Reason": f"Train R2={tr_r2:.6f} > {MEMORIZATION_R2_THRESHOLD}",
            })
            continue

        selected_row   = cand_row
        selected_model = model
        selected_tr_r2 = tr_r2
        break

    if selected_row is None:
        print(f"  WARNING: {target} — all top-{MAX_MEMORIZATION_CANDIDATES} configs memorize; falling back to rank-1.")
        top_row = candidates.iloc[0]
        selected_model = clone(config_dict[top_row["Config_Name"]])
        selected_model.fit(X_train, y_train[target])
        selected_row   = top_row
        selected_tr_r2 = float(r2_score(y_train[target], selected_model.predict(X_train)))

    y_ho_pred = selected_model.predict(X_holdout)
    y_ho_true = y_holdout[target]

    ho_r2         = float(r2_score(y_ho_true, y_ho_pred))
    ho_rmse       = float(np.sqrt(mean_squared_error(y_ho_true, y_ho_pred)))
    ho_mae        = float(mean_absolute_error(y_ho_true, y_ho_pred))
    ho_mape       = float(safe_mape(y_ho_true, y_ho_pred))
    ho_violations = int(count_physical_violations(y_ho_pred, target))

    y_tr_pred_full = selected_model.predict(X_train)
    y_tr_true      = y_train[target]
    tr_rmse_full   = float(np.sqrt(mean_squared_error(y_tr_true, y_tr_pred_full)))
    tr_mae_full    = float(mean_absolute_error(y_tr_true, y_tr_pred_full))

    stab_str = "stable" if selected_row["Stability_Rank"] == 0 else "unstable"
    rejected_note = f"  ({len(memorization_rejected)} memorizing config(s) skipped)" if memorization_rejected else ""
    print(f"  {target} -> {selected_row['Config_Name']} ({selected_row['Family']}, {stab_str}){rejected_note}")
    print(f"    CV:      R2={selected_row['CV_R2_mean']:.4f}+/-{selected_row['CV_R2_std']:.4f}  "
          f"RMSE={selected_row['CV_RMSE_mean']:.6f}  CoeffVar={selected_row['CV_RMSE_CoeffVar']:.1f}%  "
          f"violations={int(selected_row['CV_Physical_Violations_Total'])}")
    print(f"    Train:   R2={selected_tr_r2:.6f}  RMSE={tr_rmse_full:.6f}")
    print(f"    Holdout: R2={ho_r2:.6f}  RMSE={ho_rmse:.6f}  MAE={ho_mae:.6f}  "
          f"MAPE={ho_mape:.2f}%  violations={ho_violations}")

    final_selected.append({
        "Target": target,
        "Selected_Family": selected_row["Family"],
        "Selected_Config": selected_row["Config_Name"],
        "Parameters": selected_row["Parameters"],
        "CV_Rank": int(selected_row["Rank"]),
        "CV_R2_mean": selected_row["CV_R2_mean"],
        "CV_R2_std":  selected_row["CV_R2_std"],
        "CV_RMSE_mean": selected_row["CV_RMSE_mean"],
        "CV_RMSE_std":  selected_row["CV_RMSE_std"],
        "CV_RMSE_CoeffVar": selected_row["CV_RMSE_CoeffVar"],
        "Stability_Status": stab_str,
        "CV_MAE_mean": selected_row["CV_MAE_mean"],
        "CV_Physical_Violations_Total": int(selected_row["CV_Physical_Violations_Total"]),
        "Train_R2_full": selected_tr_r2,
        "Train_RMSE_full": tr_rmse_full,
        "Train_MAE_full": tr_mae_full,
        "Holdout_R2": ho_r2,
        "Holdout_RMSE": ho_rmse,
        "Holdout_MAE": ho_mae,
        "Holdout_MAPE": ho_mape,
        "Holdout_Physical_Violations": ho_violations,
        "Memorization_Configs_Rejected": len(memorization_rejected),
    })

    holdout_predictions[f"{target}_actual"]   = y_ho_true
    holdout_predictions[f"{target}_pred"]     = y_ho_pred
    holdout_predictions[f"{target}_residual"] = y_ho_true - y_ho_pred
    holdout_predictions[f"{target}_model"]    = selected_row["Config_Name"]

    train_predictions[f"{target}_actual"]   = y_tr_true
    train_predictions[f"{target}_pred"]     = y_tr_pred_full
    train_predictions[f"{target}_residual"] = y_tr_true - y_tr_pred_full
    train_predictions[f"{target}_model"]    = selected_row["Config_Name"]

final_selected_df = pd.DataFrame(final_selected)


# Baseline (Step 7) vs tuned (Step 8) comparison

step7_cv = pd.read_excel(STEP7_FILE, sheet_name="CV_Rankings")
baseline_vs_tuned = []
for target in TARGETS:
    base_sub = step7_cv[step7_cv["Target"] == target]
    t_row    = final_selected_df[final_selected_df["Target"] == target].iloc[0]
    if not base_sub.empty:
        b_row       = base_sub.iloc[0]
        rmse_diff   = t_row["CV_RMSE_mean"] - b_row["CV_RMSE_mean"]
        rmse_pct    = (rmse_diff / b_row["CV_RMSE_mean"]) * 100
        baseline_vs_tuned.append({
            "Target": target,
            "Baseline_Champion": f"{b_row['Model']} (Step 7)",
            "Tuned_Champion":    f"{t_row['Selected_Config']} ({t_row['Selected_Family']})",
            "Baseline_CV_RMSE": b_row["CV_RMSE_mean"],
            "Tuned_CV_RMSE":    t_row["CV_RMSE_mean"],
            "RMSE_Change":      rmse_diff,
            "RMSE_Pct_Change":  rmse_pct,
            "Baseline_CV_R2":   b_row["CV_R2_mean"],
            "Tuned_CV_R2":      t_row["CV_R2_mean"],
            "R2_Change":        t_row["CV_R2_mean"] - b_row["CV_R2_mean"],
            "Baseline_Violations": int(b_row["CV_Physical_Violations_Total"]),
            "Tuned_Violations":    int(t_row["CV_Physical_Violations_Total"]),
        })

baseline_vs_tuned_df = pd.DataFrame(baseline_vs_tuned)


# Write output workbook

with pd.ExcelWriter(OUTPUT_FILE, engine="openpyxl") as writer:

    readme = pd.DataFrame({
        "Item": [
            "Step", "Input", "Output", "Date",
            "CV_Strategy", "N_Splits", "N_Repeats", "Random_Seed",
            "Features", "Targets",
            "Train_rows", "Holdout_rows",
            "Total_Configurations",
            "Selection_Criteria",
            "Stability_Threshold",
            "Memorization_Threshold",
        ],
        "Value": [
            "Step 8 - ML Model Selection & Hyperparameter Tuning",
            str(INPUT_FILE.name),
            str(OUTPUT_FILE.name),
            pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
            "RepeatedKFold",
            N_SPLITS, N_REPEATS, SEED,
            ", ".join(FEATURES),
            ", ".join(TARGETS),
            X_train.shape[0], X_holdout.shape[0],
            len(all_configs),
            "violations (asc) -> stability (asc) -> RMSE (asc) -> MAE (asc) -> R2 (desc)",
            f"RMSE CoeffVar < {STABILITY_THRESHOLD}%",
            f"Train R2 > {MEMORIZATION_R2_THRESHOLD} for tree families",
        ],
    })
    readme.to_excel(writer, sheet_name="README", index=False)
    final_selected_df.to_excel(writer, sheet_name="Final_Selected_Models", index=False)
    baseline_vs_tuned_df.to_excel(writer, sheet_name="Baseline_vs_Tuned", index=False)
    ranking_df.to_excel(writer, sheet_name="CV_Rankings_All", index=False)
    holdout_predictions.to_excel(writer, sheet_name="Holdout_Predictions", index=False)
    train_predictions.to_excel(writer, sheet_name="Train_Predictions", index=False)

print(f"\nOutput written to: {OUTPUT_FILE.name}")
print("Step 8 complete.")
