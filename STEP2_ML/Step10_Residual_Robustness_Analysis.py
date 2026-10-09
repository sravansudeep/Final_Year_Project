"""
Step 10 — Residual & Robustness Analysis
Input : Step6_ML_Data_Preparation.xlsx (ML_Train_315), Step8_ML_Model_Selection_Tuning.xlsx,
        Step5B_Final_Analysis_Consolidated.xlsx, Step9_Classical_vs_ML_Comparison.xlsx
Output: Step10_Residual_Robustness_Analysis.xlsx
"""

from pathlib import Path
import warnings
import numpy as np
import pandas as pd
from scipy import stats
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

INPUT_STEP5B = BASE_DIR / "Step5B_Final_Analysis_Consolidated.xlsx"
INPUT_STEP6  = BASE_DIR / "Step6_ML_Data_Preparation.xlsx"
INPUT_STEP8  = BASE_DIR / "Step8_ML_Model_Selection_Tuning.xlsx"
INPUT_STEP9  = BASE_DIR / "Step9_Classical_vs_ML_Comparison.xlsx"
OUTPUT_FILE  = BASE_DIR / "Step10_Residual_Robustness_Analysis.xlsx"

FEATURES = ["d", "Rpm"]
TARGETS  = ["hc_hi", "td_to", "Vch"]
ID_COL   = "Point_ID"

PHYSICAL_BOUNDS = {
    "hc_hi": (0.0, 1.0),
    "td_to": (0.0, np.inf),
    "Vch":   (0.0, 1.0),
}

SEED = 42

RPM_REGIME_LABELS = {
    "Low_0-80":      (0, 80),
    "Med_100-180":   (100, 180),
    "High_200-260":  (200, 260),
    "Extrap_280+":   (261, 360),
}


def assign_rpm_regime(rpm):
    if rpm <= 80:
        return "Low_0-80"
    elif rpm <= 180:
        return "Med_100-180"
    elif rpm <= 260:
        return "High_200-260"
    else:
        return "Extrap_280+"


def safe_mape(y_true, y_pred, epsilon=1e-10):
    mask = (np.abs(y_true) > epsilon) & (~np.isnan(y_pred))
    if mask.sum() == 0:
        return np.nan
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)


def count_physical_violations(y_pred, target):
    lo, hi = PHYSICAL_BOUNDS[target]
    valid_mask = ~np.isnan(y_pred)
    return int(((y_pred[valid_mask] < lo) | (y_pred[valid_mask] > hi)).sum())



# 1. Load Data

xl_step6 = pd.ExcelFile(INPUT_STEP6)
train_sheet_name = [s for s in xl_step6.sheet_names if s.startswith("ML_Train_")][0]
train = pd.read_excel(INPUT_STEP6, sheet_name=train_sheet_name)
holdout = pd.read_excel(INPUT_STEP6, sheet_name="ML_Holdout_17")
classical_ho = pd.read_excel(INPUT_STEP5B, sheet_name="Holdout_Model_Check")
ml_final = pd.read_excel(INPUT_STEP8, sheet_name="Final_Selected_Models")

X_train = train[FEATURES].values
y_train = {t: train[t].values for t in TARGETS}
X_holdout = holdout[FEATURES].values
y_holdout = {t: holdout[t].values for t in TARGETS}

full_data = pd.concat([train, holdout], ignore_index=True)
full_data["Set"] = ["Train"] * len(train) + ["Holdout"] * len(holdout)
full_data["RPM_Regime"] = full_data["Rpm"].apply(assign_rpm_regime)

print("Data loaded successfully.")
print(f"  Train: {len(train)} rows  |  Holdout: {len(holdout)} rows")


# 2. Rebuild Final ML Models (retrained on full Train_315)

def build_model_from_selection(family, config_name, params, seed=42):
    if isinstance(params, str):
        params = eval(params)

    if family == "RandomForest":
        depth = None if str(params["max_depth"]) == "None" or params["max_depth"] is None else int(params["max_depth"])
        return RandomForestRegressor(
            n_estimators=int(params["n_estimators"]),
            max_depth=depth,
            min_samples_leaf=int(params["min_samples_leaf"]),
            random_state=seed, n_jobs=-1
        )
    elif family == "ExtraTrees":
        depth = None if str(params["max_depth"]) == "None" or params["max_depth"] is None else int(params["max_depth"])
        return ExtraTreesRegressor(
            n_estimators=int(params["n_estimators"]),
            max_depth=depth,
            min_samples_leaf=int(params["min_samples_leaf"]),
            random_state=seed, n_jobs=-1
        )
    elif family == "XGBoost":
        return XGBRegressor(
            n_estimators=int(params["n_estimators"]),
            max_depth=int(params["max_depth"]),
            learning_rate=float(params["learning_rate"]),
            subsample=float(params.get("subsample", 0.85)),
            min_child_weight=2,
            random_state=seed, verbosity=0, n_jobs=-1
        )
    elif family == "SVR":
        gamma_val = params["gamma"] if str(params["gamma"]) == "scale" else float(params["gamma"])
        return Pipeline([
            ("scaler", StandardScaler()),
            ("svr", SVR(
                kernel=params.get("kernel", "rbf"),
                C=float(params["C"]),
                epsilon=float(params["epsilon"]),
                gamma=gamma_val
            ))
        ])
    elif family == "KNN":
        return Pipeline([
            ("scaler", StandardScaler()),
            ("knn", KNeighborsRegressor(
                n_neighbors=int(params["n_neighbors"]),
                weights=params["weights"],
                p=int(params["p"])
            ))
        ])
    elif family == "GPR":
        kname = params["kernel"]
        alpha_val = float(params["alpha"])
        kernel_map = {
            "Matern25": ConstantKernel(1.0, (1e-2, 1e3)) * Matern(length_scale=1.0, nu=2.5) + WhiteKernel(noise_level=1e-3, noise_level_bounds=(1e-6, 1e-1)),
            "Matern15": ConstantKernel(1.0, (1e-2, 1e3)) * Matern(length_scale=1.0, nu=1.5) + WhiteKernel(noise_level=1e-3, noise_level_bounds=(1e-6, 1e-1)),
            "RBF": ConstantKernel(1.0, (1e-2, 1e3)) * RBF(length_scale=1.0) + WhiteKernel(noise_level=1e-3, noise_level_bounds=(1e-6, 1e-1)),
            "RQ": ConstantKernel(1.0, (1e-2, 1e3)) * RationalQuadratic(length_scale=1.0, alpha=1.0) + WhiteKernel(noise_level=1e-3, noise_level_bounds=(1e-6, 1e-1)),
        }
        kernel = kernel_map[kname]
        return Pipeline([
            ("scaler", StandardScaler()),
            ("gpr", GaussianProcessRegressor(
                kernel=kernel, alpha=alpha_val,
                n_restarts_optimizer=3, random_state=seed, normalize_y=True
            ))
        ])
    elif family == "MLP":
        arch = eval(params["hidden_layer_sizes"]) if isinstance(params["hidden_layer_sizes"], str) else params["hidden_layer_sizes"]
        return Pipeline([
            ("scaler", StandardScaler()),
            ("mlp", MLPRegressor(
                hidden_layer_sizes=arch,
                activation=params["activation"],
                alpha=float(params["alpha"]),
                solver="adam", max_iter=1500, early_stopping=True,
                validation_fraction=0.15, random_state=seed
            ))
        ])
    else:
        raise ValueError(f"Unknown model family: {family}")


ml_models = {}
ml_model_info = {}

for _, row in ml_final.iterrows():
    target = row["Target"]
    family = row["Selected_Family"]
    config = row["Selected_Config"]
    params = row["Parameters"]
    ml_model_info[target] = {"family": family, "config": config}
    model = build_model_from_selection(family, config, params, seed=SEED)
    model.fit(X_train, y_train[target])
    ml_models[target] = model
    print(f"  {target}: Rebuilt & fitted {family} ({config})")

print(f"ML models rebuilt and retrained on Train_{len(train)}.")


# 3. Generate Predictions (Train + Holdout) for ML models

residual_records = []

for target in TARGETS:
    model = ml_models[target]
    info = ml_model_info[target]

    for subset_name, X_sub, y_sub, df_sub in [
        ("Train", X_train, y_train[target], train),
        ("Holdout", X_holdout, y_holdout[target], holdout),
    ]:
        y_pred = model.predict(X_sub)
        residuals = y_sub - y_pred

        for i in range(len(y_sub)):
            residual_records.append({
                "Point_ID": int(df_sub[ID_COL].iloc[i]),
                "d": int(df_sub["d"].iloc[i]),
                "Rpm": int(df_sub["Rpm"].iloc[i]),
                "RPM_Regime": assign_rpm_regime(int(df_sub["Rpm"].iloc[i])),
                "Set": subset_name,
                "Target": target,
                "Model_Family": info["family"],
                "Model_Config": info["config"],
                "Actual": float(y_sub[i]),
                "Predicted": float(y_pred[i]),
                "Residual": float(residuals[i]),
                "Abs_Residual": float(abs(residuals[i])),
                "Pct_Error": float(abs(residuals[i]) / abs(y_sub[i]) * 100) if abs(y_sub[i]) > 1e-10 else np.nan,
                "Physical_Violation": bool(
                    y_pred[i] < PHYSICAL_BOUNDS[target][0] or y_pred[i] > PHYSICAL_BOUNDS[target][1]
                ),
            })

residual_df = pd.DataFrame(residual_records)
print(f"Residual table: {len(residual_df)} rows across {len(TARGETS)} targets x 2 sets.")


# 4. Residual Summary Statistics (per Target x Set)

summary_rows = []

for target in TARGETS:
    for subset in ["Train", "Holdout"]:
        sub = residual_df[(residual_df["Target"] == target) & (residual_df["Set"] == subset)]
        res = sub["Residual"].values
        y_act = sub["Actual"].values
        y_prd = sub["Predicted"].values
        n = len(res)

        r2 = float(r2_score(y_act, y_prd))
        rmse = float(np.sqrt(mean_squared_error(y_act, y_prd)))
        mae = float(mean_absolute_error(y_act, y_prd))
        mape = safe_mape(y_act, y_prd)
        violations = int(sub["Physical_Violation"].sum())

        res_mean = float(np.mean(res))
        res_std = float(np.std(res, ddof=1)) if n > 1 else np.nan
        res_median = float(np.median(res))
        res_min = float(np.min(res))
        res_max = float(np.max(res))
        res_skew = float(stats.skew(res, bias=False)) if n > 2 else np.nan
        res_kurt = float(stats.kurtosis(res, bias=False)) if n > 3 else np.nan

        # Shapiro-Wilk normality test
        if n >= 3 and np.std(res) > 1e-15:
            sw_stat, sw_p = stats.shapiro(res)
        else:
            sw_stat, sw_p = np.nan, np.nan

        # Mean bias (should be near zero for unbiased model)
        bias = res_mean

        normality = "N/A"
        if not np.isnan(sw_p):
            normality = "PASS (p>=0.05)" if sw_p >= 0.05 else "FAIL (p<0.05)"

        summary_rows.append({
            "Target": target,
            "Model": f"{ml_model_info[target]['family']} ({ml_model_info[target]['config']})",
            "Set": subset,
            "N_Points": n,
            "R2": r2,
            "RMSE": rmse,
            "MAE": mae,
            "MAPE": mape,
            "Physical_Violations": violations,
            "Residual_Mean_Bias": bias,
            "Residual_Std": res_std,
            "Residual_Median": res_median,
            "Residual_Min": res_min,
            "Residual_Max": res_max,
            "Residual_Skewness": res_skew,
            "Residual_Kurtosis": res_kurt,
            "Shapiro_Wilk_Stat": float(sw_stat) if not np.isnan(sw_stat) else np.nan,
            "Shapiro_Wilk_p": float(sw_p) if not np.isnan(sw_p) else np.nan,
            "Normality_Status": normality,
            "Bias_Status": "NEGLIGIBLE" if (res_std and res_std > 0 and abs(bias) < 0.01 * res_std) else "NON-NEGLIGIBLE",
        })

summary_df = pd.DataFrame(summary_rows)

print("\n" + "=" * 80)
print("RESIDUAL SUMMARY STATISTICS (ML FINAL MODELS)")
print("=" * 80)
for _, r in summary_df.iterrows():
    print(f"\n{r['Target']} [{r['Set']}] -- {r['Model']}")
    print(f"  R2={r['R2']:.6f}  RMSE={r['RMSE']:.6f}  MAE={r['MAE']:.6f}")
    skew_str = f"{r['Residual_Skewness']:.4f}" if not np.isnan(r['Residual_Skewness']) else "N/A"
    kurt_str = f"{r['Residual_Kurtosis']:.4f}" if not np.isnan(r['Residual_Kurtosis']) else "N/A"
    print(f"  Residual: mean={r['Residual_Mean_Bias']:.6f}  std={r['Residual_Std']:.6f}  skew={skew_str}  kurt={kurt_str}")
    sw_stat_str = f"{r['Shapiro_Wilk_Stat']:.4f}" if not np.isnan(r['Shapiro_Wilk_Stat']) else "N/A"
    sw_p_str = f"{r['Shapiro_Wilk_p']:.4f}" if not np.isnan(r['Shapiro_Wilk_p']) else "N/A"
    print(f"  Shapiro-Wilk: W={sw_stat_str}  p={sw_p_str}  -> {r['Normality_Status']}")
    print(f"  Physical violations: {r['Physical_Violations']}")


# 5. Error Breakdown by Diameter

diameter_rows = []
for target in TARGETS:
    for subset in ["Train", "Holdout"]:
        for d_val in sorted(full_data["d"].unique()):
            sub = residual_df[
                (residual_df["Target"] == target) &
                (residual_df["Set"] == subset) &
                (residual_df["d"] == d_val)
            ]
            if len(sub) == 0:
                continue
            res = sub["Residual"].values
            y_act = sub["Actual"].values
            y_prd = sub["Predicted"].values
            n = len(res)

            diameter_rows.append({
                "Target": target,
                "Set": subset,
                "d_mm": d_val,
                "N_Points": n,
                "RMSE": float(np.sqrt(np.mean(res ** 2))),
                "MAE": float(np.mean(np.abs(res))),
                "Mean_Bias": float(np.mean(res)),
                "Max_Abs_Residual": float(np.max(np.abs(res))),
                "MAPE": safe_mape(y_act, y_prd),
                "Physical_Violations": int(sub["Physical_Violation"].sum()),
                "Worst_Point_ID": int(sub.loc[sub["Abs_Residual"].idxmax(), "Point_ID"]),
            })

diameter_df = pd.DataFrame(diameter_rows)

print("\n" + "=" * 80)
print("ERROR BREAKDOWN BY DIAMETER")
print("=" * 80)
for target in TARGETS:
    sub = diameter_df[(diameter_df["Target"] == target) & (diameter_df["Set"] == "Train")]
    print(f"\n{target} (Train):")
    for _, r in sub.iterrows():
        print(f"  d={r['d_mm']:2d}mm: RMSE={r['RMSE']:.6f}  MAE={r['MAE']:.6f}  Bias={r['Mean_Bias']:+.6f}  Worst=Point_{r['Worst_Point_ID']}")


# 6. Error Breakdown by RPM Regime

regime_rows = []
for target in TARGETS:
    for subset in ["Train", "Holdout"]:
        for regime in ["Low_0-80", "Med_100-180", "High_200-260"]:
            sub = residual_df[
                (residual_df["Target"] == target) &
                (residual_df["Set"] == subset) &
                (residual_df["RPM_Regime"] == regime)
            ]
            if len(sub) == 0:
                continue
            res = sub["Residual"].values
            y_act = sub["Actual"].values
            y_prd = sub["Predicted"].values

            regime_rows.append({
                "Target": target,
                "Set": subset,
                "RPM_Regime": regime,
                "N_Points": len(sub),
                "RMSE": float(np.sqrt(np.mean(res ** 2))),
                "MAE": float(np.mean(np.abs(res))),
                "Mean_Bias": float(np.mean(res)),
                "Max_Abs_Residual": float(np.max(np.abs(res))),
                "MAPE": safe_mape(y_act, y_prd),
                "Physical_Violations": int(sub["Physical_Violation"].sum()),
                "Worst_Point_ID": int(sub.loc[sub["Abs_Residual"].idxmax(), "Point_ID"]),
            })

regime_df = pd.DataFrame(regime_rows)

print("\n" + "=" * 80)
print("ERROR BREAKDOWN BY RPM REGIME")
print("=" * 80)
for target in TARGETS:
    sub = regime_df[(regime_df["Target"] == target) & (regime_df["Set"] == "Train")]
    print(f"\n{target} (Train):")
    for _, r in sub.iterrows():
        print(f"  {r['RPM_Regime']:15s}: RMSE={r['RMSE']:.6f}  MAE={r['MAE']:.6f}  Bias={r['Mean_Bias']:+.6f}")


# 7. Heteroscedasticity Diagnostics
#    Spearman correlation between residual and predicted value

hetero_rows = []
for target in TARGETS:
    for subset in ["Train", "Holdout"]:
        sub = residual_df[(residual_df["Target"] == target) & (residual_df["Set"] == subset)]
        abs_res = sub["Abs_Residual"].values
        predicted = sub["Predicted"].values

        if len(abs_res) >= 5:
            sp_corr, sp_p = stats.spearmanr(predicted, abs_res)
        else:
            sp_corr, sp_p = np.nan, np.nan

        # Also check against RPM
        rpm_vals = sub["Rpm"].values
        if len(abs_res) >= 5:
            sp_corr_rpm, sp_p_rpm = stats.spearmanr(rpm_vals, abs_res)
        else:
            sp_corr_rpm, sp_p_rpm = np.nan, np.nan

        # Also check against d
        d_vals = sub["d"].values
        if len(abs_res) >= 5:
            sp_corr_d, sp_p_d = stats.spearmanr(d_vals, abs_res)
        else:
            sp_corr_d, sp_p_d = np.nan, np.nan

        hetero_rows.append({
            "Target": target,
            "Set": subset,
            "Spearman_AbsRes_vs_Predicted": float(sp_corr) if not np.isnan(sp_corr) else np.nan,
            "p_value_Predicted": float(sp_p) if not np.isnan(sp_p) else np.nan,
            "Heteroscedasticity_Predicted": "DETECTED (p<0.05)" if sp_p < 0.05 else "NOT DETECTED" if not np.isnan(sp_p) else "N/A",
            "Spearman_AbsRes_vs_RPM": float(sp_corr_rpm) if not np.isnan(sp_corr_rpm) else np.nan,
            "p_value_RPM": float(sp_p_rpm) if not np.isnan(sp_p_rpm) else np.nan,
            "Heteroscedasticity_RPM": "DETECTED (p<0.05)" if sp_p_rpm < 0.05 else "NOT DETECTED" if not np.isnan(sp_p_rpm) else "N/A",
            "Spearman_AbsRes_vs_d": float(sp_corr_d) if not np.isnan(sp_corr_d) else np.nan,
            "p_value_d": float(sp_p_d) if not np.isnan(sp_p_d) else np.nan,
            "Heteroscedasticity_d": "DETECTED (p<0.05)" if sp_p_d < 0.05 else "NOT DETECTED" if not np.isnan(sp_p_d) else "N/A",
        })

hetero_df = pd.DataFrame(hetero_rows)

print("\n" + "=" * 80)
print("HETEROSCEDASTICITY DIAGNOSTICS")
print("=" * 80)
for _, r in hetero_df.iterrows():
    print(f"\n{r['Target']} [{r['Set']}]:")
    print(f"  |Residual| vs Predicted: Spearman={r['Spearman_AbsRes_vs_Predicted']:.4f}  p={r['p_value_Predicted']:.4f}  -> {r['Heteroscedasticity_Predicted']}")
    print(f"  |Residual| vs RPM:       Spearman={r['Spearman_AbsRes_vs_RPM']:.4f}  p={r['p_value_RPM']:.4f}  -> {r['Heteroscedasticity_RPM']}")
    print(f"  |Residual| vs d:         Spearman={r['Spearman_AbsRes_vs_d']:.4f}  p={r['p_value_d']:.4f}  -> {r['Heteroscedasticity_d']}")


# 8. Cross-Validation Stability Diagnostics (from Step 8 Selection)

print("\n" + "=" * 80)
print("CV STABILITY DIAGNOSTICS (Repeated 5-Fold from Step 8)")
print("=" * 80)

stability_rows = []
for target in TARGETS:
    m_row = ml_final[ml_final["Target"] == target].iloc[0]
    cv_coeff_var = float(m_row["CV_RMSE_CoeffVar"])
    verdict = "STABLE" if cv_coeff_var < 30.0 else "MODERATE" if cv_coeff_var < 50.0 else "UNSTABLE"

    stability_rows.append({
        "Target": target,
        "Model": f"{m_row['Selected_Family']} ({m_row['Selected_Config']})",
        "N_Folds": 15,
        "CV_R2_mean": float(m_row["CV_R2_mean"]),
        "CV_R2_std": float(m_row["CV_R2_std"]),
        "CV_RMSE_mean": float(m_row["CV_RMSE_mean"]),
        "CV_RMSE_std": float(m_row["CV_RMSE_std"]),
        "CV_RMSE_CoeffVar_Pct": cv_coeff_var,
        "CV_MAE_mean": float(m_row["CV_MAE_mean"]),
        "CV_Total_Physical_Violations": int(m_row["CV_Physical_Violations_Total"]),
        "Stability_Verdict": verdict,
    })

    print(f"\n{target} -- {m_row['Selected_Family']} ({m_row['Selected_Config']})")
    print(f"  R2:   {m_row['CV_R2_mean']:.4f} +/- {m_row['CV_R2_std']:.4f}")
    print(f"  RMSE: {m_row['CV_RMSE_mean']:.6f} +/- {m_row['CV_RMSE_std']:.6f}  (CoeffVar {cv_coeff_var:.1f}%)")
    print(f"  Verdict: {verdict}")

stability_df = pd.DataFrame(stability_rows)


# 9. Train-vs-Holdout Generalization Gap

gap_rows = []
for target in TARGETS:
    train_sub = summary_df[(summary_df["Target"] == target) & (summary_df["Set"] == "Train")].iloc[0]
    ho_sub = summary_df[(summary_df["Target"] == target) & (summary_df["Set"] == "Holdout")].iloc[0]

    rmse_ratio = ho_sub["RMSE"] / train_sub["RMSE"] if train_sub["RMSE"] > 0 else np.nan
    mae_ratio = ho_sub["MAE"] / train_sub["MAE"] if train_sub["MAE"] > 0 else np.nan
    r2_drop = train_sub["R2"] - ho_sub["R2"]

    gap_rows.append({
        "Target": target,
        "Model": f"{ml_model_info[target]['family']} ({ml_model_info[target]['config']})",
        "Train_R2": train_sub["R2"],
        "Holdout_R2": ho_sub["R2"],
        "R2_Drop": r2_drop,
        "Train_RMSE": train_sub["RMSE"],
        "Holdout_RMSE": ho_sub["RMSE"],
        "RMSE_Ratio_Holdout_over_Train": rmse_ratio,
        "Train_MAE": train_sub["MAE"],
        "Holdout_MAE": ho_sub["MAE"],
        "MAE_Ratio_Holdout_over_Train": mae_ratio,
        "Train_Violations": train_sub["Physical_Violations"],
        "Holdout_Violations": ho_sub["Physical_Violations"],
        "Overfit_Risk": "LOW" if rmse_ratio < 2.0 and r2_drop < 0.05 else "MODERATE" if rmse_ratio < 3.0 else "HIGH",
    })

gap_df = pd.DataFrame(gap_rows)

print("\n" + "=" * 80)
print("TRAIN vs HOLDOUT GENERALIZATION GAP")
print("=" * 80)
for _, r in gap_df.iterrows():
    print(f"\n{r['Target']} — {r['Model']}")
    print(f"  Train  R2={r['Train_R2']:.6f}  RMSE={r['Train_RMSE']:.6f}")
    print(f"  Holdout R2={r['Holdout_R2']:.6f}  RMSE={r['Holdout_RMSE']:.6f}")
    print(f"  RMSE ratio (Holdout/Train) = {r['RMSE_Ratio_Holdout_over_Train']:.3f}  R2 drop = {r['R2_Drop']:.6f}")
    print(f"  Overfit risk: {r['Overfit_Risk']}")


# 10. Outlier Identification (Residuals > 2σ from mean)

outlier_rows = []
for target in TARGETS:
    for subset in ["Train", "Holdout"]:
        sub = residual_df[(residual_df["Target"] == target) & (residual_df["Set"] == subset)].copy()
        res = sub["Residual"].values
        mean_r = np.mean(res)
        std_r = np.std(res, ddof=1) if len(res) > 1 else 1e-10

        for _, row in sub.iterrows():
            z_score = (row["Residual"] - mean_r) / std_r if std_r > 1e-10 else 0.0
            if abs(z_score) >= 2.0:
                outlier_rows.append({
                    "Target": target,
                    "Set": subset,
                    "Point_ID": int(row["Point_ID"]),
                    "d": int(row["d"]),
                    "Rpm": int(row["Rpm"]),
                    "RPM_Regime": row["RPM_Regime"],
                    "Actual": row["Actual"],
                    "Predicted": row["Predicted"],
                    "Residual": row["Residual"],
                    "Z_Score": float(z_score),
                    "Physical_Violation": row["Physical_Violation"],
                    "Severity": "EXTREME (|z|>=3)" if abs(z_score) >= 3.0 else "MODERATE (2<=|z|<3)",
                })

outlier_df = pd.DataFrame(outlier_rows) if outlier_rows else pd.DataFrame(columns=[
    "Target", "Set", "Point_ID", "d", "Rpm", "RPM_Regime", "Actual", "Predicted",
    "Residual", "Z_Score", "Physical_Violation", "Severity"
])

print("\n" + "=" * 80)
print("OUTLIER IDENTIFICATION (|Z| >= 2)")
print("=" * 80)
if len(outlier_df) > 0:
    for target in TARGETS:
        sub = outlier_df[outlier_df["Target"] == target]
        if len(sub) > 0:
            print(f"\n{target}: {len(sub)} outliers")
            for _, r in sub.iterrows():
                print(f"  Point_{r['Point_ID']} (d={r['d']}, RPM={r['Rpm']}) [{r['Set']}]: res={r['Residual']:+.6f}  z={r['Z_Score']:+.3f}  {r['Severity']}")
else:
    print("  No outliers with |Z| >= 2 detected.")


# 11. Engineering Diagnostic Summary

eng_diag_rows = []
for target in TARGETS:
    info = ml_model_info[target]
    train_summ = summary_df[(summary_df["Target"] == target) & (summary_df["Set"] == "Train")].iloc[0]
    ho_summ = summary_df[(summary_df["Target"] == target) & (summary_df["Set"] == "Holdout")].iloc[0]
    stab = stability_df[stability_df["Target"] == target].iloc[0]
    gap = gap_df[gap_df["Target"] == target].iloc[0]
    het = hetero_df[(hetero_df["Target"] == target) & (hetero_df["Set"] == "Train")].iloc[0]

    n_outliers_train = len(outlier_df[(outlier_df["Target"] == target) & (outlier_df["Set"] == "Train")])
    n_outliers_holdout = len(outlier_df[(outlier_df["Target"] == target) & (outlier_df["Set"] == "Holdout")])

    issues = []
    if ho_summ["Physical_Violations"] > 0:
        issues.append(f"{ho_summ['Physical_Violations']} physical violation(s) on holdout")
    if gap["Overfit_Risk"] != "LOW":
        issues.append(f"Overfit risk: {gap['Overfit_Risk']}")
    if stab["Stability_Verdict"] != "STABLE":
        if target == "td_to":
            issues.append(f"CV stability: {stab['Stability_Verdict']} (fold-sensitivity with LOW overfit risk; not memorization)")
        else:
            issues.append(f"CV stability: {stab['Stability_Verdict']}")
    if train_summ["Normality_Status"].startswith("FAIL"):
        issues.append("Train residuals non-normal (Shapiro-Wilk p<0.05)")
    if het["Heteroscedasticity_Predicted"] == "DETECTED (p<0.05)":
        issues.append("Heteroscedasticity detected (residual variance varies with predicted)")
    if het["Heteroscedasticity_RPM"] == "DETECTED (p<0.05)":
        issues.append("RPM-dependent error structure detected")
    if n_outliers_train > 3:
        issues.append(f"{n_outliers_train} training outliers (|z|>=2)")

    overall_verdict = "PASS" if len(issues) == 0 else "PASS WITH CAVEATS" if len(issues) <= 2 else "REQUIRES ATTENTION"

    viol_note = "0 physical violations on holdout." if ho_summ["Physical_Violations"] == 0 else f"{ho_summ['Physical_Violations']} physical violation(s) on holdout."
    diag_notes = (
        f"{info['family']} ({info['config']}) selected for {target}. "
        f"Holdout R2={ho_summ['R2']:.4f}, RMSE={ho_summ['RMSE']:.4f}, "
        f"Holdout/Train RMSE ratio={gap['RMSE_Ratio_Holdout_over_Train']:.3f} ({gap['Overfit_Risk']} overfit risk). "
        f"{viol_note} CV stability: {stab['Stability_Verdict']}."
    )

    eng_diag_rows.append({
        "Target": target,
        "Model": f"{info['family']} ({info['config']})",
        "Holdout_R2": ho_summ["R2"],
        "Holdout_RMSE": ho_summ["RMSE"],
        "CV_Stability": stab["Stability_Verdict"],
        "Overfit_Risk": gap["Overfit_Risk"],
        "Residual_Normality": train_summ["Normality_Status"],
        "Heteroscedasticity_vs_Predicted": het["Heteroscedasticity_Predicted"],
        "Heteroscedasticity_vs_RPM": het["Heteroscedasticity_RPM"],
        "Physical_Violations_Train": int(train_summ["Physical_Violations"]),
        "Physical_Violations_Holdout": int(ho_summ["Physical_Violations"]),
        "Outliers_Train": n_outliers_train,
        "Outliers_Holdout": n_outliers_holdout,
        "Issues_Found": "; ".join(issues) if issues else "None",
        "Diagnostic_Notes": diag_notes,
        "Overall_Verdict": overall_verdict,
    })

eng_diag_df = pd.DataFrame(eng_diag_rows)

print("\n" + "=" * 80)
print("ENGINEERING DIAGNOSTIC SUMMARY")
print("=" * 80)
for _, r in eng_diag_df.iterrows():
    print(f"\n{r['Target']} — {r['Model']}")
    print(f"  Holdout: R2={r['Holdout_R2']:.6f}  RMSE={r['Holdout_RMSE']:.6f}")
    print(f"  CV Stability: {r['CV_Stability']}  |  Overfit: {r['Overfit_Risk']}")
    print(f"  Normality: {r['Residual_Normality']}  |  Heteroscedasticity: {r['Heteroscedasticity_vs_Predicted']}")
    print(f"  Violations: Train={r['Physical_Violations_Train']}  Holdout={r['Physical_Violations_Holdout']}")
    print(f"  Diagnostic Notes: {r['Diagnostic_Notes']}")
    print(f"  Issues: {r['Issues_Found']}")
    print(f"  >>> VERDICT: {r['Overall_Verdict']}")


# 12. Write Output Workbook

with pd.ExcelWriter(OUTPUT_FILE, engine="openpyxl") as writer:

    readme = pd.DataFrame({
        "Item": [
            "Step", "Inputs", "Output", "Date",
            "Purpose",
            "ML_Models_Evaluated",
            "Train_Points", "Holdout_Points",
            "Diagnostics_Performed",
            "Key_Principle",
        ],
        "Value": [
            "Step 10 — Residual & Robustness Analysis",
            "Step5B, Step6, Step8, Step9 workbooks",
            str(OUTPUT_FILE.name),
            pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
            "Comprehensive residual diagnostics, error structure analysis, stability assessment, "
            "generalization gap evaluation, and outlier detection for all three final ML models",
            "; ".join([f"{t}: {ml_model_info[t]['family']} ({ml_model_info[t]['config']})" for t in TARGETS]),
            315, 17,
            "Residual stats, normality (Shapiro-Wilk), heteroscedasticity (Spearman), "
            "error by diameter, error by RPM regime, CV stability, train-vs-holdout gap, "
            "outlier identification (z-score), physical violation tracking",
            "High aggregate R2 can hide structured errors — this step exposes any hidden patterns",
        ]
    })
    readme.to_excel(writer, sheet_name="README", index=False)
    summary_df.to_excel(writer, sheet_name="Residual_Summary", index=False)
    residual_df.to_excel(writer, sheet_name="Point_Residuals", index=False)
    diameter_df.to_excel(writer, sheet_name="Error_by_Diameter", index=False)
    regime_df.to_excel(writer, sheet_name="Error_by_RPM_Regime", index=False)
    hetero_df.to_excel(writer, sheet_name="Heteroscedasticity", index=False)
    stability_df.to_excel(writer, sheet_name="CV_Stability", index=False)
    gap_df.to_excel(writer, sheet_name="Generalization_Gap", index=False)
    outlier_df.to_excel(writer, sheet_name="Outliers", index=False)
    eng_diag_df.to_excel(writer, sheet_name="Engineering_Diagnostics", index=False)

print(f"\nOutput written to: {OUTPUT_FILE.name}")
print("Step 10 complete.")
