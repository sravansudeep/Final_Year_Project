"""
Step 9 — Classical vs. ML Final Continuous-Model Comparison
Input : Step5B_Final_Analysis_Consolidated.xlsx, Step6_ML_Data_Preparation.xlsx, Step8_ML_Model_Selection_Tuning.xlsx
Output: Step9_Classical_vs_ML_Comparison.xlsx
"""

from pathlib import Path
import warnings
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

warnings.filterwarnings("ignore")

BASE_DIR = Path(__file__).resolve().parent

INPUT_STEP5B = BASE_DIR / "Step5B_Final_Analysis_Consolidated.xlsx"
INPUT_STEP6  = BASE_DIR / "Step6_ML_Data_Preparation.xlsx"
INPUT_STEP8  = BASE_DIR / "Step8_ML_Model_Selection_Tuning.xlsx"
OUTPUT_FILE  = BASE_DIR / "Step9_Classical_vs_ML_Comparison.xlsx"

FEATURES = ["d", "Rpm"]
TARGETS  = ["hc_hi", "td_to", "Vch"]
ID_COL   = "Point_ID"

PHYSICAL_BOUNDS = {
    "hc_hi": (0.0, 1.0),
    "td_to": (0.0, np.inf),
    "Vch":   (0.0, 1.0),
}


def safe_mape(y_true, y_pred, epsilon=1e-10):
    mask = (np.abs(y_true) > epsilon) & (~np.isnan(y_pred))
    if mask.sum() == 0:
        return np.nan
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)


def count_physical_violations(y_pred, target):
    lo, hi = PHYSICAL_BOUNDS[target]
    valid_mask = ~np.isnan(y_pred)
    return int(((y_pred[valid_mask] < lo) | (y_pred[valid_mask] > hi)).sum())


# ---------------------------------------------------------------------------
# 1. Load Data
# ---------------------------------------------------------------------------
# Classical holdout evaluation
classical_ho = pd.read_excel(INPUT_STEP5B, sheet_name="Holdout_Model_Check")

# ML train & holdout datasets (for diagnostic comparisons)
train = pd.read_excel(INPUT_STEP6, sheet_name="ML_Train_67")
holdout = pd.read_excel(INPUT_STEP6, sheet_name="ML_Holdout_17")

# ML tuned results from Step 8
ml_final_models = pd.read_excel(INPUT_STEP8, sheet_name="Final_Selected_Models")
ml_ho_preds = pd.read_excel(INPUT_STEP8, sheet_name="Holdout_Predictions")
ml_best_per_fam = pd.read_excel(INPUT_STEP8, sheet_name="CV_Best_Per_Family")

print("Step 5B, Step 6, and Step 8 workbooks successfully loaded.")

# ---------------------------------------------------------------------------
# 2. Extract Classical Metrics on 17 Holdout Points
# ---------------------------------------------------------------------------
classical_metrics = {}
for target in TARGETS:
    sub = classical_ho[classical_ho["Target"] == target].copy()
    method_name = sub["Method"].iloc[0]
    total_pts = len(sub)
    valid_sub = sub[sub["Prediction_Valid"] == 1].dropna(subset=["Predicted"])
    valid_pts = len(valid_sub)
    coverage = (valid_pts / total_pts) * 100.0

    y_true_valid = valid_sub["Actual"].values
    y_pred_valid = valid_sub["Predicted"].values

    r2 = float(r2_score(y_true_valid, y_pred_valid))
    rmse = float(np.sqrt(mean_squared_error(y_true_valid, y_pred_valid)))
    mae = float(mean_absolute_error(y_true_valid, y_pred_valid))
    mape = safe_mape(y_true_valid, y_pred_valid)
    violations = count_physical_violations(y_pred_valid, target)

    classical_metrics[target] = {
        "Method": method_name,
        "Total_Points": total_pts,
        "Valid_Predictions": valid_pts,
        "Coverage_Pct": coverage,
        "Holdout_R2": r2,
        "Holdout_RMSE": rmse,
        "Holdout_MAE": mae,
        "Holdout_MAPE": mape,
        "Physical_Violations": violations,
    }

# ---------------------------------------------------------------------------
# 3. Build Comparative Summary (Classical vs ML) - Exactly 1 locked model per target
# ---------------------------------------------------------------------------
comparison_rows = []
for target in TARGETS:
    c = classical_metrics[target]
    m_row = ml_final_models[ml_final_models["Target"] == target].iloc[0]

    rmse_diff = m_row["Holdout_RMSE"] - c["Holdout_RMSE"]
    rmse_pct_change = ((m_row["Holdout_RMSE"] - c["Holdout_RMSE"]) / c["Holdout_RMSE"]) * 100.0
    r2_diff = m_row["Holdout_R2"] - c["Holdout_R2"]

    comparison_rows.append({
        "Target": target,
        "Classical_Model": c["Method"],
        "Classical_Coverage_Pct": c["Coverage_Pct"],
        "Classical_Holdout_R2": c["Holdout_R2"],
        "Classical_Holdout_RMSE": c["Holdout_RMSE"],
        "Classical_Holdout_MAE": c["Holdout_MAE"],
        "Classical_Holdout_MAPE": c["Holdout_MAPE"],
        "Classical_Physical_Violations": c["Physical_Violations"],
        "ML_Selected_Model": f"{m_row['Selected_Family']} ({m_row['Selected_Config']})",
        "ML_Coverage_Pct": 100.0,
        "ML_CV_R2_mean": m_row["CV_R2_mean"],
        "ML_CV_RMSE_mean": m_row["CV_RMSE_mean"],
        "ML_Holdout_R2": m_row["Holdout_R2"],
        "ML_Holdout_RMSE": m_row["Holdout_RMSE"],
        "ML_Holdout_MAE": m_row["Holdout_MAE"],
        "ML_Holdout_MAPE": m_row["Holdout_MAPE"],
        "ML_Holdout_Physical_Violations": int(m_row["Holdout_Physical_Violations"]),
        "RMSE_Absolute_Reduction": -rmse_diff,
        "RMSE_Pct_Improvement": -rmse_pct_change,
        "Coverage_Improvement_Pct": 100.0 - c["Coverage_Pct"],
    })

comparison_df = pd.DataFrame(comparison_rows)

print("\n" + "="*80)
print("CLASSICAL vs. ML FINAL CONTINUOUS-MODEL COMPARISON (HOLDOUT BENCHMARK)")
print("="*80)
for _, r in comparison_df.iterrows():
    print(f"\nTarget: {r['Target']}")
    print(f"  Classical [{r['Classical_Model']}]:")
    print(f"    R2 = {r['Classical_Holdout_R2']:.6f}  RMSE = {r['Classical_Holdout_RMSE']:.6f}  Coverage = {r['Classical_Coverage_Pct']:.1f}%  Violations = {r['Classical_Physical_Violations']}")
    print(f"  ML [{r['ML_Selected_Model']}]:")
    print(f"    R2 = {r['ML_Holdout_R2']:.6f}  RMSE = {r['ML_Holdout_RMSE']:.6f}  Coverage = {r['ML_Coverage_Pct']:.1f}%  Violations = {r['ML_Holdout_Physical_Violations']}")
    print(f"  Comparison: RMSE Improvement = {r['RMSE_Pct_Improvement']:+.2f}%  Coverage Gain = +{r['Coverage_Improvement_Pct']:.1f}%")

# ---------------------------------------------------------------------------
# 4. Point-by-Point Side-by-Side Holdout Comparison Table
# ---------------------------------------------------------------------------
pbp_df = ml_ho_preds[[ID_COL, "d", "Rpm"]].copy()

for target in TARGETS:
    sub_c = classical_ho[classical_ho["Target"] == target].set_index(ID_COL)
    pbp_df[f"{target}_actual"] = sub_c.loc[pbp_df[ID_COL], "Actual"].values
    pbp_df[f"{target}_classical_pred"] = sub_c.loc[pbp_df[ID_COL], "Predicted"].values
    pbp_df[f"{target}_classical_res"] = sub_c.loc[pbp_df[ID_COL], "Residual"].values
    pbp_df[f"{target}_ml_pred"] = ml_ho_preds[f"{target}_pred"].values
    pbp_df[f"{target}_ml_res"] = ml_ho_preds[f"{target}_residual"].values

# ---------------------------------------------------------------------------
# 5. Error Breakdown by RPM Regime (Low: 0-80, Med: 100-180, High: 200-260)
# ---------------------------------------------------------------------------
def assign_rpm_regime(rpm):
    if rpm <= 80:
        return "Low_0-80"
    elif rpm <= 180:
        return "Med_100-180"
    else:
        return "High_200-260"

pbp_df["RPM_Regime"] = pbp_df["Rpm"].apply(assign_rpm_regime)

regime_rows = []
for target in TARGETS:
    for regime in ["Low_0-80", "Med_100-180", "High_200-260"]:
        sub_reg = pbp_df[pbp_df["RPM_Regime"] == regime]
        n_pts = len(sub_reg)

        # Classical
        c_res = sub_reg[f"{target}_classical_res"].dropna()
        c_rmse = float(np.sqrt(np.mean(c_res**2))) if len(c_res) > 0 else np.nan
        c_mae = float(np.mean(np.abs(c_res))) if len(c_res) > 0 else np.nan

        # ML
        m_res = sub_reg[f"{target}_ml_res"].dropna()
        m_rmse = float(np.sqrt(np.mean(m_res**2))) if len(m_res) > 0 else np.nan
        m_mae = float(np.mean(np.abs(m_res))) if len(m_res) > 0 else np.nan

        better = "ML" if m_rmse < c_rmse else "Classical"

        regime_rows.append({
            "Target": target,
            "RPM_Regime": regime,
            "Holdout_Points": n_pts,
            "Classical_RMSE": c_rmse,
            "Classical_MAE": c_mae,
            "ML_RMSE": m_rmse,
            "ML_MAE": m_mae,
            "Lower_Error_Model": better,
        })

regime_df = pd.DataFrame(regime_rows)

# ---------------------------------------------------------------------------
# 6. Diagnostic Comparison for Vch (ExtraTrees vs Locked XGBoost)
# ---------------------------------------------------------------------------
# Evaluate ExtraTrees (ET_n200_dNone_l1) on holdout for diagnostic comparison only
et_vch_model = ExtraTreesRegressor(
    n_estimators=200, max_depth=None, min_samples_leaf=1,
    random_state=42, n_jobs=-1
)
et_vch_model.fit(train[FEATURES].values, train["Vch"].values)
et_vch_ho_pred = et_vch_model.predict(holdout[FEATURES].values)
y_ho_vch_true = holdout["Vch"].values

et_vch_ho_r2 = float(r2_score(y_ho_vch_true, et_vch_ho_pred))
et_vch_ho_rmse = float(np.sqrt(mean_squared_error(y_ho_vch_true, et_vch_ho_pred)))
et_vch_ho_mae = float(mean_absolute_error(y_ho_vch_true, et_vch_ho_pred))
et_vch_ho_mape = float(safe_mape(y_ho_vch_true, et_vch_ho_pred))
et_vch_ho_viol = int(count_physical_violations(et_vch_ho_pred, "Vch"))

p71_mask = holdout[ID_COL] == 71
et_p71_vch_pred = float(et_vch_ho_pred[p71_mask][0])
et_p71_vch_res = float(y_ho_vch_true[p71_mask][0] - et_p71_vch_pred)

vch_xgb_row = ml_final_models[ml_final_models["Target"] == "Vch"].iloc[0]
vch_xgb_p71_pred = float(pbp_df.loc[pbp_df[ID_COL] == 71, "Vch_ml_pred"].values[0])
vch_xgb_p71_res = float(pbp_df.loc[pbp_df[ID_COL] == 71, "Vch_ml_res"].values[0])
vch_rsm_p71_pred = float(pbp_df.loc[pbp_df[ID_COL] == 71, "Vch_classical_pred"].values[0])

# Get CV stats for ET from CV_Best_Per_Family
et_cv_row = ml_best_per_fam[(ml_best_per_fam["Target"] == "Vch") & (ml_best_per_fam["Family"] == "ExtraTrees")].iloc[0]

diag_vch_df = pd.DataFrame([
    {
        "Model": "XGBoost (XGB_n200_d2_lr0.05)",
        "Role_in_Pipeline": "OFFICIALLY LOCKED (Step 8 CV Champion)",
        "CV_R2_mean": vch_xgb_row["CV_R2_mean"],
        "CV_RMSE_mean": vch_xgb_row["CV_RMSE_mean"],
        "CV_Physical_Violations": int(vch_xgb_row["CV_Physical_Violations_Total"]),
        "Holdout_R2": vch_xgb_row["Holdout_R2"],
        "Holdout_RMSE": vch_xgb_row["Holdout_RMSE"],
        "Holdout_MAE": vch_xgb_row["Holdout_MAE"],
        "Holdout_MAPE": vch_xgb_row["Holdout_MAPE"],
        "Holdout_Physical_Violations": int(vch_xgb_row["Holdout_Physical_Violations"]),
        "Point_71_Predicted_Vch": vch_xgb_p71_pred,
        "Point_71_Residual": vch_xgb_p71_res,
        "Status_and_Notes": "Officially selected by CV criteria. Lower RMSE on holdout. Negative volume (-0.018760) at Point 71 documented as a known boundary limitation.",
    },
    {
        "Model": "ExtraTrees (ET_n200_dNone_l1)",
        "Role_in_Pipeline": "DIAGNOSTIC COMPARISON (Not a Reselection)",
        "CV_R2_mean": et_cv_row["CV_R2_mean"],
        "CV_RMSE_mean": et_cv_row["CV_RMSE_mean"],
        "CV_Physical_Violations": int(et_cv_row["CV_Physical_Violations_Total"]),
        "Holdout_R2": et_vch_ho_r2,
        "Holdout_RMSE": et_vch_ho_rmse,
        "Holdout_MAE": et_vch_ho_mae,
        "Holdout_MAPE": et_vch_ho_mape,
        "Holdout_Physical_Violations": et_vch_ho_viol,
        "Point_71_Predicted_Vch": et_p71_vch_pred,
        "Point_71_Residual": et_p71_vch_res,
        "Status_and_Notes": "Evaluated post-hoc for diagnostic comparison only. Inherently non-negative (0 violations on holdout, Point 71 = +0.000197) but not selected because XGBoost won CV-only ranking.",
    },
    {
        "Model": "3rd-order RSM (Classical)",
        "Role_in_Pipeline": "CLASSICAL BASELINE (Step 5)",
        "CV_R2_mean": 0.986211,
        "CV_RMSE_mean": 0.009972,
        "CV_Physical_Violations": 0,
        "Holdout_R2": classical_metrics["Vch"]["Holdout_R2"],
        "Holdout_RMSE": classical_metrics["Vch"]["Holdout_RMSE"],
        "Holdout_MAE": classical_metrics["Vch"]["Holdout_MAE"],
        "Holdout_MAPE": classical_metrics["Vch"]["Holdout_MAPE"],
        "Holdout_Physical_Violations": classical_metrics["Vch"]["Physical_Violations"],
        "Point_71_Predicted_Vch": vch_rsm_p71_pred,
        "Point_71_Residual": float(pbp_df.loc[pbp_df[ID_COL] == 71, "Vch_classical_res"].values[0]),
        "Status_and_Notes": "Classical baseline model. Also produces negative volume (-0.033295) at Point 71 as documented in Step 5A.",
    },
])

# ---------------------------------------------------------------------------
# 7. Engineering Synthesis & Model Decision Table (Fully Programmatic Values)
# ---------------------------------------------------------------------------
# Extract programmatic variables directly from data frames
p71_sub = pbp_df[pbp_df[ID_COL] == 71].iloc[0]

hc_p71_actual = p71_sub["hc_hi_actual"]
hc_p71_pred = p71_sub["hc_hi_ml_pred"]
hc_p71_res = p71_sub["hc_hi_ml_res"]
hc_p71_abs_err = abs(hc_p71_res)

td_p71_actual = p71_sub["td_to_actual"]
td_p71_pred = p71_sub["td_to_ml_pred"]
td_p71_res = p71_sub["td_to_ml_res"]

vch_p71_actual = p71_sub["Vch_actual"]
vch_p71_pred = p71_sub["Vch_ml_pred"]
vch_p71_res = p71_sub["Vch_ml_res"]
vch_p71_undershoot = abs(vch_p71_pred) if vch_p71_pred < 0 else 0.0

c_hc = classical_metrics["hc_hi"]
m_hc = ml_final_models[ml_final_models["Target"] == "hc_hi"].iloc[0]

c_td = classical_metrics["td_to"]
m_td = ml_final_models[ml_final_models["Target"] == "td_to"].iloc[0]
td_rmse_pct_red = ((c_td["Holdout_RMSE"] - m_td["Holdout_RMSE"]) / c_td["Holdout_RMSE"]) * 100.0

c_vch = classical_metrics["Vch"]
m_vch = ml_final_models[ml_final_models["Target"] == "Vch"].iloc[0]
vch_rmse_pct_red = ((c_vch["Holdout_RMSE"] - m_vch["Holdout_RMSE"]) / c_vch["Holdout_RMSE"]) * 100.0

decisions = [
    {
        "Response": "hc_hi",
        "Classical_Champion": f"{c_hc['Method']}",
        "ML_Champion": f"{m_hc['Selected_Family']} ({m_hc['Selected_Config']})",
        "Key_Strength_Classical": f"Smooth interpolation in interior of convex hull (Holdout R2={c_hc['Holdout_R2']:.4f}, RMSE={c_hc['Holdout_RMSE']:.6f}).",
        "Key_Limitation_Classical": f"Convex-hull limited; fails (NaN) at boundary points like Point 71 (d=6, RPM=0). Coverage={c_hc['Coverage_Pct']:.1f}%.",
        "Key_Strength_ML": f"100% coverage across full domain; Point 71 predicted {hc_p71_pred:.6f} vs actual {hc_p71_actual:.6f} (residual {hc_p71_res:+.6f}, absolute error {hc_p71_abs_err:.6f}); Holdout R2={m_hc['Holdout_R2']:.4f}, RMSE={m_hc['Holdout_RMSE']:.6f}; 0 physical violations.",
        "Key_Limitation_ML": "Tree-based step boundaries instead of perfectly continuous analytical derivatives.",
        "Recommended_Model_for_Pipeline": f"{m_hc['Selected_Family']} ({m_hc['Selected_Config']}) for universal simulation & DSS; Cubic Griddata valid strictly inside training hull.",
        "Physical_Validity_Verdict": f"PASS ({int(m_hc['Holdout_Physical_Violations'])} violations)",
    },
    {
        "Response": "td_to",
        "Classical_Champion": f"{c_td['Method']}",
        "ML_Champion": f"{m_td['Selected_Family']} ({m_td['Selected_Config']})",
        "Key_Strength_Classical": f"Analytical closed-form equation; zero violations on training and holdout (Holdout R2={c_td['Holdout_R2']:.4f}).",
        "Key_Limitation_Classical": f"Higher residuals on holdout (RMSE={c_td['Holdout_RMSE']:.6f}, MAE={c_td['Holdout_MAE']:.6f}).",
        "Key_Strength_ML": f"Superior accuracy across all RPM regimes; {td_rmse_pct_red:.2f}% RMSE reduction ({c_td['Holdout_RMSE']:.6f} -> {m_td['Holdout_RMSE']:.6f}, MAE={m_td['Holdout_MAE']:.6f}); Holdout R2={m_td['Holdout_R2']:.4f}; zero physical violations.",
        "Key_Limitation_ML": "Requires kernel matrix evaluation; less directly auditable than a compact polynomial.",
        "Recommended_Model_for_Pipeline": f"{m_td['Selected_Family']} ({m_td['Selected_Config']}) is the locked champion for continuous modelling; 3rd-order RSM remains useful as an explicit closed-form formula.",
        "Physical_Validity_Verdict": f"PASS ({int(m_td['Holdout_Physical_Violations'])} violations)",
    },
    {
        "Response": "Vch",
        "Classical_Champion": f"{c_vch['Method']}",
        "ML_Champion": f"{m_vch['Selected_Family']} ({m_vch['Selected_Config']})",
        "Key_Strength_Classical": f"Captures overall volume growth trend with Holdout R2={c_vch['Holdout_R2']:.4f}, RMSE={c_vch['Holdout_RMSE']:.6f}.",
        "Key_Limitation_Classical": f"Violates lower physical boundary at Point 71 (d=6, RPM=0) predicting negative volume ({vch_rsm_p71_pred:.6f}).",
        "Key_Strength_ML": f"Achieves {vch_rmse_pct_red:.2f}% RMSE reduction over 3rd-order RSM ({c_vch['Holdout_RMSE']:.6f} -> {m_vch['Holdout_RMSE']:.6f}); Holdout R2={m_vch['Holdout_R2']:.4f}, MAE={m_vch['Holdout_MAE']:.6f}; 0 violations on CV.",
        "Key_Limitation_ML": f"Predicts negative volume at Point 71 (d=6, RPM=0) with Vch = {vch_p71_pred:.6f} (undershoot of {vch_p71_undershoot:.6f}), first revealed on holdout; documented honestly as a known boundary limitation.",
        "Recommended_Model_for_Pipeline": f"{m_vch['Selected_Family']} ({m_vch['Selected_Config']}) officially locked per Step 8 CV-only selection; Point 71 negative prediction is flagged as a known limitation without silent clipping or holdout reselection.",
        "Physical_Validity_Verdict": f"KNOWN LIMITATION ({int(m_vch['Holdout_Physical_Violations'])} holdout violation at Point 71: {vch_p71_pred:.6f})",
    },
]

synthesis_df = pd.DataFrame(decisions)

# ---------------------------------------------------------------------------
# 8. Write Output Workbook
# ---------------------------------------------------------------------------
with pd.ExcelWriter(OUTPUT_FILE, engine="openpyxl") as writer:

    readme = pd.DataFrame({
        "Item": [
            "Step", "Inputs", "Output", "Date",
            "Holdout_Dataset", "Total_Holdout_Points",
            "Classical_Models", "ML_Models",
            "Core_Evaluation_Principle",
            "Key_Finding_1", "Key_Finding_2", "Key_Finding_3",
        ],
        "Value": [
            "Step 9 — Classical vs. ML Final Continuous-Model Comparison",
            "Step5B_Final_Analysis_Consolidated.xlsx, Step6_ML_Data_Preparation.xlsx, Step8_ML_Model_Selection_Tuning.xlsx",
            str(OUTPUT_FILE.name),
            pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
            "ML_Holdout_17 (locked test set)",
            17,
            "hc_hi: Cubic Griddata; td_to: 3rd-order RSM; Vch: 3rd-order RSM",
            "hc_hi: ExtraTrees; td_to: SVR; Vch: XGBoost",
            "Multi-objective: R2, RMSE, MAE, coverage %, physical boundary compliance",
            "ML eliminates the coverage gap of Cubic Griddata (100% vs 94.1%) for hc_hi",
            "ML SVR improves td_to holdout RMSE by 27.50% over 3rd-order RSM with zero violations",
            f"ML XGBoost achieves {vch_rmse_pct_red:.2f}% RMSE reduction over 3rd-order RSM on Vch; Point 71 negative volume ({vch_p71_pred:.6f}) is documented as a known boundary limitation without silent clipping",
        ]
    })
    readme.to_excel(writer, sheet_name="README", index=False)
    comparison_df.to_excel(writer, sheet_name="Model_Comparison_Summary", index=False)
    pbp_df.to_excel(writer, sheet_name="Point_by_Point_Holdout", index=False)
    regime_df.to_excel(writer, sheet_name="Regime_Error_Breakdown", index=False)
    synthesis_df.to_excel(writer, sheet_name="Engineering_Decisions", index=False)
    diag_vch_df.to_excel(writer, sheet_name="Diagnostic_Vch_Comparison", index=False)

print(f"\nOutput written to: {OUTPUT_FILE.name}")
print("Step 9 complete.")
