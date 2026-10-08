"""
Step 9 — Classical vs. ML Final Continuous-Model Comparison
Input : Step5B_Final_Analysis_Consolidated.xlsx, Step6_ML_Data_Preparation.xlsx (ML_Train_315), Step8_ML_Model_Selection_Tuning.xlsx
Output: Step9_Classical_vs_ML_Comparison.xlsx
"""

from pathlib import Path
import warnings
import numpy as np
import pandas as pd
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



# 1. Load Data

# Classical holdout evaluation
classical_ho = pd.read_excel(INPUT_STEP5B, sheet_name="Holdout_Model_Check")

holdout = pd.read_excel(INPUT_STEP6, sheet_name="ML_Holdout_17")

# ML tuned results from Step 8
ml_final_models = pd.read_excel(INPUT_STEP8, sheet_name="Final_Selected_Models")
ml_ho_preds = pd.read_excel(INPUT_STEP8, sheet_name="Holdout_Predictions")

print("Step 5B, Step 6, and Step 8 workbooks successfully loaded.")


# 2. Extract Classical Metrics on 17 Holdout Points

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


# 3. Build Comparative Summary (Classical vs ML) - Exactly 1 locked model per target

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


# 4. Side-by-Side Holdout Comparison Table

pbp_df = ml_ho_preds[[ID_COL, "d", "Rpm"]].copy()

for target in TARGETS:
    sub_c = classical_ho[classical_ho["Target"] == target].set_index(ID_COL)
    pbp_df[f"{target}_actual"] = sub_c.loc[pbp_df[ID_COL], "Actual"].values
    pbp_df[f"{target}_classical_pred"] = sub_c.loc[pbp_df[ID_COL], "Predicted"].values
    pbp_df[f"{target}_classical_res"] = sub_c.loc[pbp_df[ID_COL], "Residual"].values
    pbp_df[f"{target}_ml_pred"] = ml_ho_preds[f"{target}_pred"].values
    pbp_df[f"{target}_ml_res"] = ml_ho_preds[f"{target}_residual"].values


# 5. Error Breakdown by RPM Regime (Low: 0-80, Med: 100-180, High: 200-260)

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


# 6. Write Output Workbook

with pd.ExcelWriter(OUTPUT_FILE, engine="openpyxl") as writer:

    hc_comp = comparison_df[comparison_df["Target"] == "hc_hi"].iloc[0]
    td_comp = comparison_df[comparison_df["Target"] == "td_to"].iloc[0]
    vch_comp = comparison_df[comparison_df["Target"] == "Vch"].iloc[0]

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
            f"hc_hi: {hc_comp['ML_Selected_Model']}; td_to: {td_comp['ML_Selected_Model']}; Vch: {vch_comp['ML_Selected_Model']}",
            "Multi-objective: R2, RMSE, MAE, coverage %, physical boundary compliance",
            f"ML eliminates coverage gap of Cubic Griddata (100% vs {hc_comp['Classical_Coverage_Pct']:.1f}%) on hc_hi",
            f"ML achieves {td_comp['RMSE_Pct_Improvement']:.2f}% RMSE reduction over 3rd-order RSM on td_to with {int(td_comp['ML_Holdout_Physical_Violations'])} violations",
            f"ML achieves {vch_comp['RMSE_Pct_Improvement']:.2f}% RMSE reduction over 3rd-order RSM on Vch with {int(vch_comp['ML_Holdout_Physical_Violations'])} violations",
        ]
    })
    readme.to_excel(writer, sheet_name="README", index=False)
    comparison_df.to_excel(writer, sheet_name="Model_Comparison_Summary", index=False)
    pbp_df.to_excel(writer, sheet_name="Point_by_Point_Holdout", index=False)
    regime_df.to_excel(writer, sheet_name="Regime_Error_Breakdown", index=False)

print(f"\nOutput written to: {OUTPUT_FILE.name}")
print("Step 9 complete.")
