from pathlib import Path
import numpy as np
import pandas as pd
from scipy.interpolate import griddata
from scipy.spatial import Delaunay

BASE_DIR = Path(__file__).resolve().parent
STEP1_FILE = BASE_DIR / "Step1_Stratified_Split.xlsx"
STEP5_FILE = BASE_DIR / "Step5_Final_Experimental_Holdout_Validation.xlsx"
OUTPUT_FILE = BASE_DIR / "Step5A_Holdout_Audit_CrossResponse.xlsx"

TARGETS = ["hc_hi", "td_to", "Vch"]


def classify_physical(target, value):
    if not np.isfinite(value):
        return "INVALID_NONFINITE"
    if target in {"hc_hi", "Vch"}:
        if value < 0 or value > 1:
            return "PHYSICAL_VIOLATION"
        return "VALID"
    if target == "td_to":
        if value <= 0:
            return "PHYSICAL_VIOLATION"
        return "VALID"
    return "VALID"


def hull_membership(train, test):
    points = train[["d", "Rpm"]].to_numpy(float)
    queries = test[["d", "Rpm"]].to_numpy(float)

    try:
        tri = Delaunay(points)
        simplex = tri.find_simplex(queries)
        return simplex >= 0
    except Exception:
        return np.full(len(test), np.nan)


def correlation_table(df):
    rows = []

    pairs = [
        ("hc_hi", "td_to"),
        ("hc_hi", "Vch"),
        ("td_to", "Vch"),
    ]

    for a, b in pairs:
        x = pd.to_numeric(df[a], errors="coerce")
        y = pd.to_numeric(df[b], errors="coerce")
        mask = x.notna() & y.notna()

        if mask.sum() >= 2:
            pearson = x[mask].corr(y[mask], method="pearson")
            spearman = x[mask].corr(y[mask], method="spearman")
        else:
            pearson = np.nan
            spearman = np.nan

        rows.append({
            "Response_A": a,
            "Response_B": b,
            "N": int(mask.sum()),
            "Pearson_r": pearson,
            "Spearman_rho": spearman,
        })

    return pd.DataFrame(rows)


def group_summary(df):
    rows = []

    for d, group in df.groupby("d"):
        for target in TARGETS:
            rows.append({
                "Group_Type": "Diameter",
                "Group_Value": d,
                "Target": target,
                "N": len(group),
                "Mean": group[target].mean(),
                "Std": group[target].std(ddof=1),
                "Min": group[target].min(),
                "Max": group[target].max(),
            })

    for rpm, group in df.groupby("Rpm"):
        for target in TARGETS:
            rows.append({
                "Group_Type": "RPM",
                "Group_Value": rpm,
                "Target": target,
                "N": len(group),
                "Mean": group[target].mean(),
                "Std": group[target].std(ddof=1),
                "Min": group[target].min(),
                "Max": group[target].max(),
            })

    return pd.DataFrame(rows)


def main():
    print("=" * 78)
    print("STEP 5A - HOLDOUT AUDIT AND CROSS-RESPONSE CHARACTERIZATION")
    print("=" * 78)

    if not STEP1_FILE.exists():
        raise FileNotFoundError(STEP1_FILE)

    if not STEP5_FILE.exists():
        raise FileNotFoundError(STEP5_FILE)

    train = pd.read_excel(
        STEP1_FILE,
        sheet_name="Train_Anchors"
    )

    test = pd.read_excel(
        STEP1_FILE,
        sheet_name="Test_Holdout"
    )

    holdout_predictions = pd.read_excel(
        STEP5_FILE,
        sheet_name="Holdout_Predictions"
    )

    holdout_summary = pd.read_excel(
        STEP5_FILE,
        sheet_name="Holdout_Summary"
    )

    combined = pd.concat(
        [
            train[["Point_ID", "d", "Rpm"] + TARGETS],
            test[["Point_ID", "d", "Rpm"] + TARGETS],
        ],
        ignore_index=True,
    ).drop_duplicates("Point_ID")

    # ------------------------------------------------------------
    # 1. Exact holdout problem audit
    # ------------------------------------------------------------

    test_audit = test[
        ["Point_ID", "d", "Rpm"] + TARGETS
    ].copy()

    test_audit["Hull_Inside_Train"] = hull_membership(train, test)

    diagnostic_rows = []

    for target in TARGETS:
        pred = holdout_predictions[
            holdout_predictions["Target"] == target
        ].copy()

        merged = test_audit.merge(
            pred[
                [
                    "Point_ID",
                    "Predicted",
                    "Residual",
                    "Prediction_Valid",
                ]
            ],
            on="Point_ID",
            how="left",
        )

        merged["Target"] = target
        merged["Physical_Status"] = merged.apply(
            lambda r: classify_physical(
                target,
                r["Predicted"]
            ),
            axis=1,
        )

        # Diagnostics for selected model only.
        for _, row in merged.iterrows():
            diagnostic_rows.append({
                "Target": target,
                "Point_ID": row["Point_ID"],
                "d": row["d"],
                "Rpm": row["Rpm"],
                "Actual": row[target],
                "Predicted": row["Predicted"],
                "Residual": row["Residual"],
                "Prediction_Valid": row["Prediction_Valid"],
                "Physical_Status": row["Physical_Status"],
                "Hull_Inside_Train": row["Hull_Inside_Train"],
                "Absolute_Error": (
                    abs(row["Residual"])
                    if np.isfinite(row["Residual"])
                    else np.nan
                ),
            })

    diagnostic_df = pd.DataFrame(diagnostic_rows)

    problem_df = diagnostic_df[
        (diagnostic_df["Prediction_Valid"] == False) |
        (diagnostic_df["Physical_Status"] == "PHYSICAL_VIOLATION")
    ].copy()

    # ------------------------------------------------------------
    # 2. Diagnostic-only alternative checks.
    # These DO NOT change model selection.
    # ------------------------------------------------------------

    alternative_rows = []

    for target in TARGETS:
        selected_method = holdout_summary.loc[
            holdout_summary["Target"] == target,
            "Selected_Method"
        ].iloc[0]

        for _, row in test.iterrows():
            d = float(row["d"])
            rpm = float(row["Rpm"])

            # Linear/cubic Griddata diagnostics
            points = train[["d", "Rpm"]].to_numpy(float)
            values = train[target].to_numpy(float)

            q = np.array([[d, rpm]])

            linear_pred = float(
                griddata(
                    points,
                    values,
                    q,
                    method="linear"
                )[0]
            )

            cubic_pred = float(
                griddata(
                    points,
                    values,
                    q,
                    method="cubic"
                )[0]
            )

            alternative_rows.append({
                "Target": target,
                "Point_ID": row["Point_ID"],
                "d": d,
                "Rpm": rpm,
                "Selected_Method": selected_method,
                "Selected_Actual": row[target],
                "Linear_Griddata_Diagnostic": linear_pred,
                "Cubic_Griddata_Diagnostic": cubic_pred,
            })

    alternatives_df = pd.DataFrame(alternative_rows)

    # ------------------------------------------------------------
    # 3. Response relationships over all 84 real experiments
    # ------------------------------------------------------------

    correlations_df = correlation_table(combined)

    group_df = group_summary(combined)

    target_summary = []

    for target in TARGETS:
        s = pd.to_numeric(combined[target], errors="coerce")

        target_summary.append({
            "Target": target,
            "N": int(s.notna().sum()),
            "Mean": s.mean(),
            "Std": s.std(ddof=1),
            "Min": s.min(),
            "Max": s.max(),
            "Zero_Count": int((s == 0).sum()),
        })

    target_summary_df = pd.DataFrame(target_summary)

    # ------------------------------------------------------------
    # 4. Methodological decision log
    # ------------------------------------------------------------

    decisions_df = pd.DataFrame({
        "Decision": [
            "Step 5 holdout remains independent",
            "Holdout results are not used to reselect models",
            "hc_hi one missing prediction requires audit",
            "Vch physical violation requires audit",
            "Cross-response analysis uses all 84 real experiments",
            "Synthetic data used in characterization",
            "Diameter interpolation",
            "RPM extrapolation status",
        ],
        "Status": [
            "LOCKED",
            "LOCKED",
            "REVIEW REQUIRED",
            "REVIEW REQUIRED",
            "LOCKED",
            "NO",
            "NONE — only measured diameters retained",
            "Separate applicability issue",
        ]
    })

    with pd.ExcelWriter(
        OUTPUT_FILE,
        engine="openpyxl"
    ) as writer:
        holdout_summary.to_excel(
            writer,
            sheet_name="Holdout_Summary",
            index=False
        )

        diagnostic_df.to_excel(
            writer,
            sheet_name="Holdout_Diagnostic",
            index=False
        )

        problem_df.to_excel(
            writer,
            sheet_name="Problem_Points",
            index=False
        )

        alternatives_df.to_excel(
            writer,
            sheet_name="Diagnostic_Alternatives",
            index=False
        )

        target_summary_df.to_excel(
            writer,
            sheet_name="Response_Summary",
            index=False
        )

        correlations_df.to_excel(
            writer,
            sheet_name="Response_Correlations",
            index=False
        )

        group_df.to_excel(
            writer,
            sheet_name="Diameter_RPM_Summary",
            index=False
        )

        decisions_df.to_excel(
            writer,
            sheet_name="Decision_Log",
            index=False
        )

    print("-" * 78)
    print("STEP 5 HOLDOUT SUMMARY")
    print("-" * 78)
    print(
        holdout_summary.to_string(index=False)
    )

    print("-" * 78)
    print("PROBLEMATIC HOLDOUT PREDICTIONS")
    print("-" * 78)

    if problem_df.empty:
        print("None")
    else:
        print(
            problem_df.to_string(index=False)
        )

    print("-" * 78)
    print("RESPONSE CORRELATIONS — 84 REAL EXPERIMENTS")
    print("-" * 78)
    print(
        correlations_df.to_string(index=False)
    )

    print("-" * 78)
    print("COMPLETED")
    print("-" * 78)
    print("This script does not reselect any model.")
    print("It identifies the exact holdout points requiring physical/coverage audit.")
    print("It characterizes relationships among hc_hi, td_to and Vch.")
    print("Next step depends on the exact problematic holdout points.")


if __name__ == "__main__":
    main()
