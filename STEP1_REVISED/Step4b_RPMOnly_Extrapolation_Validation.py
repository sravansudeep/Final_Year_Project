from pathlib import Path
import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
INPUT_FILE = BASE_DIR / "Step1_Stratified_Split.xlsx"
OUTPUT_FILE = BASE_DIR / "Step4_RPMOnly_Extrapolation_Validation.xlsx"

TARGETS = ["hc_hi", "td_to", "Vch"]
DIAMETERS = [6, 8, 10, 12, 14, 16]
ONE_STEP_CUTOFFS = [180, 200, 220, 240]
MULTI_STEP_CUTOFFS = [160, 180, 200]
FINAL_RPMS = [280, 300, 320]

DEGREES = {
    "1st-order RPM polynomial": 1,
    "2nd-order RPM polynomial": 2,
    "3rd-order RPM polynomial": 3,
}

def fit_poly(df, target, degree):
    x = df["Rpm"].to_numpy(float)
    y = df[target].to_numpy(float)
    c = float(np.mean(x))
    s = float(np.std(x)) or 1.0
    z = (x - c) / s
    coef = np.polyfit(z, y, degree)
    return coef, c, s

def predict(model, rpm):
    coef, c, s = model
    z = (np.asarray(rpm, float) - c) / s
    return np.polyval(coef, z)

def metrics(actual, pred):
    a = np.asarray(actual, float)
    p = np.asarray(pred, float)
    m = np.isfinite(a) & np.isfinite(p)
    a, p = a[m], p[m]
    if len(a) == 0:
        return dict(N_Valid=0, R2=np.nan, RMSE=np.nan, MAE=np.nan, MSE=np.nan, MAPE_percent=np.nan)
    e = p - a
    mse = float(np.mean(e**2))
    ss_tot = float(np.sum((a - np.mean(a))**2))
    r2 = np.nan if ss_tot == 0 else 1.0 - float(np.sum(e**2)) / ss_tot
    nz = np.abs(a) > 1e-12
    mape = float(np.mean(np.abs((a[nz] - p[nz]) / a[nz])) * 100) if np.any(nz) else np.nan
    return dict(
        N_Valid=len(a),
        R2=r2,
        RMSE=float(np.sqrt(mse)),
        MAE=float(np.mean(np.abs(e))),
        MSE=mse,
        MAPE_percent=mape,
    )

def physical_check(target, pred):
    p = np.asarray(pred, float)
    if target in {"hc_hi", "Vch"}:
        bad = (~np.isfinite(p)) | (p < 0) | (p > 1)
    elif target == "td_to":
        bad = (~np.isfinite(p)) | (p <= 0)
    else:
        bad = ~np.isfinite(p)
    return int(np.sum(bad)), float(np.mean(bad) * 100) if len(p) else np.nan

def load():
    train = pd.read_excel(INPUT_FILE, sheet_name="Train_Anchors")
    test = pd.read_excel(INPUT_FILE, sheet_name="Test_Holdout")
    need = ["Point_ID", "d", "Rpm"] + TARGETS
    missing = [c for c in need if c not in train.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")
    return train, test

def run_validation(train, target, multi=False):
    rows, preds = [], []
    cutoffs = MULTI_STEP_CUTOFFS if multi else ONE_STEP_CUTOFFS

    for d in DIAMETERS:
        dd = train[train["d"] == d].copy()
        for cutoff in cutoffs:
            if multi:
                fit = dd[dd["Rpm"] <= cutoff].copy()
                val = dd[dd["Rpm"] > cutoff].copy()
            else:
                fit = dd[dd["Rpm"] <= cutoff].copy()
                val = dd[dd["Rpm"] == cutoff + 20].copy()

            if val.empty:
                continue

            for method, degree in DEGREES.items():
                if len(fit) <= degree:
                    continue
                model = fit_poly(fit, target, degree)
                pred = predict(model, val["Rpm"])

                met = metrics(val[target], pred)
                nbad, pctbad = physical_check(target, pred)

                rows.append({
                    "Target": target,
                    "Method": method,
                    "d": d,
                    "Train_Max_RPM": cutoff,
                    "N_Train": len(fit),
                    "N_Validation": len(val),
                    **met,
                    "Physical_Violations": nbad,
                    "Physical_Violation_percent": pctbad,
                })

                for j, (_, r) in enumerate(val.iterrows()):
                    preds.append({
                        "Target": target,
                        "Method": method,
                        "d": d,
                        "Train_Max_RPM": cutoff,
                        "Validation_RPM": r["Rpm"],
                        "Point_ID": r["Point_ID"],
                        "Actual": r[target],
                        "Predicted": pred[j],
                        "Residual": pred[j] - r[target],
                    })
    return pd.DataFrame(rows), pd.DataFrame(preds)

def pool(preds, mode):
    rows = []
    for target in TARGETS:
        for method in DEGREES:
            p = preds[(preds.Target == target) & (preds.Method == method)]
            if p.empty:
                continue
            met = metrics(p.Actual, p.Predicted)
            nbad, pctbad = physical_check(target, p.Predicted)
            rows.append({
                "Validation_Mode": mode,
                "Target": target,
                "Method": method,
                "N_Validation_Points": len(p),
                **met,
                "Physical_Violations": nbad,
                "Physical_Violation_percent": pctbad,
            })
    return pd.DataFrame(rows)

def rank(one, multi):
    out = []
    for target in TARGETS:
        for method in DEGREES:
            o = one[(one.Target == target) & (one.Method == method)]
            m = multi[(multi.Target == target) & (multi.Method == method)]
            if o.empty or m.empty:
                continue
            mr2 = float(m.R2.iloc[0])
            mrmse = float(m.RMSE.iloc[0])
            om = float(o.RMSE.iloc[0])
            mphys = int(m.Physical_Violations.iloc[0])
            ophys = int(o.Physical_Violations.iloc[0])

            passes = np.isfinite(mr2) and mr2 > 0 and mphys == 0
            out.append({
                "Target": target,
                "Method": method,
                "OneStep_R2": float(o.R2.iloc[0]),
                "OneStep_RMSE": om,
                "OneStep_Physical_Violations": ophys,
                "MultiStep_R2": mr2,
                "MultiStep_RMSE": mrmse,
                "MultiStep_Physical_Violations": mphys,
                "Defensible_By_Check": passes,
            })

    df = pd.DataFrame(out)
    result = []
    for target, g in df.groupby("Target"):
        g = g.sort_values(
            ["Defensible_By_Check", "MultiStep_Physical_Violations", "MultiStep_RMSE", "OneStep_RMSE"],
            ascending=[False, True, True, True]
        ).reset_index(drop=True)
        g["Rank"] = np.arange(1, len(g) + 1)
        g["Status"] = np.where(
            g["Defensible_By_Check"],
            "CANDIDATE - PASSES PREDEFINED CHECKS",
            "NO MODEL PASSES ALL PREDEFINED CHECKS"
        )
        result.append(g)
    return pd.concat(result, ignore_index=True)

def final_predictions(train, ranking):
    rows = []
    for target in TARGETS:
        passed = ranking[(ranking.Target == target) & ranking.Defensible_By_Check]
        selected = passed.iloc[0] if not passed.empty else ranking[ranking.Target == target].iloc[0]
        method = selected.Method
        degree = DEGREES[method]
        status = (
            "PROVISIONAL RPM-ONLY EXTRAPOLATION"
            if not passed.empty
            else "NO VALIDATED RPM-EXTRAPOLATION MODEL; REFERENCE ONLY"
        )
        for d in DIAMETERS:
            dd = train[train.d == d].copy()
            if len(dd) <= degree:
                continue
            model = fit_poly(dd, target, degree)
            for rpm in FINAL_RPMS:
                val = float(predict(model, [rpm])[0])
                nbad, _ = physical_check(target, [val])
                rows.append({
                    "Target": target,
                    "Method": method,
                    "d": d,
                    "Rpm": rpm,
                    "Predicted": val,
                    "Physical_Violation": bool(nbad),
                    "Applicability": "RPM_OUTSIDE_EXPERIMENTAL_RANGE",
                    "Status": status,
                })
    return pd.DataFrame(rows)

def main():
    print("=" * 78)
    print("STEP 4 - REVISED RPM-ONLY EXTRAPOLATION VALIDATION")
    print("=" * 78)

    train, test = load()
    if len(train) != 67:
        raise ValueError(f"Expected 67 Train_Anchors rows, got {len(train)}")
    if len(test) != 17:
        raise ValueError(f"Expected 17 Test_Holdout rows, got {len(test)}")

    one_f, one_p = [], []
    multi_f, multi_p = [], []

    for t in TARGETS:
        f, p = run_validation(train, t, multi=False)
        one_f.append(f); one_p.append(p)
        f, p = run_validation(train, t, multi=True)
        multi_f.append(f); multi_p.append(p)

    one_f = pd.concat(one_f, ignore_index=True)
    one_p = pd.concat(one_p, ignore_index=True)
    multi_f = pd.concat(multi_f, ignore_index=True)
    multi_p = pd.concat(multi_p, ignore_index=True)

    one_s = pool(one_p, "One-Step Pooled")
    multi_s = pool(multi_p, "Multi-Step Pooled")
    ranking = rank(one_s, multi_s)
    final = final_predictions(train, ranking)

    protocol = pd.DataFrame({
        "Item": [
            "Train_Anchors", "Test_Holdout used", "Synthetic data",
            "Diameter domain", "One-step cutoffs", "Multi-step cutoffs",
            "Final RPMs", "Diameter extrapolation in this step", "Target columns"
        ],
        "Value": [
            len(train), 0, 0, str(DIAMETERS), str(ONE_STEP_CUTOFFS),
            str(MULTI_STEP_CUTOFFS), str(FINAL_RPMS), "NO", str(TARGETS)
        ]
    })

    notes = pd.DataFrame({
        "Method": list(DEGREES.keys()),
        "Degree": list(DEGREES.values()),
        "Description": [
            "RPM polynomial fitted separately at each measured diameter.",
            "RPM polynomial fitted separately at each measured diameter.",
            "RPM polynomial fitted separately at each measured diameter.",
        ]
    })

    with pd.ExcelWriter(OUTPUT_FILE, engine="openpyxl") as w:
        one_s.to_excel(w, sheet_name="OneStep_Summary", index=False)
        multi_s.to_excel(w, sheet_name="MultiStep_Summary", index=False)
        ranking.to_excel(w, sheet_name="Method_Ranking", index=False)
        one_f.to_excel(w, sheet_name="OneStep_ByDiameter", index=False)
        multi_f.to_excel(w, sheet_name="MultiStep_ByDiameter", index=False)
        one_p.to_excel(w, sheet_name="OneStep_Predictions", index=False)
        multi_p.to_excel(w, sheet_name="MultiStep_Predictions", index=False)
        final.to_excel(w, sheet_name="RPMOnly_280_320", index=False)
        protocol.to_excel(w, sheet_name="Protocol", index=False)
        notes.to_excel(w, sheet_name="Method_Notes", index=False)

    print("-" * 78)
    print("DATA PROTOCOL")
    print(f"Train_Anchors used : {len(train)}")
    print("Test_Holdout used  : 0")
    print("Synthetic data used: 0")
    print("-" * 78)
    print("ONE-STEP POOLED RESULTS")
    print(one_s.to_string(index=False))
    print("-" * 78)
    print("MULTI-STEP POOLED RESULTS")
    print(multi_s.to_string(index=False))
    print("-" * 78)
    print("RPM-ONLY METHOD RANKING")
    print(ranking.to_string(index=False))
    print("-" * 78)
    print("280-320 RPM AT MEASURED DIAMETERS")
    print(final.to_string(index=False))
    print("-" * 78)
    print("COMPLETED")
    print("No intermediate diameters were used.")
    print("No d=4 or d=18 extrapolation was performed.")
    print("Test_Holdout remains untouched.")
    print("Run review before any diameter extrapolation.")

if __name__ == "__main__":
    main()
