from pathlib import Path
import numpy as np
import pandas as pd
from scipy.interpolate import griddata

BASE_DIR = Path(__file__).resolve().parent
INPUT_FILE = BASE_DIR / "Step1_Stratified_Split.xlsx"
OUTPUT_FILE = BASE_DIR / "Step5_Final_Experimental_Holdout_Validation.xlsx"

TARGETS = ["hc_hi", "td_to", "Vch"]
SELECTED = {
    "hc_hi": "Cubic Griddata",
    "td_to": "3rd-order RSM",
    "Vch": "3rd-order RSM",
}

def terms(degree):
    return [(i, j) for i in range(degree + 1) for j in range(degree + 1 - i)]

def fit_rsm(df, target, degree):
    d = df["d"].to_numpy(float)
    r = df["Rpm"].to_numpy(float)
    y = df[target].to_numpy(float)
    dc, rc = d.mean(), r.mean()
    ds, rs = d.std() or 1.0, r.std() or 1.0
    x, z = (d - dc) / ds, (r - rc) / rs
    ts = terms(degree)
    X = np.column_stack([(x ** i) * (z ** j) for i, j in ts])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    return beta, ts, dc, rc, ds, rs

def predict_rsm(model, d, r):
    beta, ts, dc, rc, ds, rs = model
    x = (np.asarray(d, float) - dc) / ds
    z = (np.asarray(r, float) - rc) / rs
    X = np.column_stack([(x ** i) * (z ** j) for i, j in ts])
    return X @ beta

def predict(train, test, target, method):
    if method == "Cubic Griddata":
        return griddata(
            train[["d", "Rpm"]].to_numpy(float),
            train[target].to_numpy(float),
            test[["d", "Rpm"]].to_numpy(float),
            method="cubic",
        )
    degree = {"1st-order RSM": 1, "2nd-order RSM": 2, "3rd-order RSM": 3}[method]
    return predict_rsm(fit_rsm(train, target, degree), test["d"], test["Rpm"])

def metrics(actual, pred):
    a = np.asarray(actual, float)
    p = np.asarray(pred, float)
    m = np.isfinite(a) & np.isfinite(p)
    a, p = a[m], p[m]
    if len(a) == 0:
        return 0, np.nan, np.nan, np.nan, np.nan
    e = p - a
    mse = float(np.mean(e ** 2))
    ss = float(np.sum((a - a.mean()) ** 2))
    r2 = np.nan if ss == 0 else 1 - float(np.sum(e ** 2)) / ss
    nz = np.abs(a) > 1e-12
    mape = float(np.mean(np.abs((a[nz] - p[nz]) / a[nz])) * 100) if np.any(nz) else np.nan
    return len(a), r2, float(np.sqrt(mse)), float(np.mean(np.abs(e))), mape

def physical(target, pred):
    p = np.asarray(pred, float)
    if target in {"hc_hi", "Vch"}:
        bad = (~np.isfinite(p)) | (p < 0) | (p > 1)
    else:
        bad = (~np.isfinite(p)) | (p <= 0)
    return int(np.sum(bad)), float(np.mean(bad) * 100) if len(p) else np.nan

def main():
    train = pd.read_excel(INPUT_FILE, sheet_name="Train_Anchors")
    test = pd.read_excel(INPUT_FILE, sheet_name="Test_Holdout")

    if len(train) != 67 or len(test) != 17:
        raise ValueError(f"Expected 67 train and 17 holdout rows; got {len(train)} and {len(test)}")

    summary = []
    predictions = []

    for target in TARGETS:
        method = SELECTED[target]
        pred = predict(train, test, target, method)
        n, r2, rmse, mae, mape = metrics(test[target], pred)
        nbad, pctbad = physical(target, pred)

        summary.append({
            "Target": target,
            "Selected_Method": method,
            "N_Test_Points": len(test),
            "N_Valid_Predictions": n,
            "Coverage_percent": 100 * n / len(test),
            "Physical_Violations": nbad,
            "Physical_Violation_percent": pctbad,
            "R2": r2,
            "RMSE": rmse,
            "MAE": mae,
            "MAPE_percent": mape,
            "Status": "REVIEW BEFORE FINAL LOCK",
        })

        for i, (_, row) in enumerate(test.iterrows()):
            ok = np.isfinite(pred[i])
            predictions.append({
                "Target": target,
                "Method": method,
                "Point_ID": row["Point_ID"],
                "d": row["d"],
                "Rpm": row["Rpm"],
                "Actual": row[target],
                "Predicted": float(pred[i]) if ok else np.nan,
                "Residual": float(pred[i] - row[target]) if ok else np.nan,
                "Prediction_Valid": bool(ok),
            })

    protocol = pd.DataFrame({
        "Item": [
            "Train_Anchors", "Test_Holdout", "Synthetic data used",
            "Extrapolation used", "Test_Holdout used for model selection",
            "hc_hi model", "td_to model", "Vch model"
        ],
        "Value": [
            len(train), len(test), 0, "NO", "NO",
            SELECTED["hc_hi"], SELECTED["td_to"], SELECTED["Vch"]
        ]
    })

    with pd.ExcelWriter(OUTPUT_FILE, engine="openpyxl") as w:
        pd.DataFrame(summary).to_excel(w, sheet_name="Holdout_Summary", index=False)
        pd.DataFrame(predictions).to_excel(w, sheet_name="Holdout_Predictions", index=False)
        test[["Point_ID", "d", "Rpm"] + TARGETS].to_excel(w, sheet_name="Test_Holdout_Audit", index=False)
        protocol.to_excel(w, sheet_name="Protocol", index=False)

    print("=" * 78)
    print("STEP 5 - FINAL EXPERIMENTAL HOLDOUT VALIDATION")
    print("=" * 78)
    print(f"Input:  {INPUT_FILE}")
    print(f"Output: {OUTPUT_FILE}")
    print("-" * 78)
    print("Train_Anchors used : 67")
    print("Test_Holdout used  : 17")
    print("Synthetic data used: 0")
    print("Test_Holdout is used here for validation, not model selection.")
    print("-" * 78)
    print(pd.DataFrame(summary).to_string(index=False))
    print("-" * 78)
    print("COMPLETED")
    print("RPM extrapolation remains a separate applicability issue.")
    print("Do not use the holdout results to reselect the Step 3 model.")

if __name__ == "__main__":
    main()
