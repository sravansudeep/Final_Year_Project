import numpy as np
import pandas as pd

from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error


file_path = "edited_excel.xlsx"
sheet_name = "Selected_columns"

df = pd.read_excel(file_path, sheet_name=sheet_name)

df = df[["d", "rpm", "Vmin"]].dropna()

X = df[["d", "rpm"]].values.astype(float)
y = df["Vmin"].values.astype(float)

print("=" * 70)
print("THIRD-ORDER RSM — FULL DATA")
print("=" * 70)

print("Number of observations:", len(y))

poly = PolynomialFeatures(
    degree=3,
    include_bias=True
)

X_poly = poly.fit_transform(X)

model = LinearRegression()
model.fit(X_poly, y)

preds = model.predict(X_poly)

r2 = r2_score(y, preds)
rmse = np.sqrt(mean_squared_error(y, preds))
mae = mean_absolute_error(y, preds)
mse = mean_squared_error(y, preds)

nonzero = y != 0

mape = np.mean(
    np.abs(
        (y[nonzero] - preds[nonzero])
        / y[nonzero]
    )
) * 100

print()
print("R2   :", r2)
print("RMSE :", rmse)
print("MAE  :", mae)
print("MSE  :", mse)
print("MAPE :", mape)

print()
print("=" * 70)
print("COEFFICIENTS")
print("=" * 70)

names = poly.get_feature_names_out(["d", "rpm"])

for name, coef in zip(names, model.coef_):
    print(f"{name:15s}: {coef:.12g}")

print()
print("Intercept:", model.intercept_)

print()
print("=" * 70)
print("POLYNOMIAL EQUATION")
print("=" * 70)

for name, coef in zip(names, model.coef_):
    if name == "1":
        continue

    print(f"{coef:+.12g} * {name}")

print("=" * 70)