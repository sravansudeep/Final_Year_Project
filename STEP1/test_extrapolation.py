import numpy as np
import pandas as pd

from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression


file_path = "edited_excel.xlsx"
sheet_name = "Selected_columns"

df = pd.read_excel(
    file_path,
    sheet_name=sheet_name
)

df = df[["d", "rpm", "Vmin"]].dropna()

X = df[["d", "rpm"]].values.astype(float)
y = df["Vmin"].values.astype(float)


poly = PolynomialFeatures(
    degree=3,
    include_bias=True
)

X_poly = poly.fit_transform(X)

model = LinearRegression()
model.fit(X_poly, y)


extrapolation_points = np.array([
    [4, 280],
    [4, 300],
    [4, 320],
    [18, 280],
    [18, 300],
    [18, 320]
], dtype=float)


predictions = model.predict(
    poly.transform(extrapolation_points)
)


result = pd.DataFrame(
    extrapolation_points,
    columns=["d", "rpm"]
)

result["Vmin"] = predictions
result["Data_Type"] = "Extrapolated"
result["Source_Method"] = "Third-order RSM"
result["Validation_Status"] = "Provisional"


print("=" * 70)
print("EXTRAPOLATION TEST")
print("=" * 70)

print(result.to_string(index=False))

print()
print("=" * 70)
print("CHECKS")
print("=" * 70)

print(
    "Minimum predicted Vmin:",
    result["Vmin"].min()
)

print(
    "Maximum predicted Vmin:",
    result["Vmin"].max()
)

print(
    "Negative predictions:",
    (result["Vmin"] < 0).sum()
)

print(
    "NaN predictions:",
    result["Vmin"].isna().sum()
)


result.to_excel(
    "extrapolated_points.xlsx",
    index=False
)

print()
print("Saved: extrapolated_points.xlsx")