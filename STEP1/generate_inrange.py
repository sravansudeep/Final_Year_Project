import numpy as np
import pandas as pd

from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression


file_path = "edited_excel.xlsx"
sheet_name = "Selected_columns"

output_file = "in_range_expanded_dataset.xlsx"


df = pd.read_excel(
    file_path,
    sheet_name=sheet_name
)

df = df[["d", "rpm", "Vmin"]].dropna()

df["d"] = df["d"].astype(float)
df["rpm"] = df["rpm"].astype(float)
df["Vmin"] = df["Vmin"].astype(float)


X = df[["d", "rpm"]].values
y = df["Vmin"].values


print("=" * 70)
print("IN-RANGE DATASET EXPANSION")
print("=" * 70)

print("Original observations:", len(df))

print("Unique diameters:",
      sorted(df["d"].unique()))

print("Unique RPM values:",
      sorted(df["rpm"].unique()))


poly = PolynomialFeatures(
    degree=3,
    include_bias=True
)

X_poly = poly.fit_transform(X)

model = LinearRegression()
model.fit(X_poly, y)


diameters = np.arange(6, 17, 1)
rpms = np.arange(0, 261, 10)


grid = []

for d in diameters:
    for rpm in rpms:
        grid.append([d, rpm])


grid = np.array(grid, dtype=float)


grid_poly = poly.transform(grid)

predictions = model.predict(grid_poly)


expanded = pd.DataFrame(
    grid,
    columns=["d", "rpm"]
)

expanded["Vmin"] = predictions


original_locations = set(
    zip(
        df["d"],
        df["rpm"]
    )
)


expanded["Data_Type"] = expanded.apply(
    lambda row:
    "Original"
    if (row["d"], row["rpm"])
    in original_locations
    else "Interpolated",
    axis=1
)


expanded["Source_Method"] = expanded[
    "Data_Type"
].map({
    "Original": "Experimental",
    "Interpolated": "Third-order RSM"
})


for _, row in df.iterrows():

    mask = (
        (expanded["d"] == row["d"]) &
        (expanded["rpm"] == row["rpm"])
    )

    expanded.loc[
        mask,
        "Vmin"
    ] = row["Vmin"]


expanded["Validation_Status"] = expanded[
    "Data_Type"
].map({
    "Original": "Measured",
    "Interpolated": "In-range model estimate"
})


print()
print("=" * 70)
print("RESULT")
print("=" * 70)

print("Total grid locations:", len(expanded))

print(
    "Original points:",
    (expanded["Data_Type"] == "Original").sum()
)

print(
    "Interpolated points:",
    (expanded["Data_Type"] == "Interpolated").sum()
)


print()
print("Expected:")
print("Total: 297")
print("Original: 84")
print("Interpolated: 213")


print()
print("=" * 70)
print("DATA QUALITY CHECK")
print("=" * 70)

print(
    "Missing Vmin:",
    expanded["Vmin"].isna().sum()
)

print(
    "Duplicate locations:",
    expanded.duplicated(
        subset=["d", "rpm"]
    ).sum()
)

print(
    "Minimum Vmin:",
    expanded["Vmin"].min()
)

print(
    "Maximum Vmin:",
    expanded["Vmin"].max()
)


expanded.to_excel(
    output_file,
    index=False
)


print()
print("Saved to:", output_file)