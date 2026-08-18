import numpy as np
import pandas as pd
from scipy.interpolate import griddata


file_path = "edited_excel.xlsx"
sheet_name = "Selected_columns"

df = pd.read_excel(
    file_path,
    sheet_name=sheet_name
)

df = df[["d", "rpm", "Vmin"]].dropna()

X = df[["d", "rpm"]].values.astype(float)
y = df["Vmin"].values.astype(float)


diameters = np.arange(6, 17, 1)
rpms = np.arange(0, 261, 10)

grid = np.array(
    [
        [d, rpm]
        for d in diameters
        for rpm in rpms
    ],
    dtype=float
)


pred = griddata(
    X,
    y,
    grid,
    method="linear"
)


result = pd.DataFrame(
    grid,
    columns=["d", "rpm"]
)

result["Vmin"] = pred


original_locations = set(
    zip(
        df["d"],
        df["rpm"]
    )
)

result["Data_Type"] = result.apply(
    lambda row:
    "Original"
    if (row["d"], row["rpm"]) in original_locations
    else "Interpolated",
    axis=1
)


for _, row in df.iterrows():

    mask = (
        (result["d"] == row["d"]) &
        (result["rpm"] == row["rpm"])
    )

    result.loc[
        mask,
        "Vmin"
    ] = row["Vmin"]


print("=" * 70)
print("LINEAR GRIDDATA — IN-RANGE TEST")
print("=" * 70)

print("Total grid points:", len(result))

print(
    "Original points:",
    (result["Data_Type"] == "Original").sum()
)

print(
    "Interpolated points:",
    (result["Data_Type"] == "Interpolated").sum()
)

print(
    "NaN predictions:",
    result["Vmin"].isna().sum()
)

print(
    "Negative predictions:",
    (result["Vmin"] < 0).sum()
)

print(
    "Minimum Vmin:",
    result["Vmin"].min()
)

print(
    "Maximum Vmin:",
    result["Vmin"].max()
)


negative = result[result["Vmin"] < 0]

if len(negative) > 0:

    print()
    print("NEGATIVE VALUES")
    print(negative.to_string(index=False))


nan = result[result["Vmin"].isna()]

if len(nan) > 0:

    print()
    print("NaN VALUES")
    print(nan.to_string(index=False))


result.to_excel(
    "in_range_griddata_linear.xlsx",
    index=False
)

print()
print("Saved: in_range_griddata_linear.xlsx")