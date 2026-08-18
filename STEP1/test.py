import pandas as pd

df = pd.read_excel("in_range_expanded_dataset.xlsx")

check = df[
    df["rpm"].isin([0, 10, 20, 30, 40])
].sort_values(["rpm", "d"])

print(
    check[
        ["d", "rpm", "Vmin", "Data_Type"]
    ].to_string(index=False)
)