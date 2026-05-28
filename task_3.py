import json

import numpy as np
import pandas as pd


df = pd.DataFrame(
    {
        "region": ["North", "South", "North", "East", "South", "East", "North"],
        "revenue": [1200, 850, 1540, 920, 1100, 780, 660],
        "units": [12, 9, 15, 8, 11, 7, 6],
    }
)

df_meteo = pd.DataFrame(
    {
        "city": ["Warsaw", None, "Kraków", None, "Gdańsk"],
        "temp": [12.0, np.nan, 15.5, np.nan, 11.0],
        "wind": [np.nan, np.nan, 8.0, np.nan, 5.5],
    }
)

revenue_min = int(df["revenue"].min())
revenue_mean = float(df["revenue"].mean())
revenue_median = float(df["revenue"].median())
revenue_max = int(df["revenue"].max())
revenue_sum = int(df["revenue"].sum())

print("revenue_min:", revenue_min)
print("revenue_mean:", revenue_mean)
print("revenue_median:", revenue_median)
print("revenue_max:", revenue_max)
print("revenue_sum:", revenue_sum)

summary = (
    df.groupby("region")
    .agg(
        total_revenue=("revenue", "sum"),
        average_units=("units", "mean"),
        transactions=("region", "count"),
    )
    .sort_values("total_revenue", ascending=False)
)

tot_revenue = summary["total_revenue"]
avg_units = summary["average_units"]
transactions = summary["transactions"]

print("tot_revenue:")
print(tot_revenue)
print("avg_units:")
print(avg_units)
print("transactions:")
print(transactions)

filtered_df = df[(df["revenue"] > 1000) & (df["region"] == "South")]
filtered_val = filtered_df.values.tolist()

print("filtered_df:")
print(filtered_df)
print("filtered_val:", filtered_val)

na_sum = int(df_meteo.isna().sum().sum())
print("na_sum:", na_sum)

output = {
    "revenue_min": revenue_min,
    "revenue_mean": revenue_mean,
    "revenue_median": revenue_median,
    "revenue_max": revenue_max,
    "revenue_sum": revenue_sum,
    "tot_revenue": tot_revenue.astype(int).tolist(),
    "avg_units": avg_units.astype(float).tolist(),
    "transactions": transactions.astype(int).tolist(),
    "filtered_val": filtered_val,
    "na_sum": na_sum,
}

with open("task_3.json", "w", encoding="utf-8") as f:
    json.dump(output, f, ensure_ascii=False, indent=2)
