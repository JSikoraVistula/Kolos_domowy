import calendar
import json

import pandas as pd


student_id = 72527
mod_num = student_id % 12
month_num = mod_num + 1
month_name = calendar.month_name[month_num]

weather = pd.read_csv("seattle-weather.csv")
weather["date"] = pd.to_datetime(weather["date"])

selected_month = weather[
    (weather["date"].dt.year == 2013) & (weather["date"].dt.month == month_num)
]
rainy_days = selected_month[selected_month["precipitation"] > 0]["precipitation"]

rainy_days_num = int(rainy_days.count())
rainy_days_min = float(rainy_days.min())
rainy_days_max = float(rainy_days.max())
rainy_days_mean = float(rainy_days.mean())
rainy_days_sum = float(rainy_days.sum())

print("month_num:", month_num)
print("month_name:", month_name)
print("rainy_days_num:", rainy_days_num)
print("rainy_days_min:", rainy_days_min)
print("rainy_days_max:", rainy_days_max)
print("rainy_days_mean:", rainy_days_mean)
print("rainy_days_sum:", rainy_days_sum)

output = {
    "month_num": month_num,
    "month_name": month_name,
    "rainy_days_num": rainy_days_num,
    "rainy_days_min": rainy_days_min,
    "rainy_days_max": rainy_days_max,
    "rainy_days_mean": rainy_days_mean,
    "rainy_days_sum": rainy_days_sum,
}

with open("task_4.json", "w", encoding="utf-8") as f:
    json.dump(output, f, ensure_ascii=False, indent=2)
