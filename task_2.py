import json

import numpy as np


student_id = "72527"
seed_val = int(student_id[-4:])
sumID_0 = sum(int(digit) for digit in student_id[1:])
sumID_1 = sum(int(digit) for digit in student_id[:-1])

np.random.seed(seed_val)
A = np.random.random((3, 3))
B = np.random.randint(min(sumID_0, sumID_1), max(sumID_0, sumID_1) + 1, size=(3, 3))
A_plus_B = A + B

print("A:")
print(A)
print("B:")
print(B)

A_shape = A.shape
A_min = float(A.min())
A_mean = float(A.mean())
A_med = float(np.median(A))
A_max = float(A.max())
A_sum = float(A.sum())

print("A shape:", A_shape)
print("A minimum:", A_min)
print("A mean:", A_mean)
print("A median:", A_med)
print("A maximum:", A_max)
print("A sum:", A_sum)

B_shape = A_plus_B.shape
B_min = float(A_plus_B.min())
B_mean = float(A_plus_B.mean())
B_med = float(np.median(A_plus_B))
B_max = float(A_plus_B.max())
B_sum = float(A_plus_B.sum())

print("A+B shape:", B_shape)
print("A+B minimum:", B_min)
print("A+B mean:", B_mean)
print("A+B median:", B_med)
print("A+B maximum:", B_max)
print("A+B sum:", B_sum)

output = {
    "seed": int(seed_val),
    "sumID_0": int(sumID_0),
    "sumID_1": int(sumID_1),
    "A": A.tolist(),
    "B": B.tolist(),
    "A_plus_B": A_plus_B.tolist(),
    "A_shape": list(A_shape),
    "A_min": A_min,
    "A_mean": A_mean,
    "A_med": A_med,
    "A_max": A_max,
    "A_sum": A_sum,
    "B_shape": list(B_shape),
    "B_min": B_min,
    "B_mean": B_mean,
    "B_med": B_med,
    "B_max": B_max,
    "B_sum": B_sum,
}

with open("task_2.json", "w", encoding="utf-8") as f:
    json.dump(output, f, ensure_ascii=False, indent=2)
