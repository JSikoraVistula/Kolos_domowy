import json

import numpy as np


student_id = "72527"
seed_val = int(student_id[-4:])
sumID_0 = sum(int(digit) for digit in student_id[1:])
sumID_1 = sum(int(digit) for digit in student_id[:-1])
first_three_digits = int(student_id[:3])

np.random.seed(seed_val)
A = np.random.randint(0, sumID_0, size=(1 + sumID_0, 2 + sumID_1))
A_sum = int(A.sum())
A_shape = A.shape
A_1 = A[0].copy()

print("A_sum:", A_sum)
print("A_shape:", A_shape)
print("A_1:", A_1)

np.random.seed(seed_val)
B = np.random.randint(0, sumID_1, size=(1 + sumID_1, 1 + sumID_0))
B_sum = int(B.sum())
B_shape = B.shape
B_1 = B[0].copy()

print("B_sum:", B_sum)
print("B_shape:", B_shape)
print("B_1:", B_1)

a = A[:3, :3].copy()
det_a = float(np.linalg.det(a))
a_sum = int(a.sum())

print("det_a:", det_a)
print("a_sum:", a_sum)

a[-1, -1] = first_three_digits
a_sum_new = int(a.sum())

print("a:", a)
print("a_sum_new:", a_sum_new)

output = {
    "seed_val": int(seed_val),
    "sumID_0": int(sumID_0),
    "sumID_1": int(sumID_1),
    "A_shape": list(A_shape),
    "A_sum": A_sum,
    "A_1": A_1.tolist(),
    "B_shape": list(B_shape),
    "B_sum": B_sum,
    "B_1": B_1.tolist(),
    "det_a": det_a,
    "a_sum": a_sum,
    "a": a.tolist(),
    "a_sum_new": a_sum_new,
}

with open("task_1.json", "w", encoding="utf-8") as f:
    json.dump(output, f, ensure_ascii=False, indent=2)
