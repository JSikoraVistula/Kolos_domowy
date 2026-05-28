import json
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import numpy as np
import tensorflow as tf
from sklearn.datasets import load_iris


seed_val = 2527


def build_perceptron(name):
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=(2,), name="input"),
            tf.keras.layers.Dense(1, activation="sigmoid", name="output"),
        ],
        name=name,
    )
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.1),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    return model


# Define the XOR input and output data.
X_xor = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.float32)
y_xor = np.array([[0], [1], [1], [0]], dtype=np.float32)

iris = load_iris()
iris_mask = iris.target < 2
X_iris = iris.data[iris_mask][:, [2, 3]].astype(np.float32)
y_iris = iris.target[iris_mask].reshape(-1, 1).astype(np.float32)

tf.keras.utils.set_random_seed(seed_val)
perceptron_xor = build_perceptron("perceptron_xor")
perceptron_xor.fit(X_xor, y_xor, epochs=500, verbose=0)
accuracy_xor = float(perceptron_xor.evaluate(X_xor, y_xor, verbose=0)[1])

tf.keras.utils.set_random_seed(seed_val)
perceptron_iris = build_perceptron("perceptron_iris")
perceptron_iris.fit(X_iris, y_iris, epochs=500, verbose=0)
accuracy_iris = float(perceptron_iris.evaluate(X_iris, y_iris, verbose=0)[1])

print("p_xor_layers:", len(perceptron_xor.layers))
print("p_iris_layers:", len(perceptron_iris.layers))
print("accuracy_xor:", accuracy_xor)
print("accuracy_iris:", accuracy_iris)
print("p_xor_config:", perceptron_xor.get_config())

output = {
    "p_xor_layers": len(perceptron_xor.layers),
    "p_iris_layers": len(perceptron_iris.layers),
    "accuracy_xor": accuracy_xor,
    "accuracy_iris": accuracy_iris,
    "p_xor_config": perceptron_xor.get_config(),
}

with open("task_5.json", "w", encoding="utf-8") as f:
    json.dump(output, f, ensure_ascii=False, indent=2)
