
import time
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from tensorflow.keras import Sequential
from tensorflow.keras.layers import (
    Input, Conv2D, MaxPooling2D,
    AveragePooling2D, BatchNormalization,
    GlobalAveragePooling2D, Dense
)
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split

# --------------------------------------------------
# 1. Load CIFAR-10 dataset
# --------------------------------------------------

np.random.seed(42)
tf.random.set_seed(42)

print("Loading CIFAR-10 dataset...")

(x_all, y_all), (x_test_all, y_test_all) = (
    tf.keras.datasets.cifar10.load_data()
)

y_all = y_all.flatten()
y_test_all = y_test_all.flatten()

# Select a small random subset for faster experiments
train_indices = np.random.choice(
    len(x_all), 5000, replace=False
)

test_indices = np.random.choice(
    len(x_test_all), 1000, replace=False
)

X = x_all[train_indices].astype("float32") / 255.0
y = y_all[train_indices]

X_test = x_test_all[test_indices].astype("float32") / 255.0
y_test = y_test_all[test_indices]

# Divide training data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# One-hot encode class labels
y_train = to_categorical(y_train, 10)
y_val = to_categorical(y_val, 10)
y_test_cat = to_categorical(y_test, 10)

print("Training images:", len(X_train))
print("Validation images:", len(X_val))
print("Testing images:", len(X_test))


# --------------------------------------------------
# 2. Define CNN model
# --------------------------------------------------

def create_cnn(config):

    model = Sequential()
    model.add(Input(shape=(32, 32, 3)))

    for i in range(config["layers"]):

        filters = config["filters"] * (2 ** i)

        model.add(
            Conv2D(
                filters,
                kernel_size=config["kernel"],
                padding="same",
                use_bias=not config["batch_norm"]
            )
        )

        if config["batch_norm"]:
            model.add(BatchNormalization())

        model.add(tf.keras.layers.Activation("relu"))

        if config["pooling"] == "max":
            model.add(MaxPooling2D(pool_size=(2, 2)))
        else:
            model.add(AveragePooling2D(pool_size=(2, 2)))

    model.add(GlobalAveragePooling2D())
    model.add(Dense(64, activation="relu"))
    model.add(Dense(10, activation="softmax"))

    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=config["lr"]
        ),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model


# --------------------------------------------------
# 3. Define 12 configurations
# --------------------------------------------------

configs = [
    {
        "name": "Baseline",
        "layers": 3, "kernel": 3, "filters": 16,
        "pooling": "max", "batch_norm": False,
        "lr": 0.001, "batch_size": 32
    },
    {
        "name": "2 Conv Layers",
        "layers": 2, "kernel": 3, "filters": 16,
        "pooling": "max", "batch_norm": False,
        "lr": 0.001, "batch_size": 32
    },
    {
        "name": "4 Conv Layers",
        "layers": 4, "kernel": 3, "filters": 16,
        "pooling": "max", "batch_norm": False,
        "lr": 0.001, "batch_size": 32
    },
    {
        "name": "5x5 Filter",
        "layers": 3, "kernel": 5, "filters": 16,
        "pooling": "max", "batch_norm": False,
        "lr": 0.001, "batch_size": 32
    },
    {
        "name": "32 Filters",
        "layers": 3, "kernel": 3, "filters": 32,
        "pooling": "max", "batch_norm": False,
        "lr": 0.001, "batch_size": 32
    },
    {
        "name": "8 Filters",
        "layers": 3, "kernel": 3, "filters": 8,
        "pooling": "max", "batch_norm": False,
        "lr": 0.001, "batch_size": 32
    },
    {
        "name": "Average Pooling",
        "layers": 3, "kernel": 3, "filters": 16,
        "pooling": "average", "batch_norm": False,
        "lr": 0.001, "batch_size": 32
    },
    {
        "name": "Batch Normalisation",
        "layers": 3, "kernel": 3, "filters": 16,
        "pooling": "max", "batch_norm": True,
        "lr": 0.001, "batch_size": 32
    },
    {
        "name": "Low Learning Rate",
        "layers": 3, "kernel": 3, "filters": 16,
        "pooling": "max", "batch_norm": False,
        "lr": 0.0005, "batch_size": 32
    },
    {
        "name": "Batch Size 64",
        "layers": 3, "kernel": 3, "filters": 16,
        "pooling": "max", "batch_norm": False,
        "lr": 0.001, "batch_size": 64
    },
    {
        "name": "Lightweight CNN",
        "layers": 2, "kernel": 3, "filters": 8,
        "pooling": "average", "batch_norm": False,
        "lr": 0.001, "batch_size": 64
    },
    {
        "name": "Deeper CNN",
        "layers": 4, "kernel": 3, "filters": 32,
        "pooling": "max", "batch_norm": True,
        "lr": 0.001, "batch_size": 64
    }
]


# --------------------------------------------------
# 4. Train and compare all configurations
# --------------------------------------------------

results = []

EPOCHS = 5

for number, config in enumerate(configs, start=1):

    print("\n----------------------------------")
    print("Experiment:", number)
    print("Configuration:", config["name"])
    print("----------------------------------")

    tf.keras.backend.clear_session()
    tf.keras.utils.set_random_seed(42)

    model = create_cnn(config)

    # Count trainable and non-trainable parameters
    parameter_count = model.count_params()

    # Measure training time
    start_time = time.perf_counter()

    model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=EPOCHS,
        batch_size=config["batch_size"],
        verbose=0
    )

    end_time = time.perf_counter()

    training_time = end_time - start_time

    # Evaluate on test data
    test_loss, test_accuracy = model.evaluate(
        X_test,
        y_test_cat,
        verbose=0
    )

    results.append({
        "name": config["name"],
        "accuracy": test_accuracy * 100,
        "parameters": parameter_count,
        "time": training_time
    })

    print("Test Accuracy:", round(test_accuracy * 100, 2), "%")
    print("Parameters:", parameter_count)
    print("Training Time:", round(training_time, 2), "seconds")


# --------------------------------------------------
# 5. Display comparison table
# --------------------------------------------------

print("\n========== CNN COMPARISON RESULTS ==========")

print(
    f"{'Configuration':<23}"
    f"{'Accuracy %':>13}"
    f"{'Parameters':>15}"
    f"{'Time (sec)':>13}"
)

for result in results:

    print(
        f"{result['name']:<23}"
        f"{result['accuracy']:>13.2f}"
        f"{result['parameters']:>15}"
        f"{result['time']:>13.2f}"
    )


# --------------------------------------------------
# 6. Find best accuracy and lightweight model
# --------------------------------------------------

best_model = max(results, key=lambda x: x["accuracy"])

lightest_model = min(
    results, key=lambda x: x["parameters"]
)

print("\nBest Accuracy Model:")
print(best_model["name"])
print("Accuracy:", round(best_model["accuracy"], 2), "%")

print("\nLightest Model:")
print(lightest_model["name"])
print("Parameters:", lightest_model["parameters"])


# --------------------------------------------------
# 7. Plot accuracy comparison
# --------------------------------------------------

names = [r["name"] for r in results]
accuracies = [r["accuracy"] for r in results]

plt.figure(figsize=(12, 5))
plt.bar(names, accuracies)
plt.title("CNN Architecture - Accuracy Comparison")
plt.xlabel("CNN Configurations")
plt.ylabel("Test Accuracy (%)")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.show()


# --------------------------------------------------
# 8. Plot parameter count comparison
# --------------------------------------------------

parameters = [r["parameters"] for r in results]

plt.figure(figsize=(12, 5))
plt.bar(names, parameters)
plt.title("CNN Model Parameter Comparison")
plt.xlabel("CNN Configurations")
plt.ylabel("Number of Parameters")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.show()


# --------------------------------------------------
# 9. Plot training time comparison
# --------------------------------------------------

times = [r["time"] for r in results]

plt.figure(figsize=(12, 5))
plt.bar(names, times)
plt.title("CNN Training Time Comparison")
plt.xlabel("CNN Configurations")
plt.ylabel("Training Time (seconds)")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.show()

print("\nCNN Architecture Comparison Completed!")
