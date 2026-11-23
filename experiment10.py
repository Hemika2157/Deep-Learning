
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# -------------------------------------------------
# 1. Generate a small synthetic diabetes dataset
# -------------------------------------------------

np.random.seed(42)
tf.random.set_seed(42)

records = 500
features = 8

X = np.random.normal(0, 1, (records, features))

# Create a diabetes risk score
risk = (
    0.9 * X[:, 0]
    + 0.7 * X[:, 1]
    + 0.5 * X[:, 2]
    - 0.4 * X[:, 3]
    + 0.3 * X[:, 4]
)

probability = 1 / (1 + np.exp(-risk))
y = (probability > 0.5).astype(int)

# Split into training, validation and testing data
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.4, random_state=42, stratify=y
)

X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
)

# Standardise features using training data only
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)
X_test = scaler.transform(X_test)

print("Training records:", len(X_train))
print("Validation records:", len(X_val))
print("Testing records:", len(X_test))


# -------------------------------------------------
# 2. Create MLP model
# -------------------------------------------------

def create_model(l2_value=0, dropout_rate=0):
    model = Sequential([
        Dense(
            256,
            activation="relu",
            kernel_regularizer=l2(l2_value),
            input_shape=(features,)
        ),
        Dense(
            128,
            activation="relu",
            kernel_regularizer=l2(l2_value)
        ),
        Dense(
            64,
            activation="relu",
            kernel_regularizer=l2(l2_value)
        ),
        Dense(
            32,
            activation="relu",
            kernel_regularizer=l2(l2_value)
        ),
        Dense(1, activation="sigmoid")
    ])

    # Add dropout after hidden layers when required
    if dropout_rate > 0:
        model = Sequential([
            Dense(256, activation="relu", input_shape=(features,)),
            Dropout(dropout_rate),
            Dense(128, activation="relu"),
            Dropout(dropout_rate),
            Dense(64, activation="relu"),
            Dropout(dropout_rate),
            Dense(32, activation="relu"),
            Dense(1, activation="sigmoid")
        ])

    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )

    return model


# -------------------------------------------------
# 3. Train and evaluate each model
# -------------------------------------------------

results = []
histories = {}

def train_and_evaluate(name, model, use_early_stopping=False):

    callbacks = []

    if use_early_stopping:
        early_stop = EarlyStopping(
            monitor="val_loss",
            patience=12,
            restore_best_weights=True
        )
        callbacks.append(early_stop)

    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=150,
        batch_size=16,
        verbose=0,
        callbacks=callbacks
    )

    train_loss, train_accuracy = model.evaluate(
        X_train, y_train, verbose=0
    )

    test_loss, test_accuracy = model.evaluate(
        X_test, y_test, verbose=0
    )

    gap = train_accuracy - test_accuracy

    results.append([
        name,
        train_accuracy * 100,
        test_accuracy * 100,
        gap * 100
    ])

    histories[name] = history

    print(
        f"{name}: Train Accuracy = {train_accuracy * 100:.2f}%, "
        f"Test Accuracy = {test_accuracy * 100:.2f}%, "
        f"Gap = {gap * 100:.2f}%"
    )

    return model


# -------------------------------------------------
# 4. Baseline model without regularisation
# -------------------------------------------------

print("\nTraining Baseline Model")

baseline = create_model()
train_and_evaluate("Baseline", baseline)


# -------------------------------------------------
# 5. L2 Regularisation with three lambda values
# -------------------------------------------------

print("\nTraining L2 Regularisation Models")

for value in [0.0001, 0.001, 0.01]:

    model = create_model(l2_value=value)

    train_and_evaluate(
        f"L2 lambda={value}",
        model
    )


# -------------------------------------------------
# 6. Dropout with three dropout rates
# -------------------------------------------------

print("\nTraining Dropout Models")

for rate in [0.2, 0.3, 0.5]:

    model = create_model(dropout_rate=rate)

    train_and_evaluate(
        f"Dropout rate={rate}",
        model
    )


# -------------------------------------------------
# 7. Early Stopping
# -------------------------------------------------

print("\nTraining Early Stopping Model")

early_model = create_model()

train_and_evaluate(
    "Early Stopping",
    early_model,
    use_early_stopping=True
)


# -------------------------------------------------
# 8. Display comparison table
# -------------------------------------------------

print("\n========== MODEL COMPARISON ==========")
print(
    f"{'Model':<25}"
    f"{'Train Acc %':>13}"
    f"{'Test Acc %':>13}"
    f"{'Gap %':>10}"
)

for row in results:
    print(
        f"{row[0]:<25}"
        f"{row[1]:>13.2f}"
        f"{row[2]:>13.2f}"
        f"{row[3]:>10.2f}"
    )


# -------------------------------------------------
# 9. Plot baseline overfitting
# -------------------------------------------------

baseline_history = histories["Baseline"]

plt.figure(figsize=(8, 5))
plt.plot(baseline_history.history["loss"], label="Training Loss")
plt.plot(baseline_history.history["val_loss"], label="Validation Loss")
plt.title("Baseline Model - Overfitting")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.grid(True)
plt.show()


# -------------------------------------------------
# 10. Compare validation loss of all models
# -------------------------------------------------

plt.figure(figsize=(11, 6))

for name, history in histories.items():
    plt.plot(history.history["val_loss"], label=name)

plt.title("Validation Loss Comparison")
plt.xlabel("Epochs")
plt.ylabel("Validation Loss")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


# -------------------------------------------------
# 11. Compare test accuracy
# -------------------------------------------------

model_names = [row[0] for row in results]
test_accuracies = [row[2] for row in results]

plt.figure(figsize=(11, 5))
plt.bar(model_names, test_accuracies)
plt.title("Test Accuracy of Different Models")
plt.xlabel("Models")
plt.ylabel("Test Accuracy (%)")
plt.xticks(rotation=35, ha="right")
plt.tight_layout()
plt.show()

