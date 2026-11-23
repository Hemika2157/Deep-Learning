
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    ConfusionMatrixDisplay
)

# Reproducibility
np.random.seed(42)
tf.keras.utils.set_random_seed(42)


# =====================================================
# PART 1: GENERATE TELECOM CUSTOMER DATASET
# =====================================================

n = 1500

tenure = np.random.randint(1, 73, n)
monthly_charges = np.random.uniform(20, 120, n)
complaints = np.random.randint(0, 8, n)
data_usage = np.random.uniform(1, 100, n)

contract_type = np.random.choice(
    ["Month-to-month", "One-year", "Two-year"],
    size=n,
    p=[0.55, 0.25, 0.20]
)

# Generate churn probability using customer characteristics
risk = (
    1.5
    - 0.045 * tenure
    + 0.025 * (monthly_charges - 70)
    + 0.55 * complaints
    - 0.012 * data_usage
    + 1.0 * (contract_type == "Month-to-month")
    - 0.5 * (contract_type == "Two-year")
)

probability = 1 / (1 + np.exp(-risk))

# Churn: 1 = Yes, 0 = No
churn = (np.random.random(n) < probability).astype(int)

print("TELECOM CUSTOMER DATASET")
print("Total customers:", n)
print("Churned customers:", np.sum(churn == 1))
print("Non-churned customers:", np.sum(churn == 0))


# =====================================================
# PART 2: PREPROCESSING
# =====================================================

# Encode categorical contract type using one-hot encoding
contract_encoded = np.column_stack([
    (contract_type == "Month-to-month").astype(int),
    (contract_type == "One-year").astype(int),
    (contract_type == "Two-year").astype(int)
])

# Combine numerical and encoded categorical features
X = np.column_stack((
    tenure,
    monthly_charges,
    complaints,
    data_usage,
    contract_encoded
)).astype(float)

y = churn

# Split data into training and testing
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Scale numerical features
scaler = StandardScaler()

X_train[:, :4] = scaler.fit_transform(X_train[:, :4])
X_test[:, :4] = scaler.transform(X_test[:, :4])

print("\nPREPROCESSING COMPLETED")
print("Training samples:", len(X_train))
print("Testing samples:", len(X_test))
print("Input features:", X_train.shape[1])


# =====================================================
# PART 3: BUILD MLP MODEL
# =====================================================

def create_mlp(width):

    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(7,)),

        tf.keras.layers.Dense(width, activation="relu"),

        tf.keras.layers.Dense(width // 2, activation="relu"),

        tf.keras.layers.Dense(width // 4, activation="relu"),

        tf.keras.layers.Dense(1, activation="sigmoid")
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=0.001
        ),
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )

    return model


# =====================================================
# PART 4: TRAIN MLP WITH DIFFERENT HIDDEN-LAYER WIDTHS
# =====================================================

widths = [8, 16, 32]

results = {}
histories = {}
trained_models = {}

epochs = 50

for width in widths:

    print("\nTraining MLP with hidden-layer width:", width)

    tf.keras.utils.set_random_seed(42)

    model = create_mlp(width)

    history = model.fit(
        X_train,
        y_train,
        validation_split=0.2,
        epochs=epochs,
        batch_size=32,
        verbose=0
    )

    # Evaluate using separate test data
    test_loss, test_accuracy = model.evaluate(
        X_test, y_test, verbose=0
    )

    results[width] = test_accuracy * 100
    histories[width] = history.history
    trained_models[width] = model

    print("Test loss:", round(test_loss, 4))
    print("Test accuracy:", round(test_accuracy * 100, 2), "%")


# =====================================================
# PART 5: SELECT BEST MODEL
# =====================================================

best_width = max(results, key=results.get)
best_model = trained_models[best_width]
best_history = histories[best_width]

print("\nBEST HIDDEN-LAYER WIDTH:", best_width)


# =====================================================
# PART 6: TRAINING AND VALIDATION LOSS GRAPH
# =====================================================

plt.figure(figsize=(9, 5))

plt.plot(
    best_history["loss"],
    label="Training Loss"
)

plt.plot(
    best_history["val_loss"],
    label="Validation Loss"
)

plt.xlabel("Epochs")
plt.ylabel("Binary Cross-Entropy Loss")
plt.title("Training and Validation Loss")
plt.legend()
plt.grid()
plt.show()


# =====================================================
# PART 7: TRAINING AND VALIDATION ACCURACY GRAPH
# =====================================================

plt.figure(figsize=(9, 5))

plt.plot(
    best_history["accuracy"],
    label="Training Accuracy"
)

plt.plot(
    best_history["val_accuracy"],
    label="Validation Accuracy"
)

plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.title("Training and Validation Accuracy")
plt.legend()
plt.grid()
plt.show()


# =====================================================
# PART 8: EVALUATION METRICS
# =====================================================

predicted_probability = best_model.predict(
    X_test, verbose=0
).flatten()

predictions = (predicted_probability >= 0.5).astype(int)

accuracy = accuracy_score(y_test, predictions)

precision = precision_score(
    y_test, predictions, zero_division=0
)

recall = recall_score(
    y_test, predictions, zero_division=0
)

f1 = f1_score(
    y_test, predictions, zero_division=0
)

cm = confusion_matrix(
    y_test, predictions, labels=[0, 1]
)

print("\n========== MODEL EVALUATION ==========")

print("Accuracy :", round(accuracy * 100, 2), "%")
print("Precision:", round(precision, 4))
print("Recall   :", round(recall, 4))
print("F1 Score :", round(f1, 4))

print("\nConfusion Matrix:")
print(cm)

display = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["No Churn", "Churn"]
)

display.plot(cmap="Blues")
plt.title("Customer Churn Confusion Matrix")
plt.show()


# =====================================================
# PART 9: COMPARE HIDDEN-LAYER WIDTHS
# =====================================================

print("\n========== WIDTH COMPARISON ==========")

for width in widths:
    print(
        "Width:", width,
        "| Test Accuracy:", round(results[width], 2), "%"
    )

plt.figure(figsize=(8, 5))

plt.bar(
    [str(width) for width in widths],
    [results[width] for width in widths]
)

plt.xlabel("Hidden Layer Width")
plt.ylabel("Test Accuracy (%)")
plt.title("Effect of Hidden-Layer Width")
plt.grid(axis="y")
plt.show()


print("\nExperiment 9 completed successfully.")
