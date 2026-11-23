
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

# Reproducibility
np.random.seed(42)
tf.keras.utils.set_random_seed(42)


# ==========================================================
# PART 1: IMPLEMENT GD, SGD AND MOMENTUM FROM SCRATCH
# ==========================================================

# Loss function:
# L(x, y) = (x - 3)^2 + 10(y + 2)^2

def loss_function(x, y):
    return (x - 3)**2 + 10 * (y + 2)**2


# Gradient of the loss function
def gradient(x, y):
    dx = 2 * (x - 3)
    dy = 20 * (y + 2)
    return np.array([dx, dy])


learning_rate = 0.04
iterations = 100

start = np.array([-4.0, 4.0])

# Batch Gradient Descent
position = start.copy()
gd_path = [position.copy()]

for i in range(iterations):
    grad = gradient(position[0], position[1])
    position = position - learning_rate * grad
    gd_path.append(position.copy())


# Stochastic Gradient Descent
# Random noise represents variation in individual samples.
position = start.copy()
sgd_path = [position.copy()]

for i in range(iterations):
    grad = gradient(position[0], position[1])
    noisy_grad = grad + np.random.normal(0, 0.6, 2)
    position = position - learning_rate * noisy_grad
    sgd_path.append(position.copy())


# Gradient Descent with Momentum
position = start.copy()
velocity = np.zeros(2)
momentum = 0.9
momentum_path = [position.copy()]

for i in range(iterations):
    grad = gradient(position[0], position[1])
    velocity = momentum * velocity + learning_rate * grad
    position = position - velocity
    momentum_path.append(position.copy())


# Plot loss contour and optimiser paths
x_values = np.linspace(-5, 6, 200)
y_values = np.linspace(-5, 6, 200)
X, Y = np.meshgrid(x_values, y_values)
Z = loss_function(X, Y)

plt.figure(figsize=(9, 6))
plt.contour(X, Y, Z, levels=30)
plt.plot(*np.array(gd_path).T, 'r.-', label="Batch GD")
plt.plot(*np.array(sgd_path).T, 'b.-', label="SGD")
plt.plot(*np.array(momentum_path).T, 'g.-', label="Momentum")
plt.scatter(3, -2, color="black", marker="*", s=150,
            label="Minimum")
plt.xlabel("Parameter x")
plt.ylabel("Parameter y")
plt.title("Optimiser Paths on Loss Contour")
plt.legend()
plt.grid()
plt.show()


# ==========================================================
# PART 2: EFFECT OF DIFFERENT LEARNING RATES
# ==========================================================

learning_rates = [0.005, 0.04, 0.12]

plt.figure(figsize=(8, 5))

for lr in learning_rates:
    position = start.copy()
    losses = []

    for i in range(50):
        losses.append(loss_function(position[0], position[1]))
        grad = gradient(position[0], position[1])
        position = position - lr * grad

    plt.plot(losses, label=f"Learning rate = {lr}")

plt.xlabel("Iterations")
plt.ylabel("Loss")
plt.title("Effect of Learning Rate on Gradient Descent")
plt.legend()
plt.grid()
plt.show()


# ==========================================================
# PART 3: CREATE DELIVERY-TIME DATASET
# ==========================================================

# Features:
# Distance (km), Traffic index, Package weight (kg),
# Time of day (hour)

number_of_samples = 1200

distance = np.random.uniform(1, 30, number_of_samples)
traffic = np.random.uniform(1, 10, number_of_samples)
weight = np.random.uniform(0.5, 20, number_of_samples)
time_of_day = np.random.uniform(0, 24, number_of_samples)

# Delivery time in minutes
delivery_time = (
    10
    + 2.0 * distance
    + 2.5 * traffic
    + 0.5 * weight
    + 0.3 * np.abs(time_of_day - 12)
    + np.random.normal(0, 3, number_of_samples)
)

# Combine input features
X = np.column_stack((distance, traffic, weight, time_of_day))
y = delivery_time.reshape(-1, 1)

# Standardise input features and target
X_mean = X.mean(axis=0)
X_std = X.std(axis=0)
X = (X - X_mean) / X_std

y_mean = y.mean()
y_std = y.std()
y = (y - y_mean) / y_std

# Split into training and validation data
indices = np.random.permutation(number_of_samples)
split = int(0.8 * number_of_samples)

train_indices = indices[:split]
val_indices = indices[split:]

X_train, X_val = X[train_indices], X[val_indices]
y_train, y_val = y[train_indices], y[val_indices]


# ==========================================================
# PART 4: TRAIN THE SAME KERAS MODEL WITH THREE OPTIMISERS
# ==========================================================

def create_model():
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(4,)),
        tf.keras.layers.Dense(32, activation="relu"),
        tf.keras.layers.Dense(16, activation="relu"),
        tf.keras.layers.Dense(1)
    ])

    model.compile(
        optimizer="sgd",
        loss="mse",
        metrics=["mae"]
    )
    return model


# Create one initial model to get common starting weights
base_model = create_model()
initial_weights = base_model.get_weights()

optimisers = {
    "SGD": tf.keras.optimizers.SGD(learning_rate=0.01),
    "SGD + Momentum": tf.keras.optimizers.SGD(
        learning_rate=0.01, momentum=0.9
    ),
    "Adam": tf.keras.optimizers.Adam(learning_rate=0.01)
}

histories = {}
trained_models = {}

epochs = 80

for name, optimiser in optimisers.items():

    model = create_model()
    model.set_weights(initial_weights)

    model.compile(
        optimizer=optimiser,
        loss="mse",
        metrics=["mae"]
    )

    print("\nTraining with", name)

    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=32,
        verbose=0
    )

    histories[name] = history.history
    trained_models[name] = model

    print("Final training loss:",
          round(history.history["loss"][-1], 4))
    print("Final validation loss:",
          round(history.history["val_loss"][-1], 4))


# ==========================================================
# PART 5: PLOT TRAINING AND VALIDATION LOSS
# ==========================================================

plt.figure(figsize=(10, 6))

for name in histories:
    plt.plot(histories[name]["loss"], label=name)

plt.xlabel("Epochs")
plt.ylabel("Training Loss (MSE)")
plt.title("Training Loss Comparison")
plt.legend()
plt.grid()
plt.show()


plt.figure(figsize=(10, 6))

for name in histories:
    plt.plot(histories[name]["val_loss"], label=name)

plt.xlabel("Epochs")
plt.ylabel("Validation Loss (MSE)")
plt.title("Validation Loss Comparison")
plt.legend()
plt.grid()
plt.show()


# ==========================================================
# PART 6: COMPARE DELIVERY-TIME PREDICTIONS
# ==========================================================

print("\nFINAL MODEL PERFORMANCE")

for name, model in trained_models.items():

    predictions = model.predict(X_val, verbose=0)

    # Convert standardised predictions back to minutes
    predictions = predictions * y_std + y_mean
    actual_values = y_val * y_std + y_mean

    mae = np.mean(np.abs(predictions - actual_values))

    print(name, "Mean Absolute Error:",
          round(float(mae), 2), "minutes")

print("\nExperiment completed successfully.")
