
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)


# =========================================================
# PART 1: TWO-LAYER NEURAL NETWORK FROM SCRATCH
# =========================================================

class NeuralNetwork:

    def __init__(self, input_size, hidden_size, output_size,
                 learning_rate=1.0):

        self.learning_rate = learning_rate

        # Xavier-style weight initialisation
        self.W1 = np.random.randn(input_size, hidden_size) * np.sqrt(
            1 / input_size
        )
        self.b1 = np.zeros((1, hidden_size))

        self.W2 = np.random.randn(hidden_size, output_size) * np.sqrt(
            1 / hidden_size
        )
        self.b2 = np.zeros((1, output_size))

        self.loss_history = []

    # Sigmoid activation
    def sigmoid(self, x):
        x = np.clip(x, -500, 500)
        return 1 / (1 + np.exp(-x))

    # Forward propagation
    def forward(self, X):

        self.Z1 = np.dot(X, self.W1) + self.b1
        self.A1 = self.sigmoid(self.Z1)

        self.Z2 = np.dot(self.A1, self.W2) + self.b2
        self.A2 = self.sigmoid(self.Z2)

        return self.A2

    # Binary cross-entropy loss
    def calculate_loss(self, y, predictions):

        epsilon = 1e-8
        predictions = np.clip(predictions, epsilon, 1 - epsilon)

        loss = -np.mean(
            y * np.log(predictions)
            + (1 - y) * np.log(1 - predictions)
        )

        return loss

    # Backpropagation using the chain rule
    def backward(self, X, y):

        m = X.shape[0]

        # Output layer error
        dZ2 = (self.A2 - y) / m

        dW2 = np.dot(self.A1.T, dZ2)
        db2 = np.sum(dZ2, axis=0, keepdims=True)

        # Hidden layer error
        dA1 = np.dot(dZ2, self.W2.T)

        dZ1 = dA1 * self.A1 * (1 - self.A1)

        dW1 = np.dot(X.T, dZ1)
        db1 = np.sum(dZ1, axis=0, keepdims=True)

        gradients = {
            "dW1": dW1,
            "db1": db1,
            "dW2": dW2,
            "db2": db2
        }

        return gradients

    # Update weights and biases
    def update_weights(self, gradients):

        self.W1 -= self.learning_rate * gradients["dW1"]
        self.b1 -= self.learning_rate * gradients["db1"]

        self.W2 -= self.learning_rate * gradients["dW2"]
        self.b2 -= self.learning_rate * gradients["db2"]

    # Training
    def train(self, X, y, epochs=3000, print_every=500):

        for epoch in range(epochs):

            predictions = self.forward(X)

            loss = self.calculate_loss(y, predictions)
            self.loss_history.append(loss)

            gradients = self.backward(X, y)

            self.update_weights(gradients)

            if (epoch + 1) % print_every == 0:
                print(
                    "Epoch:", epoch + 1,
                    "Loss:", round(loss, 5)
                )

        # Calculate final loss after the last update
        final_predictions = self.forward(X)
        final_loss = self.calculate_loss(y, final_predictions)

        print("Final loss:", round(final_loss, 5))

    # Prediction
    def predict(self, X):

        probabilities = self.forward(X)
        return (probabilities >= 0.5).astype(int)


# =========================================================
# PART 2: XOR DATASET
# =========================================================

X_xor = np.array([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
], dtype=float)

y_xor = np.array([
    [0],
    [1],
    [1],
    [0]
], dtype=float)

print("\n========== XOR DATASET ==========")

xor_model = NeuralNetwork(
    input_size=2,
    hidden_size=8,
    output_size=1,
    learning_rate=1.0
)

xor_model.train(
    X_xor, y_xor,
    epochs=5000,
    print_every=1000
)

xor_predictions = xor_model.predict(X_xor)

print("\nXOR Actual outputs:")
print(y_xor.astype(int).flatten())

print("XOR Predicted outputs:")
print(xor_predictions.flatten())

xor_accuracy = np.mean(xor_predictions == y_xor) * 100

print("XOR Accuracy:", round(xor_accuracy, 2), "%")


# =========================================================
# PART 3: GENERATE TWO-MOONS DATASET USING NUMPY
# =========================================================

number_per_class = 150

theta = np.linspace(0, np.pi, number_per_class)

# First moon
moon1 = np.column_stack((
    np.cos(theta),
    np.sin(theta)
))

# Second moon
moon2 = np.column_stack((
    1 - np.cos(theta),
    0.5 - np.sin(theta)
))

X_moons = np.vstack((moon1, moon2))

y_moons = np.vstack((
    np.zeros((number_per_class, 1)),
    np.ones((number_per_class, 1))
))

# Add noise
X_moons += np.random.normal(0, 0.12, X_moons.shape)

# Shuffle the dataset
indices = np.random.permutation(len(X_moons))
X_moons = X_moons[indices]
y_moons = y_moons[indices]

# Standardise input features
X_mean = X_moons.mean(axis=0)
X_std = X_moons.std(axis=0)

X_moons = (X_moons - X_mean) / X_std

print("\n========== TWO-MOONS DATASET ==========")
print("Total samples:", len(X_moons))
print("Input features:", X_moons.shape[1])
print("Output classes: 0 and 1")


# =========================================================
# PART 4: NUMERICAL GRADIENT VERIFICATION
# =========================================================

print("\n========== GRADIENT VERIFICATION ==========")

# Use a small sample and a fresh network
check_model = NeuralNetwork(
    input_size=2,
    hidden_size=8,
    output_size=1,
    learning_rate=1.0
)

# Forward and analytical gradients
predictions = check_model.forward(X_moons)
analytical_gradients = check_model.backward(X_moons, y_moons)

# Select one weight from the first layer
i, j = 0, 0
epsilon = 1e-5

original_weight = check_model.W1[i, j]

# Loss when weight is slightly increased
check_model.W1[i, j] = original_weight + epsilon
loss_plus = check_model.calculate_loss(
    y_moons, check_model.forward(X_moons)
)

# Loss when weight is slightly decreased
check_model.W1[i, j] = original_weight - epsilon
loss_minus = check_model.calculate_loss(
    y_moons, check_model.forward(X_moons)
)

# Restore original weight
check_model.W1[i, j] = original_weight

# Numerical gradient
numerical_gradient = (
    loss_plus - loss_minus
) / (2 * epsilon)

# Analytical gradient
analytical_gradient = analytical_gradients["dW1"][i, j]

print("Analytical gradient:", analytical_gradient)
print("Numerical gradient: ", numerical_gradient)

difference = abs(
    analytical_gradient - numerical_gradient
)

print("Absolute difference:", difference)

if difference < 1e-5:
    print("Gradient verification successful!")
else:
    print("Check the backpropagation implementation.")


# =========================================================
# PART 5: TRAIN TWO-MOONS DATASET
# =========================================================

print("\n========== TWO-MOONS TRAINING ==========")

moons_model = NeuralNetwork(
    input_size=2,
    hidden_size=12,
    output_size=1,
    learning_rate=1.0
)

moons_model.train(
    X_moons, y_moons,
    epochs=3000,
    print_every=500
)

moons_predictions = moons_model.predict(X_moons)

moons_accuracy = np.mean(
    moons_predictions == y_moons
) * 100

print("Two-moons accuracy:", round(moons_accuracy, 2), "%")


# =========================================================
# PART 6: PLOT LOSS AGAINST EPOCHS
# =========================================================

plt.figure(figsize=(9, 5))

plt.plot(
    xor_model.loss_history,
    label="XOR Dataset"
)

plt.plot(
    moons_model.loss_history,
    label="Two-Moons Dataset"
)

plt.xlabel("Epochs")
plt.ylabel("Binary Cross-Entropy Loss")
plt.title("Backpropagation: Loss vs Epochs")
plt.legend()
plt.grid()
plt.show()


# =========================================================
# PART 7: PLOT TWO-MOONS DECISION BOUNDARY
# =========================================================

x_min, x_max = X_moons[:, 0].min() - 0.5, X_moons[:, 0].max() + 0.5
y_min, y_max = X_moons[:, 1].min() - 0.5, X_moons[:, 1].max() + 0.5

xx, yy = np.meshgrid(
    np.linspace(x_min, x_max, 250),
    np.linspace(y_min, y_max, 250)
)

grid_points = np.column_stack((
    xx.ravel(),
    yy.ravel()
))

probabilities = moons_model.forward(grid_points)
Z = probabilities.reshape(xx.shape)

plt.figure(figsize=(8, 6))

plt.contourf(
    xx, yy, Z,
    levels=[0, 0.5, 1],
    alpha=0.3
)

plt.contour(
    xx, yy, Z,
    levels=[0.5],
    colors="black"
)

plt.scatter(
    X_moons[y_moons.flatten() == 0, 0],
    X_moons[y_moons.flatten() == 0, 1],
    label="Customer Class 0"
)

plt.scatter(
    X_moons[y_moons.flatten() == 1, 0],
    X_moons[y_moons.flatten() == 1, 1],
    label="Customer Class 1"
)

plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.title("Two-Layer Neural Network Decision Boundary")
plt.legend()
plt.grid()
plt.show()


# =========================================================
# PART 8: PLOT XOR DECISION BOUNDARY
# =========================================================

xx_xor, yy_xor = np.meshgrid(
    np.linspace(-0.3, 1.3, 200),
    np.linspace(-0.3, 1.3, 200)
)

xor_grid = np.column_stack((
    xx_xor.ravel(),
    yy_xor.ravel()
))

xor_probabilities = xor_model.forward(xor_grid)
xor_Z = xor_probabilities.reshape(xx_xor.shape)

plt.figure(figsize=(7, 5))

plt.contourf(
    xx_xor, yy_xor, xor_Z,
    levels=[0, 0.5, 1],
    alpha=0.3
)

plt.contour(
    xx_xor, yy_xor, xor_Z,
    levels=[0.5],
    colors="black"
)

plt.scatter(
    X_xor[:, 0],
    X_xor[:, 1],
    c=y_xor.flatten(),
    s=150,
    edgecolors="black"
)

plt.xlabel("Input X1")
plt.ylabel("Input X2")
plt.title("XOR Nonlinear Decision Boundary")
plt.xticks([0, 1])
plt.yticks([0, 1])
plt.grid()
plt.show()


print("\nExperiment 8 completed successfully.")
