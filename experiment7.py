
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)

# =====================================================
# PART 1: GENERATE 100 LOAN APPLICATIONS
# =====================================================

n = 100

credit_score = np.random.uniform(300, 850, n)
debt_ratio = np.random.uniform(0.1, 1.0, n)

# Normalise the input features
x1 = (credit_score - 575) / 275
x2 = (0.55 - debt_ratio) / 0.45

X = np.column_stack((x1, x2))

# Generate linearly separable loan labels
# 1 = Approved, 0 = Rejected
y = (x1 + x2 > 0.05).astype(int)

print("Total applications:", n)
print("Approved:", np.sum(y == 1))
print("Rejected:", np.sum(y == 0))


# =====================================================
# PART 2: PERCEPTRON IMPLEMENTATION
# =====================================================

class Perceptron:

    def __init__(self, learning_rate=0.1):
        self.learning_rate = learning_rate
        self.weights = np.zeros(2)
        self.bias = 0.0
        self.epoch_errors = []
        self.weight_history = []

    # Step activation function
    def step(self, value):
        return 1 if value >= 0 else 0

    # Forward propagation
    def predict_one(self, inputs):
        weighted_sum = np.dot(inputs, self.weights) + self.bias
        output = self.step(weighted_sum)
        return output

    # Training using perceptron learning rule
    def train(self, X, y, epochs=50):

        for epoch in range(epochs):

            errors = 0

            for inputs, target in zip(X, y):

                prediction = self.predict_one(inputs)

                error = target - prediction

                # Update weights and bias
                self.weights += (
                    self.learning_rate * error * inputs
                )

                self.bias += self.learning_rate * error

                if error != 0:
                    errors += 1

            self.epoch_errors.append(errors)

            self.weight_history.append(
                [self.weights[0], self.weights[1], self.bias]
            )

            print(
                "Epoch:", epoch + 1,
                "| Errors:", errors,
                "| Weights:", np.round(self.weights, 3),
                "| Bias:", round(self.bias, 3)
            )

            # Stop if all samples are classified correctly
            if errors == 0:
                print("Training converged!")
                break

    def predict(self, X):
        return np.array([
            self.predict_one(inputs) for inputs in X
        ])


# Train loan approval perceptron
print("\nLOAN APPROVAL PERCEPTRON TRAINING")

model = Perceptron(learning_rate=0.1)
model.train(X, y, epochs=50)

predictions = model.predict(X)
accuracy = np.mean(predictions == y) * 100

print("\nFinal weights:", np.round(model.weights, 3))
print("Final bias:", round(model.bias, 3))
print("Training accuracy:", round(accuracy, 2), "%")


# =====================================================
# PART 3: PLOT LOAN DECISION BOUNDARY
# =====================================================

plt.figure(figsize=(8, 6))

plt.scatter(
    X[y == 1, 0], X[y == 1, 1],
    color="green", label="Approved"
)

plt.scatter(
    X[y == 0, 0], X[y == 0, 1],
    color="red", label="Rejected"
)

# Decision boundary:
# w1*x1 + w2*x2 + bias = 0

x_line = np.linspace(-1, 1, 100)

if abs(model.weights[1]) > 1e-10:
    y_line = (
        -model.weights[0] * x_line - model.bias
    ) / model.weights[1]

    plt.plot(
        x_line, y_line,
        color="blue", label="Decision Boundary"
    )

plt.xlabel("Normalised Credit Score")
plt.ylabel("Normalised Debt Ratio")
plt.title("Loan Approval Perceptron Decision Boundary")
plt.legend()
plt.grid()
plt.show()


# =====================================================
# PART 4: PLOT WEIGHT UPDATES PER EPOCH
# =====================================================

weight_history = np.array(model.weight_history)

plt.figure(figsize=(8, 5))

plt.plot(weight_history[:, 0], label="Weight 1")
plt.plot(weight_history[:, 1], label="Weight 2")
plt.plot(weight_history[:, 2], label="Bias")

plt.xlabel("Epoch")
plt.ylabel("Parameter Value")
plt.title("Perceptron Weight and Bias Updates")
plt.legend()
plt.grid()
plt.show()


# =====================================================
# PART 5: XOR DATASET
# =====================================================

X_xor = np.array([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
])

y_xor = np.array([0, 1, 1, 0])

print("\nXOR PERCEPTRON TRAINING")

xor_model = Perceptron(learning_rate=0.1)
xor_model.train(X_xor, y_xor, epochs=20)

xor_predictions = xor_model.predict(X_xor)

print("\nActual XOR outputs:    ", y_xor)
print("Predicted XOR outputs: ", xor_predictions)

if np.array_equal(xor_predictions, y_xor):
    print("XOR learned successfully.")
else:
    print("Single perceptron cannot learn XOR.")
    print("The XOR pattern is not linearly separable.")


# Plot XOR pattern
plt.figure(figsize=(6, 5))

for i, point in enumerate(X_xor):
    if y_xor[i] == 1:
        plt.scatter(
            point[0], point[1],
            color="green", s=150, label="Output 1" if i == 1 else ""
        )
    else:
        plt.scatter(
            point[0], point[1],
            color="red", s=150, label="Output 0" if i == 0 else ""
        )

plt.xlabel("Input X1")
plt.ylabel("Input X2")
plt.title("XOR Dataset")
plt.xticks([0, 1])
plt.yticks([0, 1])
plt.legend()
plt.grid()
plt.show()


# =====================================================
# PART 6: COMPARE ACTIVATION FUNCTIONS
# =====================================================

def step_activation(x):
    return np.where(x >= 0, 1, 0)


def sigmoid_activation(x):
    return 1 / (1 + np.exp(-x))


def relu_activation(x):
    return np.maximum(0, x)


values = np.array([-3, -2, -1, 0, 1, 2, 3])

print("\nACTIVATION FUNCTION COMPARISON")
print("Input:   ", values)
print("Step:    ", step_activation(values))
print("Sigmoid: ", np.round(sigmoid_activation(values), 3))
print("ReLU:    ", relu_activation(values))


# Plot activation functions
x = np.linspace(-5, 5, 200)

plt.figure(figsize=(9, 6))

plt.plot(x, step_activation(x), label="Step")
plt.plot(x, sigmoid_activation(x), label="Sigmoid")
plt.plot(x, relu_activation(x), label="ReLU")

plt.xlabel("Input")
plt.ylabel("Activation Output")
plt.title("Comparison of Activation Functions")
plt.legend()
plt.grid()
plt.show()

print("\nExperiment 7 completed successfully.")
