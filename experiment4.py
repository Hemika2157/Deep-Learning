
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import entropy

# 1. Actual weather distribution (30 days)
true = np.array([15, 9, 6]) / 30

# Model 1 predictions
model1 = np.array([0.50, 0.30, 0.20])

# Model 2 predictions
model2 = np.array([0.70, 0.20, 0.10])

print("TRUE WEATHER DISTRIBUTION:")
print(true)

print("\nMODEL 1 PREDICTIONS:")
print(model1)

print("\nMODEL 2 PREDICTIONS:")
print(model2)

# 2. Entropy of true distribution
H_true = entropy(true, base=2)

print("\nENTROPY OF TRUE DISTRIBUTION:")
print(round(H_true, 4), "bits")

# 3. Cross-Entropy and KL Divergence
H_cross1 = -np.sum(true * np.log2(model1))
KL1 = entropy(true, model1, base=2)

H_cross2 = -np.sum(true * np.log2(model2))
KL2 = entropy(true, model2, base=2)

print("\nMODEL 1 RESULTS:")
print("Cross Entropy:", round(H_cross1, 4))
print("KL Divergence:", round(KL1, 4))
print("Entropy + KL:", round(H_true + KL1, 4))

print("\nMODEL 2 RESULTS:")
print("Cross Entropy:", round(H_cross2, 4))
print("KL Divergence:", round(KL2, 4))
print("Entropy + KL:", round(H_true + KL2, 4))

# 4. Verify the relationship
print("\nVERIFICATION:")
print("Model 1:", np.isclose(H_cross1, H_true + KL1))
print("Model 2:", np.isclose(H_cross2, H_true + KL2))

# 5. KL Divergence Asymmetry
KL_true_model1 = entropy(true, model1, base=2)
KL_model1_true = entropy(model1, true, base=2)

print("\nKL DIVERGENCE ASYMMETRY:")
print("KL(True || Model 1):", round(KL_true_model1, 4))
print("KL(Model 1 || True):", round(KL_model1_true, 4))

# 6. Binary Cross-Entropy Loss
y_true = 1

predicted = np.linspace(0.01, 0.99, 100)

loss = -(
    y_true * np.log(predicted) +
    (1 - y_true) * np.log(1 - predicted)
)

plt.figure(figsize=(8, 5))

plt.plot(predicted, loss, label="Binary Cross-Entropy")

plt.xlabel("Predicted Probability")
plt.ylabel("Cross-Entropy Loss")
plt.title("Binary Cross-Entropy Loss vs Predicted Probability")

plt.grid()
plt.legend()
plt.show()

# 7. Weather Distribution Comparison
labels = ["Sunny", "Cloudy", "Rainy"]
x = np.arange(3)
width = 0.25

plt.figure(figsize=(8, 5))

plt.bar(x - width, true, width, label="Actual")
plt.bar(x, model1, width, label="Model 1")
plt.bar(x + width, model2, width, label="Model 2")

plt.xticks(x, labels)
plt.ylabel("Probability")
plt.title("Weather Forecast Comparison")
plt.legend()
plt.show()

print("\nExperiment 4 Completed Successfully!")
