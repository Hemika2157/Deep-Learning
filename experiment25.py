
# Experiment 25: Restricted Boltzmann Machine Energy and Reconstruction

import numpy as np
import matplotlib.pyplot as plt

from scipy.special import expit
from sklearn.datasets import load_digits
from scipy.ndimage import zoom


# ------------------------------------------------
# 1. LOAD AND PREPROCESS THE DATASET
# ------------------------------------------------

digits = load_digits()

images = digits.images
labels = digits.target

# Resize 8x8 images into 28x28 images
resized_images = np.array([
    zoom(image, (28 / 8, 28 / 8), order=1)
    for image in images
])

# Normalize and convert pixels into binary values
resized_images = resized_images / 16.0
binary_images = (resized_images > 0.35).astype(np.float32)

# Flatten each image into 784 visible units
data = binary_images.reshape(len(binary_images), 784)

# Shuffle and select training samples
np.random.seed(42)
indices = np.random.permutation(len(data))
data = data[indices[:1200]]

print("Dataset shape:", data.shape)
print("Visible units:", data.shape[1])


# ------------------------------------------------
# 2. RBM INITIALIZATION
# ------------------------------------------------

n_visible = 784
n_hidden = 64

learning_rate = 0.01
epochs = 20
batch_size = 50

W = np.random.normal(
    0, 0.01, (n_visible, n_hidden)
).astype(np.float32)

visible_bias = np.zeros(n_visible, dtype=np.float32)
hidden_bias = np.zeros(n_hidden, dtype=np.float32)


# ------------------------------------------------
# 3. ENERGY FUNCTION
# ------------------------------------------------

def energy(v, h):
    return (
        -np.dot(v, visible_bias)
        -np.dot(h, hidden_bias)
        -np.dot(v, W) @ h
    )


# ------------------------------------------------
# 4. HIDDEN ACTIVATION PROBABILITY
# ------------------------------------------------

def hidden_probability(v):
    return expit(np.dot(v, W) + hidden_bias)


# ------------------------------------------------
# 5. VISIBLE ACTIVATION PROBABILITY
# ------------------------------------------------

def visible_probability(h):
    return expit(np.dot(h, W.T) + visible_bias)


# ------------------------------------------------
# 6. FREE ENERGY
# ------------------------------------------------

def free_energy(v):
    hidden_input = np.dot(v, W) + hidden_bias

    return (
        -np.dot(v, visible_bias)
        -np.sum(np.logaddexp(0, hidden_input), axis=1)
    )


# ------------------------------------------------
# 7. TRAIN RBM USING CD-1
# ------------------------------------------------

free_energy_history = []
reconstruction_history = []

print("\nTraining RBM...")

for epoch in range(epochs):

    np.random.shuffle(data)

    total_error = 0
    total_free_energy = 0
    batches = 0

    for start in range(0, len(data), batch_size):

        # Original visible data
        v0 = data[start:start + batch_size]

        # Positive phase
        ph0 = hidden_probability(v0)
        h0 = (np.random.rand(*ph0.shape) < ph0).astype(np.float32)

        # Negative phase: reconstruct visible units
        pv1 = visible_probability(h0)
        v1 = (np.random.rand(*pv1.shape) < pv1).astype(np.float32)

        # Recalculate hidden probabilities
        ph1 = hidden_probability(v1)

        # Contrastive Divergence weight update
        W += learning_rate * (
            np.dot(v0.T, ph0) - np.dot(v1.T, ph1)
        ) / len(v0)

        visible_bias += learning_rate * np.mean(
            v0 - v1, axis=0
        )

        hidden_bias += learning_rate * np.mean(
            ph0 - ph1, axis=0
        )

        # Reconstruction error
        error = np.mean((v0 - pv1) ** 2)
        total_error += error

        # Average free energy
        total_free_energy += np.mean(free_energy(v0))

        batches += 1

    avg_error = total_error / batches
    avg_energy = total_free_energy / batches

    reconstruction_history.append(avg_error)
    free_energy_history.append(avg_energy)

    print(
        f"Epoch {epoch + 1}/{epochs} | "
        f"Free Energy: {avg_energy:.4f} | "
        f"Reconstruction Error: {avg_error:.4f}"
    )


# ------------------------------------------------
# 8. ORIGINAL AND RECONSTRUCTED IMAGES
# ------------------------------------------------

test_images = data[:10]

# Get hidden probabilities
hidden_probs = hidden_probability(test_images)

# Use probabilities instead of random samples
reconstructed_probs = visible_probability(hidden_probs)

# Convert probabilities to binary images
reconstructed_images = (
    reconstructed_probs > 0.5
).astype(np.float32)


plt.figure(figsize=(12, 4))

for i in range(10):

    # Original image
    plt.subplot(2, 10, i + 1)
    plt.imshow(
        test_images[i].reshape(28, 28),
        cmap='gray'
    )
    plt.axis('off')

    # Reconstructed image
    plt.subplot(2, 10, i + 11)
    plt.imshow(
        reconstructed_images[i].reshape(28, 28),
        cmap='gray'
    )
    plt.axis('off')

plt.suptitle("Original Images (Top) and RBM Reconstructions (Bottom)")
plt.tight_layout()
plt.show()


# ------------------------------------------------
# 9. PLOT FREE ENERGY
# ------------------------------------------------

plt.figure(figsize=(8, 5))

plt.plot(
    range(1, epochs + 1),
    free_energy_history,
    marker='o'
)

plt.xlabel("Epoch")
plt.ylabel("Average Free Energy")
plt.title("RBM Average Free Energy")
plt.grid(True)
plt.show()


# ------------------------------------------------
# 10. PLOT RECONSTRUCTION ERROR
# ------------------------------------------------

plt.figure(figsize=(8, 5))

plt.plot(
    range(1, epochs + 1),
    reconstruction_history,
    marker='o'
)

plt.xlabel("Epoch")
plt.ylabel("Reconstruction Error")
plt.title("RBM Reconstruction Error")
plt.grid(True)
plt.show()


# ------------------------------------------------
# 11. DISPLAY ENERGY OF A SAMPLE
# ------------------------------------------------

sample_v = test_images[0]
sample_h = (
    np.random.rand(n_hidden) < hidden_probability(sample_v)
).astype(np.float32)

print("\nSample RBM Energy:", energy(sample_v, sample_h))

print("\nExperiment 25 Completed Successfully!")

