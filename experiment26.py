
# Experiment 26: Gibbs Sampling and Contrastive Divergence Analysis

import numpy as np
import matplotlib.pyplot as plt
import time

from scipy.special import expit, log_expit
from scipy.ndimage import zoom
from sklearn.datasets import load_digits


# ------------------------------------------------
# 1. LOAD AND PREPROCESS DATASET
# ------------------------------------------------

np.random.seed(42)

digits = load_digits()

images = digits.images

# Resize 8x8 images to 28x28
resized_images = np.array([
    zoom(img, (3.5, 3.5), order=1)
    for img in images
])

# Normalize and binarize
resized_images = resized_images / 16.0
binary_images = (resized_images > 0.35).astype(np.float32)

# Flatten images into 784 visible units
data = binary_images.reshape(len(binary_images), 784)

# Select 1200 samples
indices = np.random.permutation(len(data))
data = data[indices[:1200]]

print("Dataset shape:", data.shape)


# ------------------------------------------------
# 2. RBM PARAMETERS
# ------------------------------------------------

n_visible = 784
n_hidden = 64

learning_rate = 0.01
epochs = 10
batch_size = 50

methods = {
    "CD-1": 1,
    "CD-5": 5,
    "CD-10": 10,
    "PCD": 1
}


# ------------------------------------------------
# 3. INITIALIZE RBM
# ------------------------------------------------

def initialize_rbm(seed):

    rng = np.random.default_rng(seed)

    W = rng.normal(
        0, 0.01, (n_visible, n_hidden)
    ).astype(np.float32)

    visible_bias = np.zeros(n_visible, dtype=np.float32)
    hidden_bias = np.zeros(n_hidden, dtype=np.float32)

    return W, visible_bias, hidden_bias, rng


# ------------------------------------------------
# 4. HIDDEN AND VISIBLE PROBABILITIES
# ------------------------------------------------

def hidden_probability(v, W, hidden_bias):

    return expit(v @ W + hidden_bias)


def visible_probability(h, W, visible_bias):

    return expit(h @ W.T + visible_bias)


# ------------------------------------------------
# 5. BLOCK GIBBS SAMPLING
# ------------------------------------------------

def gibbs_step(v, W, visible_bias, hidden_bias, rng):

    # Sample hidden layer
    ph = hidden_probability(v, W, hidden_bias)

    h = (
        rng.random(ph.shape) < ph
    ).astype(np.float32)

    # Sample visible layer
    pv = visible_probability(h, W, visible_bias)

    v_new = (
        rng.random(pv.shape) < pv
    ).astype(np.float32)

    return v_new, h, pv


def gibbs_chain(v, steps, W, visible_bias, hidden_bias, rng):

    current_v = v.copy()

    for _ in range(steps):

        current_v, h, pv = gibbs_step(
            current_v,
            W,
            visible_bias,
            hidden_bias,
            rng
        )

    return current_v


# ------------------------------------------------
# 6. FREE ENERGY
# ------------------------------------------------

def free_energy(v, W, visible_bias, hidden_bias):

    hidden_input = v @ W + hidden_bias

    return (
        -v @ visible_bias
        -np.sum(np.logaddexp(0, hidden_input), axis=1)
    )


# ------------------------------------------------
# 7. RECONSTRUCTION ERROR
# ------------------------------------------------

def reconstruction_error(
    samples, W, visible_bias, hidden_bias
):

    ph = hidden_probability(samples, W, hidden_bias)

    pv = visible_probability(ph, W, visible_bias)

    return np.mean((samples - pv) ** 2)


# ------------------------------------------------
# 8. PSEUDO-LOG-LIKELIHOOD
# ------------------------------------------------

def pseudo_log_likelihood(
    samples, W, visible_bias, hidden_bias, rng
):

    # Randomly select one visible bit per sample
    bit_indices = rng.integers(0, n_visible, size=len(samples))

    flipped = samples.copy()

    rows = np.arange(len(samples))

    flipped[rows, bit_indices] = (
        1 - flipped[rows, bit_indices]
    )

    original_energy = free_energy(
        samples, W, visible_bias, hidden_bias
    )

    flipped_energy = free_energy(
        flipped, W, visible_bias, hidden_bias
    )

    # Estimate full pseudo-log-likelihood
    values = n_visible * log_expit(
        flipped_energy - original_energy
    )

    return np.mean(values)


# ------------------------------------------------
# 9. TRAIN RBM USING CD-k AND PCD
# ------------------------------------------------

def train_rbm(method, k, seed):

    W, visible_bias, hidden_bias, rng = initialize_rbm(seed)

    # Persistent chain for PCD
    persistent_v = (
        rng.random((batch_size, n_visible)) < 0.5
    ).astype(np.float32)

    error_history = []
    pll_history = []

    start_time = time.time()

    for epoch in range(epochs):

        shuffled = data[rng.permutation(len(data))]

        for start in range(
            0, len(shuffled) - batch_size + 1, batch_size
        ):

            v0 = shuffled[start:start + batch_size]

            # Positive phase
            ph0 = hidden_probability(v0, W, hidden_bias)

            # CD-k or PCD negative phase
            if method == "PCD":

                # Continue the persistent chain
                vk = persistent_v

                vk = gibbs_chain(
                    vk, 1, W, visible_bias, hidden_bias, rng
                )

                persistent_v = vk.copy()

            else:

                # Start the chain from the training data
                vk = v0.copy()

                vk = gibbs_chain(
                    vk, k, W, visible_bias, hidden_bias, rng
                )

            phk = hidden_probability(vk, W, hidden_bias)

            # Update weights and biases
            W += learning_rate * (
                (v0.T @ ph0 - vk.T @ phk) / batch_size
            )

            visible_bias += learning_rate * np.mean(
                v0 - vk, axis=0
            )

            hidden_bias += learning_rate * np.mean(
                ph0 - phk, axis=0
            )

        # Evaluate after each epoch
        error = reconstruction_error(
            data[:200], W, visible_bias, hidden_bias
        )

        pll = pseudo_log_likelihood(
            data[:200], W, visible_bias, hidden_bias, rng
        )

        error_history.append(error)
        pll_history.append(pll)

        print(
            f"{method} | Epoch {epoch + 1}/{epochs} | "
            f"Reconstruction Error: {error:.4f} | "
            f"Pseudo-Log-Likelihood: {pll:.2f}"
        )

    training_time = time.time() - start_time

    return {
        "W": W,
        "visible_bias": visible_bias,
        "hidden_bias": hidden_bias,
        "errors": error_history,
        "pll": pll_history,
        "time": training_time
    }


# ------------------------------------------------
# 10. RUN ALL FOUR TRAINING METHODS
# ------------------------------------------------

results = {}

print("\nTRAINING RBM MODELS")

for index, (method, k) in enumerate(methods.items()):

    print("\nTraining method:", method)

    results[method] = train_rbm(
        method,
        k,
        seed=100 + index
    )


# ------------------------------------------------
# 11. COMPARE TRAINING RESULTS
# ------------------------------------------------

comparison = []

for method, result in results.items():

    comparison.append({
        "Method": method,
        "Training Time (sec)": result["time"],
        "Reconstruction Error": result["errors"][-1],
        "Pseudo-Log-Likelihood": result["pll"][-1]
    })

print("\nMETHOD COMPARISON")

for row in comparison:
    print(
        f"{row['Method']:6s} | "
        f"Time: {row['Training Time (sec)']:.2f}s | "
        f"Error: {row['Reconstruction Error']:.4f} | "
        f"Pseudo-LL: {row['Pseudo-Log-Likelihood']:.2f}"
    )


# ------------------------------------------------
# 12. GIBBS SAMPLING FROM RANDOM NOISE
# ------------------------------------------------

# Use trained CD-1 model
model = results["CD-1"]

W = model["W"]
visible_bias = model["visible_bias"]
hidden_bias = model["hidden_bias"]

rng = np.random.default_rng(123)

# Start with random binary noise
initial_v = (
    rng.random((1, n_visible)) < 0.5
).astype(np.float32)

steps_to_display = [1, 10, 100, 1000]

current_v = initial_v.copy()
gibbs_samples = []

for step in range(1, max(steps_to_display) + 1):

    current_v = gibbs_chain(
        current_v, 1, W, visible_bias, hidden_bias, rng
    )

    if step in steps_to_display:
        gibbs_samples.append(current_v.copy())


# Display Gibbs chain progression
plt.figure(figsize=(12, 3))

for i, sample in enumerate(gibbs_samples):

    plt.subplot(1, 4, i + 1)

    plt.imshow(
        sample.reshape(28, 28),
        cmap="gray"
    )

    plt.title(f"{steps_to_display[i]} Steps")
    plt.axis("off")

plt.suptitle("Gibbs Sampling from Random Noise")
plt.tight_layout()
plt.show()


# ------------------------------------------------
# 13. PLOT RECONSTRUCTION ERROR
# ------------------------------------------------

plt.figure(figsize=(9, 5))

for method, result in results.items():

    plt.plot(
        range(1, epochs + 1),
        result["errors"],
        marker="o",
        label=method
    )

plt.xlabel("Epoch")
plt.ylabel("Reconstruction Error")
plt.title("Reconstruction Error Comparison")
plt.legend()
plt.grid(True)
plt.show()


# ------------------------------------------------
# 14. PLOT PSEUDO-LOG-LIKELIHOOD
# ------------------------------------------------

plt.figure(figsize=(9, 5))

for method, result in results.items():

    plt.plot(
        range(1, epochs + 1),
        result["pll"],
        marker="o",
        label=method
    )

plt.xlabel("Epoch")
plt.ylabel("Pseudo-Log-Likelihood")
plt.title("Pseudo-Log-Likelihood Comparison")
plt.legend()
plt.grid(True)
plt.show()


# ------------------------------------------------
# 15. TRAINING TIME VS SAMPLE QUALITY
# ------------------------------------------------

method_names = list(results.keys())

training_times = [
    results[m]["time"] for m in method_names
]

final_errors = [
    results[m]["errors"][-1] for m in method_names
]

plt.figure(figsize=(8, 5))

plt.scatter(training_times, final_errors, s=100)

for i, method in enumerate(method_names):

    plt.annotate(
        method,
        (training_times[i], final_errors[i]),
        xytext=(5, 5),
        textcoords="offset points"
    )

plt.xlabel("Training Time (seconds)")
plt.ylabel("Reconstruction Error")
plt.title("Training Time vs Sample Quality")
plt.grid(True)
plt.show()


# ------------------------------------------------
# 16. DISPLAY GENERATED SAMPLES FOR EACH METHOD
# ------------------------------------------------

plt.figure(figsize=(10, 5))

for i, method in enumerate(method_names):

    result = results[method]

    W = result["W"]
    visible_bias = result["visible_bias"]
    hidden_bias = result["hidden_bias"]

    rng = np.random.default_rng(500 + i)

    start = (
        rng.random((1, n_visible)) < 0.5
    ).astype(np.float32)

    generated = gibbs_chain(
        start, 100, W, visible_bias, hidden_bias, rng
    )

    plt.subplot(1, 4, i + 1)

    plt.imshow(
        generated.reshape(28, 28),
        cmap="gray"
    )

    plt.title(method)
    plt.axis("off")

plt.suptitle("Generated Samples from Different RBM Methods")
plt.tight_layout()
plt.show()


print("\nExperiment 26 Completed Successfully!")

