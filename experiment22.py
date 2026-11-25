
import numpy as np
import matplotlib.pyplot as plt

from scipy.signal import sawtooth
from sklearn.decomposition import FastICA, PCA

np.random.seed(42)

# 1. Generate three original source signals
n_samples = 2000
time = np.linspace(0, 8, n_samples)

# Source 1: Sine wave
source1 = np.sin(2 * time)

# Source 2: Square wave
source2 = np.sign(np.sin(3 * time))

# Source 3: Sawtooth wave
source3 = sawtooth(2 * np.pi * 0.5 * time)

# Combine all sources
S = np.c_[source1, source2, source3]

# Standardize source signals
S = (S - S.mean(axis=0)) / S.std(axis=0)

print("Original source shape:", S.shape)

# 2. Mix the source signals
mixing_matrix = np.array([
    [1.0, 0.5, 0.8],
    [0.4, 1.0, 0.3],
    [0.6, 0.2, 1.0]
])

X = np.dot(S, mixing_matrix.T)

print("\nMixing Matrix:")
print(mixing_matrix)

print("\nMixed signal shape:", X.shape)

# 3. Apply FastICA
ica = FastICA(
    n_components=3,
    random_state=42,
    max_iter=2000,
    tol=0.0001,
    whiten="unit-variance"
)

S_ica = ica.fit_transform(X)

print("\nFastICA completed.")

# 4. Apply PCA for comparison
pca = PCA(n_components=3)

S_pca = pca.fit_transform(X)

print("PCA completed.")

# 5. Calculate correlation-based recovery quality
def calculate_correlations(original, recovered):

    correlation_matrix = np.zeros(
        (original.shape[1], recovered.shape[1])
    )

    for i in range(original.shape[1]):
        for j in range(recovered.shape[1]):

            correlation_matrix[i, j] = np.corrcoef(
                original[:, i],
                recovered[:, j]
            )[0, 1]

    return correlation_matrix


ica_corr = calculate_correlations(S, S_ica)
pca_corr = calculate_correlations(S, S_pca)

print("\nICA CORRELATION MATRIX")
print(np.round(ica_corr, 3))

print("\nPCA CORRELATION MATRIX")
print(np.round(pca_corr, 3))

# Match recovered components with original sources
# ICA can recover signals in any order and with either sign.
ica_quality = np.mean(
    np.max(np.abs(ica_corr), axis=1)
)

pca_quality = np.mean(
    np.max(np.abs(pca_corr), axis=1)
)

print("\nRECOVERY QUALITY")
print("ICA Mean Absolute Correlation:",
      round(ica_quality, 4))

print("PCA Mean Absolute Correlation:",
      round(pca_quality, 4))

# 6. Plot original source signals
fig, axes = plt.subplots(3, 1, figsize=(12, 8))

for i in range(3):
    axes[i].plot(time, S[:, i])
    axes[i].set_title(f"Original Source {i+1}")
    axes[i].set_xlabel("Time")
    axes[i].set_ylabel("Amplitude")
    axes[i].grid()

plt.suptitle("Original Independent Signals")
plt.tight_layout()
plt.show()

# 7. Plot mixed signals
fig, axes = plt.subplots(3, 1, figsize=(12, 8))

for i in range(3):
    axes[i].plot(time, X[:, i])
    axes[i].set_title(f"Mixed Signal {i+1}")
    axes[i].set_xlabel("Time")
    axes[i].set_ylabel("Amplitude")
    axes[i].grid()

plt.suptitle("Mixed Signals")
plt.tight_layout()
plt.show()

# 8. Plot ICA recovered signals
fig, axes = plt.subplots(3, 1, figsize=(12, 8))

for i in range(3):
    axes[i].plot(time, S_ica[:, i])
    axes[i].set_title(f"ICA Recovered Component {i+1}")
    axes[i].set_xlabel("Time")
    axes[i].set_ylabel("Amplitude")
    axes[i].grid()

plt.suptitle("FastICA Recovered Signals")
plt.tight_layout()
plt.show()

# 9. Plot PCA recovered components
fig, axes = plt.subplots(3, 1, figsize=(12, 8))

for i in range(3):
    axes[i].plot(time, S_pca[:, i])
    axes[i].set_title(f"PCA Component {i+1}")
    axes[i].set_xlabel("Time")
    axes[i].set_ylabel("Amplitude")
    axes[i].grid()

plt.suptitle("PCA Components")
plt.tight_layout()
plt.show()

# 10. Compare correlation quality
plt.figure(figsize=(8, 5))

plt.bar(
    ["FastICA", "PCA"],
    [ica_quality, pca_quality]
)

plt.title("Source Recovery Quality Comparison")
plt.ylabel("Mean Absolute Correlation")
plt.ylim(0, 1.1)
plt.grid(axis="y")
plt.tight_layout()
plt.show()

print("\nExperiment 22 completed successfully.")

