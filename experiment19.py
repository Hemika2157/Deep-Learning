
import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import fetch_olivetti_faces
from sklearn.decomposition import PCA

# 1. Load Olivetti Faces dataset
print("Loading Olivetti Faces dataset...")

faces = fetch_olivetti_faces(shuffle=True, random_state=42)

X = faces.data
images = faces.images

print("Dataset shape:", X.shape)
print("Number of images:", len(images))
print("Image size:", images.shape[1], "x", images.shape[2])

# 2. Center the data
mean_face = np.mean(X, axis=0)
X_centered = X - mean_face

# 3. PCA from scratch using covariance (dual) matrix
# This uses a 400 x 400 matrix instead of a 4096 x 4096 matrix.
# Both approaches produce the same non-zero principal components.

n_samples = X_centered.shape[0]

cov_matrix = np.dot(
    X_centered, X_centered.T
) / (n_samples - 1)

# Eigenvalue decomposition
eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)

# Sort eigenvalues in descending order
sorted_indices = np.argsort(eigenvalues)[::-1]

eigenvalues = eigenvalues[sorted_indices]
eigenvectors = eigenvectors[:, sorted_indices]

# Remove tiny negative numerical values
eigenvalues = np.maximum(eigenvalues, 0)

# Convert eigenvectors into eigenfaces
# V = X_transpose * U / sqrt((n-1) * eigenvalue)

valid = eigenvalues > 1e-10

eigenvalues = eigenvalues[valid]
eigenvectors = eigenvectors[:, valid]

components = np.dot(
    X_centered.T, eigenvectors
) / np.sqrt(
    (n_samples - 1) * eigenvalues
)

# Normalize eigenfaces
components = components / np.linalg.norm(
    components, axis=0
)

# 4. Explained variance calculation
explained_variance = eigenvalues / np.sum(eigenvalues)

cumulative_variance = np.cumsum(explained_variance)

# Find components needed for 95% variance
components_95 = np.argmax(
    cumulative_variance >= 0.95
) + 1

print("\nPCA FROM SCRATCH")
print("Components for 95% variance:", components_95)
print("Variance retained:", round(
    cumulative_variance[components_95 - 1] * 100, 2
), "%")

# 5. Verify using Scikit-learn PCA
sk_pca = PCA()
sk_pca.fit(X)

print("\nSCIKIT-LEARN PCA")
print("Components for 95% variance:",
      np.argmax(np.cumsum(sk_pca.explained_variance_ratio_) >= 0.95) + 1)

print("Maximum explained variance difference:",
      np.max(np.abs(
          explained_variance -
          sk_pca.explained_variance_ratio_[:len(explained_variance)]
      )))

# 6. Scree plot
plt.figure(figsize=(9, 5))

plt.plot(
    range(1, len(explained_variance) + 1),
    explained_variance,
    marker="."
)

plt.title("PCA Scree Plot")
plt.xlabel("Principal Component")
plt.ylabel("Explained Variance Ratio")
plt.grid()
plt.tight_layout()
plt.show()

# 7. Cumulative explained variance plot
plt.figure(figsize=(9, 5))

plt.plot(
    range(1, len(cumulative_variance) + 1),
    cumulative_variance * 100
)

plt.axhline(
    y=95,
    color="red",
    linestyle="--",
    label="95% Variance"
)

plt.axvline(
    x=components_95,
    color="green",
    linestyle="--",
    label=f"{components_95} Components"
)

plt.title("Cumulative Explained Variance")
plt.xlabel("Number of Components")
plt.ylabel("Cumulative Variance (%)")
plt.legend()
plt.grid()
plt.tight_layout()
plt.show()

# 8. Visualize top eigenfaces
fig, axes = plt.subplots(2, 5, figsize=(12, 5))

for i, ax in enumerate(axes.flat):
    eigenface = components[:, i].reshape(64, 64)

    ax.imshow(eigenface, cmap="gray")
    ax.set_title(f"Eigenface {i + 1}")
    ax.axis("off")

plt.suptitle("Top 10 Eigenfaces")
plt.tight_layout()
plt.show()

# 9. Reconstruct a face using different components

sample_index = 0
original_face = X[sample_index]

component_values = [10, 50, 100]

fig, axes = plt.subplots(
    1, 4, figsize=(14, 4)
)

axes[0].imshow(
    original_face.reshape(64, 64),
    cmap="gray"
)

axes[0].set_title("Original Face")
axes[0].axis("off")

for i, k in enumerate(component_values):

    # Project onto first k principal components
    selected_components = components[:, :k]

    coefficients = np.dot(
        original_face - mean_face,
        selected_components
    )

    reconstructed = np.dot(
        coefficients,
        selected_components.T
    ) + mean_face

    axes[i + 1].imshow(
        reconstructed.reshape(64, 64),
        cmap="gray"
    )

    axes[i + 1].set_title(f"{k} Components")
    axes[i + 1].axis("off")

plt.suptitle("Face Reconstruction Using PCA")
plt.tight_layout()
plt.show()

# 10. Compare reconstruction errors
print("\nRECONSTRUCTION ERROR")

for k in component_values:

    selected_components = components[:, :k]

    coefficients = np.dot(
        X_centered,
        selected_components
    )

    reconstructed = np.dot(
        coefficients,
        selected_components.T
    ) + mean_face

    mse = np.mean((X - reconstructed) ** 2)

    print(f"{k} Components - MSE: {mse:.6f}")

print("\nExperiment 19 completed successfully.")

