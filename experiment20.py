
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

np.random.seed(42)

# 1. Generate synthetic sensor data
print("Generating sensor readings...")

n_samples = 500

# Two hidden factors affecting sensor measurements
z1 = np.random.normal(0, 1, n_samples)
z2 = np.random.normal(0, 1, n_samples)

# Sensor readings: temperature, vibration, pressure,
# current and humidity
X_clean = np.column_stack([
    30 + 5*z1 + 2*z2,
    10 + 2*z1 + 4*z2,
    100 + 8*z1 - 3*z2,
    20 + 3*z1 + 5*z2,
    50 - 4*z1 + 2*z2
])

# Add Gaussian measurement noise
noise_std = 1.5
X_noisy = X_clean + np.random.normal(
    0, noise_std, X_clean.shape
)

# Introduce occasional missing values
missing_mask = np.random.rand(*X_noisy.shape) < 0.03
X_missing = X_noisy.copy()
X_missing[missing_mask] = np.nan

columns = [
    "Temperature",
    "Vibration",
    "Pressure",
    "Current",
    "Humidity"
]

print("\nMissing values:", np.isnan(X_missing).sum())

# 2. Handle missing values
imputer = SimpleImputer(strategy="mean")
X_imputed = imputer.fit_transform(X_missing)

# Standardize data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_imputed)

print("Dataset shape:", X_scaled.shape)

# 3. Fit Probabilistic PCA
# Scikit-learn PCA estimates residual noise variance
n_components = 2

ppca = PCA(
    n_components=n_components,
    svd_solver="full"
)

Z = ppca.fit_transform(X_scaled)

# Estimated noise variance
sigma_squared = ppca.noise_variance_

print("\nPROBABILISTIC PCA")
print("Number of components:", n_components)
print("Estimated noise variance:", round(
    sigma_squared, 6
))
print("Explained variance ratio:",
      ppca.explained_variance_ratio_)

# 4. Fit standard PCA for comparison
standard_pca = PCA(n_components=n_components)

Z_standard = standard_pca.fit_transform(X_scaled)

# Reconstruct data
X_reconstructed_ppca = ppca.inverse_transform(Z)
X_reconstructed_pca = standard_pca.inverse_transform(
    Z_standard
)

# Calculate reconstruction errors
ppca_mse = np.mean(
    (X_scaled - X_reconstructed_ppca) ** 2
)

pca_mse = np.mean(
    (X_scaled - X_reconstructed_pca) ** 2
)

print("\nRECONSTRUCTION COMPARISON")
print("PPCA Reconstruction MSE:", round(ppca_mse, 6))
print("Standard PCA Reconstruction MSE:", round(
    pca_mse, 6
))

# 5. Compare reconstruction error for different components
component_values = [1, 2, 3, 4]

ppca_errors = []
pca_errors = []
noise_estimates = []

for k in component_values:

    # Probabilistic PCA
    model_ppca = PCA(
        n_components=k,
        svd_solver="full"
    )

    transformed_ppca = model_ppca.fit_transform(X_scaled)

    reconstructed_ppca = model_ppca.inverse_transform(
        transformed_ppca
    )

    error_ppca = np.mean(
        (X_scaled - reconstructed_ppca) ** 2
    )

    ppca_errors.append(error_ppca)
    noise_estimates.append(model_ppca.noise_variance_)

    # Standard PCA
    model_pca = PCA(n_components=k)

    transformed_pca = model_pca.fit_transform(X_scaled)

    reconstructed_pca = model_pca.inverse_transform(
        transformed_pca
    )

    error_pca = np.mean(
        (X_scaled - reconstructed_pca) ** 2
    )

    pca_errors.append(error_pca)

# Plot reconstruction errors
plt.figure(figsize=(8, 5))

plt.plot(
    component_values,
    ppca_errors,
    marker="o",
    label="PPCA"
)

plt.plot(
    component_values,
    pca_errors,
    marker="s",
    label="Standard PCA"
)

plt.xlabel("Number of Components")
plt.ylabel("Reconstruction MSE")
plt.title("PPCA vs Standard PCA Reconstruction Error")
plt.legend()
plt.grid()
plt.tight_layout()
plt.show()

# 6. Plot estimated noise variance
plt.figure(figsize=(8, 5))

plt.plot(
    component_values,
    noise_estimates,
    marker="o"
)

plt.xlabel("Number of Components")
plt.ylabel("Estimated Noise Variance")
plt.title("PPCA Noise Variance Estimation")
plt.grid()
plt.tight_layout()
plt.show()

# 7. Generate new synthetic samples from fitted PPCA model
# PPCA model: x = Wz + mean + epsilon
#
# Estimate W using eigenvectors and eigenvalues:
# W = U * sqrt(Lambda - sigma_squared)

eigenvalues = ppca.explained_variance_
eigenvectors = ppca.components_.T

adjusted_values = np.maximum(
    eigenvalues - sigma_squared, 0
)

W = eigenvectors * np.sqrt(adjusted_values)

# Generate latent variables and Gaussian noise
n_new = 100

latent_z = np.random.normal(
    0, 1, (n_new, n_components)
)

gaussian_noise = np.random.normal(
    0,
    np.sqrt(sigma_squared),
    (n_new, X_scaled.shape[1])
)

# Generate new samples
new_samples_scaled = (
    latent_z @ W.T +
    ppca.mean_ +
    gaussian_noise
)

# Convert back to original sensor units
new_samples = scaler.inverse_transform(
    new_samples_scaled
)

new_df = pd.DataFrame(
    new_samples,
    columns=columns
)

print("\nGENERATED SENSOR SAMPLES")
print(new_df.head(10).round(2))

# 8. Visualize original and reconstructed readings
plt.figure(figsize=(10, 5))

sensor_index = 0

plt.plot(
    X_scaled[:50, sensor_index],
    label="Original"
)

plt.plot(
    X_reconstructed_ppca[:50, sensor_index],
    label="PPCA Reconstructed"
)

plt.title("Original vs PPCA Reconstructed Temperature")
plt.xlabel("Sample Number")
plt.ylabel("Standardized Reading")
plt.legend()
plt.grid()
plt.tight_layout()
plt.show()

print("\nExperiment 20 completed successfully.")

