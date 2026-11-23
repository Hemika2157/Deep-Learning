
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 1. Generate sample data for 200 patients
np.random.seed(42)

haemoglobin = np.random.normal(13.5, 1.5, 200)
rbc = 4.5 + 0.3 * (haemoglobin - 13.5) + np.random.normal(0, 0.3, 200)
glucose = np.random.normal(110, 20, 200)
cholesterol = 180 + 0.5 * glucose + np.random.normal(0, 15, 200)
bmi = np.random.normal(25, 4, 200)

# 2. Create Pandas DataFrame
data = pd.DataFrame({
    'Haemoglobin': haemoglobin,
    'RBC': rbc,
    'Glucose': glucose,
    'Cholesterol': cholesterol,
    'BMI': bmi
})

print("PATIENT DATA:")
print(data.head())

# 3. Standardisation
standardized = (data - data.mean()) / data.std(ddof=0)

print("\nSTANDARDISED DATA:")
print(standardized.head())

# 4. Covariance Matrix
cov_matrix = standardized.cov()

print("\nCOVARIANCE MATRIX:")
print(cov_matrix)

# 5. Correlation Matrix
corr_matrix = standardized.corr()

print("\nCORRELATION MATRIX:")
print(corr_matrix)

# 6. Covariance Heatmap
plt.figure(figsize=(7, 5))
plt.imshow(cov_matrix, cmap='coolwarm', vmin=-1, vmax=1)
plt.colorbar(label='Covariance')
plt.xticks(range(5), data.columns, rotation=45)
plt.yticks(range(5), data.columns)
plt.title("Covariance Matrix Heatmap")

for i in range(5):
    for j in range(5):
        plt.text(j, i, f"{cov_matrix.iloc[i, j]:.2f}",
                 ha='center', va='center')

plt.tight_layout()
plt.show()

# 7. Correlation Heatmap
plt.figure(figsize=(7, 5))
plt.imshow(corr_matrix, cmap='coolwarm', vmin=-1, vmax=1)
plt.colorbar(label='Correlation')
plt.xticks(range(5), data.columns, rotation=45)
plt.yticks(range(5), data.columns)
plt.title("Correlation Matrix Heatmap")

for i in range(5):
    for j in range(5):
        plt.text(j, i, f"{corr_matrix.iloc[i, j]:.2f}",
                 ha='center', va='center')

plt.tight_layout()
plt.show()

# 8. Eigenvalues and Eigenvectors
eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)

# Sort in descending order
indices = np.argsort(eigenvalues)[::-1]

eigenvalues = eigenvalues[indices]
eigenvectors = eigenvectors[:, indices]

print("\nEIGENVALUES:")
print(eigenvalues)

print("\nEIGENVECTORS:")
print(eigenvectors)

# 9. Percentage of Variance
variance_percentage = (
    eigenvalues / np.sum(eigenvalues)
) * 100

print("\nVARIANCE CAPTURED BY EACH EIGENVALUE:")
for i, value in enumerate(variance_percentage):
    print(f"PC{i+1}: {value:.2f}%")

print("\nCumulative Variance of First Two Components:",
      round(np.sum(variance_percentage[:2]), 2), "%")

# 10. Project data onto first two principal components
projected = standardized.values @ eigenvectors[:, :2]

# 11. Plot 2-D data with top two eigenvectors
plt.figure(figsize=(8, 6))

plt.scatter(projected[:, 0], projected[:, 1],
            alpha=0.6, label="Patient Data")

plt.arrow(0, 0, 3, 0,
          head_width=0.2, length_includes_head=True,
          label="PC1")

plt.arrow(0, 0, 0, 3,
          head_width=0.2, length_includes_head=True,
          label="PC2")

plt.xlabel("First Principal Component (PC1)")
plt.ylabel("Second Principal Component (PC2)")
plt.title("2-D PCA Projection of Patient Data")
plt.legend()
plt.grid()
plt.show()

print("\nExperiment 2 Completed Successfully!")
