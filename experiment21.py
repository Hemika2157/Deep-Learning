
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.decomposition import FactorAnalysis, PCA
from sklearn.preprocessing import StandardScaler

np.random.seed(42)

# 1. Generate synthetic employee survey data
print("Generating employee survey data...")

n_employees = 300

# Four hidden satisfaction factors
compensation = np.random.normal(size=n_employees)
work_culture = np.random.normal(size=n_employees)
management = np.random.normal(size=n_employees)
career_growth = np.random.normal(size=n_employees)

# Ten survey questions
questions = [
    "Salary Satisfaction",
    "Benefits Satisfaction",
    "Workload Balance",
    "Team Cooperation",
    "Work Environment",
    "Manager Support",
    "Communication",
    "Promotion Opportunities",
    "Skill Development",
    "Career Growth"
]

# Generate responses based on latent factors
X = np.column_stack([
    3 + 0.9*compensation + 0.1*work_culture,
    3 + 0.8*compensation + 0.2*management,
    3 + 0.8*work_culture - 0.1*career_growth,
    3 + 0.9*work_culture + 0.1*management,
    3 + 0.8*work_culture + 0.2*compensation,
    3 + 0.9*management + 0.1*work_culture,
    3 + 0.8*management + 0.2*career_growth,
    3 + 0.9*career_growth + 0.1*compensation,
    3 + 0.8*career_growth + 0.2*management,
    3 + 0.9*career_growth + 0.1*work_culture
])

# Add random noise
X += np.random.normal(0, 0.5, X.shape)

# Convert responses to Likert scale (1 to 5)
X = np.clip(np.rint(X), 1, 5).astype(int)

df = pd.DataFrame(X, columns=questions)

# 2. Explore dataset
print("\nDATASET INFORMATION")
print(df.head())

print("\nDataset shape:", df.shape)

print("\nDescriptive statistics:")
print(df.describe())

print("\nCorrelation matrix:")
print(df.corr().round(2))

# 3. Correlation heatmap
plt.figure(figsize=(11, 8))

sns.heatmap(
    df.corr(),
    annot=True,
    cmap="coolwarm",
    vmin=-1,
    vmax=1
)

plt.title("Employee Satisfaction Correlation Matrix")
plt.tight_layout()
plt.show()

# 4. Standardize data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df)

# 5. Fit Factor Analysis with 2, 3 and 4 factors
factor_models = {}
factor_loadings = {}

for n_factors in [2, 3, 4]:

    print("\n================================")
    print("FACTOR ANALYSIS:", n_factors, "FACTORS")
    print("================================")

    fa = FactorAnalysis(
        n_components=n_factors,
        random_state=42,
        max_iter=1000
    )

    fa.fit(X_scaled)

    # sklearn loadings: features x factors
    loadings = fa.components_.T

    factor_models[n_factors] = fa
    factor_loadings[n_factors] = loadings

    loading_df = pd.DataFrame(
        loadings,
        index=questions,
        columns=[
            f"Factor {i+1}" for i in range(n_factors)
        ]
    )

    print("\nFactor Loadings:")
    print(loading_df.round(3))

    # Display loadings heatmap
    plt.figure(figsize=(9, 7))

    sns.heatmap(
        loading_df,
        annot=True,
        cmap="coolwarm",
        center=0,
        fmt=".2f"
    )

    plt.title(f"Factor Loadings - {n_factors} Factors")
    plt.tight_layout()
    plt.show()


# 6. Apply Varimax rotation
def varimax(Phi, gamma=1.0, max_iter=100, tol=1e-6):

    p, k = Phi.shape
    rotation = np.eye(k)
    previous_value = 0

    for i in range(max_iter):

        rotated = np.dot(Phi, rotation)

        u, singular_values, vh = np.linalg.svd(
            np.dot(
                Phi.T,
                rotated**3 -
                (gamma / p) * rotated @
                np.diag(np.diag(rotated.T @ rotated))
            )
        )

        rotation = u @ vh

        current_value = np.sum(singular_values)

        if previous_value != 0:
            if current_value - previous_value < tol:
                break

        previous_value = current_value

    return Phi @ rotation


print("\nVARIMAX ROTATION")

# Use 4 factors for detailed interpretation
selected_model = factor_models[4]

unrotated_loadings = selected_model.components_.T

rotated_loadings = varimax(unrotated_loadings)

rotated_df = pd.DataFrame(
    rotated_loadings,
    index=questions,
    columns=[
        "Factor 1",
        "Factor 2",
        "Factor 3",
        "Factor 4"
    ]
)

print("\nRotated Factor Loadings:")
print(rotated_df.round(3))

# Varimax heatmap
plt.figure(figsize=(10, 7))

sns.heatmap(
    rotated_df,
    annot=True,
    cmap="coolwarm",
    center=0,
    fmt=".2f"
)

plt.title("Varimax Rotated Factor Loadings")
plt.tight_layout()
plt.show()

# 7. Interpret and name factors
factor_names = [
    "Compensation",
    "Work Culture",
    "Management Support",
    "Career Growth"
]

print("\nFACTOR INTERPRETATION")

for i, name in enumerate(factor_names):

    print("\n", name)

    sorted_questions = rotated_df.iloc[:, i].abs().sort_values(
        ascending=False
    )

    for question in sorted_questions.index[:3]:
        print(
            question,
            "Loading:",
            round(rotated_df.loc[question, rotated_df.columns[i]], 3)
        )

# 8. Compare Factor Analysis and PCA
fa_model = FactorAnalysis(
    n_components=4,
    random_state=42,
    max_iter=1000
)

fa_model.fit(X_scaled)

pca_model = PCA(n_components=4)
pca_model.fit(X_scaled)

# Calculate reconstruction errors
fa_scores = fa_model.transform(X_scaled)

fa_reconstructed = (
    fa_scores @ fa_model.components_ +
    fa_model.mean_
)

pca_scores = pca_model.transform(X_scaled)
pca_reconstructed = pca_model.inverse_transform(
    pca_scores
)

fa_mse = np.mean(
    (X_scaled - fa_reconstructed) ** 2
)

pca_mse = np.mean(
    (X_scaled - pca_reconstructed) ** 2
)

print("\nFACTOR ANALYSIS VS PCA")
print("Factor Analysis Reconstruction MSE:", round(fa_mse, 4))
print("PCA Reconstruction MSE:", round(pca_mse, 4))

# Compare reconstruction errors
plt.figure(figsize=(7, 5))

plt.bar(
    ["Factor Analysis", "PCA"],
    [fa_mse, pca_mse]
)

plt.title("Factor Analysis vs PCA")
plt.ylabel("Reconstruction MSE")
plt.tight_layout()
plt.show()

# 9. Final result
print("\nExperiment 21 completed successfully.")

