
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import bernoulli, binom, poisson, uniform, norm

np.random.seed(42)

# 1. Generate probability distribution samples
n = 100000

bernoulli_data = bernoulli.rvs(0.5, size=n)
binomial_data = binom.rvs(10, 0.5, size=n)
poisson_data = poisson.rvs(5, size=n)
uniform_data = uniform.rvs(0, 10, size=n)
gaussian_data = norm.rvs(50, 10, size=n)

# 2. Plot distributions
fig, axes = plt.subplots(2, 3, figsize=(14, 8))

# Bernoulli
x = np.array([0, 1])
axes[0, 0].hist(bernoulli_data, bins=[-0.5, 0.5, 1.5],
                density=True, alpha=0.6)
axes[0, 0].plot(x, bernoulli.pmf(x, 0.5), 'ro')
axes[0, 0].set_title("Bernoulli Distribution")

# Binomial
x = np.arange(0, 11)
axes[0, 1].hist(binomial_data, bins=np.arange(-0.5, 11.5, 1),
                density=True, alpha=0.6)
axes[0, 1].plot(x, binom.pmf(x, 10, 0.5), 'ro')
axes[0, 1].set_title("Binomial Distribution")

# Poisson
x = np.arange(0, 16)
axes[0, 2].hist(poisson_data, bins=np.arange(-0.5, 16.5, 1),
                density=True, alpha=0.6)
axes[0, 2].plot(x, poisson.pmf(x, 5), 'ro')
axes[0, 2].set_title("Poisson Distribution")

# Uniform
x = np.linspace(0, 10, 100)
axes[1, 0].hist(uniform_data, bins=30, density=True, alpha=0.6)
axes[1, 0].plot(x, uniform.pdf(x, 0, 10), 'r')
axes[1, 0].set_title("Uniform Distribution")

# Gaussian
x = np.linspace(10, 90, 100)
axes[1, 1].hist(gaussian_data, bins=30, density=True, alpha=0.6)
axes[1, 1].plot(x, norm.pdf(x, 50, 10), 'r')
axes[1, 1].set_title("Gaussian Distribution")

axes[1, 2].axis("off")

plt.tight_layout()
plt.show()

# 3. Bayes' Theorem
prevalence = 0.01
sensitivity = 0.95
specificity = 0.90

false_positive_rate = 1 - specificity

numerator = sensitivity * prevalence

denominator = (sensitivity * prevalence) + (
    false_positive_rate * (1 - prevalence)
)

posterior = numerator / denominator

print("\nBAYES' THEOREM RESULT")
print("Disease Prevalence:", prevalence * 100, "%")
print("Sensitivity:", sensitivity * 100, "%")
print("Specificity:", specificity * 100, "%")
print("Probability of Disease Given Positive:",
      round(posterior * 100, 2), "%")

# 4. Simulate 100,000 patients
disease = np.random.binomial(1, prevalence, n)

test_positive = np.where(
    disease == 1,
    np.random.binomial(1, sensitivity, n),
    np.random.binomial(1, false_positive_rate, n)
)

actual_positive = np.sum(
    (disease == 1) & (test_positive == 1)
)

total_positive = np.sum(test_positive)

empirical_probability = actual_positive / total_positive

print("\nSIMULATION RESULT")
print("Total Patients:", n)
print("Patients with Disease:", np.sum(disease))
print("Positive Test Results:", total_positive)
print("True Positive Results:", actual_positive)
print("Empirical Probability:",
      round(empirical_probability * 100, 2), "%")

# 5. Posterior probability for different prevalence values
prevalences = np.linspace(0.001, 0.20, 100)

posteriors = (
    sensitivity * prevalences
) / (
    sensitivity * prevalences +
    false_positive_rate * (1 - prevalences)
)

plt.figure(figsize=(8, 5))
plt.plot(prevalences * 100, posteriors * 100, linewidth=2)

plt.scatter(1, posterior * 100, color='red',
            label="Given prevalence = 1%")

plt.xlabel("Disease Prevalence (%)")
plt.ylabel("Probability of Disease Given Positive (%)")
plt.title("Effect of Disease Prevalence on Posterior Probability")
plt.legend()
plt.grid()
plt.show()

print("\nExperiment 3 Completed Successfully!")
