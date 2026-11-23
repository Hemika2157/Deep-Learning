
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_squared_error

# 1. Generate 150 house area samples
np.random.seed(42)

X = np.random.uniform(500, 3000, 150).reshape(-1, 1)

# Scale area between 0 and 1
x_scaled = X / 3000

# Hidden non-linear relationship
def true_function(x):
    return 30 + 45*x + 20*x**2 + 12*np.sin(2*np.pi*x)

y = true_function(x_scaled.ravel()) + np.random.normal(0, 5, 150)

# 2. Split data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(
    x_scaled, y, test_size=0.2, random_state=42
)

degrees = [1, 3, 9, 15]

train_errors = []
val_errors = []
models = {}

# 3. Fit polynomial regression models
plt.figure(figsize=(12, 8))

plt.scatter(x_scaled, y, color='black',
            alpha=0.5, label='House Data')

x_plot = np.linspace(0, 1, 300).reshape(-1, 1)

for degree in degrees:

    model = make_pipeline(
        PolynomialFeatures(degree),
        LinearRegression()
    )

    model.fit(X_train, y_train)
    models[degree] = model

    train_pred = model.predict(X_train)
    val_pred = model.predict(X_val)

    train_mse = mean_squared_error(y_train, train_pred)
    val_mse = mean_squared_error(y_val, val_pred)

    train_errors.append(train_mse)
    val_errors.append(val_mse)

    y_plot = model.predict(x_plot)

    plt.plot(x_plot, y_plot,
             label=f'Degree {degree}')

    print("\nPolynomial Degree:", degree)
    print("Training Error:", round(train_mse, 3))
    print("Validation Error:", round(val_mse, 3))

plt.xlabel("Scaled House Area")
plt.ylabel("House Price (Lakhs)")
plt.title("Polynomial Regression: Underfitting and Overfitting")
plt.legend()
plt.grid()
plt.show()

# 4. Plot training and validation errors
plt.figure(figsize=(8, 5))

plt.plot(degrees, train_errors, 'o-',
         label='Training Error')

plt.plot(degrees, val_errors, 's-',
         label='Validation Error')

plt.xlabel("Polynomial Degree")
plt.ylabel("Mean Squared Error")
plt.title("Training and Validation Error vs Model Complexity")
plt.xticks(degrees)
plt.legend()
plt.grid()
plt.show()

# 5. Estimate Bias and Variance using repeated resampling
np.random.seed(42)

x_grid = np.linspace(0, 1, 100).reshape(-1, 1)
true_values = true_function(x_grid.ravel())

resampling_results = {}

for degree in degrees:

    predictions = []

    for i in range(100):

        # Randomly select training samples with replacement
        indices = np.random.choice(
            len(X_train), len(X_train), replace=True
        )

        X_sample = X_train[indices]
        y_sample = y_train[indices]

        model = make_pipeline(
            PolynomialFeatures(degree),
            LinearRegression()
        )

        model.fit(X_sample, y_sample)

        predictions.append(model.predict(x_grid))

    predictions = np.array(predictions)

    mean_prediction = np.mean(predictions, axis=0)

    bias_squared = np.mean(
        (mean_prediction - true_values)**2
    )

    variance = np.mean(
        np.var(predictions, axis=0)
    )

    resampling_results[degree] = (
        bias_squared, variance
    )

    print("\nDegree:", degree)
    print("Bias Squared:", round(bias_squared, 3))
    print("Variance:", round(variance, 3))

# 6. Plot Bias Squared and Variance
bias_values = [
    resampling_results[d][0] for d in degrees
]

variance_values = [
    resampling_results[d][1] for d in degrees
]

plt.figure(figsize=(8, 5))

plt.plot(degrees, bias_values, 'o-',
         label='Bias Squared')

plt.plot(degrees, variance_values, 's-',
         label='Variance')

plt.xlabel("Polynomial Degree")
plt.ylabel("Error")
plt.title("Bias-Variance Analysis")
plt.xticks(degrees)
plt.legend()
plt.grid()
plt.show()

# 7. Identify the best model
best_index = np.argmin(val_errors)
best_degree = degrees[best_index]

print("\nBEST MODEL")
print("Best Polynomial Degree:", best_degree)
print("Lowest Validation Error:",
      round(val_errors[best_index], 3))

print("\nExperiment 5 Completed Successfully!")
