import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV

# Load the data from an Excel file
data = pd.read_excel('example for logistic regression.xlsx')  # Replace 'example for logistic regression.xlsx' with the actual file path

# Extract features (X) and labels (y)
X = data.iloc[:, :-1].values #selected all rows and all columns except the last column
y = data.iloc[:, -1].values

# Create a polynomial logistic regression model
polyreg = make_pipeline(PolynomialFeatures(), LogisticRegression(max_iter=10000))  # No degree set initially

# Define a grid of degrees to search
param_grid = {
    'polynomialfeatures__degree': [1, 2, 3, 4, 5],  # You can adjust the range of degrees
}

# Perform grid search with cross-validation
grid_search = GridSearchCV(polyreg, param_grid, cv=5)  # You can adjust the number of cross-validation folds
#polyreg is the model to be tuned, param_grid is the grid of hyperparameters to search
grid_search.fit(X, y)

# Get the best degree from the grid search
best_degree = grid_search.best_params_['polynomialfeatures__degree']

# Print the best degree
print("Best Degree:", best_degree)

# Train the model with the best degree
polyreg = make_pipeline(PolynomialFeatures(degree=best_degree), LogisticRegression(max_iter=10000))
polyreg.fit(X, y)

# Get model parameters
intercept = polyreg.named_steps['logisticregression'].intercept_
coef = polyreg.named_steps['logisticregression'].coef_

# Print model parameters
print("Intercept:", intercept)
print("Coefficients:", coef)

# Plot the decision boundary
x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100))
Z = polyreg.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

plt.figure(1, figsize=(6, 6))
plt.pcolormesh(xx, yy, Z, cmap=plt.cm.Paired)

# Plot the data points
plt.scatter(X[:, 0], X[:, 1], c=y, edgecolors='k', cmap=plt.cm.Paired)
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')


# Define new data points for prediction
X_new = np.array([[1.5, 1.5], [2.0, 2.0], [3.0, 3.0]])

# Predict the class labels for new data points
y_new = polyreg.predict(X_new)

print("Predicted Class Labels for X_new:", y_new)

plt.show()