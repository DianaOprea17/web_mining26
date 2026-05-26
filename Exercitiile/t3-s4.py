## Exercise 4 (10 minutes): Ridge vs. Lasso

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge, Lasso
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

# 1. Create synthetic dataset
np.random.seed(42)
num_samples = 30

X1 = np.random.rand(num_samples) * 10
X2 = X1 + np.random.rand(num_samples) * 2   # correlated feature
X3 = np.random.rand(num_samples) * 10       # less relevant feature

y = 3 * X1 + 1.5 * X2 + np.random.normal(0, 5, size=num_samples)

df = pd.DataFrame({
    "X1": X1,
    "X2": X2,
    "X3": X3,
    "Target": y
})

# 2. Split features and target
X = df[["X1", "X2", "X3"]]
y = df["Target"]

# 3. Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.3,
    random_state=42
)

# 4. Train models
ridge = Ridge(alpha=1.0)
lasso = Lasso(alpha=0.1)

ridge.fit(X_train, y_train)
lasso.fit(X_train, y_train)

# 5. Predictions
y_pred_ridge = ridge.predict(X_test)
y_pred_lasso = lasso.predict(X_test)

# 6. Evaluation metrics
r2_ridge = r2_score(y_test, y_pred_ridge)
mse_ridge = mean_squared_error(y_test, y_pred_ridge)
mae_ridge = mean_absolute_error(y_test, y_pred_ridge)

r2_lasso = r2_score(y_test, y_pred_lasso)
mse_lasso = mean_squared_error(y_test, y_pred_lasso)
mae_lasso = mean_absolute_error(y_test, y_pred_lasso)

# 7. Results
print("True Relationship: y = 3*X1 + 1.5*X2 + noise")

print("\nRidge Coefficients:", ridge.coef_)
print("Ridge Intercept:", ridge.intercept_)
print(f"Ridge R²: {r2_ridge:.3f}, MSE: {mse_ridge:.3f}, MAE: {mae_ridge:.3f}")

print("\nLasso Coefficients:", lasso.coef_)
print("Lasso Intercept:", lasso.intercept_)
print(f"Lasso R²: {r2_lasso:.3f}, MSE: {mse_lasso:.3f}, MAE: {mae_lasso:.3f}")