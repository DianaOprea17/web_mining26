# Exercițiul 3: Folosiți setul de date despre prețuri imobiliare din California pentru a
# antrena un model de regresie Ridge. Tratați valorile lipsă, folosiți GridSearchCV
# pentru a căuta parametrul optim alpha și afișați eroarea MSE pe setul de test.

import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import mean_squared_error

url = "https://raw.githubusercontent.com/ageron/handson-ml/master/datasets/housing/housing.csv"
df = pd.read_csv(url)

# === START ===

# 1. separare X / y
X = df.drop("median_house_value", axis=1)
y = df["median_house_value"]

# 2. tratare valori lipsă (manual, doar cu pandas)
# numeric → median
for col in X.select_dtypes(include=["float64", "int64"]).columns:
    X[col] = X[col].fillna(X[col].median())

# categoric → mod
X["ocean_proximity"] = X["ocean_proximity"].fillna(X["ocean_proximity"].mode()[0])

# 3. encoding simplu (one-hot manual cu pandas)
X = pd.get_dummies(X, columns=["ocean_proximity"])

# 4. split train/test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 5. model + GridSearch
ridge = Ridge()

param_grid = {
    "alpha": [0.1, 1.0, 10.0, 100.0]
}

grid = GridSearchCV(
    ridge,
    param_grid,
    cv=5,
    scoring="neg_mean_squared_error"
)

grid.fit(X_train, y_train)

# 6. predicție
y_pred = grid.predict(X_test)

# 7. MSE
mse = mean_squared_error(y_test, y_pred)

print("Best alpha:", grid.best_params_)
print("Test MSE:", mse)

# === END ===