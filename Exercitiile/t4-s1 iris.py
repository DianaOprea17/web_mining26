## Exercise 1 (10 minutes): Load & Preprocess Your Dataset
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

# 1. Load the Iris dataset from scikit-learn
iris = load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)

print("Original data (first rows):")
print(df.head())

# 2. Introduce some artificial missing values (optional, for demonstration)
#    Here, we'll set a few entries to NaN in the 'petal length (cm)' column
df.iloc[5:10, 2] = np.nan

print("\nData with missing values:")
print(df.iloc[3:12])

# 3. Handle missing values
#    We'll use SimpleImputer to replace NaNs with the mean of each column
imputer = SimpleImputer(strategy="mean")

df_imputed = pd.DataFrame(
    imputer.fit_transform(df),
    columns=df.columns
)

print("\nAfter imputing missing values:")
print(df_imputed.iloc[3:12])

# 4. Scale the data
#    StandardScaler transforms each feature to have mean=0 and std=1
scaler = StandardScaler()

df_scaled = pd.DataFrame(
    scaler.fit_transform(df_imputed),
    columns=df.columns
)

# 5. Check the results
print("\nScaled data (first rows):")
print(df_scaled.head())

# 6. (Optional) Print the first few rows to confirm preprocessing