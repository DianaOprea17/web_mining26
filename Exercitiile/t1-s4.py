### Exercise 4: Feature Engineering for Classification

import pandas as pd
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler

# 1. Load dataset
df = sns.load_dataset("titanic")

print("Original data:")
print(df.head())

# 2. Select features + target
df = df[["survived", "pclass", "sex", "age", "fare", "embarked"]]

# 3. Handle missing values
df["age"] = df["age"].fillna(df["age"].mean())
df["embarked"] = df["embarked"].fillna(df["embarked"].mode()[0])

# 4. Convert categorical data
df["sex"] = df["sex"].map({"male": 0, "female": 1})

df = pd.get_dummies(df, columns=["embarked"], drop_first=True)

# 5. Split features and target
X = df.drop("survived", axis=1)
y = df["survived"]

# 6. Scale features
scaler = MinMaxScaler()
X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

print("\nProcessed data:")
print(X_scaled.head())