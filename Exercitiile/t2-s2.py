## Exercise 2 (10 minutes): K-Means Clustering

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets import load_iris

# 1. Simulated preprocessed data (Iris dataset)
iris = load_iris()

df_scaled = pd.DataFrame(iris.data, columns=iris.feature_names)

print("Data preview:")
print(df_scaled.head())

# 2. Instantiate K-Means
kmeans = KMeans(n_clusters=3, random_state=42)

# 3. Fit model
kmeans.fit(df_scaled)

# 4. Extract cluster labels
labels = kmeans.labels_

# 5. Add labels to DataFrame
df_scaled["Cluster"] = labels

# 6. Print results
print("\nClustered data (first rows):")
print(df_scaled.head())

print("\nCluster counts:")
print(df_scaled["Cluster"].value_counts())

# 7. Optional visualization (2D scatter plot)
plt.figure(figsize=(6, 4))

plt.scatter(
    df_scaled["sepal length (cm)"],
    df_scaled["sepal width (cm)"],
    c=df_scaled["Cluster"]
)

plt.title("K-Means Clustering (Iris)")
plt.xlabel("Sepal Length")
plt.ylabel("Sepal Width")
plt.show()