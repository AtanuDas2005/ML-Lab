# ==========================================
# K-Means Customer Segmentation Experiment
# ==========================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# ------------------------------------------------
# 1. Create / Load Customer Dataset
# (Example dataset: Income vs Spending Score)
# ------------------------------------------------

# Example data (you can replace with CSV file)
data = {
    "Annual_Income": [15,16,17,20,23,25,28,30,35,40,45,50,55,60,65,70,75,80,85,90],
    "Spending_Score": [39,81,6,77,40,76,6,94,3,72,5,60,15,55,10,45,20,35,25,30]
}

df = pd.DataFrame(data)

X = df[["Annual_Income", "Spending_Score"]]

# ------------------------------------------------
# 2. Feature Scaling
# ------------------------------------------------
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ------------------------------------------------
# 3. Apply K-Means Clustering
# ------------------------------------------------
kmeans = KMeans(n_clusters=3, random_state=42)
clusters = kmeans.fit_predict(X_scaled)

df["Cluster"] = clusters

# ------------------------------------------------
# 4. Plot Clusters in 2D Space
# ------------------------------------------------
plt.figure(figsize=(8,6))

plt.scatter(
    X_scaled[:,0],
    X_scaled[:,1],
    c=clusters
)

# Plot Centroids
centroids = kmeans.cluster_centers_
plt.scatter(
    centroids[:,0],
    centroids[:,1],
    marker='X',
    s=200
)

plt.title("Customer Segmentation using K-Means")
plt.xlabel("Scaled Annual Income")
plt.ylabel("Scaled Spending Score")
plt.show()

# ------------------------------------------------
# 5. Cluster Analysis
# ------------------------------------------------
print("\nCluster-wise Customer Analysis:\n")
print(df.groupby("Cluster").mean())