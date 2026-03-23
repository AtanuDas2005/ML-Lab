import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# 1. Load/Create Dataset (Using common patterns: Income vs Spending)
data = {
    'Annual_Income': [15, 16, 17, 18, 70, 72, 75, 78, 80, 85, 20, 22, 25, 90, 95],
    'Spending_Score': [39, 81, 6, 77, 40, 45, 50, 55, 60, 90, 10, 15, 20, 95, 88]
}
df = pd.DataFrame(data)

# 2. Feature Scaling (Important for K-Means distance calculations)
scaler = StandardScaler()
scaled_features = scaler.fit_transform(df)

# 3. Apply K-Means
# We'll use 4 clusters to represent the different income/spending quadrants
kmeans = KMeans(n_clusters=4, init='k-means++', random_state=42)
df['Cluster'] = kmeans.fit_predict(scaled_features)
centroids = kmeans.cluster_centers_

# 4. Plotting the Clusters
plt.figure(figsize=(10, 6))
colors = ['red', 'blue', 'green', 'orange']

for i in range(4):
    plt.scatter(scaled_features[df['Cluster'] == i, 0], 
                scaled_features[df['Cluster'] == i, 1], 
                s=100, c=colors[i], label=f'Cluster {i}')

# Mark the Centroids
plt.scatter(centroids[:, 0], centroids[:, 1], s=300, c='black', marker='X', label='Centroids')

plt.title('Customer Groups (K-Means Clustering)')
plt.xlabel('Annual Income (Scaled)')
plt.ylabel('Spending Score (Scaled)')
plt.legend()
plt.grid(True)
plt.show()