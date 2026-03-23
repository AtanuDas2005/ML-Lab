import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.cluster import AgglomerativeClustering

# dataset
points = np.array([
    [2,3],   # A
    [3,4],   # B
    [5,8],   # C
    [6,9],   # D
    [8,10]   # E
])

labels = ['A','B','C','D','E']

# Perform hierarchical clustering with average linkage
Z = linkage(points, method='average')

# Plot dendrogram
plt.figure(figsize=(6,4))
dendrogram(Z, labels=labels)
plt.title("Hierarchical Clustering (Average Linkage)")
plt.xlabel("Points")
plt.ylabel("Distance")
plt.show()

# If we want clusters (example: 2 clusters)
model = AgglomerativeClustering(n_clusters=2, linkage='average')
clusters = model.fit_predict(points)

print("Cluster assignment:")
for p,c in zip(labels, clusters):
    print(p, "-> Cluster", c)