import numpy as np
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt

# Dataset (X,Y points)
data = np.array([
    [1.0, 1.0],   # A
    [1.2, 1.1],   # B
    [0.8, 1.2],   # C
    [5.0, 5.0],   # D
    [9.0, 9.0]    # E
])

# Apply DBSCAN
dbscan = DBSCAN(eps=1.5, min_samples=2)
labels = dbscan.fit_predict(data)

# Print results
points = ['A','B','C','D','E']
for p,l in zip(points,labels):
    print(f"Point {p} -> Cluster {l}")

# Plot clusters
plt.scatter(data[:,0], data[:,1], c=labels, cmap='rainbow', s=100)

for i, txt in enumerate(points):
    plt.annotate(txt,(data[i][0],data[i][1]))

plt.title("DBSCAN Clustering")
plt.xlabel("X")
plt.ylabel("Y")
plt.show()