import numpy as np

# Points
points = {
"A1":(2,10),
"A2":(2,5),
"A3":(8,4),
"A4":(5,8),
"A5":(7,5),
"A6":(6,4),
"A7":(1,2),
"A8":(4,9)
}

# Initial centers
centers = {
"C1":points["A1"],
"C2":points["A4"],
"C3":points["A7"]
}

# Manhattan distance
def manhattan(p1,p2):
    return abs(p1[0]-p2[0]) + abs(p1[1]-p2[1])

clusters = {"C1":[],"C2":[],"C3":[]}

# Assign points to nearest cluster
for name,coord in points.items():
    
    d1 = manhattan(coord, centers["C1"])
    d2 = manhattan(coord, centers["C2"])
    d3 = manhattan(coord, centers["C3"])
    
    distances = [d1,d2,d3]
    cluster = distances.index(min(distances)) + 1
    
    clusters["C"+str(cluster)].append(name)

print("Cluster Assignment")
for c,p in clusters.items():
    print(c,"->",p)