import pandas as pd
import numpy as np

# Dataset
classes = ["2-4", "4-6", "6-8", "8-10"]
f = np.array([3, 4, 2, 1])
x = np.array([3, 5, 7, 9])  # mid values

# Mean
mean = np.sum(f * x) / np.sum(f)

# Standard Deviation
sd = np.sqrt(np.sum(f * (x - mean)**2) / np.sum(f))

# Skewness
skew = np.sum(f * (x - mean)**3) / ((np.sum(f) - 1) * sd**3)

# Kurtosis
kurt = np.sum(f * (x - mean)**4) / ((np.sum(f) - 1) * sd**4)

print("Mean =", mean)
print("Standard Deviation =", sd)
print("Skewness =", skew)
print("Kurtosis =", kurt)
