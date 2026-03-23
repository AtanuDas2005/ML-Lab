import numpy as np
import pandas as pd

print("Q1(a) Standard Error")
# Given values
n = 200
std_dev = 180

# Standard Error
standard_error = std_dev / np.sqrt(n)
print("Standard Error =", round(standard_error, 2))


print("\nQ1(b) Average Distance from Mean (Mean Absolute Deviation)")
# Given data
data = np.array([12, 23, 31, 15, 26, 24, 16, 23])

mean = np.mean(data)
mad = np.mean(np.abs(data - mean))

print("Mean =", mean)
print("Average Distance from Mean =", round(mad, 2))


print("\nQ1(c) Group Frequency Table & Skewness")

# Age groups and frequencies
age_groups = [(2,4), (4,6), (6,8), (8,10)]
frequency = np.array([16, 13, 7, 5])

# Midpoints
midpoints = np.array([(a+b)/2 for a,b in age_groups])

# Mean
group_mean = np.sum(midpoints * frequency) / np.sum(frequency)

# Median calculation
N = np.sum(frequency)
cf = np.cumsum(frequency)
median_class_index = np.where(cf >= N/2)[0][0]

L = age_groups[median_class_index][0]
f = frequency[median_class_index]
cf_prev = cf[median_class_index - 1] if median_class_index > 0 else 0
w = age_groups[0][1] - age_groups[0][0]

median = L + ((N/2 - cf_prev) / f) * w

# Mode calculation
modal_index = np.argmax(frequency)
f1 = frequency[modal_index]
f0 = frequency[modal_index - 1] if modal_index > 0 else 0
f2 = frequency[modal_index + 1] if modal_index < len(frequency)-1 else 0
L_mode = age_groups[modal_index][0]

mode = L_mode + ((f1 - f0) / (2*f1 - f0 - f2)) * w

print("Mean =", round(group_mean, 2))
print("Median =", round(median, 2))
print("Mode =", round(mode, 2))

if group_mean > median > mode:
    print("Skewness: Positively Skewed")
elif group_mean < median < mode:
    print("Skewness: Negatively Skewed")
else:
    print("Skewness: Symmetrical")


print("\nQ1(d) Stratified vs Cluster Sampling")

print("Stratified Sampling:")
print("- Population divided into homogeneous strata")
print("- Samples selected from each stratum")

print("\nCluster Sampling:")
print("- Population divided into heterogeneous clusters")
print("- Only some clusters are selected")