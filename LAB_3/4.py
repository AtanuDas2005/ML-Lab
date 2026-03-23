import math

# Given data
x = [43, 21, 25, 42, 57, 59]   # Age
y = [99, 65, 79, 75, 87, 81]   # Glucose level

n = len(x)

# Calculate required sums
sum_x = sum(x)
sum_y = sum(y)
sum_x2 = sum(i**2 for i in x)
sum_y2 = sum(i**2 for i in y)
sum_xy = sum(x[i] * y[i] for i in range(n))

# Pearson correlation formula
numerator = n * sum_xy - sum_x * sum_y
denominator = math.sqrt((n * sum_x2 - sum_x**2) * (n * sum_y2 - sum_y**2))

r = numerator / denominator

print("Pearson Correlation Coefficient (r):", r)
