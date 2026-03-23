import pandas as pd
import matplotlib.pyplot as plt

# Load dataset
df = pd.read_csv("medicalStore.csv")

x = df["Competitors"]
y = df["Sales"]

# Mean values
x_mean = x.mean()
y_mean = y.mean()

# Slope using mean-deviation formula
b = ((x - x_mean) * (y - y_mean)).sum() / ((x - x_mean) ** 2).sum()

# Intercept
a = y_mean - b * x_mean

print("Intercept (a):", a)
print("Slope (b):", b)

# Regression line
y_pred = a + b * x

# Scatter plot + regression line
plt.figure()
plt.scatter(x, y)
plt.plot(x, y_pred)
plt.xlabel("Number of Competitors")
plt.ylabel("Sales Volume")
plt.title("Sales vs Competitors (Linear Regression)")
plt.show()
