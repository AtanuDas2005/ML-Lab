import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Step 1: Create the dataset
data = {
    "YoE": [1, 1.1, 1.3, 2, 2.2, 2.7, 3, 3.2, 3.9, 4, 4.1, 4.2],
    "Salary": [32323, 45207, 39751, 43525, 39891, 56542,
               60150, 54545, 63218, 55794, 56081, 57081]
}

df = pd.DataFrame(data)

# Step 2: Independent (X) and Dependent (Y) variables
X = df["YoE"].values.reshape(-1, 1)
y = df["Salary"].values

# Step 3: Create and train Linear Regression model
model = LinearRegression()
model.fit(X, y)

# Step 4: Get slope and intercept
slope = model.coef_[0]
intercept = model.intercept_

print("Slope (m):", slope)
print("Intercept (c):", intercept)

# Step 5: Predict salary
y_pred = model.predict(X)

# Step 6: Calculate random error (residuals)
errors = y - y_pred
print("\nRandom Errors (Residuals):")
print(errors)

# Step 7: Relationship type (Correlation)
correlation = np.corrcoef(df["YoE"], df["Salary"])[0, 1]
print("\nCorrelation Coefficient:", correlation)

# Step 8: Plot regression line
plt.scatter(X, y)
plt.plot(X, y_pred)
plt.xlabel("Years of Experience")
plt.ylabel("Salary")
plt.title("Linear Regression: YoE vs Salary")
plt.show()