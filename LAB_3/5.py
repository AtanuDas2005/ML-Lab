import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Load dataset
df = pd.read_csv("studentPerformance.csv")

X = df[['StudyHours', 'Attendance', 'AssignmentScore']]
y = df['Performance']

# Train on full dataset
model = LinearRegression()
model.fit(X, y)

# Predict all values
y_pred = model.predict(X)

# Metrics
mse = mean_squared_error(y, y_pred)
r2 = r2_score(y, y_pred)

print("MSE:", mse)
print("R2 Score:", r2)

# Plot ALL points
plt.figure()
plt.scatter(y, y_pred)
plt.xlabel("Actual Performance")
plt.ylabel("Predicted Performance")
plt.title("Actual vs Predicted Student Performance (All Data)")
plt.show()
