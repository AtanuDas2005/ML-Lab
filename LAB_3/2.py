import numpy as np

# Actual and Predicted sales (in lakhs)
actual = np.array([112, 113, 114, 115, 112, 121, 122, 114])
predicted = np.array([113, 112, 116, 117, 110, 118, 121, 115])

n = len(actual)

# Mean Squared Error (MSE)
mse = np.mean((actual - predicted) ** 2)

# Root Mean Squared Error (RMSE)
rmse = np.sqrt(mse)

# Hybrid Error
hybrid_error = 0.3 * mse + 0.25 * rmse

# Mean Absolute Percentage Error (MAPE)
mape = np.mean(np.abs((actual - predicted) / actual)) * 100

# Output
print("MSE:", mse)
print("RMSE:", rmse)
print("Hybrid Error:", hybrid_error)
print("MAPE:", mape, "%")
