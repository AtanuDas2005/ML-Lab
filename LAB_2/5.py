import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# 1. Load dataset
df = pd.read_csv("house_prediction.csv")

print("Sample data:")
print(df.head())

# 2. Select feature and target
X = df[['Area']]     # Independent variable
y = df['Price']      # Dependent variable

# 3. Train-test split (increased test size to avoid R² = nan)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.4, random_state=42
)

print("Training samples:", len(X_train))
print("Testing samples:", len(X_test))

# 4. Train model
model = LinearRegression()
model.fit(X_train, y_train)

# 5. Predictions
y_pred = model.predict(X_test)

# 6. Evaluation
mse = mean_squared_error(y_test, y_pred)

# R² only if more than 1 test sample
if len(y_test) > 1:
    r2 = r2_score(y_test, y_pred)
else:
    r2 = "Not defined (too few test samples)"

print("Mean Squared Error:", mse)
print("R2 Score:", r2)

# 7. Plot regression line properly
X_sorted = X.sort_values(by='Area')
y_line = model.predict(X_sorted)

plt.scatter(X, y, label="Actual Data")
plt.plot(X_sorted, y_line, label="Regression Line")
plt.xlabel("Area (sqft)")
plt.ylabel("Price (Lakhs)")
plt.title("Simple Linear Regression - House Price Prediction")
plt.legend()
plt.show()
