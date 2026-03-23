import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression

# Dataset
X = np.array([
    0.50, 0.75, 1.00, 1.25, 1.50, 1.75, 1.75, 2.00, 2.25, 2.50,
    2.75, 3.00, 3.25, 3.50, 4.00, 4.25, 4.50, 4.75, 5.00, 5.50
]).reshape(-1, 1)

y = np.array([
    0, 0, 0, 0, 0, 0, 1, 0, 1, 0,
    1, 0, 1, 0, 1, 1, 1, 1, 1, 1
])

# Train logistic regression model
model = LogisticRegression()
model.fit(X, y)

# Generate values for smooth curve
x_curve = np.linspace(0, 6, 300).reshape(-1, 1)
y_curve = model.predict_proba(x_curve)[:, 1]

# Plot graph
plt.figure()
plt.scatter(X, y)
plt.plot(x_curve, y_curve)
plt.xlabel("Hours Studied")
plt.ylabel("Probability of Pass")
plt.title("Logistic Regression Curve (Hours vs Pass)")
plt.show()
