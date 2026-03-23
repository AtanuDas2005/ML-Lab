import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# -----------------------------
# 1. Create a simple dataset
# -----------------------------
# Features: [email_length, number_of_links]
# Label: 1 = Spam, 0 = Not Spam

X = np.array([
    [100, 0],
    [200, 1],
    [300, 2],
    [50, 0],
    [400, 5],
    [20, 0],
    [350, 4],
    [80, 1]
])

y = np.array([0, 0, 1, 0, 1, 0, 1, 0])

# -----------------------------
# 2. Train-test split
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42
)

# -----------------------------
# 3. Train Logistic Regression
# -----------------------------
model = LogisticRegression()
model.fit(X_train, y_train)

# -----------------------------
# 4. Predictions
# -----------------------------
y_pred = model.predict(X_test)

# -----------------------------
# 5. Evaluation
# -----------------------------
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
