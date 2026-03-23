# ===============================
# 1. Import Required Libraries
# ===============================
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


# ===============================
# 2. Load the Dataset
# ===============================
# Replace with your dataset file name
data = pd.read_csv("telecom_churn.csv")

print("Dataset Shape:", data.shape)
print("\nFirst 5 rows of dataset:\n", data.head())


# ===============================
# 3. Data Preprocessing
# ===============================

# Drop Customer ID column if present
if 'customerID' in data.columns:
    data.drop('customerID', axis=1, inplace=True)

# Convert categorical columns to numeric using Label Encoding
label_encoder = LabelEncoder()

for column in data.columns:
    if data[column].dtype == 'object':
        data[column] = label_encoder.fit_transform(data[column])

print("\nDataset after encoding:\n", data.head())


# ===============================
# 4. Split Features and Target
# ===============================

# Target variable (Churn column)
X = data.drop('Churn', axis=1)
y = data['Churn']


# ===============================
# 5. Train-Test Split
# ===============================
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.20,
    random_state=42
)


# ===============================
# 6. Train Random Forest Classifier
# ===============================
rf_classifier = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

rf_classifier.fit(X_train, y_train)


# ===============================
# 7. Make Predictions
# ===============================
y_pred = rf_classifier.predict(X_test)


# ===============================
# 8. Model Evaluation
# ===============================

# Accuracy
accuracy = accuracy_score(y_test, y_pred)
print("\nModel Accuracy:", accuracy)

# Confusion Matrix
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Classification Report
print("\nClassification Report:")
print(classification_report(y_test, y_pred))


# ===============================
# 9. End of Program
# ===============================
print("\nCustomer Churn Prediction using Random Forest completed successfully.")
