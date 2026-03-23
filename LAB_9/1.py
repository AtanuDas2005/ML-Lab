# ===============================
# SVM on MNIST Handwritten Digits
# ===============================

import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import seaborn as sns

# -------------------------------
# 1. Load MNIST Dataset
# -------------------------------
print("Loading MNIST dataset...")
mnist = fetch_openml('mnist_784', version=1, as_frame=False)

X = mnist.data          # 70000 images, each 784 features (28x28)
y = mnist.target.astype(int)

print("Dataset shape:", X.shape)

# -------------------------------
# 2. Train-Test Split
# -------------------------------
# Use smaller subset if system is slow
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=10000, random_state=42, stratify=y
)

# -------------------------------
# 3. Feature Scaling (IMPORTANT for SVM)
# -------------------------------
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# -------------------------------
# 4. Build SVM Model
# -------------------------------
# RBF kernel gives strong accuracy on MNIST
print("Training SVM model... (may take time)")
svm_model = SVC(kernel='rbf', gamma='scale')

svm_model.fit(X_train, y_train)

# -------------------------------
# 5. Predictions
# -------------------------------
y_pred = svm_model.predict(X_test)

# -------------------------------
# 6. Accuracy
# -------------------------------
accuracy = accuracy_score(y_test, y_pred)
print("\nAccuracy:", accuracy)

# -------------------------------
# 7. Confusion Matrix
# -------------------------------
cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(8,6))
sns.heatmap(cm, annot=False, cmap="Blues")
plt.title("Confusion Matrix")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.show()

# -------------------------------
# 8. Classification Report
# -------------------------------
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))
