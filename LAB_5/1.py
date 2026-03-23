# Week 5: Decision Trees
# Objective: Develop a decision tree classifier for Iris dataset and evaluate it

# 1. Import Required Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report


# 2. Load the Iris Dataset
iris = load_iris()

X = iris.data          # Features
y = iris.target        # Target variable (species)

# Convert to DataFrame for better understanding (optional)
iris_df = pd.DataFrame(data=X, columns=iris.feature_names)
iris_df['species'] = y

print("Iris Dataset Preview:\n")
print(iris_df.head())


# 3. Split the Dataset into Training and Testing Sets
# Split the data into training (80%) and testing (20%) sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


# 4. Train a Decision Tree Model
# Initialize and train the Decision Tree Classifier
dt_model = DecisionTreeClassifier(random_state=42)
dt_model.fit(X_train, y_train)


# 5. Evaluate the Model
# Make predictions on the test set
y_pred = dt_model.predict(X_test)

# Evaluate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"\nModel Accuracy: {accuracy:.2f}")

# Generate a detailed classification report
report = classification_report(y_test, y_pred, target_names=iris.target_names)
print("\nClassification Report:\n", report)


# 6. Visualize the Decision Tree
plt.figure(figsize=(12, 8))
plot_tree(
    dt_model,
    feature_names=iris.feature_names,
    class_names=iris.target_names,
    filled=True
)
plt.title("Decision Tree Visualization (Iris Dataset)")
plt.show()
