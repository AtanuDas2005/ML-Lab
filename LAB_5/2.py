# Decision Tree for Weather Dataset (ID3 Algorithm)

import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, classification_report

# Step 1: Encode the Weather Dataset
# Weather: Sunny=1, Cloudy=2, Rainy=3
# Temperature: Hot=1, Mild=2, Cool=3
# Humidity: High=1, Normal=2
# Wind: Weak=1, Strong=2
# Play: Yes=1, No=0

X = [
    [1,1,1,1],  # Sunny, Hot, High, Weak
    [2,1,1,1],  # Cloudy, Hot, High, Weak
    [1,2,2,2],  # Sunny, Mild, Normal, Strong
    [2,2,1,2],  # Cloudy, Mild, High, Strong
    [3,2,1,2],  # Rainy, Mild, High, Strong
    [3,3,2,2],  # Rainy, Cool, Normal, Strong
    [3,2,1,1],  # Rainy, Mild, High, Weak
    [1,1,1,2],  # Sunny, Hot, High, Strong
    [2,1,2,1],  # Cloudy, Hot, Normal, Weak
    [3,2,1,2],  # Rainy, Mild, High, Strong
]

y = [0,1,1,1,0,0,1,0,1,0]  # Play (No=0, Yes=1)

# Step 2: Create Decision Tree using ID3 (Entropy)
model = DecisionTreeClassifier(criterion="entropy", random_state=42)
model.fit(X, y)

# Step 3: Predict and Evaluate
y_pred = model.predict(X)

print("Accuracy:", accuracy_score(y, y_pred))
print("\nClassification Report:")
print(classification_report(y, y_pred, target_names=["No", "Yes"]))

# Step 4: Visualize the Decision Tree
plt.figure(figsize=(16,8))
plot_tree(
    model,
    feature_names=["Weather", "Temperature", "Humidity", "Wind"],
    class_names=["No", "Yes"],
    filled=True
)
plt.show()