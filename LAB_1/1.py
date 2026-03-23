import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Load a sample dataset

# Load the Iris dataset from seaborn
iris = sns.load_dataset('iris')
print("Sample of the Iris dataset:")
print(iris.head())

# Plot histogram of 'sepal_length'
plt.figure(figsize=(8, 6))
plt.hist(iris['sepal_length'], bins=20, color='skyblue', edgecolor='black')
plt.title('Distribution of Sepal Length')
plt.xlabel('Sepal Length')
plt.ylabel('Frequency')
plt.show()

plt.figure(figsize=(8, 6))
sns.scatterplot(x='sepal_length', y='sepal_width', hue='species', data=iris, palette='viridis')
plt.title('Scatter Plot of Sepal Length vs Sepal Width')
plt.xlabel('Sepal Length')
plt.ylabel('Sepal Width')
plt.show()

correlation_matrix = iris.drop(columns=['species']).corr()
plt.figure(figsize=(8, 6))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
plt.title('Correlation Heatmap')
plt.show()