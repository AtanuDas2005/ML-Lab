import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Load your dataset
data = pd.read_csv("covid19_trade.csv")

print("Sample of the dataset:")
print(data.head())

# ---- Plot 1: Histogram (using the first numeric column) ----
plt.figure(figsize=(8, 6))
plt.hist(data[data.columns[0]], bins=20, edgecolor='black')
plt.title(f'Distribution of {data.columns[0]}')
plt.xlabel(data.columns[0])
plt.ylabel('Frequency')
plt.show()

# ---- Plot 2: Scatter Plot (using first two numeric columns) ----
plt.figure(figsize=(8, 6))
sns.scatterplot(
    x=data.columns[0],
    y=data.columns[1],
    data=data
)
plt.title(f'Scatter Plot of {data.columns[0]} vs {data.columns[1]}')
plt.xlabel(data.columns[0])
plt.ylabel(data.columns[1])
plt.show()

# ---- Plot 3: Correlation Heatmap ----
numeric_data = data.select_dtypes(include=['float64', 'int64'])
correlation_matrix = numeric_data.corr()

plt.figure(figsize=(8, 6))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
plt.title('Correlation Heatmap')
plt.show()
