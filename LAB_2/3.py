import pandas as pd
import math

df = pd.read_csv("dataset.csv")

x = df["x"].tolist()
n = len(x)
mean = sum(x) / n

variance = sum((xi - mean) ** 2 for xi in x) / (n-1)
std_dev = math.sqrt(variance)

print("Standard Deviation =", std_dev)
