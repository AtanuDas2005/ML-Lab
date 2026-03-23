#Given a dataset containing values and their frequencies in a CSV file, 
#write a Python program to calculate the Mean, Median, and Mode of the data.

import pandas as pd

df = pd.read_csv("central_tendency.csv")

values = df["value"].tolist()
freqs = df["freq"].tolist()

total = sum(freqs)
mean = sum(values[i] * freqs[i] for i in range(len(values))) / total

data = []
for v, f in zip(values, freqs):
    data += [v] * f

data.sort()
n = len(data)

if n % 2 == 1:
    median = data[n // 2]
else:
    median = (data[n//2 - 1] + data[n//2]) / 2

max_freq = max(freqs)
mode = values[freqs.index(max_freq)]

print("Mean =", mean)
print("Median =", median)
print("Mode =", mode)
