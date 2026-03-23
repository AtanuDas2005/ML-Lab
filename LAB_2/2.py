import pandas as pd

df = pd.read_csv("soccer_player.csv")

total = df["freq"].sum()

a = df[df["upper"] <= 65.95]["freq"].sum() / total * 100

b = df[(df["lower"] >= 61.95) & (df["upper"] <= 65.95)]["freq"].sum() / total * 100

c = df[(df["lower"] >= 61.95) & (df["upper"] <= 71.95)]["freq"].sum()

print("Percentage < 65.95 inches =", a)
print("Percentage between 61.95 and 65.95 inches =", b)
print("Players between 61.95 and 71.95 inches =", c)
