import pandas as pd

data = {
    "Marks": [80, 90, 70, 85]
}

df = pd.DataFrame(data)

print("Average Marks =", df["Marks"].mean())
