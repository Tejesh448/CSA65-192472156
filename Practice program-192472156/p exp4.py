import pandas as pd

data = {
    "Name": ["Ram", "Sita", "John"],
    "Marks": [85, 92, 78]
}

df = pd.DataFrame(data)

df = df.sort_values("Marks")

print(df)
