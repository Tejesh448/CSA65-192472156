import pandas as pd

data = {
    "Name": ["Ram", "Sita", "John"],
    "Marks": [85, 90, 78]
}

df = pd.DataFrame(data)

df["Result"] = ["Pass", "Pass", "Pass"]

print(df)
