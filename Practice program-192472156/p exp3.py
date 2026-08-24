import pandas as pd

data = {
    "Name": ["Ram", "Sita", "John", "Ravi"],
    "Age": [20, 21, 22, 23]
}

df = pd.DataFrame(data)

print(df.head(2))
