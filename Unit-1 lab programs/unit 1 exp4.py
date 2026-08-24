# pip install pandas

import pandas as pd

data = {
    "Name": ["Teja", "Rahul", "Anu", "Priya"],
    "Grade": ["A", "B", "A", "C"]
}

df = pd.DataFrame(data)

print(df)
