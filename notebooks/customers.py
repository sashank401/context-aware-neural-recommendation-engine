import pandas as pd
from pathlib import Path

folder = Path(r"C:\Users\lenovo\Desktop\customers.csv")
file = [f for f in folder.iterdir() if f.is_file()][0]
print(file)

customers = pd.read_csv(file)
print(customers.shape)