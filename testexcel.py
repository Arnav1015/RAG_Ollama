
import pandas as pd
from pathlib import Path

# Define your data in a dictionary format
data = [
    {'Name': 'John', 'Age': 28, 'Country': 'USA'},
    {'Name': 'Anna', 'Age': 24, 'Country': 'UK'},
    {'Name': 'Peter', 'Age': 35, 'Country': 'Australia'},
    {'Name': 'Linda', 'Age': 32, 'Country': 'Germany'}
]

# Create DataFrame
df = pd.DataFrame(data)

# Specify the output file path and name
output_file_path = Path("data.xlsx")

# Write data to Excel file
df.to_excel(output_file_path, index=False)




# Specify the output file path and name
output_file_path = Path("data.xlsx")

# Write data to Excel file
df.to_excel(output_file_path, index=False)

output_file_path = Path("data.xlsx")

# Write data to Excel file
df.to_excel(output_file_path, index=False)