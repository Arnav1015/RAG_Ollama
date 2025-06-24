import os
import pandas as pd
import numpy as np
def load_text_data(file_path, text_column='text'):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    ext = os.path.splitext(file_path)[-1].lower()
    
    if ext == '.csv':
        df = pd.read_csv(file_path)
    elif ext in ['.xls', '.xlsx']:
        try:
            import openpyxl
        except ImportError:
            raise ImportError("Reading Excel files requires 'openpyxl'. Please install it with: pip install openpyxl")
        df = pd.read_excel(file_path)
    else:
        raise ValueError("Unsupported file type. Please use a .csv or .xlsx file.")

    if text_column not in df.columns:
        raise ValueError(f"Column '{text_column}' not found in file. Available columns: {', '.join(df.columns)}")
    print("Done step1")
    return df[text_column].dropna().astype(str).tolist()

input_file = 'data.csv'  # Change to 'data.xlsx' to use Excel instead
text_column = 'text'
texts = load_text_data(input_file, text_column)
