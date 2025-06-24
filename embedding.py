import os
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from vector_database import VectorDatabase  

# Step 1: Load Data (CSV or Excel)
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
    

# --- CONFIGURATION ---
input_file = 'data.xlsx'  # Change to 'data.xlsx' to use Excel instead
text_column = 'text'
query = "Alexander"  # Your search query
index_file = 'material/my_faiss_index.index'
print("Done Config")
# Step 2: Load text
texts = load_text_data(input_file, text_column)

# Step 3: Generate Embeddings
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
embeddings = model.encode(texts, convert_to_numpy=True, show_progress_bar=True)

# Step 4: Use FAISS vector database
dimension = embeddings.shape[1]
vector_ids = list(range(len(embeddings)))

db = VectorDatabase(
    dimension=dimension,
    index_type='ivfflat',
    metric='cosine',
    database_file=index_file
)
print("Done database")

db.add_vectors(embeddings, vector_ids)
print("Vector added")

# Step 5: Query Interface
query_embedding = model.encode([query])[0]
distances, indices = db.search_similar_vectors(query_embedding)
print("Vector added")

# Step 6: Show Results
print("\n--- Top Results ---")
for idx in indices:
    if idx != -1:
        print(f"> {texts[idx]}")
