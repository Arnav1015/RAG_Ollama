import os
import pandas as pd
import fitz
import nltk
from uuid import uuid4
from typing import List, Dict, Tuple
from nltk.tokenize import sent_tokenize
from sentence_transformers import SentenceTransformer
from paddleocr import PaddleOCR
from Vector_Database_ChromaDB import ChromaVectorStore

nltk.download("punkt")
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
ocr_model = PaddleOCR(use_angle_cls=True, lang="en")

def extract_text_from_image(file_path: str) -> str:
    result = ocr_model.ocr(file_path, cls=True)
    return " ".join([line[1][0] for line in result[0]]) if result[0] else ""

def load_csv_chunks(path: str, chunk_size: int = 1) -> Tuple[List[str], List[Dict]]:
    try:
        df = pd.read_csv(path)
    except Exception:
        df = pd.read_excel(path)
    if df.empty:
        return [], []

    rows = df.astype(str).apply(lambda row: " | ".join(row), axis=1).tolist()
    chunks, metadatas = [], []
    for i in range(0, len(rows), chunk_size):
        text_chunk = " ".join(rows[i:i+chunk_size])
        chunks.append(text_chunk)
        metadatas.append({
            "source": "csv_or_excel",
            "filename": os.path.basename(path),
            "chunk": i // chunk_size,
        })
    return chunks, metadatas

def load_pdf_chunks(path: str, max_sentences_per_chunk: int = 5) -> Tuple[List[str], List[Dict]]:
    doc = fitz.open(path)
    chunks, metadatas = [], []
    for page_num in range(len(doc)):
        text = doc[page_num].get_text()
        sentences = sent_tokenize(text)
        for i in range(0, len(sentences), max_sentences_per_chunk):
            chunk = " ".join(sentences[i:i+max_sentences_per_chunk])
            if chunk.strip():
                chunks.append(chunk)
                metadatas.append({
                    "source": "pdf",
                    "filename": os.path.basename(path),
                    "page": page_num + 1,
                    "chunk": i // max_sentences_per_chunk,
                })
    return chunks, metadatas

def load_image_chunk(path: str) -> Tuple[List[str], List[Dict]]:
    text = extract_text_from_image(path)
    return [text], [{
        "source": "image",
        "filename": os.path.basename(path),
        "chunk": 0
    }] if text.strip() else ([], [])

def embed_and_store(chunks: List[str], metadatas: List[Dict], store: ChromaVectorStore):
    ids = [str(uuid4()) for _ in chunks]
    embeddings = embedding_model.encode(chunks, show_progress_bar=True, convert_to_numpy=False)
    store.add_documents(ids, chunks, embeddings, metadatas)
    print(f"Stored {len(chunks)} chunks.")

def process_file(file_path: str, store: ChromaVectorStore, existing_files: set):
    filename = os.path.basename(file_path)
    if filename in existing_files:
        print(f" Skipping duplicate: {filename}")
        return

    ext = os.path.splitext(file_path)[1].lower()
    if ext in [".csv", ".xlsx"]:
        chunks, metas = load_csv_chunks(file_path)
    elif ext == ".pdf":
        chunks, metas = load_pdf_chunks(file_path)
    elif ext in [".jpg", ".jpeg", ".png"]:
        chunks, metas = load_image_chunk(file_path)
    else:
        print(f"Unsupported file type: {file_path}")
        return

    if chunks:
        embed_and_store(chunks, metas, store)
    else:
        print(f"No data extracted from: {file_path}")

def process_folder(folder_path: str, store: ChromaVectorStore):
    existing_files = store.get_all_filenames()
    for root, _, files in os.walk(folder_path):
        for filename in files:
            if filename.endswith((".csv", ".xlsx", ".pdf", ".jpg", ".jpeg", ".png")):
                full_path = os.path.join(root, filename)
                print(f"Processing: {full_path}")
                process_file(full_path, store, existing_files)

if __name__ == "__main__":
    folder_path = "C:/Users/Arnav/Documents/a_Database/text"
    store = ChromaVectorStore(collection_name="hybrid_data", persist_directory="chroma_db")
    process_folder(folder_path, store)
