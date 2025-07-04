import os
import re
import pickle
import numpy as np
import faiss
import fitz  # PyMuPDF

#Loading Word2Vec model
embedding_matrix = np.load("material/embedding_matrix.npy")
with open("material/word_to_id.pkl", "rb") as f:
    word_to_id = pickle.load(f)

#Tokenizer
def tokenize(text):
    pattern = re.compile(
        r"\b[a-zA-Z0-9]+(?:[-'][a-zA-Z0-9]+)*\b"  # Words and hyphenated
        r"|[#@][\w-]+"
        r"|(?:[A-Z]\.){2,}"
        r"|\$?\d+(?:\.\d+)?%?"
    )
    return pattern.findall(text.lower())

#Text Chunking
def chunk_text(text, by="paragraph", max_words=200, overlap=50):
    if by == "paragraph":
        return [chunk.strip() for chunk in text.split("\n\n") if chunk.strip()]

    elif by == "sentence":
        return re.split(r'(?<=[.!?])\s+', text.strip())

    elif by == "word_count":
        words = re.findall(r'\w+', text)
        chunks = []
        start = 0
        while start < len(words):
            end = start + max_words
            chunk = ' '.join(words[start:end])
            chunks.append(chunk)
            start += max_words - overlap
        return chunks

    else:
        raise ValueError("Invalid chunking method. Choose 'paragraph', 'sentence', or 'word_count'.")


#Chunk Embedding
def embed_chunk(chunk):
    tokens = tokenize(chunk)
    vectors = [
        embedding_matrix[word_to_id[token]]
        for token in tokens if token in word_to_id
    ]
    if not vectors:
        return None
    return np.mean(vectors, axis=0)

def read_pdf_to_text(file_path):
    """
    Extracts text from all pages of a PDF and returns it as a string.
    """
    try:
        doc = fitz.open(file_path)
        text = ""
        for page in doc:
            text += page.get_text()
            text += "\n\n"  # mimic paragraph break
        doc.close()
        print(f"✅ Extracted text from {file_path} ({len(text)} characters).")
        return text
    except Exception as e:
        print(f"❌ Error reading PDF {file_path}: {e}")
        return ""
    

#Process Multiple Text Files
folder = "text/"
texts = []
metadata = []
embeddings = []

for filename in os.listdir(folder):
    path = os.path.join(folder, filename)
    if filename.endswith(".txt"):
        with open(path, "r", encoding="utf-8") as f:
            content = f.read()
    elif filename.endswith(".pdf"):
        content = read_pdf_to_text(path)
    else:
        continue  # Skip unsupported formats

    chunks = chunk_text(content)
    for i, chunk in enumerate(chunks):
        vec = embed_chunk(chunk)
        if vec is not None:
            embeddings.append(vec)
            texts.append(chunk)
            metadata.append((filename, i))

    

#Building and Saveing FAISS Index

dim = embedding_matrix.shape[1]
index = faiss.IndexFlatL2(dim)
index.add(np.array(embeddings))

os.makedirs("material", exist_ok=True)
faiss.write_index(index, "material/rag_word2vec_index.faiss")

with open("material/rag_word2vec_texts.pkl", "wb") as f:
    pickle.dump(texts, f)

with open("material/rag_word2vec_metadata.pkl", "wb") as f:
    pickle.dump(metadata, f)

print(f"\n✅ Indexed {len(texts)} chunks from {len(set(m[0] for m in metadata))} files.")
