# query_with_word2vec.py

import pickle
import faiss
import numpy as np
from word2vecscratch import tokenize, embedding_matrix, word_to_id

def embed_query(query):
    tokens = tokenize(query)
    vectors = [
        embedding_matrix[word_to_id[token]]
        for token in tokens if token in word_to_id
    ]
    if not vectors:
        return None
    return np.mean(vectors, axis=0).reshape(1, -1)

def query_faiss(query, k :int = 10):
    # Load index and data
    index = faiss.read_index("material/rag_word2vec_index.faiss")
    with open("material/rag_word2vec_texts.pkl", "rb") as f:
        texts = pickle.load(f)
    with open("material/rag_word2vec_metadata.pkl", "rb") as f:
        metadata = pickle.load(f)

    query_vec = embed_query(query)
    if query_vec is None:
        print("No valid tokens in query.")
        return []

    distances, indices = index.search(query_vec, k)
    results = []
    for i, dist in zip(indices[0], distances[0]):
        results.append((texts[i], metadata[i], dist))
    return results

# Test 
if __name__ == "__main__":
    query = "What is machine learning?"
    results = query_faiss(query)

    for chunk, (filename, idx), dist in results:
        print(f"\n📄 From {filename} [chunk {idx}] (score={dist:.4f}):\n{chunk}")
