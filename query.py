# query_with_word2vec.py

import pickle
import faiss
import numpy as np
import os
import re
from word2vecscratch import tokenize, embedding_matrix, word_to_id
from word2vecscratch import read_pdf_to_text, chunk_text, embed_chunk

def embed_query(query):
    tokens = tokenize(query)
    vectors = [
        embedding_matrix[word_to_id[token]]
        for token in tokens if token in word_to_id
    ]
    if not vectors:
        return None
    return np.mean(vectors, axis=0).reshape(1, -1)

def query_faiss(query, k :int = 100):
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

def add_document_to_index(content, filename, max_words=200, overlap=50):
    """
    Process and add a text document to the FAISS index
    
    Args:
        content: Text content to add
        filename: Name of the file (used for metadata)
        max_words: Words per chunk
        overlap: Word overlap between chunks
    """
    try:
        # Load existing data
        try:
            index = faiss.read_index("material/rag_word2vec_index.faiss")
            with open("material/rag_word2vec_texts.pkl", "rb") as f:
                texts = pickle.load(f)
            with open("material/rag_word2vec_metadata.pkl", "rb") as f:
                metadata = pickle.load(f)
        except (FileNotFoundError, IOError):
            # Initialize if files don't exist
            print("Creating new index files")
            dim = embedding_matrix.shape[1]
            index = faiss.IndexFlatL2(dim)
            texts = []
            metadata = []
        
        # Chunk the text
        chunks = chunk_text(content, by="word_count", max_words=max_words, overlap=overlap)
        
        # Process each chunk
        new_vectors = []
        for i, chunk in enumerate(chunks):
            vec = embed_chunk(chunk)
            if vec is not None:
                new_vectors.append(vec)
                texts.append(chunk)
                metadata.append((filename, i))
        
        # Add to the FAISS index
        if new_vectors:
            index.add(np.array(new_vectors))
            
            # Save updated index and data
            faiss.write_index(index, "material/rag_word2vec_index.faiss")
            with open("material/rag_word2vec_texts.pkl", "wb") as f:
                pickle.dump(texts, f)
            with open("material/rag_word2vec_metadata.pkl", "wb") as f:
                pickle.dump(metadata, f)
                
            print(f"Added {len(new_vectors)} chunks from {filename}")
            return True
        else:
            print("No valid vectors created from document")
            return False
            
    except Exception as e:
        print(f"Error adding document to index: {e}")
        return False

def add_pdf_to_index(pdf_path, max_words=200, overlap=50):
    """
    Process and add a PDF document to the FAISS index
    
    Args:
        pdf_path: Path to PDF file
        max_words: Words per chunk
        overlap: Word overlap between chunks
    """
    try:
        # Extract text from PDF
        content = read_pdf_to_text(pdf_path)
        if not content:
            print(f"No content extracted from {pdf_path}")
            return False
            
        # Use the base filename for metadata
        filename = os.path.basename(pdf_path)
        
        # Add the extracted content
        return add_document_to_index(content, filename, max_words, overlap)
        
    except Exception as e:
        print(f"Error processing PDF {pdf_path}: {e}")
        return False

# Test 
if __name__ == "__main__":
    query = "What is machine learning?"
    results = query_faiss(query)

    for chunk, (filename, idx), dist in results[:3]:
        print(f"\n📄 From {filename} [chunk {idx}] (score={dist:.4f}):\n{chunk}")
