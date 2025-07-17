import os
import pandas as pd
import fitz
import nltk
from uuid import uuid4
from typing import List, Dict, Tuple
from nltk.tokenize import sent_tokenize
from Vector_Database_ChromaDB import ChromaVectorStore

# Set environment variables to force CPU-only mode before importing torch
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['QT_QPA_PLATFORM'] = 'offscreen'  # Prevent Qt conflicts in threads

# Lazy loading of heavy dependencies
_embedding_model = None
_ocr_model = None

def get_embedding_model():
    global _embedding_model
    if _embedding_model is None:
        try:
            import torch
            from sentence_transformers import SentenceTransformer
            
            # Force CPU-only mode to avoid CUDA segmentation faults
            device = "cpu"  # Force CPU usage to avoid CUDA issues
            
            print(f"Loading SentenceTransformer on device: {device}")
            
            # Set torch to use single thread to avoid conflicts
            torch.set_num_threads(1)
            torch.set_num_interop_threads(1)
            
            # Disable automatic mixed precision which can cause issues
            torch.backends.cudnn.enabled = False
            torch.backends.cudnn.benchmark = False
            
            # Create model with explicit CPU device
            _embedding_model = SentenceTransformer("all-MiniLM-L6-v2", device=device)
            
            # Set model to evaluation mode and ensure it's on CPU
            _embedding_model.eval()
            _embedding_model = _embedding_model.to('cpu')
            
            # Disable gradient computation completely
            for param in _embedding_model.parameters():
                param.requires_grad = False
            
            # Force garbage collection after loading
            import gc
            gc.collect()
            
            print("✓ Model loaded and configured for safe CPU operation")
            
        except Exception as e:
            print(f"Warning: Could not load SentenceTransformer: {e}")
            print("Please install torch properly for your system.")
            # Return a dummy model that will raise an error if used
            _embedding_model = None
            raise ImportError(f"Failed to load SentenceTransformer: {e}")
    return _embedding_model

def get_ocr_model():
    global _ocr_model
    if _ocr_model is None:
        from paddleocr import PaddleOCR
        _ocr_model = PaddleOCR(use_angle_cls=True, lang="en")
    return _ocr_model

# Download NLTK data if needed
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download("punkt")

def extract_text_from_image(file_path: str) -> str:
    ocr_model = get_ocr_model()
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

def load_text_chunks(path: str, max_sentences_per_chunk: int = 5) -> Tuple[List[str], List[Dict]]:
    with open(path, 'r', encoding='utf-8', errors='ignore') as f:
        text = f.read()
    
    sentences = sent_tokenize(text)
    chunks, metadatas = [], []
    
    for i in range(0, len(sentences), max_sentences_per_chunk):
        chunk = " ".join(sentences[i:i+max_sentences_per_chunk])
        if chunk.strip():
            chunks.append(chunk)
            metadatas.append({
                "source": "text",
                "filename": os.path.basename(path),
                "chunk": i // max_sentences_per_chunk,
            })
    return chunks, metadatas

def embed_and_store(chunks: List[str], metadatas: List[Dict], store: ChromaVectorStore):
    try:
        print(f"Starting to process {len(chunks)} chunks...")
        
        # Load the embedding model only once at the beginning
        print("Loading embedding model...")
        embedding_model = get_embedding_model()
        print("✓ Embedding model loaded successfully")
        
        # Process one chunk at a time to isolate any problematic chunks
        print(f"Processing {len(chunks)} chunks individually for maximum safety")
        
        successful_count = 0
        for i, (chunk, metadata) in enumerate(zip(chunks, metadatas)):
            try:
                chunk_id = str(uuid4())
                print(f"Processing chunk {i+1}/{len(chunks)}")
                
                # Force garbage collection before each chunk
                import gc
                gc.collect()
                
                # Process single chunk with maximum safety
                try:
                    # Encode single chunk with very conservative settings
                    embedding = embedding_model.encode(
                        [chunk],  # Single chunk in a list
                        show_progress_bar=False,
                        convert_to_numpy=True,
                        device="cpu",
                        batch_size=1  # Process exactly one item
                    )
                    
                    # Convert to list immediately
                    embedding_list = embedding.tolist()
                    
                    # Clear the numpy array immediately
                    del embedding
                    gc.collect()
                    
                    # Store this single chunk immediately
                    store.add_documents([chunk_id], [chunk], embedding_list, [metadata])
                    print(f"✓ Chunk {i+1} stored successfully")
                    successful_count += 1
                    
                    # Clear variables
                    del embedding_list, chunk_id
                    gc.collect()
                    
                except Exception as chunk_error:
                    print(f"⚠ Error processing chunk {i+1}: {chunk_error}")
                    print(f"   Chunk preview: {chunk[:100]}...")
                    # Continue with next chunk instead of failing
                    continue
                    
            except Exception as outer_error:
                print(f"⚠ Outer error on chunk {i+1}: {outer_error}")
                continue
        
        print(f"Successfully processed {successful_count}/{len(chunks)} chunks")
        
        if successful_count == 0:
            raise RuntimeError("No chunks were successfully processed")
        
        # Final cleanup
        import gc
        gc.collect()
        
    except Exception as e:
        print(f"Error in embed_and_store: {e}")
        import traceback
        traceback.print_exc()
        raise


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
    elif ext in [".txt", ".md", ".json"]:
        chunks, metas = load_text_chunks(file_path)
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
            if filename.endswith((".csv", ".xlsx", ".pdf", ".jpg", ".jpeg", ".png", ".txt", ".md", ".json")):
                full_path = os.path.join(root, filename)
                print(f"Processing: {full_path}")
                process_file(full_path, store, existing_files)

if __name__ == "__main__":
    folder_path = "C:/Users/Arnav/Documents/text"
    folder_path = "./text"
    store = ChromaVectorStore(collection_name="hybrid_data", persist_directory="chroma_db")
    process_folder(folder_path, store)
