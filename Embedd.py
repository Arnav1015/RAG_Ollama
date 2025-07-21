import os
import pandas as pd
import fitz
import nltk
from uuid import uuid4
from typing import List, Dict, Tuple
from nltk.tokenize import sent_tokenize
from Vector_Database_ChromaDB import ChromaVectorStore
import re
import cv2
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter, ImageOps
import io
import shutil

# Lazy loading of heavy dependencies
_embedding_model = None
_ocr_model = None
_tesseract_configured = False

def get_embedding_model():
    global _embedding_model
    if _embedding_model is None:
        try:
            from sentence_transformers import SentenceTransformer
            import torch
            
            # Force CPU usage to avoid CUDA issues
            device = "cpu"
            print(f"Loading embedding model on: {device}")
            
            _embedding_model = SentenceTransformer("all-MiniLM-L6-v2", device=device)
            
            # Ensure the model stays on CPU
            _embedding_model = _embedding_model.to(device)
            
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

def configure_tesseract():
    """Configure Tesseract OCR path"""
    global _tesseract_configured
    if _tesseract_configured:
        return True
        
    try:
        import pytesseract
        
        # Try different possible paths
        tesseract_path = shutil.which('tesseract')
        if tesseract_path:
            pytesseract.pytesseract.tesseract_cmd = tesseract_path
            print(f"Found tesseract at: {tesseract_path}")
            _tesseract_configured = True
            return True
        else:
            # Try common installation paths
            possible_paths = [
                r'C:\Program Files\Tesseract-OCR\tesseract.exe',
                r'C:\Program Files (x86)\Tesseract-OCR\tesseract.exe',
                r'C:\Users\arnas\AppData\Local\Programs\Tesseract-OCR\tesseract.exe'
            ]
            for path in possible_paths:
                if os.path.exists(path):
                    pytesseract.pytesseract.tesseract_cmd = path
                    print(f"Found tesseract at: {path}")
                    _tesseract_configured = True
                    return True
            
            print("Tesseract not found. Please install it first.")
            return False
    except ImportError:
        print("pytesseract not installed. Please install: pip install pytesseract")
        return False

# Download NLTK data if needed
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download("punkt")

def extract_text_from_image(file_path: str) -> str:
    ocr_model = get_ocr_model()
    result = ocr_model.ocr(file_path, cls=True)
    return " ".join([line[1][0] for line in result[0]]) if result[0] else ""

def is_meaningful_text(text):
    """Check if extracted text is meaningful (not just OCR noise)"""
    if not text or len(text.strip()) < 50:
        return False
    
    # Check for reasonable word/character ratio
    words = text.split()
    if len(words) < 10:
        return False
    
    # Check for too many single characters (OCR artifacts)
    single_chars = sum(1 for word in words if len(word) == 1)
    if single_chars / len(words) > 0.3:
        return False
    
    # Check for reasonable alphabetic content
    alpha_chars = sum(1 for char in text if char.isalpha())
    if alpha_chars / len(text) < 0.3:
        return False
    
    return True

def process_page_with_advanced_ocr(page, page_num):
    """Process a single page with advanced OCR (from testpadddle.py)"""
    try:
        import pytesseract
        
        # Convert page to high-resolution image
        pix = page.get_pixmap(matrix=fitz.Matrix(3, 3))
        
        # Convert to PIL Image
        img_data = pix.tobytes("ppm")
        img = Image.open(io.BytesIO(img_data))
        
        # Advanced preprocessing
        try:
            img_gray = img.convert('L')
            img_cv = cv2.cvtColor(np.array(img_gray), cv2.COLOR_GRAY2BGR)
            gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
            
            # Noise removal
            denoised = cv2.medianBlur(gray, 3)
            
            # Thresholding
            _, binary = cv2.threshold(denoised, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            
            # Morphological operations
            kernel = np.ones((1,1), np.uint8)
            cleaned = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
            
            # Deskew correction
            coords = np.column_stack(np.where(cleaned > 0))
            if len(coords) > 0:
                angle = cv2.minAreaRect(coords)[-1]
                if angle < -45:
                    angle = -(90 + angle)
                else:
                    angle = -angle
                
                if abs(angle) > 0.5:
                    print(f"  Detected skew angle: {angle:.2f}°, correcting...")
                    (h, w) = cleaned.shape[:2]
                    center = (w // 2, h // 2)
                    
                    # Calculate new size to prevent trimming
                    angle_rad = np.radians(abs(angle))
                    new_w = int(h * np.sin(angle_rad) + w * np.cos(angle_rad))
                    new_h = int(h * np.cos(angle_rad) + w * np.sin(angle_rad))
                    new_center = (new_w // 2, new_h // 2)
                    
                    M = cv2.getRotationMatrix2D(center, angle, 1.0)
                    M[0, 2] += new_center[0] - center[0]
                    M[1, 2] += new_center[1] - center[1]
                    
                    cleaned = cv2.warpAffine(cleaned, M, (new_w, new_h), 
                                           flags=cv2.INTER_CUBIC, 
                                           borderMode=cv2.BORDER_CONSTANT, 
                                           borderValue=255)
            
            img_processed = Image.fromarray(cleaned)
            img_processed = ImageOps.autocontrast(img_processed)
            
        except Exception as e:
            print(f"  Preprocessing failed: {e}, using simple conversion")
            img_processed = img.convert('L')
        
        # OCR with multiple configs
        configs = [
            r'--oem 3 --psm 6 -c preserve_interword_spaces=1',
            r'--oem 3 --psm 4',
            r'--oem 3 --psm 6',
        ]
        
        best_text = ""
        best_length = 0
        
        for config in configs:
            try:
                text = pytesseract.image_to_string(img_processed, lang='eng', config=config)
                if len(text.strip()) > best_length:
                    best_text = text
                    best_length = len(text.strip())
            except:
                continue
        
        return format_for_llm_embedding(best_text) if best_text else pytesseract.image_to_string(img_processed, lang='eng')
        
    except ImportError:
        print("  pytesseract not available, falling back to PaddleOCR")
        # Fallback to existing PaddleOCR method
        return extract_text_from_image_page(page)

def extract_text_from_image_page(page):
    """Extract text from a PDF page using PaddleOCR"""
    try:
        # Convert page to image
        pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))
        img_data = pix.tobytes("png")
        
        # Save temporarily and process with PaddleOCR
        temp_path = f"temp_page_{os.getpid()}.png"
        with open(temp_path, "wb") as f:
            f.write(img_data)
        
        text = extract_text_from_image(temp_path)
        
        # Clean up
        if os.path.exists(temp_path):
            os.remove(temp_path)
            
        return text
    except Exception as e:
        print(f"  Error in PaddleOCR extraction: {e}")
        return ""

def format_for_llm_embedding(text):
    """Format extracted text for optimal LLM embedding and understanding"""
    lines = text.split('\n')
    formatted_lines = []
    current_table = []
    in_table = False
    
    for i, line in enumerate(lines):
        line = line.strip()
        if not line:
            continue
            
        # Detect table headers and rows
        if is_table_header(line):
            if current_table:
                # Process previous table
                formatted_lines.extend(format_table(current_table))
                current_table = []
            current_table.append(line)
            in_table = True
        elif in_table and is_table_row(line):
            current_table.append(line)
        elif in_table and not is_table_continuation(line):
            # End of table
            if current_table:
                formatted_lines.extend(format_table(current_table))
                current_table = []
            in_table = False
            formatted_lines.append(clean_text_line(line))
        else:
            if current_table:
                formatted_lines.extend(format_table(current_table))
                current_table = []
                in_table = False
            formatted_lines.append(clean_text_line(line))
    
    # Handle any remaining table
    if current_table:
        formatted_lines.extend(format_table(current_table))
    
    return '\n'.join(formatted_lines)

def is_table_header(line):
    """Detect if a line is likely a table header"""
    header_patterns = [
        r'^[A-Z][^|]*\|[^|]*\|',  # Multiple columns with pipes
        r'Date.*Service.*Product',  # Common header patterns
        r'Wellsite.*Date.*Post.*processed',
        r'Service.*product.*submission',
        r'Details.*of.*Deliverables',
    ]
    return any(re.search(pattern, line, re.IGNORECASE) for pattern in header_patterns)

def is_table_row(line):
    """Detect if a line is likely a table row"""
    row_patterns = [
        r'\|.*\|',  # Contains pipes
        r'\s{3,}.*\s{3,}',  # Multiple large spaces (column separation)
        r'\d{1,2}[-/]\w{3}[-/]\d{2,4}',  # Date patterns
        r'N/A.*N/A',  # Multiple N/A values
        r'Acceptable|Not Acceptable',  # Status values
        r'[A-Z]{2,}.*[A-Z]{2,}',  # Multiple abbreviations
    ]
    return any(re.search(pattern, line) for pattern in row_patterns)

def is_table_continuation(line):
    """Check if line is continuation of previous table row"""
    continuation_patterns = [
        r'^[A-Z]{2,}$',  # All caps abbreviations
        r'^\([^)]+\)$',  # Parenthetical additions
        r'^-+$',  # Dash lines
        r'^[A-Z]+:',  # Labels like "CD/DVD:"
    ]
    return any(re.search(pattern, line) for pattern in continuation_patterns)

def format_table(table_lines):
    """Format table lines for LLM understanding"""
    if not table_lines:
        return []
    
    formatted = ["TABLE_START"]
    
    # Merge multi-line rows
    merged_rows = merge_table_rows(table_lines)
    
    for i, row in enumerate(merged_rows):
        if i == 0:
            formatted.append(f"HEADER: {clean_table_row(row)}")
        else:
            formatted.append(f"ROW: {clean_table_row(row)}")
    
    formatted.append("TABLE_END")
    return formatted

def merge_table_rows(table_lines):
    """Merge multi-line table rows into single rows"""
    merged = []
    current_row = ""
    
    for line in table_lines:
        # Check if this line starts a new row or continues previous
        if is_new_table_row(line) and current_row:
            merged.append(current_row.strip())
            current_row = line
        else:
            if current_row:
                current_row += " " + line
            else:
                current_row = line
    
    if current_row:
        merged.append(current_row.strip())
    
    return merged

def is_new_table_row(line):
    """Determine if line starts a new table row"""
    new_row_patterns = [
        r'^\d',  # Starts with number
        r'^[A-Z][a-z]',  # Starts with capital letter followed by lowercase
        r'^\w+[-/]\w+[-/]',  # Date-like pattern at start
        r'^[A-Z]{2,}-',  # Starts with abbreviation and dash
    ]
    return any(re.search(pattern, line) for pattern in new_row_patterns)

def clean_table_row(row):
    """Clean and standardize table row format"""
    # Normalize separators
    row = re.sub(r'\s*\|\s*', ' | ', row)
    row = re.sub(r'\s{2,}', ' | ', row)
    
    # Clean common OCR errors
    row = row.replace('|', ' | ')
    row = re.sub(r'\s*\|\s*', ' | ', row)
    
    # Remove leading/trailing separators
    row = re.sub(r'^\s*\|\s*', '', row)
    row = re.sub(r'\s*\|\s*$', '', row)
    
    return row.strip()

def clean_text_line(line):
    """Clean regular text lines"""
    # Remove excessive whitespace
    line = re.sub(r'\s+', ' ', line)
    
    # Fix common OCR errors
    line = line.replace('|', 'I')  # Common misread
    
    return line.strip()

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
    """Process PDFs - simple extraction for normal PDFs, advanced OCR only for scanned PDFs"""
    doc = fitz.open(path)
    chunks, metadatas = [], []
    requires_ocr = False
    
    print(f"Processing PDF: {os.path.basename(path)}")
    
    for page_num in range(len(doc)):
        print(f"  Processing page {page_num + 1}/{len(doc)}")
        page = doc[page_num]
        
        # Try text extraction first
        text = page.get_text()
        
        # Check if we got meaningful text
        if is_meaningful_text(text):
            print(f"  Page {page_num + 1}: Using direct text extraction")
            # Keep original simple processing for normal PDFs
            formatted_text = text
        else:
            print(f"  Page {page_num + 1}: Text extraction insufficient, using OCR")
            requires_ocr = True
            
            # Use advanced OCR processing only for scanned PDFs
            if configure_tesseract():
                raw_ocr_text = process_page_with_advanced_ocr(page, page_num)
                # Apply advanced formatting only to OCR text
                formatted_text = format_for_llm_embedding(raw_ocr_text)
            else:
                # Fallback to PaddleOCR
                raw_text = extract_text_from_image_page(page)
                formatted_text = format_for_llm_embedding(raw_text)
        
        # Split into chunks - different handling for OCR vs normal text
        if formatted_text.strip():
            if requires_ocr and ("TABLE_START" in formatted_text or "TABLE_END" in formatted_text):
                # Advanced chunking for OCR text with tables
                lines = formatted_text.split('\n')
                current_chunk = []
                in_table = False
                
                for line in lines:
                    if line.startswith("TABLE_START"):
                        # Save any current chunk
                        if current_chunk:
                            chunk_text = '\n'.join(current_chunk)
                            sentences = sent_tokenize(chunk_text)
                            for i in range(0, len(sentences), max_sentences_per_chunk):
                                sentence_chunk = " ".join(sentences[i:i+max_sentences_per_chunk])
                                if sentence_chunk.strip():
                                    chunks.append(sentence_chunk)
                                    metadatas.append({
                                        "source": "pdf_ocr",
                                        "filename": os.path.basename(path),
                                        "page": page_num + 1,
                                        "chunk": len(chunks),
                                        "type": "text"
                                    })
                            current_chunk = []
                        
                        # Start table collection
                        in_table = True
                        table_content = [line]
                    
                    elif line.startswith("TABLE_END"):
                        # End table and save as single chunk
                        table_content.append(line)
                        table_text = '\n'.join(table_content)
                        chunks.append(table_text)
                        metadatas.append({
                            "source": "pdf_ocr",
                            "filename": os.path.basename(path),
                            "page": page_num + 1,
                            "chunk": len(chunks),
                            "type": "table"
                        })
                        in_table = False
                        table_content = []
                    
                    elif in_table:
                        table_content.append(line)
                    
                    else:
                        current_chunk.append(line)
                
                # Handle any remaining content
                if current_chunk:
                    chunk_text = '\n'.join(current_chunk)
                    sentences = sent_tokenize(chunk_text)
                    for i in range(0, len(sentences), max_sentences_per_chunk):
                        sentence_chunk = " ".join(sentences[i:i+max_sentences_per_chunk])
                        if sentence_chunk.strip():
                            chunks.append(sentence_chunk)
                            metadatas.append({
                                "source": "pdf_ocr",
                                "filename": os.path.basename(path),
                                "page": page_num + 1,
                                "chunk": len(chunks),
                                "type": "text"
                            })
            else:
                # Simple sentence-based chunking for normal PDFs or OCR without tables
                sentences = sent_tokenize(formatted_text)
                for i in range(0, len(sentences), max_sentences_per_chunk):
                    chunk = " ".join(sentences[i:i+max_sentences_per_chunk])
                    if chunk.strip():
                        chunks.append(chunk)
                        metadatas.append({
                            "source": "pdf_ocr" if requires_ocr else "pdf_text",
                            "filename": os.path.basename(path),
                            "page": page_num + 1,
                            "chunk": i // max_sentences_per_chunk,
                            "type": "text"
                        })
    
    doc.close()
    method = "HYBRID" if requires_ocr else "TEXT_EXTRACTION"
    print(f"  Completed PDF processing using: {method}")
    print(f"  Generated {len(chunks)} chunks")
    
    return chunks, metadatas

def load_image_chunk(path: str) -> Tuple[List[str], List[Dict]]:
    text = extract_text_from_image(path)
    # Apply advanced formatting to image OCR text
    formatted_text = format_for_llm_embedding(text) if text.strip() else text
    return [formatted_text], [{
        "source": "image",
        "filename": os.path.basename(path),
        "chunk": 0
    }] if formatted_text.strip() else ([], [])

def load_text_chunks(path: str, max_sentences_per_chunk: int = 5) -> Tuple[List[str], List[Dict]]:
    with open(path, 'r', encoding='utf-8', errors='ignore') as f:
        text = f.read()
    
    # Keep original simple processing for text files
    formatted_text = text
    sentences = sent_tokenize(formatted_text)
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

def embed_and_store(chunks: List[str], metadatas: List[Dict], store: ChromaVectorStore, batch_size: int = 5):
    """Store chunks in smaller batches to avoid ChromaDB segfaults"""
    embedding_model = get_embedding_model()
    
    # Process in smaller batches
    for i in range(0, len(chunks), batch_size):
        batch_chunks = chunks[i:i+batch_size]
        batch_metas = metadatas[i:i+batch_size]
        batch_ids = [str(uuid4()) for _ in batch_chunks]
        
        try:
            print(f"Processing batch {i//batch_size + 1}/{(len(chunks) + batch_size - 1)//batch_size} ({len(batch_chunks)} chunks)")
            
            # Embed the batch
            embeddings = embedding_model.encode(batch_chunks, show_progress_bar=True, convert_to_numpy=True)
            embeddings_list = embeddings.tolist()
            
            # Store to ChromaDB
            store.add_documents(batch_ids, batch_chunks, embeddings_list, batch_metas)
            print(f"✅ Stored batch {i//batch_size + 1}: {len(batch_chunks)} chunks")
            
            # Clean up memory
            del embeddings, embeddings_list
            import gc
            gc.collect()
            
        except Exception as e:
            print(f"❌ Error storing batch {i//batch_size + 1}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    print(f"✅ Successfully processed {len(chunks)} chunks in {(len(chunks) + batch_size - 1)//batch_size} batches")

def process_file(file_path: str, store: ChromaVectorStore, existing_files: set):
    filename = os.path.basename(file_path)
    if filename in existing_files:
        print(f"⏭️ Skipping duplicate: {filename}")
        return

    try:
        # Test ChromaDB connection before processing
        print(f"📊 Testing ChromaDB connection...")
        existing_count = len(store.get_all_filenames())
        print(f"📊 ChromaDB has {existing_count} existing files")
        
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
            print(f"❌ Unsupported file type: {file_path}")
            return

        print(f"📄 Loaded {len(chunks)} chunks from {file_path}")
        
        if chunks:
            embed_and_store(chunks, metas, store)
        else:
            print(f"⚠️ No data extracted from: {file_path}")
            
    except Exception as e:
        print(f"❌ Error processing {file_path}: {e}")
        import traceback
        traceback.print_exc()

def process_folder(folder_path: str, store: ChromaVectorStore):
    existing_files = store.get_all_filenames()
    for root, _, files in os.walk(folder_path):
        for filename in files:
            if filename.endswith((".csv", ".xlsx", ".pdf", ".jpg", ".jpeg", ".png", ".txt", ".md", ".json")):
                full_path = os.path.join(root, filename)
                print(f"📁 Processing: {full_path}")
                process_file(full_path, store, existing_files)

if __name__ == "__main__":
    folder_path = "C:/Users/Arnav/Documents/text"
    folder_path = "./text"
    store = ChromaVectorStore(collection_name="hybrid_data", persist_directory="chroma_db")
    process_folder(folder_path, store)
