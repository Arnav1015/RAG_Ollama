import fitz  # PyMuPDF
import pytesseract
from PIL import Image, ImageEnhance, ImageFilter, ImageOps
import os
import io
import shutil
import cv2
import numpy as np
import re

def process_pdf_document(pdf_path: str):
    """Process both text-based and scanned PDFs intelligently"""
    print("=== SMART PDF PROCESSING ===")
    
    # First, try to extract text directly (for normal PDFs)
    doc = fitz.open(pdf_path)
    all_text = []
    requires_ocr = False
    
    for page_num in range(len(doc)):
        print(f"\n=== Analyzing Page {page_num + 1} ===")
        page = doc[page_num]
        
        # Try text extraction first
        text = page.get_text()
        
        # Check if we got meaningful text
        if is_meaningful_text(text):
            print(f"Page {page_num + 1}: Using direct text extraction")
            formatted_text = format_for_llm_embedding(text)
            all_text.append(formatted_text)
        else:
            print(f"Page {page_num + 1}: Text extraction insufficient, using OCR")
            requires_ocr = True
            # Use OCR for this page
            ocr_text = process_page_with_ocr(page, page_num)
            formatted_text = format_for_llm_embedding(ocr_text)
            all_text.append(formatted_text)
    
    doc.close()
    
    # Create output
    method = "HYBRID" if requires_ocr else "TEXT_EXTRACTION"
    structured_output = create_llm_friendly_output(all_text, pdf_path, method)
    
    # Save files
    save_outputs(all_text, structured_output, pdf_path, method)
    
    return structured_output

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

def process_page_with_ocr(page, page_num):
    """Process a single page with OCR (extracted from main function)"""
    # Convert page to high-resolution image
    pix = page.get_pixmap(matrix=fitz.Matrix(3, 3))
    
    # Convert to PIL Image
    img_data = pix.tobytes("ppm")
    img = Image.open(io.BytesIO(img_data))
    
    # Save original for debugging
    img.save(f"debug_original_page_{page_num + 1}.png")
    
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
                print(f"Detected skew angle: {angle:.2f}°, correcting...")
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
        img_processed.save(f"debug_processed_page_{page_num + 1}.png")
        
    except Exception as e:
        print(f"Preprocessing failed: {e}, using simple conversion")
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
    
    return best_text if best_text else pytesseract.image_to_string(img_processed, lang='eng')

def create_llm_friendly_output(all_text, pdf_path, method="OCR"):
    """Create final structured output optimized for LLM embedding"""
    output = []
    
    # Document header
    output.append("DOCUMENT_TYPE: Service Ticket")
    output.append(f"SOURCE_FILE: {os.path.basename(pdf_path)}")
    output.append(f"EXTRACTION_METHOD: {method}")
    output.append("")
    
    # Process each page
    for page_num, text in enumerate(all_text, 1):
        output.append(f"PAGE_{page_num}_START")
        
        # Split into sections
        lines = text.split('\n')
        current_section = []
        section_type = "TEXT"
        
        for line in lines:
            if line.startswith("TABLE_START"):
                if current_section:
                    output.append(f"SECTION_TYPE: {section_type}")
                    output.extend(current_section)
                    output.append("")
                current_section = []
                section_type = "TABLE"
            elif line.startswith("TABLE_END"):
                if current_section:
                    output.append(f"SECTION_TYPE: {section_type}")
                    output.extend(current_section)
                    output.append("")
                current_section = []
                section_type = "TEXT"
            elif line.strip():
                current_section.append(line)
        
        # Add any remaining section
        if current_section:
            output.append(f"SECTION_TYPE: {section_type}")
            output.extend(current_section)
            output.append("")
        
        output.append(f"PAGE_{page_num}_END")
        output.append("")
    
    return '\n'.join(output)

def save_outputs(all_text, structured_output, pdf_path, method):
    """Save both raw and structured outputs"""
    # Raw output
    raw_output_file = pdf_path.replace('.pdf', f'_raw_{method.lower()}_output.txt')
    with open(raw_output_file, 'w', encoding='utf-8') as f:
        f.write('\n\n=== PAGE BREAK ===\n\n'.join(all_text))
    
    # Structured output
    structured_output_file = pdf_path.replace('.pdf', f'_structured_for_llm_{method.lower()}.txt')
    with open(structured_output_file, 'w', encoding='utf-8') as f:
        f.write(structured_output)
    
    print(f"\n=== LLM-READY STRUCTURED OUTPUT ({method}) ===")
    print(structured_output)
    print(f"\nRaw text saved to: {raw_output_file}")
    print(f"LLM-ready text saved to: {structured_output_file}")

# Keep the original function for backward compatibility
def process_scanned_pdf_with_ocr(pdf_path: str):
    """Original function - use process_pdf_document instead"""
    return process_pdf_document(pdf_path)

# Main execution
if __name__ == "__main__":
    pdf_path = "Service_Ticket_1.pdf"
    print("=== SMART PDF PROCESSING WITH LLM FORMATTING ===")
    process_pdf_document(pdf_path)
