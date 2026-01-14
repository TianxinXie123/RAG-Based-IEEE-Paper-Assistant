# src/pdf_loader.py

import fitz  # PyMuPDF
from pathlib import Path
import re

def clean_text(text: str) -> str:
    """
    Clean text by joining broken lines and removing hyphenations.
    """
    # English Comment: Remove hyphenation at the end of lines
    text = re.sub(r'(\w+)-\s*\n\s*(\w+)', r'\1\2', text)
    # English Comment: Replace single newlines with a space
    text = re.sub(r'(?<!\n)\n(?!\n)', ' ', text)
    # English Comment: Clean up multiple spaces
    text = re.sub(r' +', ' ', text)
    return text.strip()

def load_pdf_text(pdf_path: str) -> str:
    """
    Advanced PDF loader with Column Sorting, Header/Footer filtering, and Smart Joining.
    """
    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")

    doc = fitz.open(pdf_path)
    # English Comment: Using a single string to accumulate all content for easier punctuation checking
    full_text = ""

    # Define margins (in pixels) to ignore headers and footers
    HEADER_MARGIN = 30 
    FOOTER_MARGIN = 30

    for page_num in range(len(doc)):
        page = doc[page_num]
        page_width = page.rect.width
        page_height = page.rect.height
        
        blocks = page.get_text("blocks")
        
        # English Comment: Filter out blocks that are in the header or footer area
        filtered_blocks = []
        for b in blocks:
            x0, y0, x1, y1 = b[:4]
            if y0 > HEADER_MARGIN and y1 < (page_height - FOOTER_MARGIN):
                filtered_blocks.append(b)
        
        # English Comment: Sorting logic for Spanning, Left, and Right columns
        def get_block_order(b):
            x0, y0, x1, y1 = b[:4]
            block_center_x = (x0 + x1) / 2
            
            if (x1 - x0) > (page_width * 0.8): # Spanning block (Title/Abstract)
                column_id = 0
            elif block_center_x < (page_width / 2): # Left column
                column_id = 1
            else: # Right column
                column_id = 2
            return (column_id, y0)

        sorted_blocks = sorted(filtered_blocks, key=get_block_order)
        
        # English Comment: Smart joining logic for blocks within the current page
        for b in sorted_blocks:
            content = b[4].strip()
            if content:
                cleaned = clean_text(content)
                
                if not full_text:
                    full_text = cleaned
                else:
                    # English Comment: Check if the existing text ends with paragraph-ending punctuation
                    if full_text.endswith(('.', '!', '?', ':')):
                        # English Comment: Add double newline if it's likely a paragraph end
                        full_text += "\n\n" + cleaned
                    else:
                        # English Comment: Add a space if the sentence likely continues in the next block
                        full_text += " " + cleaned

    doc.close()
    return full_text

if __name__ == "__main__":
    # English Comment: Local module test
    import os
    current_dir = os.path.dirname(os.path.abspath(__file__))
    test_pdf = os.path.join(current_dir, "..", "data", "sample_ieee.pdf")
    

    """
    # English Comment: Test the smart joining logic
    print("--- Testing pdf_loader with Smart Block Joining ---")
    try:
        result = load_pdf_text(test_pdf)
        print(f"Total characters: {len(result)}")
        print("\n--- Preview (First 1000 chars) ---")
        print(result[:9000])
    except Exception as e:
        print(f"Error: {e}")
    """