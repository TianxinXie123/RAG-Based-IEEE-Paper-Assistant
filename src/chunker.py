# src/chunker.py
from typing import List
import re

def smart_paragraph_chunker(text: str, chunk_size: int = 2000, chunk_overlap: int = 150) -> List[str]:
    # English Comment: Standard cleanup to remove multiple spaces
    text = re.sub(r' +', ' ', text) 
    paragraphs = text.split('\n\n')
    chunks = []
    current_chunk = ""

    for para in paragraphs:
        para = para.strip()
        if not para: continue
            
        # English Comment: Case 1: Paragraph fits in current chunk
        if len(current_chunk) + len(para) <= chunk_size:
            current_chunk += (para + "\n\n")
        else:
            # English Comment: Case 2: Paragraph is very long, split by sentences
            if len(para) > chunk_size:
                # English Comment: Use regex to split paragraph into individual sentences
                sentences = re.split(r'(?<=[.!?]) +', para)
                for sent in sentences:
                    if len(current_chunk) + len(sent) <= chunk_size:
                        current_chunk += sent + " "
                    else:
                        # English Comment: Save full chunk
                        chunks.append(current_chunk.strip())
                        
                        # --- IMPROVED OVERLAP LOGIC: Get last complete sentence ---
                        # English Comment: Split the finished chunk into sentences and take the last one
                        last_sents = re.split(r'(?<=[.!?]) +', current_chunk.strip())
                        overlap = last_sents[-1] + " " if last_sents else ""
                        
                        current_chunk = overlap + sent + " "
            else:
                # English Comment: Case 3: Paragraph doesn't fit, but isn't huge itself
                if current_chunk:
                    chunks.append(current_chunk.strip())
                
                # --- IMPROVED OVERLAP LOGIC: Get last complete sentence ---
                # English Comment: Extract the last full sentence for the next chunk's context
                last_sents = re.split(r'(?<=[.!?]) +', current_chunk.strip())
                overlap_text = last_sents[-1] + "\n\n" if last_sents else ""
                
                current_chunk = overlap_text + para + "\n\n"

    # English Comment: Save the final remaining chunk
    if current_chunk:
        chunks.append(current_chunk.strip())
        
    return chunks