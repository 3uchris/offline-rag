"""
Stage 1: Ingest - read documents + split into chunks

Input: documents under "data/raw" (PDF, docs, md, txt)

Output: chunks of text, each being (text, source info)

Why split into chunks?
An LLM's context is limited, and retrieval needs to find the relevant small passage - 
not feed in a whole 100-page document. So we split first and embed each small piece separately.

Why overlap?
If a cut lands mid-sentence, meaning breaks. A little overlap between adjacent chunks keeps imortant
sentences from being split apart.
"""

from dataclasses import dataclass
from pathlib import Path

import fitz                     # pymupdf, used to read PDFs
from docx import Document       # python-docx, used to read .docx

from config import settings

@dataclass
class Chunk:
    """
    One chunk = a piece of text + where it came from(so answer can cite sources)
    """
    text: str
    source: str         # e.g. "file.pdf"
    chunk_index: int    # which chunk number within its document

def read_pdf(file_path: Path) -> str:
    """
    Read a PDF and return its plain text.
    """
    doc = fitz.open(file_path)
    text = ""
    for page in doc:
        text += page.get_text()
    doc.close()
    return text

def read_docx(file_path: Path) -> str:
    """
    Read a .docx file and return its plain text.
    """
    doc = Document(file_path)
    text = "\n".join([para.text for para in doc.paragraphs])
    return text

def read_txt(file_path: Path) -> str:
    """
    Read a .txt file or markdown file and return its plain text.
    """
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read_text(encoding="utf-8")

def load_file(file_path: Path) -> str:
    """
    Load a file based on its extension and return its text content.
    """
    if file_path.suffix.lower() == ".pdf":
        return read_pdf(file_path)
    elif file_path.suffix.lower() == ".docx":
        return read_docx(file_path)
    elif file_path.suffix.lower() in [".txt", ".md"]:
        return read_txt(file_path)
    else:
        raise ValueError(f"Unsupported file type: {file_path.suffix}")
    
def chunk_text(text: str, chunk_size: int, chunk_overlap: int) -> list[str]:
    """
    Split text into chunks of a certain size with some overlap.
    Use settings.chunk_size and setting.chunk_overlap from config.py for tunable parameters.
    """
    chunks = []
    start = 0
    text_length = len(text)
    
    while start < text_length:
        end = min(start + chunk_size, text_length)
        chunk = text[start:end]
        chunks.append(chunk)
        start += chunk_size - chunk_overlap  # move forward by chunk_size minus the overlap
        # Advanced: split by sentence/paragragh, but use a char window first to get the pipeline running.)
        # For example, if chunk_size=1000 and chunk_overlap=200, the first chunk is text[0:1000], the second chunk starts at 800 (1000-200) and goes to 1800, etc. This way, each chunk overlaps with the previous one by 200 characters.
        # This is a simple sliding window approach. More advanced methods could try to split on sentence or paragraph boundaries, but this char-based method is a good start and ensures we don't lose information at the chunk edges.
        # Note: the last chunk may be shorter than chunk_size, which is fine.
        #   
    return chunks

def ingest_all() -> list[Chunk]:
    """
    Main Entry: scan all files under /raw/, read and chunk them, return one big chunk list.
    build_index() call this.
    """
    all_chunks = []
    for file_path in settings.raw_docs_dir.glob("*.*"):  # get all files in the raw docs directory
        try:
            text = load_file(file_path)  # read the file content as text
            chunks = chunk_text(text, settings.chunk_size, settings.chunk_overlap)  # split into chunks
            for i, chunk in enumerate(chunks):
                all_chunks.append(Chunk(text=chunk, source=file_path.name, chunk_index=i))
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
    print(all_chunks[:2])  # print the first 2 chunks for a sanity check
    return all_chunks