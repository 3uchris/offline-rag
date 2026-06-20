"""
build_index.py - build the index (indexing phrase, run once)

Read all documents in data/raw, chunk them, embed, and store into the vector DB.
Re-run it whenever the documents change.

Usage(from the project root):
    python scripts/build_index.py
"""

import sys
from pathlib import Path

# Let python find scr/ and config.py (add project root to the path)
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.ingest import ingest_all
from src.embed import embed_texts
from src.store import add_chunks, count

def main() -> None:
    print("Reading and chunking documents...")
    chunks = ingest_all()
    print(f"{len(chunks)} chunks total.")

    print("Computing chunks...")
    texts = [c.text for c in chunks]
    vectors = embed_texts(texts, show_progress=True)

    print("Writing into the vector store...")
    add_chunks(chunks, vectors)
    print(f"Done. The index now holds {count()} chunks.")

if __name__ == "__main__":
    main()