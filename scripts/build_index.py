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

def main() -> None:
    print("Reading and chunking documents...")
    chunks = ingest_all()
    print(f"{len(chunks)} chunks total.")

if __name__ == "__main__":
    main()