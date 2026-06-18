"""
[Central Configuration] for the application. This module defines all the configuration settings


Keep every tunable parameters(numbers and names) in this single place, not scattered across files.

When you swap models, change chunk size, or tune top_k, edit only this file.

Precedence: Environment variables(.env) >  the defaults here.
So the same codebase can be used in development and production by just changing environment variables(different .env).
"""

from pathlib import Path
from pydantic_settings import BaseSettings

# Project Root(the folder this files lives in)
BASE_DIR = Path(__file__).resolve().parent

class Settings(BaseSettings):
    # --- Paths ---
    raw_docs_dir: Path = BASE_DIR / "data" / "raw"             # Raw documents go here
    vectorstore_dir: Path = BASE_DIR / "data" / "vectorstore"  # Vector index persists here

    # --- Chunking(Phrase 1) ---
    chunk_size: int = 1000                                     # Number of characters per chunk
    chunk_overlap: int = 200                                   # Number of overlapping characters between chunks(avoid cutting meaning)

    # --- Embedding(Phrase 2) ---
    embedding_model: str = "BAAI/bge-small-en-v1.5"            # Small, CPU-friendly model for embedding. Swap with a larger one if you have GPU and need better performance.
    # Air-gap note: download it on a networked machine first, then move the HF cache over

    # --- Vector Search(Phrase 3) ---
    collection_name: str = "documents"                          # Name of the Chroma collection to store document vectors
    top_k: int = 5                                              # how many nearest chunks to retrieve

    # --- LLM(Phrase 4) ---
    ollama_model: str = "http://localhost:11434"                # Ollama runs on the host. air-gap note: pull the model first, then move it over.
    llm_model: str = "qwen2.5:3b"                               # Small model, runs on a CPU server; swap later
    # airgap: pull the model first, then move it over.

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"

settings = Settings()