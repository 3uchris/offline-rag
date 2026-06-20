"""
Stage 2: Embed = turn text chunks into vectors

Input: a list of text(chunk text, or the users' question)

Output: a list of vectors(list of floats); each vector is the text's semantic coordinates

Core Idea:
Texts with similar meaning have vectors that sit close together.
so "finding [relevant] chunks" = "find the [nearest-vecotor] chunks"
At indexing time we embed all chunks; at query time we embed the question, then compare distances.

Notes:
The same model must embed both chunks and questions, so the vectors share one space.
This model is fast enough on CPU, no GPU needed.
"""

from sentence_transformers import SentenceTransformer

from config import settings

# Load the model once globally(loading is slow; dont reload on every call)
# It loads the first time this module is used.

_model: SentenceTransformer | None = None

def get_model() -> SentenceTransformer:
    """
    Lazy-load the embedding model. Load once, then reuse it.
    """
    global _model
    if _model is None:
        print(f"Loading embedding model {settings.embedding_model}...")
        _model = SentenceTransformer(settings.embedding_model)
    return _model

def embed_texts(texts: list[str], show_progress: bool = False) -> list[list[float]]:
    """
    A batch of texts -> a batch of vectors
    Used at indexing time(embed many chunks at once)
    """
    model = get_model()
    return model.encode(texts, show_progress_bar=show_progress).tolist()  # convert numpy array to list for Chroma

def embed_query(text: str) -> list[float]:
    """
    A single question -> a single vector. Used at query time.
    """
    return embed_texts([text])[0]