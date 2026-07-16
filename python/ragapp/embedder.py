"""Embedding for BAAI/bge-m3.

bge-m3 dense retrieval is symmetric: no query/document prefixes, and pooling is
the [CLS] token (not mean). sentence-transformers applies CLS pooling and L2
normalization for us; the Java app reproduces the same contract by hand. All
embedding goes through this module so it cannot drift.
"""
from functools import lru_cache

from sentence_transformers import SentenceTransformer

MODEL_NAME = "BAAI/bge-m3"
# News items are short; capping the sequence keeps CPU inference fast and must
# match the Java tokenizer's max length so both truncate identically.
MAX_SEQ_LENGTH = 512


@lru_cache(maxsize=1)
def _model() -> SentenceTransformer:
    model = SentenceTransformer(MODEL_NAME)
    model.max_seq_length = MAX_SEQ_LENGTH
    return model


def embed_documents(texts: list[str]) -> list[list[float]]:
    return _model().encode(texts, normalize_embeddings=True).tolist()


def embed_query(text: str) -> list[float]:
    return _model().encode(text, normalize_embeddings=True).tolist()


def embed_raw(text: str) -> list[float]:
    """Same as the others (bge-m3 uses no prefix); kept as the explicit entry
    point for the cross-runtime vector parity test."""
    return _model().encode(text, normalize_embeddings=True).tolist()
