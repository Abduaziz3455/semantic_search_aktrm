"""Embedding for BAAI/bge-m3.

bge-m3 dense retrieval is symmetric: no query/document prefixes, and pooling is
the [CLS] token (not mean). sentence-transformers applies CLS pooling and L2
normalization for us; the Java app reproduces the same contract by hand. All
embedding goes through this module so it cannot drift.
"""
import os
from functools import lru_cache
from pathlib import Path

from sentence_transformers import SentenceTransformer

# News items are short; capping the sequence keeps CPU inference fast and must
# match the Java tokenizer's max length so both truncate identically.
MAX_SEQ_LENGTH = 512

# The model ships in the repo at models/bge-m3, so nothing is downloaded at
# runtime. Mirrors Embedder.defaultModelDir() on the Java side.
_REPO_MODEL_DIR = Path(__file__).resolve().parents[2] / "models" / "bge-m3"


def _resolve_model() -> str:
    # docker compose sets MODEL_PATH; otherwise use the repo copy, and fall back
    # to the hub name only if the copy is missing.
    env = os.environ.get("MODEL_PATH")
    if env:
        return env
    if (_REPO_MODEL_DIR / "config.json").exists():
        return str(_REPO_MODEL_DIR)
    return "BAAI/bge-m3"


MODEL_NAME = _resolve_model()


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
