"""Embedding for intfloat/multilingual-e5-small.

E5 is asymmetric: documents and queries need different prefixes, and getting
them wrong degrades recall silently. All prefixing lives in this module so it
cannot drift between ingest, CLI search, and the Gradio UI.
"""
from functools import lru_cache

from sentence_transformers import SentenceTransformer

MODEL_NAME = "intfloat/multilingual-e5-small"
PASSAGE_PREFIX = "passage: "
QUERY_PREFIX = "query: "


@lru_cache(maxsize=1)
def _model() -> SentenceTransformer:
    return SentenceTransformer(MODEL_NAME)


def embed_documents(texts: list[str]) -> list[list[float]]:
    prefixed = [PASSAGE_PREFIX + t for t in texts]
    return _model().encode(prefixed, normalize_embeddings=True).tolist()


def embed_query(text: str) -> list[float]:
    return _model().encode(QUERY_PREFIX + text, normalize_embeddings=True).tolist()


def embed_raw(text: str) -> list[float]:
    """No prefix added — used only by the vector parity test, which passes an
    already-prefixed string so both apps embed byte-identical input."""
    return _model().encode(text, normalize_embeddings=True).tolist()
