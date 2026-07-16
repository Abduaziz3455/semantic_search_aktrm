"""Qdrant collection lifecycle, upsert, and search.

`search()` is the single retrieval entry point: the CLI (search.py) and the
Gradio UI (app.py) both call it, so ranking and threshold behaviour cannot
diverge between them.
"""
import os
from dataclasses import dataclass

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

from .embedder import embed_documents, embed_query

COLLECTION = "faq"
DIMENSIONS = 384
DEFAULT_LIMIT = 3
DEFAULT_THRESHOLD = 0.80

# localhost for a native run; docker compose sets this to http://qdrant:6333.
QDRANT_URL = os.environ.get("QDRANT_URL", "http://localhost:6333")


@dataclass
class Hit:
    score: float
    question: str
    answer: str
    tags: list[str]


def client() -> QdrantClient:
    return QdrantClient(url=QDRANT_URL)


def recreate_collection(c: QdrantClient) -> None:
    if c.collection_exists(COLLECTION):
        c.delete_collection(COLLECTION)
    c.create_collection(
        collection_name=COLLECTION,
        vectors_config=VectorParams(size=DIMENSIONS, distance=Distance.COSINE),
    )


def upsert_entries(c: QdrantClient, entries: list[dict]) -> None:
    # Question carries the user's phrasing, answer carries the vocabulary
    # they might search by — embed both as one chunk.
    vectors = embed_documents(
        [e["question"] + "\n" + e["answer"] for e in entries]
    )
    points = [
        PointStruct(
            id=e["id"],
            vector=v,
            payload={"question": e["question"], "answer": e["answer"], "tags": e["tags"]},
        )
        for e, v in zip(entries, vectors)
    ]
    c.upsert(collection_name=COLLECTION, points=points, wait=True)


def search(
    query_text: str,
    limit: int = DEFAULT_LIMIT,
    threshold: float = DEFAULT_THRESHOLD,
) -> list[Hit]:
    """Returns hits sorted by score descending; empty list means no hit
    cleared the threshold (i.e. no confident match)."""
    results = client().query_points(
        collection_name=COLLECTION,
        query=embed_query(query_text),
        limit=limit,
        score_threshold=threshold,
    ).points
    return [
        Hit(
            score=r.score,
            question=r.payload["question"],
            answer=r.payload["answer"],
            tags=r.payload.get("tags", []),
        )
        for r in results
    ]
