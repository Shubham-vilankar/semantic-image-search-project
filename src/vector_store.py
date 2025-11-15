"""
Qdrant vector store wrapper.

Two operating modes, controlled by .env:
- LOCAL MODE (default): QDRANT_URL is blank -> Qdrant runs embedded, storing
  everything as files under QDRANT_LOCAL_PATH. No server process needed.
  This is what we use for local development (this step).
- SERVER MODE: QDRANT_URL is set (e.g. http://localhost:6333) -> connects to
  a real Qdrant server instead. We'll switch to this in Step 10 (Docker).

Note: local mode locks its storage directory while a client is open, so
don't run two scripts against it at the same time (e.g. ingest.py and
search.py simultaneously) — finish one before starting the other.
"""

import os
from pathlib import Path
from typing import Any

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

COLLECTION_NAME = os.getenv("QDRANT_COLLECTION_NAME", "image_search")
QDRANT_URL = os.getenv("QDRANT_URL", "").strip()
QDRANT_LOCAL_PATH = os.getenv("QDRANT_LOCAL_PATH", "./qdrant_data")

# Must match the output dimension of our CLIP model (ViT-B/32 -> 512).
# If you ever switch CLIP_MODEL_NAME to a variant with a different output
# size, this needs to change too.
VECTOR_SIZE = 512


def get_client() -> QdrantClient:
    """Returns a Qdrant client in local mode or server mode, per .env."""
    if QDRANT_URL:
        return QdrantClient(url=QDRANT_URL)
    Path(QDRANT_LOCAL_PATH).mkdir(parents=True, exist_ok=True)
    return QdrantClient(path=QDRANT_LOCAL_PATH)


def ensure_collection(client: QdrantClient, recreate: bool = False) -> None:
    """
    Creates the collection if it doesn't exist, configured for COSINE
    distance — matching the L2-normalized embeddings from src/embeddings.py.

    recreate=True drops and rebuilds the collection from scratch. Useful
    while iterating on the dataset; we'll turn this off once ingestion is
    incremental rather than a full rebuild each time.
    """
    exists = client.collection_exists(COLLECTION_NAME)
    if exists and recreate:
        client.delete_collection(COLLECTION_NAME)
        exists = False

    if not exists:
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE),
        )


def upsert_images(
    client: QdrantClient,
    ids: list[int],
    vectors,
    payloads: list[dict[str, Any]],
) -> None:
    """
    Store a batch of image vectors with their metadata (payload).
    `vectors` can be a torch.Tensor or a plain list of lists.
    """
    vectors_list = vectors.tolist() if hasattr(vectors, "tolist") else vectors

    points = [
        PointStruct(id=ids[i], vector=vectors_list[i], payload=payloads[i])
        for i in range(len(ids))
    ]
    client.upsert(collection_name=COLLECTION_NAME, points=points)


def search(client: QdrantClient, query_vector, top_k: int = 5):
    """
    Search for the top_k closest stored vectors to `query_vector` (typically
    a text embedding). Returns Qdrant's ScoredPoint objects, each with
    .id, .score, and .payload.
    """
    query_list = query_vector.tolist() if hasattr(query_vector, "tolist") else query_vector
    result = client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_list,
        limit=top_k,
    )
    return result.points