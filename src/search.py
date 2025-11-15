"""
Search pipeline (Step 6): takes a natural language query, embeds it with
CLIP's text encoder, and retrieves the closest matching images from Qdrant.

This does NOT touch the image encoder or re-embed any images — it only
embeds the query text (fast, single vector) and searches against whatever
src/ingest.py already stored.

Run: python src/search.py "a striped animal"
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from embeddings import embed_text
from vector_store import get_client
from vector_store import search as qdrant_search


def search_images(query: str, top_k: int = 5) -> list[dict]:
    """
    Returns a list of dicts, best match first:
    [{"score": 0.29, "image_id": "zebra", "filename": "zebra.jpg", ...}, ...]
    """
    # Light prompt templating (per the Step 5 experiment): short/bare
    # keyword queries tend to score lower and rank less confidently than
    # natural-sentence phrasing, since CLIP was trained on photo captions.
    if len(query.split()) <= 2:
        query_for_embedding = f"a photo of {query}"
    else:
        query_for_embedding = query

    query_vector = embed_text([query_for_embedding])[0]  # single 512-d vector

    client = get_client()
    results = qdrant_search(client, query_vector, top_k=top_k)

    return [{"score": r.score, **r.payload} for r in results]


if __name__ == "__main__":
    query = " ".join(sys.argv[1:]) or "a fluffy cat"
    print(f'Query: "{query}"\n')

    results = search_images(query)
    if not results:
        print("No results. Did you run `python src/ingest.py` first?")
    else:
        for i, r in enumerate(results, 1):
            print(f'{i}. {r["image_id"]:20s} score={r["score"]:.3f}  ({r["filename"]})')