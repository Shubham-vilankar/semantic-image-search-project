"""
Ingestion pipeline (Step 6): embeds every image in data/images and stores
the vectors + metadata in Qdrant.

Run this whenever your image dataset changes. Searching (src/search.py)
does NOT re-embed images — it only queries whatever was stored here.

Run: python src/ingest.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from dataset import load_dataset
from embeddings import embed_images
from vector_store import ensure_collection, get_client, upsert_images


def main():
    images_dir = Path(__file__).resolve().parent.parent / "data" / "images"

    print(f"Loading and validating images from {images_dir} ...")
    records = list(load_dataset(images_dir))
    if not records:
        print("No valid images found. Add some to data/images/ first.")
        return

    pil_images = [img for img, _ in records]
    metas = [rec for _, rec in records]

    print(f"Embedding {len(pil_images)} images with CLIP...")
    embeddings = embed_images(pil_images)

    client = get_client()
    # recreate=True: full rebuild each run, which is fine while our dataset
    # is tiny and changing. We'll revisit this once ingestion needs to be
    # incremental (large datasets where re-embedding everything is wasteful).
    ensure_collection(client, recreate=True)

    ids = list(range(len(metas)))
    payloads = [
        {
            "image_id": rec.id,
            "filename": rec.path.name,
            "path": str(rec.path),
            "width": rec.width,
            "height": rec.height,
        }
        for rec in metas
    ]

    print("Storing vectors in Qdrant...")
    upsert_images(client, ids, embeddings, payloads)

    print(f"\nDone. {len(ids)} images indexed into collection 'image_search'.")


if __name__ == "__main__":
    main()