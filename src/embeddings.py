"""
CLIP embedding generation — the core of the whole search system.

Both functions below (`embed_images` and `embed_text`) push data through the
SAME CLIP model, but through its two different encoder towers (see the
architecture diagram from Step 2). Because both towers were trained jointly,
the vectors they produce are directly comparable with cosine similarity —
that's the entire trick this project relies on.

Design choices explained:
- We load the model ONCE at module import time (not per-call). Loading a
  transformer model is slow (~1-3 sec); doing it per-image would make
  ingestion painfully slow.
- We batch images through the model instead of looping one at a time.
  The vision transformer is heavily optimized for batched matrix ops, so
  batching gives a large speedup even on CPU.
- We L2-normalize every output vector. Cosine similarity between two
  normalized vectors is just their dot product — cheaper to compute and
  what Qdrant expects when we configure it for cosine distance in Step 6.
- Everything runs on CPU by default (`device="cpu"`), since this project is
  designed to run for free on a normal laptop. If you have a CUDA GPU
  available, this will auto-detect and use it.
"""

import os
from pathlib import Path

import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

MODEL_NAME = os.getenv("CLIP_MODEL_NAME", "openai/clip-vit-base-patch32")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print(f"[embeddings] Loading {MODEL_NAME} on {DEVICE} ...")
_model = CLIPModel.from_pretrained(MODEL_NAME).to(DEVICE)
_processor = CLIPProcessor.from_pretrained(MODEL_NAME)
_model.eval()  # inference mode: disables dropout etc., we are not training
print("[embeddings] Model loaded.")


@torch.no_grad()  # we are not training, so skip gradient tracking entirely
def embed_images(images: list[Image.Image], batch_size: int = 16) -> torch.Tensor:
    """
    Convert a list of PIL images (already RGB, from src/dataset.py) into a
    tensor of shape (N, 512) — one L2-normalized embedding per image.
    """
    all_embeddings = []

    for i in range(0, len(images), batch_size):
        batch = images[i : i + batch_size]

        inputs = _processor(images=batch, return_tensors="pt").to(DEVICE)
        image_features = _model.get_image_features(**inputs)

        # Newer transformers versions may return a ModelOutput object instead
        # of a raw tensor — unwrap it if so.
        if not torch.is_tensor(image_features):
            image_features = getattr(image_features, "image_embeds", None) \
                or getattr(image_features, "pooler_output", None)

        # L2-normalize each row (each image's vector) to unit length
        image_features = image_features / image_features.norm(p=2, dim=-1, keepdim=True)

        all_embeddings.append(image_features.cpu())

    return torch.cat(all_embeddings, dim=0)


@torch.no_grad()
def embed_text(texts: list[str]) -> torch.Tensor:
    """
    Convert a list of text strings (search queries) into a tensor of shape
    (N, 512) — one L2-normalized embedding per string.

    Used both for the eventual search queries (Step 6) and, right now, as
    a way to sanity-check that the shared embedding space actually works.
    """
    inputs = _processor(text=texts, return_tensors="pt", padding=True).to(DEVICE)
    text_features = _model.get_text_features(**inputs)

    if not torch.is_tensor(text_features):
        text_features = getattr(text_features, "text_embeds", None) \
            or getattr(text_features, "pooler_output", None)

    text_features = text_features / text_features.norm(p=2, dim=-1, keepdim=True)
    return text_features.cpu()


if __name__ == "__main__":
    # Sanity check: embed our 12 sample images AND a few text queries, then
    # print the cosine similarity matrix so we can eyeball whether the
    # shared embedding space is behaving the way Step 2 described.
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from dataset import load_dataset

    images_dir = Path(__file__).resolve().parent.parent / "data" / "images"

    records = list(load_dataset(images_dir))
    pil_images = [img for img, _ in records]
    ids = [rec.id for _, rec in records]

    print(f"\nEmbedding {len(pil_images)} images...")
    image_embeds = embed_images(pil_images)
    print("Image embeddings shape:", tuple(image_embeds.shape))

    queries = [
        "a fluffy cat",
        "a striped animal",
        "a musical instrument",
        "a vehicle",
    ]
    print(f"\nEmbedding {len(queries)} text queries...")
    text_embeds = embed_text(queries)
    print("Text embeddings shape:", tuple(text_embeds.shape))

    # Cosine similarity = dot product, since everything is L2-normalized
    similarity = text_embeds @ image_embeds.T  # shape: (num_queries, num_images)

    print("\nTop match per query (sanity check of the shared embedding space):")
    for qi, query in enumerate(queries):
        scores = similarity[qi]
        best_idx = torch.argmax(scores).item()
        print(f'  "{query}"  ->  {ids[best_idx]}  (score={scores[best_idx]:.3f})')
