"""
Exploratory script (Step 5): compares keyword-style vs. natural-sentence
queries against our sample images, to build intuition for how to phrase
search queries later.

CLIP was trained on real photo captions scraped from the web (e.g. "a photo
of a fluffy orange cat sitting on a windowsill"), NOT on bare keywords. So
query phrasing that resembles a natural caption tends to match its training
distribution better than a single bare noun.
"""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from dataset import load_dataset
from embeddings import embed_images, embed_text

images_dir = Path(__file__).resolve().parent.parent / "data" / "images"
records = list(load_dataset(images_dir))
pil_images = [img for img, _ in records]
ids = [rec.id for _, rec in records]

print(f"Embedding {len(pil_images)} images...\n")
image_embeds = embed_images(pil_images)

# Paired queries: same intent, two different phrasings
query_pairs = [
    ("cat", "a photo of a fluffy cat"),
    ("guitar", "a person playing an acoustic guitar"),
    ("stripes", "a wild animal with black and white stripes"),
    ("clock", "a vintage analog clock on a wall"),
]

def top_k(query_embed, k=3):
    scores = (query_embed @ image_embeds.T).squeeze(0)
    top = torch.topk(scores, k)
    return [(ids[i], top.values[j].item()) for j, i in enumerate(top.indices)]

for keyword, sentence in query_pairs:
    kw_embed = embed_text([keyword])
    sent_embed = embed_text([sentence])

    print(f'Keyword:  "{keyword}"')
    for name, score in top_k(kw_embed):
        print(f"    {name:20s} {score:.3f}")

    print(f'Sentence: "{sentence}"')
    for name, score in top_k(sent_embed):
        print(f"    {name:20s} {score:.3f}")

    print()