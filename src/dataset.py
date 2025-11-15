"""
Dataset loading and preprocessing for the image search pipeline.

Why this step exists (before we ever touch CLIP):
- Real image datasets are messy: mixed formats (JPEG/PNG/WEBP), mixed color
  modes (RGB, grayscale, CMYK, RGBA-with-alpha), mixed sizes, and occasional
  corrupt/truncated files.
- CLIP's own processor (used in src/embeddings.py, Step 4) already handles
  RESIZING and NORMALIZATION internally — we do not need to resize images
  ourselves.
- What we DO need to guarantee before handing images to CLIP:
    1. The file actually opens and isn't corrupt/truncated.
    2. The image is in RGB mode (3 channels) — CLIP expects 3-channel input,
       but grayscale ('L'), palette ('P'), and RGBA images are common in the
       wild and will break or silently misbehave if passed in as-is.
- Keeping this as its own module (separate from embeddings.py) means we can
  swap the *source* of images later (e.g. an S3 bucket instead of a local
  folder) without touching any embedding code.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

from PIL import Image, UnidentifiedImageError

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


@dataclass
class ImageRecord:
    """One validated image, ready to be embedded."""
    path: Path
    id: str          # stable identifier derived from filename, used as the
                      # Qdrant point ID later in Step 6
    width: int
    height: int
    original_mode: str  # the mode it was BEFORE we normalized to RGB


def discover_image_paths(images_dir: str | Path) -> list[Path]:
    """
    Scan a directory (non-recursive by default) for supported image files.
    """
    images_dir = Path(images_dir)
    if not images_dir.exists():
        raise FileNotFoundError(f"Images directory not found: {images_dir}")

    paths = sorted(
        p for p in images_dir.iterdir()
        if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS
    )
    return paths


def load_and_validate(path: Path) -> tuple[Image.Image, ImageRecord] | None:
    """
    Open one image file, validate it, and normalize it to RGB.

    Returns None (and prints a warning) instead of raising, so that one bad
    file in a large dataset doesn't crash an entire ingestion run.
    """
    try:
        img = Image.open(path)
        img.load()  # forces a full read now, catching truncated files early
    except (UnidentifiedImageError, OSError) as e:
        print(f"[skip] Could not open {path.name}: {e}")
        return None

    original_mode = img.mode
    width, height = img.size

    if original_mode != "RGB":
        img = img.convert("RGB")

    record = ImageRecord(
        path=path,
        id=path.stem,          # e.g. "persian_cat" from "persian_cat.jpg"
        width=width,
        height=height,
        original_mode=original_mode,
    )
    return img, record


def load_dataset(images_dir: str | Path) -> Iterator[tuple[Image.Image, ImageRecord]]:
    """
    Generator over every valid, RGB-normalized image in a directory.
    Skips (with a warning) any file that fails to load.
    """
    for path in discover_image_paths(images_dir):
        result = load_and_validate(path)
        if result is not None:
            yield result


if __name__ == "__main__":
    # Quick manual test: run `python src/dataset.py` from the project root.
    images_dir = Path(__file__).resolve().parent.parent / "data" / "images"

    print(f"Scanning: {images_dir}\n")
    total = 0
    converted = 0
    for img, record in load_dataset(images_dir):
        total += 1
        note = ""
        if record.original_mode != "RGB":
            converted += 1
            note = f"  <- converted from {record.original_mode} to RGB"
        print(f"{record.id:20s} {record.width}x{record.height:<6}{note}")

    print(f"\nValid images: {total}")
    print(f"Converted to RGB: {converted}")
