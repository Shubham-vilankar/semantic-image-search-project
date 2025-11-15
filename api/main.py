"""
REST API backend (Step 7).

Exposes the search pipeline built in Step 6 over HTTP, so any client
(the Streamlit UI in Step 8, curl, a future mobile app, etc.) can query it
without needing Python or CLIP installed locally themselves.

Run from the project root:
    uvicorn api.main:app --reload --port 8000

Then visit http://localhost:8000/docs for interactive API docs (FastAPI
auto-generates this — genuinely useful for testing without writing a client).
"""

import sys
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel

# Make src/ importable from api/
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from search import search_images  # noqa: E402  (import after sys.path edit, intentional)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
IMAGES_DIR = PROJECT_ROOT / "data" / "images"

app = FastAPI(
    title="Semantic Image Search API",
    description="Text-to-image search powered by CLIP embeddings + Qdrant.",
    version="1.0.0",
)

# Allows the Streamlit app (running on a different port) to call this API
# from the browser without being blocked by CORS. Fine to leave wide open
# for local development; would tighten this for a real public deployment.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class SearchResult(BaseModel):
    score: float
    image_id: str
    filename: str
    width: int
    height: int


class SearchResponse(BaseModel):
    query: str
    results: list[SearchResult]


@app.get("/", tags=["meta"])
def root():
    return {"status": "ok", "message": "Semantic Image Search API is running."}


@app.get("/search", response_model=SearchResponse, tags=["search"])
def search(
    q: str = Query(..., min_length=1, description="Natural language search query"),
    top_k: int = Query(5, ge=1, le=50, description="Number of results to return"),
):
    """
    Search the indexed image collection with a natural language query.
    Returns metadata only — use GET /images/{filename} to fetch the actual
    image bytes for each result.
    """
    try:
        results = search_images(q, top_k=top_k)
    except Exception as e:
        # Most likely cause: ingest.py hasn't been run yet, so the Qdrant
        # collection doesn't exist. Surface that clearly instead of a raw
        # 500 with a confusing stack trace.
        raise HTTPException(
            status_code=500,
            detail=f"Search failed: {e}. Have you run `python src/ingest.py` yet?",
        )

    return SearchResponse(query=q, results=results)


@app.get("/images/{filename}", tags=["images"])
def get_image(filename: str):
    """
    Serves an image file by filename from data/images/.
    Used by the frontend to actually display search result thumbnails.
    """
    # Basic path traversal guard: reject anything containing path separators.
    if "/" in filename or "\\" in filename or ".." in filename:
        raise HTTPException(status_code=400, detail="Invalid filename.")

    image_path = IMAGES_DIR / filename
    if not image_path.exists():
        raise HTTPException(status_code=404, detail=f"Image not found: {filename}")

    return FileResponse(image_path)