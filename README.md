# LLM-Powered Semantic Image Search System

A multimodal image search system: type a natural-language query, get back the
most semantically relevant images — powered by CLIP embeddings and Qdrant
vector search.

This project is being built step by step. Each module below corresponds to
one build session; folders/files will fill in as we progress.

## Project Structure

```
semantic-image-search/
├── data/
│   └── images/          # put your image dataset here
├── src/
│   ├── embeddings.py     # CLIP embedding generation (image + text)
│   ├── vector_store.py   # Qdrant wrapper (store + search vectors)
│   ├── ingest.py         # pipeline: images -> embeddings -> Qdrant
│   └── search.py         # text query -> search logic
├── api/
│   └── main.py            # FastAPI backend
├── app/
│   └── streamlit_app.py   # Streamlit UI
├── requirements.txt
├── .env.example
├── Dockerfile
├── docker-compose.yml
└── README.md
```

## Local Setup (Step 1)

These commands are meant to be run **on your own machine**, in a terminal,
inside the unzipped `semantic-image-search/` folder.

### 1. Create and activate a virtual environment

macOS / Linux:
```bash
python3 -m venv venv
source venv/bin/activate
```

Windows (PowerShell):
```powershell
python -m venv venv
venv\Scripts\Activate.ps1
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

This installs PyTorch, HuggingFace Transformers (for CLIP), Qdrant's client,
FastAPI, Streamlit, and supporting libraries. On a laptop CPU this can take a
few minutes — PyTorch is the largest download.

### 3. Configure environment variables

```bash
cp .env.example .env
```

You don't need to edit anything yet — the defaults use Qdrant in **local
on-disk mode**, meaning no separate database server is required for now. A
real Qdrant server (via Docker) will be introduced in the deployment step.

### 4. Verify the install

Run:
```bash
python -c "import torch, transformers, qdrant_client; print('All good:', torch.__version__, transformers.__version__)"
```

If this prints version numbers with no errors, your environment is ready.

---

## Build Progress

- [x] Step 1: Environment setup & project structure
- [ ] Step 2: Multimodal AI / CLIP architecture overview
- [ ] Step 3: Dataset collection & preprocessing
- [ ] Step 4: Image embedding generation
- [ ] Step 5: Text embedding generation
- [ ] Step 6: Vector database & similarity search
- [ ] Step 7: REST API backend
- [ ] Step 8: Streamlit UI
- [ ] Step 9: LLM-based query rewriting
- [ ] Step 10: Dockerization
- [ ] Step 11: Deployment
