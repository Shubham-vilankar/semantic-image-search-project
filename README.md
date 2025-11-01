# LLM-Powered Semantic Image Search System

A multimodal image search system: type a natural-language query, get back the
most semantically relevant images — powered by CLIP embeddings and Qdrant
vector search.

This project is being built step by step. Each module below corresponds to
one build session; folders/files will fill in as we progress.


## Local Setup 

### 1. Create and activate a virtual environment

Windows (PowerShell):
```powershell
python -m venv venv
venv\Scripts\Activate.ps1
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

This installs PyTorch, HuggingFace Transformers (for CLIP), Qdrant's client, FastAPI, Streamlit, and supporting libraries. On a laptop CPU this can take a few minutes , PyTorch is the largest download.

### 3. Configure environment variables b mn 

You don't need to edit anything yet — the defaults use Qdrant in **local on-disk mode**, meaning no separate database server is required. To use a real Qdrant server (via Docker) will be introduced in the deployment step.

### 4. To Verify the installation

Run:
```bash
python -c "import torch, transformers, qdrant_client; print('All good:', torch.__version__, transformers.__version__)"
```
---