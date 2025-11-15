"""
Streamlit UI (Step 8).

This is a pure HTTP client of the FastAPI backend from Step 7 — it does NOT
import CLIP, Qdrant, or any src/ code directly. That separation matters:
this file could be swapped for a React app, a mobile app, or a CLI tool
without touching any search logic at all.

Requires api/main.py to already be running (uvicorn api.main:app --port 9000)
before you start this.

Run from the project root:
    streamlit run app/streamlit_app.py
"""

import os

import requests
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

API_BASE_URL = os.getenv("API_BASE_URL", "http://127.0.0.1:9000")

st.set_page_config(page_title="Semantic Image Search", page_icon="🔍", layout="wide")

st.title("🔍 Semantic Image Search")
st.caption("Search your image collection using natural language — powered by CLIP + Qdrant.")

with st.sidebar:
    st.header("Search settings")
    top_k = st.slider("Number of results", min_value=1, max_value=20, value=5)
    st.markdown("---")
    st.markdown(f"**API:** `{API_BASE_URL}`")
    st.caption("Change this in .env (API_BASE_URL) if your backend runs elsewhere.")


def run_search(q: str, k: int) -> dict:
    resp = requests.get(f"{API_BASE_URL}/search", params={"q": q, "top_k": k}, timeout=30)
    resp.raise_for_status()
    return resp.json()


query = st.text_input("What are you looking for?", placeholder="e.g. a striped animal")
search_clicked = st.button("Search", type="primary")

if search_clicked and not query.strip():
    st.warning("Please enter a search query.")

elif search_clicked and query.strip():
    with st.spinner("Searching..."):
        try:
            data = run_search(query, top_k)
        except requests.exceptions.ConnectionError:
            st.error(
                f"Could not reach the API at {API_BASE_URL}. "
                f"Is `uvicorn api.main:app --port 9000` running in another terminal?"
            )
            st.stop()
        except requests.exceptions.HTTPError as e:
            st.error(f"API error: {e}")
            st.stop()

    results = data["results"]
    if not results:
        st.warning("No results found. Have you run `python src/ingest.py` yet?")
    else:
        st.subheader(f'Results for "{data["query"]}"')
        cols = st.columns(4)
        for i, r in enumerate(results):
            col = cols[i % 4]
            with col:
                image_url = f"{API_BASE_URL}/images/{r['filename']}"
                st.image(image_url, use_container_width=True)
                st.caption(f"**{r['image_id']}**  \nscore: {r['score']:.3f}")