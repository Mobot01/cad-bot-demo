# app_noapi.py — LOCAL / CLOUD (no external LLM)
# Runs fully offline using sentence-transformers + a prebuilt Chroma index.
# If cadstandards_index/ is missing but cadstandards_index.zip exists,
# it auto-unzips at startup so the cloud doesn’t need to rebuild.

import os
import json
import zipfile
import chromadb
import streamlit as st
from sentence_transformers import SentenceTransformer

# ---------------- Config ----------------
DB_DIR = "cadstandards_index"
COLL   = "cad_manual"
TOP_K  = 10             # how many candidates to consider
THRESH = 0.48           # lower=stricter, higher=looser (0.32–0.55 typical)
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# ------------- Auto-unzip index ----------
# This lets you ship cadstandards_index.zip with the app, and avoid building in the cloud.
if not os.path.isdir(DB_DIR) and os.path.exists("cadstandards_index.zip"):
    with zipfile.ZipFile("cadstandards_index.zip", "r") as z:
        z.extractall(".")  # creates cadstandards_index/ next to this file

# ------------- Streamlit UI --------------
st.set_page_config(page_title="CAD Standards Q&A (Local)", layout="centered")
st.title("🧠 CAD Standards Q&A (Local, Manual-Only)")
st.caption("Short extractive answers with page citations • No external APIs")

# Optional slang → formal mappings to understand casual wording
GLOSS = {}
if os.path.exists("cad_glossary.json"):
    with open("cad_glossary.json", "r", encoding="utf-8") as f:
        GLOSS = json.load(f)

# ------------- Models & DB ---------------
# Local embedder (CPU works; first run may download weights)
embedder = SentenceTransformer(MODEL_NAME)

# Open the local Chroma DB
client = chromadb.PersistentClient(path=DB_DIR)
collection = client.get_or_create_collection(
    name=COLL,
    metadata={"hnsw:space": "cosine"}  # cosine distance
)

# ------------- Retrieval helpers ---------
def expand_query(q: str):
    """Expand casual terms using the glossary (simple rewrite)."""
    qlow = q.lower()
    out = [q]
    for canonical, variants in GLOSS.items():
        for term in [canonical] + variants:
            if term in qlow:
                out.append(qlow.replace(term, canonical))
    # dedupe; keep first 3–4 variants max
    return list(dict.fromkeys(out))[:4]

def retrieve(qs):
    """Semantic retrieval from Chroma with a loose threshold and keyword-ish fallback."""
    seen = {}
    for q in qs:
        q_emb = embedder.encode([q], normalize_embeddings=True)[0].tolist()
        res = collection.query(
            query_embeddings=[q_emb],
            n_results=TOP_K,
            include=["documents", "metadatas", "distances"]
        )
        docs  = res.get("documents", [[]])[0]
        metas = res.get("metadatas", [[]])[0]
        dists = res.get("distances", [[]])[0]
        for t, m, d in zip(docs, metas, dists):
            if d <= THRESH:
                key = (m.get("page"), t[:80])
                if key not in seen or d < seen[key]["distance"]:
                    seen[key] = {"text": t, "page": m.get("page"), "distance": d}

    # fallback: lightweight text query if nothing met THRESH
    if not seen:
        res = collection.query(
            query_texts=[qs[0]],
            n_results=TOP_K,
            include=["documents", "metadatas", "distances"]
        )
        docs  = res.get("documents", [[]])[0]
        metas = res.get("metadatas", [[]])[0]
        dists = res.get("distances", [[]])[0]
        for t, m, d in zip(docs, metas, dists):
            key = (m.get("page"), t[:80])
            seen[key] = {"text": t, "page": m.get("page"), "distance": d}

    # return sorted by distance (lower = better)
    return sorted(seen.values(), key=lambda x: x["distance"])

def make_short_answer(snips):
    """Make a short extractive answer + citations from top snippets."""
    if not snips:
        return None
    best = snips[0]["text"].strip().replace("\n", " ")
    if len(best) > 240:
        # trim at sentence boundary if possible
        cut = best[:240]
        best = cut.rsplit(". ", 1)[0] + "..."
    cites = []
    for d in snips[:3]:
        s = d["text"].strip().replace("\n", " ")
        if len(s) > 160:
            s = s[:160] + "..."
        cites.append(f"- “{s}” (p.{d['page']})")
    return best + "\n\n**Citations:**\n" + "\n".join(cites)

# ------------- App form ------------------
with st.form("ask"):
    q = st.text_input(
        "Ask a question (casual is fine):",
        placeholder="e.g., how do we name sheets?"
    )
    submitted = st.form_submit_button("Ask")

if submitted and q.strip():
    hits = retrieve(expand_query(q.strip()))
    if not hits:
        st.warning("Not found in the CAD Standards manual.")
    else:
        ans = make_short_answer(hits)
        if not ans:
            st.warning("Not found in the CAD Standards manual.")
        else:
            st.subheader("Answer")
            st.write(ans)
            with st.expander("See matching snippets"):
                for d in hits[:5]:
                    st.markdown(f"**p.{d['page']}** · distance {d['distance']:.3f}")
                    st.write(d["text"])
