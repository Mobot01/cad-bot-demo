# app_noapi.py  —  No-API “concise answer + citations” demo

# ---- 0) Patch sqlite BEFORE importing chromadb (needed on Streamlit Cloud) ----
import sys
try:
    import pysqlite3
    sys.modules["sqlite3"] = pysqlite3
except Exception:
    # Local env may already have a recent sqlite3
    pass

# ---- 1) Imports ----
import os
import re
import zipfile
import shutil
from collections import Counter

import streamlit as st
import chromadb
from sentence_transformers import SentenceTransformer

# ---- 2) Config ----
DB_DIR     = "cadstandards_index"                 # folder that holds the Chroma DB
ZIP_PATH   = "cadstandards_index.zip"             # prebuilt index shipped with the app
COLL       = "cad_manual"                         # collection name used during ingest
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
TOP_K      = 5                                    # how many chunks to retrieve

# ---- 3) Unzip helper (robust to ZIP layout) ----
def ensure_index_unzipped() -> bool:
    """
    Ensure we end up with:
        ./cadstandards_index/
            chroma.sqlite3
            <uuid>/
                *.bin, *.pickle ...
    Works whether the zip contains the folder or just the files.
    Returns True if we extracted anything this call, else False.
    """
    if os.path.isdir(DB_DIR):
        return False
    if not os.path.exists(ZIP_PATH):
        return False

    with zipfile.ZipFile(ZIP_PATH, "r") as z:
        names = z.namelist()
        # Case A: zip already has 'cadstandards_index/...'
        if all(n.startswith(DB_DIR + "/") or n.endswith("/") for n in names):
            z.extractall(".")
        else:
            # Case B: zip has index files at top-level; place them under DB_DIR
            os.makedirs(DB_DIR, exist_ok=True)
            z.extractall(DB_DIR)
    return True

# Try to ensure the index is present on startup
_ = ensure_index_unzipped()

# ---- 4) Model & collection ----
embedder = SentenceTransformer(MODEL_NAME)

def open_collection():
    client = chromadb.PersistentClient(path=DB_DIR)
    return client.get_or_create_collection(
        name=COLL,
        metadata={"hnsw:space": "cosine"},  # cosine similarity
    )

collection = open_collection()

# ---- 5) UI ----
st.set_page_config(page_title="CAD Standards Bot (Demo)", layout="centered")
st.title("CAD Standards Bot (Demo)")

# Debug/health panel
with st.expander("Data status", expanded=True):
    # Show current doc count (defensive: always produce a value)
    try:
        cnt = int(collection.count())
    except Exception as e:
        st.error(f"Could not read collection: {e}")
        cnt = 0

    st.write(f"**Documents in index:** {cnt}")

    if cnt == 0:
        if os.path.exists(ZIP_PATH):
            st.warning("Index appears empty. Try re-extracting the ZIP and reloading.")
            if st.button("Re-extract and Reload"):
                # Remove any half-baked folder first
                if os.path.isdir(DB_DIR):
                    shutil.rmtree(DB_DIR, ignore_errors=True)

                ensure_index_unzipped()

                # Reopen collection, then rerun app to refresh state
                collection = open_collection()  # local shadow is fine before rerun
                if hasattr(st, "rerun"):
                    st.rerun()
                else:
                    st.experimental_rerun()
        else:
            st.error("No cadstandards_index folder and no cadstandards_index.zip found.")

# ---- 6) Helpers: extractive heuristic ----
def _sentences(text: str):
    """Simple sentence splitter suitable for PDF-ish text."""
    sents = re.split(r'(?<=[\.\?\!])\s+', text.replace('\n', ' ').strip())
    # keep reasonable lengths to avoid junk
    return [s.strip() for s in sents if 6 <= len(s) <= 300]

_STOP = {"the","a","an","and","or","of","to","in","on","for","by","is","are","as","at","with","from","that","this","these","those","it","its"}

def _keywords(q: str):
    toks = re.findall(r"[a-z0-9\-]+", q.lower())
    return [t for t in toks if t not in _STOP and len(t) > 2]

def _score_sentence(sent: str, keys):
    if not sent:
        return 0.0
    toks = re.findall(r"[a-z0-9\-]+", sent.lower())
    bag = Counter(toks)
    # score by keyword frequency with a small length normalization
    return sum(bag[k] for k in keys) / (1 + len(toks)/40)

def _truncate(s: str, n: int = 300) -> str:
    s = s.strip().replace("\n", " ")
    return s if len(s) <= n else s[:n] + "..."

# ---- 7) Search box ----
query = st.text_input(
    "Ask me something about the CAD standards:",
    placeholder="e.g., line weights for sections"
)

# ---- 8) Retrieval + extractive concise answer ----
if query:
    # Always get a (1, dim) numpy array, then take row 0 to ensure a dense vector
    q_mat = embedder.encode(
        [query],
        normalize_embeddings=True,
        convert_to_numpy=True,
    )
    q_emb = q_mat[0].astype(float).tolist()  # 384-length vector for MiniLM

    try:
        res = collection.query(
            query_embeddings=[q_emb],
            n_results=TOP_K,
            include=["documents", "metadatas", "distances"],
        )
    except Exception as e:
        st.error(f"Query failed: {e}")
        st.stop()

    docs  = (res.get("documents") or [[]])[0]
    metas = (res.get("metadatas") or [[]])[0]
    dists = (res.get("distances") or [[]])[0]

    st.subheader("Answer (from CAD Standards):")

    if not docs:
        st.write("❌ No relevant answer found in the CAD Standards.")
    else:
        # ---- Extractive concise answer + citations ----
        keys = _keywords(query)
        candidates = []  # (score, sentence, page)
        for text, meta, dist in zip(docs, metas, dists):
            page = meta.get("page", "?")
            for s in _sentences(text or ""):
                sc = _score_sentence(s, keys)
                if sc > 0:
                    candidates.append((sc, s, page))

        candidates.sort(reverse=True, key=lambda x: x[0])
        top_sents = [s for _, s, _ in candidates[:3]]
        cited_pages = []
        for _, _, p in candidates[:3]:
            if p != "?":
                try:
                    cited_pages.append(int(p))
                except Exception:
                    pass
        cited_pages = sorted(set(cited_pages))

        if top_sents:
            st.markdown("**Concise answer:** " + " ".join(top_sents))
            if cited_pages:
                st.caption("**Citations:** CAD Standards manual — " +
                           ", ".join(f"p.{p}" for p in cited_pages))
        else:
            st.write("I couldn’t extract a concise answer. See supporting snippets below.")

        # ---- Supporting snippets (what you had before) ----
        st.markdown("---")
        st.markdown("**Supporting snippets:**")
        for text, meta, dist in zip(docs, metas, dists):
            page = meta.get("page", "?")
            st.markdown(f"- **p.{page}** · dist `{dist:.3f}` — {_truncate(text or '')}")
