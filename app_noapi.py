# app_noapi.py

# ---- patch sqlite before importing chromadb ----
import sys
try:
    import pysqlite3
    sys.modules["sqlite3"] = pysqlite3
except Exception:
    pass

import os, json, zipfile
import chromadb
import streamlit as st
from sentence_transformers import SentenceTransformer

DB_DIR   = "cadstandards_index"
ZIP_PATH = "cadstandards_index.zip"
COLL     = "cad_manual"
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"  # safest canonical name
TOP_K    = 5

# --- unzip index robustly ---
def ensure_index_unzipped():
    """
    Ensure we end up with:
      ./cadstandards_index/
        chroma.sqlite3
        <random-id>/
          *.bin / *.pickle
    Works whether the zip contains the folder or just the files.
    """
    if os.path.isdir(DB_DIR):
        return False

    if not os.path.exists(ZIP_PATH):
        return False

    with zipfile.ZipFile(ZIP_PATH, "r") as z:
        names = z.namelist()
        if all(n.startswith(DB_DIR + "/") or n.endswith("/") for n in names):
            # ZIP already contains 'cadstandards_index/...'
            z.extractall(".")
        else:
            # ZIP has files at top-level; place them inside 'cadstandards_index/'
            os.makedirs(DB_DIR, exist_ok=True)
            z.extractall(DB_DIR)
    return True

unzipped_now = ensure_index_unzipped()

# --- model & collection ---
embedder = SentenceTransformer(MODEL_NAME)

def open_collection():
    client = chromadb.PersistentClient(path=DB_DIR)
    return client.get_or_create_collection(
        name=COLL,
        metadata={"hnsw:space": "cosine"}
    )

collection = open_collection()

# --- UI ---
st.set_page_config(page_title="CAD Standards Bot (Demo)", layout="centered")
st.title("CAD Standards Bot (Demo)")

# Health/debug panel
with st.expander("Data status", expanded=True):
    try:
        cnt = collection.count()
    except Exception as e:
        cnt = 0
        st.error(f"Could not read collection: {e}")
    st.write(f"**Documents in index:** {cnt}")
    if cnt == 0:
        if os.path.exists(ZIP_PATH):
            st.warning("Index appears empty. Trying to re-extract ZIP and reload…")
  if st.button("Re-extract and Reload"):
    import shutil
    # in case a bad/empty folder exists, remove and re-extract cleanly
    if os.path.isdir(DB_DIR):
        shutil.rmtree(DB_DIR, ignore_errors=True)

    ensure_index_unzipped()
    collection = open_collection()

    # Streamlit 1.25+ uses st.rerun(); older builds still have experimental_rerun
    if hasattr(st, "rerun"):
        st.rerun()
    else:
        st.experimental_rerun()

        else:
            st.error("No cadstandards_index folder and no cadstandards_index.zip found.")

query = st.text_input("Ask me something about the CAD standards:", placeholder="e.g., line weights for sections")

if query:
    # Normalize embeddings to match ingest.py
    q_emb = embedder.encode([query], normalize_embeddings=True)[0].tolist()
    try:
        res = collection.query(
            query_embeddings=[q_emb],
            n_results=TOP_K,
            include=["documents", "metadatas", "distances"]
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
        for t, m, d in zip(docs, metas, dists):
            page = m.get("page", "?")
            snippet = (t or "").strip().replace("\n", " ")
            if len(snippet) > 300:
                snippet = snippet[:300] + "..."
            st.markdown(f"- **p.{page}** · dist `{d:.3f}` — {snippet}")
