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

DB_DIR = "cadstandards_index"
ZIP_PATH = "cadstandards_index.zip"
COLL = "cad_manual"
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"   # exact repo name is safest
TOP_K = 5

# --- unzip index if folder missing ---
def ensure_index_unzipped():
    if not os.path.isdir(DB_DIR) and os.path.exists(ZIP_PATH):
        with zipfile.ZipFile(ZIP_PATH, "r") as z:
            z.extractall(".")
        return True
    return False

unzipped_now = ensure_index_unzipped()

embedder = SentenceTransformer(MODEL_NAME)

def open_collection():
    client = chromadb.PersistentClient(path=DB_DIR)
    return client.get_or_create_collection(
        name=COLL,
        metadata={"hnsw:space": "cosine"}
    )

collection = open_collection()

# --- Streamlit UI ---
st.set_page_config(page_title="CAD Standards Bot (Demo)", layout="centered")
st.title("CAD Standards Bot (Demo)")

# Health check / helpful debug for demo
with st.expander("Data status", expanded=False):
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
                ensure_index_unzipped()
                collection = open_collection()
                st.experimental_rerun()
        else:
            st.error("No cadstandards_index folder and no cadstandards_index.zip found.")

query = st.text_input("Ask me something about the CAD standards:", placeholder="e.g., line weights for sections")

if query:
    # IMPORTANT: normalize embeddings to match how ingest.py built them
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
        # show top snippets with page citations
        for t, m, d in zip(docs, metas, dists):
            page = m.get("page", "?")
            snippet = (t or "").strip().replace("\n", " ")
            if len(snippet) > 300:
                snippet = snippet[:300] + "..."
            st.markdown(f"- **p.{page}** · dist `{d:.3f}` — {snippet}")
