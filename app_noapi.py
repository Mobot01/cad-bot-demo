# app_noapi.py

# ---- 0) Patch sqlite BEFORE importing chromadb (needed on Streamlit Cloud) ----
import sys
try:
    import pysqlite3
    sys.modules["sqlite3"] = pysqlite3
except Exception:
    # If local env already has a new sqlite3, this is fine
    pass

# ---- 1) Imports ----
import os
import zipfile
import shutil

import streamlit as st
import chromadb
from sentence_transformers import SentenceTransformer

# ---- 2) Config ----
DB_DIR     = "cadstandards_index"                 # folder that holds the Chroma DB
ZIP_PATH   = "cadstandards_index.zip"             # prebuilt index shipped with the app
COLL       = "cad_manual"                         # collection name used during ingest
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
TOP_K      = 5

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

# ---- 6) Search box ----
query = st.text_input(
    "Ask me something about the CAD standards:",
    placeholder="e.g., line weights for sections"
)

# ---- 7) Query flow ----
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
        # Show top snippets with page numbers; keep it short for a demo
        for text, meta, dist in zip(docs, metas, dists):
            page = meta.get("page", "?")
            snippet = (text or "").strip().replace("\n", " ")
            if len(snippet) > 300:
                snippet = snippet[:300] + "..."
            st.markdown(f"- **p.{page}** · dist `{dist:.3f}` — {snippet}")
