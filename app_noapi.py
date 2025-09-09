# app_noapi.py

# ---------------------------
# Patch sqlite before importing chromadb
# ---------------------------
import sys
try:
    import pysqlite3
    sys.modules["sqlite3"] = pysqlite3
except Exception:
    # if pysqlite3 isn't available (e.g. local dev), just continue
    pass

# ---------------------------
# Standard library imports
# ---------------------------
import os
import json
import zipfile

# ---------------------------
# Third-party imports
# ---------------------------
import chromadb
import streamlit as st
from sentence_transformers import SentenceTransformer

# ---------------------------
# Config
# ---------------------------
DB_DIR = "cadstandards_index"
ZIP_PATH = "cadstandards_index.zip"
COLL = "cad_manual"
MODEL_NAME = "all-MiniLM-L6-v2"

# ---------------------------
# Auto-unzip the index if needed
# ---------------------------
if not os.path.isdir(DB_DIR) and os.path.exists(ZIP_PATH):
    with zipfile.ZipFile(ZIP_PATH, "r") as z:
        z.extractall(".")

# ---------------------------
# Load sentence transformer
# ---------------------------
embedder = SentenceTransformer(MODEL_NAME)

# ---------------------------
# Open the local Chroma DB
# ---------------------------
client = chromadb.PersistentClient(path=DB_DIR)
collection = client.get_or_create_collection(
    name=COLL,
    metadata={"hnsw:space": "cosine"}  # cosine similarity
)

# ---------------------------
# Streamlit UI
# ---------------------------
st.title("CAD Standards Bot (Demo)")

query = st.text_input("Ask me something about the CAD standards:")

if query:
    # Embed the query
    q_emb = embedder.encode([query]).tolist()
    # Search in Chroma
    results = collection.query(query_embeddings=q_emb, n_results=3)

    if results and results["documents"]:
        st.subheader("Answer (from CAD Standards):")
        for doc, meta in zip(results["documents"][0], results["metadatas"][0]):
            st.markdown(f"- **Page {meta['page']}**: {doc}")
    else:
        st.write("❌ No relevant answer found in the CAD Standards.")
