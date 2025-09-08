# ingest.py  — LOCAL (no API), robust extractor + UNIQUE IDS
# Run:  python ingest.py

import os
import hashlib
from typing import List, Dict

import chromadb
from sentence_transformers import SentenceTransformer

# ---------- Config ----------
PDF_PATH = "CAD_Standards.pdf"   # change if your file has a different name
DB_DIR   = "cadstandards_index"
COLL     = "cad_manual"

CHUNK_MAX_CHARS = 1800           # larger chunk = fewer embeddings
CHUNK_OVERLAP   = 120
EMBED_MODEL     = "sentence-transformers/all-MiniLM-L6-v2"
UPSERT_BATCH    = 1000           # insert in batches to avoid big payloads


# ---------- PDF reading (PyMuPDF first, then pypdf fallback) ----------
def read_pdf(path: str) -> List[Dict]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Put your PDF at {path}")

    # 1) Try PyMuPDF (best for extraction)
    try:
        import fitz  # PyMuPDF
        doc = fitz.open(path)
        pages = []
        for i, page in enumerate(doc, start=1):
            text = page.get_text("text") or ""
            text = " ".join(text.split())
            pages.append({"page": i, "text": text})
        doc.close()
        # If at least one page has text, use it
        if any(p["text"] for p in pages):
            return pages
        else:
            print("PyMuPDF returned little/no text; falling back to pypdf...")
    except Exception as e:
        print("PyMuPDF unavailable or failed, falling back to pypdf…", e)

    # 2) Fallback to pypdf
    from pypdf import PdfReader
    reader = PdfReader(path)
    pages = []
    for i, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        text = " ".join(text.split())
        pages.append({"page": i, "text": text})
    return pages


# ---------- Chunking ----------
def chunk_text(text: str, max_chars=CHUNK_MAX_CHARS, overlap=CHUNK_OVERLAP) -> List[str]:
    chunks, i = [], 0
    n = len(text)
    while i < n:
        part = text[i:i + max_chars]
        # try to end at a sentence boundary to keep chunks readable
        last_dot = part.rfind(". ")
        if last_dot > 400:
            part = part[: last_dot + 1]
        part = part.strip()
        if part:
            chunks.append(part)
        i += max(1, len(part) - overlap)
    return chunks


def build_chunks(pages: List[Dict]) -> List[Dict]:
    out = []
    for p in pages:
        if not p["text"]:
            continue
        for c in chunk_text(p["text"]):
            out.append({"page": p["page"], "content": c})
    return out


# ---------- Main ----------
def main():
    pages  = read_pdf(PDF_PATH)
    chunks = build_chunks(pages)
    print(f"Pages: {len(pages)}, chunks: {len(chunks)}")

    # Local embedder (no API)
    model = SentenceTransformer(EMBED_MODEL)

    # Prepare records with TRULY UNIQUE IDS
    ids, docs, metas = [], [], []
    seen = set()
    for i, ch in enumerate(chunks):
        text = ch["content"]
        # base hash uses page + content; short for readability
        base = hashlib.sha1((str(ch["page"]) + "|" + text).encode("utf-8")).hexdigest()[:12]
        cid  = f"{ch['page']}-{i}-{base}"  # page + running index + short hash
        if cid in seen:
            # extremely unlikely, but guard just in case
            k = 1
            while f"{cid}-{k}" in seen:
                k += 1
            cid = f"{cid}-{k}"
        seen.add(cid)
        ids.append(cid)
        docs.append(text)
        metas.append({"page": ch["page"]})

    # Compute embeddings in batches (shows progress bar)
    embs = model.encode(docs, batch_size=64, show_progress_bar=True, normalize_embeddings=True)

    # Create a fresh collection each time (simplest and safe)
    client = chromadb.PersistentClient(path=DB_DIR)
    try:
        client.delete_collection(COLL)
    except Exception:
        pass
    coll = client.create_collection(COLL, metadata={"hnsw:space": "cosine"})

    # Upsert in manageable batches
    total = len(ids)
    for start in range(0, total, UPSERT_BATCH):
        end = min(start + UPSERT_BATCH, total)
        coll.upsert(
            ids=ids[start:end],
            documents=docs[start:end],
            metadatas=metas[start:end],
            embeddings=embs[start:end].tolist(),
        )
        print(f"Upserted {end}/{total}")

    print("Index built at:", DB_DIR)


if __name__ == "__main__":
    main()
