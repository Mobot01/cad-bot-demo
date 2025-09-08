import os, json
import chromadb
from openai import OpenAI
import streamlit as st

DB_DIR = "cadstandards_index"
COLL   = "cad_manual"
TOP_K  = 7
THRESH = 0.35  # 0.25 (strict) .. 0.45 (looser)
ANSWER_MODEL = "gpt-4o-mini"
EMB_MODEL    = "text-embedding-3-small"

client = OpenAI()

st.set_page_config(page_title="CAD Standards Q&A", layout="centered")
st.title("🧠 CAD Standards Q&A (Manual-Only)")
st.caption("Short answers with page citations • Answers only from your PDF")

# Load slang → formal mappings
GLOSS = {}
if os.path.exists("cad_glossary.json"):
    with open("cad_glossary.json", "r", encoding="utf-8") as f:
        GLOSS = json.load(f)

db = chromadb.PersistentClient(path=DB_DIR)
collection = db.get_or_create_collection(COLL, metadata={"hnsw:space": "cosine"})

def embed(texts):
    resp = client.embeddings.create(model=EMB_MODEL, input=texts)
    return [d.embedding for d in resp.data]

def expand_query(q):
    qlow = q.lower()
    expanded = [q]
    for canonical, variants in GLOSS.items():
        for term in [canonical] + variants:
            if term in qlow:
                expanded.append(qlow.replace(term, canonical))
    return list(dict.fromkeys(expanded))[:4]

def retrieve(query_list):
    seen = {}
    for q in query_list:
        q_emb = embed([q])[0]
        res = collection.query(
            query_embeddings=[q_emb],
            n_results=TOP_K,
            include=["documents","metadatas","distances"]
        )
        docs = res["documents"][0] if res["documents"] else []
        metas = res["metadatas"][0] if res["metadatas"] else []
        dists = res["distances"][0] if res["distances"] else []
        for t, m, d in zip(docs, metas, dists):
            if d <= THRESH:
                key = (m.get("page"), t[:80])
                if key not in seen or d < seen[key]["distance"]:
                    seen[key] = {"text": t, "page": m.get("page"), "distance": d}
    return sorted(seen.values(), key=lambda x: x["distance"])

SYSTEM_PROMPT = """You answer ONLY using the provided snippets from the CAD Standards manual.
Rules:
- If the answer is not clearly in the snippets, reply exactly: "Not found in the CAD Standards manual."
- Keep answers SHORT (1–3 sentences).
- After the answer, output a line 'Citations:' followed by bullet points with page numbers, quoting tiny snippets.
"""

def build_context(rdocs):
    return "\n\n---\n\n".join([f"(p.{d['page']}) {d['text']}" for d in rdocs])

def ask_llm(question, rdocs):
    context = build_context(rdocs)
    messages = [
        {"role":"system","content": SYSTEM_PROMPT},
        {"role":"user","content": f"Question: {question}\n\nSnippets:\n{context}"}
    ]
    resp = client.chat.completions.create(
        model=ANSWER_MODEL,
        temperature=0,
        messages=messages
    )
    return resp.choices[0].message.content.strip()

with st.form("ask"):
    q = st.text_input("Ask a question (casual is fine):", placeholder="e.g., how do we name sheets?")
    submitted = st.form_submit_button("Ask")

if submitted and q.strip():
    qlist = expand_query(q.strip())
    hits = retrieve(qlist)
    if not hits:
        st.warning("Not found in the CAD Standards manual.")
    else:
        answer = ask_llm(q, hits[:6])
        if "Not found in the CAD Standards manual." in answer:
            st.warning("Not found in the CAD Standards manual.")
        else:
            st.subheader("Answer")
            st.write(answer)
            with st.expander("See matching snippets"):
                for d in hits[:4]:
                    st.markdown(f"**p.{d['page']}** · distance {d['distance']:.3f}")
                    st.write(d["text"])
