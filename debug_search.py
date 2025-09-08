from sentence_transformers import SentenceTransformer
import chromadb, sys

query = " ".join(sys.argv[1:]) or "sheet naming"
print("Query:", query)

embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
c = chromadb.PersistentClient(path="cadstandards_index")
coll = c.get_or_create_collection("cad_manual")

q_emb = embedder.encode([query], normalize_embeddings=True)[0].tolist()
res = coll.query(query_embeddings=[q_emb], n_results=5, include=["documents","metadatas","distances"])

docs  = res.get("documents",[[]])[0]
metas = res.get("metadatas",[[]])[0]
dists = res.get("distances",[[]])[0]

for i,(t,m,d) in enumerate(zip(docs,metas,dists),1):
    print(f"\n{i}) distance={d:.3f}, page={m.get('page')}")
    print(t[:400].replace("\n"," "))
