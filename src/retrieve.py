import json
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

def load_meta(meta_path: str):
    meta = []
    with open(meta_path, "r", encoding="utf-8") as f:
        for line in f:
            meta.append(json.loads(line))
    return meta

class Retriever:
    def __init__(self,
                 index_path="artifacts/index.faiss",
                 meta_path="artifacts/meta.jsonl",
                 model_name="sentence-transformers/all-MiniLM-L6-v2"):
        self.index = faiss.read_index(index_path)
        self.meta = load_meta(meta_path)
        self.model = SentenceTransformer(model_name)

    def search(self, query: str, top_k: int = 6):
        q = self.model.encode([query], normalize_embeddings=True)
        q = np.asarray(q, dtype="float32")
        scores, ids = self.index.search(q, top_k)

        results = []
        for score, idx in zip(scores[0], ids[0]):
            if idx == -1:
                continue
            rec = self.meta[idx]
            results.append({
                "score": float(score),
                "chunk_id": rec["chunk_id"],
                "meta": rec["meta"],
                "text": rec["text"]
            })
        return results
