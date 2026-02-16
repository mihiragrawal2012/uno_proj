import json
import numpy as np
from pathlib import Path
import faiss
from sentence_transformers import SentenceTransformer

def load_chunks(path: str):
    items = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            items.append(json.loads(line))
    return items

def build_index(chunks_path="artifacts/chunks.jsonl",
                index_path="artifacts/index.faiss",
                meta_path="artifacts/meta.jsonl",
                model_name="sentence-transformers/all-MiniLM-L6-v2"):
    chunks = load_chunks(chunks_path)
    model = SentenceTransformer(model_name)

    texts = [c["text"] for c in chunks]
    emb = model.encode(texts, batch_size=64, show_progress_bar=True, normalize_embeddings=True)
    emb = np.asarray(emb, dtype="float32")

    dim = emb.shape[1]
    index = faiss.IndexFlatIP(dim)  # cosine via normalized + inner product
    index.add(emb)

    Path(index_path).parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, index_path)

    with open(meta_path, "w", encoding="utf-8") as f:
        for i, c in enumerate(chunks):
            f.write(json.dumps({
                "faiss_id": i,
                "chunk_id": c["chunk_id"],
                "meta": c["meta"],
                "text": c["text"]
            }, ensure_ascii=False) + "\n")

if __name__ == "__main__":
    build_index()
