```md
# Mini RAG System (Emails)

Build a Retrieval-Augmented Generation (RAG) pipeline that ingests a dataset of synthetic emails, chunks them, embeds chunks, retrieves relevant context via similarity search, and (optionally) uses an LLM to answer questions grounded in retrieved evidence.

This repo **does not use end-to-end RAG frameworks** like LangChain/LlamaIndex. Core components are implemented directly using:
- **sentence-transformers** (embeddings)
- **FAISS** (vector similarity search)

---

## Dataset

The dataset is expected in the `emails/` directory.

Each email follows a structure like:

- `Subject: ...`
- blank line
- `From: Name <email>`
- `To: Name <email>`
- blank line
- body (100+ words)

The pipeline is robust to minor variations (e.g., missing `Date:`).

---

## Repo Layout

```

mini-rag/
emails/                     # dataset (100 synthetic email txt files)
src/
ingest.py                 # parse + chunk emails -> artifacts/chunks.jsonl
embed_index.py            # embed chunks + build FAISS -> artifacts/index.faiss + artifacts/meta.jsonl
artifacts/
chunks.jsonl              # chunked docs (JSONL)
index.faiss               # FAISS index (cosine similarity)
meta.jsonl                # metadata aligned with FAISS ids (JSONL)
requirements.txt
README.md

````

---

## Setup

### 1) Create a virtual environment (optional but recommended)

```bash
python -m venv .venv
source .venv/bin/activate      # mac/linux
# .venv\Scripts\activate       # windows
````

### 2) Install dependencies

```bash
pip install -r requirements.txt
```

---

## Build the RAG Artifacts

### Step 1 — Ingest + Chunk emails → `artifacts/chunks.jsonl`

```bash
python src/ingest.py
```

Output:

* `artifacts/chunks.jsonl`

Each line is a JSON object:

```json
{
  "chunk_id": "c000012",
  "text": "Subject: ... | From: ... | To: ...\n\n<body chunk>",
  "meta": { "subject": "...", "from": "...", "to": "...", "source_file": "emails/email_012.txt" }
}
```

### Step 2 — Embed chunks + Build FAISS → `artifacts/index.faiss` + `artifacts/meta.jsonl`

```bash
python src/embed_index.py
```

Outputs:

* `artifacts/index.faiss`
* `artifacts/meta.jsonl`

`meta.jsonl` is aligned with FAISS row ids:

```json
{
  "faiss_id": 12,
  "chunk_id": "c000012",
  "meta": {...},
  "text": "..."
}
```

---

## Design Choices & Tradeoffs

### 1) Chunking Strategy (Email-aware)

**Goal:** preserve context in emails where headers and body matter together.

Approach:

* Parse structured headers (`Subject`, `From`, `To`, optional `Date`) into metadata.
* Chunk the body using **paragraph-aware splitting** (blank lines).
* If the body is a single large paragraph, fall back to **sentence-ish splitting**.
* Merge segments into ~**260 words** per chunk with **70 word overlap**.
* **Prepend a compact header string into every chunk**:

  * `Subject | From | To | Date`
  * This improves retrieval for queries like “who requested X?” or “what was the budget approval?”

**Tradeoff:** Larger chunks improve answer completeness but may reduce precision. Overlap reduces boundary loss but increases index size slightly.

### 2) Embedding Model

Model:

* `sentence-transformers/all-MiniLM-L6-v2`

Reasons:

* Fast on CPU
* Solid semantic retrieval baseline
* Easy to swap for higher-quality models later

**Tradeoff:** stronger models (e.g., BGE) may improve retrieval but increase latency and compute cost.

### 3) Retrieval

Vector store:

* **FAISS** `IndexFlatIP` with **normalized embeddings**.

Similarity:

* Cosine similarity implemented as inner product on normalized vectors.

**Tradeoff:** Flat index is simplest and accurate for small datasets (100 emails). For much larger corpora, IVF/HNSW indexes would improve speed.

### 4) Generation (Optional)

This repo builds ingestion + embeddings + retrieval artifacts.
To complete a full RAG loop:

* Embed the query
* Retrieve top-k chunks
* Construct a grounded prompt:

  * “Answer only using the provided context”
  * “Cite chunk_id(s) used”
  * “If missing, respond: Not enough information”

LLM calling code is intentionally left pluggable so you can use any provider (OpenAI-compatible API / local HF model) depending on constraints.

---

## Quality Evaluation Approach

Because the dataset is synthetic and unlabeled, evaluation is based on a lightweight but defensible workflow:

### Retrieval Metrics (Objective)

Create a small set of test queries (e.g., 30–50) where the “correct” email is known.

Measure:

* **Recall@k**: whether any of the top-k retrieved chunks come from the correct email file
* **MRR** (Mean Reciprocal Rank): how high the first correct chunk appears in results

### Generation Checks (Qualitative)

For ~20 queries, manually score:

* **Faithfulness**: answer is supported by retrieved context (no hallucinations)
* **Completeness**: captures key details (who/what/when/why)
* **Citations**: chunk_id evidence included

---

## Notes

* Ensure `emails/` exists and contains the dataset before running scripts.
* Artifacts are written to `artifacts/` and can be committed or regenerated as needed.
* The chunking/parser is tuned for the header format shown in the provided examples.

---

## Quick Commands Summary

```bash
pip install -r requirements.txt
python src/ingest.py
python src/embed_index.py
```

That’s it — you’ll have:

* `artifacts/chunks.jsonl`
* `artifacts/index.faiss`
* `artifacts/meta.jsonl`

```
```
