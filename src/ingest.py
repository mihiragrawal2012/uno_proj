import os, re, json
from pathlib import Path
from typing import List
# from .models import Chunk

HEADER_RE = re.compile(r"^(Subject|From|To|Date):\s*(.*)$", re.IGNORECASE)

def parse_email(text: str) -> tuple[dict[str, str], str]:
    lines = text.splitlines()
    meta = {}
    body_lines = []
    in_headers = True

    for line in lines:
        if in_headers:
            if line.strip() == "":
                in_headers = False
                continue
            m = HEADER_RE.match(line.strip())
            if m:
                meta[m.group(1).lower()] = m.group(2).strip()
            else:
                # If format deviates, treat as body from here
                in_headers = False
                body_lines.append(line)
        else:
            body_lines.append(line)

    body = "\n".join(body_lines).strip()
    return meta, body

def chunk_text(meta: dict[str, str], body: str,
                chunk_words: int = 260, overlap_words: int = 70) -> List[str]:
    # 1) paragraph split
    paras = [p.strip() for p in re.split(r"\n\s*\n", body) if p.strip()]

    # If synthetic emails are mostly one big paragraph, sentence-split fallback
    if len(paras) <= 1:
        paras = re.split(r"(?<=[.!?])\s+", body.strip())
        paras = [p.strip() for p in paras if p.strip()]

    # 2) merge into word-bounded chunks
    chunks, current, cur_w = [], [], 0

    def flush():
        nonlocal current, cur_w
        if current:
            chunks.append(" ".join(current).strip())
        current, cur_w = [], 0

    for p in paras:
        w = len(p.split())
        if cur_w + w <= chunk_words:
            current.append(p)
            cur_w += w
        else:
            flush()
            current.append(p)
            cur_w = w
    flush()

    # 3) overlap
    if overlap_words > 0 and len(chunks) > 1:
        out = [chunks[0]]
        prev_tail = chunks[0].split()[-overlap_words:]
        for c in chunks[1:]:
            out.append((" ".join(prev_tail) + "\n" + c).strip())
            prev_tail = c.split()[-overlap_words:]
        chunks = out

    # 4) header prefix inside chunk text (improves retrieval a lot)
    header_bits = []
    if "subject" in meta: header_bits.append(f"Subject: {meta['subject']}")
    if "from" in meta: header_bits.append(f"From: {meta['from']}")
    if "to" in meta: header_bits.append(f"To: {meta['to']}")
    header = " | ".join(header_bits)

    if header:
        chunks = [header + "\n\n" + c for c in chunks]

    return chunks


def ingest_emails(emails_dir: str, out_path: str):
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    chunk_id = 0
    with out.open("w", encoding="utf-8") as f:
        for p in sorted(Path(emails_dir).glob("**/*")):
            if p.is_file():
                text = p.read_text(encoding="utf-8", errors="ignore")
                meta, body = parse_email(text)
                meta["source_file"] = str(p)
                chunks = chunk_text(meta, body)

                for c in chunks:
                    rec = {
                        "chunk_id": f"c{chunk_id:06d}",
                        "text": c,
                        "meta": meta
                    }
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    chunk_id += 1

if __name__ == "__main__":
    ingest_emails("emails", "artifacts/chunks.jsonl")
