import argparse
from .retrieve import Retriever
from .generate import build_prompt, call_openai_compatible

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--q", required=True, help="question")
    ap.add_argument("--k", type=int, default=6)
    args = ap.parse_args()

    r = Retriever()
    hits = r.search(args.q, top_k=args.k)

    prompt = build_prompt(args.q, hits)
    answer = call_openai_compatible(prompt)

    print("\nTop hits:")
    for h in hits:
        s = h["meta"].get("subject", "")
        print(f"- {h['chunk_id']} score={h['score']:.3f} subject={s}")

    print("\nAnswer:\n", answer)

if __name__ == "__main__":
    main()
