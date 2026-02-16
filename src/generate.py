from typing import List, Dict

SYSTEM = """You are a helpful assistant answering questions using ONLY the provided email context.
If the answer is not contained in the context, say: "Not enough information in the emails."
Cite evidence by listing the chunk_id(s) you used."""

def build_prompt(question: str, contexts: List[Dict], max_chars: int = 12000) -> str:
    blocks = []
    used = 0
    for r in contexts:
        block = f"[chunk_id={r['chunk_id']} score={r['score']:.3f}]\n{r['text']}\n"
        if used + len(block) > max_chars:
            break
        blocks.append(block)
        used += len(block)

    ctx = "\n---\n".join(blocks)
    return f"{SYSTEM}\n\nCONTEXT:\n{ctx}\n\nQUESTION:\n{question}\n\nANSWER:"

# Pluggable LLM call: implement one of the following.
def call_openai_compatible(prompt: str) -> str:
    """
    Replace with your allowed endpoint / env vars.
    Keep it minimal to satisfy the assignment.
    """
    raise NotImplementedError("Add your OpenAI-compatible call here.")
