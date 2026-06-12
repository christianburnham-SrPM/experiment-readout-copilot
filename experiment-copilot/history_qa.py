"""Ask Your History — RAG Q&A over the experiment-readout library.

The textbook pattern: retrieve (BM25 top-k) -> ground -> generate with citations ->
refuse when retrieval comes back empty or weak. Same contract as the memo layer:
the model may only use what was retrieved.
"""

import os

from anthropic import Anthropic

MODEL = os.environ.get("COPILOT_MODEL", "claude-sonnet-4-6")

SYSTEM = """You answer questions about an organization's experiment history using ONLY the
retrieved readouts provided. Rules:
- Cite every claim with the experiment ID in brackets, e.g. [EXP-014].
- If the retrieved readouts don't answer the question, say so plainly and name the
  closest related experiments instead. Never invent experiments, numbers, or outcomes.
- Be concise: direct answer first, then the relevant learnings."""


def ask_history(question: str, memory, api_key: str | None = None, k: int = 4):
    hits = memory.similar(question, k=k)
    if not hits:
        return "Nothing in the experiment library matches that question.", []
    context = "\n".join(
        f"[{h['id']}] {h['title']} ({h['area']}) — Hypothesis: {h['hypothesis']} "
        f"Result: {h['result']} Decision: {h['decision']} Learning: {h['learning']}"
        for h in hits
    )
    client = Anthropic(api_key=api_key) if api_key else Anthropic()
    msg = client.messages.create(
        model=MODEL, max_tokens=500, system=SYSTEM,
        messages=[{"role": "user", "content": f"Retrieved readouts:\n{context}\n\nQuestion: {question}"}],
    )
    return "".join(b.text for b in msg.content if getattr(b, "type", "") == "text"), hits
