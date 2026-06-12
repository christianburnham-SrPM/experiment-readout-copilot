"""BM25 retrieval over the prior-experiment readout library (institutional memory).

DECISION: lexical retrieval first. Readouts are short and term-dense ("SRM",
"checkout", "novelty"); BM25 is cheap, explainable, and good enough at this corpus
size. Embeddings + rerank is the measured v2 — the eval harness exists so that
upgrade is a measurement, not an assumption.
"""

import json
import re
from pathlib import Path

from rank_bm25 import BM25Okapi

_TOKEN = re.compile(r"[a-z0-9$%.+-]+")


def tokenize(text: str):
    return _TOKEN.findall(text.lower())


class Memory:
    def __init__(self, path: Path = Path("readouts.jsonl")):
        self.docs = [json.loads(l) for l in Path(path).read_text(encoding="utf-8").splitlines() if l.strip()]
        fields = ["title", "area", "hypothesis", "result", "decision", "learning"]
        self._bm25 = BM25Okapi([tokenize(" ".join(d[f] for f in fields)) for d in self.docs])

    @property
    def size(self) -> int:
        return len(self.docs)

    def similar(self, query: str, k: int = 3):
        scores = self._bm25.get_scores(tokenize(query))
        ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
        return [{**self.docs[i], "score": float(scores[i])} for i in ranked if scores[i] > 0]
