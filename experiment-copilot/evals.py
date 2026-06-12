"""Evaluation suite, importable by the in-app Evals tab and runnable standalone.

Two layers, honestly labeled:
1. DECISION REGRESSION SUITE (deterministic, no API): 20 scenarios with locked
   expected decisions. Guarantees changes to the stats engine can't silently flip
   a SHIP into a KEEP RUNNING. This measures code correctness, not model quality.
2. MEMO FAITHFULNESS (LLM-judged, needs API key): generates the memo for a sample
   of scenarios and judges whether every number matches the stats block verbatim
   and the decision is communicated, not overruled. This measures the AI layer.
"""

import json
from pathlib import Path

from stats import analyze

GOLD = Path(__file__).parent / "gold_set.jsonl"

JUDGE = """Grade this experiment memo. Reply with ONLY JSON.
Stats block (ground truth):
{stats}

Memo:
{memo}

Keys:
- "numbers_faithful": true if every figure in the memo appears verbatim-compatible with the stats block (no new/recomputed numbers)
- "decision_respected": true if the memo's recommendation matches the DECISION line
- "invented_priors": true if the memo cites any prior experiment ID not in: {prior_ids}
"""


def load_gold():
    return [json.loads(l) for l in GOLD.read_text(encoding="utf-8").splitlines() if l.strip()]


def run_decision_suite():
    rows, passed = [], 0
    for case in load_gold():
        r = analyze(case["control_n"], case["control_conv"], case["treat_n"], case["treat_conv"])
        ok = r.decision == case["expected_decision"]
        passed += ok
        rows.append({"id": case["id"], "name": case["name"], "expected": case["expected_decision"],
                     "got": r.decision, "p_value": round(r.p_value, 4), "pass": ok})
    return rows, passed


def run_memo_faithfulness(sample_ids, api_key: str):
    from anthropic import Anthropic
    from memo import write_memo, stats_to_block, MODEL
    from retrieval import Memory

    client = Anthropic(api_key=api_key)
    mem = Memory()
    results = []
    for case in [c for c in load_gold() if c["id"] in sample_ids]:
        r = analyze(case["control_n"], case["control_conv"], case["treat_n"], case["treat_conv"])
        block = stats_to_block(r)
        priors = mem.similar(case["name"], k=2)
        m = write_memo(case["name"], block, priors, api_key=api_key)
        msg = client.messages.create(model=MODEL, max_tokens=150, messages=[{
            "role": "user", "content": JUDGE.format(stats=block, memo=m,
                                                    prior_ids=[p["id"] for p in priors] or "none")}])
        text = "".join(b.text for b in msg.content if getattr(b, "type", "") == "text")
        text = text.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
        g = json.loads(text)
        results.append({"id": case["id"], "memo": m, **g,
                        "pass": g.get("numbers_faithful") and g.get("decision_respected")
                                and not g.get("invented_priors")})
    return results


GOLD_HISTORY = Path(__file__).parent / "gold_history.jsonl"


def run_history_suite(k: int = 4):
    """Deterministic RAG retrieval eval: expected readout in top-k for history questions."""
    from retrieval import Memory
    mem = Memory()
    rows, passed = [], 0
    for case in [json.loads(l) for l in GOLD_HISTORY.read_text(encoding="utf-8").splitlines() if l.strip()]:
        top = mem.similar(case["question"], k=k)
        got = any(h["id"] == case["expected_id"] for h in top)
        passed += got
        rows.append({"id": case["id"], "question": case["question"][:55],
                     "expected": case["expected_id"],
                     "retrieved": ", ".join(h["id"] for h in top), "pass": got})
    return rows, passed
