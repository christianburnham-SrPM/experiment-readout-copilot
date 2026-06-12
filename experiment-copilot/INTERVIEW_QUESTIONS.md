# The 10 Probes — and Where Each Answer Lives

Spend one hour here before any interview. You don't defend code line-by-line; you
defend decisions. These are yours.

**1. Why this project?**
Readouts were my job at Lowe's — I scaled experimentation 5x and watched bad calls cost
more than slow writing ever did. I built the tool I'd have wanted, with the failure
modes I actually saw: SRM, underpowered verdicts, significance-vs-magnitude confusion.
→ README "The problem"

**2. Why doesn't the AI compute the statistics?**
Because LLM arithmetic is a hallucination surface in a decision tool. Code computes,
the model narrates, and the eval suite's faithfulness layer judges exactly that
contract. → memo.py system prompt; README "The one rule"

**3. Walk me through the decision policy.**
Seven rules, in priority order: data validity first (SRM, zero-conversion guards),
then significance with direction, then practical equivalence, then an honest
"keep running" that names the sample actually required. → stats.py decision block;
README policy table

**4. How do you know it works?**
Two eval layers, honestly labeled: a deterministic 20-scenario regression suite for
the engine (measures code, not the model) and an LLM-judged faithfulness layer for
the memo (measures the AI). Knowing what your eval measures is half the discipline.
→ evals.py; Evals tab

**5. Did the evals ever change the product?**
Yes — best story in the repo. The exact-null case told us to "keep running" toward
2.2 billion users per arm; that finding produced the practical-equivalence guard.
Zero-conversion arms were being treated as sampling noise; now they're a tracking
investigation. → README "earned its keep"; stats.py GUARD comments

**6. Why BM25 for memory instead of embeddings?**
Sixteen term-dense readouts: lexical retrieval is cheap, explainable, and sufficient —
and I can name its observed failure (synonym misses). Embeddings are v2, measured
against the same gold set, so the upgrade is a measurement, not a fashion choice.
→ retrieval.py docstring

**7. What are its limitations?**
Single metric, frequentist-only (peeking inflates false positives), synthetic memory
library, lexical synonym gaps. All stated in the README — calibrated claims beat
inflated ones. → README "Known limitations"

**8. What does v2 look like?**
Guardrail metrics with non-inferiority bounds, sequential testing, Bayesian readout
alongside, platform CSV imports, embedding memory, and a feedback loop where shipped
readouts become new memory. → README "V2"

**9. What did Claude build vs what did you build?**
Claude wrote most of the code under my direction. I chose the problem and user, set
the calculate/narrate rule, designed the decision policy and its guards, authored the
gold set from scenarios I've lived, and made every tradeoff above. That division —
PM judgment, AI acceleration, eval verification — is how I think product gets built now.

**10. Significant but tiny — would you ship G05?**
The engine says SHIP at p=0.009; the PM notes the lift is 2.5% relative on a huge
sample and weighs implementation cost. Statistics bound the decision; they don't
replace judgment — which is why the tool outputs a rationale, not just a verdict.
→ gold_set.jsonl G05

**11. Is this *really* RAG?**
Yes — and here's the taxonomy, because the question is fair. RAG is the
retrieve→ground→generate pattern plus evaluating groundedness; it is not defined by
vector databases. This system retrieves (BM25 over 40 readouts), grounds (model may
use only retrieved text), generates with citations, refuses on empty retrieval, and is
evaluated on retrieval hit-rate (9/10) and groundedness (invented-priors check). The
honest version of the critique is corpus size — at 16 documents retrieval was optional;
at 40 with a Q&A surface it's load-bearing for precision, and at a real org's 500+ it's
load-bearing for feasibility. The architecture doesn't change across that scale; the
evals measure it at every size. → history_qa.py; Evals tab
