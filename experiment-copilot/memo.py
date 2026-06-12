"""Executive memo generation.

HARD RULE: code calculates, the model narrates. Every number in the memo is passed
in verbatim from stats.py; the prompt forbids deriving, recomputing, or inventing
statistics. Prior-experiment context is retrieved (BM25) and may be referenced only
as provided.
"""

import os

from anthropic import Anthropic

MODEL = os.environ.get("COPILOT_MODEL", "claude-sonnet-4-6")

SYSTEM = """You write executive experiment readouts for product leadership.
Rules, non-negotiable:
- Use ONLY the numbers provided in the stats block, verbatim. Never compute, round
  differently, extrapolate, or invent any figure. If a number isn't provided, don't state one.
- The DECISION and RATIONALE are already determined by the statistics engine; your job is
  to communicate them clearly, not to overrule them.
- If prior experiments are provided, reference at most two, by ID, only for context
  (risks, guardrails, decay patterns). Do not invent prior experiments.
- Format: a tight memo — Headline (one line), Result, Decision, Risks & guardrails,
  Suggested next step. Under 220 words. Plain language, no hype."""


def write_memo(experiment_name: str, stats_block: str, priors, api_key: str | None = None) -> str:
    client = Anthropic(api_key=api_key) if api_key else Anthropic()
    prior_text = "\n".join(
        f"[{p['id']}] {p['title']} — {p['result']} — Decision: {p['decision']} — Learning: {p['learning']}"
        for p in priors
    ) or "(none)"
    msg = client.messages.create(
        model=MODEL, max_tokens=600, system=SYSTEM,
        messages=[{"role": "user", "content":
                   f"Experiment: {experiment_name}\n\nStats block (authoritative, verbatim):\n{stats_block}\n\n"
                   f"Similar prior experiments:\n{prior_text}\n\nWrite the readout memo."}],
    )
    return "".join(b.text for b in msg.content if getattr(b, "type", "") == "text")


def stats_to_block(r) -> str:
    return (
        f"Control: {r.control_conv:,}/{r.control_n:,} = {r.p_control:.2%}\n"
        f"Treatment: {r.treat_conv:,}/{r.treat_n:,} = {r.p_treat:.2%}\n"
        f"Absolute difference: {r.abs_diff:+.4f} ({r.abs_diff*100:+.2f} pts)\n"
        f"Relative lift: {r.rel_lift:+.1%}\n"
        f"Two-sided p-value: {r.p_value:.4g}\n"
        f"95% CI (absolute difference): [{r.ci_low:+.4f}, {r.ci_high:+.4f}]\n"
        f"SRM check p-value: {r.srm_p:.3g} ({'FLAGGED' if r.srm_flag else 'clean'})\n"
        f"Power: requires ~{r.required_n_per_arm:,}/arm for the observed effect "
        f"({'underpowered' if r.underpowered else 'adequately powered'})\n"
        f"DECISION: {r.decision}\nRATIONALE: {r.rationale}"
    )
