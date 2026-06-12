"""Experiment Readout Copilot — Streamlit app.

Turns raw A/B results into a statistically rigorous decision and an executive-ready
memo grounded in prior-experiment memory. Code calculates; AI narrates.
"""

import os

import pandas as pd
import streamlit as st

from stats import analyze, Result
from retrieval import Memory
from memo import write_memo, stats_to_block
from evals import run_decision_suite, run_memo_faithfulness, run_history_suite
from history_qa import ask_history

st.set_page_config(page_title="Experiment Readout Copilot", page_icon="🧪", layout="wide")


def get_api_key():
    try:
        if "ANTHROPIC_API_KEY" in st.secrets:
            return st.secrets["ANTHROPIC_API_KEY"]
    except Exception:
        pass
    return os.environ.get("ANTHROPIC_API_KEY")


@st.cache_resource
def memory() -> Memory:
    return Memory()


def decision_card(r: Result):
    color = {"SHIP": "green", "DO NOT SHIP": "red",
             "KEEP RUNNING": "orange", "INVESTIGATE": "violet"}[r.decision]
    st.markdown(f"## :{color}[{r.decision}]")
    st.markdown(f"**Rationale:** {r.rationale}")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Control", f"{r.p_control:.2%}", f"{r.control_conv:,}/{r.control_n:,}", delta_color="off")
    c2.metric("Treatment", f"{r.p_treat:.2%}", f"{r.treat_conv:,}/{r.treat_n:,}", delta_color="off")
    c3.metric("Relative lift", f"{r.rel_lift:+.1%}")
    c4.metric("p-value", f"{r.p_value:.4g}")
    with st.expander("Full statistics"):
        st.text(stats_to_block(r))


with st.sidebar:
    st.title("🧪 Experiment Readout Copilot")
    st.caption("Raw A/B results in → rigorous decision + executive memo out. "
               "Statistics are computed in code; the AI only narrates them.")
    st.markdown(f"**Prior-experiment memory:** {memory().size} readouts indexed (BM25).")
    if not get_api_key():
        st.warning("No ANTHROPIC_API_KEY in secrets — statistics and evals work; "
                   "AI memos are disabled until you add one.", icon="🔑")
    st.info("Educational prototype. Decision policy is documented in the README.", icon="ℹ️")

tab_analyze, tab_history, tab_evals, tab_about = st.tabs(["Analyze", "Ask History", "Evals", "About"])

# ----------------------------------------------------------------- ANALYZE
with tab_analyze:
    st.subheader("Experiment results")
    name = st.text_input("Experiment name", "Checkout: $0-deposit test")

    # DECISION: manual entry first. Most PMs have four numbers, not a CSV.
    mode = st.radio("Input", ["Enter numbers", "Upload CSV"], horizontal=True)
    cn = cc = tn = tc = None
    if mode == "Enter numbers":
        a, b = st.columns(2)
        with a:
            st.markdown("**Control**")
            cn = st.number_input("Users (control)", min_value=1, value=10000, step=100)
            cc = st.number_input("Conversions (control)", min_value=0, value=320, step=10)
        with b:
            st.markdown("**Treatment**")
            tn = st.number_input("Users (treatment)", min_value=1, value=10000, step=100)
            tc = st.number_input("Conversions (treatment)", min_value=0, value=380, step=10)
    else:
        st.caption("CSV with columns: variant, users, conversions — rows named control / treatment.")
        up = st.file_uploader("CSV", type=["csv"])
        if up:
            df = pd.read_csv(up)
            df.columns = [c.strip().lower() for c in df.columns]
            try:
                ctl = df[df["variant"].str.strip().str.lower() == "control"].iloc[0]
                trt = df[df["variant"].str.strip().str.lower() == "treatment"].iloc[0]
                cn, cc = int(ctl["users"]), int(ctl["conversions"])
                tn, tc = int(trt["users"]), int(trt["conversions"])
                st.success(f"Parsed — control {cc:,}/{cn:,}, treatment {tc:,}/{tn:,}")
            except Exception as exc:
                st.error(f"Couldn't parse CSV: {exc}")

    if st.button("Analyze", type="primary") and cn:
        try:
            r = analyze(cn, cc, tn, tc)
        except ValueError as exc:
            st.error(str(exc))
            st.stop()
        st.session_state["last"] = (name, r)

    if "last" in st.session_state:
        name, r = st.session_state["last"]
        st.divider()
        decision_card(r)

        priors = memory().similar(name + " " + r.decision, k=3)
        if priors:
            with st.expander("Similar prior experiments (retrieved)"):
                for p in priors:
                    st.markdown(f"**[{p['id']}] {p['title']}** — {p['result']} → *{p['decision']}*  \n{p['learning']}")

        st.divider()
        if get_api_key():
            if st.button("✍️ Write executive memo"):
                with st.spinner("Drafting (AI narrates — every number comes from the stats engine)…"):
                    st.session_state["memo"] = write_memo(name, stats_to_block(r), priors,
                                                          api_key=get_api_key())
            if "memo" in st.session_state:
                st.markdown(st.session_state["memo"])
        else:
            st.caption("Add ANTHROPIC_API_KEY in app secrets to enable the memo writer.")

# ------------------------------------------------------------- ASK HISTORY
with tab_history:
    st.subheader("Ask your experiment history")
    st.caption("RAG over the readout library: your question is matched against 40 prior "
               "experiments (BM25); the answer is grounded in the retrieved readouts and "
               "cites them by ID — or says plainly when the history doesn't cover it.")
    q = st.text_input("Question", "Have we ever tested free-shipping thresholds?")
    if st.button("Search history", type="primary"):
        if get_api_key():
            with st.spinner("Retrieving and synthesizing…"):
                answer, hits = ask_history(q, memory(), api_key=get_api_key())
            st.markdown(answer)
        else:
            hits = memory().similar(q, k=4)
            st.info("No API key — showing retrieval only (the grounded answer needs a key).")
        if hits:
            with st.expander(f"Retrieved readouts ({len(hits)})"):
                for h in hits:
                    st.markdown(f"**[{h['id']}] {h['title']}** — {h['result']} → *{h['decision']}*  \n{h['learning']}")

# ------------------------------------------------------------------- EVALS
with tab_evals:
    st.subheader("Evaluation suite")
    st.markdown(
        "**Layer 1 — Decision regression suite** (deterministic, free, instant): 20 locked "
        "scenarios guaranteeing the statistics engine's calls can't silently change. "
        "This measures code correctness.\n\n"
        "**Layer 2 — Memo faithfulness** (LLM-judged, uses your API key): does the memo use "
        "only the engine's numbers and respect its decision? This measures the AI layer."
    )
    if st.button("Run decision regression suite (free)"):
        rows, passed = run_decision_suite()
        st.metric("Result", f"{passed}/{len(rows)} pass")
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    if st.button("Run history retrieval suite (free)"):
        rows, passed = run_history_suite()
        st.metric("Retrieval hit-rate@4", f"{passed}/{len(rows)}")
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
        st.caption("The known miss (H07) is the acronym gap — query says 'sample ratio "
                   "mismatch', the readout says 'SRM'. Documented BM25 limitation; "
                   "embeddings are the measured v2.")

    sample = ["G01", "G03", "G07", "G10", "G18"]
    if get_api_key():
        if st.button(f"Run memo faithfulness on {len(sample)} scenarios (uses API)"):
            with st.spinner("Generating and judging memos…"):
                res = run_memo_faithfulness(sample, get_api_key())
            ok = sum(1 for x in res if x["pass"])
            st.metric("Memo faithfulness", f"{ok}/{len(res)} pass")
            for x in res:
                with st.expander(f"{x['id']} — {'PASS' if x['pass'] else 'FAIL'}"):
                    st.json({k: x[k] for k in ("numbers_faithful", "decision_respected", "invented_priors")})
                    st.markdown(x["memo"])
    else:
        st.caption("Add ANTHROPIC_API_KEY in app secrets to run Layer 2.")

# ------------------------------------------------------------------- ABOUT
with tab_about:
    st.markdown(
        """
**Why this exists.** PMs spend hours converting raw test results into readouts, and the
failure mode isn't slow writing — it's bad calls: peeking, underpowered verdicts,
sample-ratio mismatches treated as wins, and AI tools that hallucinate arithmetic.

**The one rule:** code calculates, AI narrates. The language model is never asked to
compute a statistic — it receives the engine's numbers verbatim and is judged (Layer 2)
on whether it stays faithful to them.

**What the engine checks:** two-proportion z-test with 95% CI, sample-ratio-mismatch
detection, power analysis with an honest *keep running* verdict, plus guards for broken
tracking and practical equivalence — both added after the eval suite caught the original
policy mis-calling them.

Built by **Christian Burnham** (product decisions, decision policy, gold-set design),
pair-built with Claude. Prior-experiment library is synthetic, modeled on real retail
experimentation patterns.
        """
    )
