"""Statistics engine for the Experiment Readout Copilot.

HARD RULE (the product decision that matters): this module computes every number.
The LLM layer narrates results; it is never asked to calculate, infer, or adjust
statistics. In a decision-support tool, arithmetic done by a language model is a
hallucination surface — so it's removed by design.

Implements, dependency-light (math.erf, no scipy):
- Two-proportion z-test (pooled SE), two-sided p-value
- 95% CI for the absolute difference (unpooled SE)
- Sample-ratio-mismatch (SRM) check vs the expected split
- Power check: required N/arm (alpha=.05, power=.80) for the observed effect
- A documented decision policy mapping stats -> SHIP / DO NOT SHIP / KEEP RUNNING / INVESTIGATE
"""

import math
from dataclasses import dataclass, asdict

Z_ALPHA = 1.959963984540054   # two-sided 95%
Z_POWER = 0.8416212335729143  # 80% power


def _phi(z: float) -> float:
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


@dataclass
class Result:
    control_n: int
    control_conv: int
    treat_n: int
    treat_conv: int
    p_control: float
    p_treat: float
    abs_diff: float
    rel_lift: float
    z: float
    p_value: float
    ci_low: float
    ci_high: float
    srm_p: float
    srm_flag: bool
    required_n_per_arm: int
    underpowered: bool
    decision: str
    rationale: str


def analyze(control_n: int, control_conv: int, treat_n: int, treat_conv: int,
            expected_split: float = 0.5) -> Result:
    if min(control_n, treat_n) <= 0:
        raise ValueError("Sample sizes must be positive.")
    if control_conv > control_n or treat_conv > treat_n:
        raise ValueError("Conversions cannot exceed sample size.")

    p1 = control_conv / control_n
    p2 = treat_conv / treat_n
    diff = p2 - p1
    lift = (diff / p1) if p1 > 0 else float("inf")

    # --- significance (pooled SE for the test)
    pooled = (control_conv + treat_conv) / (control_n + treat_n)
    se_pooled = math.sqrt(pooled * (1 - pooled) * (1 / control_n + 1 / treat_n))
    z = diff / se_pooled if se_pooled > 0 else 0.0
    p_value = 2 * (1 - _phi(abs(z)))

    # --- CI for the difference (unpooled SE)
    se_un = math.sqrt(p1 * (1 - p1) / control_n + p2 * (1 - p2) / treat_n)
    ci_low, ci_high = diff - Z_ALPHA * se_un, diff + Z_ALPHA * se_un

    # --- SRM: did traffic split as designed? (z-test on allocation; flag at p<0.001,
    # the conventional SRM threshold — a failed split invalidates the read.)
    n_total = control_n + treat_n
    exp_treat = n_total * expected_split
    se_alloc = math.sqrt(n_total * expected_split * (1 - expected_split))
    z_srm = (treat_n - exp_treat) / se_alloc if se_alloc > 0 else 0.0
    srm_p = 2 * (1 - _phi(abs(z_srm)))
    srm_flag = srm_p < 0.001

    # --- power: N/arm needed to detect the OBSERVED effect with 80% power
    if abs(diff) > 1e-12:
        pbar = (p1 + p2) / 2
        num = (Z_ALPHA * math.sqrt(2 * pbar * (1 - pbar))
               + Z_POWER * math.sqrt(p1 * (1 - p1) + p2 * (1 - p2))) ** 2
        required = math.ceil(num / diff**2)
    else:
        required = 10**9
    underpowered = min(control_n, treat_n) < required

    # --- decision policy (documented, deterministic, regression-tested in evals)
    # GUARD 1: zero conversions in both arms at any real sample size is a tracking
    # problem until proven otherwise — found by gold case G18.
    # GUARD 2: practical equivalence — with p >= 0.3 and |relative lift| under
    # NEGLIGIBLE_LIFT, "keep running" would chase astronomically large samples for an
    # effect nobody would ship anyway (gold case G06 demanded 2.2B/arm). Call it.
    NEGLIGIBLE_LIFT = 0.02
    if control_conv == 0 and treat_conv == 0:
        decision = "INVESTIGATE"
        rationale = ("Zero conversions in both arms. Verify event tracking and exposure "
                     "logging before interpreting this as a result.")
        return Result(control_n, control_conv, treat_n, treat_conv, p1, p2, diff, lift,
                      z, p_value, ci_low, ci_high, srm_p, srm_flag, required, underpowered,
                      decision, rationale)
    if srm_flag:
        decision = "INVESTIGATE"
        rationale = (f"Sample-ratio mismatch (p={srm_p:.2g}): traffic did not split as designed. "
                     "Diagnose assignment/logging before trusting any metric.")
    elif p_value < 0.05 and diff > 0:
        decision = "SHIP"
        rationale = (f"Statistically significant lift (p={p_value:.3g}); 95% CI for the absolute "
                     f"difference [{ci_low:+.4f}, {ci_high:+.4f}] excludes zero. Roll out with "
                     "guardrail monitoring.")
    elif p_value < 0.05 and diff < 0:
        decision = "DO NOT SHIP"
        rationale = f"Statistically significant decline (p={p_value:.3g}). Treatment loses."
    elif p_value >= 0.3 and abs(lift) < NEGLIGIBLE_LIFT:
        decision = "DO NOT SHIP"
        rationale = (f"No practically meaningful effect: observed relative lift {lift:+.1%} with "
                     f"p={p_value:.3g}. Detecting an effect this small would require "
                     f"~{required:,}/arm; continuing is unlikely to change the product call.")
    elif underpowered:
        decision = "KEEP RUNNING"
        rationale = (f"Not significant (p={p_value:.3g}) and underpowered for the observed effect: "
                     f"~{required:,}/arm needed for 80% power vs {min(control_n, treat_n):,} observed. "
                     "The honest call is more data, not a verdict.")
    else:
        decision = "DO NOT SHIP"
        rationale = (f"Adequately powered and not significant (p={p_value:.3g}); no detectable "
                     "effect at a sample size that should have found one.")

    return Result(control_n, control_conv, treat_n, treat_conv, p1, p2, diff, lift,
                  z, p_value, ci_low, ci_high, srm_p, srm_flag, required, underpowered,
                  decision, rationale)


def as_dict(r: Result) -> dict:
    return asdict(r)
