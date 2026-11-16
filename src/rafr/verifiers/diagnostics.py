"""II.4.2's full diagnostics table for V1 and V2 (step 6 deliverable; decisions.md D-2026-09-25c).

Pure numpy, like ``probe_analysis`` (whose functions it reuses): ``pipelines/step06_score_states.py
diagnostics`` reads the formal scores tables, y* of D_fit and the unlabeled pool (purpose
``VERIFIER_DIAGNOSTICS``; no other split), the perturbation sample, the paraphrases, E's
judgments of them and the perturbed texts' scores, and hands arrays in. No thresholds are set
here (II.4.2: "no threshold"): only numbers.

Rows of II.4.2 and where each comes from:

* discrimination -- AUROC, PR-AUC, accuracy, F1, F2 against y*;
* calibration -- ECE (15 equal-width bins, as step 2) and Brier;
* error correlation -- phi and Cohen's kappa of the two verifiers' error indicators;
* joint error pattern P(e) -- V1 only, V2 only, both wrong (after the fact, needs y*);
* disagreement rate -- the share of states on which the two verdicts differ;
* error rate on rewrite text -- FNR on the E-UNSAFE states, FPR on the E-SAFE states. The
  states ARE rewrites (G_fixed's), which is II.4.2's point: decisions happen on rewritten text
  (2026-09-25: step 2's reading);
* intrinsic stability -- per perturbation (prefix, paraphrase) and verifier, the share of
  sampled states whose verdict ``1[s >= tau]`` is the same before and after. Two columns: all,
  and the subset whose E label did not change. Only paraphrases are judged by E; a prefix
  does not change toxicity by construction, so its second column is N/A.

The verdict uses tau_i: the frozen value when ``frozen/tau.json`` exists, otherwise tau fitted
here on D_fit with step 7's rule (F1-max, plateau midpoint; the same scores, so the same
number) and marked as not yet frozen.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from rafr.verifiers import probe_analysis as pa

VERIFIERS = ("v1", "v2")
PREFIX_E_NOTE = (
    "N/A: prefixed texts are not sent to E -- a neutral, content-free prefix does not change "
    "toxicity by construction, so the 'all' column already measures the verifier's own "
    "instability (D-2026-09-25c point 4)"
)


def single(scores: np.ndarray, y: np.ndarray, tau: float, p: float) -> dict[str, float]:
    """One verifier on one population at one tau: every II.4.2 single-verifier number."""
    m = pa.threshold_metrics(scores, y, tau, p)
    keep = ("n", "n_unsafe", "auroc", "pr_auc", "accuracy", "f1", "f2", "precision", "tpr",
            "fpr", "ece", "brier", "fnr_unsafe", "fpr_safe")  # fmt: skip
    return {k: m[k] for k in keep}


def pair(s1: np.ndarray, tau1: float, s2: np.ndarray, tau2: float, y: np.ndarray) -> dict:
    m = pa.pair_metrics(s1, tau1, s2, tau2, y)
    t = np.asarray(y, dtype=int)
    e1 = (np.asarray(s1) >= tau1).astype(int) != t
    e2 = (np.asarray(s2) >= tau2).astype(int) != t
    m["p_both_right"] = float((~e1 & ~e2).mean())
    return m


def consistency(before: np.ndarray, after: np.ndarray, tau: float) -> dict[str, Any]:
    """Share of items whose verdict ``1[s >= tau]`` is the same before and after, and the
    flips in each direction."""
    b = np.asarray(before, dtype=float) >= tau
    a = np.asarray(after, dtype=float) >= tau
    n = int(len(b))
    same = int((a == b).sum())
    return {
        "n": n,
        "n_consistent": same,
        "rate": same / n if n else float("nan"),
        "flips_toxic_to_clean": int((b & ~a).sum()),
        "flips_clean_to_toxic": int((~b & a).sum()),
    }


def stability(
    sample: Sequence[Mapping[str, Any]],
    original: Mapping[str, Mapping[str, float]],
    perturbed: Mapping[str, Mapping[str, Mapping[str, float]]],
    taus: Mapping[str, float],
    paraphrases: Mapping[str, Mapping[str, Any]],
    judgments: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    """Per perturbation kind and verifier: consistency over all items and over the items whose
    E label did not change.

    ``sample`` rows carry state_id and e_label (the state's y*); ``original[v][state_id]`` and
    ``perturbed[kind][v][state_id]`` are scores; ``paraphrases[state_id]`` holds ``valid`` and
    ``invalid_reason``; ``judgments[state_id]`` holds E's ``label`` of the paraphrase (None when
    E failed) and ``e_error``.
    """
    out: dict[str, Any] = {}
    ids = [r["state_id"] for r in sample]
    y0 = {r["state_id"]: r["e_label"] for r in sample}

    # prefix: every sampled state
    kind = "prefix"
    out[kind] = {"n_sampled": len(ids), "by_verifier": {}, "e_unchanged_note": PREFIX_E_NOTE}
    for v in VERIFIERS:
        have = [s for s in ids if s in perturbed[kind][v]]
        if len(have) != len(ids):
            raise ValueError(f"{kind}/{v}: {len(ids) - len(have)} sampled state(s) not scored")
        out[kind]["by_verifier"][v] = {
            "all": consistency(
                [original[v][s] for s in ids], [perturbed[kind][v][s] for s in ids], taus[v]
            ),
            "e_unchanged": None,
        }

    # paraphrase: valid ones; the E-unchanged subset needs E's label of the paraphrase
    kind = "paraphrase"
    valid = [s for s in ids if paraphrases[s]["valid"]]
    invalid = Counter(paraphrases[s]["invalid_reason"] for s in ids if not paraphrases[s]["valid"])
    judged = [s for s in valid if (judgments.get(s) or {}).get("label") is not None]
    e_failed = [s for s in valid if s not in judged]
    unchanged = [s for s in judged if judgments[s]["label"] == y0[s]]
    changed = Counter(f"{y0[s]}->{judgments[s]['label']}" for s in judged if s not in unchanged)
    by_class = Counter(y0[s] for s in valid)
    out[kind] = {
        "n_sampled": len(ids),
        "n_valid": len(valid),
        "invalid_by_reason": dict(sorted(invalid.items())),
        "n_valid_by_state_label": dict(sorted(by_class.items())),
        "n_e_judged": len(judged),
        "n_e_failed": len(e_failed),
        "n_e_unchanged": len(unchanged),
        "e_changed": dict(sorted(changed.items())),
        "by_verifier": {},
    }
    for v in VERIFIERS:
        missing = [s for s in valid if s not in perturbed[kind][v]]
        if missing:
            raise ValueError(f"{kind}/{v}: {len(missing)} valid paraphrase(s) not scored")
        out[kind]["by_verifier"][v] = {
            "all": consistency(
                [original[v][s] for s in valid], [perturbed[kind][v][s] for s in valid], taus[v]
            ),
            "e_unchanged": consistency(
                [original[v][s] for s in unchanged],
                [perturbed[kind][v][s] for s in unchanged],
                taus[v],
            ),
        }
    return out


# ---------------------------------------------------------------------------------------
# report
# ---------------------------------------------------------------------------------------


def fmt(x: Any, nd: int = 4) -> str:
    if x is None:
        return "n/a"
    if isinstance(x, float):
        return "nan" if np.isnan(x) else f"{x:.{nd}f}"
    return str(x)


def _rate(c: Mapping[str, Any] | None) -> str:
    if c is None:
        return "N/A"
    return f"{fmt(c['rate'])} ({c['n_consistent']}/{c['n']})"


def render_report(mx: Mapping[str, Any]) -> str:
    ids = mx["verifiers"]
    out = [
        f"# Step 6 verifier diagnostics (II.4.2) -- run {mx['run_id']}",
        "",
        f"git HEAD `{mx.get('git_head')}`. D_fit + unlabeled pool only (no D_gate, D_gate_reserve "
        "or D_Exp1 label is read). y* = E's judgment of each state. No thresholds: numbers only.",
        "",
        "| verifier | config_id | model_version | tau | tau source |",
        "|---|---|---|---|---|",
    ]
    for v in VERIFIERS:
        i = ids[v]
        out.append(
            f"| {v.upper()} | `{i['config_id']}` | `{i['model_version']}` | "
            f"{mx['tau'][v]:.10g} | {mx['tau_source']} |"
        )
    out += ["", f"p (E-UNSAFE share of all states, from the manifest) = {fmt(mx['p'])}.", ""]
    head = ("| population | verifier | n | n E-UNSAFE | AUROC | PR-AUC | accuracy | F1 | F2 | "
            "ECE | Brier | FNR (E-UNSAFE) | FPR (E-SAFE) |")  # fmt: skip
    out += ["## Discrimination, calibration, error rate on rewrite text", "", head,
            "|" + "---|" * 13]  # fmt: skip
    for pop, per in mx["single"].items():
        for v in VERIFIERS:
            m = per[v]
            out.append(
                f"| {pop} | {v.upper()} | {m['n']} | {m['n_unsafe']} | {fmt(m['auroc'])} | "
                f"{fmt(m['pr_auc'])} | {fmt(m['accuracy'])} | {fmt(m['f1'])} | {fmt(m['f2'])} | "
                f"{fmt(m['ece'])} | {fmt(m['brier'])} | {fmt(m['fnr_unsafe'])} | "
                f"{fmt(m['fpr_safe'])} |"
            )
    out += [
        "",
        "ECE: 15 equal-width bins (as step 2). The states are G_fixed's rewrites, so FNR / FPR "
        "here are II.4.2's error rates on rewrite text. D_fit is in-sample for tau.",
        "",
        "## The pair: error correlation, joint error pattern, disagreement",
        "",
        "| population | phi (errors) | kappa (errors) | P(V1 only) | P(V2 only) | P(both wrong) | "
        "P(both right) | disagreement |",
        "|" + "---|" * 8,
    ]
    for pop, m in mx["pair"].items():
        out.append(
            f"| {pop} | {fmt(m['phi_errors'])} | {fmt(m['kappa_errors'])} | "
            f"{fmt(m['p_v1_only'])} | {fmt(m['p_v2_only'])} | {fmt(m['p_both_wrong'])} | "
            f"{fmt(m['p_both_right'])} | {fmt(m['disagreement'])} |"
        )
    st = mx["stability"]
    para = st["paraphrase"]
    out += [
        "",
        "## Intrinsic stability under wording perturbations",
        "",
        f"Sample: {para['n_sampled']} states of D_fit + pool, 150 E-UNSAFE and 150 E-SAFE (seed "
        f"component `perturb_sample`, {mx['sample']['sha256'][:12]}...). Verdict = 1[s >= tau].",
        "",
        "| perturbation | verifier | consistent, all | consistent, E label unchanged | "
        "flips toxic->clean | flips clean->toxic |",
        "|---|---|---|---|---|---|",
    ]
    for kind in ("prefix", "paraphrase"):
        for v in VERIFIERS:
            c = st[kind]["by_verifier"][v]
            out.append(
                f"| {kind} | {v.upper()} | {_rate(c['all'])} | {_rate(c['e_unchanged'])} | "
                f"{c['all']['flips_toxic_to_clean']} | {c['all']['flips_clean_to_toxic']} |"
            )
    out += [
        "",
        f"Prefix, E label unchanged: {st['prefix']['e_unchanged_note']}.",
        "",
        f"Paraphrases: {para['n_valid']} valid of {para['n_sampled']} (invalid, dropped: "
        f"{para['invalid_by_reason'] or 'none'}); valid by the state's E label "
        f"{para['n_valid_by_state_label']}. E judged {para['n_e_judged']} (failed "
        f"{para['n_e_failed']}); E label unchanged {para['n_e_unchanged']}, changed "
        f"{para['e_changed'] or 'none'}.",
        "",
        f"Prefix table and paraphrase prompt: {mx['prompts']}.",
        "",
    ]
    return "\n".join(out)


__all__ = ["consistency", "pair", "render_report", "single", "stability"]
