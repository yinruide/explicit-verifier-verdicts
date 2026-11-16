"""Step 10: the primary analysis, section by section.

Known-answer cases small enough to check by hand for every analysis function added or reworked
for step 10; bootstrap seeds fixed; the three parse_fail accountings of paper Sec. 4.6 checked for
direction on a small case; the fake generator, the FAKE guards and the step 10 run end to end.
(The mechanism, reliability and evidence-sensitivity functions have their known-answer cases in
tests/test_analysis.py.)
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
from analysis import (
    _common,
    design_checks,
    drift,
    error_boundary,
    exp1_did,
    exp2_tfr,
    fake_data,
    parse_fail,
    prereg_tables,
    transport,
)

from rafr.contracts.schemas import LogRecord, text_hash
from rafr.core import paths
from rafr.evaluator.panel import PanelJudgment

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# ---------------------------------------------------------------------------------------
# a hand-built Exp 1: four states, Y = accept
# ---------------------------------------------------------------------------------------

# (A1, B) accept indicators per state at each level; permuted_B copies B
Y = {
    "s1": {"high": (1, 0), "mid": (1, 0), "low": (0, 0)},  # D = (1 - 0) - (0 - 0) = 1
    "s2": {"high": (1, 1), "mid": (1, 0), "low": (1, 1)},  # D = 0
    "s3": {"high": (1, 0), "mid": (0, 0), "low": (1, 0)},  # D = 0
    "s4": {"high": (1, 0), "mid": (0, 0), "low": (1, 1)},  # D = 1
}
# FA: A1 high 1.0, B high .25 -> Delta_high .75; A1 low .75, B low .5 -> Delta_low .25;
# A1 mid .5, B mid 0 -> Delta_mid .5; D-bar = .5 = Delta_high - Delta_low


def _row(sid, arm, lam, accept, *, pf=False, st=(0.9, 0.9), reason=None, donor=None):
    return {
        "run_id": "t", "experiment": "exp1", "state_id": sid, "x0_id": f"x-{sid}", "arm": arm,
        "lam": lam, "t": 1, "x_t": f"text {sid}", "parse_fail": pf,
        "action": None if pf else ("accept" if accept else "rewrite"),
        "revised_text": None, "refusal_like": False, "s1": 0.9, "s2": 0.9,
        "s1_tilde": st[0], "s2_tilde": st[1], "tau1": 0.5, "tau2": 0.5, "e_label": "UNSAFE",
        "n_parse_retries": 2 if pf else 0, "permuted_from_state_id": donor,
        "attempts": [{"attempt": a, "parse_fail_reason": reason, "prompt_hash": f"{sid}{arm}{lam}"}
                     for a in range(3 if pf else 1)],
    }  # fmt: skip


def exp1_rows(evidence=None, pf=None):
    """``evidence[(sid, lam)] = (s1~, s2~)``; ``pf = {(sid, arm, lam)}`` parse_fails."""
    evidence, pf = evidence or {}, pf or set()
    donors = {"s1": "s2", "s2": "s3", "s3": "s4", "s4": "s1"}
    rows = []
    for sid, by_lam in Y.items():
        for lam, (a1, b) in by_lam.items():
            st = evidence.get((sid, lam), (0.9, 0.9))
            for arm, y in (("A1", a1), ("B", b), ("permuted_B", b)):
                rows.append(
                    _row(sid, arm, lam, y, pf=(sid, arm, lam) in pf, st=st,
                         donor=donors[sid] if arm == "permuted_B" else None)
                )  # fmt: skip
    return rows


def test_exp1_did_known_answer_and_three_level_curve():
    rows = exp1_rows()
    d = exp1_did.did(rows, seed=1, n_boot=200)
    assert d["estimate"] == pytest.approx(0.5) and d["n_states"] == 4
    assert d["lambdas"] == ["high", "low"]
    cells = exp1_did.cell_rates(rows)
    assert cells["A1|high"]["false_accept_rate"] == 1.0
    assert cells["B|high"]["false_accept_rate"] == 0.25
    assert cells["A1|low"]["false_accept_rate"] == 0.75
    assert cells["B|low"]["false_accept_rate"] == 0.5
    delta = exp1_did.delta_by_lambda(rows)
    assert delta == pytest.approx({"low": 0.25, "mid": 0.5, "high": 0.75})
    assert delta["high"] - delta["low"] == pytest.approx(d["estimate"])
    # permuted_B copies B: the secondary contrast is exactly 0
    assert exp1_did.did(rows, control="permuted_B", treated="B", n_boot=50)["estimate"] == 0.0
    assert exp1_did.effective_n(rows) == {**exp1_did.effective_n(rows), "n_states": 4}


def test_exp1_bootstrap_seed_is_fixed_and_recorded():
    rows = exp1_rows()
    a = exp1_did.did(rows, seed=7, n_boot=300)
    b = exp1_did.did(rows, seed=7, n_boot=300)
    assert a["ci95"] == b["ci95"] and a["bootstrap"]["seed"] == 7
    assert a["bootstrap"] == {
        "method": "percentile_bootstrap", "level": 0.95, "n_boot": 300, "seed": 7, "unit": "x0",
    }  # fmt: skip
    lo, hi = a["ci95"]
    assert lo <= a["estimate"] <= hi
    # a constant statistic has a degenerate interval whatever the seed
    assert _common.bootstrap_ci([0.5] * 5, _common.mean_or_nan, n_boot=50, seed=3) == (0.5, 0.5)


def test_stratified_bootstrap_keeps_arm_sizes_and_is_seeded():
    sizes = []

    def stat(draw):
        sizes.append((len(draw["A1"]), len(draw["B"])))
        return _common.mean_or_nan(draw["A1"]) - _common.mean_or_nan(draw["B"])

    ci = _common.stratified_bootstrap_ci({"A1": [1.0] * 3, "B": [0.0] * 2}, stat, n_boot=40, seed=2)
    assert ci == (1.0, 1.0) and set(sizes) == {(3, 2)}
    strata = {"A1": [0.0, 1.0, 1.0, 0.0], "B": [1.0, 0.0, 0.0]}
    one = _common.stratified_bootstrap_ci(strata, stat, n_boot=200, seed=9)
    assert one == _common.stratified_bootstrap_ci(strata, stat, n_boot=200, seed=9)


def test_exp1_parse_fail_three_accountings_move_in_the_expected_directions():
    """paper Sec. 4.6 on a small case: B's decision at (s1, high) fails to parse. Main counts it as
    not
    accepted (D_s1 = 1, as if B had rewritten), (a) as an accept (D_s1 = 0), (b) drops s1."""
    rows = exp1_rows(pf={("s1", "B", "high")})
    main = exp1_did.did(rows, accounting="main", n_boot=50)
    as_accept = exp1_did.did(rows, accounting="parse_fail_as_accept", n_boot=50)
    cc = exp1_did.did(rows, accounting="complete_case", n_boot=50)
    assert main["estimate"] == pytest.approx(0.5) and main["n_states"] == 4
    assert as_accept["estimate"] == pytest.approx(0.25) and as_accept["n_states"] == 4
    assert cc["estimate"] == pytest.approx(1 / 3) and cc["n_states"] == 3
    assert as_accept["estimate"] < cc["estimate"] < main["estimate"]
    # a parse_fail on the A1 side moves (a) the other way
    rows = exp1_rows(pf={("s1", "A1", "low")})
    assert exp1_did.did(rows, accounting="main", n_boot=50)["estimate"] == pytest.approx(0.5)
    assert exp1_did.did(rows, accounting="parse_fail_as_accept", n_boot=50)[
        "estimate"
    ] == pytest.approx(0.25)
    res = exp1_did.analyze(exp1_rows(pf={("s1", "B", "high")}), n_boot=50)
    assert res["primary_robust_across_accountings"] is True


def test_exp1_h2_strata_known_answer():
    # s1: V1 replaced at mid only (stratifier i); s2: the verifiers disagree at every level (ii)
    channel = []
    for sid in Y:
        for lam in ("low", "mid", "high"):
            p1 = lam == "high" or (lam == "mid" and sid == "s1")
            channel.append(
                {"state_id": sid, "lam": lam, "perturbed_v1": p1, "perturbed_v2": lam == "high"}
            )
    ev = {("s2", lam): (0.9, 0.1) for lam in ("low", "mid", "high")}
    h = exp1_did.h2_strata(exp1_rows(evidence=ev), channel)
    pert = h["any_verifier_perturbed"]["strata"]
    assert pert["True"]["delta_fa_by_lambda"] == pytest.approx({"mid": 1.0, "high": 0.75})
    assert pert["False"]["delta_fa_by_lambda"] == pytest.approx({"low": 0.25, "mid": 1 / 3})
    dis = h["observed_disagreement"]["strata"]["True"]
    assert dis["n_states"] == 1
    assert dis["delta_fa_by_lambda"] == pytest.approx({"low": 0.0, "mid": 1.0, "high": 0.0})
    assert h["any_verifier_perturbed"]["cells_differing_across_arms"] == 0
    assert h["observed_disagreement"]["cells_differing_across_arms"] == 0


def test_exp1_result_shape_reads_the_point_estimates_only():
    shape = exp1_did.result_shape(exp1_rows())
    a1 = shape["A1 - B"]
    assert a1["delta_fa_by_lambda"] == pytest.approx({"low": 0.25, "mid": 0.5, "high": 0.75})
    assert a1["rising_in_lambda"] is True and a1["negative_at_low"] is False
    pb = shape["permuted_B - B"]
    assert pb["rising_in_lambda"] is False and pb["negative_at_low"] is False  # all zero


def test_exp1_prompt_identity_ratio_c24():
    """per arm and pair of levels, the share of (state, arm) whose first prompt is the
    same; here two of four states have the same prompt hash at mid as at low."""
    from rafr.experiments.exp1 import identity_to_low_by_lambda

    rows = exp1_rows()
    for r in rows:
        same = r["state_id"] in ("s1", "s2") and r["lam"] in ("low", "mid")
        r["attempts"][0]["prompt_hash"] = f"{r['state_id']}{r['arm']}" + ("" if same else r["lam"])
    pw = identity_to_low_by_lambda(rows)["pairwise"]
    assert pw["low|mid"]["prompt_share"] == 0.5
    assert pw["low|mid"]["by_arm"]["A1"]["prompt_identical"] == 2
    assert pw["low|high"]["prompt_identical"] == 0 and pw["mid|high"]["prompt_identical"] == 0


# ---------------------------------------------------------------------------------------
# Exp 2: a hand-built set of finals
# ---------------------------------------------------------------------------------------


def _final(
    x0, arm, success, *, accepted=True, at=1, pf_t=None, rl=False, rounds=None, n_rw=0, sem=None
):
    return {
        "x0_id": x0, "x0": f"x0 {x0}", "arm": arm, "x_T": f"xT {x0}", "semantic_score": sem,
        "accepted": accepted, "accepted_at": at if accepted else None, "n_rewrites": n_rw,
        "n_parse_fail": int(pf_t is not None), "first_parse_fail_t": pf_t,
        "refusal_like_rewrites": int(rl), "x_T_refusal_like": rl,
        "e_label": "SAFE" if success else "UNSAFE", "e_score": None, "task_success": success,
        "round_e_labels": rounds or {"1": "UNSAFE"},
    }  # fmt: skip


def exp2_finals():
    return [
        _final("a1", "A1", False, at=1, sem=0.1),
        _final("a2", "A1", True, at=2, n_rw=1, sem=0.9),
        _final("a3", "A1", False, accepted=False, pf_t=2, n_rw=3,
               rounds={"1": "UNSAFE", "2": "UNSAFE", "5": "UNSAFE"}),
        _final("a4", "A1", True, at=3, rl=True, n_rw=2, sem=0.2),
        _final("b1", "B", True, at=2, rl=True, n_rw=1, sem=0.4),
        _final("b2", "B", True, at=2, rl=True, n_rw=1),
        _final("b3", "B", False, at=1),
        _final("b4", "B", True, at=1, rl=True, sem=0.3),
    ]  # fmt: skip


def test_exp2_tfr_contrast_and_the_ii82_sensitivities_known_answer():
    f = exp2_finals()
    assert exp2_tfr.tfr_by_arm(f) == {"A1": {"n": 4, "TFR": 0.5}, "B": {"n": 4, "TFR": 0.25}}
    main = exp2_tfr.contrast(f, "main", seed=4, n_boot=200)
    assert main["estimate"] == pytest.approx(0.25) and main["n_by_arm"] == {"A1": 4, "B": 4}
    assert main == exp2_tfr.contrast(f, "main", seed=4, n_boot=200)  # seed fixed
    assert main["bootstrap"]["unit"] == "trajectory (stratified by arm)"
    # (a) parse_fail as accept: a3 stops at its first parse_fail (t = 2), judged UNSAFE there
    a = exp2_tfr.contrast(f, "parse_fail_as_accept", n_boot=50)
    assert a["estimate"] == pytest.approx(0.25)
    # (b) complete case: a3 dropped -> A1 1/3
    b = exp2_tfr.contrast(f, "complete_case", n_boot=50)
    assert b["estimate"] == pytest.approx(1 / 3 - 0.25) and b["n_by_arm"] == {"A1": 3, "B": 4}
    # refusal_like: x_T's own flag -> a4, b1, b2, b4 fail -> A1 .75, B 1.0
    rl = exp2_tfr.contrast(f, "refusal_like_as_failure", n_boot=50)
    assert rl["estimate"] == pytest.approx(-0.25)
    res = exp2_tfr.analyze(f, n_boot=50)
    # paper Sec. 4.6: robust = main, (a), (b) agree; refusal_like has the other sign and does not
    # count
    assert res["robust_across_accountings"] is True


def test_exp2_parse_fail_as_accept_moves_with_the_judgment_of_the_stopping_round():
    f = exp2_finals()
    f[2]["round_e_labels"]["2"] = "SAFE"  # a3 would have ended on a SAFE text at t = 2
    a = exp2_tfr.contrast(f, "parse_fail_as_accept", n_boot=50)
    assert a["estimate"] == pytest.approx(0.0)  # A1 1/4 - B 1/4: (a) lowers A1's TFR
    assert exp2_tfr.contrast(f, "main", n_boot=50)["estimate"] == pytest.approx(0.25)


def test_exp2_secondary_and_semantic_counts_known_answer():
    s = exp2_tfr.secondary(exp2_finals())
    assert s["A1"]["n_accepted"] == 3 and s["A1"]["conditional_FAR"] == pytest.approx(1 / 3)
    assert s["A1"]["non_convergence_rate"] == 0.25 and s["B"]["non_convergence_rate"] == 0.0
    assert s["A1"]["rewrite_rounds_distribution"] == {0: 1, 1: 1, 2: 1, 3: 1}
    assert s["A1"]["rewrite_rounds_median"] == 2 and s["A1"]["right_censored_at_K"] == 4
    sem = exp2_tfr.low_preservation_among_safe(exp2_finals(), 0.35)
    # A1 SAFE: a2 (0.9), a4 (0.2) -> 1 of 2 below; B SAFE: b1 .4, b2 None, b4 .3 -> 1 of 2
    assert sem["by_arm"]["A1"] == {"n_safe": 2, "n_scored": 2, "n_below_cutpoint": 1,
                                   "share_below_cutpoint": 0.5}  # fmt: skip
    assert sem["by_arm"]["B"]["n_safe"] == 3 and sem["by_arm"]["B"]["n_scored"] == 2
    tbd = exp2_tfr.low_preservation_among_safe(exp2_finals(), None)
    assert tbd["by_arm"]["A1"]["share_below_cutpoint"] is None  # missing, never 0


def test_exp2_panel_sensitivity_known_answer():
    f = exp2_finals()
    member = {text_hash(x["x_T"]): ("UNSAFE" if x["x0_id"] in ("a1", "b3", "b4") else "SAFE")
              for x in f}  # fmt: skip
    member[text_hash("xT a2")] = None  # no verdict: a2 leaves this member's TFR only
    p = exp2_tfr.panel_sensitivity(f, {"m": member}, main_estimate=0.25, n_boot=50)
    m = p["members"]["m"]
    assert m["TFR_by_arm"] == {
        "A1": {"n": 3, "TFR": pytest.approx(1 / 3)},
        "B": {"n": 4, "TFR": 0.5},
    }
    assert m["contrast"]["estimate"] == pytest.approx(1 / 3 - 0.5)
    assert m["same_direction_as_main"] is False and m["n_without_verdict_by_arm"] == {"A1": 1}


# ---------------------------------------------------------------------------------------
# error boundary alignment, parse_fail report, transport, drift
# ---------------------------------------------------------------------------------------


def test_error_boundary_alignment_known_answer():
    rows = [
        _row("sA", "A1", "high", True, st=(0.9, 0.9)),  # evidence right, accept -> error
        _row("sB", "A1", "high", True, st=(0.1, 0.9)),  # V1 says safe on UNSAFE: misaligned
        _row("sC", "A1", "high", False, st=(0.1, 0.1)),  # misaligned, rewrite -> no error
        _row("sD", "A1", "high", False, pf=True, st=(0.1, 0.1)),  # parse_fail = not accepted
    ]
    c = error_boundary.exp1(rows)["cells"]["high"]["A1"]
    assert c["n"] == 4 and c["n_misaligned"] == 3 and c["misaligned_share"] == 0.75
    assert c["decision_error_rate_on_misaligned"] == pytest.approx(1 / 3)
    assert c["decision_error_rate_overall"] == 0.5
    # (a): the parse_fail becomes an accept on an UNSAFE text -> another error
    a = error_boundary.exp1(rows, accounting="parse_fail_as_accept")["cells"]["high"]["A1"]
    assert a["decision_error_rate_on_misaligned"] == pytest.approx(2 / 3)
    # Exp 2: E says SAFE; V1 flags it (misaligned); a rewrite of a SAFE text is an error
    r2 = {**_row("x", "B", "none", False, st=(0.9, 0.1)), "t": 2, "x_t": "clean"}
    e2 = error_boundary.exp2([r2], {text_hash("clean"): "SAFE"})["cells"]["2"]["B"]
    assert e2["n_misaligned"] == 1 and e2["decision_error_rate_on_misaligned"] == 1.0


def test_parse_fail_report_counts_truncation_on_its_own():
    rows = [
        _row("s1", "B", "high", False, pf=True, reason="malformed_json"),
        _row("s2", "B", "high", False, pf=True, reason="refusal"),
        _row("s3", "B", "high", True),
    ]
    rows[2]["attempts"] = [
        {"attempt": 0, "parse_fail_reason": "malformed_json"},
        {"attempt": 1, "parse_fail_reason": None},
    ]
    rows[2]["n_parse_retries"] = 1
    c = parse_fail.report(rows)["cells"]["B|high"]
    assert c["n"] == 3 and c["n_parse_fail"] == 2 and c["ALERT"] is True
    assert c["n_parse_fail_malformed_json"] == 1  # decisions, by their final attempt
    assert c["parse_fail_by_final_reason"] == {"malformed_json": 1, "refusal": 1}
    assert c["attempt_failure_reasons"] == {"malformed_json": 4, "refusal": 3}
    assert c["retries_distribution"] == {1: 1, 2: 2}


def test_transport_takes_exp2_unsafe_share_from_the_round_judgments():
    e1 = [_row("s1", "A1", "low", True), _row("s2", "A1", "low", True)]
    e2 = [
        {**_row("x1", "A1", "none", True), "t": 1, "x_t": "text s1", "e_label": None},
        {**_row("x1", "A1", "none", True), "t": 2, "x_t": "later a", "e_label": None},
        {**_row("x2", "B", "none", True), "t": 2, "x_t": "later b", "e_label": None},
    ]
    truth = {text_hash("text s1"): "UNSAFE", text_hash("later a"): "SAFE",
             text_hash("later b"): "UNSAFE"}  # fmt: skip
    out = transport.compare(e1, e2, truth)
    assert out["unsafe"]["exp1_states"]["mean"] == 1.0
    assert out["unsafe"]["exp2_t_ge_2"] == {**out["unsafe"]["exp2_t_ge_2"], "n": 2, "mean": 0.5}
    assert out["unsafe"]["exp2_decision_points"]["mean"] == pytest.approx(2 / 3)
    assert transport.compare(e1, e2)["unsafe"]["exp2_t_ge_2"] == {"n": 0}  # no truth, no share
    src12 = (ROOT / "pipelines" / "step12_exp2.py").read_text(encoding="utf-8")
    assert "transport.compare(rows1, rows, truth)" in src12  # step12 report passes it


def test_drift_ladder_on_d_exp1_known_answer():
    flags = {"low": (1, 1, 1, 0), "mid": (1, 0, 1, 0), "high": (0, 0, 1, 0)}  # V1, per state
    ev = {
        (sid, lam): (0.9 if flags[lam][i] else 0.1, 0.9) for lam in flags for i, sid in enumerate(Y)
    }
    rows = exp1_rows(evidence=ev)
    lad = drift.exp1_ladder(rows, seed=5, n_boot=100)
    assert {lam: lad["r"]["v1"][lam]["estimate"] for lam in flags} == {
        "low": 0.75,
        "mid": 0.5,
        "high": 0.25,
    }
    assert lad["r"]["v2"]["high"]["estimate"] == 1.0 and lad["n_states"] == 4
    assert lad == drift.exp1_ladder(rows, seed=5, n_boot=100)  # seed fixed
    cmp = drift.compare_to_gate(lad, {"v1": {"low": 0.7, "mid": 0.5, "high": 0.3}, "v2": {}})
    v1_low = next(x for x in cmp if x["verifier"] == "v1" and x["lambda"] == "low")
    assert v1_low["exp1_minus_gate"] == pytest.approx(0.05) and v1_low["R_gate"] == 0.7
    assert next(x for x in cmp if x["verifier"] == "v2")["R_gate"] is None
    rows[0]["s1_tilde"] = 0.99  # one arm sees different evidence: paper Sec. 4.3 broken, refused
    with pytest.raises(ValueError, match="different evidence"):
        drift.per_state(rows)


# ---------------------------------------------------------------------------------------
# construction checks and the E identity
# ---------------------------------------------------------------------------------------


def test_exp1_design_checks_pass_and_catch_violations():
    ok = design_checks.exp1(exp1_rows(), n_expected=4)
    assert all(c["ok"] for c in ok), [c for c in ok if not c["ok"]]
    rows = exp1_rows()
    rows[2]["permuted_from_state_id"] = "s3"  # s1's donor at high differs from its other levels
    rows[1]["s1_tilde"] = 0.2  # B sees other evidence than A1 at (s1, high)
    bad = {c["check"]: c for c in design_checks.exp1(rows, n_expected=4)}
    assert bad["permuted_b_donor_varies_across_lambda"]["ok"] is False
    assert bad["state_lambda_cells_with_different_evidence_across_arms"]["value"] == 1
    assert bad["n_states"]["ok"] is True


def test_e_identity_one_e_or_not():
    rec = {"e_prompt_sha256": "aa", "e_version": "v"}
    one = design_checks.e_identity({"exp1_outcome": [rec] * 2, "exp2_evaluations": [rec]})
    assert [r["n_records"] for r in one] == [2, 1] and all(r["single_e"] for r in one)
    two = design_checks.e_identity({"a": [rec], "b": [{**rec, "e_version": "w"}]})
    assert not any(r["single_e"] for r in two)


# ---------------------------------------------------------------------------------------
# the fake generator
# ---------------------------------------------------------------------------------------


@pytest.fixture(scope="module")
def fake_runs(tmp_path_factory):
    return fake_data.write_fake_runs(tmp_path_factory.mktemp("fake"), seed=11, n_states=300)


def _jsonl(p: Path) -> list[dict]:
    return [json.loads(x) for x in p.read_text(encoding="utf-8").splitlines() if x.strip()]


def test_fake_runs_have_the_real_schemas_and_the_required_cases(fake_runs):
    d1 = _jsonl(fake_runs.exp1_dir / "decisions.jsonl")
    d2 = _jsonl(fake_runs.exp2_dir / "decisions.jsonl")
    assert len(d1) == 300 * 3 * 3
    for r in d1 + d2:
        LogRecord.model_validate(r)
        assert r["run_id"].startswith(_common.FAKE_RUN_PREFIX)
    finals = _jsonl(fake_runs.exp2_dir / "trajectories.jsonl")
    assert len(finals) == 300 and {a: sum(f["arm"] == a for f in finals) for a in ("A1", "B")} == {
        "A1": 150,
        "B": 150,
    }
    assert max(int(r["t"]) for r in d2) <= 4
    reasons = {r["attempts"][-1]["parse_fail_reason"] for r in d1 + d2 if r["parse_fail"]}
    assert reasons == {
        "refusal",
        "action_not_in_accept_rewrite",
        "rewrite_with_empty_revised_text",
        "malformed_json",
    }
    assert any(r["refusal_like"] for r in d1) and any(r["refusal_like"] for r in d2)
    assert any(f["x_T_refusal_like"] for f in finals) and any(f["n_parse_fail"] for f in finals)
    assert any(not f["accepted"] for f in finals)
    panel = [
        PanelJudgment.model_validate(r)
        for r in _jsonl(fake_runs.exp2_dir / "panel_fake_member_a.jsonl")
    ]
    assert any(p.label is None for p in panel)
    seeds = {r["seed"] for r in _jsonl(fake_runs.exp2_dir / "assignment.jsonl")}
    manifest = json.loads(fake_runs.manifest.read_text(encoding="utf-8"))
    assert manifest["FAKE"] is True and seeds == {manifest["seeds"]["exp2_assignment"]}
    assert manifest["seeds"]["arm_shuffle"] == fake_runs.seeds.of("arm_shuffle")
    for d in (fake_runs.root, fake_runs.exp1_dir, fake_runs.exp2_dir):
        assert (d / _common.FAKE_MARKER).is_file()


def test_fake_runs_are_reproducible_from_the_seed(fake_runs, tmp_path):
    again = fake_data.write_fake_runs(tmp_path / "again", seed=11, n_states=300)

    def strip(x):
        if isinstance(x, dict):
            return {k: strip(v) for k, v in x.items() if k != "created_at"}
        return [strip(v) for v in x] if isinstance(x, list) else x

    for sub in ("FAKE_exp1_run/decisions.jsonl", "FAKE_exp2_run/decisions.jsonl",
                "FAKE_exp2_run/trajectories.jsonl"):  # fmt: skip
        assert strip(_jsonl(fake_runs.root / sub)) == strip(_jsonl(again.root / sub)), sub
    with pytest.raises(FileExistsError):  # a fake run is written once, never appended to
        fake_data.write_fake_runs(tmp_path / "again", seed=11, n_states=4)


def test_fake_exp1_arm_labels_are_a_shuffle_of_arm_blind_outcomes(fake_runs):
    """Per (state, lambda) the three outcomes are drawn without the arm and then shuffled over
    the arm labels: every design check passes and the effect is near zero."""
    d1 = _jsonl(fake_runs.exp1_dir / "decisions.jsonl")
    assert all(c["ok"] for c in design_checks.exp1(d1, n_expected=300))
    assert abs(exp1_did.did(d1, n_boot=20)["estimate"]) < 0.2


def test_exp2_design_checks_pass_on_the_fake_loop_and_catch_a_broken_one(fake_runs):
    d = fake_runs.exp2_dir
    finals, rows = _jsonl(d / "trajectories.jsonl"), _jsonl(d / "decisions.jsonl")
    assignment, rj = _jsonl(d / "assignment.jsonl"), _jsonl(d / "round_judgments.jsonl")
    e1 = _jsonl(fake_runs.exp1_dir / "decisions.jsonl")
    ok = design_checks.exp2(finals, rows, assignment, exp1_rows=e1, round_judgments=rj)
    assert all(c["ok"] for c in ok), [c for c in ok if not c["ok"]]
    # a parse_fail followed by a new text, and an x_T that is not the terminal rule's
    pf = next(r for r in rows if r["parse_fail"] and int(r["t"]) < 4)
    nxt = next(r for r in rows if r["x0_id"] == pf["x0_id"] and int(r["t"]) == int(pf["t"]) + 1)
    nxt["x_t"] = "something else"
    finals[0]["x_T"] = "not the terminal text"
    bad = {c["check"]: c for c in design_checks.exp2(finals, rows, assignment)}
    assert bad["parse_fail_followed_by_new_text"]["value"] == 1
    assert bad["x_T_not_by_the_terminal_rule"]["value"] >= 1
    assert bad["arm_sizes"]["ok"] is True


def test_fake_generator_reads_nothing_real():
    src = (ROOT / "analysis" / "fake_data.py").read_text(encoding="utf-8")
    for needle in ("rafr.core.paths", "paths.FROZEN", "load_tau", "read_labels", "load_config",
                   "DegradeChannel.load", "build_client", "load_scores"):  # fmt: skip
        assert needle not in src, needle


# ---------------------------------------------------------------------------------------
# the FAKE guards
# ---------------------------------------------------------------------------------------


def test_formal_readers_refuse_fake_data_and_step10_refuses_real_data(fake_runs, tmp_path):
    rows = _jsonl(fake_runs.exp1_dir / "decisions.jsonl")
    with pytest.raises(_common.FakeDataError):
        _common.refuse_fake(fake_runs.exp1_dir, what="t")
    with pytest.raises(_common.FakeDataError):  # the row marks are enough on their own
        _common.refuse_fake(tmp_path, rows, what="t")
    with pytest.raises(_common.FakeDataError):
        prereg_tables.load_inputs(fake_runs.exp1_dir, fake_runs.exp2_dir, fake=False)
    real = tmp_path / "real_run"
    real.mkdir()
    (real / "decisions.jsonl").write_text(
        json.dumps({**rows[0], "run_id": "exp1_x"}) + "\n", "utf-8"
    )
    with pytest.raises(_common.FakeDataError, match="no FAKE marker"):
        _common.require_fake(real, what="t")
    with pytest.raises(_common.FakeDataError):
        prereg_tables.load_inputs(real, fake_runs.exp2_dir, fake=True)
    (real / _common.FAKE_MARKER).write_text("x", "utf-8")
    with pytest.raises(_common.FakeDataError, match="not fake"):  # marker, but real rows
        _common.require_fake(real, [{"run_id": "exp1_x"}], what="t")


def test_step12_and_step11_refuse_a_fake_run_directory(fake_runs):
    from pipelines import step12_exp2

    with pytest.raises(_common.FakeDataError):
        step12_exp2.check_mode(fake_runs.exp2_dir, smoke=False)
    with pytest.raises(_common.FakeDataError):
        step12_exp2.check_mode(fake_runs.exp2_dir, smoke=True)
    src11 = (ROOT / "pipelines" / "step11_exp1.py").read_text(encoding="utf-8")
    src12 = (ROOT / "pipelines" / "step12_exp2.py").read_text(encoding="utf-8")
    assert 'refuse_fake(rd, what="step11")' in src11
    assert 'refuse_fake(rd, rows, what="step11 analysis")' in src11
    assert (
        'refuse_fake(Path(args.exp1_run_dir), rows1, what="step12 report --exp1-run-dir")' in src12
    )


# ---------------------------------------------------------------------------------------
# the section map and the tables
# ---------------------------------------------------------------------------------------


def test_tables_on_fake_runs_are_complete_and_marked(fake_runs, tmp_path):
    inp = prereg_tables.load_inputs(
        fake_runs.exp1_dir, fake_runs.exp2_dir, gate_report=fake_runs.gate_report, fake=True
    )
    res = prereg_tables.compute(inp, seed_exp1=1, seed_exp2=2, n_boot=50)
    tables = prereg_tables.build_tables(res, inp)
    assert [t.id for t in tables] == [t for s in prereg_tables.SECTIONS for t in s.tables]
    for t in tables:
        assert t.rows, t.id
        assert all(len(r) == len(t.columns) for r in t.rows), t.id
    assert all(c["ok"] for c in res["exp1_checks"] + res["exp2_checks"])
    by_id = {t.id: t for t in tables}
    assert by_id["T06_exp1_did"].rows[0][:3] == ["primary", "A1 - B", "main"]
    assert by_id["T062_r_ref"].rows[0][1] == 0.70  # the fake gate report, not step 9's
    written = prereg_tables.write_tables(tables, res, tmp_path / "tables", fake=True)
    assert written and all(p.name.startswith(_common.FAKE_FILE_PREFIX) for p in written)
    for p in written:
        if p.suffix == ".json":
            assert json.loads(p.read_text(encoding="utf-8"))["FAKE"] is True
    md = (tmp_path / "tables" / "FAKE_tables.md").read_text(encoding="utf-8")
    assert md.startswith("# FAKE") and "Rep x R_t interaction" in md
    # a formal reader refuses the tables too (FAKE_ in the name, marker above them)
    with pytest.raises(_common.FakeDataError):
        _common.refuse_fake(written[0], what="t")


def test_step10_end_to_end(tmp_path, sandbox):
    from pipelines import step10_fake_data_test as step10

    assert step10.main(["--n-states", "20", "--n-boot", "50", "--seed", "3"]) == 0
    rd = next((paths.RUNS).glob("step10_*"))
    assert (rd / _common.FAKE_MARKER).is_file() and (rd / "snapshot.json").is_file()
    tables = rd / "tables"
    assert (tables / "FAKE_tables.md").is_file()
    ids = {p.stem.removeprefix("FAKE_") for p in tables.glob("FAKE_T*.json")}
    assert ids == set(prereg_tables.BUILDERS)
    full = json.loads((tables / "FAKE_analysis_full.json").read_text(encoding="utf-8"))
    assert full["FAKE"] is True and full["settings"]["n_boot"] == 50
    assert full["exp1"]["effective_n"]["n_states"] == 20
    assert all(c["ok"] for c in full["exp1_checks"] + full["exp2_checks"])
