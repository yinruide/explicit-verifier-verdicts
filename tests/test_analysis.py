"""analysis: accounting rules, DiD, TFR, parse_fail alerts, reliability, transport, fake data;
step10 end to end; step11/12 refuse while not frozen."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
from analysis import (
    _common,
    evidence_sensitivity,
    exp1_did,
    exp2_tfr,
    fake_data,
    parse_fail,
    reliability,
    transport,
)

from rafr.contracts.schemas import Split, text_hash
from rafr.core.config import NotFrozenError, load_config
from rafr.evaluator import judge

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def test_accounting_rules():
    ok = {"parse_fail": False, "action": "accept"}
    rw = {"parse_fail": False, "action": "rewrite"}
    pf = {"parse_fail": True, "action": None}
    assert _common.accepted(ok) is True and _common.accepted(rw) is False
    assert _common.accepted(pf, "main") is False
    assert _common.accepted(pf, "parse_fail_as_accept") is True
    assert _common.accepted(pf, "complete_case") is None


def _rows_with_effect(n=40, effect_high=0.5, effect_low=0.0):
    """A1 accepts UNSAFE states more than B at high lambda -> positive DiD (B better)."""
    rows = fake_data.fake_exp1_rows(n, seed=1, parse_fail_rate=0.0)
    import random

    r = random.Random(0)
    for row in rows:
        eff = effect_high if row["lam"] == "high" else effect_low if row["lam"] == "low" else 0.2
        p_accept = 0.5 + eff / 2 if row["arm"] == "A1" else 0.5 - eff / 2
        row["action"] = "accept" if r.random() < p_accept else "rewrite"
        row["parse_fail"] = False
    return rows


def test_exp1_did_sign_and_cells():
    rows = _rows_with_effect()
    res = exp1_did.analyze(rows, seed=0, n_boot=200)
    est = res["primary_did"]["main"]["estimate"]
    assert est > 0.15  # A1 - B, positive = B better
    assert res["primary_did"]["main"]["n_states"] > 0
    d = res["secondary_delta_fa_by_lambda_main"]["values"]
    assert d["high"] > d["low"]
    assert set(res["cells_main"]) == {
        f"{a}|{lam}" for a in ("A1", "B", "permuted_B") for lam in ("low", "mid", "high")
    }
    lo, hi = res["primary_did"]["main"]["ci95"]
    assert lo <= est <= hi


def test_exp1_did_sensitivities_and_complete_case():
    rows = _rows_with_effect()
    # make one state fail to parse in a single cell: complete-case drops the whole state
    sid = rows[0]["state_id"]
    for r in rows:
        if r["state_id"] == sid and r["arm"] == "B" and r["lam"] == "high":
            r["parse_fail"], r["action"] = True, None
    res = exp1_did.analyze(rows, seed=0, n_boot=100)
    n_main = res["primary_did"]["main"]["n_states"]
    n_cc = res["primary_did"]["complete_case"]["n_states"]
    assert n_cc == n_main - 1
    assert isinstance(res["primary_robust_across_accountings"], bool)


def test_fake_rows_give_near_zero_effect():
    rows = fake_data.fake_exp1_rows(80, seed=3)
    res = exp1_did.analyze(rows, seed=0, n_boot=200)
    assert abs(res["primary_did"]["main"]["estimate"]) < 0.3


def test_parse_fail_report_alerts():
    rows = fake_data.fake_exp1_rows(30, seed=0, parse_fail_rate=0.5)
    rep = parse_fail.report(rows)
    assert rep["alerts"] and all(c["ALERT"] for c in rep["cells"].values())
    assert (
        "action_not_in_accept_rewrite"
        in next(iter(rep["cells"].values()))["attempt_failure_reasons"]
    )
    clean = parse_fail.report(fake_data.fake_exp1_rows(30, seed=0, parse_fail_rate=0.0))
    assert not clean["alerts"]
    assert "ALERT" in parse_fail.render(rep)


def test_exp2_tfr_analysis():
    finals = fake_data.fake_exp2_finals(100, seed=2)
    for f in finals:  # inject an effect: B fails less
        if f["arm"] == "B":
            f["task_success"] = True
    res = exp2_tfr.analyze(finals, seed=0, n_boot=200)
    assert res["primary_TFR"]["B"]["TFR"] == 0.0 and res["primary_TFR"]["A1"]["TFR"] > 0
    assert res["primary_contrast"]["estimate"] > 0
    # group 6d: per-round E judgments exist -> the parse_fail-as-accept sensitivity is computed
    pfa = res["sensitivity"]["parse_fail_as_accept"]
    assert pfa["rule"] == "parse_fail_as_accept" and pfa["n"] == 100
    assert res["secondary"]["A1"]["right_censored_at_K"] == 4
    assert "conditional_FAR" in res["secondary"]["A1"]
    assert res["primary_TFR"]["A1"]["n"] + res["primary_TFR"]["B"]["n"] == 100  # all trajectories
    # without round labels (pre-group-6d finals) the sensitivity is None, nothing else changes
    for f in finals:
        f.pop("round_e_labels")
    assert (
        exp2_tfr.analyze(finals, seed=0, n_boot=50)["sensitivity"]["parse_fail_as_accept"] is None
    )


def test_exp2_parse_fail_as_accept_uses_the_first_parse_fail_round_text():
    base = {
        "x0_id": "a", "arm": "A1", "x_T": "z", "accepted": False, "accepted_at": None,
        "n_rewrites": 3, "refusal_like_rewrites": 0, "e_label": "SAFE", "e_score": None,
        "task_success": True,
    }  # fmt: skip
    labels = {"1": "UNSAFE", "2": "UNSAFE", "3": "SAFE", "4": "SAFE", "5": "SAFE"}
    f_pf = {**base, "n_parse_fail": 1, "first_parse_fail_t": 2, "round_e_labels": labels}
    f_ok = {**base, "n_parse_fail": 0, "first_parse_fail_t": None, "round_e_labels": labels}
    # main: terminal SAFE -> no failure; as-accept: stopped at t=2 whose text is UNSAFE -> failure
    assert exp2_tfr._failure(f_pf, "main") is False
    assert exp2_tfr._failure(f_pf, "parse_fail_as_accept") is True
    assert exp2_tfr._failure(f_ok, "parse_fail_as_accept") is False
    assert exp2_tfr._failure({**f_pf, "round_e_labels": {}}, "parse_fail_as_accept") is None


def test_exp2_low_preservation_among_safe_is_descriptive_per_arm_and_needs_a_cutpoint():
    """(2026-09-18): among finals E judged SAFE, the share with quality(x0, x_T) below
    the cutpoint fixed before Exp 2, per arm. UNSAFE finals are not in the denominator; unscored
    finals are counted but excluded from the share; a TBD cutpoint gives counts only."""
    finals = [
        {"arm": "A1", "task_success": True, "semantic_score": 0.2},
        {"arm": "A1", "task_success": True, "semantic_score": 0.9},
        {"arm": "A1", "task_success": False, "semantic_score": 0.1},  # UNSAFE: not counted
        {"arm": "A1", "task_success": True, "semantic_score": None},  # not scored
        {"arm": "B", "task_success": True, "semantic_score": 0.5},
        {"arm": "B", "task_success": True},  # key absent = not scored
    ]
    rep = exp2_tfr.low_preservation_among_safe(finals, 0.5)
    assert rep["cutpoint"] == 0.5
    assert rep["by_arm"]["A1"] == {
        "n_safe": 3, "n_scored": 2, "n_below_cutpoint": 1, "share_below_cutpoint": 0.5
    }  # fmt: skip
    # 0.5 is not strictly below 0.5
    assert rep["by_arm"]["B"] == {
        "n_safe": 2, "n_scored": 1, "n_below_cutpoint": 0, "share_below_cutpoint": 0.0
    }  # fmt: skip
    none = exp2_tfr.low_preservation_among_safe(finals, None)
    assert none["cutpoint"] is None
    assert none["by_arm"]["A1"]["n_scored"] == 2
    assert none["by_arm"]["A1"]["n_below_cutpoint"] is None
    assert none["by_arm"]["A1"]["share_below_cutpoint"] is None
    # nothing scored -> share None, never 0
    unscored = [{"arm": "A1", "task_success": True, "semantic_score": None}]
    assert exp2_tfr.low_preservation_among_safe(unscored, 0.5)["by_arm"]["A1"] == {
        "n_safe": 1, "n_scored": 0, "n_below_cutpoint": 0, "share_below_cutpoint": None
    }  # fmt: skip
    # wired into analyze(); the primary TFR contrast does not depend on it
    fake = fake_data.fake_exp2_finals(40, seed=1)
    with_cut = exp2_tfr.analyze(fake, seed=0, n_boot=50, semantic_cutpoint=0.5)
    without = exp2_tfr.analyze(fake, seed=0, n_boot=50)
    assert with_cut["low_preservation_among_safe"]["cutpoint"] == 0.5
    assert without["low_preservation_among_safe"]["cutpoint"] is None
    assert with_cut["primary_contrast"] == without["primary_contrast"]
    a1 = with_cut["low_preservation_among_safe"]["by_arm"]["A1"]
    assert a1["n_scored"] == a1["n_safe"] > 0 and 0.0 <= a1["share_below_cutpoint"] <= 1.0


def test_exp2_mechanism_rates_from_round_judgments():
    def row(arm, t, x_t, action, revised=None, parse_fail=False):
        return {
            "arm": arm, "t": t, "x_t": x_t, "action": action,
            "parse_fail": parse_fail, "revised_text": revised,
        }  # fmt: skip

    rows = [
        # accept on a SAFE text (correct), accept on an UNSAFE text (wrong)
        row("A1", 1, "s1", "accept"),
        row("A1", 1, "u1", "accept"),
        # rewrite UNSAFE -> SAFE (improved), SAFE -> UNSAFE (degraded), UNSAFE -> UNSAFE
        row("A1", 1, "u2", "rewrite", "s2"),
        row("A1", 1, "s3", "rewrite", "u3"),
        row("A1", 2, "u4", "rewrite", "u5"),
        # parse_fail on an UNSAFE text counts as "not accepted" -> correct under main accounting
        row("A1", 2, "u6", None, parse_fail=True),
        # a text without judgment is skipped and counted
        row("B", 1, "unknown", "accept"),
    ]
    truth = {text_hash(t): ("SAFE" if t.startswith("s") else "UNSAFE")
             for t in ("s1", "u1", "u2", "s2", "s3", "u3", "u4", "u5", "u6")}  # fmt: skip
    m = exp2_tfr.mechanism(rows, truth)
    a = m["A1"]
    assert a["n_accepts"] == 2 and a["accept_decision_accuracy"] == 0.5
    assert a["n_decisions"] == 6 and a["decision_accuracy"] == pytest.approx(4 / 6)
    assert a["n_rewrites"] == 3
    assert a["rewrite_improvement_rate"] == 0.5 and a["rewrite_degradation_rate"] == 1.0
    assert a["rewrite_transitions"] == {"SAFE->UNSAFE": 1, "UNSAFE->SAFE": 1, "UNSAFE->UNSAFE": 1}
    assert m["B"]["n_decisions"] == 0 and m["B"]["n_rows_without_judgment"] == 1


def test_reliability_truth_helpers():
    rj = [{"text_id": "abc", "e_label": "UNSAFE"}, {"text_id": "def", "e_label": "SAFE"}]
    assert reliability.truth_from_round_judgments(rj) == {"abc": "UNSAFE", "def": "SAFE"}
    rows = [{"x_t": "hello", "e_label": "SAFE"}, {"x_t": "x", "e_label": None}]
    assert reliability.truth_from_rows(rows) == {text_hash("hello"): "SAFE"}


def test_reliability_fill_and_r_t():
    finals = fake_data.fake_exp2_finals(20, seed=0)
    rows = fake_data.fake_exp2_rows(finals, seed=0)
    truth = {text_hash(r["x_t"]): ("UNSAFE" if r["s1"] >= 0.5 else "SAFE") for r in rows}
    n = reliability.fill_c_t(rows, truth)
    assert n == len(rows)
    rt = reliability.r_t_by_arm(rows)
    assert set(rt) <= {"A1", "B"}
    for arm in rt:
        for v in rt[arm].values():
            assert v["v1"] == 1.0  # truth was defined from v1's own judgment
            assert 0.0 <= v["v2"] <= 1.0 and v["n"] > 0
    # paper Sec. 3: r_ref is in the gate's units -- the detection rate on the E-UNSAFE states
    # (labels == 1) -- with the accuracy over all states alongside
    ref = reliability.r_ref([0.9, 0.1, 0.6], [1, 0, 0], 0.5)
    assert ref["detection_rate"] == 1.0 and ref["n_unsafe"] == 1
    assert ref["accuracy"] == pytest.approx(2 / 3) and ref["n"] == 3
    ref2 = reliability.r_ref([0.9, 0.1, 0.6, 0.2], [1, 0, 1, 1], 0.5)
    assert ref2["detection_rate"] == pytest.approx(2 / 3) and ref2["n_unsafe"] == 3
    assert ref2["accuracy"] == pytest.approx(3 / 4)
    with pytest.raises(ValueError):
        reliability.r_ref([0.9], [1, 0], 0.5)


def _es_row(sid, arm, lam, s1, s2, action, *, t=1, parse_fail=False):
    return {
        "state_id": sid, "arm": arm, "lam": lam, "t": t, "s1_tilde": s1, "s2_tilde": s2,
        "tau1": 0.5, "tau2": 0.5, "action": None if parse_fail else action,
        "parse_fail": parse_fail,
    }  # fmt: skip


def test_evidence_sensitivity_exp1_four_cells_by_arm_and_lambda():
    """paper Sec. 4.6: per arm x lambda, the accept rate in the four cells of the observed
    (y1, y2) with n per cell; parse_fail = not accepted (main accounting); the observed pair is
    the same for every arm at a (state, lambda), so the arm-consistency count is 0."""
    rows = []
    # four states, one per observed pair; A1 accepts everything, B accepts only on (0,0)
    for sid, (s1, s2) in zip("abcd", ((0.1, 0.2), (0.1, 0.9), (0.9, 0.2), (0.9, 0.9)), strict=True):
        for lam in ("low", "high"):
            rows.append(_es_row(sid, "A1", lam, s1, s2, "accept"))
            rows.append(
                _es_row(sid, "B", lam, s1, s2, "accept" if (s1, s2) == (0.1, 0.2) else "rewrite")
            )
    # a parse_fail on A1 at high for state d: not accepted under the main accounting
    for r in rows:
        if r["state_id"] == "d" and r["arm"] == "A1" and r["lam"] == "high":
            r["parse_fail"], r["action"] = True, None
    rep = evidence_sensitivity.exp1(rows)
    assert rep["grouped_by"] == "lam" and rep["accounting"] == "main"
    assert rep["pairs"] == ["(0,0)", "(0,1)", "(1,0)", "(1,1)"]
    assert rep["cells_differing_across_arms"] == 0
    low = rep["cells"]["low"]
    assert set(low) == {"A1", "B"}
    for key in rep["pairs"]:
        assert low["A1"][key]["n"] == 1 and low["A1"][key]["accept_rate"] == 1.0
    assert low["B"]["(0,0)"]["accept_rate"] == 1.0
    assert all(low["B"][k]["accept_rate"] == 0.0 for k in ("(0,1)", "(1,0)", "(1,1)"))
    high = rep["cells"]["high"]
    assert high["A1"]["(1,1)"] == {"n": 1, "n_excluded_by_accounting": 0, "n_accepted": 0,
                                   "accept_rate": 0.0}  # fmt: skip
    # the two paper Sec. 4.6 sensitivities run through the same function
    as_accept = evidence_sensitivity.exp1(rows, accounting="parse_fail_as_accept")
    assert as_accept["cells"]["high"]["A1"]["(1,1)"]["accept_rate"] == 1.0
    cc = evidence_sensitivity.exp1(rows, accounting="complete_case")
    assert cc["cells"]["high"]["A1"]["(1,1)"]["n"] == 0
    assert cc["cells"]["high"]["A1"]["(1,1)"]["n_excluded_by_accounting"] == 1
    assert (
        cc["cells"]["high"]["A1"]["(1,1)"]["accept_rate"]
        != cc["cells"]["high"]["A1"]["(1,1)"]["accept_rate"]
    )  # NaN  # noqa: E501
    text = evidence_sensitivity.render(rep)
    assert "III.1.2" in text and "(1,1)" in text and "n/a" in evidence_sensitivity.render(cc)
    # a channel that showed different pairs to different arms would be a bug, and is counted
    rows.append(_es_row("a", "B", "low", 0.9, 0.9, "rewrite"))
    assert evidence_sensitivity.exp1(rows)["cells_differing_across_arms"] == 1


def test_evidence_sensitivity_cells_use_the_states_own_post_channel_pair_only():
    """paper Sec. 4.6 (2026-09-25): the cell is the CURRENT state's own observed pair after
    C_lambda -- for permuted-B too (the donor's block is what it was shown as derived fields, the
    scores in its prompt are its own) -- never the raw pre-channel scores and never y*."""
    row = _es_row("s", "permuted_B", "high", 0.9, 0.9, "accept")
    row.update(
        {
            "s1": 0.1,  # raw, pre-channel: (0,0) -- must not be used
            "s2": 0.1,
            "e_label": "SAFE",  # y*: must not be used
            "permuted_from_state_id": "d",
            # the donor's block as shown to permuted-B: (0,0) -- must not be used either
            "fields": {
                "y1": 0,
                "y2": 0,
                "m1": -0.4,
                "m2": -0.4,
                "agreement": 1,
                "conflict_direction": "N/A",
            },  # fmt: skip
        }
    )
    own_b = _es_row("s", "B", "high", 0.9, 0.9, "accept")
    rep = evidence_sensitivity.exp1([row, own_b])
    for arm in ("permuted_B", "B"):
        cells = rep["cells"]["high"][arm]
        assert cells["(1,1)"]["n"] == 1 and cells["(1,1)"]["accept_rate"] == 1.0
        assert all(cells[k]["n"] == 0 for k in ("(0,0)", "(0,1)", "(1,0)"))
    assert rep["cells_differing_across_arms"] == 0  # own s~ agree across the two arms
    # and the module reads none of those fields at all
    src = (ROOT / "analysis" / "evidence_sensitivity.py").read_text("utf-8")
    for key in ('"s1"', '"s2"', "e_label", "permuted_from_state_id", '"fields"', "raw_s"):
        assert key not in src, key
    assert '"s1_tilde"' in (ROOT / "analysis" / "error_boundary.py").read_text("utf-8")


def test_evidence_sensitivity_exp2_groups_by_round():
    rows = [
        _es_row("x-A1-t1", "A1", "none", 0.9, 0.9, "rewrite", t=1),
        _es_row("x-A1-t2", "A1", "none", 0.1, 0.1, "accept", t=2),
        _es_row("y-B-t1", "B", "none", 0.9, 0.9, "accept", t=1),
    ]
    rep = evidence_sensitivity.exp2(rows)
    assert rep["grouped_by"] == "t" and set(rep["cells"]) == {"1", "2"}
    assert rep["cells"]["1"]["A1"]["(1,1)"]["accept_rate"] == 0.0
    assert rep["cells"]["1"]["B"]["(1,1)"]["accept_rate"] == 1.0
    assert rep["cells"]["2"]["A1"]["(0,0)"]["accept_rate"] == 1.0
    assert "cells_differing_across_arms" not in rep  # t >= 2 evidence is post-treatment
    # the fake-data rows run through it end to end, and every cell's n adds up
    fake_rows = fake_data.fake_exp2_rows(fake_data.fake_exp2_finals(20, seed=0), seed=0)
    rep2 = evidence_sensitivity.exp2(fake_rows)
    n = sum(
        c["n"] for arms in rep2["cells"].values() for cells in arms.values() for c in cells.values()
    )
    assert n == len(fake_rows)
    # paper Sec. 4.6: both runners write the table into their report
    src11 = (ROOT / "pipelines" / "step11_exp1.py").read_text("utf-8")
    src12 = (ROOT / "pipelines" / "step12_exp2.py").read_text("utf-8")
    assert '"evidence_sensitivity": evidence_sensitivity.exp1(rows)' in src11
    assert '"evidence_sensitivity": evidence_sensitivity.exp2(rows)' in src12


def test_transport_compare():
    rows1 = fake_data.fake_exp1_rows(30, seed=0)
    finals = fake_data.fake_exp2_finals(30, seed=0)
    rows2 = fake_data.fake_exp2_rows(finals, seed=0)
    out = transport.compare(rows1, rows2)
    assert out["s1"]["exp1_states"]["n"] == 30
    assert 0.0 <= out["s1"]["ks"] <= 1.0
    assert set(out["by_round_exp2"]) <= {1, 2, 3, 4}
    # group 6d: t = 1 rows are the Exp 1 states; the t >= 2 comparison is reported apart
    assert "t >= 2" in out["note"]
    assert "E-UNSAFE by construction" in out["note"]  #
    assert out["n_exp2_rows_t1"] + out["n_exp2_rows_t_ge_2"] == len(rows2)
    assert out["s1"]["exp2_t_ge_2"]["n"] == out["n_exp2_rows_t_ge_2"]
    assert 0.0 <= out["m1"]["ks_t_ge_2"] <= 1.0
    assert transport.ks_statistic([0, 0, 0], [1, 1, 1]) == 1.0


def test_step11_and_step12_refuse_while_not_frozen(tmp_path: Path, monkeypatch, sandbox):
    """They refuse today. Since the formatter's checker descriptions were filled (the authors
    2026-09-25) step 11 and the pod's loop pass the config guard and stop at
    their preconditions (tau not frozen; the x1 file); since the Exp 2 cutpoint was computed
    (2026-09-28) the report passes the guard too and stops at the missing run directory. The
    sandbox is there for the day this test starts doing real work (2026-09-20)."""
    from pipelines import step11_exp1 as s11
    from pipelines import step12_exp2 as s12

    with pytest.raises(SystemExit, match="tau.json"):
        s11.main(["--confirm"])
    with pytest.raises(SystemExit, match="not found"):  # the pod's loop: no formal x1 file yet
        s12.main(["loop", "--run-id", "x", "--x1-file", "f", "--x1-sha256", "h", "--confirm"])
    with pytest.raises(SystemExit, match="no run directory"):  # the report: no run yet
        s12.main(["report", "--run-id", "x"])


def _filled(obj):
    """The real configs with every TBD replaced by a dummy value."""
    if isinstance(obj, dict):
        return {k: _filled(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_filled(v) for v in obj]
    return "filled" if obj == "TBD" else obj


def test_step11_and_step12_guards_are_scoped_to_what_each_step_reads():
    """(2026-09-18): a step is blocked only by the config values it actually consumes.
    The Exp 2 semantic-preservation cutpoint (exp2.yaml) must not stop Exp 1, and an Exp 1
    design value must not stop Exp 2, which reads only exp1.outcome."""
    from pipelines import step11_exp1 as s11
    from pipelines import step12_exp2 as s12

    from rafr.core.config import assert_frozen, load_config

    cfgs = _filled(load_config())
    cfgs["exp2"]["analysis"]["semantic_preservation"]["cutpoint"] = "TBD"
    assert_frozen(cfgs, only_sections=s11.GUARD_SECTIONS)  # Exp 1 runs
    with pytest.raises(NotFrozenError) as ei:
        assert_frozen(cfgs, only_sections=s12.GUARD_SECTIONS)
    assert ei.value.tbd == ["exp2.analysis.semantic_preservation.cutpoint"]
    cfgs["exp2"]["analysis"]["semantic_preservation"]["cutpoint"] = 0.5
    cfgs["exp1"]["design"]["lambdas"] = "TBD"
    with pytest.raises(NotFrozenError) as ei2:
        assert_frozen(cfgs, only_sections=s11.GUARD_SECTIONS)
    assert ei2.value.tbd == ["exp1.design.lambdas"]
    assert_frozen(cfgs, only_sections=s12.GUARD_SECTIONS)  # Exp 2 runs
    cfgs["exp1"]["design"]["lambdas"] = ["low", "mid", "high"]
    # what both must still be blocked by: the shared backbone, E, the verifiers, the
    # refusal-phrase file entry (a prompt file in scope) and the controller prompts
    for sec in ("models.controller", "models.evaluator", "models.verifiers"):
        assert sec in s11.GUARD_SECTIONS and sec in s12.GUARD_SECTIONS
    cfgs["models"]["controller"]["model"] = "TBD"
    for scope in (s11.GUARD_SECTIONS, s12.GUARD_SECTIONS):
        with pytest.raises(NotFrozenError) as ei3:
            assert_frozen(cfgs, only_sections=scope)
        assert ei3.value.tbd == ["models.controller.model"]
    cfgs["models"]["controller"]["model"] = "filled"
    del cfgs["generation"]["g_fixed"]["validity"]["refusal_phrases_file"]
    for scope in (s11.GUARD_SECTIONS, s12.GUARD_SECTIONS):
        with pytest.raises(NotFrozenError) as ei4:
            assert_frozen(cfgs, only_sections=scope)
        assert ei4.value.missing == ["generation.g_fixed.validity.refusal_phrases_file"]
    # states come frozen from step04: G_fixed's own prompt / decoding are not Exp 1's business
    assert "models.g_fixed" not in s11.GUARD_SECTIONS


def test_exp2_refusal_like_sensitivity_conditions_on_terminal_text_only():
    """paper Sec. 4.6: only a trajectory whose FINAL output carries the flag counts as a
    terminal failure; a refusal-looking rewrite in the middle does not."""
    base = {
        "x0_id": "a", "arm": "A1", "x_T": "z", "accepted": True, "accepted_at": 2,
        "n_rewrites": 1, "n_parse_fail": 0, "first_parse_fail_t": None,
        "e_label": "SAFE", "e_score": None, "task_success": True, "round_e_labels": {"1": "SAFE"},
    }  # fmt: skip
    mid_only = {**base, "refusal_like_rewrites": 1, "x_T_refusal_like": False}
    terminal = {**base, "refusal_like_rewrites": 1, "x_T_refusal_like": True}
    assert exp2_tfr._failure(mid_only, "refusal_like_as_failure") is False
    assert exp2_tfr._failure(terminal, "refusal_like_as_failure") is True
    # a trajectory that already failed stays a failure either way
    failed = {**base, "task_success": False, "refusal_like_rewrites": 0, "x_T_refusal_like": False}
    assert exp2_tfr._failure(failed, "refusal_like_as_failure") is True


def test_step12_reuses_step5_y_star_for_t1(tmp_path: Path, sandbox, monkeypatch):
    """paper Sec. 4.4: Exp 2's t = 1 texts ARE the D_Exp1 states, so their judgments are step
    5's y* read through the label gate (purpose EXP2_T1), from the configured E, covering every
    x1; step 12 refuses rather than re-buying one."""
    from pipelines import step12_exp2 as s12

    from rafr.contracts.schemas import EvalRecord, SafetyLabel, State
    from rafr.data.labels import YStar

    states = [
        State(state_id=f"x0-{i}-g1", x0_id=f"x0-{i}", x0=f"o{i}", text=f"state {i}")
        for i in range(3)
    ]
    ecfg = judge.build_config(load_config("models"))

    def rec(s, identity=True):
        return EvalRecord(
            text_id=text_hash(s.text), text=s.text, label="UNSAFE", model=ecfg.model,
            model_version=None, prompt_hash="p", e_prompt_sha256=ecfg.prompt_sha256,
            e_version=ecfg.version if identity else "other", decoding=ecfg.decoding,
        )  # fmt: skip

    seen: list = []

    def fake_read(split, purpose, **kw):
        seen.append((split, purpose))
        return [
            YStar(
                state_id=s.state_id,
                x0_id=s.x0_id,
                split=split,
                label=SafetyLabel.UNSAFE,
                record=rec(s),
            )  # fmt: skip
            for s in states[:2]
        ]

    monkeypatch.setattr(s12.L, "read_labels", fake_read)
    with pytest.raises(SystemExit, match="1 x1 state"):  # partial coverage
        s12.prior_from_y_star(states, ecfg)
    prior = s12.prior_from_y_star(states[:2], ecfg)
    assert set(prior) == {text_hash(s.text) for s in states[:2]}
    assert seen[-1] == (Split.D_EXP1, s12.L.LabelPurpose.EXP2_T1)
    # a record from a different E is refused
    monkeypatch.setattr(
        s12.L,
        "read_labels",
        lambda split, purpose, **kw: [
            YStar(
                state_id=states[0].state_id,
                x0_id=states[0].x0_id,
                split=split,
                label=SafetyLabel.UNSAFE,
                record=rec(states[0], identity=False),
            )  # fmt: skip
        ],
    )
    with pytest.raises(judge.EChangedError):
        s12.prior_from_y_star(states[:1], ecfg)
