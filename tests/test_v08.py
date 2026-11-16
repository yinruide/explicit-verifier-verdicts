"""The nine analysis-layer clauses, one test group each.

 changed nothing about the experiment design; it filled in how the numbers are formed.
Each group below names the clause it pins down.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from analysis import _common, error_boundary, exp1_did, exp2_tfr, fake_data, parse_fail

from rafr.channel.degrade import DegradeConfigError, LevelParams, check_monotone
from rafr.contracts.schemas import BOOTSTRAP_ALPHA, BOOTSTRAP_B, Lambda
from rafr.validity import gate as G

ROOT = Path(__file__).resolve().parents[1]


def _ladder(v1, v2):
    return {
        "v1": dict(zip((Lambda.LOW, Lambda.MID, Lambda.HIGH), v1, strict=True)),
        "v2": dict(zip((Lambda.LOW, Lambda.MID, Lambda.HIGH), v2, strict=True)),
    }


def _correct(n: int, rates: dict[str, tuple[float, float, float]]):
    """Per-state correctness vectors with exactly the requested per-level accuracy."""
    out: dict[str, dict[Lambda, list[int]]] = {}
    for v, (lo, mid, hi) in rates.items():
        out[v] = {
            lam: [1] * round(p * n) + [0] * (n - round(p * n))
            for lam, p in zip((Lambda.LOW, Lambda.MID, Lambda.HIGH), (lo, mid, hi), strict=True)
        }
    return out


# 1. gate verdict = point estimates; CI reported, never decisive (paper Sec. 4.3) ------------


def test_gate_verdict_is_the_point_estimate_and_ci_cannot_change_it():
    correct = _correct(200, {"v1": (0.90, 0.70, 0.50), "v2": (0.85, 0.65, 0.45)})
    ci = G.ladder_ci(correct, n_boot=200, seed=1)
    passed = G.ladder_check(_ladder((0.90, 0.70, 0.50), (0.85, 0.65, 0.45)), 0.15, ci=ci)
    assert passed.passed and passed.ci is ci or passed.ci == ci
    # the same verdict with no CI at all, and with a CI whose lower bounds sit under delta_R
    assert G.ladder_check(_ladder((0.90, 0.70, 0.50), (0.85, 0.65, 0.45)), 0.15).passed
    wide = {**ci, "gaps": {v: {k: [-1.0, 1.0] for k in ci["gaps"][v]} for v in ("v1", "v2")}}
    assert G.ladder_check(_ladder((0.90, 0.70, 0.50), (0.85, 0.65, 0.45)), 0.15, ci=wide).passed
    # a gap exactly at delta_R passes (float noise must not decide a gate), a hair under fails
    exact = G.ladder_check(_ladder((0.85, 0.70, 0.55), (170 / 200, 140 / 200, 110 / 200)), 0.15)
    assert exact.passed, exact.failed_conditions
    near = G.ladder_check(_ladder((0.90, 0.751, 0.60), (0.90, 0.75, 0.60)), 0.15)
    assert not near.passed and len(near.failed_conditions) == 1
    assert G.GAP_TOLERANCE < 1e-6  # slack for ties only, never a loosened threshold


def test_ladder_ci_covers_six_r_and_four_gaps_over_states():
    correct = _correct(120, {"v1": (0.90, 0.70, 0.50), "v2": (0.85, 0.65, 0.45)})
    ci = G.ladder_ci(correct, n_boot=300, seed=7)
    assert ci["unit"] == "state" and ci["n_states"] == 120
    assert ci["level"] == pytest.approx(1 - BOOTSTRAP_ALPHA)
    assert set(ci["r"]) == {"v1", "v2"} and set(ci["r"]["v1"]) == {"low", "mid", "high"}
    assert set(ci["gaps"]["v1"]) == {"low-mid", "mid-high"}
    assert sum(len(v) for v in ci["r"].values()) == 6
    assert sum(len(v) for v in ci["gaps"].values()) == 4
    for lam, point in (("low", 0.90), ("mid", 0.70), ("high", 0.50)):
        lo, hi = ci["r"]["v1"][lam]
        assert lo <= point <= hi
    lo, hi = ci["gaps"]["v1"]["low-mid"]
    assert lo <= 0.20 <= hi
    assert G.ladder_ci(correct, n_boot=300, seed=7) == ci  # deterministic in the seed
    assert G.ladder_ci(correct, n_boot=300, seed=8) != ci


def test_ladder_ci_refuses_misaligned_or_missing_vectors():
    correct = _correct(30, {"v1": (0.9, 0.7, 0.5), "v2": (0.9, 0.7, 0.5)})
    with pytest.raises(G.GateError):
        G.ladder_ci({"v1": correct["v1"]}, n_boot=10)
    short = {"v1": correct["v1"], "v2": {**correct["v2"], Lambda.HIGH: [1, 0]}}
    with pytest.raises(G.GateError):
        G.ladder_ci(short, n_boot=10)


def test_gate_attempt_record_carries_the_ci(tmp_path: Path):
    correct = _correct(50, {"v1": (0.9, 0.7, 0.5), "v2": (0.85, 0.65, 0.45)})
    ci = G.ladder_ci(correct, n_boot=100, seed=2)
    res = G.ladder_check(_ladder((0.9, 0.7, 0.5), (0.85, 0.65, 0.45)), 0.15, ci=ci)
    from rafr.contracts.schemas import Split

    path = tmp_path / "attempts.json"
    entry = G.record_attempt(path, res, Split.D_GATE)
    assert entry["ci"]["n_boot"] == 100 and entry["ci"]["seed"] == 2
    assert json.loads(path.read_text(encoding="utf-8"))["attempts"][0]["ci"]["unit"] == "state"


# 2. level naming; lambda_low may take p = 0 (paper Sec. 4.3) -------------------------------


def test_level_names_are_low_mid_high_everywhere():
    assert [lam.value for lam in G.LADDER] == ["low", "mid", "high"]
    assert list(exp1_did.LAMBDAS) == ["low", "mid", "high"]
    assert exp1_did.PRIMARY_LAMBDA_HIGH == "high" and exp1_did.PRIMARY_LAMBDA_LOW == "low"
    assert set(_load_yaml(ROOT / "configs" / "exp1.yaml")["design"]["lambdas"]) == {
        "low",
        "mid",
        "high",
    }
    text = (ROOT / "configs" / "channel.yaml").read_text(encoding="utf-8")
    assert "lambda1 = lambda_low" in text  # the naming is written down for the authors


def test_lambda_low_is_fully_unperturbed():
    """paper Sec. 4.3 (2026-09-26 revision): lambda_low takes p = 0 -- one level of entirely
    unperturbed evidence -- as part of the fixed 0 / 0.5 / 1; the old solve rule is gone."""
    degrade = _load_yaml(ROOT / "configs" / "channel.yaml")["degrade"]
    assert degrade["levels"]["low"] == {"v1": 0.0, "v2": 0.0}
    assert "anchor_lambda_low_at_zero" not in degrade
    zero = LevelParams.from_config({"v1": 0, "v2": 0})
    assert zero.p_v1 == 0.0 and zero.p_v2 == 0.0
    assert zero.derived_mode_probs() == {"none": 1.0, "v1_only": 0.0, "v2_only": 0.0, "both": 0.0}
    check_monotone(
        {
            Lambda.LOW: zero,
            Lambda.MID: LevelParams(0.3, 0.3),
            Lambda.HIGH: LevelParams(0.6, 0.6),
        }
    )
    with pytest.raises(DegradeConfigError):  # still monotone, still in [0, 1]
        check_monotone(
            {
                Lambda.LOW: LevelParams(0.3, 0.0),
                Lambda.MID: zero,
                Lambda.HIGH: LevelParams(0.6, 0.6),
            }
        )
    with pytest.raises(DegradeConfigError):
        LevelParams(-0.1, 0.0)


def _load_yaml(path: Path):
    import yaml

    return yaml.safe_load(path.read_text(encoding="utf-8"))


# 3. Exp 1 primary uses lambda_high vs lambda_low only (paper Sec. 4.6) --------------------


def _rows(n=40, seed=1, effect_high=0.4, effect_low=0.0, effect_mid=0.2):
    rows = fake_data.fake_exp1_rows(n, seed=seed)
    from rafr.core.seeding import py_rng

    r = py_rng(seed, "v08")
    for row in rows:
        eff = {"high": effect_high, "low": effect_low, "mid": effect_mid}[row["lam"]]
        p = 0.5 + eff / 2 if row["arm"] == "A1" else 0.5 - eff / 2
        row["action"] = "accept" if r.random() < p else "rewrite"
        row["parse_fail"] = False
    return rows


def test_primary_did_ignores_lambda_mid_and_reports_it_as_secondary():
    rows = _rows()
    res = exp1_did.analyze(rows, seed=0, n_boot=100)
    assert res["primary_lambdas"] == ["high", "low"]
    assert res["primary_did"]["main"]["lambdas"] == ["high", "low"]
    # flipping every mid-level decision must not move the primary estimate at all
    flipped = [
        {**r, "action": ("rewrite" if r["action"] == "accept" else "accept")}
        if r["lam"] == "mid"
        else r
        for r in rows
    ]
    assert (
        exp1_did.analyze(flipped, seed=0, n_boot=100)["primary_did"]["main"]["estimate"]
        == (res["primary_did"]["main"]["estimate"])
    )
    sec = res["secondary_delta_fa_by_lambda_main"]
    assert set(sec["values"]) == {"low", "mid", "high"}  # all three levels, secondary
    assert (
        exp1_did.analyze(flipped, seed=0, n_boot=100)["secondary_delta_fa_by_lambda_main"][
            "values"
        ]["mid"]
        != sec["values"]["mid"]
    )


# 4. 95% percentile bootstrap, B = 10000, seed on disk (paper Sec. 4.6) --------------------


def test_bootstrap_defaults_are_b10000_and_the_seed_is_reported():
    assert BOOTSTRAP_B == 10000 and BOOTSTRAP_ALPHA == 0.05
    res = exp1_did.analyze(_rows(), seed=123, n_boot=100)
    for block in (res["bootstrap"], res["primary_did"]["main"]["bootstrap"]):
        assert block["method"] == "percentile_bootstrap"
        assert block["level"] == 0.95 and block["seed"] == 123 and block["unit"] == "x0"
    finals = fake_data.fake_exp2_finals(60, seed=2)
    res2 = exp2_tfr.analyze(finals, seed=77, n_boot=100)
    assert res2["bootstrap"]["unit"] == "trajectory (stratified by arm)"
    assert res2["primary_contrast"]["bootstrap"]["seed"] == 77


def test_exp2_bootstrap_resamples_within_arm():
    """paper Sec. 4.6: arm-stratified, so every replicate keeps the randomized arm sizes."""
    seen: list[dict[str, int]] = []

    def stat(draw):
        seen.append({k: len(v) for k, v in draw.items()})
        return _common.mean_or_nan(draw["A1"]) - _common.mean_or_nan(draw["B"])

    strata = {"A1": [1.0] * 30, "B": [0.0] * 10}
    lo, hi = _common.stratified_bootstrap_ci(strata, stat, n_boot=50, seed=0)
    assert all(s == {"A1": 30, "B": 10} for s in seen)
    assert lo == hi == pytest.approx(1.0)  # no within-stratum variation in this toy case
    finals = fake_data.fake_exp2_finals(60, seed=2)
    c = exp2_tfr.contrast(finals, "main", seed=0, n_boot=100)
    assert c["n_by_arm"]["A1"] + c["n_by_arm"]["B"] == c["n"]


def test_one_master_seed_feeds_every_component():
    """Decision (2026-09-17): a single configured seed, derived per component by name.

    Eight separately configurable seeds carried no information this derivation does not.
    """
    from rafr.core.config import load_config, master_seed
    from rafr.core.seeding import COMPONENTS, component_seed, derive_seed

    cfgs = load_config()
    assert master_seed(cfgs) == 18042406
    seeds = {c: component_seed(master_seed(cfgs), c) for c in COMPONENTS}
    assert len(set(seeds.values())) == len(COMPONENTS)  # every component gets its own stream
    assert seeds["channel"] == derive_seed(18042406, "component", "channel")  # stable
    with pytest.raises(ValueError):
        component_seed(1, "not_a_component")
    # no seed is configured anywhere any more
    for name in ("models", "data", "channel", "exp1", "exp2", "generation"):
        text = (ROOT / "configs" / f"{name}.yaml").read_text(encoding="utf-8")
        for line in text.splitlines():
            code = line.split("#", 1)[0]
            assert "seed:" not in code or "master:" in code, f"{name}.yaml still configures {line}"


def test_bootstrap_seeds_come_from_the_master_seed():
    from pipelines import step08_tune_channel as step08
    from pipelines import step11_exp1 as step11
    from pipelines import step12_exp2 as step12

    from rafr.core.config import load_config, master_seed
    from rafr.core.seeding import component_seed

    cfgs = load_config()
    m = master_seed(cfgs)
    assert step11.bootstrap_settings(cfgs) == (BOOTSTRAP_B, component_seed(m, "exp1_run"))
    assert step12.bootstrap_settings(cfgs) == (BOOTSTRAP_B, component_seed(m, "exp2_run"))
    assert step08.ci_seed(cfgs) == component_seed(m, "gate_ci")
    for block in (
        _load_yaml(ROOT / "configs" / "exp1.yaml")["analysis"]["bootstrap"],
        _load_yaml(ROOT / "configs" / "exp2.yaml")["analysis"]["bootstrap"],
        _load_yaml(ROOT / "configs" / "channel.yaml")["gate"]["ci"],
    ):
        assert block["n_boot"] == BOOTSTRAP_B and "seed" not in block


# 5. Exp 1 N = D_Exp1 in full (paper Sec. 4.6) -----------------------------------------


def test_exp1_n_is_d_exp1_in_full():
    """(paper Sec. 4.6, 3): D_Exp1 is E-UNSAFE only, so the false accept rate's
    denominator is every D_Exp1 state; a SAFE state is a data error, not a smaller N."""
    rows = _rows()
    eff = exp1_did.analyze(rows, seed=0, n_boot=50)["effective_n"]
    sids = {r["state_id"] for r in rows}
    assert eff["n_states"] == len(sids)
    assert (
        eff["n_did_units"]
        == exp1_did.analyze(rows, seed=0, n_boot=50)["primary_did"]["main"]["n_states"]
    )
    assert "N = n_states" in eff["note"] and "III.1.3" in eff["note"]
    # every cell's FA is over all of D_Exp1
    for cell in exp1_did.cell_rates(rows).values():
        assert cell["n"] == len(sids)
    bad = [dict(r) for r in rows]
    for r in bad:
        if r["state_id"] == bad[0]["state_id"]:
            r["e_label"] = "SAFE"
    with pytest.raises(exp1_did.Exp1DataError, match="not E-UNSAFE"):
        exp1_did.analyze(bad, seed=0, n_boot=50)


# 6. H2 strata (paper Sec. 4.6) ------------------------------------------------------------


def test_h2_strata_are_arm_independent_and_split_delta_fa():
    rows = _rows()
    for r in rows:  # make stratum (i) non-trivial: perturb only the high level
        if r["lam"] == "high":
            r["s1_tilde"] = min(1.0, r["s1"] + 0.3)
    h2 = exp1_did.h2_strata(rows)
    assert set(h2) >= {"any_verifier_perturbed", "observed_disagreement"}
    for name in ("any_verifier_perturbed", "observed_disagreement"):
        block = h2[name]
        assert block["cells_differing_across_arms"] == 0  # arm-independent by construction
        assert set(block["strata"]) == {"True", "False"}
        for stratum in block["strata"].values():
            assert set(stratum["delta_fa_by_lambda"]) <= {"low", "mid", "high"}
    n_true = h2["any_verifier_perturbed"]["strata"]["True"]["n_rows"]
    n_false = h2["any_verifier_perturbed"]["strata"]["False"]["n_rows"]
    assert n_true + n_false == len(rows) and n_true > 0 and n_false > 0


def test_h2_stratum_i_prefers_the_logged_channel_assignment():
    rows = _rows(n=10)
    for r in rows:
        r["s1_tilde"], r["s2_tilde"] = r["s1"], r["s2"]  # rows alone would say "not perturbed"
    channel_rows = [
        {"state_id": r["state_id"], "lam": r["lam"], "perturbed_v1": True, "perturbed_v2": False}
        for r in rows
        if r["arm"] == "A1"
    ]
    assert exp1_did.h2_strata(rows)["any_verifier_perturbed"]["strata"]["True"]["n_rows"] == 0
    with_log = exp1_did.h2_strata(rows, channel_rows)
    assert with_log["source"] == "channel.jsonl"
    assert with_log["any_verifier_perturbed"]["strata"]["True"]["n_rows"] == len(rows)


def test_r_ref_is_reported_beside_the_strata_and_never_inside_them():
    import tokenize

    with (ROOT / "analysis" / "exp1_did.py").open("rb") as fh:
        names = {t.string for t in tokenize.tokenize(fh.readline) if t.type == tokenize.NAME}
    assert not (names & {"r_ref", "r_hat_ref"})  # paper Sec. 3: not an input to the stratification
    from pipelines import step11_exp1 as step11

    diag = ROOT / "tests" / "_missing_diagnostics.json"
    assert step11.read_r_ref(diag)["available"] is False


def test_step09_writes_r_ref_into_the_gate_diagnostics():
    src = (ROOT / "pipelines" / "step09_gate.py").read_text(encoding="utf-8")
    assert '"r_ref"' in src and "ladder_ci" in src


# 7. error boundary alignment (paper Sec. 6.2) ----------------------------------------------


def _row(arm, lam, s1, s2, action, t=1, parse_fail=False):
    return {
        "arm": arm,
        "lam": lam,
        "t": t,
        "state_id": f"s-{arm}-{lam}-{s1}-{s2}",
        "x_t": f"text {s1} {s2}",
        "s1": s1,
        "s2": s2,
        "s1_tilde": s1,
        "s2_tilde": s2,
        "tau1": 0.5,
        "tau2": 0.5,
        "action": action,
        "parse_fail": parse_fail,
        "e_label": "UNSAFE",
        "n_parse_retries": 0,
        "attempts": [],
    }


def test_error_boundary_selects_misaligned_units_and_scores_decision_errors():
    # E says UNSAFE everywhere. tau = 0.5, so s < 0.5 means the verifier says SAFE (misaligned).
    rows = [
        _row("A1", "high", 0.9, 0.9, "accept"),  # aligned evidence, accept -> not counted
        _row("A1", "high", 0.2, 0.9, "accept"),  # misaligned, accept on UNSAFE -> error
        _row("A1", "high", 0.2, 0.9, "rewrite"),  # misaligned, correct decision
        _row("B", "high", 0.2, 0.2, "rewrite"),  # misaligned, correct decision
    ]
    rep = error_boundary.exp1(rows)
    assert rep["grouped_by"] == "lam"
    a1 = rep["cells"]["high"]["A1"]
    assert a1["n"] == 3 and a1["n_misaligned"] == 2
    assert a1["decision_error_rate_on_misaligned"] == pytest.approx(0.5)
    assert rep["cells"]["high"]["B"]["decision_error_rate_on_misaligned"] == 0.0
    assert "misaligned" in error_boundary.render(rep)


def test_error_boundary_counts_parse_fail_as_not_accepted_and_uses_observed_scores():
    safe = {**_row("A1", "low", 0.9, 0.9, None, parse_fail=True), "e_label": "SAFE"}
    rep = error_boundary.exp1([safe])
    cell = rep["cells"]["low"]["A1"]
    # both verifiers say UNSAFE, E says SAFE -> misaligned; parse_fail = not accepted on a
    # SAFE text is a decision error under the main accounting (paper Sec. 4.6)
    assert cell["n_misaligned"] == 1 and cell["decision_error_rate_on_misaligned"] == 1.0
    cc = error_boundary.exp1([safe], accounting="complete_case")["cells"]["low"]["A1"]
    assert cc["n"] == 0 and cc["n_excluded_by_accounting"] == 1
    # the post-channel score decides what "observed" means
    shown_safe = {**_row("A1", "low", 0.9, 0.9, "accept"), "s1_tilde": 0.1, "s2_tilde": 0.1}
    assert error_boundary.verifier_judgments(shown_safe) == (0, 0)
    assert error_boundary.evidence_misaligned(shown_safe, "UNSAFE")


def test_error_boundary_exp2_groups_by_round():
    from rafr.contracts.schemas import text_hash

    rows = [_row("A1", "none", 0.2, 0.9, "accept", t=t) for t in (1, 2)]
    truth = {text_hash(r["x_t"]): "UNSAFE" for r in rows}
    rep = error_boundary.exp2(rows, truth)
    assert rep["grouped_by"] == "t" and set(rep["cells"]) == {"1", "2"}
    assert rep["cells"]["1"]["A1"]["decision_error_rate_on_misaligned"] == 1.0
    assert error_boundary.exp2(rows, {})["n_rows_without_judgment"] == 2


# 8. parse_fail grid: Exp 1 arm x lambda, Exp 2 arm x round (paper Sec. 4.6) ----------------


def test_parse_fail_grid_differs_by_experiment():
    rows1 = fake_data.fake_exp1_rows(20, seed=0, parse_fail_rate=0.5)
    rep1 = parse_fail.report(rows1, by=parse_fail.BY_LAMBDA)
    assert rep1["by"] == "lam"
    assert set(rep1["cells"]) == {
        f"{a}|{lam}" for a in ("A1", "B", "permuted_B") for lam in ("low", "mid", "high")
    }
    assert "lambda" in parse_fail.render(rep1)

    finals = fake_data.fake_exp2_finals(20, seed=1)
    rows2 = fake_data.fake_exp2_rows(finals, seed=1)
    rep2 = parse_fail.report(rows2, by=parse_fail.BY_ROUND)
    assert rep2["by"] == "t"
    assert all("|" in k and k.split("|")[0] in ("A1", "B") for k in rep2["cells"])
    assert {k.split("|")[1] for k in rep2["cells"]} <= {"1", "2", "3", "4"}
    assert "round" in parse_fail.render(rep2)
    with pytest.raises(ValueError):
        parse_fail.report(rows1, by="arm")


def test_exp2_run_summary_reports_parse_fail_by_round():
    from rafr.experiments.common import parse_fail_cells

    finals = fake_data.fake_exp2_finals(12, seed=4)
    rows = fake_data.fake_exp2_rows(finals, seed=4)
    cells = parse_fail_cells(rows, 0.02, by="t")
    assert all(k.split("|")[1].isdigit() for k in cells)
    src = (ROOT / "src" / "rafr" / "experiments" / "exp2.py").read_text(encoding="utf-8")
    assert 'parse_fail_cells(decision_rows, PARSE_FAIL_ALERT_THRESHOLD, by="t")' in src


# 9. "the x0 prefilter is never redone because E changed" --------------------
# Withdrawn by: prefilter, y* and outcome come from one E, and a changed E
# means rebuilding the chain. The rule -- same-E judgments are reused, another E's are
# refused -- is tested in tests/test_v09.py; so are the retried E failure and the removal of
# the candidate cap (judges the whole deduplicated pool).
