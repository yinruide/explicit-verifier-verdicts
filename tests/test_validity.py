"""validity: metrics, tau (F2 max, plateau midpoint), gate (ladder + attempts), cue diagnostics."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from rafr.contracts.schemas import GATE_MAX_ATTEMPTS, Lambda, Split
from rafr.validity import cue_diagnostics as CD
from rafr.validity import gate as G
from rafr.validity import metrics as M
from rafr.validity import tau as T

# ---------------------------------------------------------------- metrics


def test_metrics_basic():
    y = [1, 1, 0, 0]
    p = [1, 0, 0, 0]
    assert M.precision_recall(y, p) == (1.0, 0.5)
    assert M.f2(y, p) == pytest.approx((5 * 1.0 * 0.5) / (4 * 1.0 + 0.5))
    assert M.accuracy(y, p) == 0.75
    assert M.auroc([0.9, 0.8, 0.2, 0.1], y) == 1.0
    assert M.auroc([0.1, 0.2, 0.8, 0.9], y) == 0.0
    assert M.auroc([0.5, 0.5, 0.5, 0.5], y) == 0.5
    assert np.isnan(M.auroc([0.1, 0.2], [1, 1]))
    assert M.pr_auc([0.9, 0.8, 0.2, 0.1], y) == 1.0
    assert M.balanced_accuracy(y, p) == 0.75
    assert M.brier([1.0, 0.0], [1, 0]) == 0.0
    assert M.ece([0.9, 0.9, 0.1, 0.1], [1, 1, 0, 0]) == pytest.approx(0.1)
    assert M.phi([1, 1, 0, 0], [1, 1, 0, 0]) == 1.0
    assert M.cohen_kappa([1, 1, 0, 0], [1, 1, 0, 0]) == 1.0


# ---------------------------------------------------------------- tau


def test_tau_separable_takes_gap_midpoint():
    s = [0.1, 0.2, 0.3, 0.7, 0.8, 0.9]
    y = [0, 0, 0, 1, 1, 1]
    f = T.fit_tau(s, y, "v")
    assert f.f1_max == 1.0
    # candidates are midpoints, so the only F1 = 1 candidate is 0.5; it lies in the gap
    assert 0.3 < f.tau <= 0.7
    assert f.aux["recall_at_tau"] == 1.0 and f.aux["precision_at_tau"] == 1.0
    assert f.criterion == "f1_max_plateau_midpoint"


def test_tau_plateau_midpoint():
    # several thresholds tie on F1 -> midpoint of the widest run
    s = [0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.60, 0.90]
    y = [0, 1, 1, 0, 1, 1, 0, 1]
    f = T.fit_tau(s, y, "v")
    cands = T.candidate_thresholds(np.asarray(s))
    f1s = [M.f1(y, (np.asarray(s) >= c).astype(int)) for c in cands]
    assert f.f1_max == pytest.approx(max(f1s))
    assert f.plateau_lo <= f.tau <= f.plateau_hi
    assert f.tau == pytest.approx((f.plateau_lo + f.plateau_hi) / 2)
    assert f.n_plateaus >= 1


def test_tau_is_chosen_by_f1_and_f2_is_only_reported():
    """(paper Sec. 5): F1 decides, F2 is an auxiliary. On this data the two disagree:
    F2 peaks at the lowest threshold (0.1, recall 1), F1 at 0.45."""
    s = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
    y = [1, 0, 0, 0, 1, 0, 1, 1]
    arr = np.asarray(s)
    cands = T.candidate_thresholds(arr)
    f2s = [M.f2(y, (arr >= c).astype(int)) for c in cands]
    f2_best = cands[int(np.argmax(f2s))]
    f = T.fit_tau(s, y, "v")
    assert f.tau == pytest.approx(0.45) and f.f1_max == pytest.approx(0.75)
    assert f2_best == pytest.approx(0.1) and f.tau != pytest.approx(f2_best)
    assert f.aux["f2_at_tau"] == pytest.approx(M.f2(y, (arr >= f.tau).astype(int)))
    assert "f1_at_tau" not in f.aux  # F1 is the criterion itself, reported as f1_max


def test_tau_save_load(tmp_path: Path):
    f = T.fit_tau([0.1, 0.9], [0, 1], "v1")
    T.save_tau({"v1": f}, tmp_path / "tau.json", extra={"split": "d_fit"})
    back = T.load_tau(tmp_path / "tau.json")
    assert back["v1"].tau == f.tau and back["v1"].aux == f.aux
    assert json.loads((tmp_path / "tau.json").read_text())["split"] == "d_fit"


def test_tau_rejects_bad_input():
    with pytest.raises(ValueError):
        T.fit_tau([], [], "v")
    with pytest.raises(ValueError):
        T.fit_tau([0.1], [0, 1], "v")


# ---------------------------------------------------------------- gate


def _r(v1, v2):
    return {
        "v1": dict(zip((Lambda.LOW, Lambda.MID, Lambda.HIGH), v1, strict=True)),
        "v2": dict(zip((Lambda.LOW, Lambda.MID, Lambda.HIGH), v2, strict=True)),
    }


def test_ladder_check_four_conditions():
    ok = G.ladder_check(_r((0.9, 0.7, 0.5), (0.85, 0.65, 0.45)), 0.15)
    assert ok.passed and ok.failed_conditions == []
    assert ok.gaps["v1"] == {"low-mid": pytest.approx(0.2), "mid-high": pytest.approx(0.2)}
    small = G.ladder_check(_r((0.9, 0.8, 0.5), (0.85, 0.65, 0.45)), 0.15)
    assert (
        not small.passed
        and len(small.failed_conditions) == 1
        and "v1" in small.failed_conditions[0]
    )
    wrong_order = G.ladder_check(_r((0.9, 0.7, 0.5), (0.5, 0.7, 0.9)), 0.15)
    assert not wrong_order.passed and len(wrong_order.failed_conditions) == 2
    with pytest.raises(G.GateError):
        G.ladder_check({"v1": {Lambda.LOW: 0.9}}, 0.15)


def test_gate_attempts_bookkeeping(tmp_path: Path):
    path = tmp_path / "gate_attempts.json"
    fail = G.ladder_check(_r((0.9, 0.8, 0.5), (0.85, 0.65, 0.45)), 0.15)
    ok = G.ladder_check(_r((0.9, 0.7, 0.5), (0.85, 0.65, 0.45)), 0.15)
    assert G.next_attempt_split(path) is Split.D_GATE
    with pytest.raises(G.GateError):  # first attempt must use D_gate
        G.record_attempt(path, fail, Split.D_GATE_RESERVE)
    G.record_attempt(path, fail, Split.D_GATE)
    assert G.next_attempt_split(path) is Split.D_GATE_RESERVE
    with pytest.raises(G.GateError):  # same D_gate data may not be reused (paper Sec. 4.3)
        G.record_attempt(path, ok, Split.D_GATE)
    G.record_attempt(path, fail, Split.D_GATE_RESERVE)
    with pytest.raises(G.GateError):  # max 2 attempts
        G.next_attempt_split(path)
    data = json.loads(path.read_text())
    assert data["max_attempts"] == GATE_MAX_ATTEMPTS and len(data["attempts"]) == 2
    assert [a["passed"] for a in data["attempts"]] == [False, False]


def test_paired_correctness_correlation_is_phi_over_the_same_states():
    """paper App. B: per verifier and pair of levels, phi of the two correctness vectors
    over the same E-UNSAFE states, with the 2x2 counts; None where a vector is constant."""
    low = [1, 1, 1, 0, 0, 1]
    mid = [1, 1, 0, 0, 0, 1]
    high = [1, 1, 1, 1, 1, 1]  # constant: phi undefined
    vec = {v: {Lambda.LOW: low, Lambda.MID: mid, Lambda.HIGH: high} for v in G.VERIFIER_KEYS}
    out = G.paired_correctness_correlation(vec)
    lm = out["v1"]["low-mid"]
    assert (lm["n11"], lm["n10"], lm["n01"], lm["n00"], lm["n"]) == (3, 1, 0, 2, 6)
    assert lm["phi"] == pytest.approx(M.phi(low, mid))
    assert lm["phi"] == pytest.approx((3 * 2 - 1 * 0) / np.sqrt(4 * 3 * 2 * 3))
    assert out["v2"]["mid-high"]["phi"] is None and out["v1"]["low-high"]["phi"] is None
    with pytest.raises(G.GateError):
        G.paired_correctness_correlation(
            {v: {Lambda.LOW: low, Lambda.MID: mid[:-1], Lambda.HIGH: high} for v in ("v1", "v2")}
        )


def test_gate_pass_blocks_further_attempts(tmp_path: Path):
    path = tmp_path / "g.json"
    ok = G.ladder_check(_r((0.9, 0.7, 0.5), (0.85, 0.65, 0.45)), 0.15)
    G.record_attempt(path, ok, Split.D_GATE)
    with pytest.raises(G.GateError):
        G.next_attempt_split(path)


def test_reliability():
    assert G.reliability([1, 0, 1, 1], [1, 0, 0, 1]) == 0.75


# ---------------------------------------------------------------- cue diagnostics


def test_cue_diagnostics_five_items():
    rng = np.random.default_rng(0)
    n = 200
    y = rng.integers(0, 2, n)
    s1 = np.clip(0.5 + (y - 0.5) * 0.6 + rng.normal(0, 0.15, n), 0, 1)
    s2 = np.clip(0.5 + (y - 0.5) * 0.4 + rng.normal(0, 0.25, n), 0, 1)
    rep = CD.compute(CD.CueInputs(list(s1), list(s2), 0.5, 0.5, list(y)), high_margin_quantile=0.75)
    assert rep.n == n
    assert set(rep.auroc_margin_to_correct) == {"v1", "v2"}
    assert 0.0 <= rep.disagreement_rate <= 1.0
    assert set(rep.error_rate_given_agreement) == {"v1", "v2"}
    assert set(rep.high_margin_wrong_share) == {"v1", "v2"}
    assert len(rep.joint_margin_hist["counts"]) == 10
    assert rep.reliability["v1"] > rep.reliability["v2"]  # v1 is the cleaner one by construction
    assert 0.0 <= rep.p_at_least_one_wrong <= 1.0
    # AUROC(|m| -> correct) should be informative for a well-behaved scorer
    assert rep.auroc_margin_to_correct["v1"] > 0.5


# gate metric (d): R on the E-UNSAFE states (section 1) --------


def test_detection_vectors_keep_only_e_unsafe_states_in_split_order():
    y = [1, 0, 1, 1, 0]
    obs = {
        Lambda.LOW: ([0.9, 0.9, 0.2, 0.7, 0.1], [0.1, 0.9, 0.9, 0.9, 0.9]),
        Lambda.HIGH: ([0.1, 0.9, 0.1, 0.1, 0.1], [0.9, 0.1, 0.9, 0.1, 0.1]),
    }
    vec = G.detection_vectors(y, obs, 0.5, 0.5)
    assert vec["v1"][Lambda.LOW] == [1, 0, 1] and vec["v2"][Lambda.LOW] == [0, 1, 1]
    assert vec["v1"][Lambda.HIGH] == [0, 0, 0] and vec["v2"][Lambda.HIGH] == [1, 1, 0]
    r = G.reliability_by_level(vec)
    assert r["v1"][Lambda.LOW] == pytest.approx(2 / 3) and r["v2"][Lambda.HIGH] == pytest.approx(
        2 / 3
    )
    # the SAFE states (indices 1 and 4) never enter R, whatever their scores
    obs2 = {
        lam: ([a[0], 0.0, *a[2:4], 0.0], [b[0], 0.0, *b[2:4], 0.0]) for lam, (a, b) in obs.items()
    }
    assert G.detection_vectors(y, obs2, 0.5, 0.5) == vec
    desc = G.descriptive_accuracy(y, obs, 0.5, 0.5)
    assert desc["v1"]["low"]["n"] == 5
    assert desc["v1"]["low"]["accuracy"] == pytest.approx(3 / 5)  # 1,_,1 right; SAFE 0.9 wrong
    assert desc["v1"]["low"]["balanced_accuracy"] == pytest.approx((2 / 3 + 1 / 2) / 2)
    with pytest.raises(G.GateError, match="no E-UNSAFE"):
        G.detection_vectors([0, 0], {Lambda.LOW: ([0.1, 0.2], [0.1, 0.2])}, 0.5, 0.5)


def test_ladder_ci_on_the_subset_resamples_only_its_states():
    rng = np.random.default_rng(0)
    vec = {
        v: {
            lam: [int(x) for x in rng.random(30) < p]
            for lam, p in zip(G.LADDER, (0.9, 0.6, 0.3), strict=True)
        }
        for v in G.VERIFIER_KEYS
    }
    ci = G.ladder_ci(vec, n_boot=200, seed=3)
    assert ci["n_states"] == 30 and ci["seed"] == 3 and ci["n_boot"] == 200
    r = G.reliability_by_level(vec)
    res = G.ladder_check(r, 0.15, ci=ci, n_states=30, descriptive={"x": 1})
    assert res.population == "e_unsafe" and res.n_states == 30 and res.descriptive == {"x": 1}


def test_exact_resampling_floor_matches_the_sampler():
    """r_i = P_{s~q_i}(s >= tau_i) is computed exactly from the pool
    scores; it matches the channel's own inverse-CDF sampler."""
    from rafr.channel.margin_fit import fit_margin_model

    pool = list(np.random.default_rng(1).beta(0.5, 2.0, 400))
    for tau in (0.0, 0.05, 0.3, 0.5, 0.9, max(pool), max(pool) + 0.01):
        m = fit_margin_model(pool, tau, "v")
        exact = m.prob_at_least(tau)
        rng = np.random.default_rng(7)
        mc = float(np.mean([m.sample_score(rng) >= tau for _ in range(20000)]))
        assert abs(exact - mc) < 0.01, (tau, exact, mc)
    assert fit_margin_model([0.4], 0.4, "v").prob_at_least(0.4) == 1.0
    assert fit_margin_model([0.4], 0.4, "v").prob_at_least(0.41) == 0.0
    assert fit_margin_model([0.0, 1.0], 0.5, "v").prob_at_least(0.5) == pytest.approx(0.5)
