"""analysis/paper_appendix_ab.py on synthetic inputs (no real run, no data/ file is read)."""

from __future__ import annotations

import json
import math

import numpy as np
import pytest
from analysis import paper_appendix_ab as P

from rafr.validity import cue_diagnostics as CD


def test_constants_are_the_usual_80_percent_power_multipliers():
    assert P.K_TWO_SIDED == pytest.approx(1.959964 + 0.841621, abs=1e-5)
    assert P.K_ONE_SIDED == pytest.approx(1.644854 + 0.841621, abs=1e-5)


def test_paired_stats_by_hand_and_nan_dropped():
    d = [1.0, 0.0, 0.0, -1.0, 1.0, 0.0, float("nan")]
    s = P.paired_stats(d)
    v = np.array(d[:-1])
    sd = math.sqrt(((v - v.mean()) ** 2).sum() / (len(v) - 1))
    assert s["n"] == 6
    assert s["estimate"] == pytest.approx(1 / 6)
    assert s["sd_d"] == pytest.approx(sd)
    assert s["se"] == pytest.approx(sd / math.sqrt(6))
    assert s["mde80_two_sided"] == pytest.approx(P.K_TWO_SIDED * sd / math.sqrt(6))
    assert s["mde80_one_sided"] == pytest.approx(P.K_ONE_SIDED * sd / math.sqrt(6))
    assert s["share_nonzero"] == pytest.approx(0.5)
    with pytest.raises(P.AppendixError):
        P.paired_stats([1.0, float("nan")])


def test_ba_contrast_halves_the_sum_and_combines_independent_ses():
    ba = P.ba_contrast({"estimate": 0.04, "se": 0.03}, {"estimate": -0.02, "se": 0.04})
    assert ba["estimate"] == pytest.approx(0.01)
    assert ba["se"] == pytest.approx(0.025)
    assert ba["mde80_two_sided"] == pytest.approx(P.K_TWO_SIDED * 0.025)


def test_unpaired_stats_two_independent_samples():
    a, b = [1, 1, 0, 0], [1, 0, 0, 0, 0]
    s = P.unpaired_stats(a, b)
    se = math.sqrt(np.var(a, ddof=1) / 4 + np.var(b, ddof=1) / 5)
    assert (s["n_x"], s["n_y"]) == (4, 5)
    assert s["estimate"] == pytest.approx(0.5 - 0.2)
    assert s["se"] == pytest.approx(se)
    assert s["mde80_two_sided"] == pytest.approx(P.K_TWO_SIDED * se)


@pytest.mark.parametrize(("delta", "n", "power"), [(0.11, 300, 0.68), (0.06, 600, 0.45)])
def test_implied_sigma_inverts_the_one_sided_power(delta, n, power):
    sigma = P.implied_sigma(delta, n, power)
    assert P.power_one_sided(delta, n, sigma) == pytest.approx(power)


def test_prereg_table_row_011_implies_sigma_about_090():
    implied = [P.implied_sigma(0.11, n, pw) for n, pw in P.PREREG_TABLE[0.11].items()]
    assert min(implied) > 0.89
    assert max(implied) < 0.92


@pytest.mark.parametrize(("n", "k"), [(96, 24), (121, 31), (205, 52), (209, 53), (600, 150)])
def test_nominal_high_count_matches_numpy_on_distinct_values(n, k):
    assert P.nominal_high_count(n, 0.75) == k
    rng = np.random.default_rng(n)
    v = rng.permutation(np.linspace(0.01, 1, n)) * rng.choice([-1.0, 1.0], n)  # distinct |v|
    assert int(np.sum(np.abs(v) >= np.quantile(np.abs(v), 0.75))) == k


def _cue(y_true, s1, s2, tau=(0.5, 0.5)):
    inp = CD.CueInputs(list(s1), list(s2), tau[0], tau[1], list(y_true))
    return inp, CD.compute(inp, high_margin_quantile=0.75)


def test_cue_denominators_equal_direct_counts():
    rng = np.random.default_rng(3)
    n = 97
    y = rng.integers(0, 2, n)
    s1, s2 = rng.uniform(0, 1, n), rng.uniform(0, 1, n)  # distinct scores: no ties
    inp, rep = _cue(y, s1, s2)
    d = P.cue_denominators(rep.__dict__, n)
    y1, y2 = inp.judgments()
    c1, c2 = inp.correctness()
    m1, _ = inp.margins()
    assert d["n_disagree"] == int(np.sum(y1 != y2))
    assert d["n_agree"] == int(np.sum(y1 == y2))
    assert (d["n_correct_v1"], d["n_wrong_v1"]) == (int(c1.sum()), int(n - c1.sum()))
    assert (d["n_correct_v2"], d["n_wrong_v2"]) == (int(c2.sum()), int(n - c2.sum()))
    assert d["n_high_nominal"] == int(np.sum(np.abs(m1) >= np.quantile(np.abs(m1), 0.75)))


def test_auroc_is_defined_on_a_failure_only_subset():
    """y* has one class there, but the AUROC's class is the verifier's correctness: detected
    (m >= 0) against missed (m < 0). Only an all-detected or all-missed verifier gives nan."""
    y = [1] * 6
    _, rep = _cue(y, [0.9, 0.7, 0.55, 0.45, 0.2, 0.05], [0.9, 0.8, 0.7, 0.6, 0.4, 0.1])
    assert 0.0 <= rep.auroc_margin_to_correct["v1"] <= 1.0
    _, rep = _cue(y, [0.9, 0.8, 0.7, 0.6, 0.55, 0.51], [0.9, 0.8, 0.7, 0.6, 0.4, 0.1])
    assert math.isnan(rep.auroc_margin_to_correct["v1"])
    assert not math.isnan(rep.auroc_margin_to_correct["v2"])


def test_count_refuses_a_rate_that_is_not_a_count():
    with pytest.raises(P.AppendixError):
        P._count(0.3333, 10, "disagreement")
    assert P._count(0.3, 10, "disagreement") == 3


def test_recorder_checks_are_json_plain_and_flag_mismatches(tmp_path):
    rec = P.Recorder(tmp_path)
    rec.check("equal", np.int64(5), 5)
    rec.check("differs", 0.1, 0.2)
    json.dumps(rec.checks)  # numpy scalars are stored as plain floats
    assert [c["ok"] for c in rec.checks] == [True, False]
    with pytest.raises(P.AppendixError):
        rec.src("missing.json")


def test_table_escapes_pipes_inside_cells():
    lines = P._table(("a", "b"), [["|m_i|", 1]])
    assert lines[2] == "| \\|m_i\\| | 1 |"
