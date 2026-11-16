"""channel: q as independent resampling, per-verifier nested C_lambda (paper Sec. 4.3-paper Sec.
4.3;
group 6a/6b decisions, 2026-09-16)."""

from __future__ import annotations

import inspect
from pathlib import Path

import numpy as np
import pytest

from rafr.channel import Channel, ObservationBuilder
from rafr.channel.degrade import (
    FIXED_P,
    DegradeChannel,
    DegradeConfigError,
    LevelParams,
    check_monotone,
    expected_ladder,
    levels_from_config,
)
from rafr.channel.margin_fit import MarginModel, fit_margin_model, load_models, save_models
from rafr.contracts.schemas import Lambda


def test_margin_model_fit_sample_roundtrip(tmp_path: Path):
    rng = np.random.default_rng(1)
    scores = list(np.clip(rng.normal(0.5, 0.2, 500), 0, 1))
    m = fit_margin_model(scores, 0.5, "v1", n_bins=10)
    assert m.n == 500 and m.scores == sorted(m.scores)
    draws = [m.sample_score(rng) for _ in range(300)]
    assert min(draws) >= m.scores[0] and max(draws) <= m.scores[-1]
    assert abs(np.mean(draws) - np.mean(m.scores)) < 0.03
    # |m| implied by the resampled scores follows the pool's |m| distribution (margin-matched)
    assert abs(np.mean([abs(d - 0.5) for d in draws]) - np.mean(m.magnitudes)) < 0.03
    save_models({"v1": m}, tmp_path / "q.json", extra={"note": 1})
    back = load_models(tmp_path / "q.json")["v1"]
    assert back.scores == m.scores and back.tau == 0.5
    with pytest.raises(ValueError):
        fit_margin_model([], 0.5, "v")
    with pytest.raises(ValueError):  # old |m|-only files are refused
        MarginModel.from_dict({"verifier_id": "v", "tau": 0.5, "magnitudes": [0.1]})


def test_level_params_and_monotone_ladder():
    LevelParams(0.2, 0.7)
    with pytest.raises(DegradeConfigError):
        LevelParams(1.5, 0.0)
    with pytest.raises(DegradeConfigError):
        LevelParams.from_config({"v1": "TBD", "v2": 0.1})
    ok = {
        Lambda.LOW: LevelParams(0.1, 0.2),
        Lambda.MID: LevelParams(0.3, 0.2),
        Lambda.HIGH: LevelParams(0.6, 0.5),
    }
    check_monotone(ok)
    with pytest.raises(DegradeConfigError):  # p must be non-decreasing in the level
        check_monotone({**ok, Lambda.MID: LevelParams(0.05, 0.2)})
    with pytest.raises(DegradeConfigError):  # a missing level
        check_monotone({Lambda.LOW: LevelParams(0.1, 0.1)})
    # derived z modes under independence
    d = LevelParams(0.5, 0.2).derived_mode_probs()
    assert d == pytest.approx({"v1_only": 0.4, "v2_only": 0.1, "both": 0.1, "none": 0.4})


def _q():
    pool = [0.05, 0.15, 0.3, 0.45, 0.55, 0.7, 0.85, 0.95]
    return {"v1": MarginModel("v1", 0.5, pool), "v2": MarginModel("v2", 0.6, pool)}


def _channel(low=(0.0, 0.0), mid=(0.5, 0.5), high=(1.0, 1.0), seed=11):
    levels = {
        Lambda.LOW: LevelParams(*low),
        Lambda.MID: LevelParams(*mid),
        Lambda.HIGH: LevelParams(*high),
    }
    return DegradeChannel(levels, _q(), tau1=0.5, tau2=0.6, seed=seed)


def test_degrade_conforms_and_signature_has_no_arm_or_labels():
    ch = _channel()
    assert isinstance(ch, Channel)
    params = list(inspect.signature(DegradeChannel.__call__).parameters)[1:]
    assert params == ["state_id", "s1", "s2", "lam"]


def test_degrade_p_zero_and_lambda_none_are_identity():
    ch = _channel()
    assert ch("s", 0.8, 0.3, Lambda.LOW) == (0.8, 0.3)  # p = 0 at LOW
    assert ch("s", 0.8, 0.3, Lambda.NONE) == (0.8, 0.3)


def test_degrade_resamples_from_pool_without_forcing_a_flip():
    ch = _channel()
    pool = _q()["v1"].scores
    flips, keeps = 0, 0
    for i in range(200):
        t1, t2 = ch(f"s{i}", 0.8, 0.3, Lambda.HIGH)  # p = 1: both resampled
        assert pool[0] <= t1 <= pool[-1] and pool[0] <= t2 <= pool[-1]
        flips += int((t1 >= 0.5) != (0.8 >= 0.5))
        keeps += int((t1 >= 0.5) == (0.8 >= 0.5))
    assert flips > 0 and keeps > 0  # judgment may or may not flip


def test_degrade_nested_across_levels_and_monotone_enforced():
    ch = _channel(low=(0.2, 0.1), mid=(0.5, 0.4), high=(0.9, 0.8))
    sets = {lam: {"v1": set(), "v2": set()} for lam in (Lambda.LOW, Lambda.MID, Lambda.HIGH)}
    for i in range(300):
        sid = f"s{i}"
        for lam in sets:
            p = ch.perturbed(sid, lam)
            for v in ("v1", "v2"):
                if p[v]:
                    sets[lam][v].add(sid)
                    # a perturbed verifier shows the same replacement score at every level
    for v in ("v1", "v2"):
        assert sets[Lambda.LOW][v] <= sets[Lambda.MID][v] <= sets[Lambda.HIGH][v]
        assert len(sets[Lambda.LOW][v]) < len(sets[Lambda.MID][v]) < len(sets[Lambda.HIGH][v])
    # replacement value is shared across levels for a verifier perturbed at LOW
    for sid in list(sets[Lambda.LOW]["v1"])[:10]:
        lo = ch(sid, 0.8, 0.3, Lambda.LOW)[0]
        assert ch(sid, 0.8, 0.3, Lambda.MID)[0] == lo == ch(sid, 0.8, 0.3, Lambda.HIGH)[0]
    with pytest.raises(DegradeConfigError):
        _channel(low=(0.5, 0.1), mid=(0.2, 0.4))


def test_degrade_probabilities_are_respected_per_verifier():
    ch = _channel(mid=(0.5, 0.0))
    n1 = sum(ch.perturbed(f"s{i}", Lambda.MID)["v1"] for i in range(400))
    n2 = sum(ch.perturbed(f"s{i}", Lambda.MID)["v2"] for i in range(400))
    assert 150 < n1 < 250 and n2 == 0


def test_degrade_is_reproducible_in_state_and_lambda_and_shared_by_arms():
    ch = _channel(mid=(0.5, 0.5))
    a = ch("state-1", 0.8, 0.3, Lambda.HIGH)
    assert a == ch("state-1", 0.8, 0.3, Lambda.HIGH)
    assert DegradeChannel(ch.levels, _q(), 0.5, 0.6, seed=12)("state-1", 0.8, 0.3, Lambda.HIGH) != a
    # three arms of Exp 1 go through one ObservationBuilder and see the identical pair
    b = ObservationBuilder(ch, 0.5, 0.6, "v1", "v2")
    obs = [b.build("state-1", 0.8, 0.3, Lambda.MID) for _arm in ("A1", "B", "permuted_B")]
    assert len({(o.s1_obs, o.s2_obs) for o in obs}) == 1
    assert all(o.raw_s1 == 0.8 and o.raw_s2 == 0.3 for o in obs)


def test_degrade_save_load_roundtrip(tmp_path: Path):
    ch = _channel(low=(0.1, 0.2), mid=(0.3, 0.4), high=(0.6, 0.8))
    ch.save(tmp_path / "channel.json", extra={"status": "candidate"})
    back = DegradeChannel.load(tmp_path / "channel.json")
    for lam in (Lambda.LOW, Lambda.MID, Lambda.HIGH):
        assert back("s", 0.8, 0.3, lam) == ch("s", 0.8, 0.3, lam)
    assert back.seed == 11 and back.tau2 == 0.6
    d = ch.to_dict()
    assert d["geometry"] == "independent_resampling" and "derived_p_z" in d
    assert d["derived_p_z"]["high"]["both"] == pytest.approx(0.48)
    (tmp_path / "old.json").write_text('{"name": "degrade", "levels": {}, "q": {}}')
    with pytest.raises(DegradeConfigError):
        DegradeChannel.load(tmp_path / "old.json")


def test_assignment_reports_flags_derived_z_and_replacement_scores():
    """log fields: which verifier was perturbed, the derived z mode, the replacement
    score actually used. Identity (and any channel without the hook) perturbs nothing."""
    from rafr.channel import NO_PERTURBATION, channel_assignment
    from rafr.channel.degrade import z_mode
    from rafr.channel.identity import IdentityChannel

    ch = _channel(low=(0.0, 0.0), mid=(1.0, 0.0), high=(1.0, 1.0))
    lo = ch.assignment("s1", Lambda.LOW)
    assert lo["z"] == "none" and lo["replacement_s1"] is None and lo["replacement_s2"] is None
    mid = ch.assignment("s1", Lambda.MID)
    assert mid["perturbed_v1"] and not mid["perturbed_v2"] and mid["z"] == "v1_only"
    assert mid["replacement_s1"] == ch("s1", 0.8, 0.3, Lambda.MID)[0]
    assert mid["replacement_s2"] is None
    hi = ch.assignment("s1", Lambda.HIGH)
    assert hi["z"] == "both" and (hi["replacement_s1"], hi["replacement_s2"]) == ch(
        "s1", 0.8, 0.3, Lambda.HIGH
    )
    # the replacement score of a verifier perturbed at MID does not change at HIGH (nesting)
    assert hi["replacement_s1"] == mid["replacement_s1"]
    assert z_mode(False, True) == "v2_only" and z_mode(False, False) == "none"
    assert channel_assignment(IdentityChannel(), "s1", Lambda.HIGH) == NO_PERTURBATION
    assert channel_assignment(object(), "s1", Lambda.HIGH) == NO_PERTURBATION
    # the builder forwards it for the log; still no arm anywhere in the signature
    b = ObservationBuilder(ch, 0.5, 0.6, "v1", "v2")
    assert b.assignment("s1", Lambda.MID) == mid


# p fixed ex ante at 0 / 0.5 / 1 (paper Sec. 4.3, 2026-09-26) ----------------


def _degrade_cfg() -> dict:
    from rafr.core.config import load_config

    return load_config("channel")["degrade"]


def test_the_levels_are_fixed_at_0_05_1_for_both_verifiers():
    """paper Sec. 4.3: configs/channel.yaml carries p = 0 / 0.5 / 1, both verifiers alike; the
    solve rule (and its lambda_low anchor flag) is gone."""
    cfg = _degrade_cfg()
    assert "anchor_lambda_low_at_zero" not in cfg
    levels = levels_from_config(cfg)
    assert {lam: lp.to_dict() for lam, lp in levels.items()} == {
        Lambda.LOW: {"v1": 0.0, "v2": 0.0},
        Lambda.MID: {"v1": 0.5, "v2": 0.5},
        Lambda.HIGH: {"v1": 1.0, "v2": 1.0},
    }
    assert FIXED_P == {Lambda.LOW: 0.0, Lambda.MID: 0.5, Lambda.HIGH: 1.0}
    check_monotone(levels)  # monotone, as every level set must be
    assert levels[Lambda.LOW].derived_mode_probs()["none"] == 1.0
    assert levels[Lambda.MID].derived_mode_probs() == {
        "v1_only": 0.25, "v2_only": 0.25, "both": 0.25, "none": 0.25,
    }  # fmt: skip
    assert levels[Lambda.HIGH].derived_mode_probs()["both"] == 1.0


@pytest.mark.parametrize(
    ("tpr", "floor"),
    [
        ((0.6875, 0.69792), (0.332, 0.270)),  # step 7's D_fit TPR, step 2's pool r
        ((0.9, 0.85), (0.5, 0.45)),
        ((0.55, 0.52), (0.50, 0.49)),  # a span far below any delta_R
        ((1.0, 1.0), (0.0, 0.0)),
        ((0.3, 0.3), (0.3, 0.3)),  # no span at all
    ],
)
def test_p_does_not_move_with_tpr_or_r(tpr, floor):
    """p is fixed ex ante; no measurement enters it. levels_from_config takes the config
    only, and the expected ladder is TPR_i, (TPR_i + r_i)/2, r_i whatever TPR_i and r_i are."""
    assert list(inspect.signature(levels_from_config).parameters) == ["degrade_cfg"]
    levels = levels_from_config(_degrade_cfg())
    before = {lam: lp.to_dict() for lam, lp in levels.items()}
    t, r = dict(zip(("v1", "v2"), tpr, strict=True)), dict(zip(("v1", "v2"), floor, strict=True))
    ladder = expected_ladder(t, r, levels)
    assert {lam: lp.to_dict() for lam, lp in levels.items()} == before  # untouched
    assert levels_from_config(_degrade_cfg()) == levels
    for v in ("v1", "v2"):
        assert ladder[v]["low"] == pytest.approx(t[v])
        assert ladder[v]["mid"] == pytest.approx((t[v] + r[v]) / 2)
        assert ladder[v]["high"] == pytest.approx(r[v])
        # adjacent levels are (TPR - r)/2 apart in expectation (paper Sec. 4.3)
        assert ladder[v]["low"] - ladder[v]["mid"] == pytest.approx((t[v] - r[v]) / 2)
        assert ladder[v]["mid"] - ladder[v]["high"] == pytest.approx((t[v] - r[v]) / 2)


def _lv(low=(0.0, 0.0), mid=(0.5, 0.5), high=(1.0, 1.0), **extra) -> dict:
    out = {"low": low, "mid": mid, "high": high, **extra}
    return {k: {"v1": v[0], "v2": v[1]} for k, v in out.items() if v is not None}


@pytest.mark.parametrize(
    "levels",
    [
        _lv(mid=(0.4, 0.4)),
        _lv(mid=(0.5, 0.6)),  # the two verifiers apart
        _lv(high=(0.9, 0.9)),
        _lv(high=None),  # a level missing
        _lv(mid=("TBD", 0.5)),
        _lv(low=(0.5, 0.5), mid=(0.0, 0.0)),  # not monotone
        _lv(extra=(1.0, 1.0)),  # an unknown level
        None,
    ],
)
def test_levels_other_than_the_fixed_ones_are_refused(levels):
    """paper Sec. 4.3: anything but p = 0 / 0.5 / 1 for both verifiers -- another number, the two
    verifiers apart, a missing or unknown level, a TBD, a non-monotone set, no levels at all --
    is refused, never run."""
    cfg = {**_degrade_cfg(), "levels": levels}
    with pytest.raises(DegradeConfigError):
        levels_from_config(cfg)
