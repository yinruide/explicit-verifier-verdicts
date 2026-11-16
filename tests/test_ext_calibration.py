"""pi-hat calibration (post-pilot, 2026-10-31): logit-scale Platt with Platt's smoothed
targets and no regularization, eps = 1e-7 clipping with per-cell counts (stop above 1%),
equal-frequency ECE, bootstrap intervals, the frozen file with its id list and sha256, the pool's
disjointness from every other split, and the label gate of the three new purposes."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from rafr.calibration import fit as F
from rafr.calibration import platt
from rafr.calibration.apply import Calibrator, CalibratorError, ids_sha256, write_frozen
from rafr.contracts.schemas import EvalRecord, SafetyLabel, Split
from rafr.data import labels as label_gate
from rafr.experiments.ext_cohort import CohortError, draw_cohort, read_cohort, write_cohort

EPS = 1e-7


def _synthetic(n: int = 4000, a: float = 1.3, b: float = -0.7, seed: int = 3):
    r = np.random.default_rng(seed)
    z = r.normal(0.0, 2.0, n)
    s = 1.0 / (1.0 + np.exp(-z))
    y = (r.random(n) < 1.0 / (1.0 + np.exp(-(a * z + b)))).astype(int)
    return s, y


def test_platt_targets_are_platts_smoothed_values():
    t, tp, tn = platt.platt_targets(np.array([1, 1, 0, 0, 0]))
    assert tp == 3 / 4 and tn == 1 / 5
    assert list(t) == [0.75, 0.75, 0.2, 0.2, 0.2]


def test_fit_recovers_the_generating_parameters_and_is_a_stationary_point():
    s, y = _synthetic()
    f = platt.fit_platt(s, y, EPS)
    assert f.converged and abs(f.a - 1.3) < 0.15 and abs(f.b + 0.7) < 0.15
    # the gradient of the smoothed-target cross-entropy vanishes at (a, b): no regularizer
    z, _ = platt.clip_logit(s, EPS)
    t, _, _ = platt.platt_targets(y)
    p = platt.sigmoid(f.a * z + f.b)
    assert abs(np.sum((p - t) * z)) < 1e-6 * len(s) and abs(np.sum(p - t)) < 1e-6 * len(s)
    assert f.n_positive + f.n_negative == f.n == len(s)


def test_fit_is_deterministic_and_refuses_one_class():
    s, y = _synthetic(500)
    assert platt.fit_platt(s, y, EPS) == platt.fit_platt(s, y, EPS)
    with pytest.raises(platt.PlattFitError):
        platt.fit_platt(s, np.zeros_like(y), EPS)


def test_clip_counts_and_the_one_percent_stop():
    s = np.array([0.0, 1e-9, 0.5, 1.0, 0.3])
    _, clipped = platt.clip_logit(s, EPS)
    assert list(clipped) == [True, True, False, True, False]
    ok = {"pool": {"v1": [0.2] * 200 + [0.0], "v2": [0.4] * 201}}
    rep = F.clip_report(ok, EPS, 0.01)
    assert rep["cells"]["pool|v1"]["clipped"] == 1 and rep["cells"]["pool|v1"]["clipped_low"] == 1
    bad = {"pool": {"v1": [0.2] * 98 + [0.0, 1.0], "v2": [0.4] * 100}}
    with pytest.raises(F.ClipShareError) as exc:
        F.clip_report(bad, EPS, 0.01)
    assert exc.value.cells == ["pool|v1 (2/100)"] and "pool|v2" in exc.value.report["cells"]


def test_pihat_is_the_chance_the_verdict_is_wrong():
    g = np.array([0.9, 0.9, 0.2, 0.2])
    flags = np.array([1, 0, 1, 0])
    assert list(platt.pihat_from_g(g, flags)) == pytest.approx([0.1, 0.9, 0.8, 0.2])


def test_equal_frequency_bins_and_ece():
    assert platt.bin_sizes(23, 10) == [3, 3, 3, 2, 2, 2, 2, 2, 2, 2]
    p = np.linspace(0.05, 0.95, 100)
    y = (np.arange(100) % 2).astype(int)
    member = platt.bin_membership(p, 10)
    assert np.bincount(member).tolist() == [10] * 10
    ece = platt.ece_equal_frequency(p, y, 10)
    by_hand = sum(abs(p[member == j].mean() - y[member == j].mean()) / 10 for j in range(10))
    assert ece == pytest.approx(by_hand)
    # ties keep the caller's order: bins {0..4} (rate 0.2) and {5..9} (rate 0)
    tied = platt.ece_equal_frequency(np.full(10, 0.3), np.array([1] + [0] * 9), 2)
    assert tied == pytest.approx(0.5 * 0.1 + 0.5 * 0.3)


def test_diagnostics_have_intervals_and_are_seeded():
    s, y = _synthetic(600)
    d1 = platt.diagnostics(s, y, n_bins=10, n_boot=300, seed=11, alpha=0.05)
    d2 = platt.diagnostics(s, y, n_bins=10, n_boot=300, seed=11, alpha=0.05)
    assert d1 == d2
    assert d1["n"] == 600 and d1["n_positive"] == int(y.sum())
    assert d1["ece_ci"][0] <= d1["ece_ci"][1] and d1["brier_ci"][0] <= d1["brier"] <= 1
    assert len(d1["reliability"]) == 10 and sum(b["n"] for b in d1["reliability"]) == 600
    assert platt.diagnostics(s, y, n_bins=10, n_boot=300, seed=12, alpha=0.05) != d1


def _record(ids, s, y, vid=("stub-v1", "stub-v2")):
    return F.fit_record(
        domain="detox",
        ids=ids,
        failures=dict(zip(ids, y.tolist(), strict=True)),
        scores={"v1": dict(zip(ids, s, strict=True)), "v2": dict(zip(ids, s[::-1], strict=True))},
        verifier_ids={"v1": vid[0], "v2": vid[1]},
        eps=EPS,
        provenance={"test": True},
    )


def test_frozen_file_round_trip_sha256_and_no_refit(tmp_path: Path):
    s, y = _synthetic(800)
    ids = [f"x0-{i}-g1" for i in range(800)]
    rec = _record(ids, s, y)
    assert rec["fit_data"]["state_ids"] == ids
    assert rec["fit_data"]["state_ids_sha256"] == ids_sha256(ids)
    path = tmp_path / "calibration_detox.json"
    sha = write_frozen(path, rec)
    with pytest.raises(CalibratorError):
        write_frozen(path, rec)  # fitted once
    cal = Calibrator.load(path, expected_sha256=sha)
    with pytest.raises(CalibratorError):
        Calibrator.load(path, expected_sha256="0" * 64)
    a, b = cal.params("v1")
    assert cal.g("v1", 0.3) == pytest.approx(1 / (1 + np.exp(-(a * np.log(0.3 / 0.7) + b))))
    d = cal.detail(0.3, 0.0, 0.25, 0.5)  # V1 flags (0.3 >= 0.25), V2 does not; s2 is clipped
    assert d.pihat1 == pytest.approx(1 - d.g1) and d.pihat2 == pytest.approx(d.g2)
    assert d.clipped2 and not d.clipped1
    cal.check_verifiers("stub-v1", "stub-v2")
    with pytest.raises(CalibratorError):
        cal.check_verifiers("stub-v2", "stub-v1")


def _splits(tmp_path: Path, overlap: bool = False, missing: Split | None = None) -> Path:
    d = tmp_path / "splits"
    d.mkdir(parents=True)
    for k, sp in enumerate(Split):
        if sp is missing:
            continue
        ids = [f"x0-{sp.value}-{i}" for i in range(3)]
        if overlap and sp is Split.D_GATE:
            ids.append(f"x0-{Split.UNLABELED_POOL.value}-0")
        (d / f"{sp.value}.jsonl").write_text(
            "".join(json.dumps({"x0_id": x, "k": k}) + "\n" for x in ids), encoding="utf-8"
        )
    return d


def test_pool_must_be_disjoint_from_every_other_split(tmp_path: Path):
    ok = F.assert_pool_disjoint(_splits(tmp_path / "a"))
    assert ok["disjoint_by_x0"] and ok["sizes"]["unlabeled_pool"] == 3
    with pytest.raises(F.CalibrationLeakError, match="d_gate"):
        F.assert_pool_disjoint(_splits(tmp_path / "b", overlap=True))
    with pytest.raises(F.CalibrationLeakError, match="d_exp1"):
        F.assert_pool_disjoint(_splits(tmp_path / "c", missing=Split.D_EXP1))


def test_prior_shift_moves_the_intercept_by_the_logit_difference():
    """2026-10-31: delta = logit(p_R) - logit(p_pool), p_R adds the |D_Exp1| failures."""
    sh = platt.prior_shift(3784, 432, 300)
    assert sh["p_pool"] == 432 / 3784 and sh["p_R"] == 732 / 4084
    logit = lambda p: np.log(p / (1 - p))  # noqa: E731
    assert sh["delta"] == pytest.approx(logit(732 / 4084) - logit(432 / 3784), abs=1e-12)
    # a constant model at the pool's rate moves exactly to p_R
    b = float(logit(sh["p_pool"]))
    assert platt.sigmoid(np.array([b + sh["delta"]]))[0] == pytest.approx(sh["p_R"], abs=1e-12)
    assert platt.prior_shift(100, 10, 0)["delta"] == 0.0
    for bad in ((100, 0, 5), (100, 100, 5), (100, 10, -1)):
        with pytest.raises(ValueError):
            platt.prior_shift(*bad)


def test_fit_record_applies_the_prior_correction_to_b_only():
    s, y = _synthetic(800)
    ids = [f"x0-{i}-g1" for i in range(800)]
    plain = _record(ids, s, y)
    sh = platt.prior_shift(800, int(y.sum()), 50)
    rec = F.fit_record(
        domain="detox", ids=ids, failures=dict(zip(ids, y.tolist(), strict=True)),
        scores={"v1": dict(zip(ids, s, strict=True)), "v2": dict(zip(ids, s[::-1], strict=True))},
        verifier_ids={"v1": "stub-v1", "v2": "stub-v2"}, eps=EPS, provenance={},
        prior_correction=sh,
    )  # fmt: skip
    assert rec["prior_correction"]["delta"] == sh["delta"]
    for k in ("v1", "v2"):
        v, p = rec["verifiers"][k], plain["verifiers"][k]
        assert v["a"] == p["a"] and v["b_fit"] == p["b"] and v["b"] == p["b"] + sh["delta"]
    with pytest.raises(ValueError, match="counts"):
        F.fit_record(
            domain="detox", ids=ids, failures=dict(zip(ids, y.tolist(), strict=True)),
            scores={"v1": dict(zip(ids, s, strict=True)), "v2": dict(zip(ids, s, strict=True))},
            verifier_ids={"v1": "a", "v2": "b"}, eps=EPS, provenance={},
            prior_correction=platt.prior_shift(799, int(y.sum()), 50),
        )  # fmt: skip


def _drawn(n: int = 400, seed: int = 99):
    """A split drawn by the real procedure, with its inputs."""
    from rafr.data.splits import PROCEDURE, make_splits

    all_x0 = [f"x0-{i:04d}" for i in range(n)]
    r = np.random.default_rng(seed)
    unsafe = {x for x in all_x0 if r.random() < 0.3}
    sizes = {"d_fit": 40, "d_gate": 40, "d_gate_reserve": 40, "d_exp1": 30}
    sizes["unlabeled_pool"] = "rest"
    splits = make_splits(all_x0, lambda x: x in unsafe, sizes, seed)
    plan = {"procedure": PROCEDURE, "seed": seed, "sizes": sizes}
    pool_fail = [x for x in splits[Split.UNLABELED_POOL] if x in unsafe]
    return all_x0, splits, plan, pool_fail


def test_the_prior_premises_hold_for_the_procedures_draw_and_nothing_else():
    all_x0, splits, plan, pool_fail = _drawn()
    kw = dict(plan=plan, expected_seed=99, pool_failure_x0=pool_fail)
    ok = F.check_prior_premises(all_x0=all_x0, splits=splits, **kw)
    assert ok["n_all"] == 400 and ok["sizes"]["d_exp1"] == 30
    assert ok["n_pool_failures"] == len(pool_fail)

    def refused(match: str, **over):
        args = dict(all_x0=all_x0, splits=splits, **kw)
        args.update(over)
        with pytest.raises(F.PriorPremiseError, match=match):
            F.check_prior_premises(**args)

    # 1) the complement: a state that is in no split, a split x0 not in the state list
    refused("is not all x0", all_x0=[*all_x0, "x0-extra"])
    refused("not in the state list", all_x0=[x for x in all_x0 if x != splits[Split.D_FIT][0]])
    # 2) the seeded draw: another seed, another procedure
    refused("expected", plan={**plan, "seed": 98})
    refused("procedure", plan={**plan, "procedure": "something_else"})
    # D_Exp1 swapped with a later pool failure: no longer the first |D_Exp1| failures
    pool, exp1 = list(splits[Split.UNLABELED_POOL]), list(splits[Split.D_EXP1])
    swapped = dict(splits)
    swapped[Split.D_EXP1] = [*exp1[:-1], pool_fail[0]]
    swapped[Split.UNLABELED_POOL] = [exp1[-1] if x == pool_fail[0] else x for x in pool]
    refused("not what the procedure draws", splits=swapped, pool_failure_x0=pool_fail[1:])
    # a pool state shuffled before D_Exp1's last that would have been a failure: D_Exp1 would
    # have taken it, so the recorded D_Exp1 is not the procedure's draw
    from rafr.core.seeding import py_rng

    order = sorted(all_x0)
    py_rng(99, "shuffle").shuffle(order)
    last = max(order.index(x) for x in exp1)
    early = next(x for x in order[:last] if x in set(pool) and x not in set(pool_fail))
    refused("not what the procedure draws", pool_failure_x0=[*pool_fail, early])
    refused("not in the pool", pool_failure_x0=[*pool_fail, exp1[0]])


def _y_star(tmp_path: Path) -> tuple[Path, Path]:
    """Two states per split, the first UNSAFE, through the y* file the gate reads."""
    labels_dir, split_dir = tmp_path / "labels", tmp_path / "splits"
    labels_dir.mkdir()
    split_dir.mkdir()
    rows = []
    for sp in Split:
        xs = [f"x0-{sp.value}-{i}" for i in range(2)]
        (split_dir / f"{sp.value}.jsonl").write_text(
            "".join(json.dumps({"x0_id": x}) + "\n" for x in xs), encoding="utf-8"
        )
        for i, x in enumerate(xs):
            lab = SafetyLabel.UNSAFE if i == 0 else SafetyLabel.SAFE
            rec = EvalRecord(
                text_id="h", text="t", label=lab, model="e", model_version=None,
                prompt_hash="p", e_prompt_sha256="h", e_version="v", decoding={},
            )  # fmt: skip
            rows.append(
                {
                    "state_id": f"{x}-g1",
                    "x0_id": x,
                    "label": lab.value,
                    "record": rec.model_dump(mode="json"),
                    "e_identity": {},
                }  # fmt: skip
            )
    (labels_dir / label_gate.Y_STAR_FILENAME).write_text(
        "".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8"
    )
    return labels_dir, split_dir


def test_smoke_states_come_from_outside_the_formal_sets(tmp_path: Path):
    """Stop point D (2026-10-31): E-incorrect smoke = the pool's E-UNSAFE (DRY_RUN),
    E-correct smoke = D_gate's SAFE minus the cohort (EXT_SMOKE); drawn without replacement."""
    from pipelines import _ext_smoke as X

    from rafr.experiments.ext_cohort import draw_smoke

    labels_dir, split_dir = _y_star(tmp_path)
    kw = dict(labels_dir=labels_dir, split_dir=split_dir)
    ids, recs = X.smoke_candidates("e_incorrect", **kw)
    assert ids == ["x0-unlabeled_pool-0-g1"] and recs[ids[0]].label is SafetyLabel.UNSAFE
    ids, _ = X.smoke_candidates("e_correct", **kw)
    assert ids == ["x0-d_gate-1-g1"]
    assert X.smoke_candidates("e_correct", exclude={"x0-d_gate-1-g1"}, **kw)[0] == []
    assert X.SOURCES["e_correct"].purpose is label_gate.LabelPurpose.EXT_SMOKE
    pool = [f"s{i}" for i in range(50)]
    a = draw_smoke(pool, 10, 5, "e_incorrect")
    assert a == draw_smoke(list(reversed(pool)), 10, 5, "e_incorrect") and len(set(a)) == 10
    assert a != draw_smoke(pool, 10, 5, "e_correct")
    for bad in (0, 51):
        with pytest.raises(CohortError):
            draw_smoke(pool, bad, 5, "e_incorrect")


def test_the_label_gate_of_the_three_new_purposes(tmp_path: Path):
    labels_dir, split_dir = _y_star(tmp_path)
    kw = dict(labels_dir=labels_dir, split_dir=split_dir)
    fit = F.task_failures(
        Split.UNLABELED_POOL, label_gate.LabelPurpose.CALIBRATION_FIT, SafetyLabel.UNSAFE, **kw
    )
    assert fit == {"x0-unlabeled_pool-0-g1": 1, "x0-unlabeled_pool-1-g1": 0}
    diag = F.task_failures(
        Split.D_GATE, label_gate.LabelPurpose.CALIBRATION_DIAG, SafetyLabel.UNSAFE, **kw
    )
    assert len(diag) == 2
    # the fit purpose opens the pool only; the diagnostics and cohort purposes D_gate only
    purposes = label_gate.LabelPurpose
    for split, purpose in (
        (Split.D_GATE, purposes.CALIBRATION_FIT),
        (Split.D_EXP1, purposes.CALIBRATION_FIT),
        (Split.D_FIT, purposes.CALIBRATION_DIAG),
        (Split.UNLABELED_POOL, purposes.CALIBRATION_DIAG),
        (Split.D_EXP1, purposes.ECORRECT_COHORT),
        (Split.D_GATE_RESERVE, purposes.ECORRECT_COHORT),
        # stop point D's E-correct smoke opens D_gate only
        (Split.D_EXP1, purposes.EXT_SMOKE),
        (Split.UNLABELED_POOL, purposes.EXT_SMOKE),
        (Split.D_FIT, purposes.EXT_SMOKE),
        (Split.D_GATE_RESERVE, purposes.EXT_SMOKE),
    ):
        with pytest.raises(label_gate.LabelAccessError):
            label_gate.read_labels(split, purpose, **kw)


def test_ecorrect_cohort_without_replacement_or_all(tmp_path: Path):
    ids = [f"s{i:03d}" for i in range(500)]
    a = draw_cohort(ids, 300, seed=5)
    assert len(a) == len(set(a)) == 300 and set(a) <= set(ids) and a == sorted(a)
    assert draw_cohort(list(reversed(ids)), 300, seed=5) == a  # order of candidates is irrelevant
    assert draw_cohort(ids, 300, seed=6) != a
    assert draw_cohort(ids[:120], 300, seed=5) == ids[:120]  # fewer than n: all of them
    with pytest.raises(CohortError):
        draw_cohort(["a", "a"], 1, seed=1)
    path = tmp_path / "cohort.jsonl"
    sha = write_cohort(path, a, {"seed": 5})
    with pytest.raises(CohortError):
        write_cohort(path, a, {"seed": 5})
    header, back = read_cohort(path, sha)
    assert back == a and header["n"] == 300 and header["seed"] == 5
    with pytest.raises(CohortError):
        read_cohort(path, "0" * 64)
