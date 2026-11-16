"""data: cleaning/dedupe, the v0.9 split, y* access matrix, reserve lock, manifest."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from rafr.contracts.schemas import SafetyLabel, Split
from rafr.data import labels as L
from rafr.data import paradetox
from rafr.data import splits as S
from rafr.data.manifest import Manifest

# ---------------------------------------------------------------- paradetox


def test_clean_normalizes_dedupes_and_drops_empty(paradetox_sample):
    res = paradetox.clean(paradetox_sample)
    assert res.stats == {"n_raw": 12, "n_empty": 1, "n_duplicates": 2, "n_kept": 9}
    texts = [r["text"] for r in res.x0s]
    assert "this garbage post is written by clowns" in texts  # internal whitespace collapsed
    assert len({r["x0_id"] for r in res.x0s}) == 9
    assert all(r["x0_id"].startswith("x0-") for r in res.x0s)
    # deterministic ids
    assert paradetox.clean(paradetox_sample).x0s == res.x0s


def test_load_raw_reads_fixture_dir(tmp_path: Path, paradetox_sample):
    p = tmp_path / "part.jsonl"
    with p.open("w", encoding="utf-8") as fh:
        for r in paradetox_sample:
            fh.write(json.dumps(r) + "\n")
    rows = paradetox.load_raw(tmp_path)
    assert len(rows) == 12
    with pytest.raises(KeyError):
        paradetox.load_raw(tmp_path, text_field="nope")


# ---------------------------------------------------------------- splits (v0.9, II.2.4)

SIZES = {"d_fit": 10, "d_gate": 10, "d_gate_reserve": 10, "d_exp1": 5, "unlabeled_pool": "rest"}


def _ids(n: int = 80) -> list[str]:
    return [f"x0-{i:03d}" for i in range(n)]


def _unsafe_set(ids: list[str], every: int = 3) -> set[str]:
    return {x for i, x in enumerate(ids) if i % every == 0}


def test_split_order_random_first_then_e_unsafe_only():
    """II.2.4 (v0.9): D_fit, D_gate, D_gate_reserve are drawn from all x0 without looking at
    a label; D_Exp1 only from the E-UNSAFE x0 that remain; the pool takes the rest."""
    ids = _ids()
    unsafe = _unsafe_set(ids)
    asked: list[str] = []

    def is_unsafe(x: str) -> bool:
        asked.append(x)
        return x in unsafe

    m = S.make_splits(ids, is_unsafe, SIZES, seed=11)
    first_three = {x for sp in S.RANDOM_SPLITS for x in m[sp]}
    assert [len(m[sp]) for sp in S.SPLIT_ORDER] == [10, 10, 10, 5, 45]
    S.assert_disjoint(m)
    assert set().union(*map(set, m.values())) == set(ids)
    # the label is consulted only for x0 left after the three random splits
    assert asked and not (set(asked) & first_three)
    assert set(asked) == set(ids) - first_three
    # D_Exp1 is E-UNSAFE only; every other remaining x0 went to the pool
    assert set(m[Split.D_EXP1]) <= unsafe
    assert set(m[Split.UNLABELED_POOL]) == set(ids) - first_three - set(m[Split.D_EXP1])
    # the three random splits do not depend on the labels at all
    flipped = S.make_splits(ids, lambda x: x not in unsafe, SIZES, seed=11)
    for sp in S.RANDOM_SPLITS:
        assert flipped[sp] == m[sp]
    # deterministic, independent of input order, and the seed is load-bearing
    assert S.make_splits(list(reversed(ids)), lambda x: x in unsafe, SIZES, seed=11) == m
    assert S.make_splits(ids, lambda x: x in unsafe, SIZES, seed=12) != m


def test_d_exp1_is_a_random_draw_from_the_remaining_e_unsafe():
    ids = _ids(200)
    unsafe = _unsafe_set(ids, every=2)
    draws = {
        tuple(S.make_splits(ids, lambda x: x in unsafe, SIZES, seed=s)[Split.D_EXP1])
        for s in range(5)
    }
    assert len(draws) == 5 and all(set(d) <= unsafe for d in draws)


def test_too_few_e_unsafe_stops_and_shrinks_nothing():
    """II.2.4: fewer than d_exp1 E-UNSAFE x0 remaining -> stop and report."""
    ids = _ids(40)  # 30 go to the random splits, 10 remain
    unsafe = set(ids[:3])  # at most 3 of the remaining can be UNSAFE
    with pytest.raises(S.InsufficientUnsafeError) as ei:
        S.make_splits(ids, lambda x: x in unsafe, SIZES, seed=1)
    assert ei.value.need == 5 and ei.value.have <= 3 and ei.value.n_remaining == 10


def test_make_splits_rejects_bad_sizes_and_duplicates():
    with pytest.raises(S.SplitError):  # the pool must be "rest"
        S.make_splits(_ids(), lambda x: True, {**SIZES, "unlabeled_pool": 10}, seed=0)
    with pytest.raises(S.SplitError):  # the three random splits cannot overflow the pool
        S.make_splits(_ids(20), lambda x: True, SIZES, seed=0)
    with pytest.raises(S.SplitError):
        S.make_splits(["a", "a"], lambda x: True, SIZES, seed=0)
    with pytest.raises(S.SplitError):
        S.make_splits(_ids(), lambda x: True, {**SIZES, "d_exp1": "rest"}, seed=0)


def test_split_plan_comes_from_the_config_and_is_pinned_in_the_manifest(tmp_path: Path):
    """II.2.4: seed and procedure are written into the config and the manifest before step 5
    judges any state, and may not change afterwards."""
    from rafr.core.config import load_config, master_seed
    from rafr.core.seeding import COMPONENTS, component_seed
    from rafr.data.manifest import SplitPlanChangedError

    cfgs = load_config()
    plan = S.split_plan(cfgs)
    assert plan["procedure"] == S.PROCEDURE == cfgs["data"]["splits"]["procedure"]
    assert plan["seed_component"] == "splits_v09" and "splits_v09" in COMPONENTS
    assert plan["seed"] == component_seed(master_seed(cfgs), "splits_v09")
    assert plan["seed"] != component_seed(master_seed(cfgs), "splits")  # not round 1's
    assert plan["sizes"] == {
        "d_fit": 600,
        "d_gate": 600,
        "d_gate_reserve": 600,
        "d_exp1": 300,
        "unlabeled_pool": "rest",
    }
    m = Manifest()
    m.pin_split_plan(plan)
    m.pin_split_plan(dict(plan))  # the same plan again is fine
    for key, value in (
        ("seed", 1),
        ("procedure", "other"),
        ("sizes", {**plan["sizes"], "d_fit": 1}),
    ):
        with pytest.raises(SplitPlanChangedError):
            m.pin_split_plan({**plan, key: value})
    bad = {**cfgs, "data": {**cfgs["data"], "splits": {**cfgs["data"]["splits"], "procedure": "x"}}}
    with pytest.raises(S.SplitError):
        S.split_plan(bad)


def test_assert_disjoint_catches_overlap():
    with pytest.raises(S.SplitError):
        S.assert_disjoint({Split.D_FIT: ["a"], Split.D_GATE: ["a"]})


def test_save_and_load_splits_roundtrip(tmp_path: Path):
    ids = _ids(60)
    m = S.make_splits(ids, lambda x: True, SIZES, seed=3)
    S.save_splits(m, tmp_path)
    back = S.load_all_splits(tmp_path)
    assert back == m
    assert S.split_of(m[Split.D_FIT][0], back) is Split.D_FIT


# ---------------------------------------------------------------- y* (v0.9: y* = E)


def _record(text: str, label: SafetyLabel):
    from rafr.contracts.schemas import EvalRecord

    return EvalRecord(
        text_id=text,
        text=text,
        label=label,
        model="claude-opus-5",
        model_version="claude-opus-5",
        prompt_hash="p",
        e_prompt_sha256="h",
        e_version="2026-07-24",
        decoding={},
    )


def _write_y_star(tmp_path: Path, splits: dict, *, skip: set[str] = frozenset()) -> tuple:
    """Two x0 per split, the first UNSAFE; ``skip`` leaves an x0's row out of the file."""
    labels_dir, split_dir = tmp_path / "labels", tmp_path / "splits"
    labels_dir.mkdir(exist_ok=True)
    S.save_splits(splits, split_dir)
    with L.y_star_path(labels_dir).open("w", encoding="utf-8") as fh:
        for ids in splits.values():
            for i, x0 in enumerate(ids):
                if x0 in skip:
                    continue
                lab = SafetyLabel.UNSAFE if i == 0 else SafetyLabel.SAFE
                row = L.StateJudgment(
                    state_id=f"{x0}-g1",
                    x0_id=x0,
                    label=lab,
                    record=_record(x0, lab),
                    e_identity={},
                )
                fh.write(row.model_dump_json() + "\n")
    return labels_dir, split_dir


FIVE = {sp: [f"{sp.value}-a", f"{sp.value}-b"] for sp in S.SPLIT_ORDER}

ALLOWED_READS = {
    (Split.D_FIT, L.LabelPurpose.TAU_FIT),
    (Split.D_FIT, L.LabelPurpose.CHANNEL_CHECK),
    (Split.D_FIT, L.LabelPurpose.BASELINE_DIAGNOSTICS),
    (Split.D_FIT, L.LabelPurpose.VERIFIER_SELECTION),
    (Split.D_FIT, L.LabelPurpose.VERIFIER_DIAGNOSTICS),  # step 6, II.4.2 (D-2026-09-25c)
    (Split.D_FIT, L.LabelPurpose.SPLIT_REPORT),
    (Split.D_GATE, L.LabelPurpose.GATE),
    (Split.D_EXP1, L.LabelPurpose.EXP1_OUTCOME),
    (Split.D_EXP1, L.LabelPurpose.EXP2_T1),
    (Split.UNLABELED_POOL, L.LabelPurpose.VERIFIER_SELECTION),
    (Split.UNLABELED_POOL, L.LabelPurpose.VERIFIER_DIAGNOSTICS),
    (Split.UNLABELED_POOL, L.LabelPurpose.DRY_RUN),
    # post-pilot extension (2026-10-31)
    (Split.UNLABELED_POOL, L.LabelPurpose.CALIBRATION_FIT),
    (Split.D_GATE, L.LabelPurpose.CALIBRATION_DIAG),
    (Split.D_GATE, L.LabelPurpose.ECORRECT_COHORT),
    (Split.D_GATE, L.LabelPurpose.EXT_SMOKE),  # stop point D's E-correct smoke
}


@pytest.mark.parametrize("split", [sp for sp in S.SPLIT_ORDER if sp is not Split.D_GATE_RESERVE])
@pytest.mark.parametrize("purpose", list(L.LabelPurpose))
def test_y_star_access_matrix(tmp_path: Path, split, purpose):
    """Coding instructions section 4 (v0.9): who may read which split's y*."""
    labels_dir, split_dir = _write_y_star(tmp_path, FIVE)
    if (split, purpose) in ALLOWED_READS:
        ys = L.read_labels(split, purpose, labels_dir=labels_dir, split_dir=split_dir)
        assert [y.x0_id for y in ys] == FIVE[split]
        assert [y.label for y in ys] == [SafetyLabel.UNSAFE, SafetyLabel.SAFE]
        assert all(y.record.e_prompt_sha256 == "h" for y in ys)  # the E record travels along
    else:
        with pytest.raises(L.LabelAccessError):
            L.read_labels(split, purpose, labels_dir=labels_dir, split_dir=split_dir)


def test_reserve_y_star_is_unreadable_until_the_first_gate_fails(tmp_path: Path):
    """II.6.1: D_gate_reserve's y* is not read, scored or fitted before a recorded failed
    first gate attempt -- then the gate, and only the gate, may read it."""
    labels_dir, split_dir = _write_y_star(tmp_path, FIVE)
    attempts = tmp_path / "gate_attempts.json"
    kw = dict(labels_dir=labels_dir, split_dir=split_dir, gate_attempts_path=attempts)
    for purpose in L.LabelPurpose:  # no file -> locked for every purpose
        with pytest.raises(L.LabelAccessError):
            L.read_labels(Split.D_GATE_RESERVE, purpose, **kw)
    for op in L.Operation:
        with pytest.raises(L.LabelAccessError):
            L.check_split_usable(Split.D_GATE_RESERVE, op, gate_attempts_path=attempts)
    attempts.write_text(json.dumps({"attempts": [{"attempt": 1, "passed": True}]}), "utf-8")
    with pytest.raises(L.LabelAccessError):  # a PASSED first attempt does not unlock it
        L.read_labels(Split.D_GATE_RESERVE, L.LabelPurpose.GATE, **kw)
    attempts.write_text(json.dumps({"attempts": [{"attempt": 1, "passed": False}]}), "utf-8")
    assert len(L.read_labels(Split.D_GATE_RESERVE, L.LabelPurpose.GATE, **kw)) == 2
    L.check_split_usable(Split.D_GATE_RESERVE, L.Operation.SCORE, gate_attempts_path=attempts)
    for purpose in set(L.LabelPurpose) - {L.LabelPurpose.GATE}:  # still only the gate
        with pytest.raises(L.LabelAccessError):
            L.read_labels(Split.D_GATE_RESERVE, purpose, **kw)


def test_the_split_report_reads_d_fit_only(tmp_path: Path):
    """(2026-09-23): the split report gives D_fit's E-UNSAFE count and share and every
    other split's size only; D_gate's y* is read by the gate alone, and with the segment-3 total
    nothing lets the reserve's count be recovered by subtraction (II.6.1). The report works with
    every other split's judgments absent from the file -- it does not open them."""
    others = {x for sp in S.SPLIT_ORDER if sp is not Split.D_FIT for x in FIVE[sp]}
    labels_dir, split_dir = _write_y_star(tmp_path, FIVE, skip=others)
    rep = L.split_report(FIVE, labels_dir=labels_dir, split_dir=split_dir)
    assert rep["d_fit"] == {"n": 2, "n_unsafe": 1, "unsafe_rate": 0.5}
    for name in ("d_gate", "d_gate_reserve", "d_exp1", "unlabeled_pool"):
        assert rep[name] == {"n": 2}
    assert L.SPLIT_REPORT_COUNTS == {Split.D_FIT}
    for sp in S.SPLIT_ORDER:
        allowed = L.LabelPurpose.SPLIT_REPORT in L.ALLOWED[sp]
        assert allowed == (sp is Split.D_FIT)


def test_a_split_x0_without_y_star_is_an_error(tmp_path: Path):
    labels_dir, split_dir = _write_y_star(tmp_path, FIVE, skip={FIVE[Split.D_FIT][1]})
    with pytest.raises(L.LabelDataError):
        L.read_labels(
            Split.D_FIT, L.LabelPurpose.TAU_FIT, labels_dir=labels_dir, split_dir=split_dir
        )


def test_d_exp1_and_pool_are_never_annotated():
    for sp in (Split.D_EXP1, Split.UNLABELED_POOL):
        with pytest.raises(L.LabelAccessError):
            L.check_split_usable(sp, L.Operation.ANNOTATE)


# ---------------------------------------------------------------- manifest


def test_manifest_roundtrip(tmp_path: Path):
    m = Manifest()
    m.update_section("splits", {"seed": 7})
    p = m.save(tmp_path / "manifest.json")
    back = Manifest.load(p)
    assert back.splits == {"seed": 7}
    with pytest.raises(KeyError):
        m.update_section("created_at", {})
