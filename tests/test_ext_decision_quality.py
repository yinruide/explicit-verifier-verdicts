"""Post hoc decision-quality analysis (2026-11-01; analysis/ext_decision_quality.py):
hand-computed cells, the bootstrap's pairing and cohort independence, the input checks, and the
whole entry on shuffled fake data; the cells from the A1 rows' flags and the domain's seed
(the authors 2026-11-01). Only the detox regression reads runs/ (skipped where they are not);
nothing reads data/."""

from __future__ import annotations

import json
import math
import random
import shutil
from pathlib import Path

import numpy as np
import pytest
from analysis import ext_decision_quality as Q

from rafr.core import paths
from rafr.core.config import load_config, master_seed
from rafr.core.seeding import COMPONENTS, component_seed
from rafr.experiments import ext_config as C
from rafr.logging.writer import read_jsonl

LEVELS = {
    "low": {
        "p": {"v1": 0.0, "v2": 0.0},
        "q": "independent_resampling",
        "flags_independent_of_text": False,
    },
}


def _row(cohort: str, sid: str, arm: str, y: str, acc: int, lam: str = "low") -> dict:
    return {
        "run_id": f"FAKE-{cohort}",
        "state_id": sid,
        "x0_id": sid.removesuffix("-g1"),
        "arm": arm,
        "lam": lam,
        "fields": {"y1": int(y[0]), "y2": int(y[1])},
        "parse_fail": False,
        "action": "accept" if acc else "rewrite",
    }


def _small(extra_arms: dict[str, tuple[str, str]] | None = None) -> dict[str, Q.Cohort]:
    """Four states per cohort, one per (y1, y2) pattern in e_incorrect; A1 and No Verifier."""
    inc_flags = ("00", "01", "10", "11")
    cor_flags = ("00", "00", "11", "01")
    accepts = {
        "A1": ("1100", "1101"),
        "no_verifier": ("1001", "1110"),
        **(extra_arms or {}),
    }
    rows: dict[str, list[dict]] = {Q.E_INCORRECT: [], Q.E_CORRECT: []}
    for arm, (inc, cor) in accepts.items():
        for i, (y, a) in enumerate(zip(inc_flags, inc, strict=True)):
            rows[Q.E_INCORRECT].append(_row("i", f"x0-i{i}-g1", arm, y, int(a)))
        for i, (y, a) in enumerate(zip(cor_flags, cor, strict=True)):
            rows[Q.E_CORRECT].append(_row("c", f"x0-c{i}-g1", arm, y, int(a)))
    return {c: Q.build_cohort(c, r) for c, r in rows.items()}


def _est(block: dict) -> float:
    return block["estimate"]


def test_the_seed_component_and_the_seven_pairs_are_the_confirmed_ones():
    assert Q.SEED_COMPONENT in COMPONENTS
    a = C.load_ext()["analysis"]
    confirmed = [tuple(c) for c in [*a["contrasts"], *a["extra_contrasts"]["exp1"]["e_correct"]]]
    assert list(Q.BA_PAIRS) == confirmed


def test_the_rules_on_the_four_cells():
    y1, y2 = np.array([0, 0, 1, 1]), np.array([0, 1, 0, 1])
    got = {r: f(y1, y2).astype(int).tolist() for r, f in Q.RULES.items()}
    assert got == {
        "V1_only": [1, 1, 0, 0],
        "V2_only": [1, 0, 1, 0],
        "OR": [1, 0, 0, 0],
        "AND": [1, 1, 1, 0],
    }


def test_dominance_labels():
    assert Q.dominance(-0.1, -0.2) == "dominates"
    assert Q.dominance(0.1, 0.2) == "dominated"
    assert Q.dominance(-0.1, 0.2) == Q.dominance(0.1, -0.2) == "moved"
    assert Q.dominance(0.0, -0.1) == "weakly dominates"
    assert Q.dominance(0.1, 0.0) == "weakly dominated"
    assert Q.dominance(0.0, 0.0) == "equal"
    assert Q.dominance(math.nan, 0.1) == "n/a"


def test_dominance_from_the_intervals():
    """2026-11-01: 'excludes 0' = the whole 95% interval on one side of 0."""
    below, above, across = [-0.2, -0.01], [0.01, 0.3], [-0.1, 0.1]
    assert Q.dominance_ci(below, below, across) == "dominates"
    assert Q.dominance_ci(above, above, across) == "dominated"
    assert Q.dominance_ci(below, above, above) == "improved with trade-off"
    assert Q.dominance_ci(above, below, below) == "worsened with trade-off"
    assert Q.dominance_ci(above, below, across) == "shift only"
    assert (
        Q.dominance_ci(below, across, above)
        == Q.dominance_ci(across, above, below)
        == ("one-sided")
    )
    assert Q.dominance_ci(across, across, above) == "indistinguishable"
    # an interval that ends at 0 contains 0
    assert Q.dominance_ci([-0.2, 0.0], [0.0, 0.1], below) == "indistinguishable"
    assert Q.dominance_ci(below, above, [0.0, 0.1]) == "shift only"
    assert Q.dominance_ci([math.nan, math.nan], below, below) == "n/a"
    assert Q.dominance_ci(below, above, [math.nan, math.nan]) == "n/a"
    assert Q.dominance_ci(below, below, [math.nan, math.nan]) == "dominates"


def test_hand_computed_cells_rules_and_operating_points():
    rep = Q.analyze(_small(), domain="fake", levels=LEVELS, n_boot=500, seed=1)
    vf = rep["verdict_following"][Q.E_INCORRECT]["low"]
    a1, nv = vf["A1"], vf["no_verifier"]
    assert [a1["cells"][p]["n"] for p in Q.PATTERNS] == [1, 1, 1, 1]
    assert [_est(a1["cells"][p]) for p in Q.PATTERNS] == [1.0, 1.0, 0.0, 0.0]
    assert _est(a1["following_gap"]) == 1.0 and _est(nv["following_gap"]) == 0.0
    assert _est(a1["gap_minus_no_verifier"]) == 1.0
    assert nv["gap_minus_no_verifier"] == {"status": "reference"}
    assert _est(a1["rule_agreement"]["V1_only"]) == 1.0
    assert _est(a1["rule_agreement"]["AND"]) == 0.75
    # a cell of one state in four is empty in about (3/4)^4 of the replicates
    assert 0 < a1["cells"]["01"]["n_boot_undefined"] < 500

    ops = rep["operating_points"]["low"]
    pts = ops["points"]
    want = {
        ("arms", "A1"): (0.5, 0.25),
        ("arms", "no_verifier"): (0.5, 0.25),
        ("rules", "V1_only"): (0.5, 0.25),
        ("rules", "V2_only"): (0.5, 0.5),
        ("rules", "OR"): (0.25, 0.5),
        ("rules", "AND"): (0.75, 0.25),
    }
    for (kind, name), (fa, frr) in want.items():
        p = pts[kind][name]
        assert (_est(p["FA"]), _est(p["FRR"])) == (fa, frr)
        assert _est(p["BA"]) == pytest.approx(1 - (fa + frr) / 2)
    first = ops["pairs"][0]
    assert (first["X"], first["Y"], first["status"], first["dominance"]) == (
        "no_verifier",
        "A1",
        "ok",
        "equal",
    )
    assert all(p["status"].startswith("not available") for p in ops["pairs"][1:])
    vs = {(c["arm"], c["rule"]): c for c in ops["arm_minus_rule"]}
    assert _est(vs[("A1", "V2_only")]["dBA"]) == pytest.approx(0.125)
    assert vs[("A1", "V2_only")]["dominance"] == "weakly dominates"
    assert vs[("A1", "AND")]["dominance"] == "weakly dominates"
    assert vs[("A1", "OR")]["dominance"] == "moved" and _est(vs[("A1", "OR")]["dBA"]) == 0.0
    assert _est(vs[("A1", "V1_only")]["dFA"]) == 0.0


def test_arms_are_paired_within_a_cohort_and_the_cohorts_drawn_independently():
    rep = Q.analyze(
        _small({"B": ("1100", "1101")}), domain="fake", levels=LEVELS, n_boot=400, seed=2
    )
    pair = next(
        p for p in rep["operating_points"]["low"]["pairs"] if (p["X"], p["Y"]) == ("A1", "B")
    )
    for k in ("dBA", "dFA", "dFRR"):
        assert pair[k]["estimate"] == 0.0 and pair[k]["ci"] == [0.0, 0.0]
    w_inc = Q._draw_weights(4, n_boot=50, seed=2, cohort=Q.E_INCORRECT)
    w_cor = Q._draw_weights(4, n_boot=50, seed=2, cohort=Q.E_CORRECT)
    assert not np.array_equal(w_inc, w_cor)
    assert np.array_equal(w_inc, Q._draw_weights(4, n_boot=50, seed=2, cohort=Q.E_INCORRECT))
    assert (w_inc.sum(axis=1) == 4).all()


def test_end_to_end_on_shuffled_fake_data(tmp_path: Path):
    rep = Q.analyze_fake(tmp_path / "fake", n_boot=300, seed=4)
    assert rep["status"] == Q.STATUS and "POST HOC" in rep["status"]
    assert [lam for lam, v in rep["lambda_levels"].items() if v["flags_independent_of_text"]] == [
        "high"
    ]
    assert rep["arms_in_one_cohort_only"] == []
    raw: dict[str, list[dict]] = {Q.E_INCORRECT: [], Q.E_CORRECT: []}
    for c in Q.COHORTS:
        for e in rep["inputs"][c]:
            raw[c] += list(read_jsonl(Path(e["file"])))
    raw[Q.E_INCORRECT] = [r for r in raw[Q.E_INCORRECT] if r["arm"] in Q.ARM_ORDER]
    for c, by_lam in rep["verdict_following"].items():
        n = rep["cohorts"][c]["n_states"]
        for arms in by_lam.values():
            assert list(arms) == list(Q.ARM_ORDER)
            for arm, a in arms.items():
                assert sum(a["cells"][p]["n"] for p in Q.PATTERNS) == n
                assert set(a["rule_agreement"]) == set(Q.RULES)
                assert (a["gap_minus_no_verifier"]["status"] == "reference") == (
                    arm == Q.REFERENCE_ARM
                )
    for lam, ops in rep["operating_points"].items():
        for arm, p in ops["points"]["arms"].items():
            acc = {
                c: np.mean(
                    [
                        r["action"] == "accept" and not r["parse_fail"]
                        for r in raw[c]
                        if r["arm"] == arm and r["lam"] == lam
                    ]
                )
                for c in Q.COHORTS
            }
            assert _est(p["FA"]) == pytest.approx(acc[Q.E_INCORRECT])
            assert _est(p["FRR"]) == pytest.approx(1 - acc[Q.E_CORRECT])
        for p in [*ops["points"]["arms"].values(), *ops["points"]["rules"].values()]:
            assert _est(p["BA"]) == pytest.approx(1 - (_est(p["FA"]) + _est(p["FRR"])) / 2)
            assert p["BA"]["ci"][0] <= _est(p["BA"]) <= p["BA"]["ci"][1]
        pts = ops["points"]["arms"]
        assert [(p["X"], p["Y"]) for p in ops["pairs"]] == list(Q.BA_PAIRS)
        for p in ops["pairs"]:
            assert _est(p["dBA"]) == pytest.approx(
                _est(pts[p["Y"]]["BA"]) - _est(pts[p["X"]]["BA"])
            )
            assert _est(p["dFA"]) == pytest.approx(
                _est(pts[p["Y"]]["FA"]) - _est(pts[p["X"]]["FA"])
            )
            assert p["dominance"] == Q.dominance(_est(p["dFA"]), _est(p["dFRR"]))
            assert p["dominance_ci"] == Q.dominance_ci(
                p["dFA"]["ci"], p["dFRR"]["ci"], p["dBA"]["ci"]
            )
        assert len(ops["arm_minus_rule"]) == len(Q.ARM_ORDER) * len(Q.RULES)
    md = Q.render(rep)
    assert Q.STATUS in md and "†" in md
    for arm in Q.ARM_ORDER:
        assert f"| {arm} |" in md


def test_the_outputs_and_the_figure(tmp_path: Path):
    pytest.importorskip("matplotlib")
    rep = Q.analyze_fake(tmp_path / "inputs", n_boot=100, seed=5, n_inc=40, n_cor=30)
    files = Q.write_outputs(rep, tmp_path / "out", prefix="FAKE_")
    assert [f.name for f in files] == [
        "FAKE_report.json",
        "FAKE_tables.md",
        "FAKE_operating_points.png",
        "FAKE_operating_points.pdf",
    ]
    assert all(f.stat().st_size > 0 for f in files)
    assert (
        files[2].read_bytes()[:8] == b"\x89PNG\r\n\x1a\n" and files[3].read_bytes()[:4] == b"%PDF"
    )
    saved = json.loads(files[0].read_text(encoding="utf-8"))
    assert saved["status"] == Q.STATUS and saved["bootstrap"]["seed_component"] == Q.SEED_COMPONENT


def test_the_analysis_does_not_depend_on_row_order(tmp_path: Path):
    spec = Q.write_fake(tmp_path / "fake", n_inc=30, n_cor=30)
    rows = list(read_jsonl(spec[Q.E_CORRECT][0] / "decisions.jsonl"))
    inc = list(read_jsonl(spec["original_decisions"])) + list(
        read_jsonl(spec[Q.E_INCORRECT][0] / "decisions.jsonl")
    )
    shuffled = rows[:]
    random.Random(1).shuffle(shuffled)
    kw = {"domain": "fake", "levels": Q.lambda_levels(Q.FAKE_CHANNEL), "n_boot": 200, "seed": 3}
    a = Q.analyze(
        {
            Q.E_INCORRECT: Q.build_cohort(Q.E_INCORRECT, inc),
            Q.E_CORRECT: Q.build_cohort(Q.E_CORRECT, rows),
        },
        **kw,
    )
    b = Q.analyze(
        {
            Q.E_INCORRECT: Q.build_cohort(Q.E_INCORRECT, inc[::-1]),
            Q.E_CORRECT: Q.build_cohort(Q.E_CORRECT, shuffled),
        },
        **kw,
    )
    assert json.dumps(Q._clean(a)) == json.dumps(Q._clean(b))


def _load(spec: dict, domain: str = Q.FAKE_DOMAIN, **kw):
    args = {
        "original_decisions": spec["original_decisions"],
        "original_sha256": None,
        "reused_arms": spec["reused_arms"],
        "prefix": "exp1ext",
        "base": spec["base"],
        "fake": True,
    } | kw
    return Q.load_cohorts(domain, spec[Q.E_INCORRECT], spec[Q.E_CORRECT], **args)


def test_a_missing_file_or_a_wrong_input_is_an_error(tmp_path: Path):
    spec = Q.write_fake(tmp_path / "fake", n_inc=12, n_cor=10)
    cohorts, inputs, levels = _load(spec)
    assert set(cohorts) == set(Q.COHORTS) and set(levels) == set(Q.LAMBDAS)
    assert [e["role"] for e in inputs[Q.E_INCORRECT]] == [
        "original run, reused arms",
        "extension run",
    ]
    with pytest.raises(Q.DecisionQualityError, match="is a 'fake' run"):
        _load(spec, domain="detox")
    run = spec[Q.E_CORRECT][0]
    for name in Q.RUN_FILES:
        backup = tmp_path / "backup"
        shutil.copy(run / name, backup)
        (run / name).unlink()
        with pytest.raises(Q.DecisionQualityError, match="has no"):
            _load(spec)
        shutil.copy(backup, run / name)
    with pytest.raises(Q.DecisionQualityError, match="does not exist"):
        _load(spec, original_decisions=tmp_path / "nowhere" / "decisions.jsonl")
    with pytest.raises(Q.DecisionQualityError, match="does not exist"):
        Q.load_cohorts(
            Q.FAKE_DOMAIN,
            [tmp_path / "missing"],
            spec[Q.E_CORRECT],
            original_decisions=None,
            original_sha256=None,
            reused_arms=(),
            prefix="exp1ext",
            base=spec["base"],
            fake=True,
        )
    from analysis._common import FakeDataError

    with pytest.raises(FakeDataError):
        _load(spec, fake=False)


def test_inconsistent_flags_incomplete_arms_and_duplicates_are_errors(tmp_path: Path):
    spec = Q.write_fake(tmp_path / "fake", n_inc=12, n_cor=10)
    rows = list(read_jsonl(spec[Q.E_CORRECT][0] / "decisions.jsonl"))
    bad = [dict(r) for r in rows]
    i = next(i for i, r in enumerate(bad) if r["arm"] in Q.OWN_FLAG_ARMS and r["arm"] != "A1")
    bad[i] = {**bad[i], "fields": {"y1": 1 - bad[i]["fields"]["y1"], "y2": bad[i]["fields"]["y2"]}}
    with pytest.raises(Q.DecisionQualityError, match="different flags"):
        Q.build_cohort(Q.E_CORRECT, bad)
    with pytest.raises(Q.DecisionQualityError, match=f"arm {rows[0]['arm']} decided 29 of the 30"):
        Q.build_cohort(Q.E_CORRECT, rows[1:])
    with pytest.raises(Q.DecisionQualityError, match="decided twice"):
        Q.build_cohort(Q.E_CORRECT, [*rows, rows[0]])
    cohorts = {c: Q.build_cohort(c, rows) for c in (Q.E_CORRECT,)}
    with pytest.raises(Q.DecisionQualityError, match="both cohorts"):
        Q.analyze(cohorts, domain="fake", levels=Q.lambda_levels(Q.FAKE_CHANNEL), seed=0)


def test_the_entry_reads_the_domain_config_and_refuses_what_is_missing(tmp_path: Path):
    d = Q.domain_inputs("detox")
    ext = C.load_ext()["exp1"]
    assert (
        d["prefix"] == ext["run_id_prefix"]
        and d["reused_arms"] == ext["e_incorrect"]["reused_arms"]
    )
    assert d["original_sha256"] == ext["e_incorrect"]["original_sha256"]["decisions.jsonl"]
    assert d["original_decisions"].parent.name == Path(ext["e_incorrect"]["original_run"]).name
    with pytest.raises(SystemExit):
        Q.main(["--domain", "detox"])
    with pytest.raises(Q.DecisionQualityError, match="does not exist"):
        Q.main(
            [
                "--domain",
                "detox",
                "--e-incorrect",
                str(tmp_path / "a"),
                "--e-correct",
                str(tmp_path / "b"),
            ]
        )


def test_the_cells_are_the_a1_rows_flags_and_only_arms_shown_them_are_checked(tmp_path: Path):
    """2026-11-01: permuted-B may log its donor's flags (math's original run does) and No
    Verifier may log none; the cells and every number stay those of the A1 rows' flags."""
    spec = Q.write_fake(tmp_path / "fake", n_inc=20, n_cor=20)
    rows = list(read_jsonl(spec[Q.E_CORRECT][0] / "decisions.jsonl"))
    flipped = []
    for r in rows:
        r = dict(r)
        if r["arm"] == "permuted_B":
            r["fields"] = {"y1": 1 - r["fields"]["y1"], "y2": 1 - r["fields"]["y2"]}
        elif r["arm"] == "no_verifier":
            r.pop("fields")
        flipped.append(r)
    a, b = Q.build_cohort(Q.E_CORRECT, rows), Q.build_cohort(Q.E_CORRECT, flipped)
    assert a.states == b.states and a.lams == b.lams
    for lam in a.lams:
        assert all(np.array_equal(x, y) for x, y in zip(a.flags[lam], b.flags[lam], strict=True))
    assert {k: {lam: v.tolist() for lam, v in d.items()} for k, d in a.accept.items()} == {
        k: {lam: v.tolist() for lam, v in d.items()} for k, d in b.accept.items()
    }
    with pytest.raises(Q.DecisionQualityError, match="no A1 rows"):
        Q.build_cohort(Q.E_CORRECT, [r for r in rows if r["arm"] != "A1"])
    other = [{**r, "arm": "math_ceiling"} if r["arm"] == "B" else r for r in rows]
    with pytest.raises(Q.DecisionQualityError, match="neither OWN_FLAG_ARMS nor OTHER_FLAG_ARMS"):
        Q.build_cohort(Q.E_CORRECT, other)
    wrong_a1 = [
        {**r, "fields": {"y1": 1 - r["fields"]["y1"], "y2": r["fields"]["y2"]}}
        if r["arm"] == "A1"
        else r
        for r in rows
    ]
    with pytest.raises(Q.DecisionQualityError, match="different flags"):
        Q.build_cohort(Q.E_CORRECT, wrong_a1)


def test_the_bootstrap_seed_is_the_domains():
    assert Q.domain_seed("detox") == component_seed(master_seed(load_config()), Q.SEED_COMPONENT)
    math_seed = json.loads((paths.ROOT / "configs/math/formal.json").read_text("utf-8"))["seed"]
    # read from math's own run config (today the same value as configs/data.yaml's)
    assert Q.domain_seed("math") == component_seed(math_seed, Q.SEED_COMPONENT)
    with pytest.raises(Q.DecisionQualityError, match="unknown domain"):
        Q.domain_seed("chess")


# read at import: tests/test_analysis.py::test_step11_and_step12_refuse_while_not_frozen leaves
# paths.RUNS pointing at its tmp directory (two MonkeyPatch objects on the same attribute)
REAL_RUNS = paths.RUNS
GROUP64 = REAL_RUNS / "ext_decision_quality_detox_20261101T043908Z"
DETOX_EXP1 = {
    Q.E_INCORRECT: REAL_RUNS / "exp1ext_e_incorrect_20261031T112435Z",
    Q.E_CORRECT: REAL_RUNS / "exp1ext_e_correct_20261031T112435Z",
}


def _numbers(obj, path=()):
    """Every leaf of a report except the notes and the interval dominance (added 2026-11-01)."""
    if isinstance(obj, dict):
        for k, v in obj.items():
            if k not in ("notes", "dominance_ci"):
                yield from _numbers(v, (*path, k))
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            yield from _numbers(v, (*path, i))
    else:
        yield path, obj


@pytest.mark.skipif(
    not (GROUP64 / "report.json").is_file() or not all(p.is_dir() for p in DETOX_EXP1.values()),
    reason="the detox extension runs are not here",
)
def test_detox_results_are_group_64s_bit_for_bit():
    d = Q.domain_inputs("detox")
    cohorts, inputs, levels = Q.load_cohorts(
        "detox",
        [DETOX_EXP1[Q.E_INCORRECT]],
        [DETOX_EXP1[Q.E_CORRECT]],
        original_decisions=d["original_decisions"],
        original_sha256=d["original_sha256"],
        reused_arms=d["reused_arms"],
        prefix=d["prefix"],
        base=REAL_RUNS,
    )
    rep = Q.analyze(cohorts, domain="detox", levels=levels, seed=Q.domain_seed("detox"),
                    inputs=inputs)  # fmt: skip
    rep["ext_config"] = d["ext_config"]
    new = json.loads(json.dumps(Q._clean(rep)))
    old = json.loads((GROUP64 / "report.json").read_text(encoding="utf-8"))
    assert list(_numbers(new)) == list(_numbers(old))
