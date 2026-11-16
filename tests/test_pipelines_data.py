"""pipelines: step00 / step04 / step05 end-to-end on the fake provider,
step07 / step08 / step09 on synthetic data (step03: tests/test_step03.py). Offline: nothing
here reaches a real API."""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import pytest

from rafr.contracts.schemas import TBD, EvalRecord, SafetyLabel, Split
from rafr.core import paths
from rafr.core.config import (
    NotFrozenError,
    assert_frozen,
    find_tbd,
    load_config,
    master_seed,
)
from rafr.data import labels as L
from rafr.data import splits as S
from rafr.data.manifest import Manifest
from rafr.evaluator import judge
from rafr.llm.client import content_text
from rafr.logging.writer import read_jsonl

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pipelines import step00_prepare_data as step00  # noqa: E402
from pipelines import step04_build_states as step04  # noqa: E402
from pipelines import step05_label_and_split as step05  # noqa: E402

FIXTURES = ROOT / "tests" / "fixtures"


def _fake_step00(*extra: str) -> int:
    return step00.main(["--fake", "--skip-download", "--raw-dir", str(FIXTURES), *extra])


def test_step00_fake_end_to_end(sandbox: Path):
    """every deduplicated x0 is judged, nothing is split, the split plan is pinned."""
    assert _fake_step00() == 0
    root = sandbox / "data" / paths.FAKE_ROOT_NAME
    m = Manifest.load(root / "manifest.json")
    assert m.cleaning["n_kept"] == 9
    assert m.prefilter["n_candidates"] == 9 and m.prefilter["n_kept"] == 9  # fake E: UNSAFE
    assert m.prefilter["complete"] is True
    assert (
        m.prefilter["e_identity"]["prompt_sha256"]
        == (load_config("models")["evaluator"]["prompt_sha256"])
    )
    assert len(list(read_jsonl(root / "pools" / "x0_prefiltered.jsonl"))) == 9
    rows = list(read_jsonl(root / "pools" / "prefilter.jsonl"))
    assert len(rows) == 9 and all(r["e_prompt_sha256"] and r["e_version"] for r in rows)
    assert not (root / "splits").exists() or not list((root / "splits").glob("*.jsonl"))
    assert m.splits["plan"]["seed_component"] == "splits_v09" and "sizes" not in m.splits
    run_dirs = list((root / "runs").glob("step00_*"))
    assert len(run_dirs) == 1 and (run_dirs[0] / "evaluations.jsonl").exists()


def test_step00_prefilter_drops_x0_when_e_keeps_failing(sandbox: Path, monkeypatch):
    """paper Sec. 4.2: an x0 on which E still fails after its retries is dropped with its reason
    recorded; the step does not abort."""
    from rafr.core.runtime import build_client
    from rafr.llm.providers.fake import FakeProvider, FakeReply, default_responder

    def responder(req):
        if "nobody asked" in content_text(req.messages[0]["content"]):
            return FakeReply(text=None, refused=True, stop_reason="refusal")
        return default_responder(req)

    fp = FakeProvider(responder=responder)
    monkeypatch.setattr(
        step00,
        "build_client",
        lambda models_cfg, run_dir, fake: build_client(
            models_cfg, run_dir=run_dir, providers={"openai": fp, "anthropic": fp, "fake": fp}
        ),
    )
    assert _fake_step00() == 0
    root = sandbox / "data" / paths.FAKE_ROOT_NAME
    m = Manifest.load(root / "manifest.json")
    assert (m.prefilter["n_candidates"], m.prefilter["n_kept"], m.prefilter["n_e_failed"]) == (
        9,
        8,
        1,
    )
    meta = json.loads((root / "pools" / "prefilter_meta.json").read_text(encoding="utf-8"))
    assert len(meta["e_failed"]) == 1 and "refusal" in meta["e_failed"][0]["reason"]
    rows = list(read_jsonl(root / "pools" / "prefilter.jsonl"))
    bad = [r for r in rows if r["label"] is None]
    assert len(bad) == 1 and "nobody asked" in bad[0]["text"] and "e_error" in bad[0]
    kept = list(read_jsonl(root / "pools" / "x0_prefiltered.jsonl"))
    assert bad[0]["x0_id"] not in {r["x0_id"] for r in kept}
    # E was retried (3 attempts) on the failing text before giving up
    assert sum("nobody asked" in content_text(c.messages[0]["content"]) for c in fp.calls) == 3


# These two used to call main() without --fake and assert that the guard refused. the authors
# filled the last evaluator / controller TBDs on 2026-09-20, so the guard now passes -- and
# that turned the step00 one into nine live E calls before it failed further down. The guard
# is checked directly from here on; conftest.no_live_api_keys stops a repeat.


def test_step00_guard_is_released_but_still_bites(sandbox: Path):
    """step00's sections (data.* + models.evaluator) are frozen since 2026-09-20, so the
    prefilter may run. Put a TBD back into a section it reads and it refuses again."""
    cfgs = load_config()
    assert_frozen(cfgs, only_sections=step00.GUARD_SECTIONS)  # does not raise
    cfgs["models"]["evaluator"]["version"] = "TBD"
    with pytest.raises(NotFrozenError) as ei:
        assert_frozen(cfgs, only_sections=step00.GUARD_SECTIONS)
    assert ei.value.tbd == ["models.evaluator.version"]
    assert ei.value.placeholders == []  # the E prompt (annotator_arm_b.txt) is final


def test_step04_guard_is_released_but_still_bites(sandbox: Path):
    """Same for step04: the controller backbone was the last TBD it read."""
    cfgs = load_config()
    extra = [cfgs["generation"]["g_fixed"]["validity"]["refusal_phrases_file"]]
    assert_frozen(cfgs, only_sections=step04.GUARD_SECTIONS, extra_files=extra)  # does not raise
    cfgs["models"]["controller"]["model"] = "TBD"
    with pytest.raises(NotFrozenError) as ei:
        assert_frozen(cfgs, only_sections=step04.GUARD_SECTIONS, extra_files=extra)
    assert ei.value.tbd == ["models.controller.model"]
    # g_fixed.txt and refusal_phrases.yaml were filled on 2026-09-18; only TBD values block
    assert ei.value.placeholders == []


def test_step04_still_refuses_on_a_verifier_tbd_only_through_the_steps_that_read_it(
    sandbox: Path,
):
    """A TBD verifier id must not block step04, which does not read it -- the narrowed guard
    must not resurrect that block (2026-09-18). The real ids are filled since
    2026-09-25, so the TBD is put back on a copy."""
    cfgs = load_config()
    cfgs["models"]["verifiers"]["v1"]["id"] = TBD
    assert "models.verifiers.v1.id" in find_tbd(cfgs, "")
    assert_frozen(
        cfgs,
        only_sections=step04.GUARD_SECTIONS,
        extra_files=[cfgs["generation"]["g_fixed"]["validity"]["refusal_phrases_file"]],
    )


def _e_record(text: str, label: SafetyLabel) -> EvalRecord:
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


def _write_split_data(n: int = 12):
    """states_d_fit + its split file + y* (E's judgments) + two score tables in the sandbox."""
    from rafr.contracts.schemas import Score, State
    from rafr.verifiers import scores_table_path, write_scores_table

    paths.DATA_POOLS.mkdir(parents=True, exist_ok=True)
    paths.DATA_LABELS.mkdir(parents=True, exist_ok=True)
    states, s1, s2 = [], [], []
    with L.y_star_path().open("a", encoding="utf-8") as ys:
        for i in range(n):
            st = State(state_id=f"x{i}-g1", x0_id=f"x{i}", x0="o", text=f"t{i}", split=Split.D_FIT)
            states.append(st)
            unsafe = i % 2 == 0
            lab = SafetyLabel.UNSAFE if unsafe else SafetyLabel.SAFE
            row = L.StateJudgment(
                state_id=st.state_id,
                x0_id=st.x0_id,
                label=lab,
                record=_e_record(st.text, lab),
                e_identity={},
            )
            ys.write(row.model_dump_json() + "\n")
            s1.append(
                Score(
                    state_id=st.state_id,
                    verifier_id="real-v1",
                    model_version="m",
                    score=0.8 if unsafe else 0.2,
                )
            )
            s2.append(
                Score(
                    state_id=st.state_id,
                    verifier_id="real-v2",
                    model_version="m",
                    score=0.7 if unsafe else 0.4,
                )
            )
    with (paths.DATA_POOLS / "states_d_fit.jsonl").open("w", encoding="utf-8") as fh:
        for st in states:
            fh.write(st.model_dump_json() + "\n")
    S.save_splits({Split.D_FIT: [st.x0_id for st in states]}, paths.DATA_SPLITS)
    write_scores_table(s1, scores_table_path("real-v1", paths.DATA_SCORES))
    write_scores_table(s2, scores_table_path("real-v2", paths.DATA_SCORES))


def test_step07_refuses_while_verifiers_tbd(sandbox: Path, monkeypatch):
    from pipelines import step07_fit_tau as step07

    from rafr.core import config as C

    cfgs = C.load_config()
    cfgs["models"]["verifiers"]["v1"]["id"] = TBD  # the real ids are filled since 2026-09-25
    monkeypatch.setattr(step07, "load_config", lambda: cfgs)
    with pytest.raises(NotFrozenError) as ei:
        step07.main([])
    assert "models.verifiers.v1.id" in ei.value.tbd


def test_step07_end_to_end_with_frozen_verifier_ids(sandbox: Path, monkeypatch):
    from pipelines import step07_fit_tau as step07

    from rafr.core import config as C

    _write_split_data()
    real = C.load_config()
    real["models"]["verifiers"]["v1"] = {"id": "real-v1", "card": "cards/v1.md"}
    real["models"]["verifiers"]["v2"] = {"id": "real-v2", "card": "cards/v2.md"}
    monkeypatch.setattr(step07, "load_config", lambda: real)
    assert step07.main([]) == 0
    data = json.loads(paths.FROZEN_TAU.read_text(encoding="utf-8"))
    assert set(data["fits"]) == {"v1", "v2"} and data["split"] == "d_fit"
    assert 0.2 < data["fits"]["v1"]["tau"] < 0.8 and 0.4 < data["fits"]["v2"]["tau"] < 0.7
    assert data["fits"]["v1"]["f1_max"] == 1.0
    with pytest.raises(SystemExit):  # frozen: refuse to overwrite without --force
        step07.main([])


def _write_pool_scores(n: int = 40):
    """states_unlabeled_pool + two score tables spread over [0, 1] (for q)."""
    from rafr.contracts.schemas import Score, State
    from rafr.verifiers import scores_table_path, write_scores_table

    states, s1, s2 = [], [], []
    for i in range(n):
        st = State(state_id=f"p{i}-g1", x0_id=f"p{i}", x0="o", text=f"pool {i}")
        states.append(st)
        s1.append(
            Score(state_id=st.state_id, verifier_id="real-v1", model_version="m", score=i / n)
        )
        s2.append(
            Score(state_id=st.state_id, verifier_id="real-v2", model_version="m", score=i / n)
        )
    with (paths.DATA_POOLS / "states_unlabeled_pool.jsonl").open("w", encoding="utf-8") as fh:
        for st in states:
            fh.write(st.model_dump_json() + "\n")
    # append to the existing tables (D_fit rows written by _write_split_data)
    from rafr.verifiers import load_scores_table

    for vid, rows in (("real-v1", s1), ("real-v2", s2)):
        p = scores_table_path(vid, paths.DATA_SCORES)
        old = load_scores_table(p)
        merged = [
            Score(state_id=k, verifier_id=vid, model_version="m", score=v) for k, v in old.items()
        ]
        write_scores_table(merged + rows, p)


def _validity_config(sandbox: Path, *, delta_r=0.1, population="both"):
    from rafr.core import config as C

    real = C.load_config()
    real["models"]["verifiers"]["v1"] = {"id": "real-v1"}
    real["models"]["verifiers"]["v2"] = {"id": "real-v2"}
    real["channel"]["diagnostics"]["out_dir"] = str(sandbox / "diag")
    real["channel"]["gate"]["ci"]["n_boot"] = 50  # small B keeps the test fast (paper Sec. 4.3)
    # p is fixed at 0 / 0.5 / 1 (paper Sec. 4.3); delta_R is only checked against the span, which a
    # perfectly separating D_fit keeps wide, so 0.1 passes the check here.
    real["channel"]["gate"]["delta_r"] = delta_r
    real["channel"]["diagnostics"]["diagnostics_population"] = population
    return real


FIXED_LEVELS_P = {
    "low": {"v1": 0.0, "v2": 0.0},
    "mid": {"v1": 0.5, "v2": 0.5},
    "high": {"v1": 1.0, "v2": 1.0},
}


def _step08_sandbox(sandbox: Path, monkeypatch, **cfg):
    from pipelines import step07_fit_tau as step07
    from pipelines import step08_tune_channel as step08

    _write_split_data()
    _write_pool_scores()
    real = _validity_config(sandbox, **cfg)
    monkeypatch.setattr(step07, "load_config", lambda: real)
    monkeypatch.setattr(step08, "load_config", lambda: real)
    monkeypatch.setattr(step08, "CANDIDATE", sandbox / "frozen" / "channel.candidate.json")
    assert step07.main([]) == 0
    return step08, real


def test_step08_refuses_the_formal_run_while_delta_r_or_the_population_is_tbd(
    sandbox: Path, monkeypatch
):
    """the formal delta_R and diagnostics_population are
    the authors'; the formal run refuses while either is TBD. --reachability-only needs no formal
    delta_R (it falls back to the provisional one) but, since it now measures the ladder and the
    cue diagnostics too, it does need the population."""
    step08, real = _step08_sandbox(sandbox, monkeypatch, delta_r="TBD", population="TBD")
    with pytest.raises(NotFrozenError) as ei:
        step08.main([])
    assert set(ei.value.tbd) == {
        "channel.gate.delta_r",
        "channel.diagnostics.diagnostics_population",
    }
    with pytest.raises(NotFrozenError) as ei:
        step08.main(["--reachability-only"])
    assert set(ei.value.tbd) == {"channel.diagnostics.diagnostics_population"}
    real["channel"]["diagnostics"]["diagnostics_population"] = "both"
    assert step08.main(["--reachability-only"]) == 0
    assert not (sandbox / "frozen" / "channel.candidate.json").exists()
    assert not (sandbox / "diag").exists()
    rd = next(paths.RUNS.glob("step08_*"))
    out = json.loads((rd / "reachability.json").read_text(encoding="utf-8"))
    assert out["delta_r"] == 0.10 and out["delta_r_status"] == "provisional"  #
    assert out["gate_population"] == "e_unsafe" and out["stopped"] is False
    # the whole check, under the fixed p: the realized ladder with its CI and the five
    # diagnostics on both populations, lambda = 0 baseline included (paper App. A)
    assert out["levels_p"]["fixed"] is True and out["levels_p"]["values"] == FIXED_LEVELS_P
    assert set(out["ladder_on_d_fit"]["r"]) == {"v1", "v2"} and "ci" in out["ladder_on_d_fit"]
    assert set(out["cue_diagnostics"]) == {"unsafe", "all"}
    assert set(out["cue_diagnostics"]["all"]) == {"none", "low", "mid", "high"}
    real["channel"]["diagnostics"]["diagnostics_population"] = "everything"
    real["channel"]["gate"]["delta_r"] = 0.1
    with pytest.raises(SystemExit, match="diagnostics_population"):
        step08.main([])


def test_step08_reachability_is_tpr_on_the_e_unsafe_subset(sandbox: Path, monkeypatch):
    """Gate metric (d) (section 1):
    baseline TPR on D_fit's E-UNSAFE subset, the all-resampled floor r = P_{s~q}(s >= tau) from
    the pool's scores alone, the span and 2 delta_R; p fixed at 0 / 0.5 / 1, recorded as
    such, never solved; R and its CI on the same subset; accuracy / balanced accuracy on all of
    D_fit, descriptively."""
    from rafr.channel.margin_fit import fit_margin_model
    from rafr.core.seeding import component_seed
    from rafr.validity.tau import load_tau

    step08, real = _step08_sandbox(sandbox, monkeypatch)
    assert step08.main([]) == 0
    rep = json.loads((sandbox / "diag" / "step08_report.json").read_text(encoding="utf-8"))
    taus = load_tau(paths.FROZEN_TAU)
    pool = [i / 40 for i in range(40)]
    assert rep["gate_population"] == "e_unsafe" and rep["n_e_unsafe"] == 6
    for v in ("v1", "v2"):
        rc = rep["reachability"][v]
        assert rc["n_states"] == 6 and rc["population"] == "e_unsafe"
        assert rc["baseline_tpr"] == 1.0  # D_fit scores separate the labels perfectly
        floor = fit_margin_model(pool, taus[v].tau, v).prob_at_least(taus[v].tau)
        assert rc["all_resampled_floor"] == pytest.approx(floor)
        assert rc["span"] == pytest.approx(1.0 - floor)
        assert rc["expected_adjacent_gap"] == pytest.approx((1.0 - floor) / 2)
        assert rc["two_gaps_of_delta_r_reachable"] == (rc["span"] >= 2 * rep["delta_r"])
        # paper Sec. 4.3: the same three quantities in accuracy / balanced-accuracy units over
        # ALL of D_fit, descriptively; the floors follow from a label-independent judgment
        d = rc["descriptive_whole_split"]
        assert d["n_states"] == 12 and d["e_unsafe_share"] == pytest.approx(0.5)
        assert d["accuracy"]["baseline"] == 1.0 and d["balanced_accuracy"]["baseline"] == 1.0
        assert d["accuracy"]["all_resampled_floor"] == pytest.approx(
            floor * 0.5 + (1 - floor) * 0.5
        )
        assert d["balanced_accuracy"]["all_resampled_floor"] == 0.5
        assert d["accuracy"]["span"] == pytest.approx(1.0 - d["accuracy"]["all_resampled_floor"])
        assert d["balanced_accuracy"]["span"] == pytest.approx(0.5)
        # fixed p: the expected ladder is TPR, (TPR + r)/2, r
        ex = rep["expected_ladder"][v]
        assert ex["low"] == 1.0 and ex["high"] == pytest.approx(floor)
        assert ex["mid"] == pytest.approx((1.0 + floor) / 2)
    # fixed, not solved (paper Sec. 4.3): the record says so, nothing was derived from the span
    assert "solved_from" not in rep
    assert rep["levels_p"]["fixed"] is True and rep["levels_p"]["values"] == FIXED_LEVELS_P
    assert "C.27" in rep["levels_p"]["rule"]
    assert rep["derived_p_z"]["low"]["none"] == pytest.approx(1.0)
    assert rep["derived_p_z"]["high"]["both"] == pytest.approx(1.0)
    cand = json.loads((sandbox / "frozen" / "channel.candidate.json").read_text(encoding="utf-8"))
    assert cand["geometry"] == "independent_resampling" and cand["status"] == "candidate"
    assert cand["levels"] == FIXED_LEVELS_P
    ladder = rep["ladder_on_d_fit"]
    assert set(ladder["r"]) == {"v1", "v2"} and ladder["population"] == "e_unsafe"
    assert ladder["n_states"] == 6
    # (paper Sec. 4.3): the CI travels with the ladder result and is reporting material only;
    # it resamples the six E-UNSAFE states, not the twelve
    ci = ladder["ci"]
    assert ci["unit"] == "state" and ci["n_boot"] == 50 and ci["n_states"] == 6
    assert ci["seed"] == component_seed(master_seed(real), "gate_ci")  # from the master seed
    assert sum(len(v) for v in ci["r"].values()) == 6
    assert sum(len(v) for v in ci["gaps"].values()) == 4
    assert ladder["passed"] == (not ladder["failed_conditions"])
    desc = rep["descriptive_whole_split"]
    assert desc["v1"]["none"] == {"n": 12, "accuracy": 1.0, "balanced_accuracy": 1.0}
    assert set(rep["cue_diagnostics"]) == {"all", "unsafe"}
    # 2026-09-24: with `both`, E-UNSAFE primary, the whole split secondary
    assert rep["diagnostics_population"] == ["unsafe", "all"]
    assert rep["diagnostics_roles"] == {"unsafe": "primary", "all": "secondary"}
    assert rep["cue_diagnostics"]["all"]["none"]["n"] == 12
    assert rep["cue_diagnostics"]["unsafe"]["none"]["n"] == 6
    assert set(rep["cue_diagnostics"]["unsafe"]) == {"none", "low", "mid", "high"}


@pytest.mark.parametrize("reachability_only", [False, True])
def test_step08_stops_when_the_span_is_below_two_delta_r(
    sandbox: Path, monkeypatch, reachability_only: bool
):
    """paper Sec. 4.3: a span TPR_i - r_i < 2 delta_R stops step08 -- the numbers are
    printed and recorded in the run directory, nothing goes to frozen/ or data/diagnostics/, and
    p stays where it is (0 / 0.5 / 1). With the formal delta_R and, under --reachability-only,
    with the provisional one alike."""
    kw = {"delta_r": "TBD"} if reachability_only else {"delta_r": 0.45}
    step08, real = _step08_sandbox(sandbox, monkeypatch, **kw)
    if reachability_only:
        real["channel"]["gate"]["delta_r_provisional"] = 0.45
    argv = ["--reachability-only"] if reachability_only else []
    with pytest.raises(SystemExit, match=r"span TPR_i - r_i < 2 \* delta_R = 0\.900"):
        step08.main(argv)
    assert not (sandbox / "frozen" / "channel.candidate.json").exists()
    assert not (sandbox / "diag").exists()
    out = json.loads(
        (next(paths.RUNS.glob("step08_*")) / "reachability.json").read_text(encoding="utf-8")
    )
    assert out["stopped"] is True and set(out["short_span"]) == {"v1", "v2"}
    assert out["delta_r"] == 0.45
    assert out["delta_r_status"] == ("provisional" if reachability_only else "formal")
    assert out["levels_p"]["values"] == FIXED_LEVELS_P  # a short span never moves p
    for v in ("v1", "v2"):
        assert out["reachability"][v]["span"] < 0.9
        assert not out["reachability"][v]["two_gaps_of_delta_r_reachable"]


def test_step07_and_step08_outputs_are_lf_on_any_platform(sandbox: Path, monkeypatch):
    """everything step07 and step08 write -- frozen/, the run
    directories (snapshot, the configs copy, reachability.json), data/diagnostics/ -- is LF."""
    step08, _ = _step08_sandbox(sandbox, monkeypatch)
    assert step08.main(["--reachability-only"]) == 0
    assert step08.main([]) == 0
    written = [
        f
        for d in (sandbox / "frozen", paths.RUNS, sandbox / "diag")
        for f in d.rglob("*")
        if f.is_file()
    ]
    assert any(f.name == "channel.candidate.json" for f in written)
    assert any(f.name == "snapshot.json" for f in written)
    assert any(f.suffix == ".yaml" for f in written)  # the configs copy
    for f in written:
        assert b"\r" not in f.read_bytes(), f


def test_step08_reads_d_fit_labels_only_for_the_channel_check(sandbox: Path, monkeypatch):
    """paper Sec. 4.2 (2026-09-26): step08 reads y* once, of D_fit, under purpose CHANNEL_CHECK,
    which
    the gate lets through for D_fit only."""
    step08, _ = _step08_sandbox(sandbox, monkeypatch)
    seen: list[tuple] = []
    real_read = L.read_labels

    def spy(split, purpose, **kw):
        seen.append((split, purpose))
        return real_read(split, purpose, **kw)

    monkeypatch.setattr(L, "read_labels", spy)
    assert step08.main(["--reachability-only"]) == 0
    assert step08.main([]) == 0
    assert seen == [(Split.D_FIT, L.LabelPurpose.CHANNEL_CHECK)] * 2
    allowed = {sp for sp, ps in L.ALLOWED.items() if L.LabelPurpose.CHANNEL_CHECK in ps}
    assert allowed == {Split.D_FIT}
    for sp in (Split.D_GATE, Split.D_EXP1, Split.UNLABELED_POOL):
        with pytest.raises(L.LabelAccessError):
            real_read(sp, L.LabelPurpose.CHANNEL_CHECK)


def _write_gate_split(unsafe_scores, safe_scores):
    """states_d_gate + its split file + y* + both score tables (D_fit already on disk)."""
    from rafr.contracts.schemas import Score, State
    from rafr.verifiers import load_scores_table, scores_table_path, write_scores_table

    states = []
    rows = {"real-v1": [], "real-v2": []}
    with L.y_star_path().open("a", encoding="utf-8") as ys:
        for i, (lab, (a, b)) in enumerate(
            [(SafetyLabel.UNSAFE, x) for x in unsafe_scores]
            + [(SafetyLabel.SAFE, x) for x in safe_scores]
        ):
            st = State(state_id=f"g{i}-g1", x0_id=f"g{i}", x0="o", text=f"gate {i}")
            states.append(st)
            ys.write(
                L.StateJudgment(
                    state_id=st.state_id, x0_id=st.x0_id, label=lab,
                    record=_e_record(st.text, lab), e_identity={},
                ).model_dump_json() + "\n"
            )  # fmt: skip
            rows["real-v1"].append(
                Score(state_id=st.state_id, verifier_id="real-v1", model_version="m", score=a)
            )
            rows["real-v2"].append(
                Score(state_id=st.state_id, verifier_id="real-v2", model_version="m", score=b)
            )
    with (paths.DATA_POOLS / "states_d_gate.jsonl").open("w", encoding="utf-8") as fh:
        for st in states:
            fh.write(st.model_dump_json() + "\n")
    fit_ids = S.load_split(Split.D_FIT, paths.DATA_SPLITS)
    S.save_splits(
        {Split.D_FIT: fit_ids, Split.D_GATE: [st.x0_id for st in states]}, paths.DATA_SPLITS
    )
    for vid, new_rows in rows.items():
        path = scores_table_path(vid, paths.DATA_SCORES)
        old = load_scores_table(path)
        merged = [
            Score(state_id=k, verifier_id=vid, model_version="m", score=v) for k, v in old.items()
        ]
        write_scores_table(merged + new_rows, path)


def test_step09_gate_uses_the_e_unsafe_subset_and_reports_the_rest(sandbox: Path, monkeypatch):
    """paper Sec. 4.3 with gate metric (d): R and its CI on D_gate's E-UNSAFE states only; accuracy
    and balanced accuracy on all of D_gate as description; diagnostics per population written before
    the verdict; the attempt record and the reserve logic unchanged."""
    from pipelines import step07_fit_tau as step07
    from pipelines import step08_tune_channel as step08
    from pipelines import step09_gate as step09

    from rafr.validity import gate as G

    _write_split_data()
    _write_pool_scores()
    unsafe = [(0.9, 0.9)] * 8
    _write_gate_split(unsafe, [(0.1, 0.1)] * 4)
    real = _validity_config(sandbox, population="unsafe")
    real["data"]["splits"]["sizes"]["d_gate"] = 12  # the fixture's D_gate (step09 checks it)
    for step in (step07, step08, step09):
        monkeypatch.setattr(step, "load_config", lambda: real)
    cand = sandbox / "frozen" / "channel.candidate.json"
    monkeypatch.setattr(step08, "CANDIDATE", cand)
    monkeypatch.setattr(step09, "CANDIDATE", cand)
    assert step07.main([]) == 0 and step08.main([]) == 0
    assert step09.main([]) == 0
    attempts = G.read_attempts(paths.FROZEN_GATE_ATTEMPTS)
    assert len(attempts) == 1 and attempts[0]["split"] == "d_gate"
    a = attempts[0]
    assert a["population"] == "e_unsafe" and a["n_states"] == 8
    assert a["ci"]["n_states"] == 8 and a["ci"]["n_boot"] == 50
    assert a["descriptive_whole_split"]["v1"]["none"]["n"] == 12
    assert a["r"]["v1"]["low"] == 1.0  # every E-UNSAFE state is detected without perturbation
    diag = json.loads((sandbox / "diag" / "gate_diagnostics.json").read_text(encoding="utf-8"))
    assert diag["diagnostics_population"] == ["unsafe"]
    assert set(diag["cue_diagnostics"]) == {"unsafe"}
    assert diag["cue_diagnostics"]["unsafe"]["none"]["n"] == 8
    assert diag["r_ref"]["population"] == "e_unsafe" and diag["r_ref"]["n"] == 8
    assert diag["r_ref"]["by_verifier"] == {"v1": 1.0, "v2": 1.0}
    # the SAFE states do not move R: change their scores, rerun on a fresh record
    paths.FROZEN_GATE_ATTEMPTS.unlink()
    _write_gate_split(unsafe, [(0.99, 0.99)] * 4)
    assert step09.main([]) == 0
    b = G.read_attempts(paths.FROZEN_GATE_ATTEMPTS)[0]
    assert b["r"] == a["r"] and b["ci"] == a["ci"]
    assert b["descriptive_whole_split"] != a["descriptive_whole_split"]
    # attempt bookkeeping unchanged: after a failure the next attempt must use the reserve
    if b["passed"]:
        return
    assert G.next_attempt_split(paths.FROZEN_GATE_ATTEMPTS) is Split.D_GATE_RESERVE


@pytest.mark.parametrize("passes", [True, False])
def test_step09_prints_what_it_reads_reads_y_star_once_and_writes_one_report(
    sandbox: Path, monkeypatch, capsys, passes: bool
):
    """2026-09-27: before any label is read step09 prints the split and its size, the
    score tables (raw-byte sha256, rows, one identity), the GATE purpose and the reserve's lock;
    y* is read once, of D_gate, under GATE, and D_gate_reserve is neither read nor scored; the
    paired correctness correlation across levels is computed; data/diagnostics/step09_report.json
    is written pass or fail. On a failed gate frozen/channel.json is not written."""
    from pipelines import step07_fit_tau as step07
    from pipelines import step08_tune_channel as step08
    from pipelines import step09_gate as step09

    _write_split_data()
    _write_pool_scores()
    _write_gate_split([(0.9, 0.9)] * 8, [(0.1, 0.1)] * 4)
    real = _validity_config(sandbox)
    real["data"]["splits"]["sizes"]["d_gate"] = 12  # the fixture's D_gate
    for step in (step07, step08, step09):
        monkeypatch.setattr(step, "load_config", lambda: real)
    cand = sandbox / "frozen" / "channel.candidate.json"
    monkeypatch.setattr(step08, "CANDIDATE", cand)
    monkeypatch.setattr(step09, "CANDIDATE", cand)
    assert step07.main([]) == 0 and step08.main([]) == 0
    if not passes:
        real["channel"]["gate"]["delta_r"] = 0.9  # no gap reaches it
    seen: list[tuple] = []
    real_read = L.read_labels

    def spy(split, purpose, **kw):
        seen.append((split, purpose))
        return real_read(split, purpose, **kw)

    monkeypatch.setattr(L, "read_labels", spy)
    capsys.readouterr()
    assert step09.main([]) == 0
    out = capsys.readouterr().out
    first_label_line = out.index("[step09] attempt 1")
    for key in ("preflight: attempt on d_gate, 12 states", "config guard passed", "sha256",
                "purpose gate", "D_gate_reserve locked=True"):  # fmt: skip
        assert key in out[:first_label_line], key
    assert seen == [(Split.D_GATE, L.LabelPurpose.GATE)]  # once, D_gate only, never the reserve
    rep = json.loads((sandbox / "diag" / "step09_report.json").read_text(encoding="utf-8"))
    assert rep["passed"] is passes and rep["attempt"] == 1 and rep["split"] == "d_gate"
    assert rep["n_e_unsafe"] == 8 and rep["n_states"] == 12
    assert set(rep["r"]) == {"v1", "v2"} and set(rep["gaps"]["v1"]) == {"low-mid", "mid-high"}
    assert rep["ci"]["n_boot"] == 50 and isinstance(rep["ci"]["seed"], int)
    assert set(rep["p_at_least_one_missed"]) == {"low", "mid", "high"}
    assert rep["descriptive_whole_split"]["v1"]["none"]["n"] == 12
    assert set(rep["cue_diagnostics"]) == {"unsafe", "all"}
    pc = rep["paired_correctness_correlation"]
    assert set(pc["v1"]) == {"low-mid", "mid-high", "low-high"} and pc["v1"]["low-mid"]["n"] == 8
    pre = rep["preflight"]
    assert pre["d_gate_reserve"] == {"locked": True, "labels_read": False, "scored": False}
    assert pre["score_tables"]["v1"]["rows_for_split"] == 12
    assert pre["labels"] == {"split": "d_gate", "purpose": "gate", "reads": 1}
    assert (paths.FROZEN_CHANNEL.exists()) is passes
    assert rep["frozen_channel"] == (str(paths.FROZEN_CHANNEL) if passes else None)
    assert not (paths.DATA_SCORES / "reserve").exists()


# ---------------------------------------------------------------------------------------
# step00: the whole pool is judged; --limit / --budget / E-failure stops only
# ---------------------------------------------------------------------------------------


class _Log:
    def write(self, rec):  # the prefilter writes EvalRecords here
        pass


def _ecfg() -> judge.EvaluatorConfig:
    return judge.build_config(load_config("models"))  # the real, pinned E


def _stub_judge(monkeypatch, labels, *, ledger=None, usage=None, fail_texts=()):
    """Replace E with a scripted judge; optionally bill the ledger like a live call."""
    from rafr.llm.ledger import Usage

    seen: list[str] = []

    def _judge(client, ecfg, text):
        seen.append(text)
        if ledger is not None:
            ledger.record(
                purpose="evaluator",
                provider="fake",
                model="m",
                model_version="v",
                usage=usage or Usage(1000, 10),
                cached=False,
            )
        if text in fail_texts:
            raise step00.judge.EvaluatorError(
                f"E failed on {text}", ["unparseable", "unparseable", "unparseable"]
            )
        return EvalRecord(
            text_id=text,
            text=text,
            label=SafetyLabel(labels(text)),
            model=ecfg.model,
            model_version="v",
            prompt_hash="h",
            e_prompt_sha256=ecfg.prompt_sha256,
            e_version=ecfg.version,
            decoding=ecfg.decoding,
        ), []

    monkeypatch.setattr(step00.judge, "judge_with_failures", _judge)
    return seen


def _x0s(n: int, start: int = 0):
    return [{"x0_id": f"x{i}", "text": f"t{i}"} for i in range(start, start + n)]


def test_processing_order_is_a_seeded_shuffle_of_the_whole_pool():
    """Every x0 is judged; the order only decides what a --limit batch (the dry run) reaches
    first, and train.tsv is laid out in blocks by source corpus."""
    pool = _x0s(1000)
    a = step00.processing_order(pool, 12345)
    assert a == step00.processing_order(pool, 12345)  # reproducible
    assert a != pool and sorted(r["x0_id"] for r in a) == sorted(r["x0_id"] for r in pool)
    assert a != step00.processing_order(pool, 12346)  # the seed is load-bearing
    tail = sum(1 for r in a[:100] if int(r["x0_id"][1:]) >= 500)
    assert 25 < tail < 75  # a dry run of 100 is not the first block of the file


def test_step00_has_no_split_and_no_split_driven_stop():
    """the "enough to fill the splits" and "cannot fill the splits" stops are gone."""
    for name in ("candidate_cap", "wilson_interval", "shortfall_report", "plan_topup"):
        assert not hasattr(step00, name), name
    from rafr.core.runtime import ERROR_STOP_REASONS

    assert "projection" not in ERROR_STOP_REASONS


def test_prefilter_judges_every_x0_whatever_the_pass_rate(tmp_path: Path, monkeypatch):
    x0s = _x0s(60)
    seen = _stub_judge(monkeypatch, lambda t: "UNSAFE" if int(t[1:]) % 10 == 0 else "SAFE")
    r = step00.prefilter(
        None, _ecfg(), x0s, keep_label="UNSAFE", out_path=tmp_path / "p.jsonl", log=_Log()
    )
    assert r.complete and len(seen) == 60 and len(r.kept) == 6
    assert r.pass_rate == pytest.approx(0.1)


def test_prefilter_limit_stops_and_a_rerun_does_not_re_buy(tmp_path: Path, monkeypatch):
    out = tmp_path / "prefilter.jsonl"
    x0s = _x0s(20)
    seen = _stub_judge(monkeypatch, lambda t: "UNSAFE")
    kw = dict(keep_label="UNSAFE", out_path=out, log=_Log(), limit=5, echo=lambda s: None)
    r1 = step00.prefilter(None, _ecfg(), x0s, **kw)
    assert r1.stop.reason == "limit" and r1.n_judged == 5 and len(seen) == 5
    assert not r1.complete
    r2 = step00.prefilter(None, _ecfg(), x0s, **kw)
    assert r2.n_reused == 5 and r2.n_judged == 5 and len(seen) == 10  # the next five only
    r3 = step00.prefilter(None, _ecfg(), x0s, **{**kw, "limit": None})
    assert r3.complete and r3.n_reused == 10 and r3.n_judged == 10 and len(seen) == 20


def test_prefilter_rows_name_their_e(tmp_path: Path, monkeypatch):
    """paper Sec. 3: every prefilter row carries the E identity."""
    _stub_judge(monkeypatch, lambda t: "UNSAFE", fail_texts={"t1"})
    ecfg = _ecfg()
    step00.prefilter(
        None, ecfg, _x0s(3), keep_label="UNSAFE", out_path=tmp_path / "p.jsonl", log=_Log()
    )
    rows = list(read_jsonl(tmp_path / "p.jsonl"))
    assert len(rows) == 3  # the failure too
    for row in rows:
        assert step00.row_identity(row) == ecfg.identity()


def test_prefilter_retries_a_recorded_e_failure(tmp_path: Path, monkeypatch):
    """A failed E call is not a judgment: a later run retries it; successes are reused."""
    out = tmp_path / "p.jsonl"
    seen = _stub_judge(monkeypatch, lambda t: "UNSAFE", fail_texts={"t0"})
    r1 = step00.prefilter(None, _ecfg(), _x0s(2), keep_label="UNSAFE", out_path=out, log=_Log())
    assert len(r1.e_failed) == 1 and r1.complete  # complete: every x0 went through E
    _stub_judge(monkeypatch, lambda t: "UNSAFE")
    r2 = step00.prefilter(None, _ecfg(), _x0s(2), keep_label="UNSAFE", out_path=out, log=_Log())
    assert r2.n_reused == 1 and r2.n_judged == 1 and not r2.e_failed and len(r2.kept) == 2
    assert seen == ["t0", "t1"]


def test_prefilter_stops_at_the_budget(tmp_path: Path, monkeypatch):
    from rafr.llm.ledger import Ledger, Usage

    led = Ledger(tmp_path / "ledger.jsonl")
    spend = step00.Spend(led, step00.PRICES["claude-opus-5"], "evaluator")
    # 1M input + 0 output per call = $5.00 a call at list price
    _stub_judge(monkeypatch, lambda t: "UNSAFE", ledger=led, usage=Usage(1_000_000, 0))
    r = step00.prefilter(
        None,
        _ecfg(),
        _x0s(50),
        keep_label="UNSAFE",
        out_path=tmp_path / "p.jsonl",
        log=_Log(),
        spend=spend,
        budget_usd=12.0,
        echo=lambda s: None,
    )
    assert r.stop.reason == "budget" and r.n_judged == 3  # stops before the 4th call
    assert r.spend_usd == pytest.approx(15.0)  # the call that crossed it was already paid for


def test_prefilter_stops_after_consecutive_e_failures(tmp_path: Path, monkeypatch):
    x0s = _x0s(50)
    seen = _stub_judge(monkeypatch, lambda t: "UNSAFE", fail_texts={r["text"] for r in x0s})
    r = step00.prefilter(
        None,
        _ecfg(),
        x0s,
        keep_label="UNSAFE",
        out_path=tmp_path / "p.jsonl",
        log=_Log(),
        max_consecutive_e_failures=4,
        echo=lambda s: None,
    )
    assert r.stop.reason == "e_failures" and len(seen) == 4 and len(r.e_failed) == 4
    assert r.stop.is_error and "outage" in r.stop.detail


def test_prefilter_checkpoints_the_manifest_while_it_runs(tmp_path: Path, monkeypatch):
    _stub_judge(monkeypatch, lambda t: "UNSAFE")
    saved: list[int] = []
    r = step00.prefilter(
        None,
        _ecfg(),
        _x0s(25),
        keep_label="UNSAFE",
        out_path=tmp_path / "p.jsonl",
        log=_Log(),
        progress_every=5,
        checkpoint_every=10,
        on_checkpoint=lambda res: saved.append(res.n_processed),
        echo=lambda s: None,
    )
    assert saved == [10, 20] and r.n_processed == 25


def _formal_step00(monkeypatch, *extra: str, responder=None) -> int:
    """A formal step00 run (no --fake) whose E is the fake provider: real paths, no network."""
    from rafr.core.runtime import build_client
    from rafr.llm.providers.fake import FakeProvider

    fp = FakeProvider(responder=responder) if responder else FakeProvider()
    monkeypatch.setattr(
        step00,
        "build_client",
        lambda models_cfg, run_dir, fake: build_client(
            models_cfg, run_dir=run_dir, providers={"openai": fp, "anthropic": fp, "fake": fp}
        ),
    )
    return step00.main(["--skip-download", "--raw-dir", str(FIXTURES), "--budget", "5", *extra])


def test_step00_refuses_a_live_run_without_a_budget(sandbox: Path):
    """--budget is the only ceiling on what the prefilter can spend, so it is required."""
    assert step00.main(["--skip-download", "--raw-dir", str(FIXTURES)]) == 2
    assert not list(paths.RUNS.glob("step00_*"))  # refused before anything was written


def test_step00_writes_its_output_only_when_every_x0_is_judged(sandbox: Path, monkeypatch):
    """step04 builds a state for every kept x0, so x0_prefiltered.jsonl must not exist while
    the prefilter is partial; the dry run (--limit) prints the price and the projection."""
    assert _formal_step00(monkeypatch, "--limit", "4") == 0
    out = paths.DATA_POOLS / "x0_prefiltered.jsonl"
    m = Manifest.load()
    assert not out.exists() and m.prefilter["complete"] is False
    assert m.prefilter["n_judged_this_run"] == 4
    cost = m.prefilter["last_run_cost"]
    assert cost["items_bought"] == 4 and cost["items_total"] == 9 and cost["items_left"] == 5
    assert _formal_step00(monkeypatch) == 0
    m = Manifest.load()
    assert out.exists() and m.prefilter["complete"] is True
    assert m.prefilter["n_reused"] == 4 and m.prefilter["n_judged_this_run"] == 5


def test_step00_refuses_a_prefilter_written_by_another_e(sandbox: Path, monkeypatch):
    """(paper Sec. 3): judgments of a different E are never reused or mixed ."""
    assert _formal_step00(monkeypatch, "--limit", "3") == 0
    p = paths.DATA_POOLS / "prefilter.jsonl"
    rows = list(read_jsonl(p))
    rows[0]["e_prompt_sha256"] = "0" * 64  # as if judged with another prompt
    p.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    before = p.read_bytes()
    assert _formal_step00(monkeypatch) == 5
    assert p.read_bytes() == before  # refused before a single call


def test_step00_pins_the_split_plan_and_refuses_a_changed_one(sandbox: Path, monkeypatch):
    assert _formal_step00(monkeypatch, "--limit", "1") == 0
    plan = Manifest.load().splits["plan"]
    assert plan == S.split_plan(load_config())
    cfg = load_config()
    cfg["data"]["splits"]["sizes"]["d_exp1"] = 299
    monkeypatch.setattr(step00, "load_config", lambda: cfg)
    assert _formal_step00(monkeypatch, "--limit", "1") == 5
    assert Manifest.load().splits["plan"] == plan


# ---------------------------------------------------------------------------------------
# step04: one state per prefiltered x0, before any split
# ---------------------------------------------------------------------------------------


def _prepare_pool(n: int = 6, *, complete: bool = True) -> list[str]:
    """x0_prefiltered.jsonl with n x0, and a manifest saying the prefilter is complete."""
    paths.DATA_POOLS.mkdir(parents=True, exist_ok=True)
    ids = [f"x0-{i:03d}" for i in range(n)]
    with (paths.DATA_POOLS / "x0_prefiltered.jsonl").open("w", encoding="utf-8") as fh:
        for i in ids:
            fh.write(json.dumps({"x0_id": i, "text": f"you are such a {i} idiot"}) + "\n")
    m = Manifest.load()
    m.update_section("prefilter", {"complete": complete, "e_identity": _ecfg().identity()})
    m.save()
    return ids


def _fake_client(module, monkeypatch, responder=None):
    """Point a step at the fake provider; returns the provider so calls can be counted."""
    from rafr.core.runtime import build_client
    from rafr.llm.providers.fake import FakeProvider

    fp = FakeProvider(responder=responder) if responder else FakeProvider()
    monkeypatch.setattr(
        module,
        "build_client",
        lambda models_cfg, run_dir, fake: build_client(
            models_cfg, run_dir=run_dir, providers={"openai": fp, "anthropic": fp, "fake": fp}
        ),
    )
    return fp


def _states() -> list[dict]:
    p = step04.states_path()
    return list(read_jsonl(p)) if p.exists() else []


def test_step04_refuses_a_run_without_a_budget(sandbox: Path):
    _prepare_pool()
    assert step04.main([]) == 2
    assert not list(paths.RUNS.glob("step04_*"))  # refused before anything was written


def test_step04_refuses_a_model_it_cannot_price(sandbox: Path, monkeypatch):
    _prepare_pool()
    monkeypatch.setattr(step04, "prices_for", lambda model: None)
    assert step04.main(["--budget", "5"]) == 2


def test_step04_refuses_while_the_prefilter_is_partial(sandbox: Path, monkeypatch):
    _prepare_pool(complete=False)
    fp = _fake_client(step04, monkeypatch)
    assert step04.main(["--budget", "5"]) == 2 and not fp.calls and not _states()


def test_step04_builds_one_state_per_prefiltered_x0_and_resumes_for_free(
    sandbox: Path, monkeypatch
):
    ids = _prepare_pool(6)
    fp = _fake_client(step04, monkeypatch)
    assert step04.main(["--budget", "5"]) == 0
    states = _states()
    assert [r["x0_id"] for r in states] == ids and all(r["split"] is None for r in states)
    assert len(fp.calls) == 6
    m = Manifest.load()
    assert m.states["n_states"] == 6 and m.states["complete"] is True
    assert m.states["run"]["n_generated"] == 6
    assert step04.main(["--budget", "5"]) == 0
    assert len(fp.calls) == 6 and len(_states()) == 6  # not one more call, not one more row
    assert Manifest.load().states["run"]["n_reused_states"] == 6


def test_step04_records_the_unchanged_g_fixed_config_hash(sandbox: Path, monkeypatch):
    """G_fixed is rebuilt exactly as in round 1. 3750a3fc... is the hash of the round-1
    inputs (configs of runs/step04_20260920T*, prompt and refusal list at 58e1876 / ede6427)."""
    _prepare_pool(2)
    _fake_client(step04, monkeypatch)
    assert step04.main(["--budget", "5"]) == 0
    h = Manifest.load().states["g_fixed_config_hash"]
    assert h == "3750a3fcd12ec485af8135289eded1b1710b181abc2a778d7fb2c6e3590ccab5"


def test_step04_limit_stops_and_the_next_run_continues(sandbox: Path, monkeypatch):
    ids = _prepare_pool(6)
    fp = _fake_client(step04, monkeypatch)
    assert step04.main(["--budget", "5", "--limit", "2"]) == 0
    assert len(_states()) == 2 and len(fp.calls) == 2
    m = Manifest.load()
    assert m.states["run"]["stop_reason"] == "limit" and m.states["complete"] is False
    assert m.states["last_run_cost"]["items_left"] == 4
    assert step04.main(["--budget", "5"]) == 0
    assert [r["x0_id"] for r in _states()] == ids and len(fp.calls) == 6


def test_step04_stops_at_the_budget(sandbox: Path, monkeypatch):
    from rafr.llm.ledger import Usage
    from rafr.llm.providers.fake import FakeReply, default_responder

    # 1M input tokens per call = $2.00 a call at the gpt-5.6-terra list price
    def responder(req):
        return FakeReply(text=default_responder(req), usage=Usage(1_000_000, 0))

    _prepare_pool(10)
    fp = _fake_client(step04, monkeypatch, responder=responder)
    assert step04.main(["--budget", "5", "--workers", "1"]) == 0  # serial: an exact stop
    assert len(fp.calls) == 3 and len(_states()) == 3  # stops before the 4th ($6 > $5)
    run = Manifest.load().states["run"]
    assert run["stop_reason"] == "budget" and run["spend_usd"] == pytest.approx(6.0)


def test_step04_stops_after_consecutive_failures_and_does_not_retry_dropped_x0(
    sandbox: Path, monkeypatch
):
    from rafr.llm.providers.fake import FakeReply

    _prepare_pool(20)
    fp = _fake_client(
        step04,
        monkeypatch,
        responder=lambda r: FakeReply(text=None, refused=True, stop_reason="refusal"),
    )
    # serial: an exact stop (with workers the x0 in flight still land, see the tests below)
    assert step04.main(["--budget", "5", "--max-consecutive-failures", "3", "--workers", "1"]) == 4
    assert len(list(read_jsonl(step04.dropped_path()))) == 3 and not _states()
    assert len(fp.calls) == 3 * 3  # max_attempts per x0, three x0 in a row
    m = Manifest.load()
    assert m.states["run"]["stop_reason"] == "generation_failures"
    assert m.states["attempts"]["invalid_by_reason"] == {"refusal": 9}
    fp2 = _fake_client(step04, monkeypatch)
    assert step04.main(["--budget", "5", "--limit", "2"]) == 0
    assert [r["x0_id"] for r in _states()] == ["x0-003", "x0-004"]  # continues past the drops
    assert len(fp2.calls) == 2
    assert Manifest.load().states["run"]["n_skipped_dropped"] == 3


def test_step04_progress_and_checkpoints_report_the_run(sandbox: Path, monkeypatch):
    from rafr.core.runtime import Spend, build_client
    from rafr.llm.providers.fake import FakeProvider

    fp = FakeProvider()
    cfgs = load_config()
    cfg = step04.g_fixed.build_config(
        cfgs["models"], cfgs["generation"], allow_tbd=False, base_seed=7
    )
    client = build_client(
        cfgs["models"], run_dir=sandbox / "runs" / "x", providers={"openai": fp, "fake": fp}
    )
    work = [(f"x0-{i:03d}", f"text {i}") for i in range(12)]
    lines: list[str] = []
    saved: list[int] = []
    res = step04.build_states(
        client,
        cfg,
        work,
        write_state=lambda st: None,
        write_dropped=lambda row: None,
        spend=Spend(client.ledger, step04.PRICES["gpt-5.6-terra"], "g_fixed"),
        progress_every=5,
        checkpoint_every=10,
        on_checkpoint=lambda r: saved.append(r.n_done),
        echo=lines.append,
    )
    assert res.n_generated == 12 and res.stop.reason == "complete" and saved == [10]
    progress = [x for x in lines if x.startswith(("[step04] 5/12", "[step04] 10/12"))]
    assert len(progress) == 2
    for needle in ("states", "invalid", "regen", "dropped", "$", "s/x0", "ETA"):
        assert needle in progress[0], needle


def test_step04_regenerates_everything_when_the_state_file_is_deleted(sandbox: Path, monkeypatch):
    """The work list comes from the files, never from the manifest."""
    _prepare_pool(6)
    fp = _fake_client(step04, monkeypatch)
    assert step04.main(["--budget", "5"]) == 0 and len(fp.calls) == 6
    step04.states_path().unlink()
    fp2 = _fake_client(step04, monkeypatch)
    assert step04.main(["--budget", "5"]) == 0
    assert len(_states()) == 6
    # and it cost nothing: the LLM cache still held every response
    assert fp2.calls == [] and Manifest.load().states["run"]["spend_usd"] == 0.0


# ---------------------------------------------------------------------------------------
# step05: E judges every state (= y*), then the split
# ---------------------------------------------------------------------------------------

SMALL_SIZES = {"d_fit": 5, "d_gate": 5, "d_gate_reserve": 5, "d_exp1": 4, "unlabeled_pool": "rest"}


def _prepare_states(monkeypatch, n: int = 40, *, sizes=None) -> list[str]:
    """states.jsonl with n states (every third one insulting), a complete prefilter by the
    configured E, small split sizes."""
    from rafr.contracts.schemas import State

    cfg = load_config()
    cfg["data"]["splits"]["sizes"] = dict(sizes or SMALL_SIZES)
    monkeypatch.setattr(step05, "load_config", lambda: cfg)
    paths.DATA_POOLS.mkdir(parents=True, exist_ok=True)
    ids = [f"x0-{i:03d}" for i in range(n)]
    with (paths.DATA_POOLS / "states.jsonl").open("w", encoding="utf-8") as fh:
        for i, x0 in enumerate(ids):
            text = f"state {i} you idiot" if i % 3 == 0 else f"state {i} is fine"
            fh.write(
                State(state_id=f"{x0}-g1", x0_id=x0, x0=f"orig {i}", text=text).model_dump_json()
                + "\n"
            )
    m = Manifest.load()
    m.update_section("prefilter", {"complete": True, "e_identity": _ecfg().identity()})
    m.update_section("states", {"complete": True})
    m.save()
    return ids


def _fake_e(monkeypatch, fail_texts=()):
    from rafr.llm.providers.fake import FakeReply

    def responder(req):
        text = content_text(req.messages[0]["content"])
        if any(t in text for t in fail_texts):
            return FakeReply(text=None, refused=True, stop_reason="refusal")
        return json.dumps({"label": "UNSAFE" if "idiot" in text else "SAFE", "reason": "r"})

    return _fake_client(step05, monkeypatch, responder=responder)


def test_step05_judges_every_state_then_splits(sandbox: Path, monkeypatch):
    ids = _prepare_states(monkeypatch)
    fp = _fake_e(monkeypatch)
    # the dry run: a limited batch, then the rest
    assert step05.main(["judge", "--budget", "5", "--limit", "4"]) == 0
    assert len(fp.calls) == 4 and Manifest.load().labels["complete"] is False
    assert step05.main(["split"]) == 2  # no split before every state has its y*
    assert step05.main(["judge", "--budget", "5"]) == 0
    assert len(fp.calls) == 40  # one E call per state, the first four not re-bought
    lab = Manifest.load().labels
    assert lab["complete"] is True and lab["n_states"] == 40 and lab["n_unsafe"] == 14
    assert lab["sha256"] == step05.sha256_file(L.y_star_path())
    rows = L.load_judgment_rows()
    assert all(r.record.e_prompt_sha256 == _ecfg().prompt_sha256 for r in rows.values())
    assert step05.main(["judge", "--budget", "5"]) == 0 and len(fp.calls) == 40  # frozen

    assert step05.main(["split"]) == 0
    m = Manifest.load()
    splits = S.load_all_splits(paths.DATA_SPLITS)
    assert [len(splits[sp]) for sp in S.SPLIT_ORDER] == [5, 5, 5, 4, 21]
    assert sorted(x for v in splits.values() for x in v) == ids
    unsafe = {r.x0_id for r in rows.values() if r.label is SafetyLabel.UNSAFE}
    assert set(splits[Split.D_EXP1]) <= unsafe
    assert m.splits["y_star_sha256"] == lab["sha256"]
    # the report: D_fit's E-UNSAFE count and share, every other split's size only
    rep = m.splits["report"]
    assert rep["d_fit"]["n"] == 5 and set(rep["d_fit"]) == {"n", "n_unsafe", "unsafe_rate"}
    for name, n in (("d_gate", 5), ("d_gate_reserve", 5), ("d_exp1", 4), ("unlabeled_pool", 21)):
        assert rep[name] == {"n": n}  # sizes only (2026-09-23)
    for sp in S.SPLIT_ORDER:  # per-split state files carry their split
        rows_sp = list(read_jsonl(step05.split_states_path(sp)))
        assert [r["x0_id"] for r in rows_sp] == splits[sp]
        assert {r["split"] for r in rows_sp} == {sp.value}
    # the Exp 1 outcome is D_Exp1's y*, readable through the gate
    ys = L.read_labels(Split.D_EXP1, L.LabelPurpose.EXP1_OUTCOME)
    assert {y.label for y in ys} == {SafetyLabel.UNSAFE}
    # splitting again reproduces the same split and rewrites nothing
    before = {p: p.read_bytes() for p in paths.DATA_SPLITS.glob("*.jsonl")}
    assert step05.main(["split"]) == 0
    assert {p: p.read_bytes() for p in paths.DATA_SPLITS.glob("*.jsonl")} == before


def test_step05_drops_a_state_and_its_x0_when_e_keeps_failing(sandbox: Path, monkeypatch):
    """paper Sec. 3: after 2 retries the state and its x0 are dropped, before the split."""
    _prepare_states(monkeypatch)
    fp = _fake_e(monkeypatch, fail_texts=("state 7 is fine",))
    assert step05.main(["judge", "--budget", "5"]) == 0
    assert len(fp.calls) == 39 + 3  # three attempts on the failing state
    lab = Manifest.load().labels
    assert lab["n_dropped_e_failed"] == 1 and lab["dropped"][0]["x0_id"] == "x0-007"
    assert step05.main(["split"]) == 0
    splits = S.load_all_splits(paths.DATA_SPLITS)
    assert "x0-007" not in {x for v in splits.values() for x in v}
    assert sum(len(v) for v in splits.values()) == 39
    assert Manifest.load().splits["n_dropped_before_split"] == 1


def test_step05_stops_when_too_few_e_unsafe_remain(sandbox: Path, monkeypatch):
    _prepare_states(monkeypatch, sizes={**SMALL_SIZES, "d_exp1": 20})
    _fake_e(monkeypatch)
    assert step05.main(["judge", "--budget", "5"]) == 0
    assert step05.main(["split"]) == 3
    assert not any(S.load_all_splits(paths.DATA_SPLITS).values())  # nothing written


def test_step05_refuses_to_split_a_y_star_file_changed_after_its_hash(sandbox: Path, monkeypatch):
    _prepare_states(monkeypatch)
    _fake_e(monkeypatch)
    assert step05.main(["judge", "--budget", "5"]) == 0
    p = L.y_star_path()
    p.write_text(p.read_text(encoding="utf-8").replace('"SAFE"', '"UNSAFE"', 1), "utf-8")
    assert step05.main(["split"]) == 5


def test_step05_refuses_when_e_differs_from_the_prefilter_e(sandbox: Path, monkeypatch):
    _prepare_states(monkeypatch)
    fp = _fake_e(monkeypatch)
    m = Manifest.load()
    m.prefilter["e_identity"] = {**m.prefilter["e_identity"], "version": "2025-01-01"}
    m.save()
    assert step05.main(["judge", "--budget", "5"]) == 5 and not fp.calls


def test_step05_refuses_a_changed_split_plan(sandbox: Path, monkeypatch):
    _prepare_states(monkeypatch)
    fp = _fake_e(monkeypatch)
    m = Manifest.load()
    m.pin_split_plan({**S.split_plan(step05.load_config()), "seed": 1})
    m.save()
    assert step05.main(["judge", "--budget", "5"]) == 5 and not fp.calls


def test_step05_judge_needs_a_budget_and_a_complete_state_pool(sandbox: Path, monkeypatch):
    _prepare_states(monkeypatch)
    fp = _fake_e(monkeypatch)
    assert step05.main(["judge"]) == 2
    m = Manifest.load()
    m.states["complete"] = False
    m.save()
    assert step05.main(["judge", "--budget", "5"]) == 2 and not fp.calls


# ---------------------------------------------------------------------------------------
# --fake must not be able to touch a formal path (2026-09-20)
# ---------------------------------------------------------------------------------------


def test_a_fake_run_cannot_touch_formal_data(sandbox: Path, monkeypatch):
    """A formal run's prefilter and manifest survive a --fake run byte for byte."""
    assert _formal_step00(monkeypatch) == 0
    watched = [paths.DATA_POOLS / "prefilter.jsonl", paths.DATA_POOLS / "x0_prefiltered.jsonl"]
    before = {p: p.read_bytes() for p in watched}
    formal_manifest = paths.MANIFEST  # captured: the fake run redirects paths.MANIFEST
    manifest_before = formal_manifest.read_bytes()
    assert Manifest.load().run_mode == "formal"
    assert _fake_step00() == 0
    assert {p: p.read_bytes() for p in watched} == before
    assert formal_manifest.read_bytes() == manifest_before
    fake_root = sandbox / "data" / paths.FAKE_ROOT_NAME
    assert (fake_root / "pools" / "prefilter.jsonl").exists()
    assert json.loads((fake_root / "manifest.json").read_text(encoding="utf-8"))["run_mode"] == (
        "fake"
    )
    assert list(fake_root.glob("runs/step00_*"))  # even the run dir is inside the fake root


def test_fake_isolation_is_checked_not_assumed(sandbox: Path):
    """The redirect is one call at the top of a step; this is what notices when it is lost."""
    with pytest.raises(paths.FakeIsolationError):
        paths.assert_fake_isolation(True)  # not redirected, but claiming to be fake
    paths.use_fake_data_root()
    paths.assert_fake_isolation(True)
    assert paths.is_fake_root()
    with pytest.raises(paths.FakeIsolationError):
        paths.assert_fake_isolation(False)  # a formal run pointed at the fake root
    assert paths.use_fake_data_root() == paths.DATA  # idempotent: no data/_fake/_fake


# ---------------------------------------------------------------------------------------
# step00 --workers (2026-09-23): parallel E calls, written in input order
# ---------------------------------------------------------------------------------------


def _scripted_e(*, jitter_s: float = 0.004, transient: dict | None = None):
    """A fake E that answers from the text alone, deterministically: UNSAFE when the index is
    a multiple of 3; an unreadable first attempt when it is a multiple of 5 (one parse retry);
    a refusal on every attempt when it is a multiple of 7 (an E failure). It sleeps a random
    few ms so that parallel calls finish out of order. ``transient`` maps a text to how many
    429s its requests get before an answer (``-1`` = forever)."""
    import random
    import threading
    import time as _time

    from rafr.llm.client import TransientProviderError
    from rafr.llm.providers.fake import FakeReply

    lock = threading.Lock()
    seen_429: dict[str, int] = {}
    rng = random.Random(0)  # noqa: S311 - scheduling noise for a test, not security

    def responder(req):
        text = content_text(req.messages[0]["content"])
        i = int(re.search(r"item (\d+) text", text).group(1))
        with lock:
            delay = rng.random() * jitter_s
            left = (transient or {}).get(i, 0)
            n = seen_429.get(req.cache_key(), 0)
            if left == -1 or n < left:
                seen_429[req.cache_key()] = n + 1
                raise TransientProviderError("rate_limit", "429", retry_after=None, status=429)
        _time.sleep(delay)
        if i % 7 == 0:
            return FakeReply(text=None, refused=True, stop_reason="refusal")
        if i % 5 == 0 and req.attempt == 0:
            return "I think it is fine."
        label = "UNSAFE" if i % 3 == 0 else "SAFE"
        return json.dumps({"label": label, "reason": f"because {i}."})

    return responder


def _run_prefilter(tmp: Path, *, workers: int, n: int = 60, responder=None, net=None, **kw):
    import dataclasses

    from rafr.llm.cache import LLMCache
    from rafr.llm.client import LLMClient
    from rafr.llm.ledger import Ledger
    from rafr.llm.providers.fake import FakeProvider

    tmp.mkdir(parents=True, exist_ok=True)
    ecfg = _ecfg()
    ecfg = dataclasses.replace(ecfg, provider="fake", network=net or dict(ecfg.network))
    fp = FakeProvider(responder=responder or _scripted_e())
    client = LLMClient(
        providers={"fake": fp},
        cache=LLMCache(tmp / "cache"),
        ledger=Ledger(tmp / "ledger.jsonl"),
        sleep=lambda s: None,
    )
    x0s = [{"x0_id": f"x{i:03d}", "text": f"item {i} text"} for i in range(1, n + 1)]
    res = step00.prefilter(
        client,
        ecfg,
        x0s,
        keep_label="UNSAFE",
        out_path=tmp / "prefilter.jsonl",
        log=_Log(),
        workers=workers,
        echo=lambda s: None,
        **kw,
    )
    return res, client, fp


def test_parallel_prefilter_writes_exactly_what_the_serial_run_writes(tmp_path: Path):
    """Same rows, same order, same bytes; same counts, same retries, same failures -- only the
    wall clock differs. Completion order is scrambled on purpose (random per-call delays)."""
    serial, _, fp1 = _run_prefilter(tmp_path / "serial", workers=1)
    parallel, _, fp4 = _run_prefilter(tmp_path / "parallel", workers=4)
    a = (tmp_path / "serial" / "prefilter.jsonl").read_bytes()
    b = (tmp_path / "parallel" / "prefilter.jsonl").read_bytes()
    assert a == b and len(a.splitlines()) == 60
    for attr in ("n_judged", "n_processed", "n_parse_retries", "failed_attempts", "e_failed"):
        assert getattr(serial, attr) == getattr(parallel, attr), attr
    assert [r["x0_id"] for r in serial.kept] == [r["x0_id"] for r in parallel.kept]
    assert serial.complete and parallel.complete and parallel.workers == 4
    # 8 E failures (7, 14, ..., 56), each 3 refusals; parse retries: 2 per failure + 1 per
    # multiple of 5 that is not a multiple of 7
    assert len(serial.e_failed) == 8
    assert serial.n_parse_retries == 8 * 2 + sum(1 for i in range(1, 61) if i % 5 == 0 and i % 7)
    # the very same requests went out (cache keys), just not in the same order
    assert sorted(c.cache_key() for c in fp1.calls) == sorted(c.cache_key() for c in fp4.calls)


def test_step00_workers_4_and_workers_1_leave_identical_files(sandbox: Path):
    import shutil

    root = sandbox / "data" / paths.FAKE_ROOT_NAME
    out = {}
    for w in ("1", "4"):
        if root.exists():
            shutil.rmtree(root)  # fresh pool, fresh LLM cache: every call is made again
        assert _fake_step00("--workers", w) == 0
        out[w] = {
            name: (root / "pools" / name).read_bytes()
            for name in ("prefilter.jsonl", "x0_prefiltered.jsonl")
        }
        assert Manifest.load(root / "manifest.json").prefilter["workers"] == int(w)
    assert out["1"] == out["4"]


def test_rate_limits_are_retried_and_never_count_as_e_failures(tmp_path: Path):
    """Every request of every x0 is rate limited twice before it gets through. The scripted
    E's real failures (multiples of 7) are never adjacent, so a consecutive-failure limit of 2
    only fires if 429s were counted -- every x0 would count then. It runs to the end, the rows
    match a run that saw no 429 at all, and every 429 is counted."""
    clean, _, _ = _run_prefilter(tmp_path / "clean", workers=4)
    limited = _scripted_e(transient=dict.fromkeys(range(1, 61), 2))
    res, client, fp = _run_prefilter(
        tmp_path / "limited", workers=4, responder=limited, max_consecutive_e_failures=2
    )
    assert res.complete and res.stop.reason == "complete"
    assert (tmp_path / "limited" / "prefilter.jsonl").read_bytes() == (
        tmp_path / "clean" / "prefilter.jsonl"
    ).read_bytes()
    # every request that got through was refused twice first: 429s = 2/3 of all sends
    assert client.transient_stats["rate_limit"] * 3 == len(fp.calls) * 2
    assert res.transient["retries"] == client.transient_stats["rate_limit"]
    assert res.transient["exhausted"] == 0
    assert res.n_parse_retries == clean.n_parse_retries  # 429s are not parse retries
    assert len(res.e_failed) == len(clean.e_failed)  # nor E failures


def test_a_429_streak_does_not_trigger_the_consecutive_failure_stop(tmp_path: Path):
    """Heavy rate limiting on the first x0 only, and a failure limit of 1: no stop."""
    responder = _scripted_e(transient={1: 5, 2: 5, 3: 5})
    res, client, _ = _run_prefilter(
        tmp_path, workers=4, n=6, responder=responder, max_consecutive_e_failures=1
    )
    assert res.stop.reason == "complete" and res.n_processed == 6
    assert client.transient_stats["rate_limit"] == 15 and not res.e_failed


def test_exhausted_transient_retries_stop_the_run_but_are_no_e_failure(tmp_path: Path):
    net = {"transient_retries": 2, "backoff_base_s": 1, "backoff_cap_s": 1}
    responder = _scripted_e(transient={4: -1})
    res, client, _ = _run_prefilter(tmp_path, workers=1, n=8, responder=responder, net=net)
    assert res.stop.reason == "transient_errors" and res.stop.is_error
    assert not res.e_failed and res.n_transient_gave_up == 1
    rows = [
        json.loads(line)
        for line in (tmp_path / "prefilter.jsonl").read_text("utf-8").split("\n")
        if line
    ]
    assert "x004" not in {r["x0_id"] for r in rows}  # no row: judged on the next run
    assert [r["x0_id"] for r in rows] == ["x001", "x002", "x003"]
    # the next run, provider healthy again, picks x004 up and finishes
    res2, _, _ = _run_prefilter(tmp_path, workers=4, n=8)
    assert res2.complete and res2.n_reused == 3 and res2.n_judged == 5


def test_limit_and_budget_still_apply_with_workers(tmp_path: Path):
    res, _, _ = _run_prefilter(tmp_path / "limit", workers=4, limit=10)
    assert res.stop.reason == "limit" and res.n_judged == 10
    assert len((tmp_path / "limit" / "prefilter.jsonl").read_text("utf-8").splitlines()) == 10

    import dataclasses
    import re

    from rafr.llm.cache import LLMCache
    from rafr.llm.client import LLMClient
    from rafr.llm.ledger import Ledger, Usage
    from rafr.llm.providers.fake import FakeProvider, FakeReply

    def pricey(req):  # 1M input tokens = $5.00 a call at the claude-opus-5 list price
        i = int(re.search(r"item (\d+) text", content_text(req.messages[0]["content"])).group(1))
        body = json.dumps({"label": "UNSAFE" if i % 3 == 0 else "SAFE", "reason": "r."})
        return FakeReply(text=body, usage=Usage(1_000_000, 0))

    tmp = tmp_path / "budget"
    led = Ledger(tmp / "ledger.jsonl")
    client = LLMClient(
        providers={"fake": FakeProvider(responder=pricey)},
        cache=LLMCache(tmp / "cache"),
        ledger=led,
        sleep=lambda s: None,
    )
    r = step00.prefilter(
        client,
        dataclasses.replace(_ecfg(), provider="fake"),
        [{"x0_id": f"x{i:03d}", "text": f"item {i} text"} for i in range(1, 41)],
        keep_label="UNSAFE",
        out_path=tmp / "prefilter.jsonl",
        log=_Log(),
        spend=step00.Spend(led, step00.PRICES["claude-opus-5"], "evaluator"),
        budget_usd=12.0,
        workers=4,
        echo=lambda s: None,
    )
    # checked before every submission: at most workers - 1 calls beyond the serial stop
    assert r.stop.reason == "budget" and 3 <= r.n_judged <= 3 + 3
    assert len(led.rows()) == r.n_judged  # every call bought was written, none thrown away
    assert len((tmp / "prefilter.jsonl").read_text("utf-8").splitlines()) == r.n_judged


# ---------------------------------------------------------------------------------------
# step04 / step05 --workers (2026-09-23): as step00, written in input order
# ---------------------------------------------------------------------------------------


@pytest.fixture
def frozen_clock(monkeypatch):
    """``created_at`` (State, GenerationAttempt, EvalRecord) is wall-clock metadata that two
    serial runs differ in as well; frozen here so the byte comparisons cover everything else."""
    from datetime import datetime as _datetime

    from rafr.contracts import schemas

    class _Frozen(_datetime):
        @classmethod
        def now(cls, tz=None):
            return _datetime(2026, 9, 23, 12, 0, 0, tzinfo=tz)

    monkeypatch.setattr(schemas, "datetime", _Frozen)


def _jitter(jitter_s: float):
    """A seeded few-ms sleep per call, so that parallel calls finish out of order."""
    import random
    import threading
    import time as _time

    lock = threading.Lock()
    rng = random.Random(0)  # noqa: S311 - scheduling noise for a test, not security

    def sleep() -> None:
        with lock:
            delay = rng.random() * jitter_s
        _time.sleep(delay)

    return sleep


def _scripted_g(*, jitter_s: float = 0.004, transient: dict | None = None):
    """A fake G_fixed that answers from the x0 alone, deterministically: the x0 itself (invalid,
    identical -> one regeneration) on the first attempt when the index is a multiple of 5; a
    refusal on every attempt when it is a multiple of 7 (the x0 is dropped); a rewrite
    otherwise. ``transient`` maps an index to how many 429s each of its requests gets before an
    answer (``-1`` = forever)."""
    import threading

    from rafr.llm.client import TransientProviderError
    from rafr.llm.providers.fake import FakeReply

    lock = threading.Lock()
    seen_429: dict[str, int] = {}
    sleep = _jitter(jitter_s)

    def responder(req):
        text = content_text(req.messages[0]["content"])
        i = int(re.search(r"item (\d+) text", text).group(1))
        with lock:
            left = (transient or {}).get(i, 0)
            n = seen_429.get(req.cache_key(), 0)
            if left == -1 or n < left:
                seen_429[req.cache_key()] = n + 1
                raise TransientProviderError("rate_limit", "429", retry_after=None, status=429)
        sleep()
        if i % 7 == 0:
            return FakeReply(text=None, refused=True, stop_reason="refusal")
        if i % 5 == 0 and req.attempt == 0:
            return json.dumps({"revised_text": f"item {i} text"})
        return json.dumps({"revised_text": f"a calmer sentence number {i}"})

    return responder


def _offline_client(tmp: Path, provider: str, responder):
    from rafr.llm.cache import LLMCache
    from rafr.llm.client import LLMClient
    from rafr.llm.ledger import Ledger
    from rafr.llm.providers.fake import FakeProvider

    fp = FakeProvider(responder=responder)
    client = LLMClient(
        providers={provider: fp},
        cache=LLMCache(tmp / "cache"),
        ledger=Ledger(tmp / "ledger.jsonl"),
        sleep=lambda s: None,  # the backoff is counted, not slept
    )
    return client, fp


def _run_build(
    tmp: Path, *, workers: int, n: int = 60, responder=None, net=None, price=False, **kw
):
    """build_states on x0 ``item 1 text`` .. ``item n text``, files in ``tmp``; like main(),
    the x0 that already have a state or a drop record there are not in the work list."""
    import dataclasses

    from rafr.logging.writer import JsonlWriter

    tmp.mkdir(parents=True, exist_ok=True)
    cfgs = load_config()
    cfg = step04.g_fixed.build_config(
        cfgs["models"], cfgs["generation"], allow_tbd=False, base_seed=7
    )
    if net is not None:
        cfg = dataclasses.replace(cfg, network=net)
    client, fp = _offline_client(tmp, cfg.provider, responder or _scripted_g())
    done = {
        r["x0_id"] for name in ("states.jsonl", "dropped.jsonl") for r in read_jsonl(tmp / name)
    }
    work = [(f"x{i:03d}", f"item {i} text") for i in range(1, n + 1)]
    work = [(i, t) for i, t in work if i not in done]
    spend = (
        step04.Spend(client.ledger, step04.PRICES["gpt-5.6-terra"], "g_fixed") if price else None
    )
    states_fh = (tmp / "states.jsonl").open("a", encoding="utf-8")
    dropped = JsonlWriter(tmp / "dropped.jsonl")
    attempts = JsonlWriter(tmp / "attempts.jsonl")

    def write_state(st) -> None:
        states_fh.write(st.model_dump_json() + "\n")
        states_fh.flush()

    try:
        res = step04.build_states(
            client,
            cfg,
            work,
            write_state=write_state,
            write_dropped=dropped.write,
            attempts_log=attempts,
            spend=spend,
            workers=workers,
            echo=lambda s: None,
            **kw,
        )
    finally:
        states_fh.close()
        dropped.close()
        attempts.close()
    return res, client, fp


def _lines(p: Path) -> list[str]:
    return p.read_text("utf-8").splitlines() if p.exists() else []


STEP04_FILES = ("states.jsonl", "dropped.jsonl", "attempts.jsonl")


def test_parallel_build_states_writes_exactly_what_the_serial_run_writes(
    tmp_path: Path, frozen_clock
):
    """States, drop records and the attempt log: same rows, same order, same bytes; same
    counts and the same requests -- only the wall clock differs. Completion order is scrambled
    on purpose (random per-call delays)."""
    serial, _, fp1 = _run_build(tmp_path / "serial", workers=1)
    parallel, _, fp4 = _run_build(tmp_path / "parallel", workers=4)
    for name in STEP04_FILES:
        a = (tmp_path / "serial" / name).read_bytes()
        assert a == (tmp_path / "parallel" / name).read_bytes(), name
    # 8 x0 dropped (7, 14, ..., 56) after 3 refusals each; one regeneration for every
    # multiple of 5 that is not a multiple of 7
    n_regen = sum(1 for i in range(1, 61) if i % 5 == 0 and i % 7)
    assert len(_lines(tmp_path / "serial" / "states.jsonl")) == 52
    assert len(_lines(tmp_path / "serial" / "dropped.jsonl")) == 8
    assert len(_lines(tmp_path / "serial" / "attempts.jsonl")) == 52 + n_regen + 8 * 3
    for attr in (
        "n_done",
        "n_generated",
        "n_dropped",
        "n_attempts",
        "n_invalid",
        "invalid_by_reason",
    ):
        assert getattr(serial, attr) == getattr(parallel, attr), attr
    assert serial.stop.reason == parallel.stop.reason == "complete" and parallel.workers == 4
    assert sorted(c.cache_key() for c in fp1.calls) == sorted(c.cache_key() for c in fp4.calls)


def test_step04_workers_4_and_workers_1_leave_identical_files(
    sandbox: Path, monkeypatch, frozen_clock
):
    import shutil

    ids = _prepare_pool(21)
    with (paths.DATA_POOLS / "x0_prefiltered.jsonl").open("w", encoding="utf-8") as fh:
        for k, x0_id in enumerate(ids, 1):
            fh.write(json.dumps({"x0_id": x0_id, "text": f"item {k} text"}) + "\n")
    files = (
        step04.states_path(),
        step04.dropped_path(),
        paths.DATA_POOLS / "generation_attempts.jsonl",
    )
    out = {}
    for w in ("1", "4"):
        for p in files:
            p.unlink(missing_ok=True)
        shutil.rmtree(paths.DATA_CACHE, ignore_errors=True)  # every call is made again
        fp = _fake_client(step04, monkeypatch, responder=_scripted_g())
        assert step04.main(["--budget", "5", "--workers", w]) == 0
        out[w] = {p.name: p.read_bytes() for p in files}
        m = Manifest.load()
        assert m.states["run"]["workers"] == int(w) and m.states["complete"] is True
        # 14 x0 in one call, 4 in two (5, 10, 15, 20: regenerated), 3 in three (7, 14, 21: dropped)
        assert len(fp.calls) == 14 + 4 * 2 + 3 * 3
    assert out["1"] == out["4"]
    assert len(out["4"]["states.jsonl"].splitlines()) == 18


def test_step04_rate_limits_are_retried_and_never_count_as_invalid_or_as_failures(
    tmp_path: Path, frozen_clock
):
    """Every request of every x0 is rate limited twice before it gets through. The scripted
    drops (multiples of 7) are never adjacent, so a consecutive-failure limit of 2 only fires if
    429s were counted -- every x0 would count then. It runs to the end, the files match a run
    that saw no 429 at all, and every 429 is counted."""
    clean, _, _ = _run_build(tmp_path / "clean", workers=4)
    limited = _scripted_g(transient=dict.fromkeys(range(1, 61), 2))
    res, client, fp = _run_build(
        tmp_path / "limited", workers=4, responder=limited, max_consecutive_failures=2
    )
    assert res.stop.reason == "complete" and res.n_done == 60
    for name in STEP04_FILES:
        a = (tmp_path / "clean" / name).read_bytes()
        assert a == (tmp_path / "limited" / name).read_bytes(), name
    assert client.transient_stats["rate_limit"] * 3 == len(fp.calls) * 2
    assert res.transient["retries"] == client.transient_stats["rate_limit"]
    assert res.transient["exhausted"] == 0 and res.n_transient_gave_up == 0
    assert (res.n_attempts, res.n_invalid, res.n_dropped) == (
        clean.n_attempts,
        clean.n_invalid,
        clean.n_dropped,
    )  # 429s are neither attempts, nor invalid outputs, nor drops


def test_step04_a_429_streak_does_not_trigger_the_consecutive_failure_stop(tmp_path: Path):
    """Heavy rate limiting on the first x0 only, and a failure limit of 1: no stop."""
    responder = _scripted_g(transient={1: 5, 2: 5, 3: 5})
    res, client, _ = _run_build(
        tmp_path, workers=4, n=6, responder=responder, max_consecutive_failures=1
    )
    assert res.stop.reason == "complete" and res.n_done == 6 and res.n_generated == 6
    assert client.transient_stats["rate_limit"] == 15 and res.n_dropped == 0


def test_step04_exhausted_transient_retries_stop_the_run_but_drop_nothing(
    tmp_path: Path, frozen_clock
):
    net = {"transient_retries": 2, "backoff_base_s": 1, "backoff_cap_s": 1}
    responder = _scripted_g(transient={4: -1})
    res, client, _ = _run_build(tmp_path, workers=1, n=8, responder=responder, net=net)
    assert res.stop.reason == "transient_errors" and res.stop.is_error
    assert res.n_transient_gave_up == 1 and res.n_dropped == 0 and res.n_invalid == 0
    assert client.transient_stats["exhausted"] == 1
    # no state, no drop record, no logged attempt: the next run generates x004
    states = [json.loads(x)["x0_id"] for x in _lines(tmp_path / "states.jsonl")]
    assert states == ["x001", "x002", "x003"]
    assert not _lines(tmp_path / "dropped.jsonl")
    assert len(_lines(tmp_path / "attempts.jsonl")) == 3
    # the next run, provider healthy again, picks x004 up and finishes
    res2, _, _ = _run_build(tmp_path, workers=4, n=8)
    assert res2.stop.reason == "complete" and res2.n_todo == 5 and res2.n_done == 5
    states = [json.loads(x)["x0_id"] for x in _lines(tmp_path / "states.jsonl")]
    assert states == ["x001", "x002", "x003", "x004", "x005", "x006", "x008"]  # x007 dropped


def test_step04_limit_budget_and_failure_stop_still_apply_with_workers(tmp_path: Path):
    from rafr.llm.ledger import Ledger, Usage
    from rafr.llm.providers.fake import FakeReply

    # --limit counts submissions: exact
    res, _, fp = _run_build(tmp_path / "limit", workers=4, limit=10)
    assert res.stop.reason == "limit" and res.n_done == 10
    assert (
        len(_lines(tmp_path / "limit" / "states.jsonl"))
        + len(_lines(tmp_path / "limit" / "dropped.jsonl"))
        == 10
    )

    # --budget: 1M input tokens = $2.00 a call at the gpt-5.6-terra list price; checked before
    # every submission, so at most workers - 1 x0 beyond the serial stop (3 x0 = $6 > $5)
    def pricey(req):
        return FakeReply(text=json.dumps({"revised_text": "calm"}), usage=Usage(1_000_000, 0))

    res, _, fp = _run_build(
        tmp_path / "budget", workers=4, n=40, responder=pricey, price=True, budget_usd=5.0
    )
    assert res.stop.reason == "budget" and 3 <= res.n_done <= 3 + 3
    assert len(Ledger(tmp_path / "budget" / "ledger.jsonl").rows()) == len(fp.calls) == res.n_done
    assert len(_lines(tmp_path / "budget" / "states.jsonl")) == res.n_done  # none thrown away

    # consecutive x0 with nothing valid: the same bound
    def refuse(req):
        return FakeReply(text=None, refused=True, stop_reason="refusal")

    res, _, fp = _run_build(
        tmp_path / "fail", workers=4, n=40, responder=refuse, max_consecutive_failures=3
    )
    assert res.stop.reason == "generation_failures" and 3 <= res.n_dropped <= 3 + 3
    assert len(_lines(tmp_path / "fail" / "dropped.jsonl")) == res.n_dropped
    assert len(_lines(tmp_path / "fail" / "attempts.jsonl")) == len(fp.calls) == 3 * res.n_dropped


def _run_judge(
    tmp: Path, *, workers: int, n: int = 60, responder=None, net=None, price=False, **kw
):
    """judge_states on states ``item 1 text`` .. ``item n text``; y* file in ``tmp``."""
    import dataclasses

    from rafr.contracts.schemas import State

    tmp.mkdir(parents=True, exist_ok=True)
    ecfg = _ecfg()
    ecfg = dataclasses.replace(ecfg, provider="fake", network=net or dict(ecfg.network))
    client, fp = _offline_client(tmp, "fake", responder or _scripted_e())
    states = [
        State(state_id=f"x{i:03d}-g1", x0_id=f"x{i:03d}", x0=f"orig {i}", text=f"item {i} text")
        for i in range(1, n + 1)
    ]
    spend = (
        step05.Spend(client.ledger, step05.PRICES["claude-opus-5"], "evaluator") if price else None
    )
    res = step05.judge_states(
        client,
        ecfg,
        states,
        out_path=tmp / "state_judgments.jsonl",
        spend=spend,
        workers=workers,
        echo=lambda s: None,
        **kw,
    )
    return res, client, fp


def test_parallel_judge_states_writes_exactly_what_the_serial_run_writes(
    tmp_path: Path, frozen_clock
):
    """The y* file: same rows, same order, same bytes; same counts, same retries, same
    failures, the same requests. Completion order is scrambled on purpose."""
    serial, _, fp1 = _run_judge(tmp_path / "serial", workers=1)
    parallel, _, fp4 = _run_judge(tmp_path / "parallel", workers=4)
    a = (tmp_path / "serial" / "state_judgments.jsonl").read_bytes()
    assert a == (tmp_path / "parallel" / "state_judgments.jsonl").read_bytes()
    assert len(a.splitlines()) == 60
    for attr in ("n_judged", "n_processed", "n_parse_retries", "failed_attempts", "failed"):
        assert getattr(serial, attr) == getattr(parallel, attr), attr
    assert serial.complete and parallel.complete and parallel.workers == 4
    # 8 E failures (7, 14, ..., 56), each 3 refusals; parse retries: 2 per failure + 1 per
    # multiple of 5 that is not a multiple of 7
    assert len(serial.failed) == 8
    assert serial.n_parse_retries == 8 * 2 + sum(1 for i in range(1, 61) if i % 5 == 0 and i % 7)
    assert sorted(c.cache_key() for c in fp1.calls) == sorted(c.cache_key() for c in fp4.calls)


def test_step05_judge_workers_4_and_workers_1_leave_identical_files(
    sandbox: Path, monkeypatch, frozen_clock
):
    import shutil

    _prepare_states(monkeypatch)
    sleep = _jitter(0.004)

    def responder(req):
        sleep()
        text = content_text(req.messages[0]["content"])
        return json.dumps({"label": "UNSAFE" if "idiot" in text else "SAFE", "reason": "r"})

    out = {}
    for w in ("1", "4"):
        L.y_star_path().unlink(missing_ok=True)
        shutil.rmtree(paths.DATA_CACHE, ignore_errors=True)  # every call is made again
        m = Manifest.load()
        m.update_section("labels", {"complete": False})
        m.save()
        fp = _fake_client(step05, monkeypatch, responder=responder)
        assert step05.main(["judge", "--budget", "5", "--workers", w]) == 0
        assert len(fp.calls) == 40
        out[w] = L.y_star_path().read_bytes()
        lab = Manifest.load().labels
        assert lab["complete"] is True and lab["run"]["workers"] == int(w)
        assert lab["n_unsafe"] == 14
    assert out["1"] == out["4"]


def test_step05_rate_limits_are_retried_and_never_count_as_e_failures(tmp_path: Path, frozen_clock):
    """As for step00: every request rate limited twice; the scripted E's real failures
    (multiples of 7) are never adjacent, so a consecutive-failure limit of 2 only fires if
    429s were counted. It runs to the end and the y* file matches a run without any 429."""
    clean, _, _ = _run_judge(tmp_path / "clean", workers=4)
    limited = _scripted_e(transient=dict.fromkeys(range(1, 61), 2))
    res, client, fp = _run_judge(
        tmp_path / "limited", workers=4, responder=limited, max_consecutive_e_failures=2
    )
    assert res.complete and res.stop.reason == "complete"
    assert (tmp_path / "limited" / "state_judgments.jsonl").read_bytes() == (
        tmp_path / "clean" / "state_judgments.jsonl"
    ).read_bytes()
    assert client.transient_stats["rate_limit"] * 3 == len(fp.calls) * 2
    assert res.transient["retries"] == client.transient_stats["rate_limit"]
    assert res.transient["exhausted"] == 0 and res.n_transient_gave_up == 0
    assert res.n_parse_retries == clean.n_parse_retries  # 429s are not parse retries
    assert res.failed == clean.failed  # nor E failures


def test_step05_a_429_streak_does_not_trigger_the_consecutive_failure_stop(tmp_path: Path):
    """Heavy rate limiting on the first states only, and a failure limit of 1: no stop."""
    responder = _scripted_e(transient={1: 5, 2: 5, 3: 5})
    res, client, _ = _run_judge(
        tmp_path, workers=4, n=6, responder=responder, max_consecutive_e_failures=1
    )
    assert res.complete and res.n_processed == 6 and not res.failed
    assert client.transient_stats["rate_limit"] == 15


def test_step05_exhausted_transient_retries_stop_the_run_but_are_no_e_failure(tmp_path: Path):
    net = {"transient_retries": 2, "backoff_base_s": 1, "backoff_cap_s": 1}
    responder = _scripted_e(transient={4: -1})
    res, client, _ = _run_judge(tmp_path, workers=1, n=8, responder=responder, net=net)
    assert res.stop.reason == "transient_errors" and res.stop.is_error and not res.complete
    assert not res.failed and res.n_transient_gave_up == 1 and res.n_judged == 3
    rows = L.load_judgment_rows(tmp_path / "state_judgments.jsonl")
    assert list(rows) == ["x001-g1", "x002-g1", "x003-g1"]  # no row: judged on the next run
    # the next run, provider healthy again, picks x004 up and finishes
    res2, _, _ = _run_judge(tmp_path, workers=4, n=8)
    assert res2.complete and res2.n_reused == 3 and res2.n_judged == 5


def test_step05_limit_budget_and_failure_stop_still_apply_with_workers(tmp_path: Path):
    from rafr.llm.ledger import Ledger, Usage
    from rafr.llm.providers.fake import FakeReply

    res, _, _ = _run_judge(tmp_path / "limit", workers=4, limit=10)
    assert res.stop.reason == "limit" and res.n_judged == 10
    assert len(_lines(tmp_path / "limit" / "state_judgments.jsonl")) == 10

    def pricey(req):  # 1M input tokens = $5.00 a call at the claude-opus-5 list price
        body = json.dumps({"label": "SAFE", "reason": "r."})
        return FakeReply(text=body, usage=Usage(1_000_000, 0))

    res, _, fp = _run_judge(
        tmp_path / "budget", workers=4, n=40, responder=pricey, price=True, budget_usd=12.0
    )
    # checked before every submission: at most workers - 1 calls beyond the serial stop
    assert res.stop.reason == "budget" and 3 <= res.n_judged <= 3 + 3
    assert len(Ledger(tmp_path / "budget" / "ledger.jsonl").rows()) == res.n_judged
    assert len(_lines(tmp_path / "budget" / "state_judgments.jsonl")) == res.n_judged

    def refuse(req):
        return FakeReply(text=None, refused=True, stop_reason="refusal")

    res, _, fp = _run_judge(
        tmp_path / "fail", workers=4, n=40, responder=refuse, max_consecutive_e_failures=3
    )
    assert res.stop.reason == "e_failures" and 3 <= len(res.failed) <= 3 + 3
    assert len(_lines(tmp_path / "fail" / "state_judgments.jsonl")) == len(res.failed)
    assert len(fp.calls) == 3 * len(res.failed)  # three attempts per failure, none cut short
