"""Original math-layout fixtures; never real model or pilot artifact verification."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from pipelines.math_ext_data import (
    MathSource,
    fresh_output,
    prepare_calibration,
    prepare_correct_cohort,
)
from tests.math_v2_helpers import make_harness, prepare_fit

from rafr.calibration.apply import Calibrator, file_sha256
from rafr.contracts.schemas import Split
from rafr.data import labels as L
from rafr.domains.math.storage import IntegrityError
from rafr.experiments.ext_cohort import read_cohort


def hashes(root):
    names = {
        "states.jsonl",
        "initial_complete.json",
        "splits_complete.json",
        "scores.jsonl",
        "fit.json",
        "snapshot.json",
    }
    for name in ("initial_complete.json", "splits_complete.json"):
        names.update(json.loads((root / name).read_text(encoding="utf-8"))["files"])
    return {name: file_sha256(root / name) for name in names}


def open_source(root, binding=None, **kwargs):
    return MathSource(
        root,
        expected_sha256=binding or hashes(root),
        master_seed=kwargs.get("seed", 20261002),
        verifier_ids=kwargs.get("verifiers", ("fixture-V1", "fixture-V2")),
    )


@pytest.fixture
def original(tmp_path, monkeypatch):
    h = make_harness(tmp_path, monkeypatch)
    prepare_fit(h)
    return h


def test_real_math_layout_offline_preparation_is_read_only(original, tmp_path, monkeypatch):
    root = original.root
    before = {
        p.relative_to(root).as_posix(): file_sha256(p) for p in root.rglob("*") if p.is_file()
    }
    source = open_source(root)
    calls = []
    original_read = L.read_labels

    def tracked(split, purpose, **kwargs):
        calls.append((split, purpose))
        assert kwargs["gate_attempts_path"] == source.root / "gate_attempts.json"
        return original_read(split, purpose, **kwargs)

    monkeypatch.setattr(L, "read_labels", tracked)
    cal = prepare_calibration(source, tmp_path / "calibration", n_boot=25, stage="offline_stub")
    assert cal["n_fit"] == 114 and cal["n_diagnostic"] == 40
    assert cal["scientifically_frozen"] is False and cal["live_executed"] is False
    fitted = Calibrator.load(Path(cal["calibrator_file"]), expected_sha256=cal["calibrator_sha256"])
    pc = fitted.record["prior_correction"]
    assert pc["n_added_positive"] == 6
    assert pc["p_R"] > pc["p_pool"]
    assert calls == [
        (Split.UNLABELED_POOL, L.LabelPurpose.CALIBRATION_FIT),
        (Split.D_GATE, L.LabelPurpose.CALIBRATION_DIAG),
    ]
    cohort = prepare_correct_cohort(source, tmp_path / "cohort")
    header, ids = read_cohort(Path(cohort["cohort_file"]), cohort["cohort_sha256"])
    assert header["target_n"] == 300 and len(ids) == cohort["n_candidates"]
    assert len(ids) > 0 and len(set(ids)) == len(ids)
    assert calls[-1] == (Split.D_GATE, L.LabelPurpose.ECORRECT_COHORT)
    assert before == {
        p.relative_to(root).as_posix(): file_sha256(p) for p in root.rglob("*") if p.is_file()
    }
    assert not list(tmp_path.glob("**/frozen/*"))


def test_binding_required_and_tamper_detected(original):
    binding = hashes(original.root)
    source = open_source(original.root, binding)
    incomplete = dict(binding)
    incomplete.pop("states.jsonl")
    with pytest.raises(IntegrityError, match="missing source"):
        open_source(original.root, incomplete)
    (original.root / "scores.jsonl").write_text(
        (original.root / "scores.jsonl").read_text(encoding="utf-8") + "\n", encoding="utf-8"
    )
    with pytest.raises(IntegrityError, match="changed"):
        source.states(Split.D_EXP1)


def test_identity_and_private_label_boundaries(original):
    with pytest.raises(IntegrityError, match="seed"):
        open_source(original.root, seed=7)
    with pytest.raises(IntegrityError, match="verifier identities"):
        open_source(original.root, verifiers=("fixture-V2", "fixture-V1"))
    source = open_source(original.root)
    with pytest.raises(L.LabelAccessError):
        source.labels(Split.D_GATE_RESERVE, L.LabelPurpose.GATE)
    with pytest.raises(L.LabelAccessError):
        source.labels(Split.D_GATE, L.LabelPurpose.CALIBRATION_FIT)
    for name in source.provenance()["source_sha256"]:
        if name.startswith("labels/"):
            with pytest.raises(L.LabelAccessError):
                source.read_bound_rows(name)
            with pytest.raises(L.LabelAccessError):
                source.read_bound_json(name)
    altered = source.states(Split.D_EXP1)[0].model_copy(update={"text": "changed"})
    with pytest.raises(IntegrityError, match="original initial state"):
        source.pairs([altered])


def test_no_overwrite_or_source_overlap(original, tmp_path, make_symlink):
    source = open_source(original.root)
    for out in (original.root, original.root / "extension", original.root.parent):
        with pytest.raises(IntegrityError, match="overlap"):
            fresh_output(source, out)
    existing = tmp_path / "existing"
    existing.mkdir()
    with pytest.raises(IntegrityError, match="new directory"):
        fresh_output(source, existing)
    link = tmp_path / "alias"
    make_symlink(link, tmp_path / "target", target_is_directory=True)
    with pytest.raises(IntegrityError, match="symlink"):
        fresh_output(source, link / "child")
    outside = Path(__file__).resolve().parents[1] / "data" / "forbidden-extension"
    with pytest.raises(IntegrityError, match="output must"):
        fresh_output(source, outside)
    assert not outside.exists()


def test_duplicate_conflicting_scores_stop(original):
    path = original.root / "scores.jsonl"
    row = json.loads(path.read_text(encoding="utf-8").splitlines()[0])
    row["score"] = 0.4
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(row) + "\n")
    with pytest.raises(IntegrityError, match="conflicting duplicate"):
        open_source(original.root)


def test_source_symlink_and_manifest_path_stop(original, tmp_path, make_symlink):
    link = tmp_path / "source-link"
    make_symlink(link, original.root, target_is_directory=True)
    with pytest.raises(IntegrityError, match="symlink"):
        open_source(link)
    binding = hashes(original.root)
    binding["../escape"] = "0" * 64
    with pytest.raises(IntegrityError, match="unsafe"):
        open_source(original.root, binding)


def test_stub_settings_cannot_shrink_candidate_scientific_rules(original, tmp_path):
    source = open_source(original.root)
    with pytest.raises(IntegrityError, match="10000"):
        prepare_calibration(source, tmp_path / "invalid", n_boot=25)
    with pytest.raises(IntegrityError, match="target is 300"):
        prepare_correct_cohort(source, tmp_path / "invalid", n=3)
    with pytest.raises(IntegrityError, match="only offline"):
        prepare_correct_cohort(source, tmp_path / "invalid", stage="live")
    assert not (tmp_path / "invalid").exists()


def test_clipping_failure_preserves_evidence_without_fitting_labels(
    original, tmp_path, monkeypatch
):
    path = original.root / "scores.jsonl"
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    for row in rows:
        row["score"] = 0.0
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    fit_path = original.root / "fit.json"
    fit = json.loads(fit_path.read_text(encoding="utf-8"))
    for q in fit["channel"]["q"].values():
        q["scores"] = [0.0] * len(q["scores"])
    fit_path.write_text(json.dumps(fit), encoding="utf-8")
    snapshot_path = original.root / "snapshot.json"
    snapshot = json.loads(snapshot_path.read_text(encoding="utf-8"))
    snapshot["math_artifact_hashes"]["fit.json"] = file_sha256(fit_path)
    snapshot_path.write_text(json.dumps(snapshot), encoding="utf-8")
    source = open_source(original.root)

    def forbid(*args, **kwargs):
        raise AssertionError("clipping stop must occur before labels are read")

    monkeypatch.setattr(L, "read_labels", forbid)
    from rafr.calibration.fit import ClipShareError

    with pytest.raises(ClipShareError):
        prepare_calibration(source, tmp_path / "clip-stop", n_boot=25, stage="offline_stub")
    assert (tmp_path / "clip-stop" / "clip_report.json").is_file()
    assert not (tmp_path / "clip-stop" / "calibration_math_candidate.json").exists()
    assert not (tmp_path / "clip-stop" / "COMPLETE.json").exists()


def test_original_split_draw_premise_is_checked_not_assumed(original, tmp_path):
    # Swap pool/Exp1 members but preserve disjointness, counts and every bound marker hash.
    # This is an internally consistent copy with the WRONG recorded draw, not a hash failure.
    root = original.root
    pool_path = root / "splits" / f"{Split.UNLABELED_POOL.value}.jsonl"
    exp_path = root / "splits" / f"{Split.D_EXP1.value}.jsonl"
    pool = pool_path.read_text(encoding="utf-8").splitlines()
    exp = exp_path.read_text(encoding="utf-8").splitlines()
    exp_number = int(json.loads(exp[0])["x0_id"].split(":")[1])
    index = next(
        i
        for i, row in enumerate(pool)
        if int(json.loads(row)["x0_id"].split(":")[1]) % 2 == exp_number % 2
    )
    pool[index], exp[0] = exp[0], pool[index]
    pool_path.write_text("\n".join(pool) + "\n", encoding="utf-8")
    exp_path.write_text("\n".join(exp) + "\n", encoding="utf-8")
    marker_path = root / "splits_complete.json"
    marker = json.loads(marker_path.read_text(encoding="utf-8"))
    for path in (pool_path, exp_path):
        marker["files"][path.relative_to(root).as_posix()] = file_sha256(path)
    marker_path.write_text(json.dumps(marker), encoding="utf-8")
    fit_path = root / "fit.json"
    fit = json.loads(fit_path.read_text(encoding="utf-8"))
    fit["q_question_ids"] = [json.loads(row)["x0_id"] for row in pool]
    fit_path.write_text(json.dumps(fit), encoding="utf-8")
    snapshot_path = root / "snapshot.json"
    snapshot = json.loads(snapshot_path.read_text(encoding="utf-8"))
    snapshot["math_artifact_hashes"]["fit.json"] = file_sha256(fit_path)
    snapshot_path.write_text(json.dumps(snapshot), encoding="utf-8")
    source = open_source(root)
    from rafr.calibration.fit import PriorPremiseError

    with pytest.raises(PriorPremiseError):
        prepare_calibration(source, tmp_path / "wrong-draw", n_boot=25, stage="offline_stub")
    assert not (tmp_path / "wrong-draw" / "calibration_math_candidate.json").exists()


def test_source_views_do_not_mutate_held_identity(original):
    source = open_source(original.root)
    ids = source.splits[Split.D_EXP1]
    ids.clear()
    plan = source.split_plan
    plan["seed"] = 123
    assert len(source.states(Split.D_EXP1)) == 6
    assert source.split_plan["seed"] != 123


def test_scoring_phase_binding_wins_over_initially_pending_config(original):
    path = original.root / "snapshot.json"
    snapshot = json.loads(path.read_text(encoding="utf-8"))
    snapshot["configs"]["math"]["verifiers"] = []
    path.write_text(json.dumps(snapshot), encoding="utf-8")
    source = open_source(original.root)
    assert source.verifier_versions == ("offline-1", "offline-2")
    snapshot["math_phase_bindings"]["scoring"]["verifiers"][0]["revision"] = "wrong-revision"
    path.write_text(json.dumps(snapshot), encoding="utf-8")
    with pytest.raises(IntegrityError, match="revision differs"):
        open_source(original.root)


def test_resume_output_guard_and_archive(original, tmp_path, monkeypatch):
    from rafr.core import paths

    monkeypatch.setattr(paths, "ARCHIVE", tmp_path / "archive")
    from pipelines.math_ext_data import validate_output

    source = open_source(original.root)
    out = tmp_path / "ext-resume"
    out.mkdir()
    assert validate_output(out, source_root=source.root, allow_existing=True) == out.resolve()
    with pytest.raises(IntegrityError, match="archive"):
        validate_output(tmp_path / "archive" / "new")
    file = tmp_path / "file"
    file.write_text("keep", encoding="utf-8")
    with pytest.raises(IntegrityError, match="new directory"):
        validate_output(file, allow_existing=True)
