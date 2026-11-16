"""Public pod transport from actual math-layout synthetic runs; no live execution."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest
from pipelines.math_exp2_loop import text_key
from pipelines.math_ext_data import MathSource
from pipelines.math_ext_native import native_t1
from pipelines.math_ext_transport import export_public_bundle, import_public_bundle
from tests.math_v2_helpers import make_harness, prepare_gate
from tests.test_math_ext_data import hashes

from rafr.calibration.apply import file_sha256
from rafr.contracts.schemas import Arm, MathLabel, Split
from rafr.data.labels import LabelPurpose
from rafr.domains.math.storage import IntegrityError, digest
from rafr.experiments.ext_cohort import write_cohort


@pytest.fixture
def original(tmp_path, monkeypatch):
    h = make_harness(tmp_path, monkeypatch, mode="accept")
    prepare_gate(h)
    h.pipeline.exp2()
    bindings = hashes(h.root)
    bindings.update(
        {
            name: file_sha256(h.root / name)
            for name in (
                "exp2/assignment.json",
                "exp2/rounds/round_1/decisions.jsonl",
                "exp2/rounds/round_1/scores.jsonl",
            )
        }
    )
    source = MathSource(
        h.root,
        expected_sha256=bindings,
        master_seed=h.pipeline.config.seed,
        verifier_ids=("fixture-V1", "fixture-V2"),
    )
    return source


def make_bundle(source, out, **kwargs):
    states = source.states(Split.D_EXP1)
    return export_public_bundle(
        source,
        out,
        cohort="e_incorrect",
        smoke=False,
        fixture=True,
        plan={s.x0_id: [Arm.B_PIHAT] for s in states},
        **kwargs,
    )


def load(result, source, **kwargs):
    opts = {
        "expected_manifest_sha256": result["manifest_sha256"],
        "expected_source_identity_sha256": digest(source.provenance()),
        "expected_verifier_ids": source.verifier_ids,
        "expected_verifier_versions": source.verifier_versions,
        "expected_source_mode": source.execution_mode,
    }
    opts.update(kwargs)
    return import_public_bundle(Path(result["directory"]), **opts)


def edit_bundle(result, name, update):
    """Re-sign adversarial bytes to test semantic allowlists after integrity checks pass."""
    root = Path(result["directory"])
    path = root / name
    if name == "starts.jsonl":
        data = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
        update(data)
        path.write_text("".join(json.dumps(row) + "\n" for row in data), encoding="utf-8")
    else:
        data = json.loads(path.read_text(encoding="utf-8"))
        update(data)
        path.write_text(json.dumps(data), encoding="utf-8")
    sha = file_sha256(path)
    Path(str(path) + ".sha256").write_text(sha + "\n", encoding="utf-8")
    manifest = json.loads((root / "transport.json").read_text(encoding="utf-8"))
    manifest["files"][name] = sha
    (root / "transport.json").write_text(json.dumps(manifest), encoding="utf-8")
    result["manifest_sha256"] = file_sha256(root / "transport.json")
    (root / "transport.json.sha256").write_text(result["manifest_sha256"] + "\n", encoding="utf-8")


def test_native_round_one_public_bundle_roundtrip(original, tmp_path):
    before = {p: file_sha256(p) for p in original.root.rglob("*") if p.is_file()}
    wanted = native_t1(original, original.states(Split.D_EXP1))
    result = make_bundle(original, tmp_path / "bundle", native_payload=wanted)
    got = load(result, original)
    assert got.reuse_t1 and got.t1_payload == wanted
    assert len(got.states) == 6
    assert {s.x0_id for s in got.states} == set(got.plan)
    rows = [
        json.loads(line)
        for line in (Path(result["directory"]) / "starts.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert all(set(row) == {"state_id", "x0_id", "x0", "text", "arms"} for row in rows)
    for path in Path(result["directory"]).iterdir():
        assert "PRIVATE_GOLD" not in path.read_text(encoding="utf-8")
    assert before == {p: file_sha256(p) for p in original.root.rglob("*") if p.is_file()}
    assert got.manifest["transport_only"] and got.manifest["live_executed"] is False


def test_wrong_external_identity_revision_mode_and_tampering(original, tmp_path):
    result = make_bundle(original, tmp_path / "bundle")
    for kwargs in (
        {"expected_source_identity_sha256": "0" * 64},
        {"expected_verifier_versions": ("wrong", "offline-2")},
        {"expected_source_mode": "live"},
    ):
        with pytest.raises(IntegrityError):
            load(result, original, **kwargs)
    path = Path(result["directory"]) / "starts.jsonl"
    path.write_text(path.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    with pytest.raises(IntegrityError, match="changed"):
        load(result, original)


@pytest.mark.parametrize("field", ["labels", "gold", "metadata", "supporting_facts", "references"])
def test_private_start_fields_rejected_even_with_updated_hashes(original, tmp_path, field):
    result = make_bundle(original, tmp_path / "bundle")
    edit_bundle(result, "starts.jsonl", lambda rows: rows[0].update({field: "PRIVATE_SENTINEL"}))
    with pytest.raises(IntegrityError, match="unexpected or missing"):
        load(result, original)


def test_nested_t1_metadata_rejected_and_pairing_checked(original, tmp_path):
    result = make_bundle(original, tmp_path / "bundle")
    edit_bundle(
        result, "t1.json", lambda data: data["scoring_identity"].update({"reference": "SECRET"})
    )
    with pytest.raises(IntegrityError, match="scoring identity"):
        load(result, original)
    result2 = make_bundle(original, tmp_path / "bundle2")

    def swap(data):
        rows = data["states"]
        rows[0]["scores"], rows[1]["scores"] = rows[1]["scores"], rows[0]["scores"]

    edit_bundle(result2, "t1.json", swap)
    with pytest.raises(IntegrityError, match="binding differs"):
        load(result2, original)


def test_duplicate_start_and_missing_t1_coverage(original, tmp_path):
    result = make_bundle(original, tmp_path / "bundle")
    edit_bundle(result, "starts.jsonl", lambda rows: rows.append(rows[0]))
    with pytest.raises(IntegrityError, match="duplicate"):
        load(result, original)
    result2 = make_bundle(original, tmp_path / "bundle2")
    edit_bundle(result2, "t1.json", lambda data: data["states"].pop())
    with pytest.raises(IntegrityError, match="cover every"):
        load(result2, original)


def correct_cohort(source, path):
    eligible = [
        r.state_id
        for r in source.labels(Split.D_GATE, LabelPurpose.ECORRECT_COHORT)
        if r.label is MathLabel.CORRECT
    ]
    sha = write_cohort(path, eligible[:3], {"domain": "math"})
    return path, sha, eligible


def test_correct_frozen_cohort_and_smoke_use_reference_only(original, tmp_path):
    path, sha, eligible = correct_cohort(original, tmp_path / "cohort.jsonl")
    by_id = {s.state_id: s for s in original.states(Split.D_GATE)}
    for smoke, ids in ((False, eligible[:3]), (True, eligible[3:5])):
        result = export_public_bundle(
            original,
            tmp_path / f"bundle-{smoke}",
            cohort="e_correct",
            fixture=True,
            smoke=smoke,
            cohort_file=path,
            cohort_sha256=sha,
            state_ids=ids if smoke else None,
            plan={by_id[sid].x0_id: [Arm.A1] for sid in ids},
        )
        got = load(result, original)
        assert not got.reuse_t1
        assert got.manifest["cohort_sha256"] == sha
        assert {s.state_id for s in got.states} == set(ids)
    with pytest.raises(IntegrityError, match="overlaps"):
        export_public_bundle(
            original,
            tmp_path / "bad",
            cohort="e_correct",
            fixture=True,
            smoke=True,
            cohort_file=path,
            cohort_sha256=sha,
            state_ids=eligible[:1],
            plan={by_id[eligible[0]].x0_id: [Arm.A1]},
        )


def test_smoke_reuse_wrong_fill_and_fixture_relabel_refused(original, tmp_path):
    states = original.states(Split.D_EXP1)
    native = native_t1(original, states)
    pool = {s.state_id: s for s in original.states(Split.UNLABELED_POOL)}
    sid = next(
        r.state_id
        for r in original.labels(Split.UNLABELED_POOL, LabelPurpose.DRY_RUN)
        if r.label is MathLabel.INCORRECT
    )
    with pytest.raises(IntegrityError, match="reference-only"):
        export_public_bundle(
            original,
            tmp_path / "bad",
            cohort="e_incorrect",
            fixture=True,
            smoke=True,
            state_ids=[sid],
            plan={pool[sid].x0_id: [Arm.A1]},
            native_payload=native,
        )
    assignment = original.read_bound_json("exp2/assignment.json")
    with pytest.raises(IntegrityError, match="must not rerun"):
        export_public_bundle(
            original,
            tmp_path / "bad",
            cohort="e_incorrect",
            fixture=True,
            smoke=False,
            plan={s.x0_id: [assignment[s.x0_id]] for s in states},
        )
    with pytest.raises(IntegrityError, match="source mode"):
        export_public_bundle(
            original,
            tmp_path / "bad",
            cohort="e_incorrect",
            fixture=False,
            smoke=False,
            plan={s.x0_id: [Arm.B_PIHAT] for s in states},
        )


def test_no_overwrite_source_symlink_or_unapproved_output(original, tmp_path, make_symlink):
    with pytest.raises(IntegrityError, match="overlap"):
        make_bundle(original, original.root / "bundle")
    result = make_bundle(original, tmp_path / "bundle")
    with pytest.raises(IntegrityError, match="new directory"):
        make_bundle(original, tmp_path / "bundle")
    link = tmp_path / "link"
    make_symlink(link, tmp_path / "bundle", target_is_directory=True)
    result["directory"] = str(link)
    with pytest.raises(IntegrityError, match="symlinks"):
        load(result, original)
    forbidden = Path(__file__).resolve().parents[1] / "data" / "transport-forbidden"
    with pytest.raises(IntegrityError):
        make_bundle(original, forbidden)
    assert not forbidden.exists()


def test_supplied_native_payload_cannot_change_another_questions_score(original, tmp_path):
    payload = copy.deepcopy(native_t1(original, original.states(Split.D_EXP1)))
    row = payload["states"][0]
    row["scores"][0]["state_id"] = text_key("another-question", row["text"])
    with pytest.raises(IntegrityError, match="bound native source"):
        make_bundle(original, tmp_path / "bad", native_payload=payload)


def test_manifest_nesting_is_not_an_unchecked_metadata_channel(original, tmp_path):
    result = make_bundle(original, tmp_path / "bundle")
    root = Path(result["directory"])
    path = root / "transport.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    manifest["master_seed"] = {"private_reference": "PRIVATE_SENTINEL"}
    path.write_text(json.dumps(manifest), encoding="utf-8")
    result["manifest_sha256"] = file_sha256(path)
    (root / "transport.json.sha256").write_text(result["manifest_sha256"] + "\n", encoding="utf-8")
    with pytest.raises(IntegrityError, match="master seed"):
        load(result, original)


def test_correct_candidate_cannot_silently_become_formal(original, tmp_path):
    from pipelines.math_ext_transport import _read_cohort

    _, _, eligible = correct_cohort(original, tmp_path / "old_cohort.jsonl")
    path = tmp_path / "candidate.jsonl"
    sha = write_cohort(path, eligible[:3], {"scientifically_frozen": False})
    with pytest.raises(IntegrityError, match="candidate cohort"):
        _read_cohort(path, sha, fixture=False)


def test_mutating_source_mode_does_not_override_bound_snapshot(original, tmp_path):
    original.execution_mode = "live"
    with pytest.raises(IntegrityError, match="source mode"):
        export_public_bundle(
            original,
            tmp_path / "bad",
            cohort="e_incorrect",
            smoke=False,
            fixture=False,
            plan={s.x0_id: [Arm.B_PIHAT] for s in original.states(Split.D_EXP1)},
        )


def test_incorrect_segments_cannot_silently_drop_an_arm_on_some_starts(original, tmp_path):
    states = original.states(Split.D_EXP1)
    plan = {s.x0_id: [Arm.B_PIHAT] for s in states}
    plan[states[0].x0_id] = [Arm.NO_VERIFIER]
    with pytest.raises(IntegrityError, match="every selected start"):
        export_public_bundle(
            original, tmp_path / "bad", cohort="e_incorrect", smoke=False, fixture=True, plan=plan
        )
    assignment = original.read_bound_json("exp2/assignment.json")
    plan = {s.x0_id: [Arm.B_PIHAT] for s in states}
    first = states[0].x0_id
    plan[first].append(Arm.B if assignment[first] == "A1" else Arm.A1)
    with pytest.raises(IntegrityError, match="fill segment"):
        export_public_bundle(
            original, tmp_path / "bad", cohort="e_incorrect", smoke=False, fixture=True, plan=plan
        )
