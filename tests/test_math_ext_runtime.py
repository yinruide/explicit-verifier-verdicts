"""Actual CLI dispatch on original-layout fixtures; only provider/grader are injected."""

import json

import pytest
from pipelines import math_ext_run as cli
from tests import test_math_ext_integration as source_fixtures

from rafr.contracts.schemas import Split
from rafr.domains.math.storage import file_hash

native_source = source_fixtures.native_source


@pytest.fixture
def manifest(native_source, tmp_path):
    path = tmp_path / "source_manifest.json"
    path.write_text(
        json.dumps(
            {
                "source_root": str(native_source.root),
                "expected_sha256": native_source.provenance()["source_sha256"],
                "master_seed": native_source.master_seed,
                "verifier_ids": list(native_source.verifier_ids),
            }
        ),
        encoding="utf-8",
    )
    return path


def command(name, manifest, out, *extra):
    result = [name, "--manifest", str(manifest), "--stage", "offline_stub", "--out", str(out)]
    if name not in {"starts", "cohort", "calibrate"}:
        result += ["--run-id", out.name]
    return result + list(map(str, extra))


def refs(native_source, tmp_path):
    from tests.math_v2_helpers import q_number

    from rafr.domains.math.dataset import content_hash, problem_fingerprint

    path = tmp_path / "private.jsonl"
    rows = []
    for s in native_source.states(Split.D_EXP1):
        fp = problem_fingerprint(s.x0)
        answer = f"PRIVATE_GOLD_{q_number(s.x0):03}"
        rows.append(
            {
                "question_id": s.x0_id,
                "problem_sha256": fp,
                "reference_answer": answer,
                "binding_sha256": content_hash([s.x0_id, fp, answer]),
            }
        )
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    return path


def test_cli_public_loop_judge_and_ceiling(native_source, manifest, tmp_path, capsys):
    from tests.math_v2_helpers import ArithmeticGrader

    before = {p: file_hash(p) for p in native_source.root.rglob("*") if p.is_file()}
    public = tmp_path / "public"
    assert cli.main(command("starts", manifest, public)) == 0
    exported = json.loads(capsys.readouterr().out)
    assert "PRIVATE_GOLD" not in "".join(p.read_text(encoding="utf-8") for p in public.iterdir())
    loop = tmp_path / "math_exp2ext_fixture"
    common = ["--bundle", public, "--bundle-sha256", exported["manifest_sha256"]]
    assert cli.main(command("loop", manifest, loop, *common)) == 0
    assert json.loads(capsys.readouterr().out)["judged"] is False
    private = refs(native_source, tmp_path)
    e = ["--references", private, "--references-sha256", file_hash(private)]
    assert cli.main(command("judge", manifest, loop, *common, *e), grader=ArithmeticGrader()) == 0
    assert json.loads(capsys.readouterr().out)["n_results"] == 6
    ceiling = tmp_path / "math_ceiling_fixture"
    assert cli.main(command("ceiling", manifest, ceiling, *e), grader=ArithmeticGrader()) == 0
    assert json.loads(capsys.readouterr().out)["n_results"] == 6
    assert before == {p: file_hash(p) for p in native_source.root.rglob("*") if p.is_file()}


def test_cli_exp1_and_p2_fill_use_original_paths(native_source, manifest, tmp_path, capsys):
    out = tmp_path / "math_exp1ext_fixture"
    assert cli.main(command("exp1", manifest, out, "--arms", "no_verifier")) == 0
    capsys.readouterr()
    public = tmp_path / "fill"
    assert cli.main(command("starts", manifest, public, "--fill-original")) == 0
    report = json.loads(capsys.readouterr().out)
    assignment = native_source.read_bound_json("exp2/assignment.json")
    for row in map(json.loads, (public / "starts.jsonl").read_text(encoding="utf-8").splitlines()):
        assert row["arms"] == ["B" if assignment[row["x0_id"]] == "A1" else "A1"]
    assert report["manifest"]["source_mode"] == "offline_stub"


@pytest.mark.parametrize(
    "failure", ["missing_confirm", "mode_mismatch", "bad_arm", "stopped", "bad_id", "smoke_scope"]
)
def test_invalid_input_never_constructs_provider(manifest, tmp_path, capsys, failure):
    out = tmp_path / "math_exp1ext_failure"
    args = command("exp1", manifest, out, "--arms", "no_verifier")
    if failure in {"missing_confirm", "mode_mismatch"}:
        args[args.index("offline_stub")] = "live"
        if failure == "mode_mismatch":
            args += ["--confirm"]
    elif failure == "bad_arm":
        args[-1] = "oracle_feedback"
    elif failure == "smoke_scope":
        args += ["--smoke", "1"]
    elif failure == "bad_id":
        args[args.index("--run-id") + 1] = "another"
    else:
        out.mkdir()
        (out / "STOPPED.json").write_text("{}", encoding="utf-8")
    calls = []
    assert cli.main(args, provider_factory=lambda: calls.append(True)) == 2
    assert not calls
    assert json.loads(capsys.readouterr().out)["status"] == "BLOCKED"


def test_freeze_cohort_is_explicit_and_reproduces_draw(native_source, manifest, tmp_path, capsys):
    candidate = tmp_path / "candidate"
    assert cli.main(command("cohort", manifest, candidate, "--n", "2")) == 0
    prepared = json.loads(capsys.readouterr().out)
    out = tmp_path / "frozen"
    args = [
        "freeze",
        "--kind",
        "cohort",
        "--manifest",
        str(manifest),
        "--stage",
        "offline_stub",
        "--candidate",
        prepared["cohort_file"],
        "--candidate-sha256",
        prepared["cohort_sha256"],
        "--out",
        str(out),
    ]
    assert cli.main(args) == 2
    assert not out.exists()
    capsys.readouterr()
    assert cli.main(args + ["--confirm"]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["stage"] == "offline_stub" and not report["live_executed"]
    assert file_hash(out / "ecorrect_cohort_math.jsonl") == report["sha256"]


def test_private_reference_hash_and_join_guard(native_source, tmp_path):
    from pipelines.math_ext_runtime import private_references

    path = refs(native_source, tmp_path)
    states = native_source.states(Split.D_EXP1)
    assert len(private_references(path, file_hash(path), states)) == 6
    with pytest.raises(ValueError, match="hash mismatch"):
        private_references(path, "0" * 64, states)
    rows = list(map(json.loads, path.read_text(encoding="utf-8").splitlines()))
    rows[0]["problem_sha256"] = "0" * 64
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    with pytest.raises(ValueError, match="binding mismatch"):
        private_references(path, file_hash(path), states)


def test_freeze_calibration_requires_bound_review_receipts(manifest, tmp_path, capsys):
    candidate = tmp_path / "calibration_candidate"
    assert cli.main(command("calibrate", manifest, candidate, "--n-boot", "5")) == 0
    report = json.loads(capsys.readouterr().out)
    out = tmp_path / "frozen_calibration"
    args = [
        "freeze",
        "--kind",
        "calibration",
        "--manifest",
        str(manifest),
        "--stage",
        "offline_stub",
        "--confirm",
        "--candidate",
        report["calibrator_file"],
        "--candidate-sha256",
        report["calibrator_sha256"],
        "--out",
        str(out),
        "--receipt",
        str(candidate / "COMPLETE.json"),
        "--receipt-sha256",
        file_hash(candidate / "COMPLETE.json"),
        "--diagnostics",
        str(candidate / "diagnostics.json"),
        "--diagnostics-sha256",
        "0" * 64,
    ]
    assert cli.main(args) == 2
    assert not out.exists()
    capsys.readouterr()
    args[-1] = file_hash(candidate / "diagnostics.json")
    assert cli.main(args) == 0
    frozen = json.loads(capsys.readouterr().out)
    record = json.loads((out / "calibration_math.json").read_text(encoding="utf-8"))
    assert record["fit_data"]["scientifically_frozen"]
    assert record["fit_data"]["freeze_stage"] == "offline_stub"
    assert file_hash(out / "calibration_math.json") == frozen["sha256"]


def test_judge_incomplete_loop_does_not_create_e_records(native_source, manifest, tmp_path, capsys):
    public = tmp_path / "public"
    assert cli.main(command("starts", manifest, public)) == 0
    exported = json.loads(capsys.readouterr().out)
    private = refs(native_source, tmp_path)
    out = tmp_path / "math_exp2ext_incomplete"
    assert (
        cli.main(
            command(
                "judge",
                manifest,
                out,
                "--bundle",
                public,
                "--bundle-sha256",
                exported["manifest_sha256"],
                "--references",
                private,
                "--references-sha256",
                file_hash(private),
            )
        )
        == 2
    )
    assert "four completed" in capsys.readouterr().out
    assert not out.exists()


def test_original_private_bundle_is_bound_to_snapshot_and_selected_questions(
    native_source, tmp_path
):
    from dataclasses import asdict
    from types import SimpleNamespace

    from pipelines.math_ext_runtime import bundle_references

    from rafr.domains.math.dataset import PrivateReference

    private = refs(native_source, tmp_path)
    states = native_source.states(Split.D_EXP1)
    root = tmp_path / "dataset_bundle"
    root.mkdir()
    (root / "references.private.jsonl").write_text(
        "".join(
            json.dumps(
                asdict(PrivateReference(**row, original_split="train", subject="fixture", level=1))
            )
            + "\n"
            for row in map(json.loads, private.read_text(encoding="utf-8").splitlines())
        ),
        encoding="utf-8",
    )
    (root / "public_questions.jsonl").write_text(
        "".join(json.dumps({"question_id": s.x0_id, "problem": s.x0}) + "\n" for s in states),
        encoding="utf-8",
    )
    (root / "exclusions.jsonl").write_text("", encoding="utf-8")
    (root / "eligibility.jsonl").write_text("", encoding="utf-8")
    (root / "manifest.json").write_text(
        json.dumps(
            {
                "domain": "math",
                "files": {
                    p.name: file_hash(p) for p in root.iterdir() if p.name != "manifest.json"
                },
            }
        ),
        encoding="utf-8",
    )
    expected = file_hash(root / "manifest.json")
    source = SimpleNamespace(read_bound_json=lambda name: {"math_identity": {"bundle": expected}})
    assert set(bundle_references(root, source, states[:1])) == {states[0].x0_id}
    (root / "references.private.jsonl").write_text("{}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="hash mismatch"):
        bundle_references(root, source, states[:1])


def test_frozen_records_bind_the_source_by_sha256_not_by_path(native_source):
    """2026-11-05: the recorded source_root is informational; the sha256 of every bound
    file, the execution mode, the master seed and the verifiers decide."""
    from pipelines.math_ext_data import same_bound_source

    recorded = native_source.provenance()
    assert same_bound_source(recorded, native_source)
    assert same_bound_source(
        {**recorded, "source_root": "/elsewhere/math_formal_n9000"}, native_source
    )
    changed = {**recorded["source_sha256"], "scores.jsonl": "0" * 64}
    assert not same_bound_source({**recorded, "source_sha256": changed}, native_source)
    assert not same_bound_source(
        {**recorded, "master_seed": recorded["master_seed"] + 1}, native_source
    )
    assert not same_bound_source({**recorded, "source_sha256": {}}, native_source)
    assert not same_bound_source(None, native_source)


def test_real_freeze_writes_the_configured_files_once_and_refuses_out(native_source, tmp_path):
    from pipelines.math_ext_runtime import FROZEN_TARGETS, ROOT, freeze_target

    assert FROZEN_TARGETS == {
        "calibration": "frozen/calibration_math.json",
        "cohort": "data/ext/ecorrect_cohort_math.jsonl",
    }
    for kind, rel in FROZEN_TARGETS.items():
        with pytest.raises(ValueError, match="offline_stub"):
            freeze_target(kind, "real_prepare", tmp_path / "out", native_source)
        if (ROOT / rel).exists():  # after M-C: never overwritten
            with pytest.raises(ValueError, match="written once"):
                freeze_target(kind, "real_prepare", None, native_source)
        else:
            assert freeze_target(kind, "real_prepare", None, native_source) == ROOT / rel
            assert not (ROOT / rel).exists()  # the target is named, not written
        with pytest.raises(ValueError, match="needs --out"):
            freeze_target(kind, "offline_stub", None, native_source)
