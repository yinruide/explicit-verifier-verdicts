"""File-level extension contracts using original pipelines on synthetic inputs only."""

import json
import os
from pathlib import Path

import pytest
import yaml
from analysis.ext_contrasts import exp2_finals
from pipelines.math_ext_data import MathSource
from pipelines.math_ext_exp2 import grade_math_exp2, run_math_exp2, setup_from_public_bundle
from pipelines.math_ext_native import native_t1
from pipelines.math_ext_regression import replay_original
from pipelines.math_ext_snapshot import snapshot_configs
from pipelines.math_ext_transport import export_public_bundle, import_public_bundle
from tests.math_v2_helpers import make_harness, prepare_gate

from rafr.contracts.schemas import Arm, Split
from rafr.domains.math.generation import build_controller_config
from rafr.domains.math.storage import IntegrityError, file_hash
from rafr.formatter.math_extended import build_no_verifier_config, math_wording
from rafr.llm.cache import LLMCache
from rafr.llm.client import LLMClient
from rafr.llm.providers.fake import FakeProvider


@pytest.fixture
def native_source(tmp_path, monkeypatch):
    h = make_harness(tmp_path, monkeypatch)
    prepare_gate(h)
    h.pipeline.exp1()
    h.pipeline.exp2()
    hashes = {
        p.relative_to(h.root).as_posix(): file_hash(p) for p in h.root.rglob("*") if p.is_file()
    }
    return MathSource(
        h.root,
        expected_sha256=hashes,
        master_seed=h.pipeline.config.seed,
        verifier_ids=("fixture-V1", "fixture-V2"),
    )


def test_native_original_replay_is_counted_separately_from_real_evidence(native_source):
    evidence = replay_original(native_source, fixture=True)
    assert evidence["exp1"]["n_original_decisions"] == 54
    assert evidence["exp2"]["n_first_round_requests"] == 6
    assert evidence["provider_calls"] == 0
    assert evidence["execution"] == "offline_fixture"
    assert sum(evidence["all_original_log_rows"].values()) >= 60
    assert evidence["all_original_log_rows"]["exp2/rounds/round_4/decisions.jsonl"] == 0
    with pytest.raises(IntegrityError, match="requires live source"):
        replay_original(native_source)


def test_cli_and_real_evidence_gate_resolve_external_manifest_the_same_way(
    native_source, tmp_path, monkeypatch
):
    from pipelines.math_ext_run import load_source
    from tests import test_math_ext_contracts as contracts

    directory = tmp_path / "external_receipt"
    directory.mkdir()
    manifest = directory / "source_manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "source_root": os.path.relpath(native_source.root, directory),
                "expected_sha256": native_source.provenance()["source_sha256"],
                "master_seed": native_source.master_seed,
                "verifier_ids": list(native_source.verifier_ids),
            }
        ),
        encoding="utf-8",
    )
    assert load_source(manifest).root == native_source.root
    monkeypatch.setattr(contracts, "_ext", lambda: {"source_manifest": str(manifest)})
    try:
        with pytest.raises(AssertionError, match="fixture is not real-artifact"):
            contracts._real_source()
    except pytest.skip.Exception:
        pytest.fail("supplied relative manifest must be opened, not silently skipped")


def test_native_t1_public_loop_judge_and_ruide_consumer(native_source, tmp_path):
    source = native_source
    starts = source.states(Split.D_EXP1)
    export = native_t1(source, starts)
    assert "PRIVATE_GOLD" not in json.dumps(export)
    prepared = export_public_bundle(
        source,
        tmp_path / "public_bundle",
        cohort="e_incorrect",
        smoke=False,
        plan={s.x0_id: [Arm.NO_VERIFIER, Arm.VERDICT_ONLY] for s in starts},
        fixture=True,
    )
    bundle = import_public_bundle(
        Path(prepared["directory"]),
        expected_manifest_sha256=prepared["manifest_sha256"],
        expected_source_identity_sha256=prepared["manifest"]["source_identity_sha256"],
        expected_verifier_ids=source.verifier_ids,
        expected_verifier_versions=source.verifier_versions,
        expected_source_mode="offline_stub",
    )
    run_dir = tmp_path / "math_exp2ext_fixture"
    provider = FakeProvider(lambda req: '{"action":"accept","revised_text":null}')
    client = LLMClient({"openai": provider}, LLMCache(run_dir / "cache"))
    ext = yaml.safe_load(
        (Path(__file__).resolve().parents[1] / "configs/math/ext.yaml").read_text(encoding="utf-8")
    )
    setup = setup_from_public_bundle(
        bundle,
        run_id="OFFLINE-FIXTURE",
        run_dir=run_dir,
        taus=source.taus,
        score_texts=lambda *a: pytest.fail("native t1 must not be rescored"),
        wording=math_wording(),
        extension_config=ext,
        channel_config=source.read_bound_json("fit.json")["channel"],
    )
    cfg = build_controller_config(source.master_seed, 4)
    nv = build_no_verifier_config(source.master_seed, 4)
    trajectories = run_math_exp2(client, cfg, nv, setup)
    meta = json.loads((run_dir / "ext_meta.json").read_text(encoding="utf-8"))
    assert "initial_grades" not in meta and len(provider.calls) == 12
    assert not (run_dir / "math_judgments.jsonl").exists()
    # Local-only E phase attaches the known synthetic initial statuses after pod execution.
    setup.initial_grades = {s.x0_id: {"status": "incorrect", "task_success": False} for s in starts}
    rows = grade_math_exp2(
        setup,
        trajectories,
        lambda *a: pytest.fail("accepted x1 rejudged"),
        e_identity={"mode": "offline_fixture"},
    )
    consumed = exp2_finals("e_incorrect", [run_dir], prefix="math_exp2ext", base=tmp_path)
    assert len(consumed) == 12 and all(
        r["task_success"] is False and r["accepted"] for r in consumed
    )
    assert [{k: v for k, v in r.items() if k != "_panel"} for r in consumed] == rows
    final_bytes = (run_dir / "trajectories.jsonl").read_bytes()
    resumed = run_math_exp2(client, cfg, nv, setup)
    grade_math_exp2(
        setup,
        resumed,
        lambda *a: pytest.fail("resumed E called"),
        e_identity={"mode": "offline_fixture"},
    )
    assert (run_dir / "trajectories.jsonl").read_bytes() == final_bytes and len(
        provider.calls
    ) == 12
    from analysis.ext_decision_quality import lambda_levels

    levels = lambda_levels(
        yaml.safe_load((run_dir / "configs/channel.yaml").read_text(encoding="utf-8"))
    )
    assert set(levels) == {"low", "mid", "high"}
    assert (
        yaml.safe_load((run_dir / "configs/ext.yaml").read_text(encoding="utf-8"))["domain"]
        == "math"
    )


def test_reference_only_t1_difference_never_replaces_fresh_scores(tmp_path):
    from tests.test_math_ext_exp2 import harness, run

    h = harness(tmp_path, arms=(Arm.A1, Arm.B))
    h[0].initial_grades = {}  # Public pod execution has no E labels.
    h[0].t1_reference = {"q1": (0.9, 0.2)}
    run(h)
    report = json.loads((h[0].run_dir / "t1_reference_check.json").read_text(encoding="utf-8"))
    assert report["n"] == 1 and report["v1_identical"] == 0 and report["v2_identical"] == 1
    from rafr.logging.writer import read_jsonl

    assert all(r["s1"] == 0.2 for r in read_jsonl(h[0].run_dir / "decisions.jsonl"))
    calls = len(h[2].calls)
    run(h)
    assert len(h[2].calls) == calls
    h[0].t1_reference = {"q1": (0.3, 0.2)}
    with pytest.raises(RuntimeError, match="different setup"):
        run(h)


def test_gitless_snapshot_uses_core_fallback_and_fails_without_head(tmp_path, monkeypatch):
    from rafr.logging import snapshot as core_snapshot

    def no_git(*a, **kw):
        raise RuntimeError("no git archive")

    monkeypatch.setattr(core_snapshot, "git_head", no_git)
    monkeypatch.delenv("RAFR_GIT_HEAD", raising=False)
    channel = {"levels": {"low": {"v1": 0.1, "v2": 0.1}}}
    with pytest.raises(IntegrityError, match="RAFR_GIT_HEAD"):
        snapshot_configs(tmp_path / "missing", {"domain": "math"}, channel, identity={"fixture": 1})
    monkeypatch.setenv("RAFR_GIT_HEAD", "a" * 40)
    snapshot_configs(tmp_path / "okay", {"domain": "math"}, channel, identity={"fixture": 1})
    snapshot = json.loads((tmp_path / "okay/snapshot.json").read_text(encoding="utf-8"))
    assert snapshot["git_head"] == "a" * 40 and "RAFR_GIT_HEAD" in snapshot["git_head_source"]


def test_resumed_run_cannot_append_through_child_symlink(tmp_path, make_symlink):
    from tests.test_math_ext_exp2 import harness, run

    h = harness(tmp_path)
    h[0].run_dir.mkdir(parents=True, exist_ok=True)
    original = tmp_path / "original_decisions.jsonl"
    original.write_text("do not touch\n", encoding="utf-8", newline="\n")
    make_symlink(h[0].run_dir / "decisions.jsonl", original)
    with pytest.raises(IntegrityError, match="symlinks"):
        run(h)
    assert not h[2].calls and original.read_text(encoding="utf-8") == "do not touch\n"


def test_constructed_loop_uses_real_local_math_verify_only_after_four_rounds(tmp_path):
    from pipelines.math_ext_e import ExtensionMathEvaluator
    from tests.test_math_ext_exp2 import harness, run

    from rafr.logging.writer import read_jsonl

    h = harness(
        tmp_path,
        arms=(Arm.A1,),
        responder=lambda request: json.dumps({"action": "rewrite", "revised_text": r"\boxed{2}"}),
    )
    setup = h[0]
    setup.starts[0].x0 = "Compute 1+1."
    setup.starts[0].text = r"\boxed{3}"
    setup.cohort = "e_incorrect"
    setup.initial_grades = {"q1": {"status": "incorrect", "task_success": False}}
    trajectories = run(h)
    assert len(h[2].calls) == 4
    assert not (setup.run_dir / "extension_e_evaluations.jsonl").exists()
    evaluator = ExtensionMathEvaluator(
        setup.run_dir,
        questions={"q1": "Compute 1+1."},
        private_references={"q1": "2"},
        source_identity={"kind": "constructed_expression_fixture"},
        source_root=tmp_path / "original_source",
    )
    rows = grade_math_exp2(setup, trajectories, evaluator, e_identity=evaluator.identity)
    assert rows[0]["task_success"] and rows[0]["terminal_status"] == "correct"
    assert len(list(read_jsonl(setup.run_dir / "extension_e_evaluations.jsonl"))) == 1
    assert rows[0]["initial_status"] == "incorrect"
    assert len(h[2].calls) == 4  # E adds no model call.


@pytest.mark.parametrize("field", ["text", "x0"])
def test_e_refuses_changed_initial_state_before_writing_or_regrading(tmp_path, field):
    from tests.test_math_ext_exp2 import harness, run

    h = harness(tmp_path)
    trajectories = run(h)
    setattr(h[0].starts[0], field, "different initial input")
    with pytest.raises(IntegrityError, match="setup differs"):
        grade_math_exp2(
            h[0], trajectories, lambda *a: pytest.fail("E must not run"), e_identity={"fixture": 1}
        )
    assert not (h[0].run_dir / "e_identity.json").exists()
    assert not (h[0].run_dir / "math_judgments.jsonl").exists()


@pytest.mark.parametrize("change", ["recorded_head", "current_code"])
def test_snapshot_refuses_changed_code_or_commit_on_resume(tmp_path, monkeypatch, change):
    from pipelines import math_ext_snapshot

    channel = {"levels": {"low": {"v1": 0.1, "v2": 0.1}}}
    out = tmp_path / "snapshot"
    snapshot_configs(out, {"domain": "math"}, channel, identity={"fixture": 1})
    path = out / "snapshot.json"
    if change == "recorded_head":
        data = json.loads(path.read_text(encoding="utf-8"))
        data["git_head"] = "f" * 40
        path.write_text(json.dumps(data), encoding="utf-8", newline="\n")
    else:
        actual = math_ext_snapshot._execution_identity()
        monkeypatch.setattr(
            math_ext_snapshot,
            "_execution_identity",
            lambda: {**actual, "execution_sha256": "f" * 64},
        )
    before = path.read_bytes()
    with pytest.raises(IntegrityError, match="snapshot identity changed"):
        snapshot_configs(out, {"domain": "math"}, channel, identity={"fixture": 1})
    assert path.read_bytes() == before
