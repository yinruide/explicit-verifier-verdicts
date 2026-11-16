"""CLI authorization and bootstrap checks; synthetic preparation uses original-layout fixtures."""

from __future__ import annotations

import json
import subprocess
import sys

from pipelines import math_ext_run as cli


def invoke(*args, cwd=None):
    return subprocess.run(
        [sys.executable, str(cli.ROOT / "pipelines/math_ext_run.py"), *args],
        cwd=cwd,
        capture_output=True,
        text=True,
        check=False,
    )


def test_help_bootstraps_current_checkout_without_work_or_provider_import(tmp_path):
    result = invoke("--help", cwd=tmp_path)
    assert result.returncode == 0 and "preflight" in result.stdout
    assert list(tmp_path.iterdir()) == []
    script = (
        "import runpy,sys; "
        f"m=runpy.run_path({str(cli.ROOT / 'pipelines/math_ext_run.py')!r}); "
        "assert not any(n.startswith('rafr.llm.providers') for n in sys.modules); "
        f"assert sys.path[0]=={str(cli.ROOT / 'src')!r}; "
        "assert 'pipelines.math_ext_data' not in sys.modules"
    )
    checked = subprocess.run(
        [sys.executable, "-c", script], cwd=tmp_path, capture_output=True, text=True, check=False
    )
    assert checked.returncode == 0, checked.stderr
    assert list(tmp_path.iterdir()) == []


def test_unbound_preflight_fails_without_writing(tmp_path, math_ext_unbound):
    # The repository config is bound (2026-11-05); the unbound one is written apart from cwd.
    import yaml

    config = tmp_path / "config" / "ext.yaml"
    config.parent.mkdir()
    config.write_text(yaml.safe_dump(math_ext_unbound()), encoding="utf-8")
    cwd = tmp_path / "cwd"
    cwd.mkdir()
    result = invoke(
        "preflight",
        "--config",
        str(config),
        "--stage",
        "offline_stub",
        "--operation",
        "calibration",
        cwd=cwd,
    )
    assert result.returncode == 2
    body = json.loads(result.stdout)
    assert body["missing_bindings"] == ["source_manifest"]
    assert body["read_only"] and body["provider_calls"] == 0
    assert "calibration.frozen_sha256" not in body["missing_bindings"]
    assert list(cwd.iterdir()) == []


def test_real_prepare_refuses_even_before_opening_source(tmp_path):
    for command in ("calibrate", "cohort"):
        out = tmp_path / command
        result = invoke(
            command,
            "--manifest",
            str(tmp_path / "missing.json"),
            "--stage",
            "real_prepare",
            "--out",
            str(out),
        )
        assert result.returncode == 2
        assert "not authorized/enabled" in result.stdout
        assert not out.exists()


def test_manifest_tbd_hash_rejected_before_source_open(tmp_path):
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "source_root": str(tmp_path / "not-created"),
                "master_seed": 1,
                "verifier_ids": ["v1", "v2"],
                "expected_sha256": {name: "TBD" for name in cli.BASE_BINDINGS},
            }
        ),
        encoding="utf-8",
    )
    result = invoke(
        "preflight", "--manifest", str(manifest), "--stage", "offline_stub", "--operation", "cohort"
    )
    assert result.returncode == 2 and "TBD/invalid SHA256" in result.stdout
    assert set(tmp_path.iterdir()) == {manifest}


def test_synthetic_preparation_with_real_source_reader_is_read_only(tmp_path, monkeypatch, capsys):
    from tests.math_v2_helpers import make_harness, prepare_fit
    from tests.test_math_ext_data import hashes

    from rafr.domains.math.storage import file_hash

    (tmp_path / "source").mkdir()
    h = make_harness(tmp_path / "source", monkeypatch)
    prepare_fit(h)
    manifest = tmp_path / "source_manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "source_root": str(h.root),
                "expected_sha256": hashes(h.root),
                "master_seed": 20261002,
                "verifier_ids": ["fixture-V1", "fixture-V2"],
            }
        ),
        encoding="utf-8",
    )
    before = {
        p.relative_to(h.root).as_posix(): file_hash(p) for p in h.root.rglob("*") if p.is_file()
    }
    isolated = (
        "import runpy,sys; "
        f"m=runpy.run_path({str(cli.ROOT / 'pipelines/math_ext_run.py')!r}); "
        f"r=m['preflight'](manifest={str(manifest)!r},operation='calibration'); "
        "assert r['status']=='READY_FOR_REVIEW', r; "
        "assert not any(n.startswith('rafr.llm.providers') for n in sys.modules)"
    )
    checked = subprocess.run(
        [sys.executable, "-c", isolated], capture_output=True, text=True, check=False
    )
    assert checked.returncode == 0, checked.stderr
    assert (
        cli.main(
            [
                "preflight",
                "--manifest",
                str(manifest),
                "--stage",
                "offline_stub",
                "--operation",
                "calibration",
            ]
        )
        == 0
    )
    report = json.loads(capsys.readouterr().out)
    assert report["status"] == "READY_FOR_REVIEW" and not report["execution_authorized"]
    assert report["missing_bindings"] == []
    assert (
        cli.main(
            [
                "calibrate",
                "--manifest",
                str(manifest),
                "--stage",
                "offline_stub",
                "--out",
                str(tmp_path / "calibration"),
                "--n-boot",
                "5",
            ]
        )
        == 0
    )
    calibration = json.loads(capsys.readouterr().out)
    assert calibration["stage"] == "offline_stub" and not calibration["scientifically_frozen"]
    assert (
        cli.main(
            [
                "cohort",
                "--manifest",
                str(manifest),
                "--stage",
                "offline_stub",
                "--out",
                str(tmp_path / "cohort"),
                "--n",
                "2",
            ]
        )
        == 0
    )
    cohort = json.loads(capsys.readouterr().out)
    assert cohort["n"] == 2 and not cohort["live_executed"]
    assert before == {
        p.relative_to(h.root).as_posix(): file_hash(p) for p in h.root.rglob("*") if p.is_file()
    }
    # These real preparation helpers already reject reuse of an existing output directory.
    assert (
        cli.main(
            [
                "cohort",
                "--manifest",
                str(manifest),
                "--stage",
                "offline_stub",
                "--out",
                str(tmp_path / "cohort"),
                "--n",
                "2",
            ]
        )
        == 2
    )
    capsys.readouterr()


def test_replay_dispatch_preserves_explicit_fixture_scope(monkeypatch, capsys):
    from pipelines import math_ext_regression

    source = object()
    calls = []
    monkeypatch.setattr(cli, "load_source", lambda path: source)

    def replay(value, *, fixture=False):
        calls.append((value, fixture))
        return {
            "execution": "offline_fixture" if fixture else "read_only_real_artifact_replay",
            "provider_calls": 0,
        }

    monkeypatch.setattr(math_ext_regression, "replay_original", replay)
    assert cli.main(["replay", "--manifest", "fixture.json", "--fixture"]) == 0
    assert json.loads(capsys.readouterr().out)["execution"] == "offline_fixture"
    assert calls == [(source, True)]


def test_model_commands_require_explicit_arguments_and_pod_is_separate(tmp_path):
    for command in ("live", "gpu"):
        result = invoke(command, cwd=tmp_path)
        assert result.returncode == 2 and "invalid choice" in result.stderr
    assert list(tmp_path.iterdir()) == []


def test_execution_commands_have_no_implicit_stage(tmp_path):
    for command in ("loop", "judge", "exp1", "ceiling", "starts"):
        result = invoke(command, cwd=tmp_path)
        assert result.returncode == 2 and "required" in result.stderr
    assert list(tmp_path.iterdir()) == []
