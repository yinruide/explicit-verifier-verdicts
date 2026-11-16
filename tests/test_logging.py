"""logging: JSONL writer, snapshot contents (II.10), freeze-tag guard."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from rafr.contracts.schemas import Score
from rafr.logging import snapshot
from rafr.logging.schema import ROW_MODELS, RecordKind, stream_filename
from rafr.logging.writer import JsonlWriter, read_jsonl, read_jsonl_models


def test_writer_appends_and_flushes(tmp_path: Path):
    p = tmp_path / "x.jsonl"
    with JsonlWriter(p) as w:
        w.write({"a": 1})
        w.write(Score(state_id="s", verifier_id="v", model_version="m", score=0.5))
        assert p.read_text(encoding="utf-8").count("\n") == 2  # flushed per row
    with JsonlWriter(p) as w:
        w.write({"a": 2})
    rows = list(read_jsonl(p))
    assert [r.get("a") for r in rows] == [1, None, 2]
    assert rows[1]["score"] == 0.5


def test_read_models(tmp_path: Path):
    p = tmp_path / "s.jsonl"
    with JsonlWriter(p) as w:
        w.write(Score(state_id="s", verifier_id="v", model_version="m", score=0.5))
    got = read_jsonl_models(p, Score)
    assert isinstance(got[0], Score) and got[0].score == 0.5


def test_stream_names():
    assert stream_filename(RecordKind.DECISION) == "decisions.jsonl"
    assert ROW_MODELS[RecordKind.SCORE] is Score


def test_snapshot_has_git_head_and_config_copy(tmp_path: Path):
    snap = snapshot.write_snapshot(
        tmp_path / "run", step="test", configs={"k": "v"}, extra={"note": 1}
    )
    assert len(snap["git_head"]) == 40
    assert isinstance(snap["git_dirty"], bool)
    assert snap["configs"] == {"k": "v"} and snap["note"] == 1
    on_disk = json.loads((tmp_path / "run" / "snapshot.json").read_text(encoding="utf-8"))
    assert on_disk["git_head"] == snap["git_head"]
    assert (tmp_path / "run" / "configs" / "models.yaml").exists()


def test_snapshot_never_contains_api_keys(tmp_path: Path):
    snap = snapshot.write_snapshot(tmp_path / "run", step="test", configs={})
    text = json.dumps(snap)
    assert "OPENAI_API_KEY" not in text and "ANTHROPIC_API_KEY" not in text


def test_freeze_tag_name_guard():
    with pytest.raises(ValueError):
        snapshot.create_freeze_tag("not-a-freeze-tag", "msg")
    with pytest.raises(ValueError):
        snapshot.create_freeze_tag("tau-frozen", "   ")
