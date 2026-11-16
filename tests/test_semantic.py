"""Semantic preservation (rafr.domains.detox.semantic): the TextDetox 2024 official evaluation,
pinned in configs/exp2.yaml, and the preregistered cutpoint computed on D_fit (the authors
2026-09-28). Offline: fake encoders / models only; torch and sacrebleu tests skip when absent."""

from __future__ import annotations

import json
import re
import shutil
import zlib
from pathlib import Path

import numpy as np
import pytest

from rafr.contracts.schemas import Split, State
from rafr.core import paths
from rafr.core.config import get, load_config
from rafr.data import labels as L
from rafr.domains.detox import semantic
from rafr.logging.writer import read_jsonl


class FakeEncoder:
    """Deterministic 8-dim embedding per text; records every encode call."""

    def __init__(self) -> None:
        self.calls: list[list[str]] = []

    def encode(self, sentences: list[str]) -> np.ndarray:
        self.calls.append(list(sentences))
        rows = []
        for s in sentences:
            rng = np.random.default_rng(zlib.crc32(s.encode("utf-8")))
            rows.append(rng.normal(size=8).astype(np.float32))
        return np.stack(rows)


def _cos_sim(a: np.ndarray, b: np.ndarray) -> float:
    from scipy.spatial.distance import cosine

    return float(1 - cosine(a, b))


def test_the_protocol_is_pinned_in_exp2_yaml():
    cfg = semantic.build_config(load_config("exp2"))
    assert cfg.quality == "similarity" and cfg.device == "cpu"
    assert cfg.sim_model == "sentence-transformers/LaBSE"
    assert cfg.sta_model == "textdetox/xlmr-large-toxicity-classifier"  # never V1 (II.2.1)
    assert cfg.sta_model != get(load_config("models"), "verifiers.v1.id")
    assert len(cfg.sim_revision) == len(cfg.sta_revision) == 40
    assert cfg.sta_target_label == 0 and cfg.sta_batch_size == 32 and cfg.sta_max_length == 512
    assert cfg.sim_library.startswith("sentence-transformers==")
    assert cfg.fl_metric == "chrf" and cfg.fl_library.startswith("sacrebleu==")
    assert "88037f4602d013698c31985764a17f086baa451f" in cfg.reference
    rule = get(load_config("exp2"), "analysis.semantic_preservation.cutpoint_rule")
    assert rule == {"population": "d_fit", "quantile": 0.10, "method": "linear"}


def test_a_bad_protocol_block_is_refused():
    exp2 = load_config("exp2")
    exp2["analysis"]["semantic_preservation"]["protocol"]["quality"] = "fluency"
    with pytest.raises(semantic.SemanticError, match="similarity"):
        semantic.build_config(exp2)
    exp2 = load_config("exp2")
    exp2["analysis"]["semantic_preservation"]["protocol"]["similarity"]["revision"] = "main"
    with pytest.raises(semantic.SemanticError, match="40-hex"):
        semantic.build_config(exp2)


def test_similarity_is_the_reference_evaluate_sim():
    """evaluate.py: per batch of min(32, n) pairs, ONE encode call over originals + rewrites,
    then 1 - scipy cosine pair by pair."""
    enc = FakeEncoder()
    o = [f"orig {i}" for i in range(5)]
    r = [f"rew {i}" for i in range(5)]
    got = semantic.similarity(enc, o, r, batch_size=2)
    assert enc.calls == [o[0:2] + r[0:2], o[2:4] + r[2:4], o[4:5] + r[4:5]]
    want = [_cos_sim(enc.encode([a])[0], enc.encode([b])[0]) for a, b in zip(o, r, strict=True)]
    assert got == pytest.approx(want, abs=1e-6)
    enc.calls.clear()
    semantic.similarity(enc, o, r)  # default 32 -> min(32, 5): one call
    assert enc.calls == [o + r]
    assert semantic.similarity(enc, [], []) == []
    with pytest.raises(semantic.SemanticError):
        semantic.similarity(enc, o, r[:4])


def test_quality_is_one_pair_per_encode_call_and_loads_lazily():
    enc = FakeEncoder()
    loads: list[str] = []

    def loader(cfg):
        loads.append(cfg.sim_model)
        return enc

    q = semantic.QualityScorer(semantic.build_config(load_config("exp2")), loader=loader)
    assert loads == []  # building the scorer loads nothing (the pod builds none at all)
    v = q("an original", "a rewrite")
    assert loads == ["sentence-transformers/LaBSE"] and enc.calls == [["an original", "a rewrite"]]
    assert v == semantic.pair_quality(FakeEncoder(), "an original", "a rewrite")
    q("x", "y")
    assert len(loads) == 1


def test_library_versions_are_checked(monkeypatch):
    monkeypatch.setattr(semantic, "installed_version", lambda name: "9.9.9")
    with pytest.raises(semantic.SemanticError, match="pinned 6.1.0"):
        semantic.check_versions(["sentence-transformers==6.1.0"])
    monkeypatch.setattr(semantic, "installed_version", lambda name: None)
    with pytest.raises(semantic.SemanticError, match="not installed"):
        semantic.check_versions(["sacrebleu==2.6.0"])
    monkeypatch.setattr(semantic, "installed_version", lambda name: "6.1.0")
    assert semantic.check_versions(["sentence-transformers==6.1.0"]) == {
        "sentence-transformers": "6.1.0"
    }


def test_style_accuracy_is_the_softmax_of_label_0_in_batches_of_32():
    torch = pytest.importorskip("torch")

    class Batch(dict):
        def to(self, device):
            return self

    seen: list[dict] = []

    def tokenizer(texts, **kw):
        seen.append({"n": len(texts), **kw})
        return Batch(idx=torch.tensor([len(t) for t in texts], dtype=torch.float32))

    class Model:
        device = "cpu"

        def __call__(self, idx):
            logits = torch.stack([idx / 10, -idx / 10], dim=1)
            return type("Out", (), {"logits": logits})()

    texts = [("x" * (i % 7 + 1)) for i in range(33)]
    got = semantic.style_accuracy(Model(), tokenizer, texts)
    assert [s["n"] for s in seen] == [32, 1]
    assert all(
        s["padding"] and s["truncation"] and s["max_length"] == 512 and s["return_tensors"] == "pt"
        for s in seen
    )
    lens = torch.tensor([len(t) for t in texts], dtype=torch.float32)
    want = torch.softmax(torch.stack([lens / 10, -lens / 10], dim=1), -1)[:, 0]
    assert got == pytest.approx([float(x) for x in want], abs=1e-7)


def test_fluency_is_sentence_chrf_over_100_and_joint_multiplies():
    sacrebleu = pytest.importorskip("sacrebleu")
    fl = semantic.fluency(["the cat sat", "a dog"], ["the cat sat", "the cat sat"])
    assert fl[0] == pytest.approx(1.0)
    ref = sacrebleu.CHRF().sentence_score(hypothesis="a dog", references=["the cat sat"]).score
    assert fl[1] == pytest.approx(ref / 100)
    assert semantic.joint([0.5, 1.0], [0.8, 0.5], [1.0, 0.2]) == pytest.approx([0.4, 0.1])
    with pytest.raises(semantic.SemanticError):
        semantic.fluency(["a"], [])


# ------------------------------------------------------------------------------------------
# the cutpoint command
# ------------------------------------------------------------------------------------------


def _write_states(split: Split, n: int, prefix: str) -> list[State]:
    paths.DATA_POOLS.mkdir(parents=True, exist_ok=True)
    states = [
        State(state_id=f"{prefix}{i}-g1", x0_id=f"{prefix}{i}", x0=f"orig {prefix}{i}",
              text=f"rewrite {prefix}{i} {'x' * (i % 5)}")
        for i in range(n)
    ]  # fmt: skip
    with (paths.DATA_POOLS / f"states_{split.value}.jsonl").open("w", encoding="utf-8") as fh:
        for s in states:
            fh.write(s.model_dump_json() + "\n")
    return states


@pytest.fixture
def cut_env(sandbox, monkeypatch, tmp_path):
    """D_fit states, a copy of configs/ (the command writes exp2.yaml), a fake LaBSE, and
    traps on the label gate and on every other split's states."""
    from pipelines import step12_exp2 as s12

    d_fit = _write_states(Split.D_FIT, 40, "f")
    _write_states(Split.D_EXP1, 5, "e")
    cfg_dir = tmp_path / "configs_copy"
    shutil.copytree(paths.CONFIGS, cfg_dir)
    # the copy starts where the repo stood before the real value was computed
    exp2_yaml = cfg_dir / "exp2.yaml"
    exp2_yaml.write_text(
        re.sub(
            r"^    cutpoint: .*$", "    cutpoint: TBD", exp2_yaml.read_text("utf-8"), flags=re.M
        ),
        encoding="utf-8",
        newline="\n",
    )
    monkeypatch.setattr(paths, "CONFIGS", cfg_dir)

    def no_labels(*a, **k):
        raise AssertionError("the cutpoint read y*")

    real_load = s12.io.load_states

    def only_d_fit(split):
        if split is not Split.D_FIT:
            raise AssertionError(f"the cutpoint touched {split.value}")
        return real_load(split)

    monkeypatch.setattr(L, "read_labels", no_labels)
    monkeypatch.setattr(s12.io, "load_states", only_d_fit)
    monkeypatch.setattr(s12, "load_similarity_encoder", lambda scfg: FakeEncoder())
    return s12, d_fit, cfg_dir


def test_the_cutpoint_is_the_10pct_quantile_of_d_fit_quality_written_into_exp2_yaml(cut_env):
    s12, d_fit, cfg_dir = cut_env
    before = (cfg_dir / "exp2.yaml").read_text(encoding="utf-8")
    assert "    cutpoint: TBD\n" in before
    lines: list[str] = []
    assert s12.main(["cutpoint"], echo=lines.append) == 0
    q = [semantic.pair_quality(FakeEncoder(), s.x0, s.text) for s in d_fit]
    want = float(np.quantile(q, 0.10))
    after = (cfg_dir / "exp2.yaml").read_text(encoding="utf-8")
    assert f"    cutpoint: {want!r}\n" in after and "cutpoint: TBD" not in after
    # only that line changed: every comment is kept
    assert [ln for ln in before.splitlines() if "cutpoint: TBD" not in ln] == [
        ln for ln in after.splitlines() if not ln.startswith("    cutpoint: ")
    ]
    assert get(load_config("exp2"), "analysis.semantic_preservation.cutpoint") == want
    rep = json.loads((paths.DATA / "diagnostics" / s12.CUTPOINT_REPORT).read_text("utf-8"))
    assert rep["cutpoint"] == want and rep["n"] == 40 and rep["labels_read"] is False
    assert rep["population"] == "d_fit" and rep["quantile"] == 0.1 and rep["method"] == "linear"
    assert set(rep["five_number_summary"]) == {"min", "q1", "median", "q3", "max"}
    assert rep["five_number_summary"]["min"] == min(q) and rep["five_number_summary"]["max"] == max(
        q
    )
    assert rep["protocol"]["similarity"]["model"] == "sentence-transformers/LaBSE"
    assert rep["reference_batching_max_abs_diff"] < 1e-6
    assert {"git_head", "git_dirty", "git_describe"} <= set(rep)
    rows = list(read_jsonl(paths.DATA / "diagnostics" / s12.CUTPOINT_SCORES))
    assert [r["state_id"] for r in rows] == [s.state_id for s in d_fit]
    assert set(rows[0]) == {"state_id", "x0_id", "quality"}  # no text, no label
    assert any("cutpoint = 0.1 quantile (linear)" in ln for ln in lines)
    # a rerun recomputes and must agree; a different value on file is refused
    assert s12.main(["cutpoint"], echo=lines.append) == 0
    assert "recomputed identical" in lines[-3]
    (cfg_dir / "exp2.yaml").write_text(
        after.replace(f"cutpoint: {want!r}", "cutpoint: 0.5"), encoding="utf-8", newline="\n"
    )
    with pytest.raises(SystemExit, match="already holds cutpoint 0.5"):
        s12.main(["cutpoint"], echo=lines.append)


def test_write_cutpoint_needs_exactly_one_tbd_line(tmp_path):
    from pipelines import step12_exp2 as s12

    p = tmp_path / "exp2.yaml"
    p.write_text("analysis:\n  semantic_preservation:\n    cutpoint: 0.3\n", encoding="utf-8")
    with pytest.raises(SystemExit, match="found 0"):
        s12.write_cutpoint(p, 0.4)
    p.write_text("analysis:\n  semantic_preservation:\n    cutpoint: TBD\n", encoding="utf-8")
    s12.write_cutpoint(p, 0.71234)
    assert load_config("exp2", config_dir=tmp_path)["analysis"]["semantic_preservation"] == {
        "cutpoint": 0.71234
    }


def test_the_repo_never_types_the_cutpoint_by_hand():
    """The value in exp2.yaml is either TBD or the one the cutpoint report recorded."""
    cut = get(load_config("exp2"), "analysis.semantic_preservation.cutpoint")
    report = Path(paths.repo_root()) / "data" / "diagnostics" / "semantic_cutpoint.json"
    if cut == "TBD":
        return
    if not report.is_file():  # CI has no data/: the report stays on the laptop
        pytest.skip("no data/diagnostics/semantic_cutpoint.json here")
    assert json.loads(report.read_text(encoding="utf-8"))["cutpoint"] == cut
