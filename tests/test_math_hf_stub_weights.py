"""The judge and ORM through the real transformers path, on tiny random stub weights.

Nothing is downloaded: a word-level tokenizer and two 2-layer Llama checkpoints are built
in a temporary directory, pinned by a ``rafr_model_manifest.json`` and loaded from there
with ``local_files_only``. The scores mean nothing; what is checked is the plumbing --
loading, dtype, verdict-token positioning, left padding, fixed batch shape with a filled
last batch, the ORM's own last-token pooling and the score transforms. Skipped where
torch / transformers are not installed (CI installs neither).
"""

from __future__ import annotations

import hashlib
import json
import math

import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")
tokenizers = pytest.importorskip("tokenizers")

from math_v2_helpers import make_harness  # noqa: E402
from rafr.contracts.schemas import Split  # noqa: E402
from rafr.domains.math import models  # noqa: E402
from rafr.domains.math.models import (  # noqa: E402
    JUDGE_IMPLEMENTATION,
    ORM_IMPLEMENTATION,
    MathModelError,
    VerifierSpec,
    build_verifier,
    validate_verifier_pair,
)
from rafr.logging.writer import read_jsonl  # noqa: E402

PROMPT_TEXT, PROMPT_SHA = models.read_judge_prompt()
REVISION, TOKENIZER_REVISION = "a" * 40, "b" * 40
ITEMS = [
    ("Compute 1 + 1 .", "One plus one is two . \\boxed{2}"),
    ("Compute 2 + 2 .", "Two plus two is five . \\boxed{5}"),
    ("Compute 3 + 3 .", "Three plus three is six . \\boxed{6}"),
]
CHAT_TEMPLATE = (
    "{% for m in messages %}<s> {{ m['role'] }} : {{ m['content'] }} </s> {% endfor %}"
    "{% if add_generation_prompt %}<s> assistant : {% endif %}"
)


def _write_manifest(root, model_id, *, skip=()):
    files = {
        path.relative_to(root).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(root.rglob("*"))
        if path.is_file() and path.name not in skip
    }
    record = {
        "model_id": model_id,
        "revision": REVISION,
        "tokenizer_revision": TOKENIZER_REVISION,
        "files": files,
    }
    (root / "rafr_model_manifest.json").write_text(json.dumps(record), encoding="utf-8")


@pytest.fixture(scope="module")
def stub_dirs(tmp_path_factory):
    words = ["<pad>", "<s>", "</s>", "<unk>", "Yes", "No", "user", "assistant", ":"]
    for text in (PROMPT_TEXT, *(part for item in ITEMS for part in item)):
        for word in text.split():
            if word not in words and word not in ("{problem}", "{solution}"):
                words.append(word)
    vocab = {word: index for index, word in enumerate(words)}
    backend = tokenizers.Tokenizer(tokenizers.models.WordLevel(vocab, unk_token="<unk>"))
    backend.pre_tokenizer = tokenizers.pre_tokenizers.WhitespaceSplit()
    tokenizer = transformers.PreTrainedTokenizerFast(
        tokenizer_object=backend,
        bos_token="<s>",
        eos_token="</s>",
        pad_token="<pad>",
        unk_token="<unk>",
    )
    tokenizer.chat_template = CHAT_TEMPLATE
    config = transformers.LlamaConfig(
        vocab_size=len(vocab),
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=128,
        pad_token_id=vocab["<pad>"],
        bos_token_id=vocab["<s>"],
        eos_token_id=vocab["</s>"],
    )
    root = tmp_path_factory.mktemp("stub_models")
    torch.manual_seed(20261004)
    judge_dir, orm_dir = root / "judge", root / "orm"
    tokenizer.save_pretrained(judge_dir)
    transformers.LlamaForCausalLM(config).save_pretrained(judge_dir)
    config.num_labels = 1
    tokenizer.save_pretrained(orm_dir)
    transformers.LlamaForSequenceClassification(config).save_pretrained(orm_dir)
    _write_manifest(judge_dir, "stub/tiny-judge")
    _write_manifest(orm_dir, "stub/tiny-reward")
    return {"judge": judge_dir, "orm": orm_dir, "vocab": vocab}


@pytest.fixture(autouse=True)
def restore_torch_switches():
    """The adapters set the detox determinism switches globally; undo them per test."""
    state = (
        torch.are_deterministic_algorithms_enabled(),
        torch.is_deterministic_algorithms_warn_only_enabled(),
        torch.backends.cudnn.deterministic,
        torch.backends.cudnn.benchmark,
    )
    yield
    torch.use_deterministic_algorithms(state[0], warn_only=state[1])
    torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark = state[2:]


def common(path, dtype):
    return {
        "tokenizer_revision": TOKENIZER_REVISION,
        "add_special_tokens": False,
        "dtype": dtype,
        "batch_size": 2,
        "max_input_tokens": 96,
        "device_map": "cpu",
        "license_reviewed": True,
        "license_review_reference": "stub weights built by this test; no real checkpoint",
        "local_path": str(path),
    }


def judge_spec(stub_dirs, dtype="float32", **settings):
    return VerifierSpec(
        form="llm_judge",
        vendor="stub-vendor-a",
        base_model="stub-llama-a",
        model_id="stub/tiny-judge",
        revision=REVISION,
        implementation=JUDGE_IMPLEMENTATION,
        settings={
            **common(stub_dirs["judge"], dtype),
            "input_format": "chat_template",
            "yes_token": "Yes",
            "no_token": "No",
            "prompt_sha256": PROMPT_SHA,
            **settings,
        },
    )


def orm_spec(stub_dirs, dtype="float32", **settings):
    return VerifierSpec(
        form="orm",
        vendor="stub-vendor-b",
        base_model="stub-llama-b",
        model_id="stub/tiny-reward",
        revision=REVISION,
        implementation=ORM_IMPLEMENTATION,
        settings={
            **common(stub_dirs["orm"], dtype),
            "input_format": "chat_template",
            "system_prompt": "grade",
            **settings,
        },
    )


def _unpadded(verifier, result):
    """The same input alone, without any padding: the reference the adapter must match."""
    ids = verifier._tokenizer.encode(result["serialized_input"], add_special_tokens=False)
    ids = ids + result.get("answer_prefix_token_ids", [])
    assert (
        hashlib.sha256(json.dumps(ids, separators=(",", ":")).encode()).hexdigest()
        == (result["token_ids_hash"])
    )
    with torch.inference_mode():
        return verifier._model(input_ids=torch.tensor([ids])).logits.float()


@pytest.mark.parametrize("dtype", ["float32", "bfloat16"])
def test_judge_on_stub_weights(stub_dirs, dtype):
    verifier = build_verifier(judge_spec(stub_dirs, dtype))
    together = verifier.score_batch(ITEMS)
    assert verifier._model.dtype == getattr(torch, dtype)
    yes, no = stub_dirs["vocab"]["Yes"], stub_dirs["vocab"]["No"]
    for item, result in zip(ITEMS, together, strict=True):
        assert 0 < result["error_score"] < 1 and result["error_score"] == result["p_no"]
        assert result["verdict_token_ids_yes_no"] == [yes, no]
        assert result["actual_dtype"] == f"torch.{dtype}"
        assert result["execution_mode"] == "local_model"
        # fixed shape: the same score alone (last batch filled with copies) as in a batch
        assert verifier.score(*item)["error_score"] == result["error_score"]
        logits = _unpadded(verifier, result)[0, -1]
        reference = torch.softmax(logits[[yes, no]].double(), dim=0)[1].item()
        tolerance = 1e-5 if dtype == "float32" else 2e-2
        assert result["p_no"] == pytest.approx(reference, abs=tolerance)
    swapped = verifier.score_batch([ITEMS[2], ITEMS[0]])
    assert [r["error_score"] for r in swapped] == [
        together[2]["error_score"],
        together[0]["error_score"],
    ]
    card = verifier.verifier_card()
    assert card["input"]["prompt_sha256"] == PROMPT_SHA
    assert card["score"]["verdict_tokens"] == {"yes": [yes, "Yes"], "no": [no, "No"]}
    assert card["batching"]["fixed_shape"] is True
    verifier.close()


@pytest.mark.parametrize("dtype", ["float32", "bfloat16"])
def test_orm_on_stub_weights(stub_dirs, dtype):
    verifier = build_verifier(orm_spec(stub_dirs, dtype))
    together = verifier.score_batch(ITEMS)
    for item, result in zip(ITEMS, together, strict=True):
        assert 0 < result["error_score"] < 1
        assert result["serialized_input"].startswith("<s> system : grade </s> <s> user :")
        assert verifier.score(*item)["error_score"] == result["error_score"]
        logit = _unpadded(verifier, result)[0, 0].item()
        tolerance = 1e-5 if dtype == "float32" else 2e-2
        assert result["logit"] == pytest.approx(logit, abs=tolerance)
        assert result["error_score"] == pytest.approx(1 - 1 / (1 + math.exp(-logit)), abs=tolerance)
    verifier.close()


def test_stub_pair_scores_pipeline_states_without_references(stub_dirs, tmp_path, monkeypatch):
    pair = validate_verifier_pair(
        [judge_spec(stub_dirs, "bfloat16", batch_size=1), orm_spec(stub_dirs, "bfloat16")]
    )
    h = make_harness(tmp_path, monkeypatch, n=60, fit_n=10, gate_n=10, reserve_n=10)
    h.pipeline.generate()
    h.pipeline.split()
    h.pipeline.verifiers = [build_verifier(spec) for spec in pair]
    h.pipeline.score(Split.D_FIT)
    details = list(read_jsonl(h.root / "verifier_details.jsonl"))
    assert len(details) == 20 and not any("PRIVATE_GOLD" in str(row) for row in details)
    assert {row["implementation"] for row in details} == {JUDGE_IMPLEMENTATION, ORM_IMPLEMENTATION}
    assert all(0 <= row["score"] <= 1 for row in read_jsonl(h.root / "scores.jsonl"))


class OneByOne:
    """The adapter seen through score() only: the pipeline's one-by-one path."""

    def __init__(self, adapter):
        self.adapter = adapter

    def score(self, problem, solution):
        return self.adapter.score(problem, solution)


def test_stub_pipeline_scores_in_batches_bit_for_bit_as_one_by_one(
    stub_dirs, tmp_path, monkeypatch
):
    pair = [judge_spec(stub_dirs, "bfloat16"), orm_spec(stub_dirs, "bfloat16", batch_size=3)]
    written = {}
    for mode in ("batched", "one_by_one"):
        root = tmp_path / mode
        root.mkdir()
        h = make_harness(root, monkeypatch, n=60, fit_n=10, gate_n=10, reserve_n=10)
        h.pipeline.generate()
        h.pipeline.split()
        verifiers = [build_verifier(spec) for spec in pair]
        h.pipeline.verifiers = verifiers if mode == "batched" else [OneByOne(v) for v in verifiers]
        h.pipeline.score(Split.D_FIT)
        written[mode] = [
            (row["state_id"], row["verifier_id"], row["score"])
            for row in read_jsonl(h.root / "scores.jsonl")
        ]
        for verifier in verifiers:
            verifier.close()
    assert len(written["batched"]) == 20
    assert written["batched"] == written["one_by_one"]


def test_manifest_must_cover_the_chat_template_file(stub_dirs, tmp_path):
    import shutil

    copy = tmp_path / "judge"
    shutil.copytree(stub_dirs["judge"], copy)
    _write_manifest(
        copy, "stub/tiny-judge", skip=("chat_template.jinja", "rafr_model_manifest.json")
    )
    verifier = build_verifier(judge_spec({"judge": copy}))
    with pytest.raises(MathModelError, match="cover all runtime"):
        verifier.load()
