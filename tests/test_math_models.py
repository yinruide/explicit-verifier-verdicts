"""Exercise author-specific adapter logic at fake tensor/provider boundaries only.

No checkpoint is downloaded or loaded. These tests are not empirical PRM validation.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import math
import re
from dataclasses import asdict, replace
from types import SimpleNamespace

import numpy as np
import pytest

from rafr.domains.math import models
from rafr.domains.math.models import (
    ADAPTER_VERSION,
    MODEL_REPOSITORIES,
    PUBLISHED_REVISIONS,
    QWEN_SYSTEM,
    MathModelError,
    ModelAccessNotReviewedError,
    ModelInputError,
    ModelOutputError,
    PRMConfig,
    PRMVerifier,
    aggregate_steps,
    split_solution_steps,
)


class FakeTokenizer:
    """Deliberately different label IDs from the published tokenizer examples."""

    chat_template = "fake-test-template-v1"
    marker_ids = {"<extra_0>": 17, "ки": 19}

    def __init__(self):
        self.tokens = {"<bos>": 1, " ": 2, "+": 7, "-": 11, **self.marker_ids}
        self.inverse = {value: key for key, value in self.tokens.items()}
        self.calls = []
        self.messages = None

    def encode(self, text, *, add_special_tokens=True, truncation=False):
        self.calls.append((text, add_special_tokens, truncation))
        assert truncation is False
        ids = [1] if add_special_tokens else []
        for part in re.split(r"(<extra_0>|ки|\+|-)", text):
            words = [part] if part.strip() and part in self.tokens else part.split()
            for word in words:
                if word not in self.tokens:
                    token = max(self.inverse) + 1
                    self.tokens[word] = token
                    self.inverse[token] = word
                ids.append(self.tokens[word])
        return ids

    def decode(self, ids, *, skip_special_tokens=False, clean_up_tokenization_spaces=False):
        assert skip_special_tokens is False
        assert clean_up_tokenization_spaces is False
        return " ".join(self.inverse[item] for item in ids)

    def apply_chat_template(self, messages, *, tokenize, add_generation_prompt):
        assert tokenize is False
        assert add_generation_prompt is False
        self.messages = messages
        return "\n".join(f"[{item['role']}] {item['content']}" for item in messages)


class FakeTorch:
    long = "int64"
    float32 = "float32"
    float16 = "float16"
    bfloat16 = "bfloat16"

    @staticmethod
    def tensor(data, *, dtype, device):
        assert dtype == "int64"
        assert device == "cpu"
        return np.asarray(data, dtype=np.int64)

    @staticmethod
    def ones_like(data):
        return np.ones_like(data)

    @staticmethod
    def inference_mode():
        return contextlib.nullcontext()


class FakePRMModel:
    device = "cpu"
    dtype = "float32"

    def __init__(self, kind, *, pairs=None):
        self.kind = kind
        self.config = SimpleNamespace(
            max_position_embeddings=4096, _commit_hash=PUBLISHED_REVISIONS[kind]
        )
        self.pairs = pairs or [[0.0, math.log(3)], [math.log(3), 0.0]]
        self.calls = []
        self.eval_called = False
        self.output_transform = lambda logits: SimpleNamespace(logits=logits)

    def eval(self):
        self.eval_called = True
        return self

    def __call__(self, *, input_ids, attention_mask, use_cache):
        assert self.eval_called
        assert use_cache is False
        assert np.array_equal(attention_mask, np.ones_like(input_ids))
        self.calls.append(input_ids.tolist())
        labels = 2 if self.kind == "qwen_prm" else 128
        logits = np.full((1, input_ids.shape[1], labels), 1000.0)
        marker = FakeTokenizer.marker_ids["<extra_0>" if self.kind == "qwen_prm" else "ки"]
        positions = np.where(input_ids[0] == marker)[0]
        for i, position in enumerate(positions):
            pair = self.pairs[i % len(self.pairs)]
            if self.kind == "qwen_prm":
                logits[0, position] = pair
            else:
                # Negative/positive; enormous irrelevant vocabulary logits must be ignored.
                logits[0, position, 11], logits[0, position, 7] = pair
        return self.output_transform(logits)


def prm_config(kind="qwen_prm", **overrides):
    return PRMConfig(
        **{
            "kind": kind,
            "model_id": MODEL_REPOSITORIES[kind],
            "revision": PUBLISHED_REVISIONS[kind],
            "tokenizer_revision": PUBLISHED_REVISIONS[kind],
            "license_reviewed": True,
            "license_review_reference": "offline injected test only; no checkpoint reviewed",
            **overrides,
        }
    )


def verifier(kind="qwen_prm", *, config=None, tokenizer=None, model=None):
    tokenizer = tokenizer or FakeTokenizer()
    model = model or FakePRMModel(kind)
    adapter = PRMVerifier(
        config or prm_config(kind), tokenizer=tokenizer, model=model, torch_module=FakeTorch()
    )
    return adapter, tokenizer, model


@pytest.mark.parametrize("kind", ["qwen_prm", "math_shepherd"])
@pytest.mark.parametrize(
    ("aggregation", "expected"), [("min", 0.25), ("product", 0.1875), ("mean", 0.5), ("last", 0.25)]
)
def test_actual_adapter_selects_author_logits_and_records_direction(kind, aggregation, expected):
    adapter, tokenizer, model = verifier(kind, config=prm_config(kind, aggregation=aggregation))
    result = adapter.score("Compute 1 + 1.", "The sum is two.\n\n\\boxed{2}")
    assert result["step_scores"] == pytest.approx([0.75, 0.25])
    assert result["aggregate_score"] == pytest.approx(expected)
    assert result["error_score"] == pytest.approx(1 - expected)
    assert result["execution_mode"] == "injected_offline"
    assert result["n_steps"] == 2
    assert result["truncated"] is False
    assert result["revision"] == PUBLISHED_REVISIONS[kind]
    assert result["tokenizer_revision"] == PUBLISHED_REVISIONS[kind]
    assert result["transform"] == "one_minus_aggregate_not_probability_calibration"
    assert len(model.calls) == 1
    assert len(result["scoring_token_positions"]) == 2
    assert len(result["config_hash"]) == len(result["token_ids_hash"]) == 64
    assert all(call[2] is False for call in tokenizer.calls)
    if kind == "qwen_prm":
        assert tokenizer.messages == [
            {"role": "system", "content": QWEN_SYSTEM},
            {"role": "user", "content": "Compute 1 + 1."},
            {"role": "assistant", "content": "The sum is two.<extra_0>\\boxed{2}<extra_0>"},
        ]
        assert result["scoring_token_id"] == 17
        assert result["label_token_ids_negative_positive"] is None
    else:
        assert result["serialized_input"] == (
            "Compute 1 + 1. Step 1: The sum is two. ки\nStep 2: \\boxed{2} ки"
        )
        assert result["scoring_token_id"] == 19
        assert result["label_token_ids_negative_positive"] == (11, 7)


@pytest.mark.parametrize("kind", ["qwen_prm", "math_shepherd"])
def test_underflow_zero_step_retained_and_large_logits_stable(kind):
    model = FakePRMModel(kind, pairs=[[10000.0, -10000.0], [10000.0, 10000.0]])
    adapter, _, _ = verifier(kind, model=model)
    result = adapter.score("p", "step a\n\nstep b")
    assert result["step_scores"] == [0.0, 0.5]
    assert result["error_score"] == 1.0


@pytest.mark.parametrize("kind", ["qwen_prm", "math_shepherd"])
@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -float("inf")])
def test_nonfinite_selected_logit_refused(kind, bad):
    adapter, _, _ = verifier(kind, model=FakePRMModel(kind, pairs=[[bad, 0.0]]))
    with pytest.raises(ModelOutputError, match="finite logits"):
        adapter.score("p", "step")


@pytest.mark.parametrize("kind", ["qwen_prm", "math_shepherd"])
def test_overlong_inputs_rejected_before_forward_without_truncation(kind):
    adapter, _, model = verifier(kind, config=prm_config(kind, max_tokens=5))
    with pytest.raises(ModelInputError, match="truncation is forbidden"):
        adapter.score("A question with more than five words.", "step")
    assert model.calls == []


def test_model_maximum_may_reduce_config_limit():
    model = FakePRMModel("qwen_prm")
    model.config.max_position_embeddings = 3
    adapter, _, _ = verifier(model=model)
    with pytest.raises(ModelInputError, match="limit is 3"):
        adapter.score("p", "step")


@pytest.mark.parametrize("kind,marker", [("qwen_prm", "<extra_0>"), ("math_shepherd", "ки")])
@pytest.mark.parametrize("where", ["problem", "solution"])
def test_user_supplied_marker_rejected_before_model_load(kind, marker, where):
    adapter, _, model = verifier(kind)
    with pytest.raises(ModelInputError, match="literal scoring marker"):
        adapter.score(marker if where == "problem" else "p", marker if where == "solution" else "s")
    assert not model.eval_called


def test_missing_serialized_marker_is_not_silently_unscored():
    class DroppingTokenizer(FakeTokenizer):
        def apply_chat_template(self, *args, **kwargs):
            return super().apply_chat_template(*args, **kwargs).replace("<extra_0>", "")

    adapter, _, model = verifier(tokenizer=DroppingTokenizer())
    with pytest.raises(ModelInputError, match="marker count"):
        adapter.score("p", "step")
    assert model.calls == []


@pytest.mark.parametrize("kind", ["qwen_prm", "math_shepherd"])
def test_multitoken_marker_refused(kind):
    class BrokenMarker(FakeTokenizer):
        def encode(self, text, **kwargs):
            ids = super().encode(text, **kwargs)
            if text in self.marker_ids:
                return [self.tokens["+"], *ids]
            return ids

    adapter, _, _ = verifier(kind, tokenizer=BrokenMarker())
    with pytest.raises(MathModelError, match="multiple content tokens"):
        adapter.load()


def test_shepherd_whitespace_only_marker_prefix_is_checked_and_allowed():
    class SpacePrefix(FakeTokenizer):
        def encode(self, text, **kwargs):
            ids = super().encode(text, **kwargs)
            return [2, *ids] if text == "ки" else ids

    adapter, _, _ = verifier("math_shepherd", tokenizer=SpacePrefix())
    assert adapter.score("p", "s")["scoring_token_id"] == 19


def test_shepherd_changed_label_tokenization_rejected():
    class BadLabels(FakeTokenizer):
        def encode(self, text, **kwargs):
            ids = super().encode(text, **kwargs)
            return ids[:1] if text == "+ -" else ids

    adapter, _, _ = verifier("math_shepherd", tokenizer=BadLabels())
    with pytest.raises(MathModelError, match="two distinct labels"):
        adapter.load()


@pytest.mark.parametrize("mode", ["wrong_rank", "wrong_length", "wrong_head", "missing"])
def test_qwen_invalid_output_shapes_refused(mode):
    model = FakePRMModel("qwen_prm")
    changes = {
        "wrong_rank": lambda x: SimpleNamespace(logits=x[0]),
        "wrong_length": lambda x: SimpleNamespace(logits=x[:, :-1]),
        "wrong_head": lambda x: SimpleNamespace(logits=x[:, :, :1]),
        "missing": lambda x: (),
    }
    model.output_transform = changes[mode]
    adapter, _, _ = verifier(model=model)
    with pytest.raises(ModelOutputError):
        adapter.score("p", "step")


def test_tuple_output_contract_is_supported():
    model = FakePRMModel("qwen_prm")
    model.output_transform = lambda logits: (logits,)
    adapter, _, _ = verifier(model=model)
    assert adapter.score("p", "s")["step_scores"] == pytest.approx([0.75])


def test_shepherd_too_small_vocabulary_refused():
    model = FakePRMModel("math_shepherd")
    model.output_transform = lambda logits: SimpleNamespace(logits=logits[:, :, :8])
    adapter, _, _ = verifier("math_shepherd", model=model)
    with pytest.raises(ModelOutputError, match="label IDs"):
        adapter.score("p", "s")


def test_model_revision_mismatch_refused():
    model = FakePRMModel("qwen_prm")
    model.config._commit_hash = "0" * 40
    adapter, _, _ = verifier(model=model)
    with pytest.raises(MathModelError, match="different checkpoint revision"):
        adapter.load()


def test_lazy_loading_and_license_gate_precede_dependency_import(monkeypatch):
    imports = []
    monkeypatch.setattr(models.importlib, "import_module", lambda name: imports.append(name))
    adapter = PRMVerifier(prm_config(license_reviewed=False))
    assert imports == []
    with pytest.raises(ModelAccessNotReviewedError, match="license/use review"):
        adapter.load()
    assert imports == []
    adapter = PRMVerifier(prm_config(trust_remote_code=False))
    with pytest.raises(ModelAccessNotReviewedError, match="custom code"):
        adapter.load()
    assert imports == []


@pytest.mark.parametrize("kind", ["qwen_prm", "math_shepherd"])
def test_real_loader_boundary_pins_revisions_and_local_only(monkeypatch, kind):
    calls = []
    fake_tokenizer, fake_model = FakeTokenizer(), FakePRMModel(kind)

    def factory(label, output):
        def load(source, **kwargs):
            calls.append((label, source, kwargs))
            return output

        return SimpleNamespace(from_pretrained=load)

    def wrong_factory(*args, **kwargs):
        pytest.fail("the wrong Hugging Face model class was selected")

    repository_config = SimpleNamespace()  # no pad_token_id, as under transformers 5
    transformers = SimpleNamespace(
        AutoTokenizer=factory("tokenizer", fake_tokenizer),
        AutoConfig=factory("config", repository_config),
        AutoModel=factory("qwen_model", fake_model) if kind == "qwen_prm" else None,
        AutoModelForCausalLM=factory("shepherd_model", fake_model)
        if kind == "math_shepherd"
        else SimpleNamespace(from_pretrained=wrong_factory),
    )
    dependencies = {"torch": FakeTorch(), "transformers": transformers}
    monkeypatch.setattr(models.importlib, "import_module", dependencies.__getitem__)
    config = prm_config(kind, trust_remote_code=(kind == "qwen_prm"))
    adapter = PRMVerifier(config)
    assert calls == []
    adapter.load()
    adapter.load()
    # repository code (Qwen) reads its config first and gets the transformers 4.x defaults
    assert [label for label, *_ in calls][:-1] == (
        ["tokenizer", "config"] if kind == "qwen_prm" else ["tokenizer"]
    )
    for _, source, kwargs in calls:
        assert source == config.model_id
        assert kwargs["revision"] == config.revision
        assert kwargs["code_revision"] == config.revision
        assert kwargs["local_files_only"] is True
    assert calls[-1][0] == ("qwen_model" if kind == "qwen_prm" else "shepherd_model")
    assert calls[-1][2]["torch_dtype"] == "bfloat16"
    assert calls[-1][2]["device_map"] == "auto"
    if kind == "qwen_prm":
        assert calls[-1][2]["config"] is repository_config
        assert repository_config.pad_token_id is None
    else:
        assert "config" not in calls[-1][2] and not vars(repository_config)


def test_source_manifest_requires_coverage_and_content_hashes(tmp_path):
    config = prm_config(local_path=str(tmp_path))
    adapter = PRMVerifier(config)
    with pytest.raises(MathModelError, match="requires rafr_model_manifest"):
        adapter._source()
    payload = b'{"max_position_embeddings":4096}'
    (tmp_path / "config.json").write_bytes(payload)
    manifest = {
        "model_id": config.model_id,
        "revision": config.revision,
        "tokenizer_revision": config.tokenizer_revision,
        "files": {"config.json": hashlib.sha256(payload).hexdigest()},
    }
    manifest_path = tmp_path / "rafr_model_manifest.json"
    manifest_path.write_text(json.dumps(manifest))
    assert adapter._source() == str(tmp_path)
    # the Hub's bookkeeping of a local_dir download is never loaded: outside the manifest
    trees = tmp_path / ".cache" / "huggingface" / "trees"
    trees.mkdir(parents=True)
    (trees / f"{config.revision}.json").write_text("[]")
    assert adapter._source() == str(tmp_path)
    # ... but any other cache-looking directory is still checked
    (tmp_path / ".cache" / "other.json").write_text("{}")
    with pytest.raises(MathModelError, match="cover all runtime"):
        adapter._source()
    (tmp_path / ".cache" / "other.json").unlink()
    (tmp_path / "extra.py").write_text("unreviewed")
    with pytest.raises(MathModelError, match="cover all runtime"):
        adapter._source()
    (tmp_path / "extra.py").unlink()
    (tmp_path / "config.json").write_text("changed")
    with pytest.raises(MathModelError, match="hash differs"):
        adapter._source()


def test_source_manifest_rejects_paths_outside_model_directory(tmp_path):
    root = tmp_path / "model"
    root.mkdir()
    other = tmp_path / "outside.py"
    other.write_text("outside")
    config = prm_config(local_path=str(root))
    manifest = {
        "model_id": config.model_id,
        "revision": config.revision,
        "tokenizer_revision": config.tokenizer_revision,
        "files": {"../outside.py": hashlib.sha256(other.read_bytes()).hexdigest()},
    }
    (root / "rafr_model_manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(MathModelError, match="escaping"):
        PRMVerifier(config)._source()


def test_partial_injection_and_reuse_after_close_are_refused():
    with pytest.raises(MathModelError, match="together"):
        PRMVerifier(prm_config(), tokenizer=FakeTokenizer())
    adapter, _, _ = verifier()
    adapter.load()
    adapter.close()
    with pytest.raises(MathModelError, match="closed"):
        adapter.score("p", "s")


@pytest.mark.parametrize(
    "change",
    [
        {"revision": "main"},
        {"tokenizer_revision": "latest"},
        {"model_id": "other"},
        {"max_tokens": 4097},
        {"max_tokens": True},
        {"aggregation": "vote"},
        {"dtype": "int8"},
        {"local_files_only": 1},
        {"license_review_reference": None},
    ],
)
def test_invalid_prm_config_refused(change):
    with pytest.raises(MathModelError):
        prm_config(**change)


def test_prm_config_defaults_and_configuration_are_auditable():
    config = prm_config()
    assert config.local_files_only
    assert config.configuration() == {
        **asdict(config),
        "adapter_version": ADAPTER_VERSION,
        "step_rule": models.STEP_RULE,
    }
    assert not PRMConfig(
        kind="qwen_prm",
        model_id=MODEL_REPOSITORIES["qwen_prm"],
        revision=PUBLISHED_REVISIONS["qwen_prm"],
        tokenizer_revision=PUBLISHED_REVISIONS["qwen_prm"],
    ).license_reviewed


def test_checkpoint_directory_location_is_not_logical_identity():
    a = prm_config(local_path="/machine-a/models/checkpoint")
    b = prm_config(local_path="/machine-b/cache/checkpoint")
    assert a.logical_configuration() == b.logical_configuration()
    first, _, _ = verifier(config=a)
    second, _, _ = verifier(config=b)
    first_result, second_result = first.score("p", "s"), second.score("p", "s")
    assert first_result["config_hash"] == second_result["config_hash"]
    assert first_result["local_path"] != second_result["local_path"]


def test_step_rule_preserves_content_and_makes_line_endings_explicit():
    assert split_solution_steps(" a\r\nline \r\n \r\n\r\nb  ") == ["a\nline", "b"]
    assert split_solution_steps("a\nb") == ["a\nb"]


@pytest.mark.parametrize("values", [[], [True], [-0.01], [1.01], [float("nan")]])
def test_bad_step_aggregation_is_not_imputed(values):
    with pytest.raises(ModelOutputError):
        aggregate_steps(values, "min")


def spec(form="prm", **changes):
    return models.VerifierSpec(
        **{
            "form": form,
            "vendor": "test-source-a",
            "base_model": "test-backbone-a",
            "model_id": "test/model",
            "revision": "frozen-v1",
            "implementation": "reviewed-test",
            **changes,
        }
    )


def test_pair_requires_disjoint_vendor_and_backbone_not_forms():
    """paper Sec. 5: vendors and bases must differ; the same form is allowed."""
    first = spec()
    second = spec("orm", vendor="test-source-b", base_model="test-backbone-b")
    assert models.validate_verifier_pair([first, second]) == (first, second)
    same_form = replace(second, form="prm")
    assert models.validate_verifier_pair([first, same_form]) == (first, same_form)
    for change in ({"vendor": first.vendor}, {"base_model": first.base_model}):
        with pytest.raises(MathModelError, match="not eligible"):
            models.validate_verifier_pair([first, replace(second, **change)])
    with pytest.raises(MathModelError, match="exactly two"):
        models.validate_verifier_pair([first])


LINEAGE = {"vendors": ["Lab A", "Base Co"], "bases": ["Base 1"], "reasoning_data": []}


def test_pair_compares_lineage_sets():
    """A model trained from several sources is out against any model sharing one of them."""
    first = spec(lineage=LINEAGE)
    other = {"vendors": ["Lab B"], "bases": ["Base 2"], "reasoning_data": ["Teacher X"]}
    assert models.pair_ineligibility(first, spec(lineage=other)) == []
    shared = {**other, "vendors": ["Lab B", "base co"]}  # case-insensitive
    assert models.pair_ineligibility(first, spec(lineage=shared)) == ["shared vendor: base co"]
    # a spec without lineage is compared through its single vendor and base names
    plain = spec(vendor="Base Co", base_model="other-base")
    assert models.pair_ineligibility(first, plain) == ["shared vendor: base co"]


@pytest.mark.parametrize(
    "lineage",
    [
        {**LINEAGE, "vendors": ["OpenAI"]},
        {**LINEAGE, "reasoning_data": ["o3-mini traces"]},
        {**LINEAGE, "reasoning_data": ["Claude 3.5 Sonnet"]},
        {**LINEAGE, "bases": []},
        {"vendors": ["Lab A"], "bases": ["Base 1"]},
    ],
)
def test_lineage_refuses_excluded_sources_and_bad_shapes(lineage):
    with pytest.raises(MathModelError, match="lineage"):
        spec(lineage=lineage)


def test_anthropic_identity_is_rejected():
    with pytest.raises(MathModelError, match="Anthropic"):
        spec(vendor="Anthropic")


@pytest.mark.parametrize(
    "change", [{"vendor": "OpenAI"}, {"base_model": "gpt-5.6-terra"}, {"model_id": "openai/model"}]
)
def test_openai_model_sources_are_rejected(change):
    with pytest.raises(MathModelError, match="OpenAI"):
        spec(**change)


def test_orm_and_judge_take_no_scalar_backend_and_need_the_installed_adapter():
    # The adapters themselves are exercised in test_math_hf_verifiers.py.
    for kind in ("orm", "llm_judge"):
        with pytest.raises(NotImplementedError, match="not been installed"):
            models.build_verifier(spec(kind))
        with pytest.raises(TypeError):
            models.build_verifier(spec(kind), backend=lambda p, s: 0.2)


def test_prm_factory_is_lazy_and_does_not_relabel_forms():
    config = prm_config()
    selected = spec(
        "prm",
        model_id=config.model_id,
        revision=config.revision,
        implementation=config.kind,
        settings=asdict(config),
    )
    built = models.build_verifier(selected)
    assert isinstance(built, PRMVerifier)
    assert not built._prepared
    with pytest.raises(MathModelError, match="relabelled"):
        replace(selected, form="orm")
    with pytest.raises(MathModelError, match="conflict"):
        models.build_verifier(replace(selected, settings={**selected.settings, "revision": "bad"}))
