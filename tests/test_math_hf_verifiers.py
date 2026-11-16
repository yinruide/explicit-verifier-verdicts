"""LLM-judge and ORM backends (M.4) at injected tokenizer / model / torch boundaries.

No checkpoint is downloaded. The fakes stand in for any Hugging Face model: nothing in
the adapters may depend on which one. ``test_math_hf_stub_weights.py`` runs the real
transformers path on tiny random weights.
"""

from __future__ import annotations

import contextlib
import inspect
import math
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from math_v2_helpers import make_harness
from rafr.contracts.schemas import Split
from rafr.domains.math import models
from rafr.domains.math.models import (
    JUDGE_IMPLEMENTATION,
    ORM_IMPLEMENTATION,
    LLMJudgeConfig,
    LLMJudgeVerifier,
    MathModelError,
    ModelAccessNotReviewedError,
    ModelInputError,
    ModelOutputError,
    ORMConfig,
    ORMVerifier,
)
from rafr.logging.writer import read_jsonl

REVISION = "1" * 40
WRONG = "WRONG"
PROMPT_SHA = models.read_judge_prompt()[1]
YES_ID, NO_ID, PAD_ID, TURN_ID = 5, 6, 0, 2


class FakeTokenizer:
    """Whitespace words; every new word gets the next ID. Records everything it sees."""

    chat_template = "fake-chat-template-v1"
    all_special_tokens = ["<pad>", "<bos>", "<|turn|>"]
    pad_token_id = PAD_ID
    eos_token_id = TURN_ID

    def __init__(self):
        self.tokens = {"<pad>": PAD_ID, "<bos>": 1, "<|turn|>": TURN_ID, "Yes": YES_ID, "No": NO_ID}
        self.inverse = {value: key for key, value in self.tokens.items()}
        self.encoded = []
        self.chats = []

    def _id(self, word):
        if word not in self.tokens:
            token = max(self.inverse) + 1
            self.tokens[word] = token
            self.inverse[token] = word
        return self.tokens[word]

    def encode(self, text, *, add_special_tokens, truncation):
        assert truncation is False
        self.encoded.append(text)
        return ([1] if add_special_tokens else []) + [self._id(word) for word in text.split()]

    def decode(self, ids, *, skip_special_tokens, clean_up_tokenization_spaces):
        assert skip_special_tokens is False and clean_up_tokenization_spaces is False
        return " ".join(self.inverse[item] for item in ids)

    def apply_chat_template(self, messages, *, tokenize, add_generation_prompt, **kwargs):
        assert tokenize is False
        self.chats.append((messages, add_generation_prompt, kwargs))
        body = " ".join(f"<|turn|> {item['role']} {item['content']}" for item in messages)
        return body + (" <|turn|> assistant" if add_generation_prompt else " <|turn|>")


class FakeTorch:
    long = "int64"
    float32, float16, bfloat16 = "float32", "float16", "bfloat16"

    @staticmethod
    def tensor(data, *, dtype, device):
        assert dtype == "int64" and device == "cpu"
        return np.asarray(data, dtype=np.int64)

    @staticmethod
    def inference_mode():
        return contextlib.nullcontext()


def check_left_padding(input_ids, attention_mask, position_ids, pad_id):
    for ids, mask, positions in zip(input_ids, attention_mask, position_ids, strict=True):
        n = int(mask.sum())
        pad = len(ids) - n
        assert n > 0 and not mask[:pad].any() and mask[pad:].all()
        assert (ids[:pad] == pad_id).all()
        assert list(positions[pad:]) == list(range(n)) and not positions[:pad].any()


def real_tokens(input_ids, attention_mask):
    return [ids[mask == 1] for ids, mask in zip(input_ids, attention_mask, strict=True)]


class FakeJudgeModel:
    device = "cpu"
    dtype = "bfloat16"

    def __init__(self, tokenizer, *, keep_last=True, vocab=16):
        self.config = SimpleNamespace(max_position_embeddings=256, _commit_hash=REVISION)
        self.tokenizer, self.keep_last, self.vocab = tokenizer, keep_last, vocab
        self.calls, self.eval_called = [], False
        self.output_transform = lambda logits: SimpleNamespace(logits=logits)

    def eval(self):
        self.eval_called = True
        return self

    @staticmethod
    def no_logit(row, tokenizer):
        wrong = tokenizer.tokens.get(WRONG)
        return (2.0 if wrong in row else -1.0) + 0.01 * len(row)

    def __call__(self, *, input_ids, attention_mask, position_ids, use_cache, **kwargs):
        assert self.eval_called and use_cache is False
        if "logits_to_keep" in kwargs and not self.keep_last:
            raise TypeError("forward() got an unexpected keyword argument 'logits_to_keep'")
        self.calls.append((input_ids.shape, dict(kwargs), input_ids.copy()))
        check_left_padding(input_ids, attention_mask, position_ids, PAD_ID)
        batch, length = input_ids.shape
        # Enormous irrelevant vocabulary logits must not leak into the two-token softmax.
        logits = np.full((batch, length, max(self.vocab, NO_ID + 1)), 1000.0)
        for i, row in enumerate(real_tokens(input_ids, attention_mask)):
            logits[i, -1, YES_ID], logits[i, -1, NO_ID] = 0.0, self.no_logit(row, self.tokenizer)
        logits = logits[:, :, : self.vocab]
        if kwargs.get("logits_to_keep") == 1:
            logits = logits[:, -1:, :]
        return self.output_transform(logits)


class FakeORMModel:
    device = "cpu"
    dtype = "bfloat16"

    def __init__(self, tokenizer):
        self.config = SimpleNamespace(
            max_position_embeddings=256, pad_token_id=PAD_ID, _commit_hash=REVISION
        )
        self.tokenizer = tokenizer
        self.calls, self.eval_called = [], False
        self.output_transform = lambda logits: SimpleNamespace(logits=logits)

    def eval(self):
        self.eval_called = True
        return self

    @staticmethod
    def logit(row, tokenizer):
        wrong = tokenizer.tokens.get(WRONG)
        return (-1.5 if wrong in row else 3.0) - 0.01 * len(row)

    def __call__(self, *, input_ids, attention_mask, position_ids, use_cache):
        assert self.eval_called and use_cache is False
        self.calls.append(input_ids.shape)
        check_left_padding(input_ids, attention_mask, position_ids, PAD_ID)
        rows = real_tokens(input_ids, attention_mask)
        logits = np.asarray([[self.logit(row, self.tokenizer)] for row in rows])
        return self.output_transform(logits)


def judge_config(**overrides):
    return LLMJudgeConfig(
        **{
            "model_id": "test-org/any-instruct-model",
            "revision": REVISION,
            "tokenizer_revision": REVISION,
            "input_format": "chat_template",
            "add_special_tokens": False,
            "yes_token": "Yes",
            "no_token": "No",
            "dtype": "bfloat16",
            "batch_size": 2,
            "max_input_tokens": 128,
            "prompt_sha256": PROMPT_SHA,
            "license_reviewed": True,
            "license_review_reference": "offline injected test only; no checkpoint reviewed",
            **overrides,
        }
    )


def orm_config(**overrides):
    return ORMConfig(
        **{
            "model_id": "test-org/any-reward-model",
            "revision": REVISION,
            "tokenizer_revision": REVISION,
            "input_format": "chat_template",
            "add_special_tokens": False,
            "dtype": "bfloat16",
            "batch_size": 2,
            "max_input_tokens": 64,
            "license_reviewed": True,
            "license_review_reference": "offline injected test only; no checkpoint reviewed",
            **overrides,
        }
    )


def judge(config=None, *, tokenizer=None, model=None, prompt_dir=None):
    tokenizer = tokenizer or FakeTokenizer()
    model = model or FakeJudgeModel(tokenizer)
    adapter = LLMJudgeVerifier(
        config or judge_config(),
        prompt_dir=prompt_dir,
        tokenizer=tokenizer,
        model=model,
        torch_module=FakeTorch(),
    )
    return adapter, tokenizer, model


def orm(config=None, *, tokenizer=None, model=None):
    tokenizer = tokenizer or FakeTokenizer()
    model = model or FakeORMModel(tokenizer)
    adapter = ORMVerifier(
        config or orm_config(), tokenizer=tokenizer, model=model, torch_module=FakeTorch()
    )
    return adapter, tokenizer, model


def sigmoid(x):
    return 1 / (1 + math.exp(-x))


ITEMS = [
    ("Compute 1 + 1.", "One plus one is two. \\boxed{2}"),
    ("Compute 2 + 2.", "WRONG reasoning gives five. \\boxed{5}"),
    ("Compute 3 + 3.", "Three plus three is six. \\boxed{6}"),
]


# ---- LLM judge ------------------------------------------------------------------------


def test_judge_score_is_p_no_over_the_two_verdict_logits():
    adapter, tokenizer, model = judge()
    results = adapter.score_batch(ITEMS)
    assert [round(r["error_score"], 12) for r in results] == [round(r["p_no"], 12) for r in results]
    for result in results:
        row = [tokenizer.tokens[word] for word in result["serialized_input"].split()]
        assert result["input_tokens"] == len(row)
        expected = sigmoid(FakeJudgeModel.no_logit(row, tokenizer))
        assert result["error_score"] == pytest.approx(expected, abs=1e-15)
        assert result["verdict_logits_yes_no"][0] == 0.0
        assert result["verdict_token_ids_yes_no"] == [YES_ID, NO_ID]
        assert result["score_direction"] == "higher_error_score_means_more_problematic"
        assert result["truncated"] is False and result["execution_mode"] == "injected_offline"
        assert result["prompt_sha256"] == PROMPT_SHA
    # the wrong solution scores as more likely wrong
    assert results[1]["error_score"] > max(results[0]["error_score"], results[2]["error_score"])
    # one user turn: the prompt with problem and solution, then the generation prompt
    messages, generation_prompt, kwargs = tokenizer.chats[0]
    assert generation_prompt is True and kwargs == {}
    assert [m["role"] for m in messages] == ["user"]
    prompt = models.read_judge_prompt()[0]
    assert messages[0]["content"] == prompt.replace("{problem}", ITEMS[0][0]).replace(
        "{solution}", ITEMS[0][1]
    )
    assert "Answer only Yes or No." in messages[0]["content"]


def test_judge_batches_have_a_fixed_shape_and_the_last_batch_is_filled():
    adapter, _, model = judge()
    together = adapter.score_batch(ITEMS)
    assert [call[0] for call in model.calls] == [(2, 128), (2, 128)]
    # the short last batch carries a copy of its last row; its score is dropped
    last = model.calls[1][2]
    assert (last[0] == last[1]).all()
    assert len(together) == 3
    for item, result in zip(ITEMS, together, strict=True):
        alone = adapter.score(*item)
        assert alone["error_score"] == result["error_score"]
        assert alone["token_ids_hash"] == result["token_ids_hash"]
    assert all(call[1] == {"logits_to_keep": 1} for call in model.calls)


def test_judge_falls_back_when_the_model_has_no_logits_to_keep():
    tokenizer = FakeTokenizer()
    adapter, _, model = judge(tokenizer=tokenizer, model=FakeJudgeModel(tokenizer, keep_last=False))
    first = adapter.score(*ITEMS[1])
    second = adapter.score(*ITEMS[1])
    assert first["error_score"] == second["error_score"]
    assert [call[1] for call in model.calls] == [{}, {}]


def test_judge_answer_prefix_is_teacher_forced_as_token_ids():
    adapter, tokenizer, model = judge(judge_config(answer_prefix="Verdict :"))
    result = adapter.score(*ITEMS[0])
    prefix = [tokenizer.tokens["Verdict"], tokenizer.tokens[":"]]
    assert result["answer_prefix_token_ids"] == prefix
    sent = model.calls[0][2][0]
    assert list(sent[-2:]) == prefix
    assert result["input_tokens"] == len(result["serialized_input"].split()) + 2


def test_judge_raw_format_uses_the_configured_turn_markers():
    template = "<start> user {prompt} <end> <start> model"
    adapter, tokenizer, _ = judge(judge_config(input_format="raw", raw_template=template))
    result = adapter.score(*ITEMS[0])
    assert tokenizer.chats == []
    assert result["serialized_input"].startswith("<start> user Here is a math problem")
    assert result["serialized_input"].endswith("Answer only Yes or No. <end> <start> model")


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"yes_token": "Yes please"}, "2 tokens"),
        ({"yes_token_id": NO_ID + 1, "no_token_id": NO_ID}, "decodes to"),
    ],
)
def test_judge_verdict_tokens_are_checked_against_the_tokenizer(change, message):
    tokenizer = FakeTokenizer()
    tokenizer._id("filler")
    adapter, _, _ = judge(judge_config(**change), tokenizer=tokenizer)
    with pytest.raises(MathModelError, match=message):
        adapter.load()


def test_judge_configured_token_ids_are_used_when_they_decode_to_the_verdicts():
    adapter, _, _ = judge(judge_config(yes_token_id=YES_ID, no_token_id=NO_ID))
    assert adapter.score(*ITEMS[0])["verdict_token_ids_yes_no"] == [YES_ID, NO_ID]


def test_judge_logits_shape_and_vocabulary_are_checked():
    tokenizer = FakeTokenizer()
    adapter, _, _ = judge(tokenizer=tokenizer, model=FakeJudgeModel(tokenizer, vocab=NO_ID))
    with pytest.raises(ModelOutputError, match="verdict token ids"):
        adapter.score(*ITEMS[0])
    model = FakeJudgeModel(tokenizer)
    model.output_transform = lambda logits: SimpleNamespace(logits=logits[0])
    adapter, _, _ = judge(tokenizer=tokenizer, model=model)
    with pytest.raises(ModelOutputError, match="shape"):
        adapter.score(*ITEMS[0])


def test_judge_prompt_is_pinned_by_hash_and_guarded(tmp_path):
    text = models.read_judge_prompt()[0]
    with pytest.raises(MathModelError, match="prompt_sha256"):
        judge(judge_config(prompt_sha256="0" * 64))
    (tmp_path / "llm_judge.txt").write_bytes((text + "\n").replace("\n", "\r\n").encode())
    # CRLF checkouts hash like the LF pod copy
    assert models.read_judge_prompt(directory=tmp_path)[1] == PROMPT_SHA
    judge(prompt_dir=tmp_path)
    for bad, message in (
        (text + "\nReference answer: {reference}", "slots"),
        (text.replace("{solution}", "the solution"), "slots"),
        ("PLACEHOLDER {problem} {solution}", "PLACEHOLDER"),
    ):
        (tmp_path / "llm_judge.txt").write_text(bad, encoding="utf-8")
        with pytest.raises(MathModelError, match=message):
            judge(prompt_dir=tmp_path)


def test_judge_card_records_prompt_hash_tokens_and_batching():
    adapter, _, _ = judge(judge_config(answer_prefix="Verdict :"))
    card = adapter.verifier_card()
    assert card["form"] == "llm_judge" and card["implementation"] == JUDGE_IMPLEMENTATION
    assert card["input"]["prompt_sha256"] == PROMPT_SHA
    assert card["input"]["prompt_file"] == "prompts/llm_judge.txt"
    assert card["input"]["prompt_text"] == models.read_judge_prompt()[0]
    assert card["score"]["verdict_tokens"] == {"yes": [YES_ID, "Yes"], "no": [NO_ID, "No"]}
    assert card["batching"]["batch_size"] == 2 and card["batching"]["fixed_shape"] is True
    assert card["batching"]["max_input_tokens"] == 128
    assert card["reference_answer_input"] is False
    assert card["configuration"]["prompt_sha256"] == PROMPT_SHA
    assert card["config_hash"] == adapter.score(*ITEMS[0])["config_hash"]


# ---- ORM ------------------------------------------------------------------------------


def test_orm_score_is_one_minus_sigmoid_of_the_scalar_logit():
    adapter, tokenizer, model = orm(orm_config(system_prompt="Judge the solution."))
    results = adapter.score_batch(ITEMS)
    assert model.calls == [(2, 64), (2, 64)]
    for result in results:
        row = [tokenizer.tokens[word] for word in result["serialized_input"].split()]
        logit = FakeORMModel.logit(row, tokenizer)
        assert result["logit"] == logit
        assert result["error_score"] == pytest.approx(1 - sigmoid(logit), abs=1e-15)
        assert result["error_score"] + result["sigmoid_logit"] == pytest.approx(1, abs=1e-15)
        assert result["score_definition"] == "one_minus_sigmoid_of_scalar_logit"
    assert results[1]["error_score"] > max(results[0]["error_score"], results[2]["error_score"])
    messages, generation_prompt, _ = tokenizer.chats[0]
    assert generation_prompt is False
    assert messages == [
        {"role": "system", "content": "Judge the solution."},
        {"role": "user", "content": ITEMS[0][0]},
        {"role": "assistant", "content": ITEMS[0][1]},
    ]
    for item, result in zip(ITEMS, results, strict=True):
        assert adapter.score(*item)["error_score"] == result["error_score"]


def test_orm_raw_template_spells_out_every_separator():
    template = "<sys> grade <usr> {problem} <asst> {solution} <eos>"
    adapter, tokenizer, _ = orm(orm_config(input_format="raw", raw_template=template))
    result = adapter.score(*ITEMS[0])
    assert tokenizer.chats == []
    assert result["serialized_input"] == template.replace("{problem}", ITEMS[0][0]).replace(
        "{solution}", ITEMS[0][1]
    )


def test_orm_needs_a_scalar_head_and_its_own_pad_token():
    tokenizer = FakeTokenizer()
    model = FakeORMModel(tokenizer)
    model.output_transform = lambda logits: SimpleNamespace(logits=np.hstack([logits, logits]))
    adapter, _, _ = orm(tokenizer=tokenizer, model=model)
    with pytest.raises(ModelOutputError, match="one scalar logit"):
        adapter.score(*ITEMS[0])
    model = FakeORMModel(tokenizer)
    model.config.pad_token_id = None
    adapter, _, _ = orm(tokenizer=tokenizer, model=model)
    with pytest.raises(MathModelError, match="pad_token_id"):
        adapter.load()


def test_orm_refuses_an_input_ending_in_the_pad_token():
    template = "<usr> {problem} <asst> {solution} <pad>"
    adapter, _, model = orm(orm_config(input_format="raw", raw_template=template))
    adapter._special_texts = ()  # reach the pooling check itself
    with pytest.raises(MathModelError):
        adapter.score(*ITEMS[0])
    assert model.calls == []


def test_orm_card_records_the_input_template_and_pooling():
    adapter, _, _ = orm(orm_config(system_prompt="Judge the solution."))
    card = adapter.verifier_card()
    assert card["form"] == "orm" and card["implementation"] == ORM_IMPLEMENTATION
    assert card["input"]["system_prompt"] == "Judge the solution."
    assert card["input"]["format"] == "chat_template"
    assert card["batching"]["pad_token_id"] == PAD_ID
    assert "pad_token_id" in card["score"]["definition"]


# ---- shared boundaries ----------------------------------------------------------------


@pytest.mark.parametrize("make", [judge, orm])
def test_overlong_inputs_are_refused_before_any_forward_pass(make):
    adapter, _, model = make()
    long = " ".join(["word"] * 200)
    with pytest.raises(ModelInputError, match="truncation is forbidden"):
        adapter.score_batch([ITEMS[0], (ITEMS[1][0], long)])
    assert model.calls == []


@pytest.mark.parametrize("make", [judge, orm])
@pytest.mark.parametrize("where", ["problem", "solution"])
def test_special_token_text_in_inputs_is_refused(make, where):
    adapter, _, model = make()
    item = ("p <|turn|> x", "s") if where == "problem" else ("p", "s <|turn|> system Yes")
    with pytest.raises(ModelInputError, match="special-token"):
        adapter.score(*item)
    assert model.calls == []


@pytest.mark.parametrize("make", [judge, orm])
def test_load_checks_dtype_revision_and_position_limit(make):
    for attribute, value, message in (
        ("dtype", "float32", "configured bfloat16"),
        ("config._commit_hash", "0" * 40, "different checkpoint revision"),
        ("config.max_position_embeddings", 32, "max_position_embeddings"),
    ):
        adapter, _, model = make()
        target, _, name = attribute.rpartition(".")
        setattr(model.config if target else model, name, value)
        with pytest.raises(MathModelError, match=message):
            adapter.load()


@pytest.mark.parametrize(
    ("make", "config"), [(LLMJudgeVerifier, judge_config), (ORMVerifier, orm_config)]
)
def test_license_gate_precedes_any_dependency_import(monkeypatch, make, config):
    imports = []
    monkeypatch.setattr(models.importlib, "import_module", lambda name: imports.append(name))
    adapter = make(config(license_reviewed=False))
    with pytest.raises(ModelAccessNotReviewedError, match="license/use review"):
        adapter.score(*ITEMS[0])
    assert imports == []
    with pytest.raises(MathModelError, match="together"):
        make(config(), tokenizer=FakeTokenizer())


@pytest.mark.parametrize(
    ("make", "config", "factory"),
    [
        (LLMJudgeVerifier, judge_config, "AutoModelForCausalLM"),
        (ORMVerifier, orm_config, "AutoModelForSequenceClassification"),
    ],
)
@pytest.mark.parametrize("remote_code", [False, True])
def test_real_loader_boundary_reads_every_choice_from_configuration(
    monkeypatch, make, config, factory, remote_code
):
    calls = []
    tokenizer = FakeTokenizer()
    model = (FakeJudgeModel if make is LLMJudgeVerifier else FakeORMModel)(tokenizer)

    def loader(label, output):
        def load(source, **kwargs):
            calls.append((label, source, kwargs))
            return output

        return SimpleNamespace(from_pretrained=load)

    transformers = SimpleNamespace(
        AutoTokenizer=loader("tokenizer", tokenizer),
        AutoModelForCausalLM=loader("AutoModelForCausalLM", model),
        AutoModelForSequenceClassification=loader("AutoModelForSequenceClassification", model),
    )
    determinism = []
    dependencies = {
        "torch": FakeTorch(),
        "transformers": transformers,
        "rafr.verifiers.candidates.torch_backend": SimpleNamespace(
            configure_determinism=lambda: determinism.append(True)
        ),
    }
    monkeypatch.setattr(models.importlib, "import_module", dependencies.__getitem__)
    chosen = config(
        dtype="float16",
        device_map="cuda:1",
        attn_implementation="sdpa",
        trust_remote_code=remote_code,
    )
    adapter = make(chosen)
    assert calls == []
    model.dtype = "float16"
    adapter.load()
    adapter.load()
    assert determinism == [True]
    assert [label for label, _, _ in calls] == ["tokenizer", factory]
    (_, tok_source, tok_kwargs), (_, model_source, model_kwargs) = calls
    assert tok_source == model_source == chosen.model_id
    assert tok_kwargs["revision"] == chosen.tokenizer_revision
    assert model_kwargs["revision"] == chosen.revision
    assert model_kwargs["dtype"] == "float16"
    assert model_kwargs["device_map"] == "cuda:1"
    assert model_kwargs["attn_implementation"] == "sdpa"
    for kwargs in (tok_kwargs, model_kwargs):
        assert kwargs["local_files_only"] is True
        assert kwargs["trust_remote_code"] is remote_code
        assert ("code_revision" in kwargs) is remote_code


@pytest.mark.parametrize(
    ("config", "change"),
    [
        (judge_config, {"revision": "main"}),
        (judge_config, {"dtype": "int8"}),
        (judge_config, {"batch_size": 0}),
        (judge_config, {"batch_size": True}),
        (judge_config, {"max_input_tokens": 1.5}),
        (judge_config, {"add_special_tokens": 1}),
        (judge_config, {"input_format": "plain"}),
        (judge_config, {"input_format": "raw"}),
        (judge_config, {"input_format": "raw", "raw_template": "{problem} {solution}"}),
        (judge_config, {"raw_template": "{prompt}"}),
        (judge_config, {"chat_template_kwargs": {"enable_thinking": object()}}),
        (judge_config, {"yes_token": " No"}),
        (judge_config, {"yes_token_id": 5}),
        (judge_config, {"yes_token_id": 5, "no_token_id": 5}),
        (judge_config, {"prompt_file": "../llm_judge.txt"}),
        (judge_config, {"prompt_sha256": "abc"}),
        (judge_config, {"input_format": "raw", "raw_template": "{prompt} {reference}"}),
        (
            judge_config,
            {"input_format": "raw", "raw_template": "{prompt}", "chat_template_kwargs": {"a": 1}},
        ),
        (orm_config, {"input_format": "raw", "raw_template": "{problem} only"}),
        (orm_config, {"input_format": "raw", "raw_template": "{problem} {solution} {reference}"}),
        (orm_config, {"system_prompt": " "}),
        (
            orm_config,
            {"input_format": "raw", "raw_template": "{problem} {solution}", "system_prompt": "s"},
        ),
        (
            orm_config,
            {
                "input_format": "raw",
                "raw_template": "{problem} {solution}",
                "chat_template_kwargs": {"a": 1},
            },
        ),
    ],
)
def test_invalid_configurations_are_refused(config, change):
    with pytest.raises(MathModelError):
        config(**change)


# ---- one entry point ------------------------------------------------------------------


def spec(form, **changes):
    settings = (
        {
            "tokenizer_revision": REVISION,
            "input_format": "chat_template",
            "add_special_tokens": False,
            "yes_token": "Yes",
            "no_token": "No",
            "dtype": "bfloat16",
            "batch_size": 4,
            "max_input_tokens": 8192,
            "prompt_sha256": PROMPT_SHA,
        }
        if form == "llm_judge"
        else {
            "tokenizer_revision": REVISION,
            "input_format": "raw",
            "raw_template": "{problem}\n{solution}",
            "add_special_tokens": True,
            "dtype": "bfloat16",
            "batch_size": 4,
            "max_input_tokens": 8192,
        }
    )
    return models.VerifierSpec(
        **{
            "form": form,
            "vendor": "test-source",
            "base_model": "test-backbone",
            "model_id": "test-org/any-model",
            "revision": REVISION,
            "implementation": JUDGE_IMPLEMENTATION if form == "llm_judge" else ORM_IMPLEMENTATION,
            "settings": settings,
            **changes,
        }
    )


@pytest.mark.parametrize(
    ("form", "adapter", "config_class"),
    [("llm_judge", LLMJudgeVerifier, LLMJudgeConfig), ("orm", ORMVerifier, ORMConfig)],
)
def test_build_verifier_is_the_entry_point_and_loads_nothing(form, adapter, config_class):
    selected = spec(form)
    built = models.build_verifier(selected)
    assert type(built) is adapter and isinstance(built.config, config_class)
    assert not built._prepared and built._model is None
    assert built.config.model_id == selected.model_id and built.config.revision == REVISION
    assert (
        models.build_verifier({**selected.__dict__, "settings": dict(selected.settings)}).config
        == built.config
    )
    with pytest.raises(MathModelError, match="conflict"):
        models.build_verifier(
            replace(selected, settings={**selected.settings, "revision": "2" * 40})
        )
    with pytest.raises(MathModelError, match="incomplete or unknown"):
        models.build_verifier(replace(selected, settings={**selected.settings, "temperature": 0}))
    settings = dict(selected.settings)
    settings.pop("dtype")
    with pytest.raises(MathModelError, match="incomplete or unknown"):
        models.build_verifier(replace(selected, settings=settings))
    with pytest.raises(NotImplementedError, match="not been installed"):
        models.build_verifier(replace(selected, implementation="some-other-backend"))
    other = "orm" if form == "llm_judge" else "llm_judge"
    with pytest.raises(MathModelError, match="relabelled"):
        replace(selected, form=other)
    with pytest.raises(MathModelError, match="relabelled"):
        replace(selected, form="prm")


# ---- the reference answer never reaches a backend -------------------------------------


@pytest.mark.parametrize("adapter", [LLMJudgeVerifier, ORMVerifier])
def test_inputs_are_exactly_problem_and_solution(adapter):
    assert list(inspect.signature(adapter.score).parameters) == ["self", "problem", "solution"]
    assert list(inspect.signature(adapter.score_batch).parameters) == ["self", "items"]
    instance, tokenizer, model = (judge if adapter is LLMJudgeVerifier else orm)()
    for bad in (
        [("q", "s", "2")],
        [{"problem": "q", "solution": "s", "reference_answer": "2"}],
        {"problem": "q", "solution": "s"},
        "q",
    ):
        with pytest.raises(ModelInputError):
            instance.score_batch(bad)
    with pytest.raises(ModelInputError):
        instance.score({"problem": "q", "reference_answer": "2"}, "s")
    assert tokenizer.encoded == [] and model.calls == []


def test_pipeline_scoring_never_passes_the_reference_to_judge_or_orm(tmp_path, monkeypatch):
    h = make_harness(tmp_path, monkeypatch, n=60, fit_n=10, gate_n=10, reserve_n=10)
    pipe = h.pipeline
    pipe.generate()
    pipe.split()
    judge_adapter, judge_tokenizer, _ = judge(judge_config(max_input_tokens=200))
    orm_adapter, orm_tokenizer, _ = orm(orm_config(max_input_tokens=200))
    pipe.verifiers = [judge_adapter, orm_adapter]
    pipe.score(Split.D_FIT)
    references = [ref.reference_answer for ref in h.pipeline.bundle.private_references.values()]
    assert references and all(ref.startswith("PRIVATE_GOLD_") for ref in references)
    chats = judge_tokenizer.chats + orm_tokenizer.chats
    seen = judge_tokenizer.encoded + orm_tokenizer.encoded + [repr(chat) for chat in chats]
    assert len(judge_tokenizer.chats) == len(orm_tokenizer.chats) == 10
    assert not any("PRIVATE_GOLD" in text for text in seen)
    details = list(read_jsonl(h.root / "verifier_details.jsonl"))
    assert len(details) == 20
    assert not any("PRIVATE_GOLD" in str(row) for row in details)
    scores = list(read_jsonl(h.root / "scores.jsonl"))
    assert len(scores) == 20 and all(0 <= row["score"] <= 1 for row in scores)


# ---- pipeline scoring in batches ------------------------------------------------------


class OneByOne:
    """The same adapter seen through score() only: the pipeline's one-by-one path."""

    def __init__(self, adapter, *, fail_at=None):
        self.adapter, self.fail_at, self.calls = adapter, fail_at, 0

    def score(self, problem, solution):
        self.calls += 1
        if self.calls == self.fail_at:
            raise RuntimeError("interrupted")
        return self.adapter.score(problem, solution)


class InterruptedBatches:
    def __init__(self, adapter, *, fail_at):
        self.adapter, self.config, self.fail_at, self.calls = adapter, adapter.config, fail_at, 0

    def score_batch(self, items):
        self.calls += 1
        if self.calls == self.fail_at:
            raise RuntimeError("interrupted")
        return self.adapter.score_batch(items)


def scoring_run(tmp_path, monkeypatch, name):
    root = tmp_path / name
    root.mkdir()
    h = make_harness(root, monkeypatch, n=60, fit_n=10, gate_n=10, reserve_n=10)
    h.pipeline.generate()
    h.pipeline.split()
    judge_adapter, _, judge_model = judge(judge_config(max_input_tokens=200, batch_size=2))
    orm_adapter, _, orm_model = orm(orm_config(max_input_tokens=200, batch_size=3))
    return h, (judge_adapter, orm_adapter), (judge_model, orm_model)


def written(h):
    """scores.jsonl and verifier_details.jsonl in file order, wall-clock fields dropped."""

    def rows(name, clock):
        return [
            {key: value for key, value in row.items() if key != clock}
            for row in read_jsonl(h.root / name)
        ]

    return rows("scores.jsonl", "timestamp"), rows(
        "verifier_details.jsonl", "batch_elapsed_seconds"
    )


def test_batched_pipeline_scoring_is_identical_to_one_by_one(tmp_path, monkeypatch):
    batched, adapters, models_batched = scoring_run(tmp_path, monkeypatch, "batched")
    batched.pipeline.verifiers = list(adapters)
    batched.pipeline.score(Split.D_FIT)
    single, adapters, models_single = scoring_run(tmp_path, monkeypatch, "single")
    single.pipeline.verifiers = [OneByOne(adapter) for adapter in adapters]
    single.pipeline.score(Split.D_FIT)
    scores, details = written(batched)
    assert len(scores) == len(details) == 20
    # same rows, same values bit for bit, same order: state by state, V1 then V2
    assert (scores, details) == written(single)
    assert [row["verifier_id"] for row in scores] == ["fixture-V1", "fixture-V2"] * 10
    # 10 states: chunks of lcm(2, 3) = 6 and 4; judge 3 + 2 batches, ORM 2 + 2 (one filled)
    assert [len(model.calls) for model in models_batched] == [5, 4]
    assert [len(model.calls) for model in models_single] == [10, 10]
    assert {call[0] for call in models_batched[0].calls} == {(2, 200)}
    assert set(models_batched[1].calls) == {(3, 200)}


@pytest.mark.parametrize("interrupted", ["batched", "one_by_one"])
def test_batched_scoring_resumes_after_an_interruption(tmp_path, monkeypatch, interrupted):
    reference, adapters, _ = scoring_run(tmp_path, monkeypatch, "reference")
    reference.pipeline.verifiers = list(adapters)
    reference.pipeline.score(Split.D_FIT)
    h, (judge_adapter, orm_adapter), _ = scoring_run(tmp_path, monkeypatch, "resumed")
    if interrupted == "batched":
        # the second chunk's ORM batch fails: nothing of that chunk is written
        h.pipeline.verifiers = [judge_adapter, InterruptedBatches(orm_adapter, fail_at=2)]
    else:
        # _score goes through _score_items (2026-10-13): both V score a state before any
        # row is written, so the eighth state, whose ORM call fails, has no row at all
        h.pipeline.verifiers = [judge_adapter, OneByOne(orm_adapter, fail_at=8)]
    states = h.pipeline._states()
    with pytest.raises(RuntimeError, match="interrupted"):
        if interrupted == "batched":
            h.pipeline.score(Split.D_FIT)
        else:
            for qid in h.pipeline._split_ids(Split.D_FIT):
                h.pipeline._score(qid, states[qid].text, states[qid].state_id)
    partial = len(list(read_jsonl(h.root / "scores.jsonl")))
    assert partial == (12 if interrupted == "batched" else 14)
    h.reopen()
    h.pipeline.verifiers = [judge_adapter, orm_adapter]
    h.pipeline.score(Split.D_FIT)
    assert written(h) == written(reference)
