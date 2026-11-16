"""The real V1 / V2 candidates against score_spec.md (II.4.1, App. A.2).

Offline and torch-free by construction: every test drives a fake backend that returns mock
logits, so nothing here downloads weights, imports torch or touches the network. What the
fakes cannot cover -- that the label prefixes and spellings match what the weights actually
emit -- is covered by ``batch_score --self-check`` on the GPU box and is flagged as such in
the registry.
"""

from __future__ import annotations

import ast
import math
from collections.abc import Sequence
from pathlib import Path

import pytest

from rafr.core.interfaces import Verifier
from rafr.verifiers.candidates import base, encoder, guard_lm, registry
from rafr.verifiers.candidates.base import (
    Truncation,
    VerifierConfigError,
    chunked,
    logsumexp,
    resolve_batch_size,
    sigmoid,
    softmax,
)
from rafr.verifiers.candidates.encoder import (
    MULTILABEL_SIGMOID,
    SINGLE_LABEL_SOFTMAX,
    EncoderToxicityVerifier,
)
from rafr.verifiers.candidates.guard_lm import (
    CHAT,
    CONTROVERSIAL_AS_SAFE,
    CONTROVERSIAL_AS_UNSAFE,
    CONTROVERSIAL_EXCLUDE,
    RAW,
    GuardLMVerifier,
    LabelGroup,
)

CANDIDATES_DIR = Path(base.__file__).resolve().parent


# ---------------------------------------------------------------------------------------
# fakes
# ---------------------------------------------------------------------------------------


class FakeEncoderBackend:
    """Mock class logits. Records the shapes it was asked to run, for the batching tests."""

    revision = "fakerev"

    def __init__(self, id2label: dict[int, str], rows: dict[str, list[float]]) -> None:
        self.id2label = id2label
        self._rows = rows
        self.calls: list[list[str]] = []

    def raw_rows(self, items: Sequence[object]) -> list[list[float]]:
        texts = [str(t) for t in items]
        self.calls.append(texts)
        return [list(self._rows[t]) for t in texts]


class FakeGuardBackend:
    """Mock label-token logits plus a word-level tokenizer, so truncation is observable."""

    revision = "fakerev"

    def __init__(
        self,
        group_sizes: tuple[int, ...],
        rows: dict[str, list[float]],
        max_words: int = 4,
    ) -> None:
        self.group_sizes = group_sizes
        self._rows = rows
        self.max_words = max_words
        self.prompts: list[str] = []
        self.truncated: list[str] = []

    def truncate_text(self, text: str) -> str:
        words = text.split()
        if len(words) <= self.max_words:
            return text
        cut = " ".join(words[: self.max_words])
        self.truncated.append(cut)
        return cut

    def raw_rows(self, items: Sequence[object]) -> list[list[float]]:
        out = []
        for prompt in items:
            self.prompts.append(str(prompt))
            out.append(list(self._rows[str(prompt)]))
        return out


TOXIC_BERT_LABELS = {
    0: "toxic",
    1: "severe_toxic",
    2: "obscene",
    3: "threat",
    4: "insult",
    5: "identity_hate",
}


def make_encoder(head: str, rows: dict[str, list[float]], **kwargs: object) -> tuple:
    labels = TOXIC_BERT_LABELS if head == MULTILABEL_SIGMOID else {0: "neutral", 1: "toxic"}
    backend = FakeEncoderBackend(labels, rows)
    v = EncoderToxicityVerifier(
        verifier_id="fake-encoder",
        repo_id="fake/encoder",
        revision="deadbeef",
        head=head,
        toxic_label="toxic",
        truncation=Truncation(max_length=512),
        backend=backend,
        **kwargs,  # type: ignore[arg-type]
    )
    return v, backend


def make_guard(rows: dict[str, list[float]], **kwargs: object) -> tuple:
    labels = kwargs.pop("labels", None) or (
        LabelGroup("safe", ("safe",)),
        LabelGroup("unsafe", ("unsafe",)),
    )
    sizes = kwargs.pop("group_sizes", None) or tuple(1 for _ in labels)
    backend = FakeGuardBackend(tuple(sizes), rows)
    v = GuardLMVerifier(
        verifier_id="fake-guard",
        repo_id="fake/guard",
        revision="deadbeef",
        labels=labels,
        label_prefix=kwargs.pop("label_prefix", ""),
        prompt_style=kwargs.pop("prompt_style", RAW),
        raw_template=kwargs.pop("raw_template", "JUDGE:\n{text}\nANSWER:"),
        truncation=Truncation(max_length=4),
        backend=backend,
        **kwargs,  # type: ignore[arg-type]
    )
    return v, backend


# ---------------------------------------------------------------------------------------
# pure numerics
# ---------------------------------------------------------------------------------------


def test_pure_numerics():
    assert sigmoid(0.0) == 0.5
    assert sigmoid(800.0) == 1.0  # no OverflowError on either tail
    assert sigmoid(-800.0) == 0.0
    assert softmax([5.0]) == [1.0]
    p = softmax([1.0, 2.0, 3.0])
    assert math.isclose(sum(p), 1.0) and p[2] > p[1] > p[0]
    assert math.isclose(softmax([1000.0, 1000.0])[0], 0.5)
    # logsumexp of two equal logits is that logit + log 2
    assert math.isclose(logsumexp([2.0, 2.0]), 2.0 + math.log(2))
    assert math.isclose(logsumexp([700.0]), 700.0)
    with pytest.raises(VerifierConfigError):
        base.clamp_score(float("nan"), where="test")


def test_batch_size_resolution(monkeypatch):
    monkeypatch.delenv(base.BATCH_SIZE_ENV, raising=False)
    assert resolve_batch_size(None) == base.DEFAULT_INFERENCE_BATCH_SIZE
    assert resolve_batch_size(3) == 3
    monkeypatch.setenv(base.BATCH_SIZE_ENV, "16")
    assert resolve_batch_size(None) == 16
    assert resolve_batch_size(2) == 2  # explicit still wins
    monkeypatch.setenv(base.BATCH_SIZE_ENV, "zero")
    with pytest.raises(VerifierConfigError):
        resolve_batch_size(None)
    with pytest.raises(VerifierConfigError):
        resolve_batch_size(0)


def test_chunked_preserves_order_and_covers_the_input():
    items = [f"t{i}" for i in range(7)]
    chunks = chunked(items, 3)
    assert [len(c) for c in chunks] == [3, 3, 1]
    assert [t for c in chunks for t in c] == items


# ---------------------------------------------------------------------------------------
# score_spec.md
# ---------------------------------------------------------------------------------------


def test_encoder_conforms_to_score_spec():
    rows = {
        "a": [2.0, -3.0, -1.0, -5.0, 0.5, -4.0],
        "b": [-2.0, -3.0, -1.0, -5.0, 0.5, -4.0],
    }
    v, _ = make_encoder(MULTILABEL_SIGMOID, rows)
    assert isinstance(v, Verifier)
    assert v.model_version == "fake/encoder@deadbeef"
    assert v.score([]) == []
    scores = v.score(["a", "b"])
    assert scores == [sigmoid(2.0), sigmoid(-2.0)]
    assert all(0.0 <= s <= 1.0 for s in scores)
    with pytest.raises(TypeError):
        v.score("a")
    with pytest.raises(TypeError):
        v.score([1])  # type: ignore[list-item]


def test_encoder_reads_the_toxic_class_by_name_not_by_position():
    """severe_toxic sits at index 1; scoring it instead of 'toxic' must be impossible."""
    rows = {"a": [0.0, 9.0, 0.0, 0.0, 0.0, 0.0]}
    v, _ = make_encoder(MULTILABEL_SIGMOID, rows)
    assert v.toxic_index == 0
    assert v.score(["a"]) == [sigmoid(0.0)]

    shuffled = FakeEncoderBackend({0: "insult", 1: "toxic", 2: "obscene"}, {"a": [9.0, 1.0, 9.0]})
    v2 = EncoderToxicityVerifier(
        verifier_id="fake",
        repo_id="fake/enc",
        revision="r",
        head=MULTILABEL_SIGMOID,
        toxic_label="toxic",
        truncation=Truncation(max_length=8),
        backend=shuffled,
    )
    assert v2.toxic_index == 1 and v2.score(["a"]) == [sigmoid(1.0)]

    missing = FakeEncoderBackend({0: "LABEL_0", 1: "LABEL_1"}, {"a": [0.0, 1.0]})
    v3 = EncoderToxicityVerifier(
        verifier_id="fake",
        repo_id="fake/enc",
        revision="r",
        head=SINGLE_LABEL_SOFTMAX,
        toxic_label="toxic",
        truncation=Truncation(max_length=8),
        backend=missing,
    )
    with pytest.raises(VerifierConfigError, match="exactly one class named 'toxic'"):
        v3.score(["a"])


def test_encoder_softmax_head_matches_the_two_class_sigmoid():
    rows = {"a": [-1.0, 2.0]}
    v, _ = make_encoder(SINGLE_LABEL_SOFTMAX, rows)
    got = v.score(["a"])[0]
    assert math.isclose(got, softmax([-1.0, 2.0])[1])
    assert math.isclose(got, sigmoid(2.0 - (-1.0)))


def test_guard_score_is_p_unsafe_over_the_label_tokens():
    rows = {"JUDGE:\nhello\nANSWER:": [1.0, 3.0]}
    v, _ = make_guard(rows)
    assert v.score(["hello"]) == [softmax([1.0, 3.0])[1]]


def test_guard_folds_several_spellings_of_one_label():
    """Two token ids spelling 'unsafe' add up in probability space, i.e. logsumexp."""
    labels = (LabelGroup("safe", ("No", "no")), LabelGroup("unsafe", ("Yes", "yes")))
    rows = {"JUDGE:\nhello\nANSWER:": [0.0, 0.0, 1.0, 1.0]}
    v, _ = make_guard(rows, labels=labels, group_sizes=(2, 2))
    expected = softmax([logsumexp([0.0, 0.0]), logsumexp([1.0, 1.0])])[1]
    assert v.score(["hello"]) == [expected]
    probs = v.label_probabilities(rows["JUDGE:\nhello\nANSWER:"])
    assert math.isclose(sum(probs.values()), 1.0)


# ---------------------------------------------------------------------------------------
# determinism, batching, order
# ---------------------------------------------------------------------------------------


def test_determinism_bit_for_bit_across_calls():
    rows = {f"t{i}": [float(i), 1.0 - i, 0.0, 0.0, 0.0, 0.0] for i in range(10)}
    v, _ = make_encoder(MULTILABEL_SIGMOID, rows, inference_batch_size=3)
    texts = list(rows)
    first = v.score(texts)
    second = v.score(texts)
    assert first == second  # exact equality, not isclose
    assert all(a.hex() == b.hex() for a, b in zip(first, second, strict=True))


def test_batching_is_internal_and_preserves_order():
    rows = {f"t{i}": [float(i), 0.0, 0.0, 0.0, 0.0, 0.0] for i in range(7)}
    v, backend = make_encoder(MULTILABEL_SIGMOID, rows, inference_batch_size=3)
    texts = list(rows)
    scores = v.score(texts)
    assert [len(c) for c in backend.calls] == [3, 3, 1]
    assert [t for c in backend.calls for t in c] == texts
    assert scores == [sigmoid(float(i)) for i in range(7)]
    # A different batch size must give the identical numbers.
    v2, _ = make_encoder(MULTILABEL_SIGMOID, rows, inference_batch_size=7)
    assert v2.score(texts) == scores


def test_duplicate_texts_keep_their_positions():
    rows = {"x": [1.0, 0, 0, 0, 0, 0], "y": [-1.0, 0, 0, 0, 0, 0]}
    v, _ = make_encoder(MULTILABEL_SIGMOID, rows, inference_batch_size=2)
    scores = v.score(["x", "y", "x", "x", "y"])
    assert scores == [sigmoid(1.0), sigmoid(-1.0), sigmoid(1.0), sigmoid(1.0), sigmoid(-1.0)]


def test_batch_limit_is_enforced():
    rows = {"x": [0.0] * 6}
    v, _ = make_encoder(MULTILABEL_SIGMOID, rows, batch_limit=4, inference_batch_size=2)
    v.score(["x"] * 4)
    with pytest.raises(ValueError, match="exceeds batch_limit"):
        v.score(["x"] * 5)
    with pytest.raises(VerifierConfigError, match="exceeds batch_limit"):
        make_encoder(MULTILABEL_SIGMOID, rows, batch_limit=2, inference_batch_size=8)


# ---------------------------------------------------------------------------------------
# truncation
# ---------------------------------------------------------------------------------------


def test_truncation_policy_is_declared_and_described():
    t = Truncation(max_length=512, side="right")
    text = t.describe()
    assert "512" in text and "head" in text and "padded to max_length" in text
    assert "tail" in Truncation(max_length=8, side="left").describe()
    with pytest.raises(VerifierConfigError):
        Truncation(max_length=0)
    with pytest.raises(VerifierConfigError):
        Truncation(max_length=8, side="middle")


def test_guard_truncates_the_text_not_the_template():
    long_text = "one two three four five six"
    short = "one two three four"
    rows = {
        f"JUDGE:\n{short}\nANSWER:": [0.0, 2.0],
        "JUDGE:\none two\nANSWER:": [0.0, 5.0],
    }
    v, backend = make_guard(rows)  # fake backend cuts at 4 words
    assert v.score([long_text]) == [softmax([0.0, 2.0])[1]]
    assert backend.truncated == [short]
    # the template survived the cut
    assert backend.prompts[0].startswith("JUDGE:\n") and backend.prompts[0].endswith("\nANSWER:")
    # a text inside the budget is passed through untouched
    v.score(["one two"])
    assert backend.truncated == [short]


def test_encoder_truncation_is_passed_to_the_tokenizer_not_done_by_hand():
    """The torch backend must hand max_length / truncation to the tokenizer itself."""
    src = (CANDIDATES_DIR / "torch_backend.py").read_text(encoding="utf-8")
    assert "truncation=True" in src
    assert "max_length=self.truncation.max_length" in src
    assert "self.tokenizer.truncation_side = truncation.side" in src


def test_guard_prompts_are_never_truncated_by_the_tokenizer():
    """Cutting a guard prompt from the right would remove the label position itself, so an
    over-long prompt must raise rather than be silently trimmed."""
    src = (CANDIDATES_DIR / "torch_backend.py").read_text(encoding="utf-8")
    assert "truncation=False" in src
    assert "above the measured " in src
    # the prompt-length probe goes through the verifier's own templating, not the bare text
    assert "probe = self.render(self.template_fn(filler))" in src
    assert "template_fn=self.build_prompt" in (CANDIDATES_DIR / "guard_lm.py").read_text(
        encoding="utf-8"
    )


def test_self_check_decodes_without_the_teacher_forced_prefix():
    """Handing the model the prefix would make the self-check unable to falsify it."""
    src = (CANDIDATES_DIR / "torch_backend.py").read_text(encoding="utf-8")
    assert "self.render(prompt, include_prefix=False)" in src


# ---------------------------------------------------------------------------------------
# prompts
# ---------------------------------------------------------------------------------------


def test_raw_prompt_slots_are_filled_in_one_pass():
    """A text containing '{guideline}' must not pull the guideline in (fill_slots)."""
    template = "P: {guideline}\nT: {text}\nA:"
    rows = {"P: no slurs\nT: {guideline} is literal\nA:": [0.0, 1.0]}
    v, _ = make_guard(rows, raw_template=template, guideline="no slurs")
    assert v.score(["{guideline} is literal"]) == [softmax([0.0, 1.0])[1]]


def test_chat_prompt_is_a_conversation_the_backend_templates():
    v, _ = make_guard({}, prompt_style=CHAT, raw_template=None)
    assert v.build_prompt("hello") == [{"role": "user", "content": "hello"}]


def test_empty_guideline_is_refused():
    with pytest.raises(VerifierConfigError, match="guideline is set but empty"):
        make_guard({}, guideline="   ")


# ---------------------------------------------------------------------------------------
# the third verdict
# ---------------------------------------------------------------------------------------


def _three_way(policy):
    labels = (
        LabelGroup("safe", ("Safe",)),
        LabelGroup("unsafe", ("Unsafe",)),
        LabelGroup("controversial", ("Controversial",)),
    )
    rows = {"JUDGE:\nhello\nANSWER:": [0.0, 1.0, 2.0]}
    v, _ = make_guard(rows, labels=labels, controversial_policy=policy)
    return v, softmax([0.0, 1.0, 2.0])


def test_controversial_policy_has_no_default_and_three_explicit_behaviours():
    with pytest.raises(VerifierConfigError, match="design decision, not a default"):
        _three_way(None)
    with pytest.raises(VerifierConfigError, match="must be one of"):
        _three_way("maybe")

    v, p = _three_way(CONTROVERSIAL_AS_UNSAFE)
    assert math.isclose(v.score(["hello"])[0], p[1] + p[2])
    v, p = _three_way(CONTROVERSIAL_AS_SAFE)
    assert math.isclose(v.score(["hello"])[0], p[1])
    v, p = _three_way(CONTROVERSIAL_EXCLUDE)
    assert math.isclose(v.score(["hello"])[0], p[1] / (p[0] + p[1]))


def test_controversial_policy_is_refused_on_a_binary_model():
    with pytest.raises(VerifierConfigError, match="no 'controversial' label"):
        make_guard({}, controversial_policy=CONTROVERSIAL_AS_UNSAFE)


def test_guard_requires_both_binary_labels():
    with pytest.raises(VerifierConfigError, match="no 'unsafe' label"):
        make_guard({}, labels=(LabelGroup("safe", ("no",)), LabelGroup("other", ("x",))))


# ---------------------------------------------------------------------------------------
# registry
# ---------------------------------------------------------------------------------------


def test_registry_covers_both_families_and_the_named_candidates():
    expected = {
        "toxic-bert": "unitary/toxic-bert",
        "roberta-toxicity": "s-nlp/roberta_toxicity_classifier",
        "llama-guard-3-8b": "meta-llama/Llama-Guard-3-8B",
        "llama-guard-3-1b": "meta-llama/Llama-Guard-3-1B",
        "qwen3guard-gen-0.6b": "Qwen/Qwen3Guard-Gen-0.6B",
        "qwen3guard-gen-4b": "Qwen/Qwen3Guard-Gen-4B",
        "qwen3guard-gen-8b": "Qwen/Qwen3Guard-Gen-8B",
        "shieldgemma-2b": "google/shieldgemma-2b",
        "shieldgemma-9b": "google/shieldgemma-9b",
        "granite-guardian-3.3-8b": "ibm-granite/granite-guardian-3.3-8b",
    }
    assert {n: registry.spec(n).repo_id for n in registry.names()} == expected
    kinds = {registry.spec(n).kind for n in registry.names()}
    assert kinds == {registry.ENCODER, registry.GUARD_LM}
    assert all(s.hub_sha_at_check for s in registry.CANDIDATES.values())


def test_no_openai_or_anthropic_model_is_a_candidate():
    """II.9.1 / App. D: the controller is GPT-5.6 and E is Claude Opus 5, so V1/V2 may
    come from neither source. gpt-oss-safeguard is covered by the openai/ prefix."""
    for spec in registry.CANDIDATES.values():
        lowered = spec.repo_id.lower()
        assert not lowered.startswith(registry.FORBIDDEN_VENDOR_PREFIXES)
        assert "gpt-oss" not in lowered
        assert "claude" not in lowered


def test_factories_are_zero_argument_and_resolve():
    import importlib

    assert set(registry.FACTORIES) == set(registry.names())
    for name, spec_str in registry.FACTORIES.items():
        mod, _, fn = spec_str.partition(":")
        factory = getattr(importlib.import_module(mod), fn)
        assert callable(factory), name


def test_open_candidates_build_without_touching_the_network(monkeypatch):
    monkeypatch.setenv(registry.MANIFEST_ENV, "/nonexistent/model_manifest.json")
    for name in ("toxic-bert", "roberta-toxicity", "llama-guard-3-8b", "llama-guard-3-1b"):
        v = registry.build(name)
        assert isinstance(v, Verifier)
        assert v.model_version.endswith("@unpinned")
        assert v.truncation.max_length == registry.TEXT_MAX_TOKENS
        assert v._backend is None  # nothing loaded


def test_candidates_with_a_pending_decision_refuse_to_build():
    for name, needle in (
        ("qwen3guard-gen-8b", "design decision"),
        ("shieldgemma-2b", "SHIELDGEMMA_GUIDELINE"),
        ("granite-guardian-3.3-8b", "GRANITE_CRITERIA_ID"),
    ):
        with pytest.raises(VerifierConfigError, match=needle):
            registry.build(name)


def test_pending_decisions_are_none_and_marked():
    src = (CANDIDATES_DIR / "registry.py").read_text(encoding="utf-8")
    for const in (
        "QWEN3GUARD_CONTROVERSIAL_POLICY",
        "SHIELDGEMMA_GUIDELINE",
        "GRANITE_CRITERIA_ID",
    ):
        assert getattr(registry, const) is None
        line = next(ln for ln in src.splitlines() if ln.startswith(f"{const}:"))
        assert "TODO(decision)" in line


def test_granite_criteria_reaches_the_chat_template():
    """The criterion must survive the merge with the spec's own chat_template_kwargs, and
    supplying only `think` must not smuggle past the pending decision."""
    with pytest.raises(VerifierConfigError, match="GRANITE_CRITERIA_ID"):
        registry.build("granite-guardian-3.3-8b", chat_template_kwargs={"think": False})

    v = registry.build(
        "granite-guardian-3.3-8b",
        chat_template_kwargs={"guardian_config": {"criteria_id": "harm"}},
    )
    assert v.chat_template_kwargs["guardian_config"] == {"criteria_id": "harm"}
    assert v.chat_template_kwargs["think"] is False  # 3.3 thinking mode stays off


def test_manifest_pins_the_revision(tmp_path, monkeypatch):
    import json

    manifest = tmp_path / "model_manifest.json"
    manifest.write_text(
        json.dumps({"models": {"unitary/toxic-bert": {"revision": "abc123"}}}), encoding="utf-8"
    )
    monkeypatch.setenv(registry.MANIFEST_ENV, str(manifest))
    v = registry.build("toxic-bert")
    assert v.revision == "abc123"
    assert v.model_version == "unitary/toxic-bert@abc123"
    assert v.is_pinned
    assert registry.pinned_revision("google/shieldgemma-2b") is None


def test_registry_cli_lists_the_pool(capsys):
    registry._cli(["--format", "repos"])
    printed = capsys.readouterr().out.split()
    assert printed == [registry.spec(n).repo_id for n in registry.names()]

    registry._cli(["--format", "tsv", "--gated", "only"])
    rows = [ln.split("\t") for ln in capsys.readouterr().out.strip().splitlines()]
    assert {r[0] for r in rows} == {
        "llama-guard-3-8b",
        "llama-guard-3-1b",
        "shieldgemma-2b",
        "shieldgemma-9b",
    }
    assert all(r[4] == "gated" for r in rows)

    registry._cli(["--format", "tsv", "--name", "toxic-bert"])
    assert capsys.readouterr().out.strip().split("\t")[1] == "unitary/toxic-bert"


# ---------------------------------------------------------------------------------------
# no torch at import time, no sampling anywhere on the scoring path
# ---------------------------------------------------------------------------------------


def test_nothing_imports_torch_or_transformers_at_module_level():
    """The whole suite must pass with no GPU dependencies installed."""
    offenders = []
    for path in sorted(CANDIDATES_DIR.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Import | ast.ImportFrom):
                continue
            names_ = (
                [a.name for a in node.names]
                if isinstance(node, ast.Import)
                else [node.module or ""]
            )
            if not any(n.split(".")[0] in ("torch", "transformers") for n in names_):
                continue
            if node.col_offset == 0:  # module level
                offenders.append(f"{path.name}:{node.lineno}")
    assert offenders == [], f"module-level GPU imports: {offenders}"


def test_the_scoring_path_never_generates_or_samples():
    """score() is one forward pass. Only the GPU self-check may call generate, and greedily."""
    src = (CANDIDATES_DIR / "torch_backend.py").read_text(encoding="utf-8")
    tree = ast.parse(src)
    generating = []
    for fn in ast.walk(tree):
        if not isinstance(fn, ast.FunctionDef):
            continue
        for node in ast.walk(fn):
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "generate"
            ):
                generating.append(fn.name)
                kwargs = {kw.arg: kw.value for kw in node.keywords}
                assert "do_sample" in kwargs and kwargs["do_sample"].value is False
    assert generating == ["greedy_head"], generating

    for forbidden in ("multinomial", "temperature", "top_p", "top_k", ".train()"):
        assert forbidden not in src, forbidden
    assert "model.eval()" in src
    assert "torch.inference_mode()" in src
    assert '"attn_implementation": "eager"' in src


def test_fixed_batch_shape_is_implemented_not_just_claimed():
    """The card states a fixed batch shape; pad_batch is what makes that true."""
    from rafr.verifiers.candidates.torch_backend import pad_batch

    padded, n_real = pad_batch(["a", "b", "c"], 8)
    assert len(padded) == 8 and n_real == 3
    assert padded[:3] == ["a", "b", "c"] and set(padded[3:]) == {"c"}
    assert pad_batch(["a"] * 8, 8) == (["a"] * 8, 8)  # already full
    assert pad_batch([], 8) == ([], 0)  # nothing to pad

    src = (CANDIDATES_DIR / "torch_backend.py").read_text(encoding="utf-8")
    # both backends must pad, and must slice the fillers back off
    assert src.count("pad_batch(list(items), self.batch_size)") == 2
    assert src.count("[:n_real]") == 2


def test_dynamic_padding_env_reaches_the_truncation_flags_and_the_card(monkeypatch):
    """The verifier card must describe what the run did, not what the default would be."""
    monkeypatch.delenv(base.DYNAMIC_PADDING_ENV, raising=False)
    t = registry.truncation()
    assert t.pad_to_max_length and t.fixed_batch_shape
    assert "padded to max_length" in t.describe()

    monkeypatch.setenv(base.DYNAMIC_PADDING_ENV, "1")
    t = registry.truncation()
    assert not t.pad_to_max_length and not t.fixed_batch_shape
    assert "dynamic padding" in t.describe() and "ragged last batch" in t.describe()
    assert registry.build("toxic-bert").truncation.pad_to_max_length is False

    # the torch backend reads the flags, never the environment directly: one source of
    # truth, so the card cannot describe a padding scheme the run did not use
    src = (CANDIDATES_DIR / "torch_backend.py").read_text(encoding="utf-8")
    assert "dynamic_padding_requested" not in src
    assert "self.truncation.pad_to_max_length" in src
    assert "self.truncation.fixed_batch_shape" in src


def test_padding_mode_switches_both_flags_and_restores_them():
    """Selection runs dynamic, steps 6 and 12 run fixed; the comparison must be able to do
    both on one loaded model rather than paying to load an 8B twice."""
    rows = {"a": [1.0, 0.0, 0.0, 0.0, 0.0, 0.0]}
    v, backend = make_encoder(MULTILABEL_SIGMOID, rows)
    backend.truncation = v.truncation
    assert v.truncation.pad_to_max_length and v.truncation.fixed_batch_shape

    with base.padding_mode(v, fixed=False):
        assert not v.truncation.pad_to_max_length and not v.truncation.fixed_batch_shape
        assert backend.truncation is v.truncation  # the backend sees the same variant
        assert v.truncation.max_length == 512  # everything else is carried over
    assert v.truncation.pad_to_max_length and v.truncation.fixed_batch_shape
    assert backend.truncation is v.truncation

    # restored even when the body raises
    with pytest.raises(RuntimeError), base.padding_mode(v, fixed=False):
        raise RuntimeError("boom")
    assert v.truncation.fixed_batch_shape


def test_padding_comparison_counts_flips_with_and_without_a_threshold():
    """A judgment y = 1[s >= tau] flips exactly when tau lands in (min, max]."""
    fixed = [0.10, 0.40, 0.80, 0.55]
    dynamic = [0.10, 0.60, 0.81, 0.55]

    c = base.compare_scores(fixed, dynamic)
    assert c.n == 4
    assert math.isclose(c.max_abs_diff, 0.20)
    assert c.n_differing == 2  # items 1 and 2; 0 and 3 are identical
    assert c.threshold is None and c.n_flips_at_threshold is None
    # no single tau sits inside both (0.40, 0.60] and (0.80, 0.81]
    assert c.worst_case_flips == 1
    assert not c.clean()

    # tau = 0.50 is inside (0.40, 0.60] only
    assert base.compare_scores(fixed, dynamic, 0.50).n_flips_at_threshold == 1
    # tau = 0.805 is inside (0.80, 0.81] only
    assert base.compare_scores(fixed, dynamic, 0.805).n_flips_at_threshold == 1
    # tau = 0.30 is inside neither
    assert base.compare_scores(fixed, dynamic, 0.30).n_flips_at_threshold == 0
    # the interval is half-open on the left: tau exactly at the lower score does not flip
    assert base.compare_scores([0.4], [0.6], 0.40).n_flips_at_threshold == 0
    assert base.compare_scores([0.4], [0.6], 0.60).n_flips_at_threshold == 1


def test_padding_comparison_is_clean_when_the_modes_agree():
    same = [0.1, 0.9, 0.5]
    c = base.compare_scores(same, list(same), 0.5)
    assert c.clean() and c.worst_case_flips == 0 and c.n_flips_at_threshold == 0
    assert c.max_abs_diff == 0.0 and c.n_differing == 0
    assert base.compare_scores([], []).clean()


def test_padding_comparison_finds_overlapping_flip_intervals():
    """Two items whose intervals overlap can be flipped by one tau; the worst case is 2."""
    c = base.compare_scores([0.30, 0.35], [0.70, 0.65])
    assert c.worst_case_flips == 2
    assert c.worst_case_threshold in (0.65, 0.70)
    with pytest.raises(VerifierConfigError):
        base.compare_scores([0.1], [0.1, 0.2])


def test_determinism_switches_are_set():
    src = (CANDIDATES_DIR / "torch_backend.py").read_text(encoding="utf-8")
    for switch in (
        "cudnn.deterministic = True",
        "cudnn.benchmark = False",
        "matmul.allow_tf32 = False",
        "use_deterministic_algorithms(True",
        "CUBLAS_WORKSPACE_CONFIG",
        'padding="max_length"',
    ):
        assert switch in src, switch


def test_guard_backend_pads_left_and_rebuilds_position_ids():
    """The score is read at the last position, so right padding would read a pad token."""
    src = (CANDIDATES_DIR / "torch_backend.py").read_text(encoding="utf-8")
    assert 'self.tokenizer.padding_side = "left"' in src
    assert "position_ids = (attention_mask.cumsum(-1) - 1).clamp(min=0)" in src
    assert "logits[:, -1, :]" in src


def test_encoder_and_guard_modules_stay_pure():
    """encoder.py / guard_lm.py / base.py hold the numerics; none of them may mention torch."""
    for name in ("base.py", "encoder.py", "guard_lm.py", "registry.py"):
        src = (CANDIDATES_DIR / name).read_text(encoding="utf-8")
        assert "import torch" not in src, name
        assert "import transformers" not in src, name


def test_score_semantics_and_normalization_are_stated_for_every_buildable_candidate():
    """App. A.2: the card needs "which logit" and "how normalised" for each model."""
    for name in ("toxic-bert", "roberta-toxicity", "llama-guard-3-8b"):
        v = registry.build(name)
        assert v.score_semantics and v.normalization
        assert "[0, 1]" not in v.score_semantics  # range lives in the card, not here
    v = registry.build("toxic-bert")
    assert "sigmoid" in v.normalization and "toxic" in v.score_semantics
    v = registry.build("roberta-toxicity")
    assert "softmax" in v.normalization
    v = registry.build("llama-guard-3-8b")
    assert "logits" in v.score_semantics and "Nothing is sampled" in v.score_semantics


def test_module_docstrings_point_at_the_contract():
    assert "score_spec.md" in (encoder.__doc__ or "") + (base.__doc__ or "")
    assert "II.4.1" in (guard_lm.__doc__ or "")
