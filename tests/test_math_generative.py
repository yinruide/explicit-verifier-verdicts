"""Generative math verifiers (rafr.domains.math.generative; M.4 v1.2, II.10 generative variant).

Configuration checks run everywhere. The decoding loop runs on a scripted fake model (a word
tokenizer and a model that emits a fixed continuation per prompt) where torch is installed: stop
string, end-of-sequence, truncation at L_gen, the forced suffix, the readouts, and the three
missing-verdict rules of the step side. A tiny random Llama then checks the transformers path:
left padding is transparent, a row does not depend on its batch, and two runs agree bit for bit.
"""

from __future__ import annotations

import copy
import hashlib
import math
from dataclasses import replace
from types import SimpleNamespace

import pytest

from rafr.domains.math import generative as G
from rafr.domains.math.models import MathModelError, VerifierSpec, build_verifier

JUDGE_PROMPT = "Problem: {problem} Solution: {solution} Is it right ?"
VERDICT_PROMPT = "Final verdict :"
STEP_PROMPT = "Check each step . Problem: {problem} Steps: {solution}"
REV = "a" * 40


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


@pytest.fixture
def prompt_dir(tmp_path):
    for name, text in (
        ("judge.txt", JUDGE_PROMPT),
        ("verdict.txt", VERDICT_PROMPT),
        ("step.txt", STEP_PROMPT),
    ):
        (tmp_path / name).write_text(text + "\n", encoding="utf-8", newline="\n")
    return tmp_path


def judge_config(**changes):
    base = dict(
        model_id="test/judge",
        revision=REV,
        tokenizer_revision=REV,
        model_class="AutoModelForCausalLM",
        prompt_file="judge.txt",
        prompt_sha256=_sha(JUDGE_PROMPT),
        verdict_prompt_file="verdict.txt",
        verdict_prompt_sha256=_sha(VERDICT_PROMPT),
        stop_string="</think>",
        eos_token_ids=[1],
        positive_token="Yes",
        negative_token="No",
        dtype="float32",
        batch_size=2,
        max_input_tokens=24,
        max_new_tokens=6,
        max_forced_tokens=8,
        license_reviewed=True,
        license_review_reference="test",
        device_map="cpu",
    )
    return G.GenerativeJudgeConfig(**{**base, **changes})


def step_config(**changes):
    base = dict(
        model_id="test/prm",
        revision=REV,
        tokenizer_revision=REV,
        model_class="AutoModelForCausalLM",
        prompt_file="step.txt",
        prompt_sha256=_sha(STEP_PROMPT),
        step_format="Step {k} : {step}",
        step_joiner=" ",
        verdict_pattern=r"Step\s*(\d+)\s*is\s*\\boxed\{",
        forced_verdict_prefix="Step {k} is \\boxed{",
        verdict_close={"positive": "}", "negative": "}"},
        missing_verdicts="force_remaining",
        stop_string="</think>",
        eos_token_ids=[1],
        positive_token="correct",
        negative_token="incorrect",
        dtype="float32",
        batch_size=2,
        max_input_tokens=40,
        max_new_tokens=12,
        max_forced_tokens=40,
        license_reviewed=True,
        license_review_reference="test",
        device_map="cpu",
    )
    return G.GenerativeStepConfig(**{**base, **changes})


# ---------------------------------------------------------------------------------------
# configuration (no torch)
# ---------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "change, match",
    [
        ({"max_new_tokens": "TBD"}, "still TBD"),
        ({"trust_remote_code": True}, "native transformers"),
        ({"model_class": "AutoModel"}, "model_class"),
        ({"stop_string": None}, "stop string"),
        ({"positive_token": "No"}, "must differ"),
        ({"eos_token_ids": []}, "eos_token_ids"),
        ({"dtype": "int4"}, "quantization"),
    ],
)
def test_judge_config_refuses(change, match):
    with pytest.raises(MathModelError, match=match):
        judge_config(**change)


@pytest.mark.parametrize(
    "change, match",
    [
        ({"missing_verdicts": "TBD"}, "still TBD"),
        ({"missing_verdicts": "drop"}, "missing_verdicts"),
        ({"verdict_pattern": r"Step (\d+) (\w+)"}, "at most one group"),
        ({"step_format": "Step: {step}"}, "slots"),
        ({"verdict_close": {"positive": "}"}}, "verdict_close"),
        ({"aggregation": "mean"}, "minimum"),
    ],
)
def test_step_config_refuses(change, match):
    with pytest.raises(MathModelError, match=match):
        step_config(**change)


def test_placeholder_prompts_are_refused(prompt_dir):
    (prompt_dir / "judge.txt").write_text("PLACEHOLDER: the authors write it {problem} {solution}")
    with pytest.raises(MathModelError, match="PLACEHOLDER"):
        G.GenerativeJudgeVerifier(judge_config(), prompt_dir=prompt_dir)


def test_build_verifier_routes_generative_implementations(prompt_dir, monkeypatch):
    monkeypatch.setattr(G, "PROMPT_DIR", prompt_dir)
    settings = {
        k: v
        for k, v in judge_config().logical_configuration().items()
        if k not in ("model_id", "revision")
    }
    spec = VerifierSpec(
        form="llm_judge",
        vendor="Lab A",
        base_model="Base A",
        model_id="test/judge",
        revision=REV,
        implementation=G.GEN_JUDGE_IMPLEMENTATION,
        settings=settings,
    )
    verifier = build_verifier(spec)  # nothing is loaded
    assert isinstance(verifier, G.GenerativeJudgeVerifier)
    with pytest.raises(MathModelError, match="relabelled"):
        replace(spec, form="prm")
    with pytest.raises(MathModelError, match="still TBD"):
        build_verifier(replace(spec, settings={**settings, "batch_size": "TBD"}))


# ---------------------------------------------------------------------------------------
# the decoding loop on a scripted fake model
# ---------------------------------------------------------------------------------------

WORDS = (
    "<pad> <eos> <user> <assistant> </think> Yes No correct incorrect } Problem: Solution: Is "
    "it right ? Final verdict : Check each step . Steps: Step is \\boxed{ think more x y z "
    "P1 P2 P3 S1 S2 S3 0 1 2 3 4 5 6 7 8 9"
).split()


class WordTokenizer:
    """Whitespace words; enough of the tokenizer interface for the adapter."""

    pad_token_id, eos_token_id = 0, 1
    all_special_tokens = ["<pad>", "<eos>"]
    chat_template = "<user> {content} <assistant>"

    def __init__(self) -> None:
        self.vocab = {w: i for i, w in enumerate(WORDS)}
        self.words = list(WORDS)

    def encode(self, text, add_special_tokens=False, truncation=False):
        return [self.vocab[w] for w in text.split()]

    def decode(self, ids, skip_special_tokens=False, clean_up_tokenization_spaces=False):
        return " ".join(self.words[i] for i in ids)

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True, **kw):
        return f"<user> {messages[0]['content']} <assistant>"


class ScriptedModel:
    """Emits, for each row, the continuation scripted for the prompt word it contains; at a
    verdict readout the logits of the two verdict tokens come from ``verdict``."""

    def __init__(self, torch, tok, scripts, verdict, words):
        self.torch, self.tok, self.scripts, self.verdict = torch, tok, scripts, verdict
        self.words, self.eos = words, 1
        self.config = SimpleNamespace(max_position_embeddings=10**6, _commit_hash=None)
        self.dtype = torch.float32
        self.calls = 0

    def eval(self):
        return self

    def get_input_embeddings(self):
        return SimpleNamespace(weight=SimpleNamespace(device="cpu"))

    def __call__(
        self,
        input_ids,
        attention_mask,
        position_ids,
        past_key_values,
        cache_position,
        use_cache,
        logits_to_keep,
    ):
        torch = self.torch
        ids = input_ids.tolist()
        if input_ids.shape[1] > 1:  # prefill: the real tokens of each row
            mask = attention_mask.tolist()
            self.hist = [
                [t for t, m in zip(r, mr, strict=False) if m]
                for r, mr in zip(ids, mask, strict=True)
            ]
            self.prompt_len = [len(h) for h in self.hist]
        else:
            for h, r in zip(self.hist, ids, strict=True):
                h.append(r[0])
        self.calls += 1
        logits = torch.full((len(ids), 1, len(WORDS)), -5.0)
        for i, h in enumerate(self.hist):
            key = next(w for w in self.scripts if self.tok.vocab[w] in h[: self.prompt_len[i]])
            script = [self.tok.vocab[w] for w in self.scripts[key].split()]
            done = len(h) - self.prompt_len[i]
            nxt = script[done] if done < len(script) else self.eos
            pos, neg = self.verdict(self.tok.decode(h))
            for word, value in zip(self.words, (pos, neg), strict=True):
                logits[i, 0, self.tok.vocab[word]] = value
            if self.tok.words[nxt] not in self.words:  # a scripted verdict comes from verdict()
                logits[i, 0, nxt] = 5.0
        return SimpleNamespace(logits=logits)


def fake(cls, config, prompt_dir, scripts, verdict):
    torch = pytest.importorskip("torch")
    tok = WordTokenizer()
    words = ("Yes", "No") if cls is G.GenerativeJudgeVerifier else ("correct", "incorrect")
    model = ScriptedModel(torch, tok, scripts, verdict, words)
    verifier = cls(config, prompt_dir=prompt_dir, tokenizer=tok, model=model, torch_module=torch)
    verifier._new_cache = lambda: None  # the fake model keeps its own history
    return verifier, tok


def test_judge_stop_eos_and_truncation(prompt_dir):
    scripts = {
        "P1": "think x </think> y",  # stops at the stop string
        "P2": "think x",  # end-of-sequence before the stop string
        "P3": "think more more more more more more more",  # reaches L_gen = 6
    }
    verifier, tok = fake(
        G.GenerativeJudgeVerifier,
        judge_config(),
        prompt_dir,
        scripts,
        lambda text: (0.0, 2.0) if text.endswith("verdict :") else (-9.0, -9.0),
    )
    results = verifier.score_batch([("P1", "S1"), ("P2", "S2"), ("P3", "S3")])
    one, two, three = results
    assert (one["stop_reason"], one["generated_text"]) == ("stop", "think x </think>")
    assert tok.decode(one["forced_token_ids"]) == VERDICT_PROMPT  # nothing else forced
    assert (two["stop_reason"], two["generated_text"]) == ("eos", "think x")  # eos not fed
    assert tok.decode(two["forced_token_ids"]) == "</think> " + VERDICT_PROMPT
    assert three["stop_reason"] == "length" and three["truncated"] is True
    assert three["generated_tokens"] == 6
    assert tok.decode(three["forced_token_ids"]) == "</think> " + VERDICT_PROMPT
    for r in results:
        assert len(r["readouts"]) == 1 and r["readouts"][0]["logits"] == [0.0, 2.0]
        assert r["error_score"] == pytest.approx(math.exp(2) / (1 + math.exp(2)))
        assert r["readouts"][0]["greedy"] == tok.vocab["<eos>"]  # what the fake emits there
        assert r["fixed_shape"]["cache_length"] == 24 + 6 + 8


def test_step_verdicts_found_in_the_generated_text(prompt_dir):
    scripts = {"P1": "Step 1 is \\boxed{ correct } Step 2 is \\boxed{ incorrect } </think>"}

    def verdict(text):
        return (3.0, 0.0) if text.endswith("Step 1 is \\boxed{") else (0.0, 1.0)

    verifier, _ = fake(G.GenerativeStepVerifier, step_config(), prompt_dir, scripts, verdict)
    r = verifier.score("P1", "S1\n\nS2\n\nS3")
    assert r["n_steps"] == 3 and r["n_steps_judged_generated"] == 2
    assert r["missing_steps"] == []  # step 3 follows the first incorrect step: not missing
    assert r["step_scores"][2] is None and r["step_score_sources"][:2] == ["generated"] * 2
    assert r["step_scores"][0] == pytest.approx(1 / (1 + 2.718281828459045**-3))
    assert r["error_score"] == pytest.approx(1 - r["min_step_score"])
    assert r["min_step_score"] == pytest.approx(r["step_scores"][1])
    assert r["forced_token_ids"] == []


@pytest.mark.parametrize("rule", G.MISSING_RULES)
def test_missing_verdict_rules(prompt_dir, rule):
    # judges step 1 only, then is cut at L_gen
    scripts = {"P1": "Step 1 is \\boxed{ correct } think more more more more more more"}

    def verdict(text):
        return (2.0, 0.0) if text.endswith("Step 1 is \\boxed{") else (1.0, 0.0)

    config = step_config(missing_verdicts=rule)
    verifier, tok = fake(G.GenerativeStepVerifier, config, prompt_dir, scripts, verdict)
    r = verifier.score("P1", "S1\n\nS2\n\nS3")
    assert r["truncated"] and r["missing_steps"] == [2, 3]
    if rule == "force_remaining":
        assert tok.decode(r["forced_token_ids"]) == (
            "Step 2 is \\boxed{ correct } Step 3 is \\boxed{ correct }"
        )
        assert r["step_score_sources"] == ["generated", "forced", "forced"]
    elif rule == "judged_only":
        assert r["forced_token_ids"] == [] and r["step_scores"][1:] == [None, None]
        assert r["error_score"] == pytest.approx(1 - r["step_scores"][0])
    else:
        assert r["step_scores"][1:] == [0.0, 0.0] and r["error_score"] == 1.0


def test_a_pattern_without_group_reads_the_verdicts_in_order(prompt_dir):
    """ThinkPRM's authors read the j-th boxed verdict as step j's (extract_step_labels)."""
    scripts = {"P1": "x \\boxed{ correct } y \\boxed{ x } z \\boxed{ incorrect } </think>"}

    def verdict(text):
        return (0.0, 2.0) if text.endswith("z \\boxed{") else (2.0, 0.0)

    config = step_config(verdict_pattern=r"\\boxed\{")
    verifier, _ = fake(G.GenerativeStepVerifier, config, prompt_dir, scripts, verdict)
    r = verifier.score("P1", "S1\n\nS2\n\nS3")
    # the box around "x" is not a verdict token: the two verdicts are steps 1 and 2
    assert r["n_steps_judged_generated"] == 2 and r["step_score_sources"][:2] == ["generated"] * 2
    assert r["step_scores"][1] < 0.5 < r["step_scores"][0]
    assert r["missing_steps"] == []  # step 3 follows the first incorrect step


def test_judged_only_forces_every_step_when_none_was_judged(prompt_dir):
    scripts = {"P1": "think more more more more more more more more more more more"}
    config = step_config(missing_verdicts="judged_only")
    verifier, _ = fake(G.GenerativeStepVerifier, config, prompt_dir, scripts, lambda t: (1.0, 0.0))
    r = verifier.score("P1", "S1\n\nS2")
    assert r["n_steps_judged_generated"] == 0
    assert r["step_score_sources"] == ["forced", "forced"]


def test_inputs_over_l_in_or_forced_budget_are_refused(prompt_dir):
    verifier, _ = fake(
        G.GenerativeJudgeVerifier,
        judge_config(max_input_tokens=8),
        prompt_dir,
        {"P1": "x"},
        lambda t: (0.0, 0.0),
    )
    with pytest.raises(G.ModelInputError, match="truncation is forbidden"):
        verifier.score("P1", "S1")
    verifier, _ = fake(
        G.GenerativeStepVerifier,
        step_config(max_forced_tokens=5),
        prompt_dir,
        {"P1": "x"},
        lambda t: (0.0, 0.0),
    )
    with pytest.raises(G.ModelInputError, match="forced suffix"):
        verifier.score("P1", "S1\n\nS2")


# ---------------------------------------------------------------------------------------
# the transformers path on a tiny random Llama
# ---------------------------------------------------------------------------------------


@pytest.fixture(scope="module")
def tiny_llama():
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    tokenizers = pytest.importorskip("tokenizers")
    words = ["<pad>", "<eos>", "<unk>", "<user>", "<assistant>", "</think>", "Yes", "No"]
    words += sorted(set((JUDGE_PROMPT + " " + VERDICT_PROMPT).split()) - set(words))
    words += [f"w{i}" for i in range(40)]
    vocab = {w: i for i, w in enumerate(words)}
    backend = tokenizers.Tokenizer(tokenizers.models.WordLevel(vocab, unk_token="<unk>"))
    backend.pre_tokenizer = tokenizers.pre_tokenizers.WhitespaceSplit()
    tok = transformers.PreTrainedTokenizerFast(
        tokenizer_object=backend, eos_token="<eos>", pad_token="<pad>", unk_token="<unk>"
    )
    tok.chat_template = (
        "{% for m in messages %}<user> {{ m['content'] }} {% endfor %}"
        "{% if add_generation_prompt %}<assistant> {% endif %}"
    )
    config = transformers.LlamaConfig(
        vocab_size=len(vocab),
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=256,
        pad_token_id=0,
        eos_token_id=1,
    )
    torch.manual_seed(20261011)
    model = transformers.LlamaForCausalLM(config).to(torch.float32).eval()
    return torch, tok, model


ITEMS = [("w1 w2 w3", "w4 w5"), ("w6", "w7 w8 w9 w10 w11 w12"), ("w13 w14", "w15")]


def _tiny(tiny_llama, prompt_dir, **changes):
    torch, tok, model = tiny_llama
    config = judge_config(eos_token_ids=[1], max_new_tokens=10, **changes)
    return G.GenerativeJudgeVerifier(
        config, prompt_dir=prompt_dir, tokenizer=tok, model=model, torch_module=torch
    )


def _fingerprint(r):
    return (
        r["generated_token_ids"],
        r["forced_token_ids"],
        [x["logits"] for x in r["readouts"]],
        r["error_score"],
    )


def test_tiny_llama_rows_do_not_depend_on_batch_or_padding(tiny_llama, prompt_dir):
    batch = _tiny(tiny_llama, prompt_dir).score_batch(ITEMS)
    again = _tiny(tiny_llama, prompt_dir).score_batch(ITEMS)
    alone = [_tiny(tiny_llama, prompt_dir).score(p, s) for p, s in ITEMS]
    assert [_fingerprint(r) for r in batch] == [_fingerprint(r) for r in again]  # bit for bit
    assert [_fingerprint(r) for r in batch] == [_fingerprint(r) for r in alone]
    # more left padding (a longer L_in) changes no token and no verdict logit beyond rounding
    wide = _tiny(tiny_llama, prompt_dir, max_input_tokens=60).score_batch(ITEMS)
    for a, b in zip(batch, wide, strict=True):
        assert a["generated_token_ids"] == b["generated_token_ids"]
        for x, y in zip(a["readouts"][0]["logits"], b["readouts"][0]["logits"], strict=True):
            assert abs(x - y) < 1e-5
    assert all(r["stop_reason"] in ("stop", "eos", "length") for r in batch)


def test_grouped_decode_attention_is_sdpa_without_the_copies(tiny_llama):
    torch, _, model = tiny_llama
    sdpa = pytest.importorskip("transformers.integrations.sdpa_attention")
    layer = model.model.layers[0].self_attn  # 4 query heads on 2 key / value heads
    gen = torch.Generator().manual_seed(3)
    batch, kv, n_rep, dim, length = 3, 2, 2, 8, 11
    key = torch.randn(batch, kv, length, dim, generator=gen)
    value = torch.randn(batch, kv, length, dim, generator=gen)
    mask = torch.ones(batch, 1, 1, length, dtype=torch.bool)
    mask[0, ..., :4] = False  # left padding
    mask[1, ..., -3:] = False  # cache positions not yet written
    query = torch.randn(batch, kv * n_rep, 1, dim, generator=gen)
    ours, _ = G.grouped_decode_attention(layer, query, key, value, mask, scaling=0.3)
    ref, _ = sdpa.sdpa_attention_forward(layer, query, key, value, mask, scaling=0.3)
    assert ours.shape == ref.shape == (batch, 1, kv * n_rep, dim)
    assert torch.allclose(ours, ref, atol=1e-6)
    # several query positions (the prefill) are transformers' sdpa itself
    query = torch.randn(batch, kv * n_rep, 4, dim, generator=gen)
    wide = mask.expand(batch, 1, 4, length)
    assert torch.equal(
        G.grouped_decode_attention(layer, query, key, value, wide)[0],
        sdpa.sdpa_attention_forward(layer, query, key, value, wide)[0],
    )


def test_tiny_llama_on_grouped_decode_attention(tiny_llama, prompt_dir):
    torch, tok, model = tiny_llama
    transformers = pytest.importorskip("transformers")
    G.register_attention()
    grouped = transformers.AutoModelForCausalLM.from_config(  # from_config edits its config
        copy.deepcopy(model.config), attn_implementation=G.GROUPED_DECODE_ATTENTION
    )
    grouped.load_state_dict(model.state_dict())
    grouped = grouped.to(torch.float32).eval()
    assert grouped.config._attn_implementation == G.GROUPED_DECODE_ATTENTION
    assert model.config._attn_implementation != G.GROUPED_DECODE_ATTENTION

    def make():
        config = judge_config(
            eos_token_ids=[1], max_new_tokens=10, attn_implementation=G.GROUPED_DECODE_ATTENTION
        )
        return G.GenerativeJudgeVerifier(
            config, prompt_dir=prompt_dir, tokenizer=tok, model=grouped, torch_module=torch
        )

    batch = make().score_batch(ITEMS)
    again = make().score_batch(ITEMS)
    alone = [make().score(p, s) for p, s in ITEMS]
    assert [_fingerprint(r) for r in batch] == [_fingerprint(r) for r in again]  # bit for bit
    assert [_fingerprint(r) for r in batch] == [_fingerprint(r) for r in alone]
    assert all(r["attn_implementation"] == G.GROUPED_DECODE_ATTENTION for r in batch)
    for ours, ref in zip(batch, _tiny(tiny_llama, prompt_dir).score_batch(ITEMS), strict=True):
        assert ours["generated_token_ids"] == ref["generated_token_ids"]
        for x, y in zip(ours["readouts"][0]["logits"], ref["readouts"][0]["logits"], strict=True):
            assert abs(x - y) < 1e-5


def test_tiny_llama_matches_hf_generate(tiny_llama, prompt_dir):
    torch, tok, model = tiny_llama
    verifier = _tiny(tiny_llama, prompt_dir)
    result = verifier.score(*ITEMS[1])
    ids = verifier._row(*ITEMS[1])[1]
    with torch.inference_mode():
        out = model.generate(
            input_ids=torch.tensor([ids]),
            attention_mask=torch.ones(1, len(ids), dtype=torch.long),
            do_sample=False,
            max_new_tokens=10,
            eos_token_id=1,
            pad_token_id=1,
        )
    hf = out[0, len(ids) :].tolist()
    hf = hf[: hf.index(1)] if 1 in hf else hf  # our loop does not feed the end of sequence
    stop = tok.convert_tokens_to_ids("</think>")
    hf = hf[: hf.index(stop) + 1] if stop in hf else hf  # nor generate past the stop string
    assert result["generated_token_ids"] == hf
