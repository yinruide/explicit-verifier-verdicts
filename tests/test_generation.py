"""generation: validity rules (paper Sec. 4.2) and G_fixed invariants."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from rafr.contracts.schemas import InvalidReason
from rafr.generation import g_fixed
from rafr.generation import validity as V
from rafr.llm.client import LLMClient
from rafr.llm.providers.fake import FakeProvider, FakeReply
from rafr.logging.writer import JsonlWriter, read_jsonl

# ---------------------------------------------------------------- validity


def test_equality_rule_is_whitespace_normalized_and_case_sensitive():
    assert V.texts_equal("  a   b\tc ", "a b c")
    assert not V.texts_equal("a b", "A b")
    assert V.normalize_ws("\n x \n\n y ") == "x y"


@pytest.mark.parametrize(
    "text,ratio",
    [
        ("hello world", 0.0),
        ("héllo wörld, señor!", 0.0),  # accented Latin counts as Latin
        ("привет мир", 1.0),
        ("hello мир", 0.5),  # 5 Latin, 3 Cyrillic -> 3/8? no: "hello"=5, "мир"=3 -> 0.375
        ("123 !!! 😀", 0.0),  # no letters -> ratio 0, letters 0
    ],
)
def test_non_latin_ratio(text, ratio):
    got, n = V.non_latin_ratio(text)
    if text == "hello мир":
        assert n == 8 and got == pytest.approx(3 / 8)
    else:
        assert got == pytest.approx(ratio)


def test_judge_validity_order_and_reasons():
    phrases = ["i can't help with", "sorry, i can't"]  # prefixes (2026-09-18)
    j = V.judge_validity
    assert (
        j("x", "x0", provider_refused=True, phrases=phrases, non_english_threshold=0.5)
        is InvalidReason.REFUSAL
    )
    assert (
        j(None, "x0", provider_refused=False, phrases=phrases, non_english_threshold=0.5)
        is InvalidReason.EMPTY
    )
    assert (
        j("   ", "x0", provider_refused=False, phrases=phrases, non_english_threshold=0.5)
        is InvalidReason.EMPTY
    )
    assert (
        j(
            "Sorry, I can't help with that.",
            "x0",
            provider_refused=False,
            phrases=phrases,
            non_english_threshold=0.5,
        )
        is InvalidReason.REFUSAL
    )
    assert (
        j(
            " you  are wrong ",
            "you are wrong",
            provider_refused=False,
            phrases=phrases,
            non_english_threshold=0.5,
        )
        is InvalidReason.IDENTICAL_TO_X0
    )
    assert (
        j("это плохо", "x0", provider_refused=False, phrases=phrases, non_english_threshold=0.5)
        is InvalidReason.NON_ENGLISH
    )
    assert (
        j("!!! 123", "x0", provider_refused=False, phrases=phrases, non_english_threshold=0.5)
        is InvalidReason.EMPTY
    )
    assert (
        j(
            "you are mistaken",
            "you are wrong",
            provider_refused=False,
            phrases=phrases,
            non_english_threshold=0.5,
        )
        is None
    )
    # threshold None disables the language rule (dry run only)
    assert (
        j("это плохо", "x0", provider_refused=False, phrases=phrases, non_english_threshold=None)
        is None
    )


def test_refusal_prefixes_loaded_normalized(tmp_path: Path):
    p = tmp_path / "r.yaml"
    p.write_text("prefixes:\n  - '  I  CANNOT help '\n  - ''\n", encoding="utf-8")
    ph = V.load_refusal_phrases(p)
    assert ph == ["i cannot help"]
    assert V.matches_refusal("i cannot   help with this", ph)
    assert not V.matches_refusal("here is a polite rewrite", ph)


def test_refusal_match_is_prefix_not_substring():
    """(2026-09-18): the list matches the START of the whole output only. A substring
    rule would flag ordinary rewrites that merely contain one of the phrases mid-sentence."""
    ph = ["i can't help", "i cannot help", "as an ai"]
    assert V.matches_refusal("I can't help with that request.", ph)
    assert not V.matches_refusal("Honestly, I can't help but think this policy is wrong.", ph)
    assert not V.matches_refusal("Treating people as an AI would is not the point here.", ph)
    # leading / trailing whitespace and quotation marks are ignored, case is ignored
    assert V.matches_refusal('  "I CANNOT help with this."  ', ph)
    assert V.matches_refusal("\u201cI cannot help.\u201d", ph)
    assert V.matches_refusal("'i can't help'", ph)
    assert not V.matches_refusal("", ph)


def test_refusal_file_with_legacy_phrases_key_is_refused(tmp_path: Path):
    """The old ``phrases:`` key meant substring matching; reading it as prefixes would silently
    change the semantics, so it is an error."""
    p = tmp_path / "r.yaml"
    p.write_text("phrases:\n  - 'i cannot help'\n", encoding="utf-8")
    with pytest.raises(ValueError, match="prefixes"):
        V.load_refusal_phrases(p)


def test_repo_refusal_list_is_prefixes_and_narrow():
    from rafr.core import paths

    ph = V.load_refusal_phrases(paths.resolve("configs/refusal_phrases.yaml"))
    assert ph and all(p == p.lower() == V.normalize_ws(p) for p in ph)
    # detoxified sentences that keep their stance must not look like refusals (narrowed
    # 2026-09-18: bare "i won't" / "i will not" / "unfortunately, i" were dropped for this)
    assert not V.matches_refusal("This policy is wrong and I cannot see why anyone backs it.", ph)
    assert not V.matches_refusal("I won't apologize for saying this.", ph)
    assert not V.matches_refusal("I will not pretend this decision was reasonable.", ph)
    assert not V.matches_refusal("Unfortunately, I think this plan is a mistake.", ph)
    assert V.matches_refusal("I won't rewrite this.", ph)
    assert V.matches_refusal("I will not help with that request.", ph)


# Review of 2026-09-18 (the authors' ruling): five prefixes dropped because they open ordinary
# stance-keeping rewrites; ten added for refusals the list was missing. "i can not" was
# proposed and then dropped the same day (it opens "I can not stress enough ..."). One case
# per example below.
_REVIEW_FALSE_POSITIVES = [
    "I can't help thinking this plan is a mistake.",  # was "i can't help"
    "I'm not able to see why anyone would defend this policy.",  # was "i'm not able to"
    "I am unable to take this argument seriously.",  # was "i am unable to"
    "I can not stress enough how wrong this is.",  # "i can not" was never added
]
_REVIEW_MISSES = [
    "I can't rewrite this without changing what it says.",  # "i can't rewrite"
    "I cannot fulfill this request.",  # "i cannot fulfill"
    "I'm sorry, but I can't produce that text.",  # "i'm sorry, but"
    "I apologize, but this is not something I will write.",  # "i apologize"
    "I'm not comfortable rewriting this sentence.",  # "i'm not comfortable"
    "I cannot do that.",  # "i cannot do that"
]


@pytest.mark.parametrize("text", _REVIEW_FALSE_POSITIVES)
def test_review_false_positive_is_not_a_refusal(text: str):
    from rafr.core import paths

    ph = V.load_refusal_phrases(paths.resolve("configs/refusal_phrases.yaml"))
    assert not V.matches_refusal(text, ph), text


@pytest.mark.parametrize("text", _REVIEW_MISSES)
def test_review_miss_is_now_a_refusal(text: str):
    from rafr.core import paths

    ph = V.load_refusal_phrases(paths.resolve("configs/refusal_phrases.yaml"))
    assert V.matches_refusal(text, ph), text


def test_refusal_head_folds_curly_apostrophes_and_strips_leading_markdown(tmp_path: Path):
    """Review (2026-09-18): a refusal typed with U+2019 or wrapped in markdown was not
    matched by the ASCII prefixes. Only the HEAD is cleaned; markdown later in the text and
    the rest of the content are untouched."""
    ph = ["i'm sorry, but", "i can't rewrite", "i cannot help"]
    assert V.refusal_head("“I’m sorry, but”") == "i'm sorry, but"
    assert V.matches_refusal("I’m sorry, but I can’t rewrite that.", ph)
    assert V.matches_refusal("**I can't rewrite** this text.", ph)
    assert V.matches_refusal("> I cannot help with this.", ph)
    assert V.matches_refusal("- *I cannot help.*", ph)
    assert V.matches_refusal("### I'm sorry, but no.", ph)
    assert V.matches_refusal("`I cannot help`", ph)
    # the list itself may be written with curly apostrophes and still match ASCII output
    p = tmp_path / "r.yaml"
    p.write_text('prefixes:\n  - "i’m sorry, but"\n', encoding="utf-8")
    assert V.load_refusal_phrases(p) == ["i'm sorry, but"]
    assert V.matches_refusal("I'm sorry, but no.", V.load_refusal_phrases(p))
    # a marker inside the text is content, not markup: the head is still the first word
    assert V.refusal_head("This *is* fine - really.") == "this *is* fine - really."
    assert not V.matches_refusal("This *is* fine - really.", ph)
    assert V.refusal_head("") == "" and V.refusal_head("***") == ""


# ---------------------------------------------------------------- G_fixed config


def _models(temp=0.7, seed=11):
    return {
        "controller": {
            "provider": "fake",
            "api": None,
            "model": "ctrl-model",
            "version": "v1",
            "network": {},
        },
        "g_fixed": {
            "inherit_backbone_from": "controller",
            "decoding": {
                "temperature": temp,
                "top_p": None,
                "max_output_tokens": 200,
                "seed": seed,
            },
            "prompt_file": "unused",
        },
    }


def _gen(max_attempts=3, thr=0.5):
    return {
        "g_fixed": {"max_attempts": max_attempts, "validity": {"non_english": {"threshold": thr}}}
    }


def test_gfixed_inherits_controller_backbone_and_requires_temperature_gt_0():
    cfg = g_fixed.build_config(_models(temp=0.7), _gen(), prompt_text="Rewrite: {x0}", base_seed=11)
    assert cfg.model == "ctrl-model" and cfg.version == "v1" and cfg.provider == "fake"
    for bad in (0, 0.0, -1):
        with pytest.raises(g_fixed.GFixedConfigError):
            g_fixed.build_config(
                _models(temp=bad), _gen(), prompt_text="Rewrite: {x0}", base_seed=11
            )
    with pytest.raises(g_fixed.GFixedConfigError):
        g_fixed.build_config(_models(temp="TBD"), _gen(), prompt_text="Rewrite: {x0}", base_seed=11)
    # dry run may pass TBD through
    g_fixed.build_config(
        _models(temp="TBD", seed="TBD"),
        _gen(max_attempts="TBD", thr="TBD"),
        allow_tbd=True,
        prompt_text="{x0}",
        base_seed=None,
    )


def test_gfixed_accepts_an_unsent_temperature_and_records_it():
    """2026-09-20: the backbone rejects temperature, so the frozen config sets null.
    The design's `> 0` is then met by the provider default (1) -- accepted, but never silently:
    the key stays in the decoding block (so every logged attempt carries it) and the run
    snapshot gets an explicit record saying the parameter was not sent."""
    cfg = g_fixed.build_config(_models(temp=None), _gen(), prompt_text="{x0}", base_seed=11)
    assert "temperature" in cfg.decoding and cfg.decoding["temperature"] is None
    rec = g_fixed.temperature_record(cfg)
    assert rec["sent"] is False and rec["value"] is None
    assert "provider default" in rec["satisfied_by"] and "II.2.3" in rec["rule"]
    # a configured value is still reported as the thing that satisfies the rule
    sent = g_fixed.temperature_record(
        g_fixed.build_config(_models(temp=0.7), _gen(), prompt_text="{x0}", base_seed=11)
    )
    assert sent["sent"] is True and sent["value"] == 0.7 and "0.7" in sent["satisfied_by"]


def test_gfixed_attempts_log_the_unsent_temperature():
    """The design records every decoding parameter; `null` on disk is the record of "not sent"."""
    client, _ = _client(lambda r: json.dumps({"revised_text": "please stop that"}))
    cfg = g_fixed.build_config(_models(temp=None), _gen(), prompt_text="{x0}", base_seed=11)
    _, attempts = g_fixed.generate_state(client, cfg, "x0-1", "you idiot")
    assert attempts and all(a.decoding["temperature"] is None for a in attempts)


def test_gfixed_request_carries_no_feedback():
    cfg = g_fixed.build_config(
        _models(), _gen(), prompt_text="Rewrite politely: {x0}", base_seed=11
    )
    req = g_fixed.build_request(cfg, "x0-1", "you idiot", attempt=0)
    assert req.messages == [{"role": "user", "content": "Rewrite politely: you idiot"}]
    assert req.system is None
    assert req.schema.__name__ == "GFixedOutput"
    # the request is a pure function of (config, x0, attempt): no scores/observation can enter
    assert req.cache_key() == g_fixed.build_request(cfg, "x0-1", "you idiot", attempt=0).cache_key()
    assert req.cache_key() != g_fixed.build_request(cfg, "x0-1", "you idiot", attempt=1).cache_key()


def _client(responder) -> tuple[LLMClient, FakeProvider]:
    fp = FakeProvider(responder=responder)
    return LLMClient(providers={"fake": fp}), fp


def test_gfixed_regenerates_on_invalid_then_succeeds(tmp_path: Path):
    outs = iter(
        [
            FakeReply(text=None, refused=True, stop_reason="refusal"),  # refusal
            json.dumps({"revised_text": "you idiot"}),  # identical to x0
            json.dumps({"revised_text": "you are mistaken"}),  # valid
        ]
    )
    client, fp = _client(lambda r: next(outs))
    cfg = g_fixed.build_config(_models(), _gen(max_attempts=5), prompt_text="{x0}", base_seed=11)
    with JsonlWriter(tmp_path / "gen.jsonl") as log:
        state, attempts = g_fixed.generate_state(client, cfg, "x0-1", "you idiot", log=log)
    assert state is not None and state.text == "you are mistaken" and state.gen_attempt == 2
    assert state.state_id == "x0-1-g1" and state.t == 1 and state.source == "g_fixed"
    assert [a.invalid_reason for a in attempts] == [
        InvalidReason.REFUSAL,
        InvalidReason.IDENTICAL_TO_X0,
        None,
    ]
    # every attempt logged; seeds differ per attempt ("next seed")
    rows = list(read_jsonl(tmp_path / "gen.jsonl"))
    assert len(rows) == 3 and len({r["seed"] for r in rows}) == 3
    assert [r["valid"] for r in rows] == [False, False, True]
    assert len(fp.calls) == 3


def test_gfixed_drops_x0_after_max_attempts(tmp_path: Path):
    client, fp = _client(lambda r: json.dumps({"revised_text": ""}))
    cfg = g_fixed.build_config(_models(), _gen(max_attempts=3), prompt_text="{x0}", base_seed=11)
    res = g_fixed.generate_pool(client, cfg, [("x0-1", "a"), ("x0-2", "b")])
    assert res.states == [] and len(res.dropped) == 2 and res.n_attempts == 6
    assert res.dropped[0]["reasons"] == ["empty"] * 3
    assert len(fp.calls) == 6


def test_gfixed_one_state_per_x0_and_resume():
    client, fp = _client(lambda r: json.dumps({"revised_text": "fine text"}))
    cfg = g_fixed.build_config(_models(), _gen(), prompt_text="{x0}", base_seed=11)
    res = g_fixed.generate_pool(client, cfg, [("x0-1", "a"), ("x0-2", "b")])
    assert [s.x0_id for s in res.states] == ["x0-1", "x0-2"]
    assert len(fp.calls) == 2
    existing = {s.x0_id: s for s in res.states}
    res2 = g_fixed.generate_pool(client, cfg, [("x0-1", "a"), ("x0-3", "c")], existing=existing)
    assert [s.x0_id for s in res2.states] == ["x0-1", "x0-3"]
    assert len(fp.calls) == 3  # x0-1 reused, only x0-3 generated


def test_g_fixed_prompt_copies_an_x0_containing_a_slot_name_verbatim():
    """Review (2026-09-18): one-pass slot filling -- an x0 that contains \"{x0}\" is not
    expanded twice."""
    cfg = g_fixed.GFixedConfig(
        provider="fake", api=None, model="m", version=None, decoding={"temperature": 1.0},
        base_seed=1, prompt_template="S: {x0} END", max_attempts=1,
    )  # fmt: skip
    req = g_fixed.build_request(cfg, "x", "weird {x0} {x0} text", 0)
    assert req.messages[0]["content"] == "S: weird {x0} {x0} text END"
