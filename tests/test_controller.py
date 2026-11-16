"""controller: structured output, the three parse-fail categories, 2 retries, accounting."""

from __future__ import annotations

import inspect
import json
import re

import pytest

from rafr.contracts.schemas import Action, Arm, Lambda, ParseFailReason
from rafr.controller import decide as D
from rafr.controller.parse import parse_attempt
from rafr.llm.client import LLMClient, LLMResponse
from rafr.llm.ledger import Usage
from rafr.llm.providers.fake import FakeProvider, FakeReply


def _resp(text, refused=False) -> LLMResponse:
    return LLMResponse(
        text=text,
        parsed=None,
        refused=refused,
        stop_reason="stop",
        model_version="m",
        usage=Usage(),
        prompt_hash="p",
        cache_key="k",
    )


def test_parse_categories():
    r = parse_attempt(_resp(None, refused=True), "x", [])
    assert r.fail_reason is ParseFailReason.REFUSAL
    r = parse_attempt(_resp('{"action": "skip", "revised_text": "a"}'), "x", [])
    assert r.fail_reason is ParseFailReason.BAD_ACTION
    r = parse_attempt(_resp('{"action": "rewrite", "revised_text": "   "}'), "x", [])
    assert r.fail_reason is ParseFailReason.EMPTY_REVISED_TEXT
    r = parse_attempt(_resp('{"action": "rewrite", "revised_text": '), "x", [])
    assert r.fail_reason is ParseFailReason.MALFORMED_JSON
    r = parse_attempt(_resp('{"action": "accept", "revised_text": ""}'), "x", [])
    assert r.ok and r.action is Action.ACCEPT and r.revised_text is None
    # rewrite with revised_text == x_t is a VALID empty rewrite
    r = parse_attempt(
        _resp('{"action": "rewrite", "revised_text": "  the   text "}'), "the text", []
    )
    assert r.ok and r.action is Action.REWRITE and r.empty_rewrite is True
    # refusal_like is a flag only
    r = parse_attempt(
        _resp('{"action": "rewrite", "revised_text": "I cannot help with that"}'),
        "x",
        ["i cannot help"],
    )
    assert r.ok and r.refusal_like is True and r.fail_reason is None


def _models(scheme="per_state"):
    return {
        "controller": {
            "provider": "fake",
            "api": None,
            "model": "ctrl",
            "version": "v",
            "decoding": {"temperature": 0.3, "max_output_tokens": 100, "seed": 5},
            "seed_scheme": scheme,
            "system_prompt_file": "unused",
            "rewrite_prompt_file": "unused",
            "parse_retry_prompt_file": "unused",
            "network": {},
        }
    }


def _cfg(scheme="per_state", show_x0=True) -> D.ControllerConfig:
    return D.build_config(
        _models(scheme),
        {"controller": {"show_x0": show_x0}},
        base_seed=11,
        system_prompt="SYS",
        rewrite_template="X0: {x0}\nXT: {x_t}\nOBS: {observation}",
        retry_prompt="Answer with the JSON object only.\n",
    )


def _labels(arm=Arm.A1, lam=Lambda.LOW, t=1, experiment="exp1") -> D.DecisionLabels:
    return D.DecisionLabels(state_id="s1", arm=arm, lam=lam, t=t, experiment=experiment)


def _client(responder):
    fp = FakeProvider(responder=responder)
    return LLMClient(providers={"fake": fp}), fp


def test_decide_receives_only_the_formatted_string():
    params = inspect.signature(D.decide).parameters
    assert params["observation"].annotation in ("str", str)
    assert "obs" not in params and "scores" not in params


def test_decide_uses_structured_output_and_records_prompt():
    client, fp = _client(lambda r: json.dumps({"action": "accept", "revised_text": ""}))
    d = D.decide(client, _cfg(), x0="ORIG", x_t="CUR", observation="OBS-TEXT", labels=_labels())
    assert d.action is Action.ACCEPT and not d.parse_fail and d.n_parse_retries == 0
    req = fp.calls[0]
    assert req.schema.__name__ == "ControllerOutput" and req.system == "SYS"
    assert req.messages == [{"role": "user", "content": "X0: ORIG\nXT: CUR\nOBS: OBS-TEXT"}]
    assert d.attempts[0].prompt_hash == req.prompt_hash()


def test_decision_rows_carry_the_configured_reasoning_effort():
    """2026-09-20: the controller's tier is sent explicitly, so every Decision row on
    disk must name it -- a null would only have recorded "not sent" (paper Sec. 4.6)."""
    from rafr.core.config import load_config

    assert load_config()["models"]["controller"]["decoding"]["reasoning_effort"] == "medium"
    client, fp = _client(lambda r: json.dumps({"action": "accept", "revised_text": ""}))
    cfg = _cfg()
    cfg.decoding = {**cfg.decoding, "reasoning_effort": "medium"}
    d = D.decide(client, cfg, x0="a", x_t="b", observation="o", labels=_labels())
    assert d.decoding["reasoning_effort"] == "medium"
    assert fp.calls[0].decoding["reasoning_effort"] == "medium"


FENCED = (
    "Task line.\n"
    "\n"
    "{x0_section_start}\n"
    "The original is shown for reference.\n"
    "\n"
    "Original:\n---\n{x0}\n---\n"
    "{x0_section_end}\n"
    "\n"
    "Current:\n---\n{x_t}\n---\n"
    "\n"
    "Checker:\n---\n{observation}\n---\n"
)


def _fenced_cfg(show_x0: bool) -> D.ControllerConfig:
    return D.build_config(
        _models(),
        {"controller": {"show_x0": show_x0}},
        base_seed=11,
        system_prompt="SYS",
        rewrite_template=FENCED,
        retry_prompt="Answer with the JSON object only.",
    )


def test_decide_hides_x0_when_configured():
    client, fp = _client(lambda r: json.dumps({"action": "accept", "revised_text": ""}))
    D.decide(client, _fenced_cfg(False), x0="ORIG", x_t="CUR", observation="O", labels=_labels())
    assert "ORIG" not in fp.calls[0].messages[0]["content"]


def test_show_x0_section_is_kept_whole_or_removed_whole_without_an_empty_block():
    """Review (2026-09-18): with show_x0 = false the explanatory sentence AND the Original
    block go together, and the paragraphs around them close up. With show_x0 = true only
    the two marker lines disappear."""
    shown = _fenced_cfg(True).rewrite_template
    hidden = _fenced_cfg(False).rewrite_template
    assert shown == FENCED.replace("{x0_section_start}\n", "").replace("{x0_section_end}\n", "")
    assert (
        hidden == "Task line.\n\nCurrent:\n---\n{x_t}\n---\n\nChecker:\n---\n{observation}\n---\n"
    )
    assert "reference" not in hidden and "Original" not in hidden and "{x0}" not in hidden
    assert "\n\n\n" not in hidden  # no empty block where the section used to be
    assert D.X0_SECTION_START not in shown + hidden and D.X0_SECTION_END not in shown + hidden
    # the real template: same properties, and the non-x0 text is byte-identical either way
    from rafr.core import paths
    from rafr.core.config import load_config

    tpl = paths.resolve(load_config("models")["controller"]["rewrite_prompt_file"]).read_text(
        encoding="utf-8"
    )
    real_shown = D.resolve_x0_section(tpl, show_x0=True)
    real_hidden = D.resolve_x0_section(tpl, show_x0=False)
    assert "Original sentence" in real_shown and "{x0}" in real_shown
    assert (
        "Original sentence" not in real_hidden and "original sentence is shown" not in real_hidden
    )
    assert "{x0}" not in real_hidden and "\n\n\n" not in real_hidden
    assert real_hidden.startswith(
        real_shown.split("\n\nThe original sentence")[0] + "\n\nCurrent text:"
    )


def test_show_x0_false_refuses_a_template_with_x0_outside_the_section():
    """Blanking {x0} silently would send the model a prompt with an empty slot; refuse."""
    with pytest.raises(D.ControllerConfigError, match="outside"):
        _cfg(show_x0=False)  # the plain test template has no fence
    for broken in (
        FENCED.replace("{x0_section_end}\n", ""),  # unbalanced
        FENCED.replace("{x0_section_start}", "{x0_section_end}"),  # two ends
        FENCED + "{x0_section_start}\n{x0_section_end}\n",  # two pairs
    ):
        with pytest.raises(D.ControllerConfigError, match="pair"):
            D.resolve_x0_section(broken, show_x0=True)


def test_slot_values_are_inserted_verbatim_even_when_they_contain_slot_names():
    """Review (2026-09-18): chained replace() let an x_t containing the literal string
    "{x0}" pull the original sentence into the current-text block. One-pass filling copies
    every value as-is."""
    cfg = _cfg()
    out = D.render_user_turn(cfg, "ORIG", "cur {x0} {observation} {x_t}", "obs {x_t}")
    assert out == "X0: ORIG\nXT: cur {x0} {observation} {x_t}\nOBS: obs {x_t}"
    out2 = D.render_user_turn(cfg, "o {x_t}", "c", "{x0}")
    assert out2 == "X0: o {x_t}\nXT: c\nOBS: {x0}"


def test_decide_retries_twice_with_fixed_prompt_then_parse_fail():
    client, fp = _client(lambda r: '{"action": "skip", "revised_text": ""}')
    d = D.decide(client, _cfg(), x0="a", x_t="b", observation="o", labels=_labels())
    assert d.parse_fail and d.action is None and d.n_parse_retries == 2
    assert len(d.attempts) == 3 and [a.attempt for a in d.attempts] == [0, 1, 2]
    assert all(a.parse_fail_reason is ParseFailReason.BAD_ACTION for a in d.attempts)
    assert all(a.raw_output for a in d.attempts)  # raw output of every attempt kept
    # retry = the ORIGINAL request with the fixed sentence appended; the previous raw output
    # is not carried over (design decision)
    first = fp.calls[0].messages
    assert first == [{"role": "user", "content": "X0: a\nXT: b\nOBS: o"}]
    for call in fp.calls[1:]:
        assert len(call.messages) == 1 and call.messages[0]["role"] == "user"
        assert (
            call.messages[0]["content"]
            == "X0: a\nXT: b\nOBS: o\n\nAnswer with the JSON object only."
        )
        assert "skip" not in call.messages[0]["content"]  # no previous output in the retry
    assert [c.attempt for c in fp.calls] == [0, 1, 2]
    # attempt index -> independent seed and cache key for every retry
    assert len({c.seed for c in fp.calls}) == 3 and len({c.cache_key() for c in fp.calls}) == 3
    assert [a.seed for a in d.attempts] == [c.seed for c in fp.calls]


def test_retry_sentence_is_identical_for_every_arm():
    client, fp = _client(lambda r: '{"action": "skip", "revised_text": ""}')
    for arm in (Arm.A1, Arm.B, Arm.PERMUTED_B):
        D.decide(client, _cfg(), x0="a", x_t="b", observation="o", labels=_labels(arm))
    tails = {c.messages[0]["content"].rsplit("\n\n", 1)[-1] for c in fp.calls if c.attempt > 0}
    assert tails == {"Answer with the JSON object only."}


def test_decide_recovers_on_retry():
    outs = iter(
        [
            FakeReply(text=None, refused=True, stop_reason="refusal"),
            json.dumps({"action": "rewrite", "revised_text": "better"}),
        ]
    )
    client, fp = _client(lambda r: next(outs))
    d = D.decide(client, _cfg(), x0="a", x_t="b", observation="o", labels=_labels())
    assert not d.parse_fail and d.action is Action.REWRITE and d.revised_text == "better"
    assert d.n_parse_retries == 1 and d.attempts[0].parse_fail_reason is ParseFailReason.REFUSAL


def test_seed_schemes():
    fixed, per_state, per_cell = _cfg("fixed"), _cfg("per_state"), _cfg("per_cell")
    a1, b = _labels(Arm.A1), _labels(Arm.B)
    assert D.call_seed(fixed, a1) == D.call_seed(fixed, b) == 11
    assert D.call_seed(per_state, a1) == D.call_seed(per_state, b) != 5
    assert D.call_seed(per_cell, a1) != D.call_seed(per_cell, b)
    # per_state: all lambdas of a state share the seed; different t does not
    assert D.call_seed(per_state, _labels(lam=Lambda.HIGH)) == D.call_seed(per_state, a1)
    assert D.call_seed(per_state, _labels(t=2)) != D.call_seed(per_state, a1)
    # retries: attempt 0 keeps the call seed, later attempts derive their own
    s0 = D.call_seed(per_state, a1)
    assert D.attempt_call_seed(s0, 0) == s0
    assert len({D.attempt_call_seed(s0, k) for k in range(3)}) == 3
    assert D.attempt_call_seed(None, 1) is None


def test_per_state_seed_differs_between_exp1_and_exp2():
    """Design decision: same state, same t, same prompt -> different seed in exp1 vs exp2,
    hence different cache keys, so the two experiments sample independently."""
    cfg = _cfg("per_state")
    l1, l2 = _labels(experiment="exp1"), _labels(experiment="exp2")
    assert D.call_seed(cfg, l1) != D.call_seed(cfg, l2)
    msgs = [{"role": "user", "content": D.render_user_turn(cfg, "a", "b", "o")}]
    r1 = D.build_request(cfg, msgs, D.call_seed(cfg, l1), 0)
    r2 = D.build_request(cfg, msgs, D.call_seed(cfg, l2), 0)
    assert r1.prompt_hash() == r2.prompt_hash() and r1.cache_key() != r2.cache_key()
    # per_cell inherits the same namespace separation
    cell = _cfg("per_cell")
    assert D.call_seed(cell, l1) != D.call_seed(cell, l2)
    with pytest.raises(D.ControllerConfigError):
        D.build_config(
            _models("TBD"),
            {},
            base_seed=11,
            system_prompt="s",
            rewrite_template="r",
            retry_prompt="p",
        )
    with pytest.raises(D.ControllerConfigError):  # the caller must supply the component seed
        D.build_config(_models(), {}, system_prompt="s", rewrite_template="r", retry_prompt="p")
    D.build_config(
        _models("TBD"),
        {},
        allow_tbd=True,
        system_prompt="s",
        rewrite_template="r",
        retry_prompt="p",
    )


# ---------------------------------------------------------------- the authors' prompt files


def _paragraphs(text: str) -> list[str]:
    return [p.strip() for p in text.strip().split("\n\n")]


def test_real_system_prompt_k_is_filled_by_the_caller_per_experiment():
    """(2026-09-18): ``{k}`` is the decision budget the controller is told about and it
    differs by experiment (Exp 1: 1, Exp 2: K_MAX_ROUNDS). build_config keeps the slot; the
    runner fills it with with_round_budget; decide() refuses an unfilled slot."""
    from rafr.contracts.schemas import K_MAX_ROUNDS
    from rafr.core.config import load_config

    m = _models()
    m["controller"]["system_prompt_file"] = load_config("models")["controller"][
        "system_prompt_file"
    ]
    cfg = D.build_config(m, {}, base_seed=1, rewrite_template="r", retry_prompt="p")
    assert D.SYSTEM_K_SLOT in cfg.system_prompt and cfg.round_budget is None
    one = D.with_round_budget(cfg, 1)
    four = D.with_round_budget(cfg, K_MAX_ROUNDS)
    # the singular with K = 1, the plural otherwise; nothing else changes
    assert "You have at most 1 decision in total." in one.system_prompt and one.round_budget == 1
    assert "1 decisions" not in one.system_prompt
    assert f"You have at most {K_MAX_ROUNDS} decisions in total." in four.system_prompt
    assert one.system_prompt.replace("1 decision in", "{k} decisions in") == cfg.system_prompt
    assert four.system_prompt.replace(f"{K_MAX_ROUNDS} decisions", "{k} decisions") == (
        cfg.system_prompt
    )
    assert "{k} decisions in total" in cfg.system_prompt  # the prompt file is unchanged
    assert D.SYSTEM_K_SLOT not in one.system_prompt + four.system_prompt
    assert D.SYSTEM_K_SLOT in cfg.system_prompt  # pure: the input config is untouched
    assert one.system_prompt != four.system_prompt
    with pytest.raises(D.ControllerConfigError, match="round budget"):
        D.with_round_budget(cfg, 0)
    client, fp = _client(lambda r: json.dumps({"action": "accept", "revised_text": ""}))
    with pytest.raises(D.ControllerConfigError, match="with_round_budget"):
        D.decide(client, cfg, x0="a", x_t="b", observation="o", labels=_labels())
    assert fp.calls == []


def test_real_rewrite_template_fills_every_slot_and_leaves_no_brace_slot():
    from rafr.core import paths
    from rafr.core.config import load_config

    c = load_config("models")["controller"]
    tpl = paths.resolve(c["rewrite_prompt_file"]).read_text(encoding="utf-8")
    retry = paths.resolve(c["parse_retry_prompt_file"]).read_text(encoding="utf-8")
    assert all(slot in tpl for slot in ("{x0}", "{x_t}", "{observation}"))
    assert len(retry.strip().splitlines()) == 1  # one fixed sentence, same for every arm
    m = _models()
    cfg = D.build_config(
        m, {}, base_seed=1, system_prompt="S", rewrite_template=tpl, retry_prompt=retry
    )
    out = D.render_user_turn(cfg, "ORIG", "CURRENT", "OBS", attempt=1)
    assert "ORIG" in out and "CURRENT" in out and "OBS" in out
    assert not re.search(r"\{(x0|x_t|observation|k|x0_section_start|x0_section_end)\}", out)
    assert out.endswith(retry.strip())


_CATEGORY_LIST = re.compile(r"profanity.*?hostile ridicule", re.IGNORECASE | re.DOTALL)


def _toxicity_lists(text: str) -> list[str]:
    """Every toxicity category list in ``text`` (from \"profanity\" to \"hostile ridicule\"),
    whitespace-normalized so that line wrapping does not count as a difference."""
    return [" ".join(m.group(0).split()) for m in _CATEGORY_LIST.finditer(text)]


def _controller_task_side() -> tuple[list[str], list[str], str]:
    """(system task-segment paragraphs, rewrite paragraphs, rewrite text with the x0 fence
    resolved as shown). The task segment of system.txt is what describes the job and what
    toxicity means -- not the options, the checker caveat, the budget or the JSON format."""
    from rafr.core import paths
    from rafr.core.config import load_config

    c = load_config("models")["controller"]
    system = _paragraphs(paths.resolve(c["system_prompt_file"]).read_text(encoding="utf-8"))
    task = [p for p in system if p.startswith("Toxicity means:") or "is being revised" in p]
    assert len(task) == 2, system
    tpl = paths.resolve(c["rewrite_prompt_file"]).read_text(encoding="utf-8")
    shown = D.resolve_x0_section(tpl, show_x0=True)
    return task, _paragraphs(shown), shown


def test_g_fixed_and_rewrite_prompts_carry_the_same_detox_instruction():
    """paper Sec. 4.2 (the authors' ruling, 2026-09-18): G_fixed is compared against the whole of
    what the controller is told about the task -- the task segment of system.txt PLUS the
    rewrite prompt -- minus any verifier feedback.

    * rewrite paragraph 1 == g_fixed paragraph 1 up to the referent (\"the sentence below\"
      vs \"the current text\", the authors' wording);
    * paragraph 2 (preservation constraints) and paragraph 3 (the output-length sentence,
      added to rewrite.txt on 2026-09-18 so that G_fixed and the controller share the
      same output shape) are identical verbatim;
    * the toxicity category list is identical verbatim between g_fixed.txt and rewrite.txt
      (the system.txt copy is checked by the next test);
    * g_fixed.txt has no x_t / observation slot and no score / verifier / checker /
      threshold vocabulary."""
    from rafr.core import paths
    from rafr.core.config import load_config

    models = load_config("models")
    g_text = paths.resolve(models["g_fixed"]["prompt_file"]).read_text(encoding="utf-8")
    g = _paragraphs(g_text)
    task, r, r_text = _controller_task_side()
    assert g[1] == r[1]
    assert g[2] == r[2]
    assert g[2].startswith("Return the rewritten sentence only.")
    assert g[0] == r[0].replace("Rewrite the current text", "Rewrite the sentence below", 1)
    assert g[0] != r[0]  # the one documented difference is real, not a no-op
    g_lists, r_lists = _toxicity_lists(g_text), _toxicity_lists(r_text)
    assert len(g_lists) == 1 and r_lists == g_lists
    assert "{x0}" in g_text
    assert "{x_t}" not in g_text and "{observation}" not in g_text
    for word in ("score", "verifier", "checker", "threshold"):
        assert word not in g_text.lower()
    # the task description (system.txt) no longer enumerates what the controller sees
    # (2026-09-18): no "those scores, the original sentence, and the current text"
    assert (
        "those scores" not in task[0] and "original sentence, and the current text" not in task[0]
    )


def test_system_prompt_toxicity_list_matches_g_fixed_verbatim():
    """(2026-09-18, option b): system.txt uses the same short list as g_fixed.txt /
    rewrite.txt; the "(whether or not it is aimed at anyone)" qualifier was withdrawn."""
    from rafr.core import paths
    from rafr.core.config import load_config

    models = load_config("models")
    g_text = paths.resolve(models["g_fixed"]["prompt_file"]).read_text(encoding="utf-8")
    task, _, _ = _controller_task_side()
    sys_lists = _toxicity_lists(task[1])
    assert len(sys_lists) == 1
    assert sys_lists == _toxicity_lists(g_text)
