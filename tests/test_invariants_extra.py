"""Group 5: remaining section-4 invariants not covered elsewhere.

* network retries (the SDK knob, off in the frozen config since 2026-09-23, and the client's
  counted backoff) are separate from the 2 parse retries (II.8.2);
* provider parameter mapping (seed only on OpenAI Chat Completions; E has no seed);
* Exp 2 scores every active trajectory as ONE batch per round (II.8.3);
* every decision row on disk validates as a LogRecord with all II.10 fields;
* G_fixed inherits the controller backbone in the real configs, and their decoding blocks
  match except where the protocol licenses a difference (II.2.3);
* formal steps (step11/12 later) are blocked by the formatter template placeholder.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from rafr.channel.identity import IdentityChannel
from rafr.contracts.schemas import PARSE_MAX_RETRIES, Arm, EvalRecord, LogRecord, SafetyLabel, State
from rafr.controller import decide as ctrl
from rafr.core.config import NotFrozenError, assert_frozen, find_placeholders, load_config
from rafr.experiments import exp2
from rafr.formatter import TEMPLATES_PATH
from rafr.llm.client import LLMClient, LLMRequest
from rafr.llm.providers import anthropic as anthropic_provider
from rafr.llm.providers.anthropic import AnthropicProvider
from rafr.llm.providers.fake import FakeProvider
from rafr.llm.providers.openai import OpenAIProvider
from rafr.logging.writer import read_jsonl
from rafr.verifiers.stub import stub_pair


def _req(provider: str, seed: int | None = 7) -> LLMRequest:
    return LLMRequest(
        purpose="controller",
        provider=provider,
        model="m",
        version="v",
        messages=[{"role": "user", "content": "hi"}],
        system="sys",
        decoding={
            "temperature": 0.4,
            "top_p": None,
            "max_output_tokens": 55,
            "reasoning_effort": None,
        },
        seed=seed,
    )


def _e_req(**decoding: object) -> LLMRequest:
    """An E-shaped request: the evaluator's frozen decoding block (no sampling params)."""
    d: dict = {"temperature": None, "effort": "low", "max_output_tokens": 55}
    d.update(decoding)
    return LLMRequest(
        purpose="evaluator",
        provider="anthropic",
        model="m",
        version="v",
        messages=[{"role": "user", "content": "hi"}],
        decoding=d,
        seed=None,
    )


def test_network_retries_are_sdk_level_and_independent_of_parse_retries():
    o = OpenAIProvider(api_key="sk-test", max_retries=7, timeout_s=30)
    a = AnthropicProvider(api_key="sk-test", max_retries=9, timeout_s=30)
    assert o._client.max_retries == 7 and a._client.max_retries == 9
    assert PARSE_MAX_RETRIES == 2  # the parse budget never depends on the SDK setting


def test_openai_params_use_chat_completions_seed_and_decoding():
    o = OpenAIProvider(api_key="sk-test")
    p = o._params(_req("openai"))
    assert p == {"model": "m", "temperature": 0.4, "max_completion_tokens": 55, "seed": 7}
    assert "top_p" not in p and "reasoning_effort" not in p  # null = not sent


def test_openai_sends_a_configured_reasoning_effort():
    """2026-09-20: the controller's tier is sent explicitly, not left to the model's
    own default. A null stays unsent -- what changed is the config, not the mapping."""
    o = OpenAIProvider(api_key="sk-test")
    req = _req("openai")
    for level in ("low", "medium", "high"):  # all three verified against the API 2026-09-20
        req.decoding["reasoning_effort"] = level
        assert o._params(req)["reasoning_effort"] == level
    req.decoding["reasoning_effort"] = None
    assert "reasoning_effort" not in o._params(req)


def test_controller_reasoning_effort_is_explicit_and_moves_the_cache_key():
    """II.8.2 / II.10: a decoding parameter counts as recorded only if changing it moves the
    cache key -- otherwise a re-run at another tier could be served the old tier's output.
    Checked on the real config, not a fixture."""
    decoding = load_config()["models"]["controller"]["decoding"]
    assert decoding["reasoning_effort"] == "medium"  # explicit, not null
    cfg = _ccfg()
    cfg.decoding = dict(decoding)
    msgs = [{"role": "user", "content": "obs"}]
    req = ctrl.build_request(cfg, msgs, seed=1, attempt=0)
    assert req.decoding["reasoning_effort"] == "medium"
    other = ctrl.build_request(cfg, msgs, seed=1, attempt=0)
    other.decoding = {**req.decoding, "reasoning_effort": "low"}
    assert req.cache_key() != other.cache_key()


def test_anthropic_params_have_no_seed():
    a = AnthropicProvider(api_key="sk-test")
    p = a._params(_e_req())
    assert p == {"model": "m", "max_tokens": 55, "output_config": {"effort": "low"}}
    assert "seed" not in p
    # ...but the seed still lives in the cache key so a config change never hits stale cache
    assert _req("anthropic", 1).cache_key() != _req("anthropic", 2).cache_key()


def test_anthropic_refuses_sampling_params_instead_of_dropping_them():
    """Decision 14 / 2026-09-20: this model generation rejects temperature, top_p and
    top_k. Silently dropping a configured value would turn a frozen control variable into a
    provider default nobody chose, so the provider raises."""
    a = AnthropicProvider(api_key="sk-test")
    for name in ("temperature", "top_p", "top_k"):
        with pytest.raises(anthropic_provider.UnsupportedDecodingParamError) as ei:
            a._params(_e_req(**{name: 0.4}))
        assert name in str(ei.value)
    # temperature = 0 is a value too, not "unset"
    with pytest.raises(anthropic_provider.UnsupportedDecodingParamError):
        a._params(_e_req(temperature=0))


def test_anthropic_effort_is_optional_and_validated():
    a = AnthropicProvider(api_key="sk-test")
    assert "output_config" not in a._params(_e_req(effort=None))
    for level in anthropic_provider.EFFORT_LEVELS:
        assert a._params(_e_req(effort=level))["output_config"] == {"effort": level}
    with pytest.raises(anthropic_provider.UnsupportedDecodingParamError):
        a._params(_e_req(effort="lowest"))


class _BatchCountingScorer:
    v1_id, v2_id, v1_version, v2_version = "stub-v1", "stub-v2", "s", "s"

    def __init__(self) -> None:
        self.batches: list[int] = []
        self._v1, self._v2 = stub_pair()

    def score_pair(self, texts):
        self.batches.append(len(texts))
        return list(zip(self._v1.score(list(texts)), self._v2.score(list(texts)), strict=True))


def _eval(text: str) -> EvalRecord:
    return EvalRecord(
        text_id="h",
        text=text,
        label=SafetyLabel.SAFE,
        model="e",
        model_version=None,
        prompt_hash="p",
        e_prompt_sha256="h",
        e_version="v",
        decoding={},
    )


def _ccfg() -> ctrl.ControllerConfig:
    models = {
        "controller": {
            "provider": "fake",
            "model": "ctrl",
            "version": "v",
            "decoding": {"temperature": 0.2, "seed": 1},
            "seed_scheme": "fixed",
            "network": {},
        }
    }
    return ctrl.build_config(
        models,
        {},
        base_seed=1,
        system_prompt="S",
        rewrite_template="{x_t}|{observation}",
        retry_prompt="R",
    )


def test_exp2_scores_all_active_trajectories_as_one_batch_per_round(tmp_path: Path):
    states = [State(state_id=f"x{i}-g1", x0_id=f"x{i}", x0="o", text=f"t{i}") for i in range(5)]

    # accept the first two x0 at t=1, everything else keeps rewriting
    def responder(r):
        x_t = r.messages[0]["content"].split("|")[0]
        if x_t in ("t0", "t1"):
            return json.dumps({"action": "accept", "revised_text": ""})
        return json.dumps({"action": "rewrite", "revised_text": "r " + x_t})

    scorer = _BatchCountingScorer()
    client = LLMClient(providers={"fake": FakeProvider(responder=responder)})
    exp2.run_exp2(
        client,
        _ccfg(),
        exp2.Exp2Setup(
            run_id="t",
            run_dir=tmp_path / "e2",
            x1_states=states,
            scorer=scorer,
            tau1=0.5,
            tau2=0.5,
            channel=IdentityChannel(),
            arms=[Arm.A1, Arm.B],
            k=4,
            assignment_seed=1,
            run_seed=1,
            judge_fn=_eval,
        ),
    )
    assert scorer.batches == [5, 3, 3, 3]  # one batch per round, only non-accepted texts
    rows = [LogRecord.model_validate(r) for r in read_jsonl(tmp_path / "e2" / "decisions.jsonl")]
    assert len(rows) == 5 + 3 * 3
    assert all(r.c_t is None and r.r_ref is None for r in rows)  # post-hoc only


def test_real_configs_gfixed_inherits_controller_backbone():
    cfg = load_config("models")
    assert cfg["g_fixed"]["inherit_backbone_from"] == "controller"
    assert "model" not in cfg["g_fixed"] and "provider" not in cfg["g_fixed"]


#: Decoding keys G_fixed and the controller are allowed to differ on (2026-09-20).
#: `temperature` -- II.2.3 puts a `> 0` floor on G_fixed that no rule puts on the controller
#: (II.8.2 wanted the controller LOW for power), and the two were deliberately 1.0 vs 0.2
#: until 2026-09-20. `seed` -- never a yaml value: both build_config()s pop it and the caller
#: passes a per-component seed, so the per-call seeds necessarily differ. Nothing else in the
#: block has a rule licensing a difference.
DECODING_MAY_DIFFER: frozenset[str] = frozenset({"temperature", "seed"})


def test_gfixed_and_controller_decoding_match_except_where_the_protocol_allows():
    """II.2.3: G_fixed runs the controller's backbone and version, and that identity is what
    the transport argument rests on -- in Exp 2 the seam is inside one trajectory (x1 from
    G_fixed, x2.. from the controller). A decoding parameter that silently drifts apart makes
    "same backbone, same version" hollow; on a reasoning model `reasoning_effort` is exactly
    such a parameter. So: same keys, same values, except DECODING_MAY_DIFFER.

    A new decoding key added to one block and not the other fails here too -- that is the
    point. To let a key diverge, add it to DECODING_MAY_DIFFER with the rule that licenses it.
    """
    m = load_config("models")
    g, c = m["g_fixed"]["decoding"], m["controller"]["decoding"]
    assert set(g) == set(c), f"decoding keys differ: {set(g) ^ set(c)}"
    mismatched = {k: (g[k], c[k]) for k in set(g) - DECODING_MAY_DIFFER if g[k] != c[k]}
    assert not mismatched, f"G_fixed vs controller decoding differs on {mismatched}"
    # the tier in particular: null would have meant "not sent", i.e. the model's own default
    assert g["reasoning_effort"] == c["reasoning_effort"] == "medium"


def test_templates_placeholder_blocks_formal_steps(tmp_path):
    """A templates file still carrying the placeholder word blocks the formal steps; the real
    one no longer does (descriptions filled in 2026-09-25, D-2026-09-25e)."""
    src = TEMPLATES_PATH.read_text(encoding="utf-8")
    marked = tmp_path / "templates.py"
    marked.write_text(src.replace("Checker A is a text", "PLACE" + "HOLDER Checker A"), "utf-8")
    with pytest.raises(NotFrozenError) as ei:
        assert_frozen({}, extra_files=[marked])
    assert str(marked) in ei.value.placeholders
    assert find_placeholders([TEMPLATES_PATH]) == []  # nothing to refuse
