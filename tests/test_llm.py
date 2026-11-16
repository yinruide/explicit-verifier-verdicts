"""llm: cache key composition, cache hits, ledger, fake provider, strict schema."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest

from rafr.contracts.schemas import ControllerOutput, EvaluatorOutput
from rafr.llm.cache import LLMCache
from rafr.llm.client import LLMClient, LLMRequest
from rafr.llm.ledger import Ledger, Usage
from rafr.llm.providers import MissingAPIKeyError, load_api_key, strict_json_schema
from rafr.llm.providers.fake import FakeProvider, FakeReply


def _req(**kw) -> LLMRequest:
    base = dict(
        purpose="controller",
        provider="fake",
        model="m",
        version="v1",
        messages=[{"role": "user", "content": "hi"}],
        system="sys",
        schema=ControllerOutput,
        decoding={"temperature": 0.7, "max_output_tokens": 100},
        seed=42,
        attempt=0,
    )
    base.update(kw)
    return LLMRequest(**base)


@pytest.mark.parametrize(
    "change",
    [
        {"model": "m2"},
        {"version": "v2"},
        {"decoding": {"temperature": 0.8, "max_output_tokens": 100}},
        {"decoding": {"temperature": 0.7, "max_output_tokens": 101}},
        {"seed": 43},
        {"system": "sys2"},
        {"messages": [{"role": "user", "content": "hello"}]},
        {"schema": EvaluatorOutput},
        {"attempt": 1},
    ],
)
def test_cache_key_changes_with_each_ii10_component(change):
    assert _req().cache_key() != replace(_req(), **change).cache_key()


def test_cache_key_ignores_purpose_and_network():
    other = replace(_req(), purpose="x", network={"max_retries": 1})
    assert _req().cache_key() == other.cache_key()


def test_prompt_hash_covers_system_messages_schema():
    h = _req().prompt_hash()
    assert h != replace(_req(), system="other").prompt_hash()
    assert h != replace(_req(), schema=None).prompt_hash()
    assert h == _req().prompt_hash()


def test_client_caches_and_records(llm: LLMClient, fake_provider: FakeProvider):
    r1 = llm.complete(_req())
    r2 = llm.complete(_req())
    assert r1.cached is False and r2.cached is True
    assert r1.parsed == r2.parsed == {"action": "rewrite", "revised_text": "[fake rewrite] hi"}
    assert len(fake_provider.calls) == 1
    rows = llm._ledger.rows()
    assert [r["cached"] for r in rows] == [False, True]
    assert rows[0]["input_tokens"] == 10 and rows[0]["output_tokens"] == 5
    s = llm._ledger.summary()["controller"]
    assert s["calls"] == 2 and s["cached_calls"] == 1


def test_ledger_reports_tokens_and_carries_no_prices(tmp_path: Path):
    """Decision (2026-09-17): the ledger records what a run produces — calls, cache
    hits and tokens. Prices were four more frozen-config values for a multiplication."""
    led = Ledger(tmp_path / "l.jsonl")
    led.record(
        purpose="evaluator",
        provider="fake",
        model="m",
        model_version=None,
        usage=Usage(3, 4),
        cached=False,
    )
    s = led.summary()["evaluator"]
    assert s == {
        "calls": 1,
        "cached_calls": 0,
        "input_tokens": 3,
        "output_tokens": 4,
        "cache_creation_input_tokens": 0,
        "cache_read_input_tokens": 0,
    }
    assert "cost_usd" not in led.rows()[0]
    with pytest.raises(TypeError):  # the price arguments are gone for good
        led.record(
            purpose="evaluator",
            provider="fake",
            model="m",
            model_version=None,
            usage=Usage(1, 1),
            cached=False,
            price_in=1.0,
        )


def test_fake_provider_refusal_and_invalid_json():
    p = FakeProvider(responder=lambda r: FakeReply(text=None, refused=True, stop_reason="refusal"))
    r = p.complete(_req())
    assert r.refused and r.parsed is None
    p2 = FakeProvider(responder=lambda r: '{"action": "skip", "revised_text": ""}')
    r2 = p2.complete(_req())
    assert r2.parsed is None and r2.error and "schema_validation" in r2.error
    assert r2.text == '{"action": "skip", "revised_text": ""}'  # raw always kept


def test_cache_roundtrip(tmp_path: Path):
    c = LLMCache(tmp_path)
    assert c.get("ab" * 32) is None
    payload = {
        "text": "x",
        "parsed": None,
        "refused": False,
        "stop_reason": "stop",
        "model_version": "m",
        "usage": {"input_tokens": 1, "output_tokens": 2},
        "prompt_hash": "p",
        "cache_key": "ab" * 32,
    }
    c.put("ab" * 32, payload, meta={"purpose": "t"})
    got = c.get("ab" * 32)
    assert got["text"] == "x" and "_meta" not in got
    assert ("ab" * 32) in c and len(c) == 1


def test_strict_schema_requires_all_properties_and_closes_objects():
    s = strict_json_schema(EvaluatorOutput)
    assert s["additionalProperties"] is False
    assert set(s["required"]) == {"label", "score", "reason"}
    assert "default" not in s["properties"]["score"]
    json.dumps(s)  # serializable


def test_api_key_only_from_dotenv(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "from-environment")
    with pytest.raises(MissingAPIKeyError):
        load_api_key("OPENAI_API_KEY", env_file=tmp_path / ".env")
    (tmp_path / ".env").write_text("OPENAI_API_KEY=sk-test\n", encoding="utf-8")
    assert load_api_key("OPENAI_API_KEY", env_file=tmp_path / ".env") == "sk-test"


def test_fill_slots_is_a_single_pass_over_the_template():
    from rafr.llm.client import fill_slots

    assert fill_slots("a {x} b {y} c", {"x": "{y}", "y": "{x}"}) == "a {y} b {x} c"
    assert fill_slots("{x}{x}", {"x": "{x}"}) == "{x}{x}"
    # unknown braces (a JSON example) are untouched; empty mapping is the identity
    assert fill_slots('{"k": "{x}"} {other}', {"x": "1"}) == '{"k": "1"} {other}'
    assert fill_slots("{x}", {}) == "{x}"


def test_ledger_separates_provider_cache_tokens_from_our_own_cache(tmp_path: Path):
    """(2026-09-20): prompt caching must be visible in the ledger, and it is not the
    same thing as a hit in data/cache/llm — one is billed (cheaply), the other is not billed
    at all."""
    led = Ledger(tmp_path / "l.jsonl")
    live = dict(purpose="evaluator", provider="anthropic", model="claude-opus-5")
    led.record(
        **live,
        model_version="claude-opus-5",
        usage=Usage(70, 13, cache_creation_input_tokens=1800),
        cached=False,
    )
    led.record(
        **live,
        model_version="claude-opus-5",
        usage=Usage(70, 13, cache_read_input_tokens=1800),
        cached=False,
    )
    led.record(  # served from our own cache: replayed tokens, nothing bought
        **live,
        model_version="claude-opus-5",
        usage=Usage(70, 13, cache_read_input_tokens=1800),
        cached=True,
    )
    rows = led.rows()
    assert rows[0]["cache_creation_input_tokens"] == 1800
    assert rows[0]["cache_read_input_tokens"] == 0
    assert rows[1]["cache_read_input_tokens"] == 1800
    t = led.totals
    assert t.calls == 3 and t.cached_calls == 1 and t.live_calls == 2
    assert t.input_tokens == 140 and t.output_tokens == 26  # the cached row adds nothing
    assert t.cache_creation_input_tokens == 1800 and t.cache_read_input_tokens == 1800
    assert t.by_purpose["evaluator"]["calls"] == 3
    assert Usage(70, 13, cache_read_input_tokens=1800).prompt_tokens == 1870


def test_content_text_reads_a_block_list_like_a_string():
    from rafr.llm.client import content_text

    assert content_text("plain") == "plain"
    blocks = [
        {"type": "text", "text": "A", "cache_control": {"type": "ephemeral"}},
        {"type": "text", "text": "B"},
    ]
    assert content_text(blocks) == "AB"


# ------------------------------------ transient provider errors (2026-09-23)


def _net_req(**net) -> LLMRequest:
    return LLMRequest(
        purpose="evaluator",
        provider="fake",
        model="m",
        version="v",
        messages=[{"role": "user", "content": "hi"}],
        network=net,
    )


def test_backoff_is_exponential_capped_jittered_and_honours_retry_after():
    from rafr.llm.client import BackoffPolicy

    p = BackoffPolicy(retries=8, base_s=2, cap_s=60, jitter=0.2)
    assert [p.delay(k) for k in range(8)] == [2, 4, 8, 16, 32, 60, 60, 60]  # u = 0.5: no jitter
    assert p.delay(0, u=0.0) == pytest.approx(1.6) and p.delay(0, u=1.0) == pytest.approx(2.4)
    assert p.delay(0, retry_after=7) == 7  # never shorter than the server asks
    assert BackoffPolicy.from_network({"transient_retries": 3, "backoff_base_s": 1}).retries == 3
    assert BackoffPolicy.from_network(None).retries == 0  # no config: no retries at this layer


def test_transient_errors_are_retried_counted_and_leave_the_cache_key_alone(tmp_path: Path):
    """A 429 / network / 5xx is re-sent unchanged after a backoff; the response that finally
    arrives is cached under the same key as a first-time success."""
    from rafr.llm.client import TransientProviderError

    kinds = iter(["rate_limit", "network", "server"])

    def responder(req):
        k = next(kinds, None)
        if k is not None:
            raise TransientProviderError(k, "boom", retry_after=None, status=429)
        return "ok"

    slept: list[float] = []
    fp = FakeProvider(responder=responder)
    client = LLMClient(
        providers={"fake": fp},
        cache=LLMCache(tmp_path / "c"),
        ledger=Ledger(tmp_path / "l.jsonl"),
        sleep=slept.append,
        transient_log=tmp_path / "t.jsonl",
    )
    req = _net_req(transient_retries=5, backoff_base_s=2, backoff_cap_s=60)
    resp = client.complete(req)
    assert resp.text == "ok" and len(fp.calls) == 4
    assert all(c.cache_key() == req.cache_key() for c in fp.calls)  # the same request
    s = client.transient_stats
    assert (s["rate_limit"], s["network"], s["server"], s["retries"], s["exhausted"]) == (
        1,
        1,
        1,
        3,
        0,
    )
    assert 1.6 <= slept[0] <= 2.4 and 3.2 <= slept[1] <= 4.8 and 6.4 <= slept[2] <= 9.6
    assert len(Ledger(tmp_path / "l.jsonl").rows()) == 1  # a failed send buys nothing
    assert len((tmp_path / "t.jsonl").read_text(encoding="utf-8").splitlines()) == 3
    assert client.complete(req).cached  # the final answer sits under the ordinary key


def test_transient_retries_run_out_with_their_own_error(tmp_path: Path):
    from rafr.llm.client import TransientProviderError, TransientRetriesExhaustedError

    def responder(req):
        raise TransientProviderError("rate_limit", "slow down", retry_after=1.0, status=429)

    client = LLMClient(providers={"fake": FakeProvider(responder=responder)}, sleep=lambda s: None)
    with pytest.raises(TransientRetriesExhaustedError) as ei:
        client.complete(_net_req(transient_retries=2))
    assert ei.value.attempts == 3 and ei.value.last.kind == "rate_limit"
    assert client.transient_stats["rate_limit"] == 3 and client.transient_stats["exhausted"] == 1


def _sdk_http(sdk):
    """The HTTP library ``sdk`` builds its errors on, to make real SDK errors in tests.

    anthropic >= 1 and openai >= 3 are built on ``httpx2``, older releases on ``httpx``; neither
    is a dependency of ours, and a fresh install (CI) may carry only the one the SDK pulls in.
    """
    import importlib

    exc_mod = importlib.import_module(f"{sdk.__name__}._exceptions")
    for name in ("httpx2", "httpx"):
        mod = getattr(exc_mod, name, None)
        if mod is not None:
            return mod
    for name in ("httpx2", "httpx"):
        try:
            return importlib.import_module(name)
        except ImportError:
            continue
    raise ImportError(f"no HTTP library found for {sdk.__name__}")


def test_anthropic_sdk_errors_are_classified_as_transient_or_not():
    import anthropic

    httpx = _sdk_http(anthropic)

    from rafr.llm.providers.anthropic import as_transient

    def status_error(cls, code, headers=None):
        req = httpx.Request("POST", "https://api.anthropic.com/v1/messages")
        return cls(
            "x", response=httpx.Response(code, headers=headers or {}, request=req), body=None
        )

    rl = as_transient(status_error(anthropic.RateLimitError, 429, {"retry-after": "7"}))
    assert (rl.kind, rl.retry_after, rl.status) == ("rate_limit", 7.0, 429)
    assert as_transient(status_error(anthropic.InternalServerError, 500)).kind == "server"
    assert as_transient(status_error(anthropic.APIStatusError, 529)).kind == "server"
    conn = anthropic.APIConnectionError(request=httpx.Request("POST", "https://x"))
    assert as_transient(conn).kind == "network"
    assert as_transient(status_error(anthropic.BadRequestError, 400)) is None
    assert as_transient(ValueError("not an API error")) is None


def test_the_anthropic_provider_raises_transient_errors_for_the_client_to_retry():
    import anthropic

    httpx = _sdk_http(anthropic)

    from rafr.llm.client import TransientProviderError
    from rafr.llm.providers.anthropic import AnthropicProvider

    class _Messages:
        def create(self, **kw):
            req = httpx.Request("POST", "https://x")
            raise anthropic.RateLimitError(
                "rl", response=httpx.Response(429, request=req), body=None
            )

    prov = AnthropicProvider(api_key="test-key-not-used", max_retries=0)
    prov._client = type("C", (), {"messages": _Messages()})()
    with pytest.raises(TransientProviderError) as ei:
        prov.complete(
            LLMRequest(
                purpose="evaluator",
                provider="anthropic",
                model="claude-opus-5",
                version="v",
                messages=[{"role": "user", "content": "x"}],
                decoding={"max_output_tokens": 10},
            )
        )
    assert ei.value.kind == "rate_limit"


def test_e_sdk_retries_are_off_and_the_counted_backoff_is_on():
    """The SDK must not retry silently, or a 429 it absorbed would never be counted."""
    from rafr.core.config import load_config

    net = load_config("models")["evaluator"]["network"]
    assert net["max_retries"] == 0 and net["transient_retries"] > 0


def test_openai_sdk_errors_are_classified_as_transient_or_not():
    import openai

    httpx = _sdk_http(openai)

    from rafr.llm.providers.openai import as_transient

    def status_error(cls, code, headers=None, body=None):
        req = httpx.Request("POST", "https://api.openai.com/v1/chat/completions")
        return cls(
            "x", response=httpx.Response(code, headers=headers or {}, request=req), body=body
        )

    rl = as_transient(status_error(openai.RateLimitError, 429, {"retry-after": "7"}))
    assert (rl.kind, rl.retry_after, rl.status) == ("rate_limit", 7.0, 429)
    # an exhausted account is a 429 as well, but waiting does not cure it: raised as-is
    quota = status_error(openai.RateLimitError, 429, body={"code": "insufficient_quota"})
    assert as_transient(quota) is None
    assert as_transient(status_error(openai.InternalServerError, 500)).kind == "server"
    assert as_transient(status_error(openai.APIStatusError, 503)).kind == "server"
    req = httpx.Request("POST", "https://x")
    assert as_transient(openai.APIConnectionError(request=req)).kind == "network"
    assert as_transient(openai.APITimeoutError(request=req)).kind == "network"
    assert as_transient(status_error(openai.BadRequestError, 400)) is None
    assert as_transient(ValueError("not an API error")) is None


def test_the_openai_provider_raises_transient_errors_for_the_client_to_retry():
    import openai

    httpx = _sdk_http(openai)

    from rafr.llm.client import TransientProviderError
    from rafr.llm.providers.openai import OpenAIProvider

    status = {"code": 429}

    class _Completions:
        def create(self, **kw):
            req = httpx.Request("POST", "https://x")
            resp = httpx.Response(status["code"], request=req)
            if status["code"] == 429:
                raise openai.RateLimitError("rl", response=resp, body=None)
            raise openai.BadRequestError("bad", response=resp, body=None)

    prov = OpenAIProvider(api_key="test-key-not-used", max_retries=0)
    prov._client = type("C", (), {"chat": type("Chat", (), {"completions": _Completions()})()})()
    req = LLMRequest(
        purpose="g_fixed",
        provider="openai",
        model="gpt-5.6-terra",
        version="v",
        messages=[{"role": "user", "content": "x"}],
        decoding={"max_output_tokens": 10},
    )
    with pytest.raises(TransientProviderError) as ei:
        prov.complete(req)
    assert ei.value.kind == "rate_limit" and ei.value.status == 429
    status["code"] = 400  # a real error is not retried: it passes through untouched
    with pytest.raises(openai.BadRequestError):
        prov.complete(req)


def test_openai_sdk_retries_are_off_and_the_counted_backoff_is_on(sandbox: Path, monkeypatch):
    """Same rule as E (2026-09-23): the OpenAI SDK must not retry silently, or a 429
    it absorbed would never be counted. G_fixed inherits the controller's network block, so the
    one setting covers both, and build_client hands 0 to both SDKs."""
    from rafr.core import runtime
    from rafr.core.config import load_config
    from rafr.generation import g_fixed
    from rafr.llm.providers import anthropic as anthropic_mod
    from rafr.llm.providers import openai as openai_mod

    cfgs = load_config()
    net = cfgs["models"]["controller"]["network"]
    assert net["max_retries"] == 0 and net["transient_retries"] > 0
    assert g_fixed.build_config(cfgs["models"], cfgs["generation"], base_seed=7).network == net

    seen: dict[str, int] = {}

    def capture(name):
        class _Provider:
            def __init__(self, *, max_retries, timeout_s):
                seen[name] = max_retries

        return _Provider

    monkeypatch.setattr(openai_mod, "OpenAIProvider", capture("openai"))
    monkeypatch.setattr(anthropic_mod, "AnthropicProvider", capture("anthropic"))
    runtime.build_client(cfgs["models"], run_dir=sandbox / "runs" / "x")
    assert seen == {"openai": 0, "anthropic": 0}
