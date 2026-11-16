"""The panel (II.3, III.1.2; v0.9): E's prompt, per-member analysis, providers, boundaries."""

from __future__ import annotations

import copy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from analysis import exp2_tfr

from rafr.contracts.schemas import text_hash
from rafr.core.config import load_config
from rafr.evaluator import judge
from rafr.evaluator import panel as PN
from rafr.llm.cache import LLMCache
from rafr.llm.client import LLMClient
from rafr.llm.providers import MissingAPIKeyError
from rafr.llm.providers import openai_compatible as OC
from rafr.llm.providers.fake import FakeProvider, FakeReply

ROOT = Path(__file__).resolve().parents[1]

QWEN_MEMBER = {
    "provider": "dashscope",
    "family": "qwen3.8-max",
    "model": "qwen3.8-max-0902",
    "version": "created=1788313616 (2026-09-02T01:46:56Z)",
    "decoding": {"temperature": 0.0, "reasoning_params": {}, "max_output_tokens": 4096},
}
"""The member removed from the real config (D-2026-09-28b); the DashScope path stays tested."""


def _with_qwen(m: dict) -> dict:
    m["panel"]["members"]["qwen"] = copy.deepcopy(QWEN_MEMBER)
    return m


def _filled_models() -> dict:
    m = _with_qwen(copy.deepcopy(load_config("models")))
    m["panel"]["on_failure"] = "exclude_and_report"
    for name, member in m["panel"]["members"].items():
        member["model"] = f"{name}-model"
        member["version"] = "v1"
        member["decoding"]["temperature"] = None
    return m


def test_the_real_config_carries_ruides_app_d_values():
    """2026-09-24 (decisions.md D-2026-09-24c): the ids and versions the 2026-09-22
    selection test actually called, temperature 0.0, exclude_and_report. 2026-09-28
    (D-2026-09-28b): the detox pilot's panel is Grok only (DashScope unreachable)."""
    m = load_config("models")
    members = PN.build_members(m)
    assert PN.on_failure_rule(m) == "exclude_and_report"
    assert set(members) == {"grok"}
    assert members["grok"].provider == "xai"
    assert (members["grok"].model, members["grok"].version) == ("grok-4.6", "1.0@fp_9a755add8c")
    assert all(mm.decoding["temperature"] == 0.0 for mm in members.values())
    assert all(mm.decoding["max_output_tokens"] == 4096 for mm in members.values())
    assert all(mm.decoding["reasoning_params"] == {} for mm in members.values())


def test_tbd_panel_values_refuse_outside_the_dry_run():
    m = _with_qwen(copy.deepcopy(load_config("models")))
    m["panel"]["on_failure"] = "TBD"
    m["panel"]["members"]["grok"]["model"] = "TBD"
    m["panel"]["members"]["qwen"]["decoding"]["temperature"] = "TBD"
    with pytest.raises(PN.PanelConfigError, match="TBD"):
        PN.build_members(m)
    with pytest.raises(PN.PanelConfigError, match="on_failure"):
        PN.on_failure_rule(m)
    members = PN.build_members(m, allow_tbd=True)  # dry run: placeholders, nothing sent for TBD
    assert members["grok"].model == "grok-placeholder"
    assert members["qwen"].decoding["temperature"] is None


def test_members_are_shown_exactly_e_prompt_as_one_plain_message():
    """II.3: the same rubric-free prompt as E, plain prompt, no system, no schema."""
    m = _filled_models()
    ecfg = judge.build_config(m)
    members = PN.build_members(m)
    text = 'He said "{text}" and left.'  # a slot name inside the text is judged verbatim
    for mm in members.values():
        assert mm.prompt_sha256 == ecfg.prompt_sha256 == m["evaluator"]["prompt_sha256"]
        req = PN.build_request(mm, text)
        assert req.system is None and req.schema is None and req.seed is None
        assert req.messages == [{"role": "user", "content": judge.render_prompt(ecfg, text)}]
        # E's own request carries the same bytes (as prefix + tail blocks)
        e_req = judge.build_request(ecfg, text)
        e_text = "".join(b["text"] for b in e_req.messages[0]["content"])
        assert req.messages[0]["content"] == e_text
    bad = copy.deepcopy(m)
    bad["panel"]["members"]["grok"]["prompt_file"] = "x.txt"
    with pytest.raises(PN.PanelConfigError, match="E's"):
        PN.build_members(bad)


def _client(tmp_path: Path, responder) -> tuple[LLMClient, FakeProvider]:
    fp = FakeProvider(responder=responder)
    return LLMClient(providers={"xai": fp, "dashscope": fp}, cache=LLMCache(tmp_path / "c")), fp


def test_judge_text_retries_twice_then_records_a_failure(tmp_path: Path):
    members = PN.build_members(_filled_models())
    replies = iter(
        [
            FakeReply(text=None, refused=True, stop_reason="blocked"),
            FakeReply(text="no json here"),
            FakeReply(text='{"label": "unsafe", "reason": "r."}'),
        ]
    )
    client, fp = _client(tmp_path, lambda r: next(replies))
    j = PN.judge_text(client, members["grok"], "a text")
    assert j.label is not None and j.label.value == "UNSAFE" and j.reason == "r."
    assert j.failures == ["blocked", "unparseable"] and len(fp.calls) == 3
    assert [c.attempt for c in fp.calls] == [0, 1, 2]
    client2, _ = _client(tmp_path / "b", lambda r: FakeReply(text=None, refused=True))
    j2 = PN.judge_text(client2, members["qwen"], "another")
    assert j2.label is None and j2.failures == ["refusal"] * 3


def _finals() -> list[dict]:
    return [
        {"x0_id": "a", "arm": "A1", "x_T": "same text", "task_success": True},
        {"x0_id": "b", "arm": "B", "x_T": "same text", "task_success": True},
        {"x0_id": "c", "arm": "A1", "x_T": "bad text", "task_success": False},
        {"x0_id": "d", "arm": "B", "x_T": "fine text", "task_success": True},
    ]


def test_rejudge_dedupes_by_text_one_file_per_member_and_resumes(tmp_path: Path):
    members = PN.build_members(_filled_models())

    def responder(req):
        body = req.messages[0]["content"]
        label = "UNSAFE" if "bad text" in body else "SAFE"
        if req.provider == "dashscope" and "fine text" in body:
            return FakeReply(text=None, refused=True, stop_reason="blocked")
        return json.dumps({"label": label, "reason": "x."})

    client, fp = _client(tmp_path, responder)
    out = PN.rejudge_x_t(client, members, _finals(), tmp_path, on_failure="exclude_and_report")
    assert set(out) == {"grok", "qwen"}
    assert out["grok"]["n_texts"] == 3 and out["grok"]["n_failed"] == 0
    assert out["qwen"]["n_failed"] == 1  # blocked three times -> no verdict, recorded
    assert out["qwen"]["n_excluded_by_arm"] == {"B": 1} and out["grok"]["n_excluded_by_arm"] == {}
    assert (tmp_path / "panel_grok.jsonl").exists() and (tmp_path / "panel_qwen.jsonl").exists()
    grok_calls = [c for c in fp.calls if c.provider == "xai"]
    assert len(grok_calls) == 3  # three distinct texts, not four trajectories
    n_before = len(fp.calls)
    again = PN.rejudge_x_t(client, members, _finals(), tmp_path, on_failure="exclude_and_report")
    assert again["grok"]["labels"] == out["grok"]["labels"]
    # only the failed text is asked again (from our cache: nothing new is bought)
    assert len(fp.calls) == n_before
    with pytest.raises(PN.PanelError, match="abort"):
        PN.rejudge_x_t(client, members, _finals(), tmp_path / "abort", on_failure="abort")


def test_panel_sensitivity_is_per_member_and_never_merged():
    finals = [
        {"x0_id": f"x{i}", "arm": "A1" if i % 2 else "B", "x_T": f"t{i}", "task_success": True}
        for i in range(8)
    ]
    labels_a = {text_hash(f["x_T"]): ("UNSAFE" if f["arm"] == "A1" else "SAFE") for f in finals}
    labels_b = {text_hash(f["x_T"]): "SAFE" for f in finals}
    labels_b[text_hash("t1")] = None  # no verdict: out of member b's TFR only
    res = exp2_tfr.panel_sensitivity(
        finals, {"a": labels_a, "b": labels_b}, main_estimate=0.25, seed=0, n_boot=50
    )
    a, b = res["members"]["a"], res["members"]["b"]
    assert a["contrast"]["estimate"] == 1.0 and a["same_direction_as_main"] is True
    assert b["contrast"]["estimate"] == 0.0 and b["same_direction_as_main"] is False
    assert b["n_without_verdict_by_arm"] == {"A1": 1} and b["contrast"]["n"] == 7
    assert a["contrast"]["n"] == 8
    assert "not merged" in res["note"]


class _FakeCompletions:
    def __init__(self, reply=None, stream=None, exc=None):
        self.reply, self.stream, self.exc, self.kwargs = reply, stream, exc, None

    def create(self, **kwargs):
        self.kwargs = kwargs
        if self.exc is not None:
            raise self.exc
        return self.stream if kwargs.get("stream") else self.reply


def _provider(cls, completions):
    p = cls.__new__(cls)
    p.base_url = "http://fake"
    p._client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
    return p


def test_openai_compatible_providers_send_what_the_config_says(tmp_path: Path):
    members = PN.build_members(_filled_models())
    usage = SimpleNamespace(prompt_tokens=100, completion_tokens=10, total_tokens=150)
    reply = SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(content='{"label":"SAFE"}', refusal=None),
                finish_reason="stop",
            )
        ],
        model="grok-4.6",
        system_fingerprint="fp_1",
        usage=usage,
        id="r1",
    )
    comp = _FakeCompletions(reply=reply)
    xai = _provider(OC.XAIProvider, comp)
    resp = xai.complete(PN.build_request(members["grok"], "t"))
    assert comp.kwargs["max_tokens"] == 4096
    assert "temperature" not in comp.kwargs and "extra_body" not in comp.kwargs
    assert comp.kwargs["messages"][0]["role"] == "user" and len(comp.kwargs["messages"]) == 1
    assert resp.text == '{"label":"SAFE"}' and resp.model_version == "grok-4.6@fp_1"
    assert resp.usage.input_tokens == 100 and resp.usage.output_tokens == 50  # thinking billed

    chunks = [
        SimpleNamespace(
            model="qwen", id="q1", system_fingerprint=None, usage=None,
            choices=[SimpleNamespace(delta=SimpleNamespace(content='{"label":', refusal=None),
                                     finish_reason=None)],
        ),
        SimpleNamespace(
            model="qwen", id="q1", system_fingerprint=None,
            usage=SimpleNamespace(prompt_tokens=90, completion_tokens=40, total_tokens=130),
            choices=[SimpleNamespace(delta=SimpleNamespace(content=' "UNSAFE"}', refusal=None),
                                     finish_reason="stop")],
        ),
    ]  # fmt: skip
    qcomp = _FakeCompletions(stream=iter(chunks))
    ds = _provider(OC.DashScopeProvider, qcomp)
    req = PN.build_request(members["qwen"], "t")
    req.decoding = {**req.decoding, "temperature": 0.0, "reasoning_params": {"x": 1}}
    resp = ds.complete(req)
    assert qcomp.kwargs["stream"] is True and qcomp.kwargs["temperature"] == 0.0
    assert qcomp.kwargs["extra_body"] == {"x": 1}
    assert resp.text == '{"label": "UNSAFE"}' and resp.usage.output_tokens == 40

    class BlockedError(Exception):
        status_code = 400

    bcomp = _FakeCompletions(exc=BlockedError("Error code: 400 - data_inspection_failed"))
    blocked = _provider(OC.DashScopeProvider, bcomp).complete(req)
    assert blocked.refused and blocked.stop_reason == "blocked"

    class BrokenError(Exception):
        status_code = 400

    with pytest.raises(BrokenError):
        _provider(OC.XAIProvider, _FakeCompletions(exc=BrokenError("bad request"))).complete(req)


def test_panel_keys_are_never_read_in_tests():
    with pytest.raises(MissingAPIKeyError):
        OC.XAIProvider()
    with pytest.raises(MissingAPIKeyError):
        OC.DashScopeProvider()


def test_panel_never_reaches_tau_gate_or_exp1():
    """App. B.1 (v0.9): the panel may not enter tau, the gate or Exp 1."""
    files = [
        *sorted((ROOT / "src" / "rafr" / "validity").glob("*.py")),
        *sorted((ROOT / "src" / "rafr" / "channel").glob("*.py")),
        ROOT / "src" / "rafr" / "experiments" / "exp1.py",
        ROOT / "analysis" / "exp1_did.py",
        *[ROOT / "pipelines" / f for f in (
            "step07_fit_tau.py", "step08_tune_channel.py", "step09_gate.py", "step11_exp1.py",
            "_validity_io.py",
        )],
    ]  # fmt: skip
    bad = [f.name for f in files if "panel" in f.read_text(encoding="utf-8").lower()]
    assert bad == []
