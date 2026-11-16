"""The panel: two substitute judges for Exp 2's sensitivity analysis (II.3, III.1.2; v0.9).

Grok 4.6 and Qwen3.8-Max each re-judge the x_T of BOTH arms after Exp 2 has finished. Each
member is its own substitute judge: its labels give its own TFR_A1 - TFR_B and CI
(``analysis.exp2_tfr.panel_sensitivity``), nothing is merged across members and there is no
tie rule. The panel never enters tau, the gate or Exp 1 (II.3, App. B.1); it runs on the laptop
after the loop, never on the pod.

**Same prompt as E** (II.3): the member configs carry no prompt of their own. The prompt is
``models.evaluator.prompt_file``, read and checked against ``models.evaluator.prompt_sha256``
by :func:`rafr.evaluator.judge.build_config`, rendered by the same one-pass slot fill, sent as
ONE plain user message with no system prompt and no structured output, and the answer text is
parsed by E's parser (:func:`rafr.evaluator.judge.parse_response`). The decoding block is each
member's own (``models.panel.members.<name>.decoding``), because the knobs differ by provider.

Failures (II.9.1 rule for E, applied here): a refusal / block or an unreadable answer is retried
twice with the attempt index bumped. What a member's final failure does to the analysis is
``models.panel.on_failure``: ``abort`` stops the step; ``exclude_and_report`` (2026-09-24)
records the failure, leaves that trajectory out of THAT member's TFR only, and reports the count
per arm.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from rafr.contracts.schemas import (
    PARSE_MAX_RETRIES,
    TBD,
    SafetyLabel,
    text_hash,
    utc_now_iso,
)
from rafr.evaluator import judge
from rafr.llm.client import LLMClient, LLMRequest, fill_slots
from rafr.logging.writer import JsonlWriter, read_jsonl

PURPOSE_PREFIX = "panel"
PROVIDERS: tuple[str, ...] = ("xai", "dashscope")
ON_FAILURE: tuple[str, ...] = ("abort", "exclude_and_report")


class PanelConfigError(ValueError):
    pass


class PanelError(RuntimeError):
    """A member gave no verdict after every attempt and ``on_failure`` is ``abort``."""


@dataclass
class PanelMember:
    name: str
    provider: str
    model: str
    version: str | None
    decoding: dict[str, Any]
    prompt_template: str  # E's prompt, verbatim
    prompt_sha256: str  # E's prompt hash (II.3: same prompt)
    network: dict[str, Any] = field(default_factory=dict)

    @property
    def purpose(self) -> str:
        return f"{PURPOSE_PREFIX}:{self.name}"

    def identity(self) -> dict[str, Any]:
        return {
            "member": self.name,
            "provider": self.provider,
            "model": self.model,
            "version": self.version,
            "decoding": self.decoding,
            "prompt_sha256": self.prompt_sha256,
        }


def on_failure_rule(models_cfg: Mapping[str, Any], *, allow_tbd: bool = False) -> str:
    rule = (models_cfg.get("panel") or {}).get("on_failure", TBD)
    if rule == TBD and allow_tbd:
        return "exclude_and_report"  # dry run only: records the failure, stops nothing
    if rule not in ON_FAILURE:
        raise PanelConfigError(f"models.panel.on_failure must be one of {ON_FAILURE}, got {rule!r}")
    return str(rule)


def build_members(
    models_cfg: Mapping[str, Any],
    *,
    allow_tbd: bool = False,
    placeholder_ids: Mapping[str, str] | None = None,
) -> dict[str, PanelMember]:
    """Every configured member, each with E's prompt (read and hash-checked through E's own
    builder). ``allow_tbd`` is for the dry run: a TBD model id is then replaced by
    ``placeholder_ids[name]`` (or ``"<name>-placeholder"``) and a TBD decoding value by null
    (not sent); the members built that way can only talk to a fake provider."""
    ecfg = judge.build_config(models_cfg, allow_tbd=allow_tbd)
    panel = models_cfg.get("panel") or {}
    members = panel.get("members") or {}
    if not members:
        raise PanelConfigError("models.panel.members is empty")
    network = dict(panel.get("network") or {})
    out: dict[str, PanelMember] = {}
    for name, m in members.items():
        if m.get("provider") not in PROVIDERS:
            raise PanelConfigError(f"models.panel.members.{name}.provider must be in {PROVIDERS}")
        if m.get("prompt_file") or m.get("prompt"):
            raise PanelConfigError(
                f"models.panel.members.{name} sets its own prompt; the panel uses E's (II.3)"
            )
        decoding = dict(m.get("decoding") or {})
        model, version = m.get("model", TBD), m.get("version", TBD)
        tbd = [k for k in ("model", "version") if m.get(k, TBD) in (None, TBD)]
        tbd += [f"decoding.{k}" for k, v in decoding.items() if v == TBD]
        if tbd and not allow_tbd:
            raise PanelConfigError(
                f"models.panel.members.{name}: {', '.join(tbd)} still TBD (App. D: the authors, "
                "before the panel re-judges x_T)"
            )
        if model in (None, TBD):
            model = (placeholder_ids or {}).get(name, f"{name}-placeholder")
        decoding = {k: (None if v == TBD else v) for k, v in decoding.items()}
        out[name] = PanelMember(
            name=name,
            provider=str(m["provider"]),
            model=str(model),
            version=None if version in (None, TBD) else str(version),
            decoding=decoding,
            prompt_template=ecfg.prompt_template,
            prompt_sha256=ecfg.prompt_sha256,
            network=network,
        )
    return out


def render_prompt(member: PanelMember, text: str) -> str:
    """E's prompt with ``{text}`` filled in one pass -- the same string E is shown."""
    return fill_slots(member.prompt_template, {"text": text})


def build_request(member: PanelMember, text: str, attempt: int = 0) -> LLMRequest:
    return LLMRequest(
        purpose=member.purpose,
        provider=member.provider,
        api="chat_completions",
        model=member.model,
        version=member.version,
        system=None,
        messages=[{"role": "user", "content": render_prompt(member, text)}],
        schema=None,
        decoding=member.decoding,
        seed=None,
        attempt=attempt,
        network=member.network,
    )


class PanelJudgment(BaseModel):
    """One member's judgment of one x_T (or the failure that left it without one)."""

    model_config = ConfigDict(extra="forbid")

    member: str
    text_id: str
    text: str
    label: SafetyLabel | None
    reason: str | None = None
    failures: list[str] = Field(default_factory=list)
    model: str
    version: str | None
    model_version: str | None = None
    prompt_sha256: str
    prompt_hash: str | None = None
    decoding: dict[str, Any]
    cached: bool = False
    created_at: str = Field(default_factory=utc_now_iso)


def judge_text(client: LLMClient, member: PanelMember, text: str) -> PanelJudgment:
    """One member, one text: up to ``PARSE_MAX_RETRIES + 1`` attempts, as for E (II.9.1).

    Returns a judgment with ``label=None`` and every attempt's reason when all attempts fail;
    what that means for the run is the caller's ``on_failure`` rule.
    """
    failures: list[str] = []
    last_version: str | None = None
    last_hash: str | None = None
    for attempt in range(PARSE_MAX_RETRIES + 1):
        resp = client.complete(build_request(member, text, attempt))
        last_version, last_hash = resp.model_version, resp.prompt_hash
        if resp.refused:
            failures.append("blocked" if resp.stop_reason == "blocked" else "refusal")
            continue
        label, reason = judge.parse_response(resp.text)
        if label is None:
            failures.append(
                resp.error or ("empty" if not (resp.text or "").strip() else "unparseable")
            )
            continue
        return PanelJudgment(
            member=member.name,
            text_id=text_hash(text),
            text=text,
            label=SafetyLabel(label),
            reason=reason,
            failures=failures,
            model=member.model,
            version=member.version,
            model_version=resp.model_version,
            prompt_sha256=member.prompt_sha256,
            prompt_hash=resp.prompt_hash,
            decoding=member.decoding,
            cached=resp.cached,
        )
    return PanelJudgment(
        member=member.name,
        text_id=text_hash(text),
        text=text,
        label=None,
        failures=failures,
        model=member.model,
        version=member.version,
        model_version=last_version,
        prompt_sha256=member.prompt_sha256,
        prompt_hash=last_hash,
        decoding=member.decoding,
    )


def panel_path(run_dir: Path, member: str) -> Path:
    return Path(run_dir) / f"panel_{member}.jsonl"


def rejudge_x_t(
    client: LLMClient,
    members: Mapping[str, PanelMember],
    finals: Sequence[Mapping[str, Any]],
    run_dir: Path,
    *,
    on_failure: str,
    echo: Callable[[str], None] | None = None,
) -> dict[str, dict[str, Any]]:
    """Every member re-judges the x_T of every trajectory, both arms (III.1.2).

    One file per member, ``panel_<member>.jsonl``, one row per distinct x_T text (the members
    see text only, so identical terminal texts share a judgment); rows are appended as they are
    bought, so an interrupted pass resumes where it stopped. Returns per member the counts and
    ``labels``: ``text_id -> "SAFE" | "UNSAFE" | None``.
    """
    if on_failure not in ON_FAILURE:
        raise PanelConfigError(f"on_failure must be one of {ON_FAILURE}, got {on_failure!r}")
    say = echo or (lambda _s: None)
    texts: dict[str, str] = {}
    for f in finals:
        texts.setdefault(text_hash(f["x_T"]), f["x_T"])
    out: dict[str, dict[str, Any]] = {}
    for name, member in members.items():
        path = panel_path(run_dir, name)
        done: dict[str, PanelJudgment] = {}
        for row in read_jsonl(path):
            j = PanelJudgment.model_validate(row)
            if j.member != name or j.model != member.model or j.decoding != member.decoding:
                raise PanelConfigError(
                    f"{path} holds judgments of a different {name} configuration; move it away"
                )
            if j.prompt_sha256 != member.prompt_sha256:
                raise PanelConfigError(f"{path} was judged with a different prompt (II.3)")
            done[j.text_id] = j
        with JsonlWriter(path) as w:
            for tid, text in texts.items():
                if tid in done and done[tid].label is not None:
                    continue
                j = judge_text(client, member, text)
                w.write(j)
                done[tid] = j
                if j.label is None and on_failure == "abort":
                    raise PanelError(
                        f"panel member {name} gave no verdict on {tid} after "
                        f"{PARSE_MAX_RETRIES + 1} attempts ({j.failures}); on_failure = abort"
                    )
        labels = {tid: (done[tid].label.value if done[tid].label else None) for tid in texts}
        by_arm: dict[str, int] = {}
        for f in finals:
            if labels[text_hash(f["x_T"])] is None:
                by_arm[str(f.get("arm"))] = by_arm.get(str(f.get("arm")), 0) + 1
        out[name] = {
            "file": str(path),
            "n_texts": len(texts),
            "n_failed": sum(1 for v in labels.values() if v is None),
            # trajectories left out of THIS member's TFR under exclude_and_report, per arm
            "n_excluded_by_arm": dict(sorted(by_arm.items())),
            "identity": member.identity(),
            "labels": labels,
        }
        say(
            f"[panel] {name}: {len(texts)} x_T, {out[name]['n_failed']} without a verdict "
            f"(excluded by arm: {out[name]['n_excluded_by_arm']})"
        )
    return out
