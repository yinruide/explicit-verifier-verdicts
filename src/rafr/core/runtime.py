"""What a pipeline step needs at run time: the LLM client, and what it is spending.

Kept out of ``rafr.llm`` so that the provider modules stay import-free of config code.

The price list lives here rather than in the frozen config (decision, 2026-09-17: no
price list in config — it blocked formal runs on typing in four numbers for a
multiplication). It is used for the ``--budget`` ceiling and the progress line of the two
steps that buy thousands of calls (step00, step04); the ledger still records raw tokens, and
nothing scientific depends on these numbers.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from rafr.core import paths
from rafr.llm.cache import LLMCache
from rafr.llm.client import LLMClient
from rafr.llm.ledger import TOKEN_FIELDS, Ledger
from rafr.llm.providers import Provider


@dataclass(frozen=True)
class Prices:
    """USD per million tokens."""

    input_: float
    output: float
    cache_write: float
    cache_read: float

    def cost(self, tokens: Mapping[str, int]) -> float:
        return (
            int(tokens.get("input_tokens", 0)) * self.input_
            + int(tokens.get("output_tokens", 0)) * self.output
            + int(tokens.get("cache_creation_input_tokens", 0)) * self.cache_write
            + int(tokens.get("cache_read_input_tokens", 0)) * self.cache_read
        ) / 1_000_000


PRICES: dict[str, Prices] = {
    # Anthropic list price, checked 2026-09-20: $5 / $25 per MTok, cache write 1.25x
    # (5-minute TTL), cache read 0.1x.
    "claude-opus-5": Prices(input_=5.00, output=25.00, cache_write=6.25, cache_read=0.50),
    # OpenAI list price, developers.openai.com/docs/pricing, checked 2026-09-20: standard
    # tier, SHORT context ($2 / $12; the long-context column is $4 / $18, and a G_fixed or
    # controller request is a couple of hundred tokens, nowhere near that tier).
    # cache_read / cache_write are carried for completeness: OpenAI's automatic caching needs
    # a prompt of at least ~1K tokens, which these prompts never reach, and the provider does
    # not report cached tokens separately anyway — so every input token is priced at the full
    # rate here. That errs high, which is the right direction for a ceiling.
    "gpt-5.6-terra": Prices(input_=2.00, output=12.00, cache_write=2.50, cache_read=0.20),
    # The two panel models (II.3), as the annotator-selection config recorded them on
    # 2026-09-22 (docs.x.ai pricing; Alibaba Model Studio, Beijing region converted at
    # Alibaba's own rate). Neither provider bills a cache write; cache_write = input.
    "grok-4.6": Prices(input_=2.00, output=6.00, cache_write=2.00, cache_read=0.50),
    "qwen3.8-max-0902": Prices(input_=1.65, output=4.951, cache_write=1.65, cache_read=0.21),
}


def prices_for(model: str) -> Prices | None:
    return PRICES.get(model)


def fmt_duration(seconds: float) -> str:
    s = int(max(0.0, seconds))
    h, rem = divmod(s, 3600)
    m, sec = divmod(rem, 60)
    return f"{h}h{m:02d}m" if h else (f"{m}m{sec:02d}s" if m else f"{sec}s")


def fmt_tokens(n: int) -> str:
    if n >= 1_000_000:
        return f"{n / 1_000_000:.2f}M"
    return f"{n / 1000:.1f}K" if n >= 1000 else str(n)


class Spend:
    """Live token counts and their cost, read from the run's ledger as it is written.

    Only live calls count: a response served from ``data/cache/llm`` bought nothing.
    ``prices`` is None when no price list is known for the model — then the tokens are still
    reported and ``--budget`` is refused rather than silently unenforced.
    """

    def __init__(self, ledger: Ledger | None, prices: Prices | None, purpose: str) -> None:
        self._ledger = ledger
        self.prices = prices
        self._purpose = purpose

    @property
    def tokens(self) -> dict[str, int]:
        per = None if self._ledger is None else self._ledger.totals.by_purpose.get(self._purpose)
        if per is None:
            return dict.fromkeys(TOKEN_FIELDS, 0)
        return {k: int(per.get(k, 0)) for k in TOKEN_FIELDS}

    @property
    def usd(self) -> float | None:
        return None if self.prices is None else self.prices.cost(self.tokens)

    @property
    def live_calls(self) -> int:
        if self._ledger is None:
            return 0
        per = self._ledger.totals.by_purpose.get(self._purpose) or {}
        return int(per.get("calls", 0)) - int(per.get("cached_calls", 0))

    def describe(self) -> str:
        t = self.tokens
        usd = self.usd
        money = "n/a" if usd is None else f"${usd:,.2f}"
        cache = (
            f" (cache read {fmt_tokens(t['cache_read_input_tokens'])} / "
            f"write {fmt_tokens(t['cache_creation_input_tokens'])})"
            if t["cache_read_input_tokens"] or t["cache_creation_input_tokens"]
            else ""
        )
        return (
            f"in {fmt_tokens(t['input_tokens'])}{cache} "
            f"out {fmt_tokens(t['output_tokens'])} | {money}"
        )


ERROR_STOP_REASONS: frozenset[str] = frozenset(
    {
        "e_failures",  # step00 / step05: E failed N times in a row
        "transient_errors",  # step00 / 04 / 05: a request's 429 / network / 5xx retries ran out
        "generation_failures",  # step04: N x0 in a row produced nothing valid
    }
)
"""Stop reasons that need a decision before the step can finish. The others (a --limit or
--budget ceiling) are healthy pauses."""


def cost_projection(
    spend: Spend, *, items_bought: int, items_total: int, items_left: int
) -> dict[str, Any]:
    """Measured unit price and the full-run estimate from what this run has bought.

    2026-09-23: every paid segment starts with a dry run (``--limit N``) that prints
    the measured price per item and what the whole step will cost, then stops; the full run's
    ``--budget`` is 1.5x that estimate. An "item" is what the step works through -- an x0 for
    step00 and step04, a state for step05 -- so retries and regenerations are inside the price.
    """
    usd = spend.usd
    out: dict[str, Any] = {
        "items_bought": items_bought,
        "items_total": items_total,
        "items_left": items_left,
        "spend_usd": usd,
        "usd_per_item": None,
        "projected_total_usd": None,
        "projected_remaining_usd": None,
        "budget_at_1_5x_remaining_usd": None,
    }
    if usd is None or items_bought <= 0:
        return out
    unit = usd / items_bought
    out.update(
        usd_per_item=unit,
        projected_total_usd=unit * items_total,
        projected_remaining_usd=unit * items_left,
        budget_at_1_5x_remaining_usd=1.5 * unit * items_left,
    )
    return out


def render_projection(p: Mapping[str, Any], *, step: str, item: str) -> str:
    if p.get("usd_per_item") is None:
        return f"[{step}] cost projection: nothing bought this run, no measured price"
    return (
        f"[{step}] measured price: ${p['usd_per_item']:.5f} per {item} over "
        f"{p['items_bought']} {item}s bought this run (${p['spend_usd']:.4f})\n"
        f"[{step}] projection: all {p['items_total']} {item}s ~ ${p['projected_total_usd']:.2f}; "
        f"the {p['items_left']} still to buy ~ ${p['projected_remaining_usd']:.2f} "
        f"(1.5x = ${p['budget_at_1_5x_remaining_usd']:.2f})"
    )


@dataclass
class Stop:
    """Why a long, paid loop stopped, in the words the operator needs.

    ``reason`` is a short tag the step branches on; ``detail`` says what happened in numbers
    and ``resume`` says where a re-run picks up. Every stop of step00 / step04 fills all
    three — a run that stops without saying how to continue it is a run nobody dares restart.
    """

    reason: str
    detail: str = ""
    resume: str = ""

    @property
    def is_error(self) -> bool:
        return self.reason in ERROR_STOP_REASONS


def transient_suffix(stats: Mapping[str, float] | None) -> str:
    """`` | 429 n net n 5xx n`` for a progress line, or "" while there has been none."""
    s = stats or {}
    if not (s.get("rate_limit") or s.get("network") or s.get("server")):
        return ""
    return (
        f" | 429 {int(s.get('rate_limit', 0))} net {int(s.get('network', 0))} "
        f"5xx {int(s.get('server', 0))}"
    )


POD_OPENAI_KEY_ENV = "OPENAI_API_KEY"


class PodKeyMissingError(RuntimeError):
    pass


def build_pod_client(
    models_cfg: Mapping[str, Any],
    *,
    run_dir: paths.Path,
    cache_dir: paths.Path,
    providers: Mapping[str, Provider] | None = None,
) -> LLMClient:
    """The controller's client on the rented pod (Exp 2's loop, task 2026-09-24 section 5).

    OpenAI only: the loop needs no Anthropic, xAI or Qwen key (E and the panel run on the
    laptop afterwards). The key is the pod's ``OPENAI_API_KEY`` environment variable, which
    the authors set in the RunPod UI: its presence is checked, its value goes straight into the SDK
    client and is never printed, logged or written (the one exception to "keys come from .env
    only", because nothing is uploaded). The cache lives outside the shipped code directory so
    that re-shipping the code does not throw away bought answers.
    """
    import os

    if providers is not None:
        provs = dict(providers)
    else:
        from rafr.llm.providers.openai import OpenAIProvider

        if not os.environ.get(POD_OPENAI_KEY_ENV):
            raise PodKeyMissingError(
                f"{POD_OPENAI_KEY_ENV} is not set on this machine (set it as a pod environment "
                "variable in the RunPod UI)"
            )
        cn = models_cfg.get("controller", {}).get("network", {})
        provs = {
            "openai": OpenAIProvider(
                api_key=os.environ[POD_OPENAI_KEY_ENV],
                max_retries=int(cn.get("max_retries", 0)),
                timeout_s=float(cn.get("timeout_s", 120)),
            )
        }
    return LLMClient(
        providers=provs,
        cache=LLMCache(paths.Path(cache_dir)),
        ledger=Ledger(paths.Path(run_dir) / "ledger.jsonl"),
    )


def build_panel_client(
    models_cfg: Mapping[str, Any],
    *,
    run_dir: paths.Path,
    fake: bool = False,
    providers: Mapping[str, Provider] | None = None,
    cache_dir: paths.Path | None = None,
) -> LLMClient:
    """The panel's client (II.3): xAI and DashScope only, built for the members configured in
    ``models.panel.members``. ``fake=True`` registers the fake provider under both names.
    ``cache_dir`` replaces the shared ``data/cache`` (a smoke keeps its own, 2026-10-31)."""
    provs: dict[str, Provider]
    wanted = {
        m.get("provider") for m in (models_cfg.get("panel") or {}).get("members", {}).values()
    }
    if providers is not None:
        provs = dict(providers)
    elif fake:
        from rafr.llm.providers.fake import FakeProvider

        fp = FakeProvider()
        provs = dict.fromkeys(sorted(p for p in wanted if p), fp)
    else:
        from rafr.llm.providers.openai_compatible import DashScopeProvider, XAIProvider

        net = (models_cfg.get("panel") or {}).get("network", {})
        kw = {
            "max_retries": int(net.get("max_retries", 0)),
            "timeout_s": float(net.get("timeout_s", 120)),
        }
        provs = {}
        if "xai" in wanted:
            provs["xai"] = XAIProvider(**kw)
        if "dashscope" in wanted:
            provs["dashscope"] = DashScopeProvider(**kw)
    return LLMClient(
        providers=provs,
        cache=LLMCache(_cache_root(cache_dir, fake)),
        ledger=Ledger(run_dir / "ledger.jsonl"),
    )


def _cache_root(cache_dir: paths.Path | None, fake: bool) -> paths.Path:
    if cache_dir is not None:
        return paths.Path(cache_dir)
    return paths.DATA_CACHE / ("fake" if fake else "llm")


def build_client(
    models_cfg: Mapping[str, Any],
    *,
    run_dir: paths.Path,
    fake: bool = False,
    providers: Mapping[str, Provider] | None = None,
    cache_dir: paths.Path | None = None,
) -> LLMClient:
    """Real OpenAI + Anthropic providers by default; ``fake=True`` registers the fake
    provider under BOTH names so a dry run exercises the same code paths offline.
    ``cache_dir`` replaces the shared ``data/cache`` (a smoke keeps its own, 2026-10-31)."""
    provs: dict[str, Provider]
    if providers is not None:
        provs = dict(providers)
    elif fake:
        from rafr.llm.providers.fake import FakeProvider

        fp = FakeProvider()
        provs = {"openai": fp, "anthropic": fp, "fake": fp}
    else:
        from rafr.llm.providers.anthropic import AnthropicProvider
        from rafr.llm.providers.openai import OpenAIProvider

        cn = models_cfg.get("controller", {}).get("network", {})
        en = models_cfg.get("evaluator", {}).get("network", {})
        provs = {
            "openai": OpenAIProvider(
                max_retries=int(cn.get("max_retries", 5)), timeout_s=float(cn.get("timeout_s", 120))
            ),
            "anthropic": AnthropicProvider(
                max_retries=int(en.get("max_retries", 5)), timeout_s=float(en.get("timeout_s", 120))
            ),
        }
    return LLMClient(
        providers=provs,
        cache=LLMCache(_cache_root(cache_dir, fake)),
        ledger=Ledger(run_dir / "ledger.jsonl"),
    )
