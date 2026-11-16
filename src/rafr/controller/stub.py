"""The always-rewrite stub controller: for tests only (II.4.4, 2026-09-25).

Protocol II.4.4 (in-place revision of 2026-09-25) moves the coverage of rounds 3 and 4 and of the
resume path out of any smoke run and into the tests: a controller that never accepts pushes every
trajectory through all K rounds without a single API call. It is a scripted responder for
:class:`rafr.llm.providers.fake.FakeProvider` plus a matching :class:`ControllerConfig` whose model
id starts with ``stub-`` -- the same rule as the stub verifiers: every formal step refuses it
(:func:`is_stub_controller`; ``pipelines/step11_exp1.py``, ``pipelines/step12_exp2.py``).

The rewrite appends :data:`REWRITE_MARK` to the current text, so the number of marks in ``x_t`` is
the number of rewrites so far, i.e. ``t - 1`` for the text decided at round ``t``. That is what
lets the responder fail to parse at one chosen round (``parse_fail_at_round``) without any state
of its own: all three attempts of that decision return invalid JSON and the runner records a
parse_fail, x_{t+1} = x_t (II.8.2).
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable

from rafr.controller.decide import ControllerConfig
from rafr.llm.client import LLMRequest, content_text

STUB_PREFIX = "stub-"
"""The verifiers' stub prefix (``rafr.verifiers.stub.STUB_PREFIX``), spelled here so that the
controller package does not import the verifier package."""

STUB_CONTROLLER_MODEL = f"{STUB_PREFIX}controller-always-rewrite"
"""Model id of the stub controller. Starts with ``stub-``: formal steps refuse it."""

REWRITE_MARK = " [rw]"
"""Appended by every stub rewrite; ``x_t.count(REWRITE_MARK) == t - 1``."""

_X_T_OPEN, _X_T_CLOSE = "<<X_T>>", "<</X_T>>"
_X_T_RE = re.compile(re.escape(_X_T_OPEN) + r"(.*?)" + re.escape(_X_T_CLOSE), re.DOTALL)

STUB_SYSTEM_PROMPT = "stub controller; at most {k} decisions in total"
STUB_REWRITE_TEMPLATE = "{x0}\n" + _X_T_OPEN + "{x_t}" + _X_T_CLOSE + "\n{observation}"
STUB_RETRY_PROMPT = "Answer with the JSON object only."


def is_stub_controller(model: str | None) -> bool:
    """True for a controller model id that starts with ``stub-`` (the verifiers' rule)."""
    return str(model or "").startswith(STUB_PREFIX)


def rewrites_so_far(x_t: str) -> int:
    return x_t.count(REWRITE_MARK)


def always_rewrite_responder(
    *, parse_fail_at_round: int | None = None
) -> Callable[[LLMRequest], str]:
    """A FakeProvider responder that rewrites at every round (never ``accept``).

    ``parse_fail_at_round = t`` makes every attempt of the decision at round ``t`` return text
    that is not the JSON schema, so that decision is a parse_fail (all three attempts fail) and
    the trajectory keeps x_t for the next round (II.8.2).
    """

    def responder(req: LLMRequest) -> str:
        user = content_text(
            next((m["content"] for m in reversed(req.messages) if m["role"] == "user"), "")
        )
        m = _X_T_RE.search(user)
        if m is None:
            raise AssertionError("stub controller: the user turn carries no <<X_T>> block")
        x_t = m.group(1)
        t = rewrites_so_far(x_t) + 1
        if parse_fail_at_round is not None and t == parse_fail_at_round:
            return "not json at all"
        return json.dumps({"action": "rewrite", "revised_text": x_t + REWRITE_MARK})

    return responder


def stub_controller_config(*, base_seed: int = 1, provider: str = "fake") -> ControllerConfig:
    """The ControllerConfig the stub answers to: fake provider, ``stub-`` model id, a template
    with a marked x_t block, per-state seeds. ``{k}`` is filled by the runner as usual."""
    return ControllerConfig(
        provider=provider,
        api=None,
        model=STUB_CONTROLLER_MODEL,
        version="stub",
        decoding={},
        base_seed=base_seed,
        seed_scheme="per_state",
        system_prompt=STUB_SYSTEM_PROMPT,
        rewrite_template=STUB_REWRITE_TEMPLATE,
        retry_prompt=STUB_RETRY_PROMPT,
        show_x0=True,
        refusal_phrases=[],
        network={},
    )


__all__ = [
    "REWRITE_MARK",
    "STUB_CONTROLLER_MODEL",
    "always_rewrite_responder",
    "is_stub_controller",
    "rewrites_so_far",
    "stub_controller_config",
]
