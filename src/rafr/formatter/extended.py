"""Extension representations (post-pilot, 2026-10-31): shared and domain-agnostic.

Four arms join A1 / B / permuted-B. They are never part of the primary runs; their own
runners are ``rafr.experiments.exp1_ext`` and ``exp2_ext``. Each is assembled from blocks the
domain already renders, so nothing in the shared functions knows a domain's wording:

* ``no_verifier``: the controller sees x0 and x_t and no verifier output at all. The observation
  is the empty string, and the runner gives the controller its No Verifier prompts, which have no
  observation slot.
* ``verdict_only``: A1 + separator + the domain's verdict block -- B's derived block with the
  two distances (m1, m2) removed; y1, y2, agreement and the conflict direction stay.
* ``B_pihat``: B's text, unchanged byte for byte, + separator + the domain's pi-hat block
  (pi-hat_1, pi-hat_2 written with :func:`fixed_significant`'s 5 significant digits).
* ``B_pihat_mismatched``: the same, with ONLY pi-hat_1 and pi-hat_2 taken from the donor state
  (its pi-hat at the current lambda); B's six fields and scores stay this state's.

pi-hat is computed off the decision path (``rafr.calibration.apply``, from s_obs only) and
reaches this module as two numbers. A domain wires in its wording through :class:`ExtWording`;
detox's is :func:`detox_wording`. Its two text files live in ``prompts/detox/``: the verdict
block is ``templates.DERIVED_BLOCK`` with the two distance lines and the two distance sentences
deleted, nothing else changed (2026-10-31); the pi-hat block still carries the
placeholder marker, so every formal extension step refuses to run until it is written.
"""

from __future__ import annotations

import math
import re
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path

from rafr.contracts.schemas import Arm, Observation, UncertaintyFields
from rafr.formatter import fixed_significant, format_observation, render_a1
from rafr.formatter import templates as tpl
from rafr.formatter.fields import compute_fields

EXTENDED_ARMS: tuple[Arm, ...] = (
    Arm.NO_VERIFIER,
    Arm.VERDICT_ONLY,
    Arm.B_PIHAT,
    Arm.B_PIHAT_MISMATCHED,
)
PIHAT_ARMS: tuple[Arm, ...] = (Arm.B_PIHAT, Arm.B_PIHAT_MISMATCHED)
NO_OBSERVATION = ""
"""What the No Verifier controller is given in place of the checker output."""

DETOX_PROMPT_DIR = Path(__file__).with_name("prompts") / "detox"
DETOX_VERDICT_ONLY_PATH = DETOX_PROMPT_DIR / "verdict_only.txt"
DETOX_PIHAT_PATH = DETOX_PROMPT_DIR / "pihat.txt"

_SLOT_RE = re.compile(r"\{([A-Za-z_][A-Za-z0-9_]*)\}")


@dataclass(frozen=True)
class PiHat:
    """pi-hat_1, pi-hat_2 of one (state, lambda): the estimated chance each verdict is wrong."""

    pihat1: float
    pihat2: float

    def __post_init__(self) -> None:
        for name in ("pihat1", "pihat2"):
            v = getattr(self, name)
            if not (isinstance(v, float | int) and math.isfinite(v) and 0.0 <= v <= 1.0):
                raise ValueError(f"{name} must be finite and in [0, 1], got {v!r}")


@dataclass(frozen=True)
class ExtWording:
    """Everything a domain supplies. ``render_b`` must be the domain's own B arm, so that the
    B + pi-hat text starts with B's text byte for byte; ``pihat_template`` has the two slots
    ``{pihat1}`` and ``{pihat2}`` and nothing else left to fill."""

    render_a1: Callable[[Observation], str]
    render_b: Callable[[Observation], str]
    render_verdict: Callable[[UncertaintyFields], str]
    pihat_template: str
    separator: str


def fill_slots(template: str, values: Mapping[str, str], *, complete: bool = True) -> str:
    """Fill ``{name}`` slots in one pass (a value containing a slot name is copied verbatim).

    With ``complete`` a slot left unfilled is an error rather than a literal brace pair in the
    controller's prompt.
    """

    def sub(m: re.Match[str]) -> str:
        return values[m.group(1)] if m.group(1) in values else m.group(0)

    out = _SLOT_RE.sub(sub, template)
    if complete:
        left = sorted({m.group(1) for m in _SLOT_RE.finditer(template)} - set(values))
        if left:
            raise ValueError(f"template slots left unfilled: {left}")
    return out


def render_pihat(template: str, pihat: PiHat) -> str:
    return fill_slots(
        template,
        {"pihat1": fixed_significant(pihat.pihat1), "pihat2": fixed_significant(pihat.pihat2)},
    )


def format_extended(
    arm: Arm,
    obs: Observation | None,
    wording: ExtWording,
    *,
    pihat: PiHat | None = None,
    donor_pihat: PiHat | None = None,
) -> str:
    """The observation string for one of the four extension arms.

    ``pihat`` is this state's (``B_pihat`` only); ``donor_pihat`` is the donor state's at the
    same lambda (``B_pihat_mismatched`` only). ``obs`` is None for ``no_verifier`` alone.
    """
    if arm is Arm.NO_VERIFIER:
        if obs is not None or pihat is not None or donor_pihat is not None:
            raise ValueError("no_verifier is shown no verifier output")
        return NO_OBSERVATION
    if obs is None:
        raise ValueError(f"{arm.value} needs the observation")
    if arm is Arm.VERDICT_ONLY:
        if pihat is not None or donor_pihat is not None:
            raise ValueError("verdict_only shows no pi-hat")
        return (
            wording.render_a1(obs) + wording.separator + wording.render_verdict(compute_fields(obs))
        )
    if arm is Arm.B_PIHAT:
        if pihat is None or donor_pihat is not None:
            raise ValueError("B_pihat needs this state's pi-hat and no donor's")
        shown = pihat
    elif arm is Arm.B_PIHAT_MISMATCHED:
        if donor_pihat is None or pihat is not None:
            raise ValueError("B_pihat_mismatched needs the donor's pi-hat and not its own")
        shown = donor_pihat
    else:
        raise ValueError(f"{arm!r} is not an extension arm")
    return wording.render_b(obs) + wording.separator + render_pihat(wording.pihat_template, shown)


def read_block_template(path: Path) -> str:
    """A block template file as the controller will see it: LF line ends, and the file's final
    newline dropped (a block, like ``templates.DERIVED_BLOCK``, ends with its last character)."""
    text = Path(path).read_text(encoding="utf-8").replace("\r\n", "\n")
    return text.removesuffix("\n")


def detox_wording(
    *, verdict_path: Path = DETOX_VERDICT_ONLY_PATH, pihat_path: Path = DETOX_PIHAT_PATH
) -> ExtWording:
    """Detox's wording: its A1 and B unchanged, plus its two extension blocks."""
    verdict = read_block_template(verdict_path)
    names = {"v1_name": tpl.V1_NAME, "v2_name": tpl.V2_NAME}

    def render_verdict(fields: UncertaintyFields) -> str:
        return fill_slots(
            verdict,
            {
                **names,
                "y1": tpl.FLAG_WORDS[fields.y1],
                "y2": tpl.FLAG_WORDS[fields.y2],
                "agreement": tpl.AGREEMENT_WORDS[fields.agreement],
                "conflict_direction": tpl.CONFLICT_WORDS[fields.conflict_direction.value],
            },
        )

    def render_b(obs: Observation) -> str:
        return format_observation(Arm.B, obs)

    return ExtWording(
        render_a1=render_a1,
        render_b=render_b,
        render_verdict=render_verdict,
        pihat_template=fill_slots(read_block_template(pihat_path), names, complete=False),
        separator=tpl.B_SEPARATOR,
    )


__all__ = [
    "DETOX_PIHAT_PATH",
    "DETOX_VERDICT_ONLY_PATH",
    "EXTENDED_ARMS",
    "NO_OBSERVATION",
    "PIHAT_ARMS",
    "ExtWording",
    "PiHat",
    "detox_wording",
    "fill_slots",
    "format_extended",
    "read_block_template",
    "render_pihat",
]
