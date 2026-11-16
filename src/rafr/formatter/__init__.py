"""Observation formatter — the ONLY place in the code base that may branch on arm (paper App. E).

``format_observation`` is a pure function of the s^obs interface (``Observation``: s1_obs,
s2_obs, tau1, tau2), the arm, and — for permuted-B only — the donor block. It never reads
``raw_s1 / raw_s2`` (pre-channel scores), any label, E output, or text. The controller
receives the returned string and nothing else.

Invariants with tests:

* A1 contains tau1 and tau2 (paper Sec. 3).
* B = A1 + one blank line + the derived block; nothing else differs (paper App. E, Sec. 5).
* B's fields are computed by :func:`rafr.formatter.fields.compute_fields` from
  (s1_obs, s2_obs, tau1, tau2) only; conflict direction is ``N/A`` on agreement (paper Sec. 3).
* permuted-B renders the donor's whole block verbatim over the state's own A1 (paper Sec. 3).
* Rendering rules: every score, threshold and distance has ``templates.SIGNIFICANT_DIGITS`` (5)
  significant digits, as a plain decimal with its trailing zeros (``0.83412``, ``0.0031513``,
  ``0.000033283``; zero ``0.0000``); distances are signed (``+0.33412`` / ``-0.062000``) and a
  zero distance is ``+0.0000`` (2026-09-25). Flags and
  agreement ``yes`` / ``no``; checkers are ``Checker A`` / ``Checker B``, never the models'
  real names (2026-09-18).
"""

from __future__ import annotations

import math
from pathlib import Path

from rafr.contracts.schemas import Arm, Observation, UncertaintyFields
from rafr.formatter import templates as tpl
from rafr.formatter.fields import compute_fields

TEMPLATES_PATH = Path(__file__).with_name("templates.py")


def fixed_significant(x: float, digits: int = tpl.SIGNIFICANT_DIGITS) -> str:
    """``|x|`` with ``digits`` significant digits as a plain decimal, trailing zeros kept.

    The rounding is Python's own (``format(x, ".{digits-1}e")``: round-half-even on the exact
    binary value); only the notation changes, from scientific to positional. No sign: the
    callers add it. Zero has no significant digit and is written with ``digits - 1`` decimals
    (``0.0000``), as that same format writes it (2026-09-25).
    """
    if not math.isfinite(x):
        raise ValueError(f"cannot display {x!r}")
    mantissa, exponent = f"{abs(x):.{digits - 1}e}".split("e")
    body = mantissa.replace(".", "")
    point = int(exponent) + 1  # the decimal point sits after this many digits of ``body``
    if point <= 0:
        return "0." + "0" * -point + body
    if point >= len(body):
        return body + "0" * (point - len(body))
    return body[:point] + "." + body[point:]


def _number(x: float) -> str:
    if x < 0:
        raise ValueError(f"a score or threshold is never negative, got {x!r}")
    return fixed_significant(x)


def _signed(x: float) -> str:
    # A zero distance (negative zero included) reads ``+0.0000``: the checker flags at s == tau,
    # and the text says positive means at or above the threshold. A nonzero distance keeps its
    # sign however small it is, so the sign always agrees with the flag (IEEE subtraction:
    # s - tau == 0 only when s == tau).
    return ("-" if x < 0 else "+") + fixed_significant(x)


def render_a1(obs: Observation) -> str:
    return tpl.A1_BLOCK.format(
        v1_name=tpl.V1_NAME,
        v2_name=tpl.V2_NAME,
        s1=_number(obs.s1_obs),
        s2=_number(obs.s2_obs),
        tau1=_number(obs.tau1),
        tau2=_number(obs.tau2),
        v1_description=tpl.V1_DESCRIPTION,
        v2_description=tpl.V2_DESCRIPTION,
    )


def render_derived(fields: UncertaintyFields) -> str:
    """The block B appends to A1 (the six paper Sec. 3 fields plus their reading guide)."""
    return tpl.DERIVED_BLOCK.format(
        v1_name=tpl.V1_NAME,
        v2_name=tpl.V2_NAME,
        y1=tpl.FLAG_WORDS[fields.y1],
        y2=tpl.FLAG_WORDS[fields.y2],
        m1=_signed(fields.m1),
        m2=_signed(fields.m2),
        agreement=tpl.AGREEMENT_WORDS[fields.agreement],
        conflict_direction=tpl.CONFLICT_WORDS[fields.conflict_direction.value],
    )


def format_observation(
    arm: Arm, obs: Observation, *, permuted_fields: UncertaintyFields | None = None
) -> str:
    """Render the observation string for ``arm``.

    ``permuted_fields`` is required for ``Arm.PERMUTED_B`` (the donor state's block) and
    must be ``None`` for the other arms.
    """
    a1 = render_a1(obs)
    if arm is Arm.A1:
        if permuted_fields is not None:
            raise ValueError("permuted_fields only applies to permuted_B")
        return a1
    if arm is Arm.B:
        if permuted_fields is not None:
            raise ValueError("permuted_fields only applies to permuted_B")
        return a1 + tpl.B_SEPARATOR + render_derived(compute_fields(obs))
    if arm is Arm.PERMUTED_B:
        if permuted_fields is None:
            raise ValueError("permuted_B needs the donor state's fields")
        return a1 + tpl.B_SEPARATOR + render_derived(permuted_fields)
    raise ValueError(f"unknown arm {arm!r}")


__all__ = [
    "TEMPLATES_PATH",
    "compute_fields",
    "fixed_significant",
    "format_observation",
    "render_a1",
    "render_derived",
]
