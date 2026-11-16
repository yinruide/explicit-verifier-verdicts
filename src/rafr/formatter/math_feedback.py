"""Math feedback derived solely from observed scores and frozen checker descriptions.

No question, candidate solution, reference answer or evaluator label is accepted here.
The shared numerical field calculation is reused, while its internal label names are
rendered by the unchanged detox six-field renderer (paper Sec. 5, App. D).
"""

from __future__ import annotations

import math
import re
from collections.abc import Iterable
from pathlib import Path

from rafr.contracts.schemas import Arm, ConflictDirection, Observation, UncertaintyFields
from rafr.core.config import assert_frozen
from rafr.formatter import fixed_significant
from rafr.formatter import render_derived as core_render_derived
from rafr.formatter import templates as tpl
from rafr.formatter.fields import compute_fields

SIGNIFICANT_DIGITS = tpl.SIGNIFICANT_DIGITS
SEPARATOR = tpl.B_SEPARATOR
PROMPT_DIR = Path(__file__).with_name("prompts") / "math"
PROMPT_FILES = ("a1.txt", "checker_a.txt", "checker_b.txt", "score_direction.txt")
FORBIDDEN_WORDS = ("confidence", "reliability", "certain", "probability", "uncertainty")


def load_prompts(prompt_dir: Path | None = None) -> dict[str, str]:
    """Only the authors' finalized, file-backed descriptions may be used (paper Sec. 3)."""
    prompt_dir = PROMPT_DIR if prompt_dir is None else Path(prompt_dir)
    files = [prompt_dir / name for name in PROMPT_FILES]
    assert_frozen({}, extra_files=files, only_sections=[])
    values = {path.stem: path.read_text(encoding="utf-8").strip() for path in files}
    for key in ("checker_a", "checker_b", "score_direction"):
        value = _description(values[key], key)
        if any(word in value.casefold() for word in FORBIDDEN_WORDS):
            raise ValueError(f"{key} contains forbidden score-semantics wording")
    return values


def assert_anonymous_descriptions(prompts: dict[str, str], model_names: Iterable[str]) -> None:
    """Reject explicit selected identities without matching generic short prose words."""
    generic = {"model", "math", "prm", "orm", "llm", "test", "none", "unknown", "research"}
    text = "\n".join(prompts[key] for key in ("checker_a", "checker_b", "score_direction"))
    for name in model_names:
        name = name.strip()
        if len(name) < 4 or name.casefold() in generic:
            continue
        if re.search(r"(?<!\w)" + re.escape(name) + r"(?!\w)", text, flags=re.IGNORECASE):
            raise ValueError("M.7 checker descriptions must not name the selected model or source")


def _bounded(value: float, lo: float, hi: float, name: str) -> None:
    # Recheck even an Observation made with model_construct/model_copy, which bypass
    # Pydantic's normal field validation. Raw, pre-channel values are never inspected.
    try:
        valid = math.isfinite(value) and lo <= value <= hi
    except TypeError:
        valid = False
    if not valid:
        raise ValueError(f"{name} must be finite and in [{lo}, {hi}]")


def _validate_observation(obs: Observation) -> None:
    for name in ("s1_obs", "s2_obs", "tau1", "tau2"):
        _bounded(getattr(obs, name), 0.0, 1.0, name)


def _description(value: str, name: str) -> str:
    if not isinstance(value, str) or not value.strip() or "\n" in value or "\r" in value:
        raise ValueError(f"{name} must be a nonempty, single-line checker description")
    return value.strip()


def _number(value: float) -> str:
    return fixed_significant(value, digits=SIGNIFICANT_DIGITS)


def _validate_fields(fields: UncertaintyFields) -> None:
    for name in ("m1", "m2"):
        _bounded(getattr(fields, name), -1.0, 1.0, name)
    if fields.y1 not in (0, 1) or fields.y2 not in (0, 1):
        raise ValueError("checker judgments must be binary")
    if fields.y1 != int(fields.m1 >= 0) or fields.y2 != int(fields.m2 >= 0):
        raise ValueError("checker judgments must match the signs of their margins")
    agreement = int(fields.y1 == fields.y2)
    if fields.agreement != agreement:
        raise ValueError("agreement must match both checker judgments")
    expected = (
        ConflictDirection.NA
        if agreement
        else (
            ConflictDirection.V1_UNSAFE_V2_SAFE
            if fields.y1
            else ConflictDirection.V1_SAFE_V2_UNSAFE
        )
    )
    if fields.conflict_direction != expected:
        raise ValueError("conflict direction must match both checker judgments")


def render_a1(obs: Observation, *, prompt_dir: Path | None = None) -> str:
    """Detox layout; only the three domain-description paragraphs are substituted."""
    _validate_observation(obs)
    prompts = load_prompts(prompt_dir)
    return prompts["a1"].format(
        v1_name=tpl.V1_NAME,
        v2_name=tpl.V2_NAME,
        s1=_number(obs.s1_obs),
        s2=_number(obs.s2_obs),
        tau1=_number(obs.tau1),
        tau2=_number(obs.tau2),
        v1_description=prompts["checker_a"],
        v2_description=prompts["checker_b"],
        score_direction=prompts["score_direction"],
    )


def render_derived(fields: UncertaintyFields) -> str:
    """Validate a whole donor block, then use the unchanged detox renderer."""
    _validate_fields(fields)
    return core_render_derived(fields)


def format_observation(
    arm: Arm | str,
    obs: Observation,
    *,
    prompt_dir: Path | None = None,
    permuted_fields: UncertaintyFields | None = None,
    donor_observation: Observation | None = None,
) -> str:
    """A1, B, or A1 plus a caller-provided whole donor block for permuted_B.

    The caller owns the frozen permutation. This function never samples or shuffles.
    Supplying the donor observation additionally verifies that the fields are exactly its
    derived fields, at the same severity and thresholds, and from a different state.
    The file-backed descriptions must be frozen across arms and states (paper Sec. 3).
    """
    arm = Arm(arm)
    if arm is not Arm.PERMUTED_B and (permuted_fields is not None or donor_observation is not None):
        raise ValueError("donor information is only allowed for permuted_B")
    a1 = render_a1(obs, prompt_dir=prompt_dir)
    if arm is Arm.A1:
        return a1
    if arm is Arm.B:
        return a1 + SEPARATOR + render_derived(compute_fields(obs))
    if permuted_fields is None:
        raise ValueError("permuted_B requires the complete donor fields")
    _validate_fields(permuted_fields)
    if donor_observation is not None:
        _validate_observation(donor_observation)
        if donor_observation.state_id == obs.state_id:
            raise ValueError("a permuted_B donor must be a different state")
        if donor_observation.lam != obs.lam:
            raise ValueError("a donor must have the same evidence severity")
        if (donor_observation.tau1, donor_observation.tau2) != (obs.tau1, obs.tau2):
            raise ValueError("a donor must use the same checker thresholds")
        if permuted_fields != compute_fields(donor_observation):
            raise ValueError("the complete field block must match the donor observation")
    return a1 + SEPARATOR + render_derived(permuted_fields)


__all__ = ["format_observation", "render_a1", "render_derived", "load_prompts", "PROMPT_DIR"]
