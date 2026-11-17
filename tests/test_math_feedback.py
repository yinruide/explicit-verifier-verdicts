"""Math feedback preserves the information boundary and the whole donor block."""

from __future__ import annotations

import inspect
import math
import re
from pathlib import Path

import pytest

from rafr.contracts.schemas import Arm, ConflictDirection, Lambda, Observation
from rafr.core.config import NotFrozenError
from rafr.formatter import math_feedback, templates
from rafr.formatter import render_derived as core_render_derived
from rafr.formatter.fields import compute_fields
from rafr.formatter.math_feedback import format_observation, render_a1, render_derived


@pytest.fixture(autouse=True)
def frozen_test_prompts(tmp_path, monkeypatch):
    source = math_feedback.PROMPT_DIR
    (tmp_path / "a1.txt").write_text((source / "a1.txt").read_text())
    (tmp_path / "checker_a.txt").write_text("Checker A evaluates steps of a mathematical solution.")
    (tmp_path / "checker_b.txt").write_text(
        "Checker B evaluates the complete mathematical solution."
    )
    (tmp_path / "score_direction.txt").write_text(
        "For both checkers, a higher score means the checker considers the solution more "
        "likely to be wrong. The two scores are on different scales and cannot be "
        "compared directly."
    )
    monkeypatch.setattr(math_feedback, "PROMPT_DIR", tmp_path)
    return tmp_path


def obs(**changes) -> Observation:
    values = dict(
        state_id="recipient",
        s1_obs=0.834,
        s2_obs=0.538,
        tau1=0.5,
        tau2=0.6,
        lam=Lambda.LOW,
    )
    return Observation(**(values | changes))


@pytest.mark.parametrize("arm", [Arm.A1, Arm.B, Arm.PERMUTED_B])
def test_no_gold_or_candidate_arguments_and_no_unknown_arguments(arm):
    parameters = inspect.signature(format_observation).parameters
    assert not {"gold", "reference_answer", "problem", "solution", "e_label"} & parameters.keys()
    assert all(p.kind is not inspect.Parameter.VAR_KEYWORD for p in parameters.values())
    with pytest.raises(TypeError, match="unexpected keyword"):
        format_observation(arm, obs(), reference_answer="42")


def test_a1_is_exact_prefix_and_feedback_ignores_raw_scores():
    observation = obs()
    a1 = format_observation(Arm.A1, observation)
    b = format_observation(Arm.B, observation)
    assert b == a1 + "\n\n" + render_derived(compute_fields(observation))
    assert b == format_observation("B", observation)
    changed = observation.model_copy(update={"raw_s1": 0.01, "raw_s2": 0.99})
    assert format_observation(Arm.B, changed) == b
    assert "Checker A flags this text:" not in a1
    assert "Disagreement direction:" not in a1


def test_mathematical_wording_and_scale_limits():
    text = format_observation(Arm.B, obs())
    for forbidden in ("toxic", "safe", "confidence", "probability"):
        assert forbidden not in text.lower()
    assert "a higher score means the checker considers the solution more likely to be wrong" in text
    assert "different scales and cannot be compared directly" in text
    assert "Distances from different checkers are not comparable" in text
    assert "Checker A flags this text: yes" in text
    assert "Checker B flags this text: no" in text
    assert "Disagreement direction: Checker A flags, Checker B does not" in text


def test_fixed_candidate_descriptions_are_identical_between_arms(frozen_test_prompts):
    description = "Checker A uses the least-supported step score."
    (frozen_test_prompts / "checker_a.txt").write_text(description)
    a1 = format_observation("A1", obs())
    b = format_observation("B", obs())
    assert b.startswith(a1 + "\n\n")
    assert description in a1
    assert render_derived(compute_fields(obs())) == core_render_derived(compute_fields(obs()))
    assert "at or above" not in a1 and "ERROR" not in a1
    assert "flags this text" not in a1 and "reaches its threshold" not in a1


@pytest.mark.parametrize("description", ["", "   ", "first\nsecond", "first\rsecond"])
def test_checker_description_cannot_inject_extra_lines(frozen_test_prompts, description):
    (frozen_test_prompts / "checker_a.txt").write_text(description)
    with pytest.raises(ValueError, match="single-line"):
        format_observation("A1", obs())


@pytest.mark.parametrize("name", ["checker_a", "checker_b", "score_direction"])
def test_pending_or_missing_prompt_blocks_calls(frozen_test_prompts, name):
    path = frozen_test_prompts / (name + ".txt")
    path.write_text("PLACEHOLDER")
    with pytest.raises(NotFrozenError):
        format_observation("A1", obs())
    path.unlink()
    with pytest.raises(NotFrozenError):
        format_observation("A1", obs())


@pytest.mark.parametrize("word", math_feedback.FORBIDDEN_WORDS)
def test_semantics_cannot_claim_calibration(frozen_test_prompts, word):
    (frozen_test_prompts / "checker_b.txt").write_text("Checker B gives " + word)
    with pytest.raises(ValueError, match="forbidden"):
        format_observation("A1", obs())


M71_SENTENCE = (
    "Checker {x} is a language model that reads the problem and the solution, reasons about "
    "them, and then judges whether the final answer is correct; its score is its output for "
    "the answer being wrong."
)
M71_DIRECTION = (
    "For both checkers, a higher score means the checker considers the solution more likely to "
    "be wrong. The two scores are on different scales and cannot be compared directly."
)


def test_repo_descriptions_are_the_final_m71_sentences():
    """paper Sec. 3 (2026-10-12): two identical sentences, Checker A = V1, B = V2."""
    import json

    real = Path(math_feedback.__file__).with_name("prompts") / "math"
    prompts = math_feedback.load_prompts(real)
    assert prompts["checker_a"] == M71_SENTENCE.format(x="A")
    assert prompts["checker_b"] == M71_SENTENCE.format(x="B")
    assert prompts["score_direction"] == M71_DIRECTION
    rendered = format_observation("A1", obs(), prompt_dir=real)
    assert M71_SENTENCE.format(x="A") in rendered and M71_DIRECTION in rendered
    formal = Path(__file__).resolve().parents[1] / "configs" / "math" / "formal.json"
    specs = json.loads(formal.read_text(encoding="utf-8"))["verifiers"]
    math_feedback.assert_anonymous_descriptions(
        prompts, (n for s in specs for n in (s["model_id"], s["base_model"], s["vendor"]))
    )


def test_boundary_equal_is_error_with_positive_zero_margin():
    text = format_observation("B", obs(s1_obs=0.5, s2_obs=0.6))
    assert "Checker A flags this text: yes" in text
    assert "Checker B flags this text: yes" in text
    assert text.count("distance from its threshold: +0.0000") == 2
    assert "The two checkers agree: yes" in text
    assert "Disagreement direction: N/A" in text


def test_five_significant_digits_plain_decimals_and_signed_margins():
    text = format_observation("B", obs())
    assert "score 0.83400 (threshold 0.50000)" in text
    assert "score 0.53800 (threshold 0.60000)" in text
    assert "distance from its threshold: +0.33400" in text
    assert "distance from its threshold: -0.062000" in text
    small = format_observation("B", obs(s1_obs=3.3283459953388764e-05, tau1=0.003151302991566165))
    assert "score 0.000033283 (threshold 0.0031513)" in small
    assert "distance from its threshold: -0.0031180" in small
    assert not re.search(r"\d[eE][+-]\d", small)
    limits = render_a1(obs(s1_obs=0.0, s2_obs=1.0))
    assert "score 0.0000" in limits and "score 1.0000" in limits


def test_rounding_does_not_change_side_of_threshold():
    threshold = 0.5
    text = format_observation(
        "B",
        obs(s1_obs=math.nextafter(threshold, 0.0), s2_obs=threshold, tau2=threshold),
    )
    # Displayed scores can round alike, but the signed margin retains the actual side.
    assert "Checker A flags this text: no" in text
    assert "Checker A distance from its threshold: -0." in text
    assert "Checker B flags this text: yes" in text


@pytest.mark.parametrize("field", ["s1_obs", "s2_obs", "tau1", "tau2"])
@pytest.mark.parametrize("value", [math.nan, math.inf, -math.inf, -0.001, 1.001])
def test_invalid_numeric_inputs_rejected_even_after_pydantic_validation_bypass(field, value):
    invalid = obs().model_copy(update={field: value})
    with pytest.raises(ValueError, match="finite and in"):
        format_observation("B", invalid)


def test_permuted_b_uses_whole_external_donor_block_and_own_a1():
    recipient = obs()
    donor = obs(state_id="donor", s1_obs=0.1, s2_obs=0.9)
    fields = compute_fields(donor)
    text = format_observation(
        "permuted_B", recipient, permuted_fields=fields, donor_observation=donor
    )
    assert text == render_a1(recipient) + "\n\n" + render_derived(fields)
    assert "Checker A flags this text: no" in text
    assert "Checker B flags this text: yes" in text
    # Recipient's true derived fields are not accidentally shown as a second block.
    assert text.count("Checker A flags this text:") == 1
    assert text == format_observation("permuted_B", recipient, permuted_fields=fields)


@pytest.mark.parametrize("arm", ["A1", "B"])
def test_nonpermuted_arms_reject_donor_information(arm):
    with pytest.raises(ValueError, match="only allowed"):
        format_observation(arm, obs(), permuted_fields=compute_fields(obs()))
    with pytest.raises(ValueError, match="only allowed"):
        format_observation(arm, obs(), donor_observation=obs(state_id="donor"))


def test_permuted_b_requires_complete_fields():
    with pytest.raises(ValueError, match="complete donor fields"):
        format_observation("permuted_B", obs())


@pytest.mark.parametrize(
    "donor, message",
    [
        (obs(), "different state"),
        (obs(state_id="donor", lam=Lambda.HIGH), "same evidence severity"),
        (obs(state_id="donor", tau1=0.4), "same checker thresholds"),
    ],
)
def test_optional_donor_provenance_checks(donor, message):
    with pytest.raises(ValueError, match=message):
        format_observation(
            "permuted_B",
            obs(),
            permuted_fields=compute_fields(donor),
            donor_observation=donor,
        )


def test_valid_but_wrong_donor_fields_are_rejected():
    donor = obs(state_id="donor", s1_obs=0.1, s2_obs=0.9)
    with pytest.raises(ValueError, match="match the donor"):
        format_observation(
            "permuted_B",
            obs(),
            permuted_fields=compute_fields(obs()),
            donor_observation=donor,
        )


@pytest.mark.parametrize(
    "changes",
    [
        {"m1": math.nan},
        {"m2": math.inf},
        {"m1": 1.01},
        {"m1": -0.1},
        {"y1": 2},
        {"agreement": 1},
        {"conflict_direction": ConflictDirection.V1_SAFE_V2_UNSAFE},
    ],
)
def test_inconsistent_external_field_blocks_rejected(changes):
    fields = compute_fields(obs()).model_copy(update=changes)
    with pytest.raises(ValueError):
        format_observation("permuted_B", obs(), permuted_fields=fields)


def test_unknown_arm_rejected():
    with pytest.raises(ValueError):
        format_observation("unknown", obs())


def test_math_template_preserves_every_core_word_outside_three_domain_slots(frozen_test_prompts):
    expected = templates.A1_BLOCK.replace(
        "Each score runs from 0 to 1; higher means more toxic.", "{score_direction}"
    )
    assert (frozen_test_prompts / "a1.txt").read_text().strip() == expected


@pytest.mark.parametrize("field", ["checker_a", "checker_b", "score_direction"])
def test_selected_names_cannot_be_rendered_in_score_semantics(frozen_test_prompts, field):
    (frozen_test_prompts / f"{field}.txt").write_text("Checker A is based on Mistral-7B.")
    with pytest.raises(ValueError, match="must not name"):
        math_feedback.assert_anonymous_descriptions(
            math_feedback.load_prompts(), ["research", "mistral-7b", "some/model"]
        )


def test_anonymity_guard_avoids_short_generic_names_and_partial_word_matches(frozen_test_prompts):
    (frozen_test_prompts / "checker_a.txt").write_text(
        "Checker A uses one model to evaluate transformation steps."
    )
    math_feedback.assert_anonymous_descriptions(
        math_feedback.load_prompts(), ["one", "model", "form", "research"]
    )
