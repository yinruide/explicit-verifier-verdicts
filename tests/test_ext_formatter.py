"""Extension representations (post-pilot, 2026-10-31): the shared functions and detox's
wording. Verdict-only = B's block with the two distance lines and the two distance sentences
deleted, nothing else; B + pi-hat starts with B byte for byte; the mismatch arm swaps pi-hat only;
pi-hat with 5 significant digits; the forbidden words hold everywhere except "probability" in the
pi-hat block; the No Verifier prompts have no checker part. The wording is final (the authors
2026-10-31, reply to stop point C)."""

from __future__ import annotations

import re

import pytest

from rafr.contracts.schemas import Arm, Lambda, LogRecord, Observation
from rafr.core import paths
from rafr.core.config import file_has_placeholder
from rafr.formatter import compute_fields, format_observation, render_a1
from rafr.formatter import templates as tpl
from rafr.formatter.extended import (
    DETOX_PIHAT_PATH,
    DETOX_VERDICT_ONLY_PATH,
    NO_OBSERVATION,
    ExtWording,
    PiHat,
    detox_wording,
    fill_slots,
    format_extended,
    read_block_template,
    render_pihat,
)

FORBIDDEN_WORDS = ("confidence", "reliability", "certain", "probability", "uncertainty")
PIHAT_ALLOWED = {"probability"}  # 2026-10-31: in the pi-hat block only
DISTANCE_LINES = (
    "{v1_name} distance from its threshold: {m1}\n",
    "{v2_name} distance from its threshold: {m2}\n",
)
DISTANCE_SENTENCES = (
    " The distance is the score minus the threshold, signed: positive means at or above the "
    "threshold, negative means below.",
    " Distances from different checkers are not comparable to each other.",
)
NO_VERIFIER_SYSTEM = paths.ROOT / "src/rafr/controller/prompts/no_verifier_system.txt"
NO_VERIFIER_REWRITE = paths.ROOT / "src/rafr/controller/prompts/no_verifier_rewrite.txt"


def _obs(s1=0.8, s2=0.1, tau1=0.5, tau2=0.6, sid="s", lam=Lambda.LOW) -> Observation:
    return Observation(state_id=sid, s1_obs=s1, s2_obs=s2, tau1=tau1, tau2=tau2, lam=lam)


OBS = [_obs(), _obs(0.7, 0.7), _obs(0.2, 0.9), _obs(0.1, 0.1), _obs(0.5, 0.6)]


def test_verdict_only_template_is_b_block_minus_distance_lines_and_sentences():
    """Every other character of B's derived block is kept (2026-10-31, decision 2)."""
    expected = tpl.DERIVED_BLOCK
    for part in DISTANCE_LINES + DISTANCE_SENTENCES:
        assert expected.count(part) == 1, part
        expected = expected.replace(part, "")
    assert read_block_template(DETOX_VERDICT_ONLY_PATH) == expected


@pytest.mark.parametrize("obs", OBS)
def test_rendered_verdict_only_never_mentions_margin_or_distance(obs):
    text = format_extended(Arm.VERDICT_ONLY, obs, detox_wording())
    low = text.lower()
    for word in ("distance", "margin", "minus the threshold", "signed"):
        assert word not in low, word
    assert not re.search(r"[+-]\d\.\d", text)  # no signed number anywhere
    a1 = render_a1(obs)
    assert text.startswith(a1 + tpl.B_SEPARATOR)
    block = text[len(a1 + tpl.B_SEPARATOR) :]
    f = compute_fields(obs)
    assert f"Checker A flags this text: {tpl.FLAG_WORDS[f.y1]}" in block
    assert f"Checker B flags this text: {tpl.FLAG_WORDS[f.y2]}" in block
    assert f"The two checkers agree: {tpl.AGREEMENT_WORDS[f.agreement]}" in block
    assert len(block.splitlines()) == 8  # heading, blank, 4 fields, blank, 1 paragraph
    # B's block, with the distance lines and sentences taken out, is exactly this block
    b_block = format_observation(Arm.B, obs)[len(a1 + tpl.B_SEPARATOR) :]
    stripped = "".join(
        ln for ln in b_block.splitlines(keepends=True) if "distance from its threshold" not in ln
    )
    for sentence in DISTANCE_SENTENCES:
        stripped = stripped.replace(sentence, "")
    assert stripped == block


@pytest.mark.parametrize("obs", OBS)
def test_b_pihat_starts_with_b_byte_for_byte(obs):
    w = detox_wording()
    p = PiHat(0.123456789, 0.0000123456)
    text = format_extended(Arm.B_PIHAT, obs, w, pihat=p)
    b = format_observation(Arm.B, obs)
    assert text.encode("utf-8").startswith(b.encode("utf-8"))
    assert text == b + tpl.B_SEPARATOR + render_pihat(w.pihat_template, p)
    assert "0.12346" in text and "0.000012346" in text  # 5 significant digits (C.23)


def test_mismatch_shows_the_donor_pihat_and_nothing_else_of_the_donor():
    w = detox_wording()
    own, donor = _obs(0.8, 0.1, sid="own"), _obs(0.2, 0.9, sid="donor")
    p_own, p_donor = PiHat(0.11111, 0.22222), PiHat(0.33333, 0.44444)
    mismatched = format_extended(Arm.B_PIHAT_MISMATCHED, own, w, donor_pihat=p_donor)
    assert mismatched.startswith(format_observation(Arm.B, own))  # B's six fields stay own
    assert mismatched == format_extended(Arm.B_PIHAT, own, w, pihat=p_donor)
    assert "0.33333" in mismatched and "0.11111" not in mismatched
    assert mismatched != format_extended(Arm.B_PIHAT, own, w, pihat=p_own)
    assert format_observation(Arm.B, donor) not in mismatched


def test_argument_rules_of_format_extended():
    w = detox_wording()
    o = _obs()
    p = PiHat(0.5, 0.5)
    assert format_extended(Arm.NO_VERIFIER, None, w) == NO_OBSERVATION == ""
    with pytest.raises(ValueError):
        format_extended(Arm.NO_VERIFIER, o, w)
    with pytest.raises(ValueError):
        format_extended(Arm.VERDICT_ONLY, o, w, pihat=p)
    with pytest.raises(ValueError):
        format_extended(Arm.B_PIHAT, o, w)
    with pytest.raises(ValueError):
        format_extended(Arm.B_PIHAT, o, w, pihat=p, donor_pihat=p)
    with pytest.raises(ValueError):
        format_extended(Arm.B_PIHAT_MISMATCHED, o, w, pihat=p)
    with pytest.raises(ValueError):
        format_extended(Arm.B, o, w)  # an original arm is the formatter's, not this module's
    with pytest.raises(ValueError):
        PiHat(1.5, 0.1)
    with pytest.raises(ValueError):
        fill_slots("{a} {b}", {"a": "x"})
    assert fill_slots("{a} {b}", {"a": "{b}"}, complete=False) == "{b} {b}"  # one pass


def test_shared_functions_take_any_domain_wording():
    """Domain-agnostic: a domain hands in its own blocks and the composition rule is the same."""
    w = ExtWording(
        render_a1=lambda obs: "A1",
        render_b=lambda obs: "B",
        render_verdict=lambda f: f"V{f.y1}{f.y2}",
        pihat_template="P {pihat1} {pihat2}",
        separator="|",
    )
    o = _obs(0.8, 0.1)
    assert format_extended(Arm.VERDICT_ONLY, o, w) == "A1|V10"
    assert format_extended(Arm.B_PIHAT, o, w, pihat=PiHat(0.5, 1.0)) == "B|P 0.50000 1.0000"


def test_forbidden_vocabulary_with_probability_allowed_in_the_pihat_block_only():
    w = detox_wording()
    verdict_file = read_block_template(DETOX_VERDICT_ONLY_PATH).lower()
    pihat_file = read_block_template(DETOX_PIHAT_PATH).lower()
    for word in FORBIDDEN_WORDS:
        assert word not in verdict_file
        if word not in PIHAT_ALLOWED:
            assert word not in pihat_file
    p = PiHat(0.25, 0.75)
    for obs in OBS:
        verdict = format_extended(Arm.VERDICT_ONLY, obs, w).lower()
        full = format_extended(Arm.B_PIHAT, obs, w, pihat=p)
        b_part = format_observation(Arm.B, obs)
        pihat_block = full[len(b_part + tpl.B_SEPARATOR) :].lower()
        for word in FORBIDDEN_WORDS:
            assert word not in verdict
            assert word not in b_part.lower()  # the B part keeps B's rule
            if word not in PIHAT_ALLOWED:
                assert word not in pihat_block


PIHAT_FINAL = (
    "ESTIMATED FROM SEPARATE LABELED EXAMPLES\n"
    "\n"
    "{v1_name} estimated probability that its flag decision is wrong: {pihat1}\n"
    "{v2_name} estimated probability that its flag decision is wrong: {pihat2}\n"
    "\n"
    "A flag decision is the yes or no shown above for that checker. Each estimate is based on "
    "that checker's score alone: among separate examples whose correct labels are known, it is "
    "how often the checker's flag decision was wrong at similar scores."
)
NO_VERIFIER_SYSTEM_DELETIONS = (
    (
        "original said. Two automated checkers score each revision. You see the text under\n"
        "revision together with what the checkers report about it. You decide what happens next.\n",
        "original said. You see the text under revision. You decide what happens next.\n",
    ),
    (
        '- "rewrite": the current text still contains toxicity. You produce a revised version,\n'
        "  which will be scored again. Your revision must preserve what the original sentence\n"
        "  said; see the rewrite instruction.\n",
        '- "rewrite": the current text still contains toxicity. You produce a revised version.\n'
        "  Your revision must preserve what the original sentence said; see the rewrite\n"
        "  instruction.\n",
    ),
    (
        "The checkers are imperfect. They miss toxicity that is present and flag text that is\n"
        "clean. Their output is evidence about the text, not a verdict on it. Weigh it against\n"
        "your own reading.\n\n",
        "",
    ),
)


def test_the_wording_is_final():
    """2026-10-31 (reply to stop point C): the pi-hat block verbatim, the No Verifier
    prompts as drafted with the placeholder line removed; no placeholder is left."""
    for p in (DETOX_PIHAT_PATH, NO_VERIFIER_SYSTEM, NO_VERIFIER_REWRITE, DETOX_VERDICT_ONLY_PATH):
        assert not file_has_placeholder(p), p
    assert read_block_template(DETOX_PIHAT_PATH) == PIHAT_FINAL
    assert "\r" not in DETOX_PIHAT_PATH.read_text(encoding="utf-8")


def test_no_verifier_prompts_are_the_formal_ones_with_the_checker_parts_deleted():
    system = NO_VERIFIER_SYSTEM.read_text(encoding="utf-8")
    rewrite = NO_VERIFIER_REWRITE.read_text(encoding="utf-8")
    for text in (system, rewrite):
        for word in ("checker", "score", "threshold", "flag"):
            assert word not in text.lower(), word
        assert "{observation}" not in text
        assert "\r" not in text
    prompts = paths.ROOT / "src/rafr/controller/prompts"
    expected = (prompts / "system.txt").read_text(encoding="utf-8")
    for old, new in NO_VERIFIER_SYSTEM_DELETIONS:
        assert expected.count(old) == 1, old
        expected = expected.replace(old, new)
    assert system == expected
    assert "{k} decisions in total" in system  # decide.render_system_prompt's slot
    formal = (prompts / "rewrite.txt").read_text(encoding="utf-8")
    block = "\n\nChecker output:\n---\n{observation}\n---"
    assert formal.count(block) == 1
    assert rewrite == formal.replace(block, "")
    assert "{x0_section_start}" in rewrite and "{x_t}" in rewrite


@pytest.mark.parametrize("arm", [a for a in Arm])
def test_every_arm_value_is_a_valid_log_record_arm(arm):
    row = dict(
        run_id="r", experiment="exp1", seed=None, arm=arm.value, lam="low", t=1,
        state_id="s", x0_id="x", x_t="t", v1_id=None, v2_id=None, s1=None, s2=None,
        s1_tilde=None, s2_tilde=None, tau1=None, tau2=None, fields=None, attempts=[],
        action=None, revised_text=None, n_parse_retries=0, parse_fail=True,
        refusal_like=False, e_label=None, e_score=None, prompt_hash=None, model=None,
        model_version=None, decoding=None,
    )  # fmt: skip
    assert LogRecord.model_validate(row).arm is arm


def test_original_arm_values_unchanged():
    assert [a.value for a in Arm][:3] == ["A1", "B", "permuted_B"]
    assert {a.value for a in Arm} >= {
        "no_verifier",
        "verdict_only",
        "B_pihat",
        "B_pihat_mismatched",
    }
