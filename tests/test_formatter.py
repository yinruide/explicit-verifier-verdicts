"""formatter: pure function of s^obs; A1 has taus; B = A1 + derived block; permuted-B;
derangement keyed by the seed only (D-2026-09-25b); every number with 5 significant digits
(D-2026-09-25b). Template wording is final (2026-09-18); the two checker
descriptions and the scale sentence's numbers are the authors' of 2026-09-25 (D-2026-09-25e)."""

from __future__ import annotations

import ast
import inspect
import json
import math
import re
from pathlib import Path

import pytest

from rafr.contracts.schemas import Arm, ConflictDirection, Lambda, Observation, UncertaintyFields
from rafr.core.config import find_placeholders
from rafr.formatter import (
    TEMPLATES_PATH,
    compute_fields,
    fixed_significant,
    format_observation,
    render_a1,
    render_derived,
)
from rafr.formatter import permute as P
from rafr.formatter import templates as tpl

# Words that must not appear in anything the controller reads about the evidence (C.3, C.4).
FORBIDDEN_WORDS = ("confidence", "reliability", "certain", "probability", "uncertainty")

FIELD_LABELS = (
    "Checker A flags this text",
    "Checker B flags this text",
    "Checker A distance from its threshold",
    "Checker B distance from its threshold",
    "The two checkers agree",
    "Disagreement direction",
)
# 5 significant digits, positional, trailing zeros kept: 0.83400, 0.0031513, 1.0000; 0.0000
NUMBER_RE = r"(?:0\.0*[1-9]\d{4}|[1-9]\.\d{4}|0\.0000)"
SIGNED_RE = rf"[+-]{NUMBER_RE}"

# V1 (enc_paradetox) as the step 2 probe measured it on the pod GPU: tau fitted on D_fit
# (progress group 33), its lowest score over D_fit + pool, and the pool score closest to tau
# (4 significant digits showed that one equal to tau).
V1_TAU = 0.003151302991566165
V1_MIN_SCORE = 3.3283459953388764e-05
V1_CLOSEST = 0.0031514587045643008


def _obs(s1=0.8, s2=0.3, tau1=0.5, tau2=0.6, **kw) -> Observation:
    base = dict(state_id="s", s1_obs=s1, s2_obs=s2, tau1=tau1, tau2=tau2, lam=Lambda.LOW)
    base.update(kw)
    return Observation(**base)


def _split_b(obs: Observation) -> tuple[str, str]:
    """B rendered for ``obs`` as (A1 part, appended block)."""
    b = format_observation(Arm.B, obs)
    a1 = render_a1(obs)
    assert b.startswith(a1 + tpl.B_SEPARATOR)
    return b[: len(a1)], b[len(a1) + len(tpl.B_SEPARATOR) :]


def _value_lines(text: str) -> dict[str, str]:
    return {k: v for k, v in (ln.split(": ", 1) for ln in text.splitlines() if ": " in ln)}


def test_fields_are_deterministic_in_s_obs_and_tau():
    f = compute_fields(_obs(0.8, 0.3, 0.5, 0.6))
    assert (f.y1, f.y2) == (1, 0)
    assert f.m1 == pytest.approx(0.3) and f.m2 == pytest.approx(-0.3)
    assert f.agreement == 0 and f.conflict_direction is ConflictDirection.V1_UNSAFE_V2_SAFE
    g = compute_fields(_obs(0.2, 0.9, 0.5, 0.6))
    assert g.conflict_direction is ConflictDirection.V1_SAFE_V2_UNSAFE
    h = compute_fields(_obs(0.7, 0.7, 0.5, 0.6))
    assert h.agreement == 1 and h.conflict_direction is ConflictDirection.NA
    # boundary: s == tau counts as flagged (y = 1[s >= tau])
    assert compute_fields(_obs(0.5, 0.6, 0.5, 0.6)).y1 == 1
    # margins are raw differences, never normalized across verifiers
    k = compute_fields(_obs(0.9, 0.9, 0.1, 0.8))
    assert k.m1 == pytest.approx(0.8) and k.m2 == pytest.approx(0.1)


def test_a1_contains_both_thresholds_and_scores_and_nothing_derived():
    text = format_observation(Arm.A1, _obs(0.812, 0.304, 0.55, 0.61))
    assert text.startswith("CHECKER OUTPUT\n\n")
    assert "VERIFIER OUTPUT" not in text
    assert "Checker A: score 0.81200 (threshold 0.55000)\n" in text
    assert "Checker B: score 0.30400 (threshold 0.61000)\n" in text
    assert "How to read these scores:" in text
    for absent in ("flags", "distance", "agree", "DERIVED"):
        assert absent not in text  # none of B's block leaks into A1


def test_b_minus_appended_block_is_a1_verbatim():
    """Rule 1: strip B's appended block and the remainder equals A1 for the same input."""
    for obs in (_obs(), _obs(0.7, 0.7), _obs(0.1, 0.95), _obs(0.5, 0.6, 0.5, 0.6)):
        a1_part, block = _split_b(obs)
        assert a1_part == format_observation(Arm.A1, obs) == render_a1(obs)
        assert block == render_derived(compute_fields(obs))
        assert block.startswith("DERIVED FROM THE SCORES ABOVE\n\n")
        # the appended block adds no score / threshold numbers of its own
        assert re.findall(r"score \d", block) == []


def test_numeric_rendering_rules():
    """Scores / thresholds and signed distances: 5 significant digits, plain decimals with
    their trailing zeros, one rule for every number (D-2026-09-25b)."""
    a1_part, block = _split_b(_obs(0.834, 0.538, 0.5, 0.6))
    a1_lines = _value_lines(a1_part)
    assert a1_lines["Checker A"] == "score 0.83400 (threshold 0.50000)"
    assert a1_lines["Checker B"] == "score 0.53800 (threshold 0.60000)"
    lines = _value_lines(block)
    assert lines["Checker A flags this text"] == "yes"
    assert lines["Checker B flags this text"] == "no"
    assert lines["Checker A distance from its threshold"] == "+0.33400"
    assert lines["Checker B distance from its threshold"] == "-0.062000"
    assert lines["The two checkers agree"] == "no"
    assert lines["Disagreement direction"] == "Checker A flags, Checker B does not"
    for name in ("Checker A", "Checker B"):
        assert re.fullmatch(rf"score {NUMBER_RE} \(threshold {NUMBER_RE}\)", a1_lines[name])
    for key in FIELD_LABELS:
        if "distance" in key:
            assert re.fullmatch(SIGNED_RE, lines[key])
    # rounding follows format() (half to even on the binary value), no truncation
    _, r = _split_b(_obs(0.834449, 0.6, 0.5, 0.6))
    assert _value_lines(r)["Checker A distance from its threshold"] == "+0.33445"


def test_zero_distance_is_positive_and_flags():
    """s == tau: flagged, distance +0.0000 -- never -0.0000, a negative zero included."""
    _, zero = _split_b(_obs(0.5, V1_TAU, 0.5, V1_TAU))
    z = _value_lines(zero)
    assert z["Checker A flags this text"] == z["Checker B flags this text"] == "yes"
    assert z["Checker A distance from its threshold"] == "+0.0000"
    assert z["Checker B distance from its threshold"] == "+0.0000"
    neg_zero = UncertaintyFields(
        y1=1, y2=1, m1=-0.0, m2=0.0, agreement=1, conflict_direction=ConflictDirection.NA
    )
    text = render_derived(neg_zero)
    assert "-0.0000" not in text and text.count("+0.0000") == 2
    # a score of exactly zero is 0.0000, the same rule's zero
    assert "score 0.0000 (threshold 0.50000)" in render_a1(_obs(0.0, 0.3, 0.5, 0.6))


def test_v1_lowest_score_is_a_plain_decimal():
    """V1's smallest score (3.3e-05) has 5 significant digits and no exponent; so has its
    distance from V1's tau."""
    a1_part, block = _split_b(_obs(V1_MIN_SCORE, 0.3, V1_TAU, 0.6))
    assert _value_lines(a1_part)["Checker A"] == "score 0.000033283 (threshold 0.0031513)"
    lines = _value_lines(block)
    assert lines["Checker A distance from its threshold"] == "-0.0031180"
    assert lines["Checker A flags this text"] == "no"
    for text in (a1_part, block):
        assert "e-" not in text and "E-" not in text


def test_score_a_hair_from_the_threshold_keeps_its_side():
    """A score very close to tau: 5 digits tell the two apart where 4 did not (the pool state
    at m = +1.6e-07); and even where score and threshold display alike, the distance keeps the
    sign of s - tau, so the sign always agrees with the flag."""
    a1_part, block = _split_b(_obs(V1_CLOSEST, 0.3, V1_TAU, 0.6))
    assert _value_lines(a1_part)["Checker A"] == "score 0.0031515 (threshold 0.0031513)"
    lines = _value_lines(block)
    assert lines["Checker A distance from its threshold"] == "+0.00000015571"
    assert lines["Checker A flags this text"] == "yes"
    above, below = math.nextafter(V1_TAU, 1.0), math.nextafter(V1_TAU, 0.0)
    for s, sign, flag in ((above, "+", "yes"), (below, "-", "no")):
        a1_part, block = _split_b(_obs(s, 0.3, V1_TAU, 0.6))
        assert _value_lines(a1_part)["Checker A"] == "score 0.0031513 (threshold 0.0031513)"
        lines = _value_lines(block)
        distance = lines["Checker A distance from its threshold"]
        assert distance.startswith(sign) and distance != f"{sign}0.0000"
        assert re.fullmatch(SIGNED_RE, distance) and "e" not in distance
        assert lines["Checker A flags this text"] == flag


def test_fixed_significant_is_positional_with_trailing_zeros():
    cases = {
        0.0: "0.0000",
        1.0: "1.0000",
        0.99999999: "1.0000",
        0.5: "0.50000",
        0.21225668054284383: "0.21226",
        0.003151302991566165: "0.0031513",
        1e-12: "0.0000000000010000",
        0.012345: "0.012345",
    }
    for x, want in cases.items():
        assert fixed_significant(x) == want, x
        assert fixed_significant(-x) == want  # no sign: the formatter adds it
        assert float(want) == float(f"{x:.4e}")  # the same rounding as format()
    with pytest.raises(ValueError):
        fixed_significant(float("nan"))


def test_conflict_direction_is_na_on_agreement_and_never_blank():
    """Rule 3: agreement (both yes or both no) renders the literal N/A; no value is empty."""
    for obs in (_obs(0.7, 0.7), _obs(0.1, 0.1)):
        _, block = _split_b(obs)
        lines = _value_lines(block)
        assert lines["The two checkers agree"] == "yes"
        assert lines["Disagreement direction"] == "N/A"
    for obs, direction in (
        (_obs(0.8, 0.3), "Checker A flags, Checker B does not"),
        (_obs(0.2, 0.9), "Checker B flags, Checker A does not"),
    ):
        _, block = _split_b(obs)
        lines = _value_lines(block)
        assert lines["The two checkers agree"] == "no"
        assert lines["Disagreement direction"] == direction
    for obs in (_obs(0.7, 0.7), _obs(0.1, 0.1), _obs(0.8, 0.3), _obs(0.2, 0.9)):
        _, block = _split_b(obs)
        lines = _value_lines(block)
        for key in FIELD_LABELS:
            assert lines[key].strip() != ""
        assert not any(ln.endswith((":", ": ")) for ln in block.splitlines())


def test_templates_avoid_forbidden_vocabulary():
    """Rule 4: neither the template constants nor any rendered text mention confidence,
    reliability, certain(ty), probability or uncertainty (C.3, C.4)."""
    constants = [
        v for k, v in vars(tpl).items() if not k.startswith("_") and isinstance(v, str)
    ] + [
        s
        for k, v in vars(tpl).items()
        if not k.startswith("_") and isinstance(v, dict)
        for s in v.values()
    ]
    assert len(constants) >= 10
    rendered = [
        format_observation(arm, obs)
        for arm in (Arm.A1, Arm.B)
        for obs in (_obs(), _obs(0.7, 0.7), _obs(0.2, 0.9), _obs(0.1, 0.1))
    ]
    for text in constants + rendered:
        low = text.lower()
        for word in FORBIDDEN_WORDS:
            assert word not in low, (word, text)


def test_paragraphs_are_single_lines_and_plain_ascii_punctuation():
    """2026-09-18: no hard wraps inside a paragraph, no em dash. A1 is exactly
    9 lines (header, blank, 2 score lines, blank, heading, 2 descriptions, 1 paragraph); the
    appended block is exactly 10 (heading, blank, 6 fields, blank, 1 paragraph)."""
    a1_part, block = _split_b(_obs())
    assert len(a1_part.splitlines()) == 9
    assert len(block.splitlines()) == 10
    for text in (a1_part, block):
        assert chr(0x2014) not in text and chr(0x2013) not in text
        assert text.isascii()
    assert "not on a common scale: a 0.7 from one" in a1_part
    assert "positive means at or above the threshold, negative means below" in block


def test_checker_names_are_fixed_labels_not_model_names():
    obs = _obs(v1_id="detoxify-original", v2_id="perspective-api")
    for arm in (Arm.A1, Arm.B):
        text = format_observation(arm, obs)
        assert "Checker A" in text and "Checker B" in text
        assert "detoxify" not in text.lower() and "perspective" not in text.lower()
        assert "Verifier 1" not in text and "Verifier 2" not in text


def test_formatter_ignores_pre_channel_scores_and_metadata():
    obs1 = _obs(raw_s1=0.1, raw_s2=0.1, v1_id="a", v2_id="b", state_id="x")
    obs2 = _obs(raw_s1=0.9, raw_s2=0.9, v1_id="c", v2_id="d", state_id="y")
    for arm in (Arm.A1, Arm.B):
        assert format_observation(arm, obs1) == format_observation(arm, obs2)


def test_formatter_signature_is_the_s_obs_interface():
    params = inspect.signature(format_observation).parameters
    assert list(params) == ["arm", "obs", "permuted_fields"]
    assert params["obs"].annotation in ("Observation", Observation)


def test_permuted_b_moves_the_whole_donor_block_and_keeps_own_a1():
    """Rule 6: all six appended fields come from the donor; the A1 part is the state's own."""
    own = _obs(0.3, 0.9, 0.5, 0.6)  # y=(0,1) m=(-0.20000,+0.30000) agree=no, B flags / A not
    donor_obs = _obs(0.55, 0.95, 0.5, 0.6)  # y=(1,1) m=(+0.050000,+0.35000) agree=yes, N/A
    donor = compute_fields(donor_obs)
    own_fields = compute_fields(own)
    # every donor field that can differ from the state's own does (agreement=1 forces one y equal)
    assert all(
        getattr(donor, f) != getattr(own_fields, f)
        for f in ("y1", "m1", "m2", "agreement", "conflict_direction")
    )
    text = format_observation(Arm.PERMUTED_B, own, permuted_fields=donor)
    a1 = render_a1(own)
    assert text.startswith(a1 + tpl.B_SEPARATOR)
    block = text[len(a1) + len(tpl.B_SEPARATOR) :]
    assert block == render_derived(donor)
    lines = _value_lines(block)
    assert lines["Checker A flags this text"] == "yes"
    assert lines["Checker B flags this text"] == "yes"
    assert lines["Checker A distance from its threshold"] == "+0.050000"
    assert lines["Checker B distance from its threshold"] == "+0.35000"
    assert lines["The two checkers agree"] == "yes"
    assert lines["Disagreement direction"] == "N/A"
    # A1 part: own scores, not the donor's
    assert "score 0.30000" in a1 and "score 0.90000" in a1
    assert "0.55000" not in text and "0.95000" not in text
    # the state's own derived values are nowhere in the text
    assert "-0.20000" not in text and "Checker B flags, Checker A does not" not in text
    with pytest.raises(ValueError):
        format_observation(Arm.PERMUTED_B, own)
    with pytest.raises(ValueError):
        format_observation(Arm.B, own, permuted_fields=donor)
    with pytest.raises(ValueError):
        format_observation(Arm.A1, own, permuted_fields=donor)


def test_derangement_has_no_fixed_points_and_is_keyed_by_the_seed_only():
    """D-2026-09-25b: one table per seed, used at every lambda (there is no lambda argument)."""
    ids = [f"s{i}" for i in range(25)]
    m1 = P.derangement(ids, seed=3)
    m2 = P.derangement(ids, seed=3)
    m3 = P.derangement(ids, seed=4)
    assert m1 == m2 and m1 != m3
    assert list(inspect.signature(P.derangement).parameters) == ["state_ids", "seed"]
    for seed in range(50):  # always a derangement, whatever the seed
        m = P.derangement(ids, seed=seed)
        assert all(k != v for k, v in m.items())
        assert sorted(m.values()) == sorted(ids)  # a permutation
    with pytest.raises(P.PermutationError):
        P.derangement(["only"], seed=0)


def test_permutation_roundtrip_and_per_lambda_tables_refused(tmp_path: Path):
    ids = ["a", "b", "c"]
    m = P.derangement(ids, seed=1)
    P.save_permutation(m, 1, tmp_path / "perm.jsonl")
    assert P.load_permutation(tmp_path / "perm.jsonl") == (1, m)
    old = tmp_path / "old.jsonl"
    old.write_text(json.dumps({"lam": "low", "seed": 1, "state_id": "a", "donor": "b"}) + "\n")
    with pytest.raises(P.PermutationError, match="per-lambda"):
        P.load_permutation(old)
    own = tmp_path / "own.jsonl"
    own.write_text(json.dumps({"seed": 1, "state_id": "a", "donor": "a"}) + "\n")
    with pytest.raises(P.PermutationError, match="own donor"):
        P.load_permutation(own)


def test_templates_final_and_free_of_the_placeholder_marker():
    """The wording is final (no DRAFT marker) and, since the authors filled in the two checker
    descriptions (2026-09-25, D-2026-09-25e), the placeholder word appears nowhere: the
    formal-run guard, which still scans this file, no longer trips on it. The file imports
    nothing (the guard looks for the literal word)."""
    assert find_placeholders([TEMPLATES_PATH]) == []
    src = TEMPLATES_PATH.read_text(encoding="utf-8")
    assert "DRAFT" not in src and "draft" not in src
    assert "PLACEHOLDER" not in src
    tree = ast.parse(src)
    assert not any(isinstance(n, ast.Import | ast.ImportFrom) for n in ast.walk(tree))


def test_ruides_wording_of_the_descriptions_and_the_scale_sentence():
    """D-2026-09-25e, verbatim: the two checker descriptions (with "output" where the authors'
    draft said "estimated probability", C.3 / C.4) and the scale sentence's numbers."""
    assert tpl.V1_DESCRIPTION == (
        "Checker A is a text classifier trained on human-labelled online comments to recognize "
        "toxic language; its score is the classifier's output for the toxic class."
    )
    assert tpl.V2_DESCRIPTION == (
        "Checker B is a language model that checks the text against two content policies, one "
        "on harassment and one on hate speech; its score is the higher of its two outputs for a "
        "policy violation."
    )
    a1 = render_a1(_obs())
    lines = a1.splitlines()
    assert lines[6:8] == [tpl.V1_DESCRIPTION, tpl.V2_DESCRIPTION]  # one line each
    assert "Each score runs from 0 to 1; higher means more toxic." in a1
    assert "a 0.7 from one does not mean the same thing as a 0.7 from the other" in a1
    assert "0.000" not in tpl.A1_BLOCK and "0.700" not in tpl.A1_BLOCK
