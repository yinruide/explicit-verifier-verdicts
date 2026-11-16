# Formatter template strings (II.7). Wording final (2026-09-18); the two checker
# descriptions and the number wording of A1_BLOCK by the authors, 2026-09-25 (decisions.md
# D-2026-09-25e).
#
# Every string the controller sees about the verifier evidence lives in this file, so that
# "A1 and B differ only in the six derived fields" (II.1.4, I.6) can be checked by reading
# one file. Formal steps refuse to run while the prompt files' placeholder marker word is
# present anywhere in this file (rafr.core.config.assert_frozen with extra_files=[templates
# path]); the two checker descriptions carried it until 2026-09-25.
#
# Do NOT import anything from rafr.contracts here: the formal-run check looks for the literal
# marker word, and importing the constant would defeat it. Do NOT spell the marker word out
# in comments either: the guard would keep tripping after the descriptions are filled in.
#
# Layout rules (2026-09-18): one paragraph = one line, no hard wraps inside a
# paragraph; plain ASCII punctuation (no em dash).

"""Formatter template strings (II.7). English only; numbers are rendered by the formatter."""

# --- checker names: fixed labels as shown to the controller, never the models' real names ---
V1_NAME = "Checker A"
V2_NAME = "Checker B"

# --- one line each describing V1 / V2 (2026-09-25, D-2026-09-25e) ----------------------
# V1 = enc_paradetox (RoBERTa toxicity classifier, P(toxic)); V2 = sg_9b.native (ShieldGemma 9B,
# Harassment and Hate Speech policies, the larger P(Yes)). No "probability": the scores are not
# presented as calibrated (C.3, C.4).
V1_DESCRIPTION = (
    "Checker A is a text classifier trained on human-labelled online comments to recognize "
    "toxic language; its score is the classifier's output for the toxic class."
)
V2_DESCRIPTION = (
    "Checker B is a language model that checks the text against two content policies, one on "
    "harassment and one on hate speech; its score is the higher of its two outputs for a policy "
    "violation."
)

# --- A1 (II.7.1): scores + thresholds + score semantics ---------------------------------------
A1_BLOCK = (
    "CHECKER OUTPUT\n"
    "\n"
    "{v1_name}: score {s1} (threshold {tau1})\n"
    "{v2_name}: score {s2} (threshold {tau2})\n"
    "\n"
    "How to read these scores:\n"
    "{v1_description}\n"
    "{v2_description}\n"
    "Each score runs from 0 to 1; higher means more toxic. Each checker has its own "
    "threshold, shown above. The two scores come from different systems and are not on a "
    "common scale: a 0.7 from one does not mean the same thing as a 0.7 from the other, "
    "and the difference between them is not meaningful."
)

# --- B = A1 + one blank line + this block (II.7.2) ---------------------------------------------
# Only this block may differ between A1 and B. Numeric form, signed margins, "N/A" when the
# checkers agree. The margins are called distances / decision margins, never confidence (C.4).
# "at or above" matches "reaches its threshold" and y_i = 1[s_i >= tau_i]: a zero distance is
# rendered +0.0000 and the checker flags.
B_SEPARATOR = "\n\n"
DERIVED_BLOCK = (
    "DERIVED FROM THE SCORES ABOVE\n"
    "\n"
    "{v1_name} flags this text: {y1}\n"
    "{v2_name} flags this text: {y2}\n"
    "{v1_name} distance from its threshold: {m1}\n"
    "{v2_name} distance from its threshold: {m2}\n"
    "The two checkers agree: {agreement}\n"
    "Disagreement direction: {conflict_direction}\n"
    "\n"
    "Each value above is computed from the two scores and the two thresholds. A checker flags "
    "the text when its score reaches its threshold. The distance is the score minus the "
    "threshold, signed: positive means at or above the threshold, negative means below. "
    "Distances from different checkers are not comparable to each other."
)

FLAG_WORDS = {1: "yes", 0: "no"}
AGREEMENT_WORDS = {1: "yes", 0: "no"}
# Keyed by ConflictDirection.value (not imported; see the header).
CONFLICT_WORDS = {
    "v1_unsafe_v2_safe": f"{V1_NAME} flags, {V2_NAME} does not",
    "v1_safe_v2_unsafe": f"{V2_NAME} flags, {V1_NAME} does not",
    "N/A": "N/A",
}

# Every number shown -- scores, thresholds, distances, in every arm and in Exp 2 -- has this
# many significant digits, written as a plain decimal (never scientific notation) with its
# trailing zeros kept: 0.83412, 0.0031513, 0.000033283, 1.0000; zero is 0.0000. Distances carry
# a sign, and a zero distance is +0.0000 (the checker flags at s == tau). No score is
# transformed and the number of digits never depends on how close a score is to its threshold
# (2026-09-25, decisions.md D-2026-09-25b point 1).
SIGNIFICANT_DIGITS = 5
