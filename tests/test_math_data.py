"""Question identity, eligibility, sampling and gold-data separation."""

from __future__ import annotations

from dataclasses import replace

import pytest

from rafr.domains.math.data import (
    MathDataError,
    MathQuestion,
    assert_disjoint_question_groups,
    extract_last_boxed,
    from_math_row,
    normalize_problem,
    problem_fingerprint,
    screen_questions,
    stable_sample,
)


def question(qid="test/algebra/1.json", problem="Find $1+1$.", answer="2", split="test"):
    return MathQuestion(qid, problem, answer, split, "Algebra", 1)


def test_original_path_survives_adaptation_and_hub_storage_split_is_ignored():
    row = {
        "unique_id": "test/algebra/42.json",
        "problem": "Compute $2+2$.",
        "answer": "4",
        "subject": "Algebra",
        "level": "Level 2",
        "split": "train",
    }
    q = from_math_row(row, source="HuggingFaceH4/MATH-500", source_revision="abc")
    assert (q.original_split, q.question_id, q.level) == ("test", row["unique_id"], 2)
    assert q.source_revision == "abc"
    with pytest.raises(MathDataError, match="conflicting original split"):
        from_math_row(row, original_split="train")


def test_merged_storage_split_never_becomes_original_training_split():
    q = from_math_row({"problem": "Compute $1+1$.", "answer": "2", "split": "train"})
    assert q.original_split == "unknown"
    assert q.question_id.startswith("content:")
    assert screen_questions([q]).exclusions[0].reason == "unknown_original_split"


def test_content_identity_is_not_derived_from_gold_or_input_row_order():
    row = {"problem": "Find  $1+1$.\n", "answer": "2"}
    q = from_math_row(row)
    other = from_math_row({**row, "problem": "Find $1+1$.", "answer": "3"})
    assert q.question_id == other.question_id
    assert q.problem == row["problem"]


def test_raw_math_solution_answer_extracts_balanced_last_box():
    row = {"problem": "Compute.", "solution": r"First \boxed{1}; then \boxed{\frac{2}{3}}."}
    assert from_math_row(row, original_split="train").reference_answer == r"\frac{2}{3}"
    with pytest.raises(MathDataError, match="missing explicit reference"):
        from_math_row({**row, "solution": r"Old \boxed{1}; unfinished \boxed{2"})


@pytest.mark.parametrize(
    "text,answer",
    [
        (r"\boxed{\{1,2\}}", r"\{1,2\}"),
        (r"\fbox{42}", "42"),
        (r"No final number 42", None),
        (r"\boxed{ }", None),
    ],
)
def test_box_extraction(text, answer):
    assert extract_last_boxed(text) == answer


def test_public_payload_does_not_leak_reference_or_solution():
    q = question(answer="GOLD_SECRET")
    assert q.public_payload() == {"question_id": q.question_id, "problem": q.problem}
    payload = q.verifier_payload("CANDIDATE_ONLY")
    assert payload == {**q.public_payload(), "candidate_solution": "CANDIDATE_ONLY"}
    assert "GOLD_SECRET" not in repr(payload)
    assert "GOLD_SECRET" not in repr(q)


@pytest.mark.parametrize(
    "problem,reason",
    [
        ("Use the diagram to compute the angle.", "possible_visual_dependency"),
        ("Compute the area. [asy]draw((0,0)--(1,1));[/asy]", "possible_visual_dependency"),
        (r"Find x. \includegraphics{triangle.png}", "possible_visual_dependency"),
        ("Use the graph shown below.", "possible_visual_dependency"),
        ("Prove that the claim is true.", "proof_or_non_answer_task"),
        ("Show that $a=b$.", "proof_or_non_answer_task"),
        ("Sketch the curve.", "proof_or_non_answer_task"),
    ],
)
def test_conservative_eligibility(problem, reason):
    result = screen_questions([question(problem=problem)])
    assert result.questions == []
    assert result.exclusions[0].reason == reason


def test_geometry_text_alone_is_not_blanket_excluded():
    q = question(problem="A triangle has sides 3, 4, 5. What is its area?")
    assert screen_questions([q]).questions == [q]


def test_dedupe_is_order_independent_and_conflicts_fail_loudly():
    a = question(qid="test/algebra/a.json")
    b = replace(a, question_id="test/algebra/b.json", problem="Find  $1+1$. ")
    assert screen_questions([b, a, a]) == screen_questions([a, b, a])
    result = screen_questions([b, a, a])
    assert result.questions == [a]
    assert len(result.duplicates) == 2
    with pytest.raises(MathDataError, match="conflicting records"):
        screen_questions([a, replace(a, reference_answer="3")])
    with pytest.raises(MathDataError, match="conflicting reference"):
        screen_questions([a, replace(b, reference_answer="3")])


def test_cross_original_split_duplicates_are_both_excluded():
    a = question()
    b = replace(a, question_id="train/algebra/2.json", original_split="train")
    result = screen_questions([a, b])
    assert not result.questions
    assert len(result.exclusions) == 2
    assert {e.reason for e in result.exclusions} == {"duplicate_split_conflict"}


def test_sampling_is_stable_uses_ids_and_fails_on_shortfall():
    rows = [question(qid=f"test/algebra/{i}.json", problem=f"Compute {i}+1.") for i in range(20)]
    sample = stable_sample(rows, 8, seed="math-feas-v1")
    assert sample == stable_sample(reversed(rows), 8, seed="math-feas-v1")
    assert sample != stable_sample(rows, 8, seed="math-feas-v2")
    assert len({q.question_id for q in sample}) == 8
    with pytest.raises(MathDataError, match="requested"):
        stable_sample(rows, 21, seed=1)
    with pytest.raises(MathDataError, match="deduplicate"):
        stable_sample([rows[0], rows[0]], 1, seed=1)


@pytest.mark.parametrize(
    "changes",
    [
        {"problem": ""},
        {"reference_answer": ""},
        {"question_id": ""},
        {"level": True},
        {"level": 8},
        {"original_split": "validation"},
    ],
)
def test_question_validation(changes):
    with pytest.raises(MathDataError):
        replace(question(), **changes)


def test_public_fingerprint_is_unicode_and_whitespace_stable_without_altering_math():
    problem = "Find the cafe\u0301 count: $x+X$.\n"
    normalized = "Find the café count: $x+X$."
    assert normalize_problem(problem) == normalized
    assert problem_fingerprint(problem) == problem_fingerprint("  " + normalized + "\t")
    assert len(problem_fingerprint(problem)) == 64
    assert problem_fingerprint("Find $x$.") != problem_fingerprint("Find $X$.")
    assert problem_fingerprint("Find $x+1$.") != problem_fingerprint("Find $x-1$.")
    row = from_math_row({"problem": problem, "answer": "2"})
    assert row.question_id == f"content:{problem_fingerprint(problem)}"


def test_disjoint_groups_reject_same_content_under_different_mirror_ids():
    original = question(answer="SECRET_GOLD")
    mirror = replace(
        original, question_id="mirror:17", problem="Find  $1+1$.\n", source="other-mirror"
    )
    with pytest.raises(MathDataError, match=r"problem_fingerprint \(1\)") as exc:
        assert_disjoint_question_groups([original], [mirror])
    assert "SECRET_GOLD" not in str(exc.value)
    assert original.problem not in str(exc.value)


def test_disjoint_groups_reject_reused_id_even_when_question_text_changes():
    original = question()
    edited = replace(original, problem="Compute $3+4$.")
    with pytest.raises(MathDataError, match=r"question_id \(1\)"):
        assert_disjoint_question_groups([original], [edited])


def test_disjoint_groups_accept_public_records_and_generators_without_gold():
    development = [question().public_payload()]
    evaluation = [question(qid="different-id", problem="Compute $3+4$.").public_payload()]
    assert_disjoint_question_groups(iter(development), iter(evaluation))
    assert_disjoint_question_groups([], evaluation)
    assert_disjoint_question_groups(development, [])
    with pytest.raises(MathDataError, match="question groups overlap"):
        assert_disjoint_question_groups(iter(development), iter(development))


@pytest.mark.parametrize(
    "bad_record",
    [
        {"problem": "Find $1+1$."},
        {"question_id": "test/algebra/1.json"},
        {"question_id": "test/algebra/1.json", "problem": " "},
        {"question_id": "test/algebra/1.json", "problem": 2},
    ],
)
def test_disjoint_groups_fail_on_incomplete_identity_instead_of_silently_skipping(bad_record):
    with pytest.raises(MathDataError):
        assert_disjoint_question_groups([question()], [bad_record])
