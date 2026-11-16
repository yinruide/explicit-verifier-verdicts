"""Small, provenance-preserving adapters for answer-based MATH experiments.

No downloads occur here. A Hugging Face storage split is NOT necessarily the
original MATH split: ``qwedsacf/competition_math`` merges the original splits.
Supply a documented original split, or retain ``unknown``. A path-style original
ID such as ``test/algebra/123.json`` also supplies that provenance.
"""

from __future__ import annotations

import hashlib
import math
import re
import unicodedata
from collections import defaultdict
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from typing import Any

ORIGINAL_SPLITS = frozenset({"train", "test", "unknown"})
PROBLEM_FINGERPRINT_VERSION = "nfc_whitespace_v1"


class MathDataError(ValueError):
    """Missing or conflicting question-level data; never silently repair it."""


@dataclass(frozen=True)
class MathQuestion:
    question_id: str
    problem: str
    reference_answer: str = field(repr=False)
    original_split: str
    subject: str
    level: int | float | str | None
    source: str = ""
    source_revision: str = ""
    source_kind: str = "math"
    difficulty: float | None = None
    source_partition: str | None = None

    def __post_init__(self) -> None:
        for name in ("question_id", "problem", "reference_answer"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise MathDataError(f"{name} must be a nonempty string")
        if self.original_split not in ORIGINAL_SPLITS:
            raise MathDataError("original_split must be train, test, or unknown")
        if self.source_kind not in {"math", "deepmath", "harp"}:
            raise MathDataError("unknown source kind")
        if (
            self.source_kind == "math"
            and self.level is not None
            and (type(self.level) is not int or self.level not in range(1, 6))
        ):
            raise MathDataError("MATH level must be an integer from 1 to 5, or None")
        if self.level is not None and (
            isinstance(self.level, bool)
            or not isinstance(self.level, (int, float, str))
            or isinstance(self.level, (int, float))
            and not math.isfinite(self.level)
        ):
            raise MathDataError("source level must be finite numeric or text metadata")
        if self.difficulty is not None and (
            isinstance(self.difficulty, bool)
            or not isinstance(self.difficulty, (int, float))
            or not math.isfinite(self.difficulty)
        ):
            raise MathDataError("source difficulty must be finite numeric metadata")
        if self.source_partition is not None and (
            not isinstance(self.source_partition, str) or not self.source_partition.strip()
        ):
            raise MathDataError("source_partition must be nonempty when supplied")
        if not all(
            isinstance(getattr(self, name), str)
            for name in ("subject", "source", "source_revision")
        ):
            raise MathDataError("subject/source/source_revision must be strings")

    def public_payload(self) -> dict[str, str]:
        """Explicit allowlist: never serialize the entire question to a model."""
        return {"question_id": self.question_id, "problem": self.problem}

    def verifier_payload(self, candidate_solution: str) -> dict[str, str]:
        if not isinstance(candidate_solution, str):
            raise TypeError("candidate_solution must be a string")
        return {**self.public_payload(), "candidate_solution": candidate_solution}


def _normalized(text: str) -> str:
    # Preserve mathematical case, punctuation, and all original text in MathQuestion.
    return " ".join(unicodedata.normalize("NFC", text).split())


def normalize_problem(problem: str) -> str:
    """NFC plus whitespace folding, retaining mathematical case and punctuation.

    This deliberately does not canonicalize LaTeX, rename variables, or detect
    paraphrases. Keep the original problem alongside this development-only key.
    """
    if not isinstance(problem, str) or not problem.strip():
        raise MathDataError("problem must be a nonempty string")
    return _normalized(problem)


def problem_fingerprint(problem: str) -> str:
    """Stable SHA256 of UTF-8 ``normalize_problem(problem)``; contains no gold.

    Record ``PROBLEM_FINGERPRINT_VERSION`` in exported exclusion manifests so a
    future normalization change cannot silently weaken cross-mirror matching.
    """
    return hashlib.sha256(normalize_problem(problem).encode("utf-8")).hexdigest()


def assert_disjoint_question_groups(
    development: Iterable[MathQuestion | Mapping[str, Any]],
    evaluation: Iterable[MathQuestion | Mapping[str, Any]],
) -> None:
    """Reject shared IDs OR shared normalized question text across the two groups.

    Public mappings only need ``question_id`` and ``problem``: no reference answer
    is read. All source development questions, including unselected/excluded ones,
    should be passed by the caller. Passing does not rule out semantic near-duplicates
    or exposure during model training.
    """

    def keys(rows: Iterable[MathQuestion | Mapping[str, Any]]) -> tuple[set[str], set[str]]:
        ids: set[str] = set()
        fingerprints: set[str] = set()
        for row in rows:
            if isinstance(row, MathQuestion):
                qid, problem = row.question_id, row.problem
            elif isinstance(row, Mapping):
                qid, problem = row.get("question_id"), row.get("problem")
            else:
                raise MathDataError("question groups require MathQuestion or public mappings")
            if not isinstance(qid, str) or not qid.strip():
                raise MathDataError("question_id must be a nonempty string")
            ids.add(qid)
            fingerprints.add(problem_fingerprint(problem))
        return ids, fingerprints

    development_ids, development_fingerprints = keys(development)
    evaluation_ids, evaluation_fingerprints = keys(evaluation)
    shared_ids = sorted(development_ids & evaluation_ids)
    shared_fingerprints = sorted(development_fingerprints & evaluation_fingerprints)
    if shared_ids or shared_fingerprints:
        # Bounded error evidence, never reference answers or full question text.
        raise MathDataError(
            f"question groups overlap: question_id ({len(shared_ids)}) {shared_ids[:8]}; "
            f"problem_fingerprint ({len(shared_fingerprints)}) {shared_fingerprints[:8]}"
        )


def extract_last_boxed(text: str) -> str | None:
    """Read a balanced final \\boxed{...}/\\fbox{...}, including nested braces.

    An incomplete LAST box invalidates extraction; do not fall back to an earlier
    intermediate answer. Escaped braces do not change the nesting depth.
    """
    starts = list(re.finditer(r"\\(?:boxed|fbox)\s*\{", text))
    if not starts:
        return None
    start = starts[-1].end()
    depth = 1
    for i in range(start, len(text)):
        if text[i] not in "{}":
            continue
        n_backslashes = 0
        j = i - 1
        while j >= 0 and text[j] == "\\":
            n_backslashes += 1
            j -= 1
        if n_backslashes % 2:
            continue
        depth += 1 if text[i] == "{" else -1
        if depth == 0:
            return text[start:i].strip() or None
    return None


def _nonempty(row: Mapping[str, Any], *names: str) -> str | None:
    for name in names:
        value = row.get(name)
        if value is not None:
            if not isinstance(value, str):
                raise MathDataError(f"{name} must be a string")
            if value.strip():
                return value
    return None


def from_math_row(
    row: Mapping[str, Any],
    *,
    original_split: str | None = None,
    source: str = "",
    source_revision: str = "",
) -> MathQuestion:
    """Adapt MATH-500, original MATH, or explicitly normalized question records.

    ``split`` is deliberately ignored. Callers may pass ``original_split`` ONLY
    after checking source provenance. IDs lacking an original path use a stable
    content ID (not a row index), visibly prefixed ``content:``.
    """
    problem = _nonempty(row, "problem")
    if problem is None:
        raise MathDataError("missing problem")
    answer = _nonempty(row, "reference_answer", "answer", "ground_truth_answer")
    if answer is None:
        solution = _nonempty(row, "solution", "ground_truth_solution")
        answer = extract_last_boxed(solution) if solution else None
    if answer is None:
        raise MathDataError("missing explicit reference answer or balanced final box")
    qid = _nonempty(row, "question_id", "unique_id")
    original = _nonempty(row, "original_split")
    from_id = None
    if qid:
        match = re.fullmatch(r"(?:MATH/)?(train|test)/[^/]+/[^/]+\.json", qid)
        if match:
            from_id = match[1]
    known = {s for s in (original_split, original, from_id) if s not in (None, "unknown")}
    if len(known) > 1:
        raise MathDataError("conflicting original split provenance")
    split = next(iter(known), "unknown")
    if qid is None:
        qid = f"content:{problem_fingerprint(problem)}"
    level = row.get("level")
    if isinstance(level, str):
        match = re.fullmatch(r"(?:Level\s+)?([1-5])", level.strip())
        if not match:
            raise MathDataError("unrecognized MATH level")
        level = int(match[1])
    subject = _nonempty(row, "subject", "type") or "unknown"
    return MathQuestion(qid, problem, answer, split, subject, level, source, source_revision)


_VISUAL = re.compile(
    r"\[asy\]|\\(?:includegraphics|begin\{(?:tikzpicture|picture)\})"
    r"|<img\b|!\[[^\]]*\]\("
    r"|\b(?:diagram|figure|picture|illustration)\b"
    r"|\b(?:shown|displayed|depicted)\s+(?:below|above)\b"
    r"|\b(?:following|accompanying)\s+(?:graph|chart|table)\b"
    r"|\b(?:graph|chart|table)\s+(?:shown|below|above)\b",
    re.IGNORECASE,
)
_NON_ANSWER = re.compile(
    r"\b(?:prove|disprove|sketch|draw|construct)\b|\bshow\s+that\b"
    r"|\bgive\s+(?:a\s+)?(?:proof|counterexample)\b",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class ExcludedQuestion:
    question_id: str
    reason: str


@dataclass(frozen=True)
class DuplicateQuestion:
    question_id: str
    kept_question_id: str
    reason: str


@dataclass
class ScreenResult:
    questions: list[MathQuestion]
    exclusions: list[ExcludedQuestion]
    duplicates: list[DuplicateQuestion]


def screen_questions(
    questions: Iterable[MathQuestion],
    *,
    require_original_split: bool = True,
) -> ScreenResult:
    """Conservative automatic eligibility screen; not a claim of human validation.

    Remove visual/proof tasks. Deduplicate by ID and whitespace-normalized problem,
    retaining the lexicographically first ID. Conflicting answer text stops; copies
    spanning original splits are ALL excluded to avoid accidental split leakage.
    """
    by_id: dict[str, MathQuestion] = {}
    duplicates: list[DuplicateQuestion] = []
    exclusions: list[ExcludedQuestion] = []
    for q in questions:
        if q.question_id in by_id:
            if q != by_id[q.question_id]:
                raise MathDataError(f"conflicting records for question_id {q.question_id}")
            duplicates.append(DuplicateQuestion(q.question_id, q.question_id, "duplicate_id"))
        else:
            by_id[q.question_id] = q
    groups: dict[str, list[MathQuestion]] = defaultdict(list)
    for q in sorted(by_id.values(), key=lambda item: item.question_id):
        groups[normalize_problem(q.problem)].append(q)
    retained = []
    for group in groups.values():
        if len({_normalized(q.reference_answer) for q in group}) > 1:
            raise MathDataError(f"conflicting reference answers for {group[0].question_id}")
        if len({q.original_split for q in group}) > 1:
            exclusions.extend(
                ExcludedQuestion(q.question_id, "duplicate_split_conflict") for q in group
            )
            continue
        q = group[0]
        reason = None
        if require_original_split and q.original_split == "unknown":
            reason = "unknown_original_split"
        elif _VISUAL.search(q.problem):
            reason = "possible_visual_dependency"
        elif _NON_ANSWER.search(q.problem):
            reason = "proof_or_non_answer_task"
        if reason:
            exclusions.extend(ExcludedQuestion(item.question_id, reason) for item in group)
            continue
        retained.append(q)
        duplicates.extend(
            DuplicateQuestion(item.question_id, q.question_id, "duplicate_problem")
            for item in group[1:]
        )
    return ScreenResult(
        sorted(retained, key=lambda q: q.question_id),
        sorted(exclusions, key=lambda q: q.question_id),
        sorted(duplicates, key=lambda q: (q.question_id, q.kept_question_id)),
    )


def stable_sample(questions: Iterable[MathQuestion], n: int, seed: int | str) -> list[MathQuestion]:
    """Sample without replacement, invariant to file order and Python RNG changes."""
    rows = list(questions)
    if type(n) is not int or n < 0 or n > len(rows):
        raise MathDataError(f"requested {n!r} questions from {len(rows)} eligible questions")
    if len({q.question_id for q in rows}) != len(rows):
        raise MathDataError("deduplicate question IDs before sampling")

    def key(q: MathQuestion) -> tuple[str, str]:
        digest = hashlib.sha256(f"{seed}\0{q.question_id}".encode()).hexdigest()
        return digest, q.question_id

    return sorted(rows, key=key)[:n]
