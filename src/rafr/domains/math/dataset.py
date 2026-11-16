"""Versioned local math data contracts; no downloads or model calls.

Formal loading always rechecks the pinned, complete MATH-500 development source.
Public records and private references have separate schemas. Hashes establish
identity and association, not semantic correctness or freedom from training data
contamination. A caller must retain every additional development source too.
"""

from __future__ import annotations

import hashlib
import inspect
import json
import os
import re
import zipfile
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Any

from rafr.domains.math.data import (
    PROBLEM_FINGERPRINT_VERSION,
    MathDataError,
    MathQuestion,
    assert_disjoint_question_groups,
    from_math_row,
    normalize_problem,
    problem_fingerprint,
    screen_questions,
)

DATA_CONTRACT_VERSION = "math_dataset_v7_label_references"
MATH500_SOURCE = "HuggingFaceH4/MATH-500"
MATH500_REVISION = "6e4ed1a2a79af7d8630a6b768ec859cb5af4d3be"
MATH500_SHA256 = "35dc41080a3680858b27fa7e0533d2d547825316fc5dafe5d316f4ccc5a06132"
MATH500_COUNT = 500
STAGES = frozenset({"development", "stub", "smoke", "formal"})
Validator = Callable[[str], str | None]


def content_hash(value: Any) -> str:
    """Canonical JSON hash used for contracts, private bindings and split plans."""
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def file_hash(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _nonempty(value: Any, name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise MathDataError(f"{name} must be a nonempty string")


def _sha(value: Any, name: str) -> None:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(c not in "0123456789abcdef" for c in value)
    ):
        raise MathDataError(f"{name} must be a lowercase SHA256")


def _read_rows(path: Path) -> list[dict[str, Any]]:
    rows = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise MathDataError(f"invalid JSON at {path.name}:{number}") from exc
        if not isinstance(row, dict):
            raise MathDataError(f"expected JSON object at {path.name}:{number}")
        rows.append(row)
    return rows


def _write_json(path: Path, value: Any) -> None:
    if path.exists():
        raise MathDataError("refuse to replace an existing dataset manifest")
    partial = path.with_suffix(path.suffix + ".partial")
    with partial.open("x", encoding="utf-8", newline="\n") as output:
        output.write(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
        output.flush()
        os.fsync(output.fileno())
    partial.replace(path)


def _write_rows(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    with path.open("x", encoding="utf-8", newline="\n") as output:
        for row in rows:
            output.write(json.dumps(row, ensure_ascii=False) + "\n")


@dataclass(frozen=True)
class SourceMetadata:
    source: str
    revision: str
    source_sha256: str
    license_id: str
    license_url: str
    license_status: str
    original_split: str | None = None
    source_kind: str = "math"
    source_partition: str | None = None

    def __post_init__(self) -> None:
        for name in ("source", "revision", "license_id", "license_url"):
            _nonempty(getattr(self, name), name)
        _sha(self.source_sha256, "source_sha256")
        if self.license_status not in {"approved", "review_required"}:
            raise MathDataError("license_status must be approved or review_required")
        if self.original_split not in {None, "train", "test", "unknown"}:
            raise MathDataError("invalid documented original_split")
        if self.source_kind not in {"math", "deepmath", "harp"}:
            raise MathDataError("source_kind must be math, deepmath or harp")
        if self.source_partition is not None:
            _nonempty(self.source_partition, "source_partition")


@dataclass(frozen=True)
class PublicQuestion:
    question_id: str
    problem: str

    def __post_init__(self) -> None:
        _nonempty(self.question_id, "question_id")
        _nonempty(self.problem, "problem")

    def public_payload(self) -> dict[str, str]:
        return {"question_id": self.question_id, "problem": self.problem}

    def verifier_payload(self, candidate_solution: str) -> dict[str, str]:
        if not isinstance(candidate_solution, str):
            raise MathDataError("candidate_solution must be a string")
        return {**self.public_payload(), "candidate_solution": candidate_solution}


@dataclass(frozen=True)
class PrivateReference:
    question_id: str
    problem_sha256: str
    reference_answer: str = field(repr=False)
    binding_sha256: str
    original_split: str
    subject: str
    level: int | float | str | None
    source_kind: str = "math"
    difficulty: float | None = None
    source_partition: str | None = None


def _reference(question: MathQuestion) -> PrivateReference:
    fingerprint = problem_fingerprint(question.problem)
    return PrivateReference(
        question.question_id,
        fingerprint,
        question.reference_answer,
        content_hash([question.question_id, fingerprint, question.reference_answer]),
        question.original_split,
        question.subject,
        question.level,
        question.source_kind,
        question.difficulty,
        question.source_partition,
    )


@dataclass(frozen=True)
class DatasetBundle:
    public_questions: tuple[PublicQuestion, ...]
    private_references: Mapping[str, PrivateReference] = field(repr=False)
    manifest: Mapping[str, Any]
    manifest_sha256: str

    def public_payloads(self) -> list[dict[str, str]]:
        return [question.public_payload() for question in self.public_questions]


def validate_public_question(value: PublicQuestion | Mapping[str, Any]) -> PublicQuestion:
    """Model input guard: reject gold-bearing objects and extra mapping fields."""
    if isinstance(value, PublicQuestion):
        return value
    if not isinstance(value, Mapping) or set(value) != {"question_id", "problem"}:
        raise MathDataError("public question schema violation; private fields forbidden")
    return PublicQuestion(**value)


def from_deepmath_row(row: Mapping[str, Any], metadata: SourceMetadata) -> MathQuestion:
    """Map native DeepMath fields; never serialize any r1_solution into model inputs."""
    question, answer = row.get("question"), row.get("final_answer")
    _nonempty(question, "question")
    _nonempty(answer, "final_answer")
    topic = row.get("topic", "unknown")
    _nonempty(topic, "topic")
    return MathQuestion(
        question_id=f"deepmath:{problem_fingerprint(question)}",
        problem=question,
        reference_answer=answer,
        original_split=metadata.original_split or "unknown",
        subject=topic,
        level=None,
        source=metadata.source,
        source_revision=metadata.revision,
        source_kind="deepmath",
        difficulty=row.get("difficulty"),
        source_partition=metadata.source_partition,
    )


def from_harp_row(row: Mapping[str, Any], metadata: SourceMetadata) -> MathQuestion:
    """HARP's source identity is year + contest + number; keep its difficulty scale."""
    for name in ("problem", "answer"):
        _nonempty(row.get(name), name)
    identity = []
    for name in ("year", "contest", "number"):
        value = row.get(name)
        if isinstance(value, bool) or not isinstance(value, (str, int)) or not str(value).strip():
            raise MathDataError(f"HARP requires source identity field {name}")
        identity.append(str(value).strip())
    # JSON avoids composite-ID delimiter collisions while preserving the original fields.
    qid = "harp:" + json.dumps(identity, ensure_ascii=False, separators=(",", ":"))
    return MathQuestion(
        question_id=qid,
        problem=row["problem"],
        reference_answer=row["answer"],
        original_split=metadata.original_split or "unknown",
        subject=row.get("subject", "unknown"),
        level=row.get("level"),
        source=metadata.source,
        source_revision=metadata.revision,
        source_kind="harp",
        source_partition=metadata.source_partition,
    )


def _read_source_rows(path: Path, metadata: SourceMetadata) -> list[dict[str, Any]]:
    if path.suffix == ".parquet":
        import pyarrow.parquet as pq

        # The three DeepSeek reasoning columns are not task inputs or golds and
        # account for most of the official file. Never materialize them here.
        columns = (
            ["question", "final_answer", "difficulty", "topic"]
            if metadata.source_kind == "deepmath"
            else None
        )
        return pq.read_table(path, columns=columns).to_pylist()
    if path.suffix == ".zip":
        if metadata.source_kind != "harp":
            raise MathDataError("zip source is supported for the official HARP JSONL packages")
        with zipfile.ZipFile(path) as archive:
            names = [name for name in archive.namelist() if name.endswith(".jsonl")]
            if len(names) != 1:
                raise MathDataError("HARP archive must contain one unambiguous JSONL member")
            # Read directly; never extract arbitrary archive paths into the workspace.
            contents = archive.read(names[0]).decode("utf-8")
        rows = [json.loads(line) for line in contents.splitlines() if line.strip()]
        if any(not isinstance(row, dict) for row in rows):
            raise MathDataError("source rows must be JSON objects")
        return rows
    return _read_rows(path)


def native_questions(
    rows: Sequence[Mapping[str, Any]], metadata: SourceMetadata
) -> tuple[list[MathQuestion], list[dict[str, str]]]:
    """Map source rows to questions plus rows excluded before mapping.

    A DeepMath row whose final_answer is present but empty is an explicit exclusion
    (reference_empty: M.2.1 needs an E-parseable reference). An absent or non-string
    final_answer is still a damaged source and raises in from_deepmath_row.
    """
    if metadata.source_kind == "deepmath":
        questions, exclusions = [], []
        for row in rows:
            answer = row.get("final_answer")
            if isinstance(answer, str) and not answer.strip():
                _nonempty(row.get("question"), "question")
                exclusions.append(
                    {
                        "question_id": f"deepmath:{problem_fingerprint(row['question'])}",
                        "reason": "reference_empty",
                    }
                )
            else:
                questions.append(from_deepmath_row(row, metadata))
        return questions, exclusions
    if metadata.source_kind == "harp":
        return [from_harp_row(row, metadata) for row in rows], []
    return [
        from_math_row(
            row,
            source=metadata.source,
            source_revision=metadata.revision,
            original_split=metadata.original_split,
        )
        for row in rows
    ], []


def _source_questions(
    path: Path, metadata: SourceMetadata
) -> tuple[list[MathQuestion], list[dict[str, str]], int]:
    """Questions, rows excluded before mapping, and the number of source rows."""
    if file_hash(path) != metadata.source_sha256:
        raise MathDataError("source file SHA256 differs from the declared source")
    rows = _read_source_rows(path, metadata)
    questions, source_exclusions = native_questions(rows, metadata)
    # Repeated problems are deduplicated during screening; one ID naming two different
    # problems is a damaged source and still stops here.
    problems: dict[str, str] = {}
    for question in questions:
        fingerprint = problem_fingerprint(question.problem)
        if problems.setdefault(question.question_id, fingerprint) != fingerprint:
            raise MathDataError(f"question_id names different problems: {question.question_id}")
    if not questions:
        raise MathDataError("source contains no questions")
    return questions, source_exclusions, len(rows)


def _deduplicate(
    questions: Sequence[MathQuestion],
) -> tuple[list[MathQuestion], list[dict[str, str]]]:
    """Keep one copy per normalized problem; every other copy is an explicit exclusion.

    The kept copy has the smallest question_id, then the earliest source row. Copies
    whose (whitespace-normalized) reference answers or documented original splits
    differ are all excluded: no gold or split is chosen between them. Never raises.
    """
    groups: dict[str, list[tuple[str, int, MathQuestion]]] = {}
    for position, question in enumerate(questions):
        groups.setdefault(problem_fingerprint(question.problem), []).append(
            (question.question_id, position, question)
        )
    kept, exclusions = [], []
    for members in groups.values():
        members.sort(key=lambda member: member[:2])
        copies = [member[2] for member in members]
        if len({normalize_problem(q.reference_answer) for q in copies}) > 1:
            reason, keep = "duplicate_problem_conflicting_reference", None
        elif len({q.original_split for q in copies}) > 1:
            reason, keep = "duplicate_split_conflict", None
        else:
            reason, keep = "duplicate_normalized_problem", copies[0]
        if keep is not None:
            kept.append(keep)
        exclusions.extend(
            {"question_id": q.question_id, "reason": reason} for q in copies if q is not keep
        )
    return kept, exclusions


def _assert_unique(questions: Sequence[MathQuestion | PublicQuestion]) -> None:
    ids: set[str] = set()
    fingerprints: set[str] = set()
    for question in questions:
        fingerprint = problem_fingerprint(question.problem)
        if question.question_id in ids:
            raise MathDataError(f"duplicate question_id: {question.question_id}")
        if fingerprint in fingerprints:
            raise MathDataError(f"duplicate normalized problem: {question.question_id}")
        ids.add(question.question_id)
        fingerprints.add(fingerprint)


def _development_exclusions(
    source_path: Path | None,
    additional_paths: Sequence[Path],
) -> tuple[list[dict[str, str]], dict[str, Any]]:
    if source_path is None:
        raise MathDataError("formal loading requires the complete pinned MATH-500 source")
    if file_hash(source_path) != MATH500_SHA256:
        raise MathDataError("development MATH-500 source SHA256 mismatch")
    # Every row participates, including visually excluded and previously unselected rows.
    development = [from_math_row(row).public_payload() for row in _read_rows(source_path)]
    if len(development) != MATH500_COUNT or len({q["question_id"] for q in development}) != 500:
        raise MathDataError("development exclusion must contain all 500 unique MATH-500 IDs")
    additional = []
    for path in additional_paths:
        rows = _read_rows(path)
        for row in rows:
            if set(row) != {"question_id", "problem"}:
                raise MathDataError("additional development files require public question schema")
            PublicQuestion(**row)
        development.extend(rows)
        additional.append({"sha256": file_hash(path), "count": len(rows)})
    audit = {
        "source": MATH500_SOURCE,
        "revision": MATH500_REVISION,
        "source_sha256": MATH500_SHA256,
        "math500_count": MATH500_COUNT,
        "fingerprint_version": PROBLEM_FINGERPRINT_VERSION,
        "exclusion_identity_sha256": content_hash(
            sorted((q["question_id"], problem_fingerprint(q["problem"])) for q in development)
        ),
        "additional_development_sources": additional,
        "limitation": (
            "ID/text identity only; semantic near-duplicates and training exposure unchecked"
        ),
    }
    return development, audit


def _validators(
    answer_validator: Validator | None,
    question_validator: Validator | None,
    answer_contract_version: str | None,
) -> tuple[Validator, Validator, str]:
    if answer_validator is None or question_validator is None or answer_contract_version is None:
        from rafr.domains.math.answer_contract import (
            ANSWER_CONTRACT_VERSION,
            question_exclusion_reason,
            reference_exclusion_reason,
        )

        answer_validator = answer_validator or reference_exclusion_reason
        question_validator = question_validator or question_exclusion_reason
        answer_contract_version = answer_contract_version or ANSWER_CONTRACT_VERSION
    _nonempty(answer_contract_version, "answer_contract_version")
    return answer_validator, question_validator, answer_contract_version


def _eligibility_configuration(answer_validator: Validator, version: str) -> dict[str, Any]:
    """Bind reusable reference eligibility to its evaluator and implementation.

    Default production validation records the full pinned Math-Verify config.
    Explicitly injected offline-test validators remain separately identified;
    changing one cannot promote its evidence into a production bundle.
    """
    from rafr.domains.math.answer_contract import reference_exclusion_reason
    from rafr.domains.math.grading import MathGrader

    if answer_validator is reference_exclusion_reason:
        evaluator = MathGrader(require_boxed_prediction=False).configuration()
    else:
        try:
            source = inspect.getsource(answer_validator)
        except (OSError, TypeError) as exc:
            raise MathDataError("injected eligibility validator needs inspectable source") from exc
        evaluator = {
            "injected_validator": f"{answer_validator.__module__}.{answer_validator.__qualname__}",
            "source_sha256": hashlib.sha256(source.encode()).hexdigest(),
        }
    return {
        "evidence_version": "reference-eligibility-v1",
        "answer_contract_version": version,
        "evaluator": evaluator,
        "implementation_sha256": {
            name: file_hash(Path(__file__).with_name(name))
            for name in ("dataset.py", "data.py", "answer_contract.py", "grading.py")
        },
    }


def _answer_hash(answer: str) -> str:
    return hashlib.sha256(answer.encode("utf-8")).hexdigest()


def _eligibility_evidence(root: Path, manifest: dict[str, Any], configuration: dict[str, Any]):
    identity = manifest.get("reference_eligibility", {})
    if (
        identity.get("configuration") != configuration
        or identity.get("source_sha256") != manifest["source"]["source_sha256"]
    ):
        raise MathDataError("reference eligibility configuration/source changed; rebuild bundle")
    results: dict[str, str | None] = {}
    for row in _read_rows(root / "eligibility.jsonl"):
        if set(row) != {"reference_sha256", "exclusion_reason"}:
            raise MathDataError("reference eligibility evidence schema violation")
        key, reason = row["reference_sha256"], row["exclusion_reason"]
        _sha(key, "reference_sha256")
        if reason is not None:
            _nonempty(reason, "reference exclusion reason")
        if key in results:
            raise MathDataError("duplicate reference eligibility evidence")
        results[key] = reason
    if identity.get("n_unique_references") != len(results):
        raise MathDataError("reference eligibility evidence count differs")
    return results


_VERDICT_ANSWERS = frozenset({"yes", "no", "true", "false"})
_CHOICE_LETTERS = frozenset("ABCDE")
_TEXT_COMMAND = re.compile(r"^\\(?:text|textbf|textrm|mathrm|mathbf|mathsf)\s*\{(.*)\}$", re.S)
_ANSWER_WRAPPERS = (
    ("$$", "$$"),
    ("$", "$"),
    (r"\(", r"\)"),
    (r"\[", r"\]"),
    ("**", "**"),
    ("(", ")"),
)


def _answer_core(answer: str) -> str:
    """Strip presentation only (math mode, text commands, parentheses, a final period)."""
    text = answer
    while True:
        old = text
        text = text.strip().removesuffix(".").strip()
        for left, right in _ANSWER_WRAPPERS:
            if text.startswith(left) and text.endswith(right) and len(text) > len(left + right):
                text = text[len(left) : -len(right)]
                break
        if match := _TEXT_COMMAND.match(text):
            text = match.group(1)
        if text == old:
            return text


def reference_type_exclusion_reason(answer: str) -> str | None:
    """M.2.1 (2026-10-03): the reference must be a mathematical object.

    Verdicts (Yes / No / True / False, any case) and single option letters (uppercase
    A-E) are excluded: a wrong first answer there is a flipped judgment with a guessing
    baseline, not a reasoning error. Constants such as G or e are kept.
    """
    core = _answer_core(answer)
    if core.casefold() in _VERDICT_ANSWERS:
        return "reference_verdict_answer"
    if core in _CHOICE_LETTERS:
        return "reference_choice_letter"
    return None


_ACCENTS = (
    "hat|bar|tilde|check|breve|dot|ddot|vec|widehat|widetilde|overline|underline"
    "|mathbf|boldsymbol|mathcal|mathscr|mathfrak|mathrm|mathit"
)
_INDEX_BRACES = re.compile(r"_\s*\{\s*([A-Za-z0-9]+)\s*\}")
_ACCENT_BRACES = re.compile(rf"\\({_ACCENTS})\s*\{{\s*([^{{}}]*?)\s*\}}")
# One symbol: accent commands, a base (one Latin letter or a command such as \theta) and an
# optional subscript. A Latin base may not be part of a word or of a command name.
_SYMBOL_TOKEN = re.compile(
    rf"(?P<mods>(?:\\(?:{_ACCENTS})(?![A-Za-z])\s?)*)"
    r"(?P<base>\\[A-Za-z]+|(?<![A-Za-z\\])[A-Za-z](?![A-Za-z]))"
    r"(?P<index>_[A-Za-z0-9]+)?"
)
# Math-Verify parses i as an ordinary Symbol; pi, gamma and infinity are not Symbols.
_CONSTANT_SYMBOLS = frozenset({"e", "i"})


def _symbol_text(text: str) -> str:
    """x_{12} -> x_12, \\hat{x} -> \\hat x (nested accents too); other braces -> spaces."""
    text = _INDEX_BRACES.sub(r"_\1", text)
    while True:
        old, text = text, _ACCENT_BRACES.sub(r"\\\1 \2", text)
        if text == old:
            return text.replace("{", " ").replace("}", " ")


def _variant(match: re.Match) -> tuple[str, str]:
    return re.sub(r"\s", "", match["mods"]), match["index"] or ""


def reference_label_exclusion_reason(problem: str, answer: str) -> str | None:
    """M.2.1 (2026-10-03): the reference is one of several same-base symbols in the problem.

    Excluded when the reference parses (Math-Verify) as a single non-constant Symbol and
    the problem also contains a sibling: the same base with a different subscript or
    accent (theta-hat-1 / theta-hat-2). A symbol standing alone (n, L, beta, P, K) is kept;
    standalone labels such as P or D_4 are a documented boundary (answer_contract.md).
    Unparseable references are left to the E eligibility check.
    """
    core = _symbol_text(_answer_core(answer)).strip()
    reference = _SYMBOL_TOKEN.fullmatch(core)
    if reference is None or core in _CONSTANT_SYMBOLS:
        return None
    from rafr.domains.math.grading import MathParseError, MathVerifyAdapter

    try:
        parsed = MathVerifyAdapter().parse_answer(answer, "reference")
    except (MathParseError, TimeoutError, RuntimeError):
        return None
    if not getattr(parsed[0], "is_Symbol", False):
        return None
    variants = {
        _variant(match)
        for match in _SYMBOL_TOKEN.finditer(_symbol_text(problem))
        if match["base"] == reference["base"]
    }
    return "reference_is_label" if variants - {_variant(reference)} else None


def _screen_source(
    questions: Sequence[MathQuestion],
    answer_validator: Validator,
    question_validator: Validator,
    development: Sequence[dict[str, str]],
    *,
    allow_unknown_original_split: bool = False,
    source_exclusions: Sequence[dict[str, str]] = (),
) -> tuple[list[MathQuestion], list[dict[str, str]]]:
    development_ids = {q["question_id"] for q in development}
    development_hashes = {problem_fingerprint(q["problem"]) for q in development}
    questions, exclusions = _deduplicate(questions)
    exclusions.extend(source_exclusions)
    screened = screen_questions(questions, require_original_split=not allow_unknown_original_split)
    reasons = {q.question_id: q.reason for q in screened.exclusions}
    accepted = []
    for question in sorted(questions, key=lambda q: q.question_id):
        if question.question_id in development_ids:
            reason = "development_question_id"
        elif problem_fingerprint(question.problem) in development_hashes:
            reason = "development_problem_fingerprint"
        else:
            reason = reasons.get(question.question_id)
            if reason is None:
                reason = question_validator(question.problem)
            if reason is None:
                reason = reference_type_exclusion_reason(question.reference_answer)
            if reason is None:
                reason = reference_label_exclusion_reason(
                    question.problem, question.reference_answer
                )
            if reason is None:
                reason = answer_validator(question.reference_answer)
        if reason is not None:
            _nonempty(reason, "exclusion reason")
            exclusions.append({"question_id": question.question_id, "reason": reason})
        else:
            accepted.append(question)
    return accepted, sorted(exclusions, key=lambda row: (row["question_id"], row["reason"]))


def prepare_dataset_bundle(
    source_path: Path | str,
    output_dir: Path | str,
    *,
    metadata: SourceMetadata,
    stage: str,
    development_source_path: Path | str | None = None,
    additional_development_paths: Sequence[Path | str] = (),
    answer_validator: Validator | None = None,
    question_validator: Validator | None = None,
    answer_contract_version: str | None = None,
) -> dict[str, Any]:
    """Prepare a fresh bundle, retaining every exclusion without silently repairing rows.

    Missing records and an ID naming two different problems fail. Repeated normalized
    problems are deduplicated and every dropped copy is an explicit exclusion, counted
    in excluded_by_reason. Unsupported task or answer types are explicit exclusions,
    preceding any outcomes or split assignment.
    Formal preparation records development overlaps as explicit exclusions; the
    loader independently recomputes those exclusions and rejects leaked rows.
    """
    if stage not in STAGES:
        raise MathDataError("unknown dataset stage")
    source_path, output_dir = Path(source_path), Path(output_dir)
    if output_dir.exists():
        raise MathDataError("output directory already exists; refuse overwrite")
    answer_validator, question_validator, version = _validators(
        answer_validator, question_validator, answer_contract_version
    )
    questions, source_exclusions, n_source_rows = _source_questions(source_path, metadata)
    development_audit = None
    development = []
    if stage == "formal":
        if metadata.license_status != "approved":
            raise MathDataError("formal source license review is not approved")
        development, development_audit = _development_exclusions(
            Path(development_source_path) if development_source_path is not None else None,
            [Path(path) for path in additional_development_paths],
        )
    eligibility_configuration = _eligibility_configuration(answer_validator, version)
    reference_results: dict[str, str | None] = {}

    def check_reference(answer: str) -> str | None:
        key = _answer_hash(answer)
        if key not in reference_results:
            reference_results[key] = answer_validator(answer)
        return reference_results[key]

    accepted, exclusions = _screen_source(
        questions,
        check_reference,
        question_validator,
        development,
        allow_unknown_original_split=stage == "stub" or metadata.source_kind != "math",
        source_exclusions=source_exclusions,
    )
    if not accepted:
        raise MathDataError("no eligible questions after explicit answer/task screening")
    output_dir.mkdir(parents=True, exist_ok=False)
    _write_rows(output_dir / "public_questions.jsonl", [q.public_payload() for q in accepted])
    _write_rows(output_dir / "references.private.jsonl", [asdict(_reference(q)) for q in accepted])
    _write_rows(output_dir / "exclusions.jsonl", sorted(exclusions, key=lambda q: q["question_id"]))
    _write_rows(
        output_dir / "eligibility.jsonl",
        [
            {"reference_sha256": key, "exclusion_reason": reason}
            for key, reason in sorted(reference_results.items())
        ],
    )
    manifest = {
        "domain": "math",
        "data_contract_version": DATA_CONTRACT_VERSION,
        "answer_contract_version": version,
        "reference_eligibility": {
            "configuration": eligibility_configuration,
            "source_sha256": metadata.source_sha256,
            "n_unique_references": len(reference_results),
        },
        "stage": stage,
        "source": asdict(metadata),
        "source_count": n_source_rows,
        "eligible_count": len(accepted),
        "excluded_count": len(exclusions),
        "excluded_by_reason": dict(sorted(Counter(q["reason"] for q in exclusions).items())),
        "screening": {
            "original_split_required": stage != "stub" and metadata.source_kind == "math",
            "rules": (
                "one copy per normalized problem (conflicting copies all excluded); "
                "no visual/proof tasks; task/answer contract; declared development exclusions"
            ),
        },
        "fingerprint_version": PROBLEM_FINGERPRINT_VERSION,
        "development_exclusion": development_audit,
        "identity_checks": (
            "one problem per ID; bundle IDs and normalized problems unique after "
            "deduplication; source-bound private references"
        ),
        "not_checked": [
            "semantic near-duplicates",
            "model training contamination",
            "gold semantics",
        ],
        "files": {
            name: file_hash(output_dir / name)
            for name in (
                "public_questions.jsonl",
                "references.private.jsonl",
                "exclusions.jsonl",
                "eligibility.jsonl",
            )
        },
    }
    # Written last: absent manifest means preparation never completed.
    _write_json(output_dir / "manifest.json", manifest)
    return manifest


def load_dataset_bundle(
    directory: Path | str,
    *,
    stage: str,
    source_path: Path | str,
    development_source_path: Path | str | None = None,
    additional_development_paths: Sequence[Path | str] = (),
    answer_validator: Validator | None = None,
    question_validator: Validator | None = None,
    answer_contract_version: str | None = None,
) -> DatasetBundle:
    """Fail closed on namespace, provenance, association, eligibility or leakage errors."""
    root = Path(directory)
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (
        stage not in STAGES
        or manifest.get("domain") != "math"
        or manifest.get("stage") != stage
        or manifest.get("data_contract_version") != DATA_CONTRACT_VERSION
        or manifest.get("fingerprint_version") != PROBLEM_FINGERPRINT_VERSION
    ):
        raise MathDataError("dataset stage/domain/contract mismatch")
    answer_validator, question_validator, version = _validators(
        answer_validator, question_validator, answer_contract_version
    )
    if manifest.get("answer_contract_version") != version:
        raise MathDataError("answer contract version mismatch")
    names = {
        "public_questions.jsonl",
        "references.private.jsonl",
        "exclusions.jsonl",
        "eligibility.jsonl",
    }
    if set(manifest.get("files", {})) != names:
        raise MathDataError("dataset file manifest is incomplete or has unexpected files")
    for name, expected in manifest["files"].items():
        if file_hash(root / name) != expected:
            raise MathDataError(f"dataset artifact SHA256 mismatch: {name}")
    metadata = SourceMetadata(**manifest["source"])
    questions, source_exclusions, n_source_rows = _source_questions(Path(source_path), metadata)
    reference_results = _eligibility_evidence(
        root, manifest, _eligibility_configuration(answer_validator, version)
    )
    used_references: set[str] = set()

    def checked_reference(answer: str) -> str | None:
        key = _answer_hash(answer)
        if key not in reference_results:
            raise MathDataError("source reference is missing bound eligibility evidence")
        used_references.add(key)
        return reference_results[key]

    by_id: dict[str, MathQuestion] = {}
    for question in questions:
        # Same-ID copies of one problem: _deduplicate keeps the earliest source row.
        by_id.setdefault(question.question_id, question)
    development = []
    if stage == "formal":
        if metadata.license_status != "approved":
            raise MathDataError("formal source license review is not approved")
        development, audit = _development_exclusions(
            Path(development_source_path) if development_source_path is not None else None,
            [Path(path) for path in additional_development_paths],
        )
        if audit != manifest.get("development_exclusion"):
            raise MathDataError("development exclusion manifest differs from current full sources")
    public = []
    for row in _read_rows(root / "public_questions.jsonl"):
        if set(row) != {"question_id", "problem"}:
            raise MathDataError("public question schema violation; private fields forbidden")
        public.append(PublicQuestion(**row))
    _assert_unique(public)
    references = {}
    for row in _read_rows(root / "references.private.jsonl"):
        try:
            reference = PrivateReference(**row)
        except TypeError as exc:
            raise MathDataError("private reference schema violation") from exc
        if reference.question_id in references:
            raise MathDataError("duplicate private reference ID")
        references[reference.question_id] = reference
    if set(references) != {q.question_id for q in public}:
        raise MathDataError("public/private question identities are not one-to-one")
    for question in public:
        source = by_id.get(question.question_id)
        if source is None or source.problem != question.problem:
            raise MathDataError("public question differs from its original source association")
        if references[question.question_id] != _reference(source):
            raise MathDataError("private reference differs from its original source association")
    accepted, expected_exclusions = _screen_source(
        questions,
        checked_reference,
        question_validator,
        development,
        allow_unknown_original_split=stage == "stub" or metadata.source_kind != "math",
        source_exclusions=source_exclusions,
    )
    eligible = {q.question_id for q in accepted}
    if used_references != set(reference_results):
        raise MathDataError("unused reference eligibility evidence is not source-bound")
    if set(references) != eligible:
        raise MathDataError("bundle eligibility differs from the declared source and screening")
    excluded = _read_rows(root / "exclusions.jsonl")
    if excluded != expected_exclusions:
        raise MathDataError("exclusion audit does not account for every ineligible source row")
    if (
        manifest.get("eligible_count") != len(public)
        or manifest.get("source_count") != n_source_rows
        or manifest.get("excluded_count") != len(excluded)
        or manifest.get("excluded_by_reason")
        != dict(sorted(Counter(q["reason"] for q in excluded).items()))
    ):
        raise MathDataError("dataset row counts differ from manifest")
    if stage == "formal":
        assert_disjoint_question_groups(development, [q.public_payload() for q in public])
    return DatasetBundle(
        tuple(public),
        MappingProxyType(references),
        MappingProxyType(manifest),
        file_hash(manifest_path),
    )
