"""Shared machinery for the real V1 / V2 candidates (II.4.1, App. A.2).

Everything numeric in this module is plain Python. A *backend* turns texts into rows of raw
logits (plain floats); this module chunks the input, normalises those rows into a score in
[0, 1] and hands the scores back in input order. ``torch`` and ``transformers`` are imported
lazily inside :mod:`rafr.verifiers.candidates.torch_backend` and nowhere else, so importing
(and testing) a candidate needs neither GPU dependencies nor the network.

Contract: ``src/rafr/contracts/score_spec.md``. Scores are in [0, 1], higher = more toxic,
deterministic in ``(model_version, text)``, text-only.
"""

from __future__ import annotations

import math
import os
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, replace
from typing import Protocol, runtime_checkable

from rafr.contracts.schemas import SCORE_MAX, SCORE_MIN

#: Internal inference batch size. Purely an engineering knob: every batch is padded to a
#: fixed shape (see :class:`Truncation` and the torch backend), so changing it cannot change
#: a score. Resolution order: explicit argument > ``RAFR_VERIFIER_BATCH_SIZE`` > this value.
DEFAULT_INFERENCE_BATCH_SIZE = 8
BATCH_SIZE_ENV = "RAFR_VERIFIER_BATCH_SIZE"

#: Contract-level cap on one ``score()`` call (score_spec.md). Generous on purpose: the
#: memory-relevant number is the inference batch size above.
DEFAULT_BATCH_LIMIT = 4096

#: Marker for a revision that ``env/download_models.sh`` never pinned. Formal scoring
#: refuses it (see :mod:`rafr.verifiers.batch_score`).
UNPINNED = "unpinned"

#: Set to 1 to trade the cross-batch determinism guarantee for speed. Read once, when the
#: candidate is built, and turned into the :class:`Truncation` flags below, so the verifier
#: card always states what actually happened rather than what the default would have been.
DYNAMIC_PADDING_ENV = "RAFR_VERIFIER_DYNAMIC_PADDING"


def dynamic_padding_requested() -> bool:
    return os.environ.get(DYNAMIC_PADDING_ENV, "").strip() not in ("", "0", "false", "False")


class VerifierConfigError(RuntimeError):
    """A candidate is wired up in a way that cannot produce a well-defined score."""


# ---------------------------------------------------------------------------------------
# pure numerics
# ---------------------------------------------------------------------------------------


def sigmoid(x: float) -> float:
    """Logistic function, overflow-safe on both tails."""
    if x >= 0.0:
        return 1.0 / (1.0 + math.exp(-x))
    e = math.exp(x)
    return e / (1.0 + e)


def softmax(xs: Sequence[float]) -> list[float]:
    """Softmax with the usual max-subtraction. ``softmax([x]) == [1.0]``."""
    if not xs:
        raise ValueError("softmax of an empty vector")
    m = max(xs)
    exps = [math.exp(x - m) for x in xs]
    total = math.fsum(exps)
    if total <= 0.0 or not math.isfinite(total):
        raise VerifierConfigError(f"softmax did not normalise: logits={list(xs)!r}")
    return [e / total for e in exps]


def logsumexp(xs: Sequence[float]) -> float:
    """``log(sum(exp(x)))``, overflow-safe.

    Used to fold the several token ids that spell one label (``"unsafe"`` vs ``" unsafe"``
    vs ``"Unsafe"``) into a single logit: in probability space this is exactly the sum of
    those tokens' unnormalised masses, which is what the vendors' own recipes add up.
    """
    if not xs:
        raise ValueError("logsumexp of an empty vector")
    m = max(xs)
    if not math.isfinite(m):
        raise VerifierConfigError(f"non-finite logit in {list(xs)!r}")
    return m + math.log(math.fsum(math.exp(x - m) for x in xs))


def clamp_score(value: float, *, where: str) -> float:
    """Guard the score_spec range. NaN / inf are contract violations, so they raise."""
    v = float(value)
    if not math.isfinite(v):
        raise VerifierConfigError(f"{where} produced a non-finite score: {value!r}")
    return float(min(SCORE_MAX, max(SCORE_MIN, v)))


# ---------------------------------------------------------------------------------------
# truncation
# ---------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Truncation:
    """How a text longer than the model's window is cut (recorded in the verifier card).

    ``max_length`` counts tokens of *this model's* tokenizer, special tokens included.
    ``side="right"`` keeps the head of the text and drops the tail, which is what every
    HuggingFace tokenizer does by default; ``side="left"`` keeps the tail.

    ``pad_to_max_length`` and ``fixed_batch_shape`` exist for determinism rather than for
    speed: score_spec.md requires the same ``(model_version, text)`` to give the same score
    bit-for-bit *across calls and processes*, and a float reduction order that depends on
    the padded sequence length, or on how full the last batch happens to be, would break
    that. Both cost throughput; neither can change a score.
    """

    max_length: int
    side: str = "right"
    pad_to_max_length: bool = True
    fixed_batch_shape: bool = True

    def __post_init__(self) -> None:
        if self.max_length <= 0:
            raise VerifierConfigError(f"max_length must be positive, got {self.max_length}")
        if self.side not in ("right", "left"):
            raise VerifierConfigError(f"truncation side must be right|left, got {self.side!r}")

    def describe(self) -> str:
        kept = "head" if self.side == "right" else "tail"
        pad = "every batch padded to max_length" if self.pad_to_max_length else "dynamic padding"
        shape = "fixed batch shape" if self.fixed_batch_shape else "ragged last batch"
        return (
            f"truncate to {self.max_length} tokens of this model's tokenizer "
            f"(special tokens included), keeping the {kept} of the text; {pad}; {shape}"
        )


# ---------------------------------------------------------------------------------------
# backend
# ---------------------------------------------------------------------------------------


@runtime_checkable
class RawScoreBackend(Protocol):
    """Turns texts into rows of raw logits. The only place torch may appear.

    ``revision`` is the commit the weights were actually loaded from; it goes into
    ``model_version`` and therefore into every row of the scores table (App. A.2).
    ``raw_rows`` must preserve input order and return one row per text. What a row *means*
    is fixed by the candidate: class logits for an encoder, one logit per label-token group
    for a guard LM.
    """

    revision: str

    def raw_rows(self, items: Sequence[object]) -> list[list[float]]: ...


# ---------------------------------------------------------------------------------------
# base verifier
# ---------------------------------------------------------------------------------------


class BaseVerifier:
    """Chunking, ordering and range enforcement shared by every real candidate.

    Subclasses implement :meth:`normalise_row` (pure) and :meth:`build_backend` (lazy).
    """

    #: Human-readable, copied verbatim into the verifier card (App. A.2).
    score_semantics: str = ""
    normalization: str = ""

    def __init__(
        self,
        *,
        verifier_id: str,
        repo_id: str,
        revision: str | None,
        truncation: Truncation,
        batch_limit: int = DEFAULT_BATCH_LIMIT,
        inference_batch_size: int | None = None,
        backend: RawScoreBackend | None = None,
    ) -> None:
        self.verifier_id = verifier_id
        self.repo_id = repo_id
        self.revision = revision or UNPINNED
        self.truncation = truncation
        self.batch_limit = int(batch_limit)
        self.inference_batch_size = resolve_batch_size(inference_batch_size)
        if self.inference_batch_size > self.batch_limit:
            raise VerifierConfigError(
                f"inference_batch_size={self.inference_batch_size} exceeds "
                f"batch_limit={self.batch_limit}"
            )
        self._backend = backend

    # -- identity -----------------------------------------------------------------------

    @property
    def model_version(self) -> str:
        """``<hf repo>@<commit sha>`` (App. A.2). ``@unpinned`` until the download manifest
        pins a revision; formal scoring refuses that."""
        return f"{self.repo_id}@{self.revision}"

    @property
    def is_pinned(self) -> bool:
        return self.revision != UNPINNED

    # -- backend ------------------------------------------------------------------------

    def build_backend(self) -> RawScoreBackend:
        raise NotImplementedError

    @property
    def backend(self) -> RawScoreBackend:
        """Built on first use, never at import time."""
        if self._backend is None:
            self._backend = self.build_backend()
        return self._backend

    # -- scoring ------------------------------------------------------------------------

    def normalise_row(self, row: Sequence[float]) -> float:
        raise NotImplementedError

    def prepare(self, texts: Sequence[str]) -> list[object]:
        """Texts -> whatever the backend consumes, one item per text, same order.

        An encoder consumes the text itself; a guard LM consumes a finished prompt (built in
        :mod:`rafr.verifiers.candidates.guard_lm`, which is pure).
        """
        return list(texts)

    def score(self, texts: list[str]) -> list[float]:
        """score_spec.md: same length, same order, every value in [0, 1], higher = more toxic."""
        if isinstance(texts, str):
            raise TypeError("score() takes a list of texts, not a single string")
        items = list(texts)
        if not items:
            return []
        if any(not isinstance(t, str) for t in items):
            raise TypeError("score() takes a list of str")
        if len(items) > self.batch_limit:
            raise ValueError(f"batch of {len(items)} exceeds batch_limit={self.batch_limit}")

        out: list[float] = []
        for chunk in chunked(items, self.inference_batch_size):
            prepared = self.prepare(chunk)
            if len(prepared) != len(chunk):
                raise VerifierConfigError(
                    f"{self.verifier_id}: prepare() returned {len(prepared)} items for "
                    f"{len(chunk)} texts"
                )
            rows = self.backend.raw_rows(prepared)
            if len(rows) != len(chunk):
                raise VerifierConfigError(
                    f"{self.verifier_id}: backend returned {len(rows)} rows for {len(chunk)} texts"
                )
            for row in rows:
                out.append(clamp_score(self.normalise_row(row), where=self.verifier_id))
        return out


def chunked(items: Sequence[str], size: int) -> list[list[str]]:
    """Split preserving order. The concatenation of the chunks is the input."""
    if size <= 0:
        raise VerifierConfigError(f"batch size must be positive, got {size}")
    return [list(items[i : i + size]) for i in range(0, len(items), size)]


def resolve_batch_size(explicit: int | None) -> int:
    """Explicit argument > ``RAFR_VERIFIER_BATCH_SIZE`` > :data:`DEFAULT_INFERENCE_BATCH_SIZE`."""
    if explicit is not None:
        value = int(explicit)
    else:
        raw = os.environ.get(BATCH_SIZE_ENV, "").strip()
        if not raw:
            return DEFAULT_INFERENCE_BATCH_SIZE
        try:
            value = int(raw)
        except ValueError as exc:
            raise VerifierConfigError(f"{BATCH_SIZE_ENV}={raw!r} is not an integer") from exc
    if value <= 0:
        raise VerifierConfigError(f"batch size must be positive, got {value}")
    return value


# ---------------------------------------------------------------------------------------
# padding mode: switching it, and measuring what it costs
# ---------------------------------------------------------------------------------------


@contextmanager
def padding_mode(verifier: BaseVerifier, *, fixed: bool) -> Iterator[None]:
    """Run a block with the two padding flags forced on or off, then restore them.

    The flags live on :class:`Truncation`, which the backend reads per call, so this can
    switch modes on an already-loaded model instead of paying to load the weights twice.
    """
    variant = replace(verifier.truncation, pad_to_max_length=fixed, fixed_batch_shape=fixed)
    targets = [verifier]
    backend = verifier._backend  # noqa: SLF001 - same package, and loading here is the point
    if backend is not None and hasattr(backend, "truncation"):
        targets.append(backend)
    saved = [t.truncation for t in targets]
    try:
        for t in targets:
            t.truncation = variant
        yield
    finally:
        for t, old in zip(targets, saved, strict=True):
            t.truncation = old


@dataclass(frozen=True)
class PaddingComparison:
    """What the fixed-shape determinism guarantee is worth on a concrete batch.

    ``worst_case_flips`` needs no threshold: a judgment y = 1[s >= tau] differs between the
    two modes exactly when ``min(s_fixed, s_dynamic) < tau <= max(...)``, so this is the
    largest number of items any single tau could flip. At selection time tau is not fitted
    yet (step 7 comes after step 6), which is why the threshold-free number is the one to
    read. Zero everywhere means the two modes agree on every judgment for every possible
    threshold.
    """

    n: int
    max_abs_diff: float
    mean_abs_diff: float
    n_differing: int
    threshold: float | None
    n_flips_at_threshold: int | None
    worst_case_flips: int
    worst_case_threshold: float | None

    def clean(self) -> bool:
        return self.worst_case_flips == 0


def compare_scores(
    fixed: Sequence[float], dynamic: Sequence[float], threshold: float | None = None
) -> PaddingComparison:
    """Compare the same texts scored under both padding modes. Pure; no torch."""
    if len(fixed) != len(dynamic):
        raise VerifierConfigError(f"padding comparison got {len(fixed)} and {len(dynamic)} scores")
    n = len(fixed)
    if n == 0:
        return PaddingComparison(
            0, 0.0, 0.0, 0, threshold, None if threshold is None else 0, 0, None
        )

    diffs = [abs(float(a) - float(b)) for a, b in zip(fixed, dynamic, strict=True)]
    spans = [
        (min(float(a), float(b)), max(float(a), float(b)))
        for a, b in zip(fixed, dynamic, strict=True)
    ]

    flips_at: int | None = None
    if threshold is not None:
        flips_at = sum(1 for lo, hi in spans if lo < threshold <= hi)

    # (lo, hi] is the set of thresholds that flip that item, so the maximum overlap is
    # attained at one of the hi values.
    worst, worst_tau = 0, None
    for _, hi in spans:
        count = sum(1 for lo2, hi2 in spans if lo2 < hi <= hi2)
        if count > worst:
            worst, worst_tau = count, hi

    return PaddingComparison(
        n=n,
        max_abs_diff=max(diffs),
        mean_abs_diff=math.fsum(diffs) / n,
        n_differing=sum(1 for lo, hi in spans if lo != hi),
        threshold=threshold,
        n_flips_at_threshold=flips_at,
        worst_case_flips=worst,
        worst_case_threshold=worst_tau,
    )


def pick_label_index(id2label: dict[int, str], wanted: str, *, where: str) -> int:
    """Resolve a class by NAME, never by position.

    The ordering of a classification head is a property of the checkpoint, so hard-coding an
    index is how a wrapper silently scores the wrong class after a repo update.
    """
    table = {int(i): str(name) for i, name in id2label.items()}
    matches = [i for i, name in sorted(table.items()) if name.lower() == wanted.lower()]
    if len(matches) != 1:
        raise VerifierConfigError(
            f"{where}: expected exactly one class named {wanted!r} in id2label="
            f"{dict(sorted(table.items()))}, found {len(matches)}"
        )
    return matches[0]
