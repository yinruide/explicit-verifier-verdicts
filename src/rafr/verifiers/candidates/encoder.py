"""Encoder classifier candidates (II.4.1: "encoder classifier" half of the heterogeneous pair).

Two shapes of head are covered, because the two candidates differ:

``multilabel_sigmoid``
    One independent BCE logit per label (``unitary/toxic-bert``: toxic, severe_toxic,
    obscene, threat, insult, identity_hate). The score is ``sigmoid`` of the *toxic* logit
    alone; the other five are ignored.

``single_label_softmax``
    One softmax over the classes (``s-nlp/roberta_toxicity_classifier``: neutral / toxic).
    The score is the softmax probability of the *toxic* class. For a two-class head this is
    the same number as ``sigmoid(logit_toxic - logit_neutral)``.

In both cases the class is resolved by NAME through ``config.id2label`` (see
:func:`rafr.verifiers.candidates.base.pick_label_index`), never by position.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol, runtime_checkable

from rafr.verifiers.candidates.base import (
    BaseVerifier,
    Truncation,
    VerifierConfigError,
    pick_label_index,
    sigmoid,
    softmax,
)

MULTILABEL_SIGMOID = "multilabel_sigmoid"
SINGLE_LABEL_SOFTMAX = "single_label_softmax"
HEADS = (MULTILABEL_SIGMOID, SINGLE_LABEL_SOFTMAX)


@runtime_checkable
class EncoderBackend(Protocol):
    """A loaded sequence-classification model. ``id2label`` comes from its ``config``."""

    revision: str
    id2label: dict[int, str]

    def raw_rows(self, texts: Sequence[str]) -> list[list[float]]: ...


class EncoderToxicityVerifier(BaseVerifier):
    """score_spec.md-conformant wrapper around a HuggingFace text-classification checkpoint."""

    def __init__(
        self,
        *,
        verifier_id: str,
        repo_id: str,
        revision: str | None,
        head: str,
        toxic_label: str,
        truncation: Truncation,
        dtype: str = "float32",
        **kwargs: object,
    ) -> None:
        if head not in HEADS:
            raise VerifierConfigError(f"head must be one of {HEADS}, got {head!r}")
        super().__init__(
            verifier_id=verifier_id,
            repo_id=repo_id,
            revision=revision,
            truncation=truncation,
            **kwargs,  # type: ignore[arg-type]
        )
        self.head = head
        self.toxic_label = toxic_label
        self.dtype = dtype
        self._toxic_index: int | None = None

    # -- card text ----------------------------------------------------------------------

    @property
    def score_semantics(self) -> str:
        if self.head == MULTILABEL_SIGMOID:
            return (
                f"the single independent BCE logit of the class named {self.toxic_label!r}, "
                "resolved by name through config.id2label; the other label logits are not read"
            )
        return (
            f"the softmax probability of the class named {self.toxic_label!r}, "
            "resolved by name through config.id2label"
        )

    @property
    def normalization(self) -> str:
        if self.head == MULTILABEL_SIGMOID:
            return "sigmoid(logit_toxic); already in [0, 1], no further rescaling"
        return "softmax over all class logits, then take the toxic class; no further rescaling"

    # -- backend ------------------------------------------------------------------------

    def build_backend(self) -> EncoderBackend:
        from rafr.verifiers.candidates.torch_backend import TorchEncoderBackend

        return TorchEncoderBackend(
            repo_id=self.repo_id,
            revision=None if not self.is_pinned else self.revision,
            truncation=self.truncation,
            batch_size=self.inference_batch_size,
            dtype=self.dtype,
        )

    @property
    def toxic_index(self) -> int:
        if self._toxic_index is None:
            self._toxic_index = pick_label_index(
                self.backend.id2label, self.toxic_label, where=self.verifier_id
            )
        return self._toxic_index

    # -- pure normalisation -------------------------------------------------------------

    def normalise_row(self, row: Sequence[float]) -> float:
        idx = self.toxic_index
        if idx >= len(row):
            raise VerifierConfigError(
                f"{self.verifier_id}: class index {idx} out of range for a row of {len(row)} logits"
            )
        if self.head == MULTILABEL_SIGMOID:
            return sigmoid(float(row[idx]))
        return softmax([float(x) for x in row])[idx]
