"""Real V1 / V2 candidates (II.4.1, App. A.2).

Layout::

    base.py           chunking, ordering, range enforcement, pure numerics
    encoder.py        encoder classifiers (sigmoid / softmax over a classification head)
    guard_lm.py       guard LMs (label-token logits, one forward pass, nothing generated)
    registry.py       the candidate pool, the download manifest, the impl factories
    torch_backend.py  the only module that imports torch / transformers, and it is lazy

Importing this package loads no weights and needs no GPU dependencies. The factories listed
in :data:`rafr.verifiers.candidates.registry.FACTORIES` are what
``configs/models.yaml: verifiers.v{1,2}.impl`` points at once the authors pick V1 and V2.
"""

from __future__ import annotations

from rafr.verifiers.candidates.base import (
    BaseVerifier,
    RawScoreBackend,
    Truncation,
    VerifierConfigError,
)
from rafr.verifiers.candidates.encoder import EncoderToxicityVerifier
from rafr.verifiers.candidates.guard_lm import GuardLMVerifier, LabelGroup
from rafr.verifiers.candidates.registry import CANDIDATES, FACTORIES, build, names, spec

__all__ = [
    "CANDIDATES",
    "FACTORIES",
    "BaseVerifier",
    "EncoderToxicityVerifier",
    "GuardLMVerifier",
    "LabelGroup",
    "RawScoreBackend",
    "Truncation",
    "VerifierConfigError",
    "build",
    "names",
    "spec",
]
