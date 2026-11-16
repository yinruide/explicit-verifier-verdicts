"""Deterministic stub verifier for the dry run (II.4.4) and tests.

Implements the score() contract of ``src/rafr/contracts/score_spec.md`` exactly:
scores in [0, 1], higher = more toxic, deterministic in (model_version, text), text-only.
Numbers are meaningless: a hash of the text plus a small bump for a fixed word list, so that
two stubs with different salts disagree on some inputs. ``verifier_id`` starts with ``stub-``
and formal steps refuse such a verifier.
"""

from __future__ import annotations

import hashlib

from rafr.contracts.schemas import SCORE_MAX, SCORE_MIN
from rafr.generation.validity import normalize_ws

STUB_PREFIX = "stub-"
_LEXICON = (
    "idiot",
    "moron",
    "stupid",
    "loser",
    "trash",
    "garbage",
    "scum",
    "dumb",
    "pathetic",
    "clown",
)


class StubVerifier:
    def __init__(
        self, verifier_id: str = "stub-v1", salt: str = "v1", batch_limit: int = 64
    ) -> None:
        if not verifier_id.startswith(STUB_PREFIX):
            raise ValueError(f"stub verifier ids must start with {STUB_PREFIX!r}")
        self.verifier_id = verifier_id
        self.model_version = f"stub-{salt}-0"
        self.batch_limit = batch_limit
        self._salt = salt

    def _one(self, text: str) -> float:
        norm = normalize_ws(text).lower()
        h = hashlib.sha256(f"{self._salt}|{norm}".encode()).digest()
        base = int.from_bytes(h[:8], "big") / 2**64  # uniform in [0, 1)
        bump = 0.25 * sum(1 for w in _LEXICON if w in norm)
        score = 0.6 * base + min(bump, 0.4)
        return float(min(SCORE_MAX, max(SCORE_MIN, score)))

    def score(self, texts: list[str]) -> list[float]:
        if len(texts) > self.batch_limit:
            raise ValueError(f"batch of {len(texts)} exceeds batch_limit={self.batch_limit}")
        return [self._one(t) for t in texts]


def is_stub(verifier_id: str) -> bool:
    return verifier_id.startswith(STUB_PREFIX)


def stub_pair() -> tuple[StubVerifier, StubVerifier]:
    return StubVerifier("stub-v1", salt="v1"), StubVerifier("stub-v2", salt="v2")


ALWAYS_TOXIC_IDS: tuple[str, str] = ("stub-always-toxic-v1", "stub-always-toxic-v2")


class ConstantStubVerifier:
    """A stub that gives every text the same score -- for Exp 2 smoke runs only.

    ``always_toxic_pair`` (score = SCORE_MAX for every text) keeps both checkers flagging every
    text, so that a smoke run's trajectories are pushed through all K rounds and rounds 3 and 4
    make real controller calls on the pod (2026-09-24). The ids start with ``stub-``:
    every formal step refuses them.
    """

    def __init__(self, verifier_id: str, score: float, batch_limit: int = 64) -> None:
        if not verifier_id.startswith(STUB_PREFIX):
            raise ValueError(f"stub verifier ids must start with {STUB_PREFIX!r}")
        if not (SCORE_MIN <= float(score) <= SCORE_MAX):
            raise ValueError(f"score must be in [{SCORE_MIN}, {SCORE_MAX}], got {score}")
        self.verifier_id = verifier_id
        self.config_id = verifier_id
        self.constant = float(score)
        self.model_version = f"stub-constant-{self.constant!r}"
        self.batch_limit = batch_limit

    def score(self, texts: list[str]) -> list[float]:
        if isinstance(texts, str):
            raise TypeError("score() takes a list of texts, not a single string")
        if len(texts) > self.batch_limit:
            raise ValueError(f"batch of {len(texts)} exceeds batch_limit={self.batch_limit}")
        return [self.constant for _ in texts]

    def describe(self) -> dict[str, object]:
        return {
            "config_id": self.verifier_id,
            "stub": True,
            "constant_score": self.constant,
            "smoke_only": True,
            "note": "always-toxic stub: forces Exp 2 smoke trajectories through K rounds",
        }

    def close(self) -> None:
        return None


def always_toxic_pair() -> tuple[ConstantStubVerifier, ConstantStubVerifier]:
    return (
        ConstantStubVerifier(ALWAYS_TOXIC_IDS[0], SCORE_MAX),
        ConstantStubVerifier(ALWAYS_TOXIC_IDS[1], SCORE_MAX),
    )
