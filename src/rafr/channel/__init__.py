"""Evidence-degradation channel C_lambda (II.5) and its interception point (II.10).

A channel is a pure function ``(state_id, s1, s2, lam) -> (s1_tilde, s2_tilde)``. Its
signature is the enforcement of II.5.2: it receives the two verifier scores and the
``(state_id, lambda)`` randomness key, and nothing else — no y*, no c_t, no E, no arm.
Randomness keyed by ``(state_id, lambda)`` also guarantees II.5.1: every arm sees the same
perturbed scores for the same state and lambda.

This package must never import ``rafr.annotation``, ``rafr.data.labels`` or
``rafr.evaluator`` (tests/test_boundaries.py).
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from rafr.contracts.schemas import Lambda, Observation

NO_PERTURBATION: dict[str, Any] = {
    "perturbed_v1": False,
    "perturbed_v2": False,
    "z": "none",
    "replacement_s1": None,
    "replacement_s2": None,
}
"""What a channel without perturbation reports (II.10 log fields)."""


def channel_assignment(channel: object, state_id: str, lam: Lambda) -> dict[str, Any]:
    """II.10: which verifier was perturbed, the derived z mode, and the replacement scores.

    A channel may expose ``assignment(state_id, lam)``; one that does not (identity, test
    doubles) perturbs nothing.
    """
    fn = getattr(channel, "assignment", None)
    return dict(fn(state_id, lam)) if callable(fn) else dict(NO_PERTURBATION)


@runtime_checkable
class Channel(Protocol):
    name: str

    def __call__(self, state_id: str, s1: float, s2: float, lam: Lambda) -> tuple[float, float]: ...


class ObservationBuilder:
    """The interception point: raw scores -> channel -> Observation (II.1.2, II.10).

    Results are memoized by ``(state_id, lam)`` so that the three arms of Exp 1 receive the
    identical ``(s1_tilde, s2_tilde)`` (II.5.1) and so that the pair is computed exactly once
    per state and level.
    """

    def __init__(self, channel: Channel, tau1: float, tau2: float, v1_id: str, v2_id: str) -> None:
        self.channel = channel
        self.tau1 = tau1
        self.tau2 = tau2
        self.v1_id = v1_id
        self.v2_id = v2_id
        self._memo: dict[tuple[str, Lambda], tuple[float, float]] = {}

    def assignment(self, state_id: str, lam: Lambda) -> dict[str, Any]:
        """The channel's perturbation assignment for this (state, lambda), for the log."""
        return channel_assignment(self.channel, state_id, lam)

    def build(self, state_id: str, s1: float, s2: float, lam: Lambda) -> Observation:
        key = (state_id, lam)
        if key not in self._memo:
            self._memo[key] = self.channel(state_id, s1, s2, lam)
        t1, t2 = self._memo[key]
        return Observation(
            state_id=state_id,
            s1_obs=t1,
            s2_obs=t2,
            tau1=self.tau1,
            tau2=self.tau2,
            lam=lam,
            raw_s1=s1,
            raw_s2=s2,
            v1_id=self.v1_id,
            v2_id=self.v2_id,
        )
