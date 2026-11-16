"""channel: signature cannot receive labels; identity; memo by (state, lambda) across arms."""

from __future__ import annotations

import inspect

from rafr.channel import Channel, ObservationBuilder
from rafr.channel.identity import IdentityChannel
from rafr.contracts.schemas import Lambda

FORBIDDEN_PARAM_WORDS = ("label", "y_star", "ystar", "c_t", "correct", "eval", "arm", "truth")


def test_identity_channel_is_identity_and_conforms():
    ch = IdentityChannel()
    assert isinstance(ch, Channel)
    assert ch("s", 0.3, 0.9, Lambda.NONE) == (0.3, 0.9)
    assert ch("s", 0.3, 0.9, Lambda.HIGH) == (0.3, 0.9)


def test_channel_signature_has_no_label_or_arm_inputs():
    params = list(inspect.signature(IdentityChannel.__call__).parameters)[1:]
    assert params == ["state_id", "s1", "s2", "lam"]
    for p in params:
        assert not any(w in p.lower() for w in FORBIDDEN_PARAM_WORDS)


class _CountingChannel:
    """Stand-in for C_lambda: deterministic perturbation, counts calls."""

    name = "counting"

    def __init__(self) -> None:
        self.calls: list[tuple[str, Lambda]] = []

    def __call__(self, state_id: str, s1: float, s2: float, lam: Lambda) -> tuple[float, float]:
        self.calls.append((state_id, lam))
        delta = {Lambda.LOW: 0.05, Lambda.MID: 0.1, Lambda.HIGH: 0.2}.get(lam, 0.0)
        return (min(1.0, s1 + delta), max(0.0, s2 - delta))


def test_observation_builder_shares_perturbation_across_arms_and_logs_raw():
    ch = _CountingChannel()
    b = ObservationBuilder(ch, tau1=0.5, tau2=0.5, v1_id="v1", v2_id="v2")
    seen = []
    for _arm in ("A1", "B", "permuted_B"):  # arms are irrelevant to the channel by design
        for lam in (Lambda.LOW, Lambda.MID, Lambda.HIGH):
            seen.append(b.build("s1", 0.4, 0.6, lam))
    # exactly one channel call per (state, lambda)
    assert sorted(ch.calls) == sorted(
        [("s1", lam) for lam in (Lambda.LOW, Lambda.MID, Lambda.HIGH)]
    )
    by_lam: dict[Lambda, set[tuple[float, float]]] = {}
    for o in seen:
        by_lam.setdefault(o.lam, set()).add((o.s1_obs, o.s2_obs))
    assert all(len(v) == 1 for v in by_lam.values())  # identical across arms
    assert all(o.raw_s1 == 0.4 and o.raw_s2 == 0.6 for o in seen)  # pre-channel kept
    assert seen[0].s1_obs != seen[0].raw_s1  # and post-channel differs
