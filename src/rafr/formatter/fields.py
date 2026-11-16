"""B's six fields (paper Sec. 3): deterministic transforms of (s1_obs, s2_obs, tau1, tau2)."""

from __future__ import annotations

from rafr.contracts.schemas import ConflictDirection, Observation, UncertaintyFields


def compute_fields(obs: Observation) -> UncertaintyFields:
    y1 = int(obs.s1_obs >= obs.tau1)
    y2 = int(obs.s2_obs >= obs.tau2)
    m1 = obs.s1_obs - obs.tau1  # signed decision margin; no cross-verifier normalization
    m2 = obs.s2_obs - obs.tau2
    agreement = int(y1 == y2)
    if agreement:
        direction = ConflictDirection.NA
    elif y1 == 1:
        direction = ConflictDirection.V1_UNSAFE_V2_SAFE
    else:
        direction = ConflictDirection.V1_SAFE_V2_UNSAFE
    return UncertaintyFields(
        y1=y1, y2=y2, m1=m1, m2=m2, agreement=agreement, conflict_direction=direction
    )
