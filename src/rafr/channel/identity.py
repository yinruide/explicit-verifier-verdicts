"""Identity channel: ``s_obs = s`` (Exp 2, II.1.3; and lambda = none in Exp 1 diagnostics)."""

from __future__ import annotations

from typing import Any

from rafr.channel import NO_PERTURBATION
from rafr.contracts.schemas import Lambda


class IdentityChannel:
    name = "identity"

    def __call__(self, state_id: str, s1: float, s2: float, lam: Lambda) -> tuple[float, float]:
        return float(s1), float(s2)

    def assignment(self, state_id: str, lam: Lambda) -> dict[str, Any]:
        """Nothing is ever perturbed here (II.10 log fields)."""
        return dict(NO_PERTURBATION)
