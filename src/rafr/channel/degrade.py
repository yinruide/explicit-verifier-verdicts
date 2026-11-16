"""C_lambda — observation-conditioned evidence degradation (II.5).

    C_lambda = ( p_lambda^(1), p_lambda^(2), q )     (II.5.1, v0.7)

Decisions (group 6a/6b, 2026-09-16) fixing the geometry and the assignment:

* **q = independent resampling.** A perturbed verifier's score is redrawn from that verifier's
  empirical score distribution on the unlabeled pool (``margin_fit``). The judgment may or
  may not flip; nothing is forced across the threshold (naive reflection stays excluded, C.2).
* **Per-verifier, nested assignment.** Each level lambda has one perturbation probability per
  verifier, ``p_lambda^(i)``, non-decreasing in the level. They are FIXED ex ante (protocol
  II.5.1, C.27; 2026-09-26): p = 0 / 0.5 / 1 at low / mid / high, the same for both
  verifiers, read from ``configs/channel.yaml: degrade.levels`` by :func:`levels_from_config`,
  which refuses any other values. Nothing is solved from data or hand-filled; delta_R is only
  the gate's threshold and step08's reachability check. For every ``(state, verifier)`` a
  single uniform ``u`` is drawn; verifier i is perturbed at level lambda iff
  ``u < p_lambda^(i)``. The set perturbed at a higher level therefore CONTAINS the set
  perturbed at a lower level (nested coupling). The replacement score is also drawn once per
  ``(state, verifier)``, so a verifier perturbed at ``mid`` shows the same replaced score at
  ``high``.
* The former four z modes (v1_only / v2_only / both / none) are no longer parameters; their
  implied probabilities are derived from ``(p^(1), p^(2))`` under independence and written to
  ``frozen/channel.json`` / the step08 report for the record.

Per call the channel is keyed by ``(state_id, lambda)`` and never by arm (II.5.1): all three
arms see the same ``(s1~, s2~)``. Inputs are the two scores and the key only; the function
cannot see y*, c_t or E (II.5.2) and never normalizes across verifiers.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from rafr.channel.margin_fit import MarginModel
from rafr.contracts.schemas import SCORE_MAX, SCORE_MIN, TBD, Lambda
from rafr.core.seeding import rng as seeded_rng
from rafr.core.textio import crlf_rewrite_notice

VERIFIERS = ("v1", "v2")
LADDER = (Lambda.LOW, Lambda.MID, Lambda.HIGH)
MODES = ("v1_only", "v2_only", "both", "none")  # derived, for the record only


class DegradeConfigError(ValueError):
    pass


@dataclass(frozen=True)
class LevelParams:
    """Perturbation probability per verifier at one level: ``p_lambda^(1), p_lambda^(2)``."""

    p_v1: float
    p_v2: float

    def __post_init__(self) -> None:
        for name, p in (("v1", self.p_v1), ("v2", self.p_v2)):
            if not (0.0 <= p <= 1.0):
                raise DegradeConfigError(f"p.{name} must lie in [0, 1], got {p}")

    @classmethod
    def from_config(cls, d: dict[str, Any]) -> LevelParams:
        vals = {}
        for k in VERIFIERS:
            v = d.get(k, TBD)
            if v == TBD or v is None or isinstance(v, bool):
                raise DegradeConfigError(f"p.{k} is TBD")
            vals[k] = float(v)
        return cls(vals["v1"], vals["v2"])

    def p(self, verifier: str) -> float:
        return self.p_v1 if verifier == "v1" else self.p_v2

    def derived_mode_probs(self) -> dict[str, float]:
        """Implied P(z) under independent per-verifier perturbation (record only)."""
        a, b = self.p_v1, self.p_v2
        return {
            "v1_only": round(a * (1 - b), 9),
            "v2_only": round((1 - a) * b, 9),
            "both": round(a * b, 9),
            "none": round((1 - a) * (1 - b), 9),
        }

    def to_dict(self) -> dict[str, float]:
        return {"v1": self.p_v1, "v2": self.p_v2}


def z_mode(p1: bool, p2: bool) -> str:
    """The perturbation mode z derived from the two per-verifier draws (II.5.1: z is no
    longer a parameter, it is read off the assignment and logged)."""
    if p1 and p2:
        return "both"
    if p1:
        return "v1_only"
    if p2:
        return "v2_only"
    return "none"


def check_monotone(levels: dict[Lambda, LevelParams]) -> None:
    """p_lambda^(i) must be non-decreasing along low -> mid -> high (nesting requirement)."""
    missing = [lam for lam in LADDER if lam not in levels]
    if missing:
        raise DegradeConfigError(f"missing channel levels: {[m.value for m in missing]}")
    for v in VERIFIERS:
        ps = [levels[lam].p(v) for lam in LADDER]
        if not (ps[0] <= ps[1] <= ps[2]):
            raise DegradeConfigError(
                f"p^({v}) must be non-decreasing over low <= mid <= high, got {ps}"
            )


class DegradeChannel:
    name = "degrade"

    def __init__(
        self,
        levels: dict[Lambda, LevelParams],
        q: dict[str, MarginModel],  # {"v1": model, "v2": model}
        tau1: float,
        tau2: float,
        seed: int,
    ) -> None:
        for v in VERIFIERS:
            if v not in q:
                raise DegradeConfigError("q needs margin models for 'v1' and 'v2'")
        check_monotone(levels)
        self.levels = levels
        self.q = q
        self.tau1 = float(tau1)
        self.tau2 = float(tau2)
        self.seed = int(seed)

    # ---- the per-(state, verifier) draws --------------------------------------------
    def draws(self, state_id: str) -> dict[str, tuple[float, float]]:
        """``{"v1": (u, replacement_score), "v2": (u, replacement_score)}`` for this state.

        Keyed by ``(seed, state_id, verifier)`` only — independent of lambda and arm — which
        is what makes the perturbed sets nested across levels and identical across arms.
        """
        out: dict[str, tuple[float, float]] = {}
        for v in VERIFIERS:
            r = seeded_rng(self.seed, "channel", state_id, v)
            u = float(r.random())
            s_new = self.q[v].sample_score(r)
            out[v] = (u, float(min(SCORE_MAX, max(SCORE_MIN, s_new))))
        return out

    def perturbed(self, state_id: str, lam: Lambda) -> dict[str, bool]:
        """Which verifiers are perturbed for this (state, lambda)."""
        if lam is Lambda.NONE:
            return {v: False for v in VERIFIERS}
        if lam not in self.levels:
            raise DegradeConfigError(f"no parameters for lambda={lam.value}")
        d = self.draws(state_id)
        return {v: d[v][0] < self.levels[lam].p(v) for v in VERIFIERS}

    def __call__(self, state_id: str, s1: float, s2: float, lam: Lambda) -> tuple[float, float]:
        if lam is Lambda.NONE:
            return float(s1), float(s2)
        if lam not in self.levels:
            raise DegradeConfigError(f"no parameters for lambda={lam.value}")
        d = self.draws(state_id)
        params = self.levels[lam]
        t1 = d["v1"][1] if d["v1"][0] < params.p_v1 else float(s1)
        t2 = d["v2"][1] if d["v2"][0] < params.p_v2 else float(s2)
        return t1, t2

    def assignment(self, state_id: str, lam: Lambda) -> dict[str, Any]:
        """II.10 log fields: whether each verifier was perturbed at this level, the derived z
        mode, and the replacement scores actually used (``None`` where nothing was replaced).
        ``u_v*`` are the per-(state, verifier) uniforms that make the levels nested."""
        d = self.draws(state_id)
        per = self.perturbed(state_id, lam)
        return {
            "perturbed_v1": per["v1"],
            "perturbed_v2": per["v2"],
            "z": z_mode(per["v1"], per["v2"]),
            "replacement_s1": d["v1"][1] if per["v1"] else None,
            "replacement_s2": d["v2"][1] if per["v2"] else None,
            "u_v1": d["v1"][0],
            "u_v2": d["v2"][0],
        }

    def resample_all(self, state_id: str) -> tuple[float, float]:
        """The pair the state would show if BOTH verifiers were perturbed (p = 1). step08's
        reachability floor r_i is the same limit taken in expectation over q_i (II.5.2, v1.0)."""
        d = self.draws(state_id)
        return d["v1"][1], d["v2"][1]

    # ---- persistence (frozen/channel.json) -------------------------------------------
    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "geometry": "independent_resampling",
            "assignment": "per_verifier_nested",
            "seed": self.seed,
            "tau1": self.tau1,
            "tau2": self.tau2,
            "levels": {lam.value: p.to_dict() for lam, p in self.levels.items()},
            "derived_p_z": {lam.value: p.derived_mode_probs() for lam, p in self.levels.items()},
            "q": {k: m.to_dict() for k, m in self.q.items()},
        }

    def save(self, path: Path, extra: dict[str, Any] | None = None) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with crlf_rewrite_notice(path), path.open("w", encoding="utf-8", newline="\n") as fh:
            json.dump({**self.to_dict(), **(extra or {})}, fh, ensure_ascii=False, indent=1)

    @classmethod
    def load(cls, path: Path) -> DegradeChannel:
        with Path(path).open("r", encoding="utf-8") as fh:
            d = json.load(fh)
        if d.get("geometry") != "independent_resampling":
            raise DegradeConfigError(
                f"{path}: channel file predates the group 6a decision (geometry="
                f"{d.get('geometry')!r}); re-run step08"
            )
        levels = {Lambda(k): LevelParams.from_config(v) for k, v in d["levels"].items()}
        q = {k: MarginModel.from_dict(v) for k, v in d["q"].items()}
        return cls(levels, q, d["tau1"], d["tau2"], d["seed"])


FIXED_P: dict[Lambda, float] = {Lambda.LOW: 0.0, Lambda.MID: 0.5, Lambda.HIGH: 1.0}
"""p_lambda^(i) per level, the same for both verifiers (protocol II.5.1, C.27; 2026-09-26).

Fixed ex ante: not solved from delta_R, TPR_i or r_i and not hand-filled. lambda_low leaves the
evidence unperturbed; lambda_high replaces every score by a draw that carries no information
about the state (the floor r_i of II.5.2); lambda_mid is the midpoint. Under independent
resampling the expected detection rates are TPR_i, (TPR_i + r_i)/2 and r_i
(:func:`expected_ladder`), so adjacent levels are (TPR_i - r_i)/2 apart in expectation."""


def levels_from_config(degrade_cfg: Mapping[str, Any]) -> dict[Lambda, LevelParams]:
    """The three levels from ``configs/channel.yaml: degrade.levels``.

    The config must carry exactly :data:`FIXED_P` for both verifiers (II.5.1, C.27); any other
    value -- a level missing, a TBD, the two verifiers apart, a number that is not the fixed one
    -- is refused rather than run. The result depends on the config alone: no measurement
    (TPR_i, r_i, delta_R) enters it."""
    raw = degrade_cfg.get("levels")
    if not isinstance(raw, Mapping):
        raise DegradeConfigError(
            "channel.degrade.levels is missing (II.5.1: p fixed at 0 / 0.5 / 1)"
        )
    levels: dict[Lambda, LevelParams] = {}
    for lam in LADDER:
        if lam.value not in raw:
            raise DegradeConfigError(f"channel.degrade.levels.{lam.value} is missing")
        levels[lam] = LevelParams.from_config(raw[lam.value])
    extra = sorted(set(raw) - {lam.value for lam in LADDER})
    if extra:
        raise DegradeConfigError(f"channel.degrade.levels has unknown levels {extra}")
    check_monotone(levels)
    for lam, lp in levels.items():
        if (lp.p_v1, lp.p_v2) != (FIXED_P[lam], FIXED_P[lam]):
            raise DegradeConfigError(
                f"channel.degrade.levels.{lam.value} must be p = {FIXED_P[lam]} for both "
                f"verifiers (protocol II.5.1, C.27: fixed ex ante), got {lp.to_dict()}"
            )
    return levels


def expected_ladder(
    tpr: Mapping[str, float], floor: Mapping[str, float], levels: Mapping[Lambda, LevelParams]
) -> dict[str, dict[str, float]]:
    """E[R_lambda^(i)] = (1 - p) TPR_i + p r_i under the given levels (II.5.2), for the report.
    With the fixed p this is TPR_i, (TPR_i + r_i)/2, r_i. The levels are an input, never an
    output: nothing here feeds back into p."""
    return {
        v: {
            lam.value: (1 - levels[lam].p(v)) * float(tpr[v]) + levels[lam].p(v) * float(floor[v])
            for lam in LADDER
        }
        for v in VERIFIERS
        if v in tpr and v in floor
    }
