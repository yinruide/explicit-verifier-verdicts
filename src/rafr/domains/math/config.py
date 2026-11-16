"""paper Sec. 5, App. D execution configuration: scientific choices remain visibly pending.

The controller backbone and decoding are inherited from Core, never configured here.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from rafr.domains.math.models import validate_verifier_pair
from rafr.llm.client import canonical_json, sha256_hex


class MathRunConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    schema_version: Literal["math-execution-v2"] = "math-execution-v2"
    stage: Literal["stub", "development", "smoke", "formal"]
    execution_mode: Literal["offline_stub", "live"]
    source_path: str
    bundle_path: str
    development_source_path: str | None = None
    additional_development_paths: list[str] = Field(default_factory=list)
    seed: int | None = None
    fit_n: int = Field(default=1500, ge=2)
    gate_n: int = Field(default=1500, ge=2)
    reserve_n: int = Field(default=1500, ge=2)
    experiment_n: int = Field(default=300, ge=2)
    smoke_n: int = Field(default=30, ge=1)
    delta_r: float | None = Field(default=None, gt=0, le=1)
    bootstrap_n: int = Field(default=10000, ge=100)
    max_decisions: Literal[4] = 4
    max_initial_attempts: int | None = Field(default=None, ge=1)
    math_verify_version: str = "0.9.0"
    verifiers: list[dict[str, Any]] = Field(default_factory=list)
    reviewed_by: str | None = None
    research_configuration_approved: bool = False
    pricing_reviewed: bool = False
    data_reviewed: bool = False
    live_smoke_run: str | None = None
    model_probe_path: str | None = None

    @model_validator(mode="after")
    def consistency(self):
        if (self.stage == "stub") != (self.execution_mode == "offline_stub"):
            raise ValueError("offline boundaries are restricted to stage=stub")
        if self.stage == "formal" and (
            self.fit_n,
            self.gate_n,
            self.reserve_n,
            self.experiment_n,
            self.smoke_n,
            self.bootstrap_n,
        ) != (1500, 1500, 1500, 300, 30, 10000):
            raise ValueError("formal split sizes and bootstrap count are fixed by Appendix M")
        if self.stage == "smoke" and self.smoke_n != 30:
            raise ValueError("live smoke requires 30 A1-only trajectories (M.10)")
        return self

    @classmethod
    def load(cls, path: Path):
        return cls.model_validate(json.loads(Path(path).read_text(encoding="utf-8")))

    def assert_ready(self, phase: str, *, allow_pending: bool = False) -> None:
        """Check only choices consumed by the requested phase; no implicit freezing."""
        if phase not in {
            "generation",
            "scoring",
            "stability",
            "gate",
            "experiment",
            "smoke",
            "analysis",
        }:
            raise ValueError("unknown mathematics phase")
        if allow_pending:
            if self.stage != "stub":
                raise ValueError("pending scientific choices are permitted only for offline stub")
            return
        if self.seed is None:
            raise ValueError("master seed is pending (M.12)")
        if not self.math_verify_version.strip() or self.math_verify_version == "TBD":
            raise ValueError("Math-Verify version is pending (M.12)")
        if phase == "generation" and self.max_initial_attempts is None:
            raise ValueError("G_fixed invalid-output attempt cap is pending (M.12)")
        specs = None
        if phase in {"scoring", "stability", "gate", "experiment", "smoke"}:
            specs = validate_verifier_pair(self.verifiers)
        if phase in {"gate", "experiment"} and self.delta_r is None:
            raise ValueError("delta_R is pending until the D_fit channel check (M.6)")
        if phase == "stability":
            from rafr.domains.math.generation import PROMPT_DIR
            from rafr.verifiers.perturb import lock_payload

            lock_payload(PROMPT_DIR / "perturb")
        if phase in {"experiment", "smoke"}:
            from rafr.formatter.math_feedback import assert_anonymous_descriptions, load_prompts

            prompts = load_prompts()
            assert_anonymous_descriptions(
                prompts,
                (name for spec in specs for name in (spec.model_id, spec.base_model, spec.vendor)),
            )

    def profile_hash(self) -> str:
        """Identity for compatibility checks; this does not grant scientific approval."""
        from rafr.domains.math.generation import (
            PROMPT_DIR,
            frozen_models,
            read_controller_retry_prompt,
        )
        from rafr.formatter.math_feedback import PROMPT_DIR as FEEDBACK_DIR

        prompts = {
            str(path.relative_to(path.parent.parent)): path.read_text(encoding="utf-8")
            for root in (PROMPT_DIR, FEEDBACK_DIR)
            for path in sorted(root.glob("*.txt"))
        }
        return sha256_hex(
            canonical_json(
                {
                    "controller": frozen_models()["controller"],
                    "controller_retry_prompt": read_controller_retry_prompt(),
                    "verifiers": self.verifiers,
                    "math_verify_version": self.math_verify_version,
                    "prompts": prompts,
                    "max_decisions": self.max_decisions,
                }
            )
        )

    def assert_live_allowed(self, *, confirm_live: bool):
        if self.execution_mode != "live" or self.stage == "stub":
            raise ValueError("live command requires an explicitly live configuration")
        if not confirm_live:
            raise ValueError("live request requires operator --confirm-live")
        if not (
            self.reviewed_by
            and self.research_configuration_approved
            and self.pricing_reviewed
            and self.data_reviewed
        ):
            raise ValueError(
                "the authors must review research scope, data and pricing before live work"
            )
        # smoke is checked immediately before Exp 2, not before initial generation.
