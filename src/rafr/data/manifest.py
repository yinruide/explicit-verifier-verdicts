"""``data/manifest.json``: what the data on disk is and where it came from (II.10).

Each pipeline step updates its own section; the file is the single place to check which
raw files, which prefilter run and which split seed produced the pools in use.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from rafr.contracts.schemas import utc_now_iso
from rafr.core import paths
from rafr.core.textio import crlf_rewrite_notice


class SplitPlanChangedError(RuntimeError):
    """The configured split plan is not the one pinned in the manifest (II.2.4)."""


class Manifest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    created_at: str = Field(default_factory=utc_now_iso)
    updated_at: str = Field(default_factory=utc_now_iso)
    # "formal" | "fake": what kind of run last wrote this file (2026-09-20).
    # A fake run writes its own manifest under data/_fake/, so a formal manifest should only
    # ever say "formal"; the field is what the split guard reads before overwriting anything,
    # and a missing value is read as "formal" -- unknown provenance is treated as precious.
    run_mode: str | None = None
    last_run: dict[str, Any] = Field(default_factory=dict)  # step, mode, ts, run_dir
    source: dict[str, Any] = Field(default_factory=dict)  # repo, revision, files + sha256
    cleaning: dict[str, Any] = Field(default_factory=dict)  # stats
    # how the candidate pool was drawn from the cleaned pool: seed, method, sizes
    # (2026-09-20 — the sample is random, not a head truncation)
    sampling: dict[str, Any] = Field(default_factory=dict)
    prefilter: dict[str, Any] = Field(default_factory=dict)  # E identity, counts, file sha256
    # v0.9: `plan` (seed + procedure, pinned before step 5 judges anything), then the split
    # itself: sizes and file sha256 per split
    splits: dict[str, Any] = Field(default_factory=dict)
    states: dict[str, Any] = Field(default_factory=dict)  # n, dropped, G_fixed config hash
    labels: dict[str, Any] = Field(default_factory=dict)  # y*: E identity, file sha256, drops
    scores: dict[str, Any] = Field(default_factory=dict)  # per verifier: file, sha256, n
    notes: list[str] = Field(default_factory=list)

    @classmethod
    def load(cls, path: Path | None = None) -> Manifest:
        p = Path(path) if path else paths.MANIFEST
        if not p.exists():
            return cls()
        with p.open("r", encoding="utf-8") as fh:
            return cls.model_validate(json.load(fh))

    def save(self, path: Path | None = None) -> Path:
        p = Path(path) if path else paths.MANIFEST
        p.parent.mkdir(parents=True, exist_ok=True)
        self.updated_at = utc_now_iso()
        # LF on every platform; a manifest written before 2026-09-27 is CRLF, and rewriting it
        # changes its raw-byte hash -- reported, never silent (D-2026-09-27)
        with crlf_rewrite_notice(p), p.open("w", encoding="utf-8", newline="\n") as fh:
            json.dump(self.model_dump(mode="json"), fh, ensure_ascii=False, indent=2)
        return p

    def note_run(self, *, step: str, fake: bool, run_dir: Path | str) -> None:
        """Record which kind of run is writing (II.10 provenance; 2026-09-20)."""
        self.run_mode = "fake" if fake else "formal"
        self.last_run = {
            "step": step,
            "mode": self.run_mode,
            "ts": utc_now_iso(),
            "run_dir": str(run_dir),
        }

    def pin_split_plan(self, plan: dict[str, Any]) -> None:
        """Record the split plan (seed + procedure) the first time; refuse a different one.

        II.2.4 (v0.9): the split seed is fixed before step 5 judges any state, and neither it
        nor the procedure may change afterwards. The plan is pinned by the first v0.9 step
        that writes this manifest; every later step re-derives it from the config and calls
        this, so an edited seed component, procedure or size stops the step.
        """
        pinned = self.splits.get("plan")
        if pinned is None:
            self.splits["plan"] = dict(plan)
            return
        if pinned != plan:
            diff = sorted(k for k in set(pinned) | set(plan) if pinned.get(k) != plan.get(k))
            raise SplitPlanChangedError(
                f"the split plan pinned in {paths.MANIFEST} differs from the configured one "
                f"in {diff}. II.2.4: seed and procedure are fixed before step 5 and never "
                "changed; restore configs/data.yaml (splits.*, seeds.master)."
            )

    def update_section(self, name: str, values: dict[str, Any]) -> None:
        cur = getattr(self, name)
        if not isinstance(cur, dict):
            raise KeyError(name)
        cur.update(values)
