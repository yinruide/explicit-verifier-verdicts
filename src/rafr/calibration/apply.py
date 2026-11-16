"""The frozen calibrator: s_obs -> pi-hat. Reads no label (it is what the runners import).

The frozen file (``frozen/calibration_<domain>.json``, written once by
``pipelines/ext_calibrate.py`` and then only read) holds both verifiers' (a, b), eps, the target
rule and the id list of the fitting data with its sha256. Exp 1 and Exp 2 load it with the
sha256 recorded in ``configs/ext.yaml``, so the same calibrator serves both (2026-10-31).
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from rafr.calibration import platt

FROZEN_KIND = "platt_logit"
VERIFIER_KEYS = ("v1", "v2")


class CalibratorError(RuntimeError):
    pass


def file_sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def ids_sha256(ids: Sequence[str]) -> str:
    """sha256 of the id list as written (one id per line, LF, final LF)."""
    return hashlib.sha256(("".join(f"{i}\n" for i in ids)).encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class PiHatDetail:
    """One (state, lambda)'s pi-hat with what it was computed from (the run's side stream)."""

    s1_obs: float
    s2_obs: float
    g1: float
    g2: float
    pihat1: float
    pihat2: float
    clipped1: bool
    clipped2: bool

    def as_dict(self) -> dict[str, Any]:
        return {
            "s1_obs": self.s1_obs,
            "s2_obs": self.s2_obs,
            "g1": self.g1,
            "g2": self.g2,
            "pihat1": self.pihat1,
            "pihat2": self.pihat2,
            "clipped1": self.clipped1,
            "clipped2": self.clipped2,
        }


class Calibrator:
    def __init__(self, record: dict[str, Any], sha256: str, path: Path | None = None) -> None:
        if record.get("kind") != FROZEN_KIND:
            raise CalibratorError(f"not a {FROZEN_KIND} calibrator: kind {record.get('kind')!r}")
        if record.get("targets") != platt.TARGETS:
            raise CalibratorError(f"unexpected target rule {record.get('targets')!r}")
        self.record = record
        self.sha256 = sha256
        self.path = path
        self.eps = float(record["epsilon"])
        self._ab = {}
        for k in VERIFIER_KEYS:
            v = record["verifiers"][k]
            a, b = float(v["a"]), float(v["b"])
            if not (np.isfinite(a) and np.isfinite(b)) or not v.get("converged"):
                raise CalibratorError(f"{k}: the frozen fit is not a converged, finite fit")
            self._ab[k] = (a, b)

    @classmethod
    def load(cls, path: Path, *, expected_sha256: str | None) -> Calibrator:
        """The frozen calibrator; refused if its sha256 is not the expected one."""
        path = Path(path)
        digest = file_sha256(path)
        if expected_sha256 is not None and digest != expected_sha256:
            raise CalibratorError(
                f"{path} has sha256 {digest}, the configuration expects {expected_sha256}"
            )
        return cls(json.loads(path.read_text(encoding="utf-8")), digest, path)

    def verifier_id(self, key: str) -> str:
        return str(self.record["verifiers"][key]["verifier_id"])

    def check_verifiers(self, v1_id: str, v2_id: str) -> None:
        got = (self.verifier_id("v1"), self.verifier_id("v2"))
        if got != (v1_id, v2_id):
            raise CalibratorError(f"calibrator is for verifiers {got}, the run uses {v1_id, v2_id}")

    def params(self, key: str) -> tuple[float, float]:
        """(a, b) of verifier ``key``."""
        return self._ab[key]

    def g(self, key: str, s: float) -> float:
        a, b = self._ab[key]
        return float(platt.predict(a, b, [s], self.eps)[0])

    def detail(self, s1: float, s2: float, tau1: float, tau2: float) -> PiHatDetail:
        """pi-hat of one observation; y_i = 1[s_i >= tau_i], exactly B's flag (fields.py)."""
        out: dict[str, Any] = {}
        for k, s, tau in (("v1", s1, tau1), ("v2", s2, tau2)):
            a, b = self._ab[k]
            _, clipped = platt.clip_logit([s], self.eps)
            g = float(platt.predict(a, b, [s], self.eps)[0])
            out[k] = (g, float(platt.pihat_from_g([g], [int(s >= tau)])[0]), bool(clipped[0]))
        return PiHatDetail(
            s1_obs=float(s1),
            s2_obs=float(s2),
            g1=out["v1"][0],
            g2=out["v2"][0],
            pihat1=out["v1"][1],
            pihat2=out["v2"][1],
            clipped1=out["v1"][2],
            clipped2=out["v2"][2],
        )

    def provenance(self) -> dict[str, Any]:
        return {
            "file": str(self.path) if self.path else None,
            "sha256": self.sha256,
            "epsilon": self.eps,
            "fit_ids_sha256": self.record.get("fit_data", {}).get("state_ids_sha256"),
            "verifiers": {k: self.verifier_id(k) for k in VERIFIER_KEYS},
        }


def write_frozen(path: Path, record: dict[str, Any]) -> str:
    """Write the frozen calibrator (LF, indented) and return its sha256. Refuses to overwrite:
    a calibrator is fitted once."""
    path = Path(path)
    if path.exists():
        raise CalibratorError(f"{path} exists; the calibrator is frozen and is not refitted")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as fh:
        fh.write(json.dumps(record, ensure_ascii=False, indent=1) + "\n")
    return file_sha256(path)


__all__ = [
    "FROZEN_KIND",
    "VERIFIER_KEYS",
    "Calibrator",
    "CalibratorError",
    "PiHatDetail",
    "file_sha256",
    "ids_sha256",
    "write_frozen",
]
