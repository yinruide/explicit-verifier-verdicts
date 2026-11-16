"""The pod's numeric check before any formal score (2026-09-24).

After loading a verifier on the pod, step 6 (the RunPod driver (not released)) and Exp 2's loop
(the RunPod driver (not released)) re-score 203 D_fit states drawn with a fixed seed and compare
every score with the step 2 probe's, bit for bit. One difference stops the job before a single
formal score is written; the error names the largest absolute difference and how many of the 203
differ.

    laptop:  build_reference(config_ids, ...) -> data/step06/numeric_check.jsonl (+ .sha256)
    pod:     check_verifier(verifier, reference_rows_of_that_config) -> raises on a difference

The reference holds, per configuration, the drawn states' state_id and text (D_fit is
verifier-selection data: the probe already scored it on a pod; no label is read or written) and
the probe's score, taken from the probe table that the configuration's **cuda** record lists
(its meta ``table_sha256``): formal scores come from the GPU only, so the reference does too.
The draw is ``N_CHECK`` states of D_fit, seed component ``pod_numeric_check`` of the master seed;
203 is not a multiple of any probe batch size, so the check also runs a short, filled-up batch.
"""

from __future__ import annotations

import hashlib
import json
import struct
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

from rafr.contracts.schemas import State
from rafr.core.seeding import py_rng
from rafr.logging.writer import read_jsonl

N_CHECK = 203
SEED_COMPONENT = "pod_numeric_check"
FIELDS = ("config_id", "state_id", "text", "probe_score")


class NumericCheckError(RuntimeError):
    """The pod does not reproduce the probe's scores bit for bit: nothing is scored.
    ``result`` holds the comparison (count, largest difference, examples) when there is one."""

    def __init__(self, message: str, result: Mapping[str, Any] | None = None) -> None:
        super().__init__(message)
        self.result = dict(result or {})


def draw_states(d_fit: Sequence[State], seed: int, n: int | None = None) -> list[State]:
    """``n`` (default :data:`N_CHECK`) D_fit states by a seeded shuffle of the state_id order
    (no label involved)."""
    n = N_CHECK if n is None else int(n)
    order = sorted(d_fit, key=lambda s: s.state_id)
    if len(order) < n:
        raise ValueError(f"D_fit has {len(order)} states, the check needs {n}")
    py_rng(seed, SEED_COMPONENT).shuffle(order)
    return order[:n]


def sha256_file(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def probe_scores(table: Path, want_sha256: str | None) -> dict[str, float]:
    """``state_id -> score`` of a probe table, which must be the one the record lists."""
    if want_sha256 is not None and sha256_file(table) != want_sha256:
        raise NumericCheckError(
            f"{table} is not the table the probe record lists (sha256 {want_sha256}); a table of "
            "another run (e.g. the laptop CPU's) is no reference for the pod"
        )
    return {str(r["state_id"]): float(r["score"]) for r in read_jsonl(table)}


def build_reference(
    config_ids: Sequence[str],
    states: Sequence[State],
    table_for: Callable[[str], tuple[Path, str | None]],
) -> list[dict[str, Any]]:
    """One row per (configuration, drawn state): state_id, text and the probe's score.
    ``table_for(config_id)`` gives the probe table and the sha256 its cuda record lists."""
    rows: list[dict[str, Any]] = []
    for cid in config_ids:
        table, sha = table_for(cid)
        scores = probe_scores(table, sha)
        missing = [s.state_id for s in states if s.state_id not in scores]
        if missing:
            raise NumericCheckError(f"{table}: no probe score for {len(missing)} drawn state(s)")
        rows += [
            {"config_id": cid, "state_id": s.state_id, "text": s.text,
             "probe_score": scores[s.state_id]}
            for s in states
        ]  # fmt: skip
    return rows


def write_reference(path: Path, rows: Sequence[Mapping[str, Any]]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as fh:
        for r in rows:
            fh.write(json.dumps({k: r[k] for k in FIELDS}, ensure_ascii=False) + "\n")
    digest = sha256_file(path)
    path.with_name(path.name + ".sha256").write_text(
        f"{digest}  {path.name}\n", "utf-8", newline="\n"
    )
    return digest


def read_reference(path: Path, want_sha256: str) -> dict[str, list[dict[str, Any]]]:
    """``config_id -> rows`` of a reference file whose sha256 must be ``want_sha256``."""
    if sha256_file(path) != want_sha256.strip().lower():
        raise NumericCheckError(f"{path}: sha256 does not match {want_sha256}")
    out: dict[str, list[dict[str, Any]]] = {}
    for i, r in enumerate(read_jsonl(path), start=1):
        if tuple(sorted(r)) != tuple(sorted(FIELDS)):
            raise NumericCheckError(f"{path}:{i}: fields {sorted(r)}, expected {sorted(FIELDS)}")
        out.setdefault(str(r["config_id"]), []).append(dict(r))
    return out


def _bits(x: float) -> bytes:
    return struct.pack("<d", float(x))


def compare(now: Sequence[float], rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Bit-for-bit comparison (the IEEE-754 double, so -0.0 != 0.0 and NaN == NaN only if the
    bits are the same) of this run's scores with the probe's."""
    if len(now) != len(rows):
        raise NumericCheckError(f"{len(now)} scores for {len(rows)} reference rows")
    diffs = [
        (r["state_id"], float(r["probe_score"]), float(x))
        for x, r in zip(now, rows, strict=True)
        if _bits(x) != _bits(r["probe_score"])
    ]
    max_abs = max((abs(a - b) for _, a, b in diffs), default=0.0)
    return {
        "n": len(rows),
        "n_different": len(diffs),
        "max_abs_diff": max_abs,
        "identical": not diffs,
        "examples": [{"state_id": s, "probe": a, "now": b} for s, a, b in diffs[:5]],
    }


def check_verifier(
    verifier: Any, rows: Sequence[Mapping[str, Any]], echo: Callable[[str], None] | None = None
) -> dict[str, Any]:
    """Re-score the reference texts with ``verifier`` (in the order drawn, through its own
    ``score``, chunks of its ``batch_limit``) and compare; raise on any difference."""
    say = echo or (lambda _s: None)
    cid = getattr(verifier, "config_id", None) or verifier.verifier_id
    if not rows:
        raise NumericCheckError(f"{cid}: no reference rows (build them on the laptop first)")
    if any(r["config_id"] != cid for r in rows):
        raise NumericCheckError(f"{cid}: reference rows of another configuration")
    texts = [str(r["text"]) for r in rows]
    limit = int(getattr(verifier, "batch_limit", len(texts)) or len(texts))
    now: list[float] = []
    for i in range(0, len(texts), limit):
        now += [float(x) for x in verifier.score(texts[i : i + limit])]
    result = {"config_id": cid, **compare(now, rows)}
    if not result["identical"]:
        raise NumericCheckError(
            f"{cid}: {result['n_different']} of {result['n']} D_fit scores differ from the "
            f"probe's (max |diff| {result['max_abs_diff']:.3e}); stopping before any formal "
            f"score. Examples: {result['examples']}",
            result,
        )
    say(f"[numeric check] {cid}: {result['n']} D_fit scores identical to the probe's, bit for bit")
    return result
