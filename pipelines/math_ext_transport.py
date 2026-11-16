"""Public, hash-bound math pod transport; file preparation is never execution permission.

The local selector alone reads purpose-gated labels. Transferred starts contain exactly five
public fields. Import returns public State objects, arm plans and t1 score evidence; no E
context, provider, model loader, private reference or initial grade is constructed here.
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from pipelines.math_exp2_loop import text_key
from pipelines.math_ext_data import MathSource, validate_output
from pipelines.math_ext_native import native_t1
from rafr.calibration.apply import file_sha256
from rafr.contracts.schemas import Arm, MathLabel, Score, Split, State
from rafr.core import paths
from rafr.data.labels import LabelPurpose
from rafr.domains.math.storage import IntegrityError, digest
from rafr.experiments.ext_cohort import read_cohort
from rafr.logging.writer import JsonlWriter, read_jsonl

_SHA = re.compile(r"[0-9a-f]{64}\Z")
_START_KEYS = {"state_id", "x0_id", "x0", "text", "arms"}
_ARMS = {Arm.A1, Arm.B, Arm.NO_VERIFIER, Arm.VERDICT_ONLY, Arm.B_PIHAT}
_NATIVE_KEYS = {
    "domain",
    "source",
    "source_sha256",
    "original_decisions_sha256",
    "assignment_sha256",
    "verifier_ids",
    "verifier_versions",
    "reuse_decision",
}
_REFERENCE_KEYS = {"domain", "source", "source_sha256", "verifier_ids", "verifier_versions"}
_MANIFEST_KEYS = {
    "format",
    "transport_only",
    "live_executed",
    "cohort",
    "smoke",
    "fixture",
    "source_mode",
    "source_identity_sha256",
    "master_seed",
    "verifier_ids",
    "verifier_versions",
    "n_starts",
    "cohort_sha256",
    "assignment_sha256",
    "t1_mode",
    "files",
}
_FILES = {"starts.jsonl", "t1.json"}


def _sha(value: Any, name: str) -> str:
    if not isinstance(value, str) or not _SHA.fullmatch(value):
        raise IntegrityError(f"{name} requires an explicit SHA256")
    return value


def _exact(value: Any, keys: set[str], name: str):
    if not isinstance(value, dict) or set(value) != keys:
        raise IntegrityError(f"{name} has unexpected or missing fields")


def _input_file(path: Path) -> Path:
    path = Path(path).absolute()
    if any(p.is_symlink() for p in (path, *path.parents)) or paths.is_archived(path):
        raise IntegrityError("transport input may not traverse symlinks or an archive")
    if not path.is_file():
        raise IntegrityError(f"missing transport input: {path.name}")
    return path


def _read_cohort(path: Path | None, sha: str | None, *, fixture: bool) -> list[str]:
    if path is None:
        raise IntegrityError("E-correct selection requires the frozen cohort file")
    header, ids = read_cohort(_input_file(path), _sha(sha, "cohort"))
    if not fixture and header.get("scientifically_frozen") is False:
        raise IntegrityError("a candidate cohort is not a frozen formal cohort")
    if not ids or len(ids) != len(set(ids)) or len(ids) > 300:
        raise IntegrityError("cohort must contain 1..300 unique state IDs")
    return ids


def _select(source, *, cohort, smoke, fixture, state_ids, cohort_file, cohort_sha256):
    if type(smoke) is not bool or type(fixture) is not bool:
        raise IntegrityError("smoke and fixture require explicit booleans")
    if cohort not in {"e_incorrect", "e_correct"}:
        raise IntegrityError("unknown cohort")
    expected_mode = "offline_stub" if fixture else "live"
    snapshot = source.read_bound_json("snapshot.json")
    configured = snapshot.get("configs", {}).get("math", {})
    actual_mode = snapshot.get("execution_mode", configured.get("execution_mode"))
    if source.master_seed != configured.get("seed"):
        raise IntegrityError("source seed differs from bound snapshot")
    if source.execution_mode != expected_mode or source.execution_mode != actual_mode:
        raise IntegrityError("fixture/formal source mode differs; never relabel a fixture as live")
    assignment = None
    if cohort == "e_incorrect":
        split = Split.UNLABELED_POOL if smoke else Split.D_EXP1
        purpose = LabelPurpose.DRY_RUN if smoke else LabelPurpose.EXP2_T1
    else:
        split = Split.D_GATE
        purpose = LabelPurpose.EXT_SMOKE if smoke else LabelPurpose.ECORRECT_COHORT
    available = source.states(split)
    by_id = {s.state_id: s for s in available}
    if smoke:
        if state_ids is None:
            raise IntegrityError("smoke requires explicit selected state IDs")
        ids = list(state_ids)
        if cohort == "e_correct" and set(ids) & set(
            _read_cohort(cohort_file, cohort_sha256, fixture=fixture)
        ):
            raise IntegrityError("E-correct smoke overlaps the frozen formal cohort")
    elif cohort == "e_correct":
        ids = _read_cohort(cohort_file, cohort_sha256, fixture=fixture)
        if state_ids is not None and list(state_ids) != ids:
            raise IntegrityError("selected correct starts differ from frozen cohort")
    else:
        ids = [s.state_id for s in available]
        if state_ids is not None and list(state_ids) != ids:
            raise IntegrityError("incorrect starts must reuse the complete original D_Exp1")
        if not fixture and len(ids) != 300:
            raise IntegrityError("formal incorrect transport requires all 300 original starts")
        assignment = source.read_bound_json("exp2/assignment.json")
        if set(assignment) != {s.x0_id for s in available} or any(
            a not in {Arm.A1.value, Arm.B.value} for a in assignment.values()
        ):
            raise IntegrityError("original assignment must cover all starts with A1/B")
    if not ids or len(ids) != len(set(ids)) or set(ids) - by_id.keys():
        raise IntegrityError("starts must be a unique nonempty subset of the required source split")
    labels = {r.state_id: r.label for r in source.labels(split, purpose)}
    wanted = MathLabel.INCORRECT if cohort == "e_incorrect" else MathLabel.CORRECT
    if any(labels[sid] is not wanted for sid in ids):
        raise IntegrityError("selected source y* disagrees with cohort")
    return [by_id[sid] for sid in ids], assignment


def _plan(plan: Mapping[str, Sequence[Arm]], states: Sequence[State], assignment=None):
    if set(plan) != {s.x0_id for s in states}:
        raise IntegrityError("plan does not cover exactly the selected starts")
    result = {}
    for qid, choices in plan.items():
        if isinstance(choices, (str, bytes)):
            raise IntegrityError("arms must be a list, not a string")
        try:
            arms = [Arm(a) for a in choices]
        except (ValueError, TypeError) as exc:
            raise IntegrityError("unknown transport arm") from exc
        if not arms or len(arms) != len(set(arms)) or set(arms) - _ARMS:
            raise IntegrityError("empty, repeated or unsupported pod arms")
        if assignment is not None and Arm(assignment[qid]) in arms:
            raise IntegrityError("incorrect fill must not rerun its original assigned A1/B arm")
        result[qid] = arms
    new_sets = {frozenset(a for a in arms if a not in {Arm.A1, Arm.B}) for arms in result.values()}
    if len(new_sets) != 1:
        raise IntegrityError("each segment must cover its new arms on every selected start")
    if assignment is not None:
        has_fill = any(set(arms) & {Arm.A1, Arm.B} for arms in result.values())
        if has_fill and any(
            set(arms) & {Arm.A1, Arm.B} != {Arm.B if assignment[qid] == "A1" else Arm.A1}
            for qid, arms in result.items()
        ):
            raise IntegrityError("fill segment must cover the missing original arm on every start")
    return result


def _reference(source, states):
    # Preserve the actual original Score record, including its timestamp. Only its join
    # key is translated from the initial state ID to the pod question/text identity.
    by_key = {}
    for raw in source.read_bound_rows("scores.jsonl"):
        row = Score.model_validate(raw)
        by_key[row.state_id, row.verifier_id] = row
    return {
        "scoring_identity": {
            "domain": "math",
            "source": "scores.jsonl",
            "source_sha256": source.provenance()["source_sha256"]["scores.jsonl"],
            "verifier_ids": list(source.verifier_ids),
            "verifier_versions": list(source.verifier_versions),
        },
        "states": [
            {
                "x0_id": s.x0_id,
                "problem": s.x0,
                "text": s.text,
                "scores": [
                    by_key[s.state_id, v]
                    .model_copy(update={"state_id": text_key(s.x0_id, s.text)})
                    .model_dump(mode="json")
                    for v in source.verifier_ids
                ],
            }
            for s in states
        ],
    }


def _t1(payload, states, *, mode, verifier_ids, verifier_versions):
    _exact(payload, {"scoring_identity", "states"}, "t1 payload")
    identity = payload["scoring_identity"]
    reuse = mode == "reuse_original_round1"
    if mode not in {"reuse_original_round1", "reference_only"}:
        raise IntegrityError("unknown t1 mode")
    _exact(identity, _NATIVE_KEYS if reuse else _REFERENCE_KEYS, "t1 scoring identity")
    if (
        identity["domain"] != "math"
        or identity["verifier_ids"] != list(verifier_ids)
        or identity["verifier_versions"] != list(verifier_versions)
    ):
        raise IntegrityError("t1 verifier identity or revision differs")
    expected_source = "exp2/rounds/round_1/scores.jsonl" if reuse else "scores.jsonl"
    if identity["source"] != expected_source:
        raise IntegrityError("t1 source is not the required original artifact")
    _sha(identity["source_sha256"], "t1 source")
    if reuse:
        for key in ("original_decisions_sha256", "assignment_sha256"):
            _sha(identity[key], key)
        if identity["reuse_decision"] != "2026-11-01 handoff 3.5/6.1":
            raise IntegrityError("unexpected t1 reuse declaration")
    if not isinstance(payload["states"], list):
        raise IntegrityError("t1 states must be an array")
    by_q = {s.x0_id: s for s in states}
    seen = set()
    for row in payload["states"]:
        _exact(row, {"x0_id", "problem", "text", "scores"}, "t1 state")
        qid = row["x0_id"]
        if qid not in by_q or qid in seen:
            raise IntegrityError("unexpected or duplicate t1 question")
        seen.add(qid)
        state = by_q[qid]
        if (row["problem"], row["text"]) != (state.x0, state.text):
            raise IntegrityError("t1 public question/text does not match starts")
        if not isinstance(row["scores"], list) or len(row["scores"]) != 2:
            raise IntegrityError("t1 needs exactly two scores per start")
        for i, raw in enumerate(row["scores"]):
            _exact(raw, set(Score.model_fields), "t1 score")
            if type(raw["score"]) not in (int, float) or not isinstance(raw["timestamp"], str):
                raise IntegrityError("t1 score has invalid scalar types")
            try:
                datetime.fromisoformat(raw["timestamp"])
            except ValueError as exc:
                raise IntegrityError(
                    "t1 score timestamp must be the original ISO timestamp"
                ) from exc
            score = Score.model_validate(raw)
            if (
                score.state_id != text_key(qid, state.text)
                or score.verifier_id != verifier_ids[i]
                or score.model_version != verifier_versions[i]
            ):
                raise IntegrityError("t1 score question/verifier/revision binding differs")
            if score.sub_scores and any(not k.isdecimal() for k in score.sub_scores):
                raise IntegrityError("t1 sub-scores permit numeric step keys only")
    if seen != set(by_q):
        raise IntegrityError("t1 evidence must cover every selected start exactly")


def _write(path, value, *, rows=False):
    if rows:
        with JsonlWriter(path) as writer:
            writer.write_many(value)
    else:
        with Path(path).open("x", encoding="utf-8", newline="\n") as fh:
            fh.write(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    sha = file_sha256(path)
    with Path(str(path) + ".sha256").open("x", encoding="utf-8", newline="\n") as fh:
        fh.write(sha + "\n")
    return sha


def export_public_bundle(
    source: MathSource,
    out: Path,
    *,
    cohort: str,
    plan: Mapping[str, Sequence[Arm]],
    smoke: bool,
    fixture: bool,
    state_ids: Sequence[str] | None = None,
    cohort_file: Path | None = None,
    cohort_sha256: str | None = None,
    native_payload: dict | None = None,
    reference_payload: dict | None = None,
) -> dict[str, Any]:
    """Prepare immutable public files locally. No pod launch, scoring or API call occurs."""
    source.assert_unchanged()
    states, assignment = _select(
        source,
        cohort=cohort,
        smoke=smoke,
        fixture=fixture,
        state_ids=state_ids,
        cohort_file=cohort_file,
        cohort_sha256=cohort_sha256,
    )
    selected_plan = _plan(plan, states, assignment)
    reuse = cohort == "e_incorrect" and not smoke
    if reuse and reference_payload is not None or not reuse and native_payload is not None:
        raise IntegrityError(
            "smoke/correct t1 is reference-only; incorrect formal reuses round one"
        )
    wanted = native_t1(source, states) if reuse else _reference(source, states)
    supplied = native_payload if reuse else reference_payload
    if supplied is not None and supplied != wanted:
        raise IntegrityError("supplied t1 payload differs from its bound native source")
    payload = wanted if supplied is None else supplied
    mode = "reuse_original_round1" if reuse else "reference_only"
    _t1(
        payload,
        states,
        mode=mode,
        verifier_ids=source.verifier_ids,
        verifier_versions=source.verifier_versions,
    )
    out = validate_output(out, source_root=source.root)
    source.assert_unchanged()
    out.mkdir(parents=True, exist_ok=False)
    starts = [
        {
            "state_id": s.state_id,
            "x0_id": s.x0_id,
            "x0": s.x0,
            "text": s.text,
            "arms": [a.value for a in selected_plan[s.x0_id]],
        }
        for s in states
    ]
    files = {
        "starts.jsonl": _write(out / "starts.jsonl", starts, rows=True),
        "t1.json": _write(out / "t1.json", payload),
    }
    manifest = {
        "format": "math-public-transport-v1",
        "transport_only": True,
        "live_executed": False,
        "cohort": cohort,
        "smoke": smoke,
        "fixture": fixture,
        "source_mode": source.execution_mode,
        "source_identity_sha256": digest(source.provenance()),
        "master_seed": source.master_seed,
        "verifier_ids": list(source.verifier_ids),
        "verifier_versions": list(source.verifier_versions),
        "n_starts": len(states),
        "cohort_sha256": cohort_sha256 if cohort == "e_correct" else None,
        "assignment_sha256": source.provenance()["source_sha256"]["exp2/assignment.json"]
        if reuse
        else None,
        "t1_mode": mode,
        "files": files,
    }
    source.assert_unchanged()
    manifest_sha = _write(out / "transport.json", manifest)
    return {"manifest": manifest, "manifest_sha256": manifest_sha, "directory": str(out)}


@dataclass(frozen=True)
class PublicBundle:
    states: tuple[State, ...]
    plan: Mapping[str, tuple[Arm, ...]]
    t1_payload: dict[str, Any]
    manifest: dict[str, Any]
    t1_path: Path

    @property
    def reuse_t1(self) -> bool:
        return self.manifest["t1_mode"] == "reuse_original_round1"


def _checked(path, expected):
    path = _input_file(path)
    _sha(expected, path.name)
    sidecar = _input_file(Path(str(path) + ".sha256")).read_text(encoding="utf-8")
    if sidecar != expected + "\n" or file_sha256(path) != expected:
        raise IntegrityError(f"transport file or sidecar changed: {path.name}")
    return path


def import_public_bundle(
    bundle_dir: Path,
    *,
    expected_manifest_sha256: str,
    expected_source_identity_sha256: str,
    expected_verifier_ids: tuple[str, str],
    expected_verifier_versions: tuple[str, str],
    expected_source_mode: str,
) -> PublicBundle:
    """Read files against independent expected identities, never sidecars alone."""
    root = Path(bundle_dir)
    path = _checked(root / "transport.json", expected_manifest_sha256)
    manifest = json.loads(path.read_text(encoding="utf-8"))
    _exact(manifest, _MANIFEST_KEYS, "transport manifest")
    if (
        manifest["format"] != "math-public-transport-v1"
        or manifest["transport_only"] is not True
        or manifest["live_executed"] is not False
    ):
        raise IntegrityError("not a public transport manifest")
    _sha(expected_source_identity_sha256, "expected source identity")
    if manifest["source_identity_sha256"] != expected_source_identity_sha256:
        raise IntegrityError("transport is bound to another original source")
    if (
        expected_source_mode not in {"offline_stub", "live"}
        or manifest["source_mode"] != expected_source_mode
    ):
        raise IntegrityError("transport source mode differs")
    if (
        type(manifest["fixture"]) is not bool
        or type(manifest["smoke"]) is not bool
        or manifest["fixture"] != (expected_source_mode == "offline_stub")
    ):
        raise IntegrityError("transport fixture/source mode mismatch")
    if manifest["verifier_ids"] != list(expected_verifier_ids) or manifest[
        "verifier_versions"
    ] != list(expected_verifier_versions):
        raise IntegrityError("transport verifier identities/revisions differ")
    if type(manifest["master_seed"]) is not int:
        raise IntegrityError("transport master seed must be an integer")
    if (
        len(expected_verifier_ids) != 2
        or len(set(expected_verifier_ids)) != 2
        or len(expected_verifier_versions) != 2
        or any(
            not isinstance(v, str) or not v
            for v in (*expected_verifier_ids, *expected_verifier_versions)
        )
    ):
        raise IntegrityError("transport requires two explicit verifier IDs and revisions")
    _exact(manifest["files"], _FILES, "transport file table")
    for name, sha in manifest["files"].items():
        _checked(root / name, sha)
    rows = list(read_jsonl(root / "starts.jsonl"))
    states, plan = [], {}
    for row in rows:
        _exact(row, _START_KEYS, "public start")
        if any(not isinstance(row[k], str) or not row[k] for k in _START_KEYS - {"arms"}):
            raise IntegrityError("public start strings must be nonempty")
        if row["x0_id"] in plan:
            raise IntegrityError("duplicate public question")
        states.append(State.model_validate({k: row[k] for k in _START_KEYS - {"arms"}}))
        if not isinstance(row["arms"], list):
            raise IntegrityError("public arms must be an array")
        plan[row["x0_id"]] = row["arms"]
    if not states or len({s.state_id for s in states}) != len(states):
        raise IntegrityError("duplicate or empty public states")
    if type(manifest["n_starts"]) is not int or manifest["n_starts"] != len(states):
        raise IntegrityError("manifest start count differs")
    parsed_plan = _plan(plan, states)
    cohort, smoke = manifest["cohort"], manifest["smoke"]
    if cohort not in {"e_incorrect", "e_correct"}:
        raise IntegrityError("unknown transport cohort")
    reuse = cohort == "e_incorrect" and not smoke
    if manifest["t1_mode"] != ("reuse_original_round1" if reuse else "reference_only"):
        raise IntegrityError("smoke/correct transport cannot reuse t1 scores")
    if reuse:
        _sha(manifest["assignment_sha256"], "original assignment")
        if not manifest["fixture"] and len(states) != 300:
            raise IntegrityError("formal incorrect transport must carry 300 starts")
    elif manifest["assignment_sha256"] is not None:
        raise IntegrityError("unexpected assignment for reference-only transport")
    if cohort == "e_correct":
        _sha(manifest["cohort_sha256"], "correct cohort")
    elif manifest["cohort_sha256"] is not None:
        raise IntegrityError("unexpected frozen correct cohort binding")
    payload = json.loads((root / "t1.json").read_text(encoding="utf-8"))
    _t1(
        payload,
        states,
        mode=manifest["t1_mode"],
        verifier_ids=expected_verifier_ids,
        verifier_versions=expected_verifier_versions,
    )
    if reuse and payload["scoring_identity"]["assignment_sha256"] != manifest["assignment_sha256"]:
        raise IntegrityError("t1 source assignment differs from manifest")
    return PublicBundle(
        tuple(states),
        {k: tuple(v) for k, v in parsed_plan.items()},
        payload,
        manifest,
        root / "t1.json",
    )
