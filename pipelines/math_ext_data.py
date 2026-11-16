"""Read-only math pilot inputs and isolated extension preparation.

This adapter never opens MathPipeline (its constructor writes a snapshot), initializes a
provider, or reads private labels outside Core's purpose gate. Preparation writes candidates,
not an authorization or a scientific freeze. Real pilot inputs must be supplied separately.
"""

from __future__ import annotations

import copy
import json
import math
import re
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from rafr.calibration import fit as calibration_fit
from rafr.calibration import platt
from rafr.calibration.apply import Calibrator, file_sha256, write_frozen
from rafr.channel.degrade import LADDER, DegradeChannel, LevelParams
from rafr.channel.margin_fit import MarginModel
from rafr.contracts.schemas import Lambda, MathLabel, Score, Split, State
from rafr.core import paths
from rafr.core.seeding import component_seed
from rafr.data import labels as label_access
from rafr.data.splits import SEED_COMPONENT, SPLIT_ORDER, load_all_splits
from rafr.domains.math.storage import IntegrityError
from rafr.experiments.ext_cohort import draw_cohort, read_cohort, write_cohort
from rafr.logging.writer import read_jsonl

_REQUIRED = frozenset(
    {
        "states.jsonl",
        "initial_complete.json",
        "splits_complete.json",
        "scores.jsonl",
        "fit.json",
        "snapshot.json",
    }
)
_SHA = re.compile(r"[0-9a-f]{64}\Z")
_REPO = Path(__file__).resolve().parents[1]


def _relative(name: str) -> Path:
    p = Path(name)
    if not name or p.is_absolute() or ".." in p.parts or p.as_posix() != name:
        raise IntegrityError(f"unsafe source-relative path: {name!r}")
    return p


def validate_output(
    out: Path, *, source_root: Path | None = None, allow_existing: bool = False
) -> Path:
    """Validate isolation without creating directories; resumable callers opt in explicitly."""
    raw = Path(out).absolute()
    for ancestor in (raw, *raw.parents):
        if ancestor.is_symlink():
            raise IntegrityError("extension output may not traverse a symlink")
    out = raw.resolve()
    if paths.is_archived(out):
        raise IntegrityError("extension output may not write an archive")
    if source_root is not None:
        source = Path(source_root).resolve()
        if out == source or source in out.parents or out in source.parents:
            raise IntegrityError("extension output overlaps the original math run")
    permitted = (
        (_REPO / "runs").resolve(),
        Path(tempfile.gettempdir()).resolve(),
    )
    if out.is_relative_to(permitted[0]) and (
        out.parent != permitted[0]
        or not out.name.startswith(
            (
                "math_exp1ext_",
                "math_exp2ext_",
                "math_ceiling_",
                "math_ext_calibration_",
                "math_ext_cohort_",
                "math_ext_inputs_",
            )
        )
    ):
        raise IntegrityError("math extension runs must be direct runs/ children with math prefixes")
    if not any(base in out.parents for base in permitted):
        raise IntegrityError("output must be a math extension run or system temporary storage")
    if out.exists() and (not allow_existing or not out.is_dir()):
        raise IntegrityError("extension output must be a new directory; never overwrite a run")
    if out.is_dir() and any(child.is_symlink() for child in out.rglob("*")):
        raise IntegrityError("extension output children may not be symlinks")
    return out


def fresh_output(source: MathSource, out: Path) -> Path:
    """Create one isolated extension run directly under runs/ or system temporary storage."""
    out = validate_output(out, source_root=source.root)
    source.assert_unchanged()
    out.mkdir(parents=True, exist_ok=False)
    return out


class MathSource:
    """SHA256-bound initial states, original splits, scores, thresholds and channel.

    The caller supplies hashes obtained independently of this constructor. Completion-marker
    children must also be explicitly bound; trusting a newly calculated hash is not enough.
    Additional original artifacts can be bound and read with read_bound_json/rows.
    """

    def __init__(
        self,
        root: Path,
        *,
        expected_sha256: Mapping[str, str],
        master_seed: int,
        verifier_ids: tuple[str, str],
    ):
        raw_root = Path(root).absolute()
        if any(p.is_symlink() for p in (raw_root, *raw_root.parents)):
            raise IntegrityError("source root may not traverse a symlink")
        self.root = raw_root.resolve()
        if not self.root.is_dir():
            raise IntegrityError(
                "original math run is unavailable; real-artifact verification blocked"
            )
        if type(master_seed) is not int or len(verifier_ids) != 2 or len(set(verifier_ids)) != 2:
            raise IntegrityError(
                "explicit integer seed and two distinct verifier identities required"
            )
        self.master_seed, self.verifier_ids = master_seed, tuple(verifier_ids)
        self._hashes = dict(expected_sha256)
        if _REQUIRED - self._hashes.keys():
            raise IntegrityError(
                f"missing source SHA256 bindings: {sorted(_REQUIRED - self._hashes.keys())}"
            )
        for name, sha in self._hashes.items():
            _relative(name)
            if not isinstance(sha, str) or not _SHA.fullmatch(sha):
                raise IntegrityError(f"invalid explicit SHA256 binding: {name}")
        self.assert_unchanged()
        initial = self.read_bound_json("initial_complete.json")
        self._split_plan = self.read_bound_json("splits_complete.json")
        for marker in (initial, self.split_plan):
            if not isinstance(marker.get("files"), dict) or not marker["files"]:
                raise IntegrityError("completion marker lacks file bindings")
            for name, sha in marker["files"].items():
                _relative(name)
                if self._hashes.get(name) != sha:
                    raise IntegrityError(f"completion marker child is unbound or disagrees: {name}")
        if not {"states.jsonl", "initial_outcomes.jsonl"} <= initial["files"].keys():
            raise IntegrityError("initial marker lacks its state/outcome input bindings")
        # The label gate's only input directory must be covered by the initial marker.
        label_files = [
            p.relative_to(self.root).as_posix() for p in (self.root / "labels").glob("*.jsonl")
        ]
        if not label_files or any(name not in initial["files"] for name in label_files):
            raise IntegrityError("initial marker must bind all label input files")
        expected_splits = {f"splits/{sp.value}.jsonl" for sp in SPLIT_ORDER}
        if set(self.split_plan["files"]) != expected_splits:
            raise IntegrityError("split completion marker does not bind all five Core splits")
        snapshot = self.read_bound_json("snapshot.json")
        if snapshot.get("domain") != "math":
            raise IntegrityError("source snapshot is not a math run")
        cfg = snapshot.get("configs", {}).get("math", {})
        if cfg.get("seed") != master_seed:
            raise IntegrityError("source snapshot master seed differs")
        # Choices may have been pending when the initial snapshot was created. The
        # phase binding is the selected configuration that actually produced scores.
        specs = snapshot.get("math_phase_bindings", {}).get("scoring", {}).get("verifiers")
        if not isinstance(specs, list):
            raise IntegrityError("source snapshot lacks the frozen scoring verifier binding")
        if tuple(v.get("model_id") for v in specs) != self.verifier_ids:
            raise IntegrityError("source snapshot verifier identities differ")
        self.verifier_versions = tuple(str(v["revision"]) for v in specs)
        self.execution_mode = snapshot.get("execution_mode", cfg.get("execution_mode"))
        if snapshot.get("math_artifact_hashes", {}).get("fit.json") != self._hashes["fit.json"]:
            raise IntegrityError("original fit is not bound by its Core snapshot")
        self._state_rows = [State.model_validate(r) for r in self.read_bound_rows("states.jsonl")]
        if any(s.t != 1 or s.source != "g_fixed" for s in self._state_rows):
            raise IntegrityError("source population must contain G_fixed initial states only")
        self._by_x0 = {s.x0_id: s for s in self._state_rows}
        self._by_id = {s.state_id: s for s in self._state_rows}
        if len(self._by_x0) != len(self._state_rows) or len(self._by_id) != len(self._state_rows):
            raise IntegrityError("duplicate initial state or question identity")
        if not self._state_rows or initial.get("n_valid") != len(self._state_rows):
            raise IntegrityError("initial completion population count differs")
        outcomes = self.read_bound_rows("initial_outcomes.jsonl")
        if len({r["question_id"] for r in outcomes}) != len(outcomes):
            raise IntegrityError("duplicate initial generation outcome")
        valid = {r["question_id"]: r.get("state_id") for r in outcomes if r["status"] == "valid"}
        excluded = sum(r["status"] == "excluded" for r in outcomes)
        if (
            valid != {s.x0_id: s.state_id for s in self._state_rows}
            or len(outcomes) != initial.get("n_questions")
            or excluded != initial.get("n_excluded")
            or len(valid) + excluded != len(outcomes)
        ):
            raise IntegrityError("initial outcomes do not match the completed population")
        self._splits = load_all_splits(self.root / "splits")
        if any(not ids for ids in self.splits.values()):
            raise IntegrityError("all five original splits must be nonempty")
        if {x for ids in self.splits.values() for x in ids} != set(self._by_x0):
            raise IntegrityError("split population differs from initial states")
        if self.split_plan.get("counts") != {sp.value: len(ids) for sp, ids in self.splits.items()}:
            raise IntegrityError("split completion counts differ")
        if self.split_plan.get("seed") != component_seed(master_seed, SEED_COMPONENT):
            raise IntegrityError("source split seed differs")
        self._scores: dict[tuple[str, str], Score] = {}
        for raw in self.read_bound_rows("scores.jsonl"):
            row = Score.model_validate(raw)
            if row.state_id not in self._by_id or row.verifier_id not in self.verifier_ids:
                continue  # Later trajectory scores do not enter initial-state calibration.
            version = self.verifier_versions[self.verifier_ids.index(row.verifier_id)]
            if row.model_version != version:
                raise IntegrityError(
                    "initial score verifier revision differs from original snapshot"
                )
            key = row.state_id, row.verifier_id
            if key in self._scores and self._scores[key].score != row.score:
                raise IntegrityError("conflicting duplicate initial verifier score")
            self._scores[key] = row
        for sp in (Split.D_FIT, Split.D_GATE, Split.D_EXP1, Split.UNLABELED_POOL):
            self.pairs(self.states(sp))  # Require coverage; reserve scores remain unnecessary.
        fit = self.read_bound_json("fit.json")
        if (
            fit.get("q_question_ids") != self.splits[Split.UNLABELED_POOL]
            or fit.get("q_reads_labels") is not False
        ):
            raise IntegrityError("original channel q population is not the post-D_Exp1 pool")
        channel = self.channel()
        if channel.seed != component_seed(master_seed, "channel"):
            raise IntegrityError("original channel seed differs")
        if any(not math.isfinite(t) or not 0 <= t <= 1 for t in self.taus):
            raise IntegrityError("invalid original thresholds")
        if tuple(float(r["tau"]) for r in fit["tau"]) != self.taus:
            raise IntegrityError("original channel thresholds disagree with fitted thresholds")
        pool_pairs = self.pairs(self.states(Split.UNLABELED_POOL))
        for i, key in enumerate(("v1", "v2")):
            if channel.q[key].tau != self.taus[i] or channel.q[key].scores != sorted(
                pair[i] for pair in pool_pairs
            ):
                raise IntegrityError("original q distribution disagrees with bound pool scores")
        if any(
            channel.levels[level] != LevelParams(p, p)
            for level, p in zip(LADDER, (0.0, 0.5, 1.0), strict=True)
        ):
            raise IntegrityError("original channel is not the fixed protocol ladder")
        self.assert_unchanged()

    def _path(self, name: str) -> Path:
        p = self.root / _relative(name)
        if name not in self._hashes:
            raise IntegrityError(f"unbound source artifact: {name}")
        if any(q.is_symlink() for q in (p, *p.parents) if q != self.root.parent):
            raise IntegrityError(f"source artifact traverses a symlink: {name}")
        if not p.is_file():
            raise IntegrityError(f"missing original source artifact: {name}")
        return p

    def assert_unchanged(self) -> dict[str, Any]:
        for name, expected in self._hashes.items():
            if file_sha256(self._path(name)) != expected:
                raise IntegrityError(f"original source artifact changed: {name}")
        return {
            "bound_source_unchanged": True,
            "n_bound_files": len(self._hashes),
            "scope": "explicitly SHA256-bound inputs only",
        }

    @property
    def splits(self):
        return copy.deepcopy(self._splits)

    @property
    def split_plan(self):
        return copy.deepcopy(self._split_plan)

    def read_bound_json(self, name: str) -> dict[str, Any]:
        self.assert_unchanged()
        if _relative(name).parts[0] == "labels":
            raise label_access.LabelAccessError("private labels require the Core purpose gate")
        return json.loads(self._path(name).read_text(encoding="utf-8"))

    def read_bound_rows(self, name: str) -> list[dict[str, Any]]:
        self.assert_unchanged()
        # Raw label streams must be consumed only by labels(), never this generic accessor.
        if _relative(name).parts[0] == "labels":
            raise label_access.LabelAccessError("private labels require the Core purpose gate")
        return list(read_jsonl(self._path(name)))

    def states(self, split: Split) -> list[State]:
        self.assert_unchanged()
        return [self._by_x0[x].model_copy(deep=True) for x in self.splits[Split(split)]]

    def labels(self, split: Split, purpose: label_access.LabelPurpose):
        self.assert_unchanged()
        split = Split(split)
        if split is Split.D_GATE_RESERVE:
            raise label_access.LabelAccessError("extension never reads reserve labels")
        rows = label_access.read_labels(
            split,
            purpose,
            labels_dir=self.root / "labels",
            split_dir=self.root / "splits",
            # Reserve is rejected above, regardless of any shared gate history.
            # Never fall back to detox's global gate-attempts file.
            gate_attempts_path=self.root / "gate_attempts.json",
        )
        expected = {s.state_id: s.x0_id for s in self.states(split)}
        if {r.state_id: r.x0_id for r in rows} != expected or len(rows) != len(expected):
            raise IntegrityError("purpose-gated label identities differ from initial states")
        if any(r.label not in {MathLabel.CORRECT, MathLabel.INCORRECT} for r in rows):
            raise IntegrityError("extension requires resolved MathLabel judgments")
        self.assert_unchanged()
        return rows

    def pairs(self, states: Sequence[State]) -> list[tuple[float, float]]:
        self.assert_unchanged()
        out = []
        for state in states:
            if state.state_id not in self._by_id or state != self._by_id[state.state_id]:
                raise IntegrityError("score request is not an original initial state")
            try:
                out.append(
                    tuple(float(self._scores[state.state_id, v].score) for v in self.verifier_ids)
                )
            except KeyError as exc:
                raise IntegrityError("incomplete original initial verifier scores") from exc
        return out

    def channel(self) -> DegradeChannel:
        data = self.read_bound_json("fit.json")["channel"]
        return DegradeChannel(
            {
                Lambda(k): LevelParams(**{f"p_{v}": p for v, p in val.items()})
                for k, val in data["levels"].items()
            },
            {k: MarginModel.from_dict(val) for k, val in data["q"].items()},
            data["tau1"],
            data["tau2"],
            data["seed"],
        )

    @property
    def taus(self) -> tuple[float, float]:
        channel = self.channel()
        return channel.tau1, channel.tau2

    def provenance(self) -> dict[str, Any]:
        return {
            "source_root": str(self.root),
            "source_sha256": dict(self._hashes),
            "source_execution_mode": self.execution_mode,
            "master_seed": self.master_seed,
            "verifier_ids": list(self.verifier_ids),
            "verifier_versions": list(self.verifier_versions),
        }


SOURCE_IDENTITY_KEYS = ("source_execution_mode", "master_seed", "verifier_ids", "verifier_versions")


def same_bound_source(recorded: Mapping[str, Any] | None, source: MathSource) -> bool:
    """A frozen record (calibrator fit_data, cohort header, candidate receipt) names its source
    by sha256, not by where it sits on disk (2026-11-05): every file the record bound has
    the same sha256 among this source's bindings, and the execution mode, master seed and
    verifier identities agree. The recorded ``source_root`` is informational only."""
    if not isinstance(recorded, Mapping):
        return False
    current = source.provenance()
    hashes = recorded.get("source_sha256")
    if not isinstance(hashes, Mapping) or not hashes:
        return False
    if any(current["source_sha256"].get(name) != sha for name, sha in hashes.items()):
        return False
    return all(recorded.get(key) == current[key] for key in SOURCE_IDENTITY_KEYS)


def _score_map(source: MathSource, split: Split):
    states = source.states(split)
    pairs = source.pairs(states)
    ids = [s.state_id for s in states]
    return ids, {
        key: {sid: pair[i] for sid, pair in zip(ids, pairs, strict=True)}
        for i, key in enumerate(("v1", "v2"))
    }


def _failures(source: MathSource, split: Split, purpose: label_access.LabelPurpose):
    return {r.state_id: int(r.label is MathLabel.INCORRECT) for r in source.labels(split, purpose)}


def _write_json(path: Path, value: dict[str, Any]):
    with path.open("x", encoding="utf-8", newline="\n") as fh:
        fh.write(json.dumps(value, indent=2, allow_nan=False) + "\n")


def prepare_calibration(
    source: MathSource, out: Path, *, n_boot: int = 10000, stage: str = "offline_candidate"
) -> dict[str, Any]:
    """Fit a candidate with Core, including pool prior correction and held-out diagnostics.

    Uses detox's specified eps / clipping / binning settings; no quality gate is invented.
    A real math freeze and scientific approval remain separate from this preparation.
    """
    if stage not in {"offline_candidate", "offline_stub"}:
        raise IntegrityError("only an offline candidate or explicit stub is authorized here")
    if type(n_boot) is not int or n_boot < 1 or (stage != "offline_stub" and n_boot != 10000):
        raise IntegrityError("candidate diagnostics require 10000 bootstrap replicates")
    if stage == "offline_stub" and source.execution_mode != "offline_stub":
        raise IntegrityError("stub settings may not operate on live source artifacts")
    disjoint = calibration_fit.assert_pool_disjoint(source.root / "splits")
    sets = {sp: _score_map(source, sp) for sp in (Split.UNLABELED_POOL, Split.D_GATE, Split.D_EXP1)}
    out = fresh_output(source, out)
    scores_by_split = {
        sp.value: {key: [sc[key][sid] for sid in ids] for key in ("v1", "v2")}
        for sp, (ids, sc) in sets.items()
    }
    try:
        clip = calibration_fit.clip_report(scores_by_split, 1e-7, 0.01)
    except calibration_fit.ClipShareError as exc:
        _write_json(out / "clip_report.json", exc.report)
        source.assert_unchanged()
        raise
    _write_json(out / "clip_report.json", clip)
    pool_ids, pool_scores = sets[Split.UNLABELED_POOL]
    pool_y = _failures(source, Split.UNLABELED_POOL, label_access.LabelPurpose.CALIBRATION_FIT)
    x0_of = {s.state_id: s.x0_id for s in source.states(Split.UNLABELED_POOL)}
    premises = calibration_fit.check_prior_premises(
        all_x0=list(source._by_x0),
        splits=source.splits,
        plan=source.split_plan,
        expected_seed=component_seed(source.master_seed, SEED_COMPONENT),
        pool_failure_x0=[x0_of[sid] for sid, y in pool_y.items() if y],
    )
    correction = {
        **platt.prior_shift(len(pool_ids), sum(pool_y.values()), len(source.splits[Split.D_EXP1])),
        "added_split": Split.D_EXP1.value,
        "premises": premises,
    }
    record = calibration_fit.fit_record(
        domain="math",
        ids=pool_ids,
        failures=pool_y,
        scores=pool_scores,
        verifier_ids=dict(zip(("v1", "v2"), source.verifier_ids, strict=True)),
        eps=1e-7,
        provenance={
            **source.provenance(),
            "stage": stage,
            "scientifically_frozen": False,
            "pool_disjoint": disjoint,
        },
        prior_correction=correction,
    )
    path = out / "calibration_math_candidate.json"
    sha = write_frozen(path, record)  # Core write-once serializer, candidate namespace only.
    calibrator = Calibrator.load(path, expected_sha256=sha)
    gate_ids, gate_scores = sets[Split.D_GATE]
    gate_y = _failures(source, Split.D_GATE, label_access.LabelPurpose.CALIBRATION_DIAG)
    diag = calibration_fit.diagnose(
        calibrator,
        ids=gate_ids,
        failures=gate_y,
        scores=gate_scores,
        taus=dict(zip(("v1", "v2"), source.taus, strict=True)),
        n_bins=10,
        n_boot=n_boot,
        seed=component_seed(source.master_seed, "ext_calibration_boot"),
        alpha=0.05,
    )
    _write_json(out / "diagnostics.json", diag)
    result = {
        "stage": stage,
        "scientifically_frozen": False,
        "live_executed": False,
        "calibrator_file": str(path),
        "calibrator_sha256": sha,
        "diagnostics_sha256": file_sha256(out / "diagnostics.json"),
        "clip_report_sha256": file_sha256(out / "clip_report.json"),
        "n_fit": len(pool_ids),
        "n_diagnostic": len(gate_ids),
        "n_boot": n_boot,
        "prior_correction": correction,
        "source": source.provenance(),
        "source_unchanged": source.assert_unchanged(),
    }
    _write_json(out / "COMPLETE.json", result)
    return result


def prepare_correct_cohort(
    source: MathSource, out: Path, *, n: int = 300, stage: str = "offline_candidate"
) -> dict[str, Any]:
    """Draw a candidate E-CORRECT cohort via the shared Core sampling/file contract."""
    if stage not in {"offline_candidate", "offline_stub"}:
        raise IntegrityError("only offline cohort preparation is authorized")
    if stage != "offline_stub" and n != 300:
        raise IntegrityError("candidate cohort target is 300; never shrink the scientific rule")
    if stage == "offline_stub" and source.execution_mode != "offline_stub":
        raise IntegrityError("stub settings may not operate on live source artifacts")
    rows = source.labels(Split.D_GATE, label_access.LabelPurpose.ECORRECT_COHORT)
    candidates = [r.state_id for r in rows if r.label is MathLabel.CORRECT]
    if not candidates:
        raise IntegrityError("no E-CORRECT D_gate states for the extension cohort")
    seed = component_seed(source.master_seed, "ext_ecorrect_cohort")
    ids = draw_cohort(candidates, n, seed)
    out = fresh_output(source, out)
    path = out / "ecorrect_cohort_candidate.jsonl"
    sha = write_cohort(
        path,
        ids,
        {
            "domain": "math",
            "split": Split.D_GATE.value,
            "stage": stage,
            "target_n": n,
            "n_candidates": len(candidates),
            "seed": seed,
            "source": source.provenance(),
            "scientifically_frozen": False,
        },
    )
    header, checked_ids = read_cohort(path, sha)
    if ids != checked_ids or len(set(ids)) != len(ids):
        raise IntegrityError("cohort read-back differs from the seeded draw")
    result = {
        "stage": stage,
        "scientifically_frozen": False,
        "live_executed": False,
        "cohort_file": str(path),
        "cohort_sha256": sha,
        "n": header["n"],
        "target_n": n,
        "n_candidates": len(candidates),
        "source_unchanged": source.assert_unchanged(),
    }
    _write_json(out / "COMPLETE.json", result)
    return result
