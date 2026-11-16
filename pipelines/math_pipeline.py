"""Appendix M orchestration using Core decisions, streams, labels and channels.

Only the E boundary reads private references. The controller and verifiers receive
public problem/solution values. Engineering fixtures inject external boundaries;
they never become formal results.
"""

from __future__ import annotations

import math
import sys
from dataclasses import asdict
from pathlib import Path

from pipelines.math_exp2_loop import OVERLENGTH, input_overlength, run_rounds, score_record
from rafr.channel import ObservationBuilder
from rafr.channel.degrade import LADDER, DegradeChannel, LevelParams
from rafr.channel.margin_fit import fit_margin_model
from rafr.contracts.schemas import Arm, EvalRecord, Lambda, MathLabel, Score, Split, State
from rafr.controller import decide as controller
from rafr.core.seeding import component_seed, derive_seed
from rafr.data import labels as label_access
from rafr.data.splits import load_split
from rafr.domains.math.answer_contract import GradeResult
from rafr.domains.math.data import problem_fingerprint
from rafr.domains.math.dataset import validate_public_question
from rafr.domains.math.generation import build_controller_config
from rafr.domains.math.storage import IntegrityError, atomic_json, digest, file_hash, read_json
from rafr.experiments.common import done_keys, make_log_record
from rafr.experiments.exp2 import (
    Trajectory,
    assemble_streams,
    balanced_assignment,
    complete_rounds,
    load_round,
)
from rafr.formatter.fields import compute_fields
from rafr.formatter.math_feedback import format_observation
from rafr.formatter.permute import derangement
from rafr.logging.snapshot import write_snapshot
from rafr.logging.writer import JsonlWriter, read_jsonl
from rafr.validity.cue_diagnostics import CueInputs
from rafr.validity.cue_diagnostics import compute as compute_cues
from rafr.validity.gate import detection_vectors, ladder_check, ladder_ci, reliability_by_level
from rafr.validity.tau import fit_tau

# M.2.1 (2026-10-04 revision): >= 110 E-INCORRECT in each of D_fit and D_gate is a
# scale design target. A shortfall is reported loudly and recorded in the step's
# report; it never stops step 7 / 9 and never adjusts a split.
INCORRECT_DESIGN_TARGET = 110

# M.9 in-place revision 2026-10-08: in the Exp 2 judgments, a text whose E still times out
# on the Math-Verify worker wall clock after two retries is UNRESOLVED; its E record keeps
# this reason. The detail string is the one MathGrader returns (domains/math/grading.py).
WALL_CLOCK_TIMEOUT = "worker_wall_clock_deadline"
E_TIMEOUT_UNRESOLVED_REASON = f"e_timeout_after_retries:{WALL_CLOCK_TIMEOUT}"


def incorrect_count_report(split, errors):
    n_incorrect = sum(errors)
    below = n_incorrect < INCORRECT_DESIGN_TARGET
    if below:
        bar = "!" * 78
        print(
            f"{bar}\nM.2.1 SCALE SHORTFALL: {split.value} has {n_incorrect} E-INCORRECT of "
            f"{len(errors)} states, below the design target of {INCORRECT_DESIGN_TARGET}.\n"
            "Continuing; the count is recorded as e_incorrect_count in this step's report "
            f"and the splits are not adjusted.\n{bar}",
            file=sys.stderr,
            flush=True,
        )
    return {
        "split": split.value,
        "n_states": len(errors),
        "n_incorrect": n_incorrect,
        "design_target": INCORRECT_DESIGN_TARGET,
        "below_design_target": below,
        "rule": "M.2.1 design target; a shortfall is reported, splits are not adjusted",
    }


def finite_json(value):
    if isinstance(value, dict):
        return {str(key): finite_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [finite_json(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def profile_hash_by_separator(config):
    """``MathRunConfig.profile_hash`` under both path separators: it keys each prompt file by
    ``str`` of its relative path, so one configuration hashes as ``prompts\\rewrite.txt`` on the
    laptop (Windows) and ``prompts/rewrite.txt`` on the pod (Linux), every content equal (the
    smoke of 2026-10-16). Same formula as the Core method, whose own value is one of the two."""
    from rafr.domains.math.generation import (
        PROMPT_DIR,
        frozen_models,
        read_controller_retry_prompt,
    )
    from rafr.formatter.math_feedback import PROMPT_DIR as FEEDBACK_DIR
    from rafr.llm.client import canonical_json, sha256_hex

    files = [path for root in (PROMPT_DIR, FEEDBACK_DIR) for path in sorted(root.glob("*.txt"))]
    hashes = {}
    for name, sep in (("windows", "\\"), ("posix", "/")):
        prompts = {
            sep.join(path.relative_to(path.parent.parent).parts): path.read_text(encoding="utf-8")
            for path in files
        }
        hashes[name] = sha256_hex(
            canonical_json(
                {
                    "controller": frozen_models()["controller"],
                    "controller_retry_prompt": read_controller_retry_prompt(),
                    "verifiers": config.verifiers,
                    "math_verify_version": config.math_verify_version,
                    "prompts": prompts,
                    "max_decisions": config.max_decisions,
                }
            )
        )
    if config.profile_hash() not in hashes.values():
        raise IntegrityError("MathRunConfig.profile_hash no longer matches this formula")
    return hashes


def write_rows(path: Path, rows):
    """Use Core row serialization with an atomic publication of a completed stream."""
    partial = path.with_suffix(path.suffix + ".partial")
    partial.unlink(missing_ok=True)
    with JsonlWriter(partial) as writer:
        writer.write_many(rows)
    partial.replace(path)


class MathPipeline:
    def __init__(self, config, root, bundle, *, client, verifiers, grader, feedback=None):
        self.config, self.root, self.bundle = config, Path(root), bundle
        self.client, self.verifiers, self.grader = client, list(verifiers), grader
        if config.math_verify_version != grader.configuration().get("math_verify_version"):
            raise IntegrityError("configured Math-Verify version differs from the E identity")
        self.feedback = feedback or format_observation
        for question in bundle.public_questions:
            validate_public_question(question)
        self.questions = {row.question_id: row for row in bundle.public_questions}
        if config.seed is None:
            raise IntegrityError("M.12: explicitly configure the math seed before execution")
        self.root.mkdir(parents=True, exist_ok=True)
        identity = {
            "bundle": bundle.manifest_sha256,
            "stage": config.stage,
            "execution_mode": config.execution_mode,
            "grader": grader.configuration(),
        }
        snapshot = self.root / "snapshot.json"
        if snapshot.exists():
            if read_json(snapshot).get("math_identity") != identity:
                raise IntegrityError("run identity changed; use a new math run directory")
        else:
            if any(self.root.iterdir()):
                raise IntegrityError("new run directory must be empty")
            write_snapshot(
                self.root,
                step="math-v1.1",
                configs={"math": config.model_dump(mode="json")},
                extra={
                    "domain": "math",
                    "stage": config.stage,
                    "execution_mode": config.execution_mode,
                    "math_identity": identity,
                },
            )
        self._evaluations = {
            row["text_id"]: EvalRecord.model_validate(row)
            for row in read_jsonl(self.root / "evaluations.jsonl")
        }
        self._scores = {
            (row["state_id"], row["verifier_id"]): Score.model_validate(row)
            for row in read_jsonl(self.root / "scores.jsonl")
        }
        # Pending M.12 choices may be filled later; consumed choices may never drift.
        for phase, expected in read_json(snapshot).get("math_phase_bindings", {}).items():
            if expected != self._phase_values(phase):
                raise IntegrityError(f"frozen {phase} configuration changed; rebuild in a new run")

    def _phase_values(self, phase):
        if phase == "generation":
            from rafr.domains.math.generation import PROMPT_DIR, frozen_models

            return {
                "seed": self.config.seed,
                "max_attempts": self.config.max_initial_attempts,
                "models": frozen_models(),
                "prompts": {
                    name: file_hash(PROMPT_DIR / name) for name in ("g_fixed.txt", "task.txt")
                },
            }
        if phase == "splits":
            return {
                "seed": self.config.seed,
                "sizes": [
                    self.config.fit_n,
                    self.config.gate_n,
                    self.config.reserve_n,
                    self.config.experiment_n,
                ],
            }
        if phase == "scoring":
            return {"verifiers": self.config.verifiers}
        if phase == "gate":
            return {"delta_r": self.config.delta_r, "bootstrap_n": self.config.bootstrap_n}
        if phase == "feedback":
            return {"profile_hash": self.config.profile_hash()}
        raise IntegrityError("unknown freeze phase")

    def _bind(self, phase):
        path = self.root / "snapshot.json"
        snapshot = read_json(path)
        bindings = snapshot.setdefault("math_phase_bindings", {})
        value = self._phase_values(phase)
        if phase in bindings:
            if bindings[phase] != value:
                raise IntegrityError(f"frozen {phase} configuration changed")
            return
        bindings[phase] = value
        atomic_json(path, snapshot)

    @property
    def gate_attempts_path(self):
        """All reviewed revisions of one cohort consume the same Core history."""
        snapshot = read_json(self.root / "snapshot.json")
        shared = snapshot.get("math_gate_history_path")
        if shared is None:
            if (self.root / "reserve_revision.json").exists():
                raise IntegrityError(
                    "reserve revision initialization is incomplete; resume it first"
                )
            return self.root / "gate_attempts.json"
        if not (self.root / "reserve_revision_complete.json").exists():
            raise IntegrityError("reserve revision initialization is incomplete; resume it first")
        self._read_fixed(self.root / "reserve_revision_complete.json")
        revision = self._read_fixed(self.root / "reserve_revision.json")
        if Path(shared).resolve() != Path(revision["history_path"]).resolve():
            raise IntegrityError("reserve revision gate history identity changed")
        attempts = label_access.read_gate_attempts(shared)
        if not attempts or digest(attempts[0]) != revision["first_attempt_sha256"]:
            raise IntegrityError("reserve revision first gate history changed")
        return Path(shared)

    def revise_for_reserve(self, prior_root, decision):
        from pipelines.math_revision import initialize_revision

        return initialize_revision(self, prior_root, decision)

    def _grade_key(self, qid, candidate):
        question = self.questions[qid]
        reference = self.bundle.private_references[qid].reference_answer
        return digest([qid, self.grader.cache_key(question.problem, reference, candidate)])

    def grade_once(self, qid, candidate, *, deferred=None):
        """Single E call; callers implement M.2.3's two retries for actual E errors.

        With ``deferred`` (a list), the E log row is appended to it instead of written: a
        worker thread of the initial generation grades without writing, and the main thread
        writes its rows in input order with :meth:`commit_grades`.
        """
        key = self._grade_key(qid, candidate)
        cached = self._evaluations.get(key)
        if cached is None and deferred is not None:
            cached = next(
                (row for kind, row in deferred if kind == "evaluation" and row.text_id == key),
                None,
            )
        if cached is not None:
            status = cached.label.value.lower()
            return GradeResult(
                status=status,
                prediction_answer=cached.extracted_answer,
                detail=cached.reason,
                contract_version=self.grader.configuration().get("contract_version"),
                math_verify_version=cached.math_verify_version,
                cache_key=key,
                n_boxes=cached.n_boxes,
                boxes_all_equivalent=cached.boxes_all_equivalent,
            )
        reference = self.bundle.private_references[qid].reference_answer
        result = self.grader.grade(self.questions[qid].problem, reference, candidate)
        if result.status not in {"correct", "incorrect", "unresolved"}:
            row = {"question_id": qid, "text_id": key, **asdict(result)}
            if deferred is None:
                self.commit_grades([("e_error", row)])
            else:
                deferred.append(("e_error", row))
            return result
        record = self._eval_record(qid, key, candidate, reference, result)
        if deferred is None:
            self.commit_grades([("evaluation", record)])
        else:
            deferred.append(("evaluation", record))
        return result

    def _eval_record(self, qid, key, candidate, reference, result):
        settings = self.grader.configuration()
        return EvalRecord(
            text_id=key,
            text=candidate,
            label=MathLabel(result.status.upper()),
            model="Math-Verify",
            model_version=self.config.math_verify_version,
            prompt_hash=key,
            e_prompt_sha256=digest(settings),
            e_version=self.config.math_verify_version,
            decoding=settings,
            reason=result.detail,
            question_id=qid,
            reference_answer=reference,
            extracted_answer=result.prediction_answer,
            unresolved=result.status == "unresolved",
            math_verify_version=self.config.math_verify_version,
            n_boxes=result.n_boxes,
            boxes_all_equivalent=result.boxes_all_equivalent,
        )

    def commit_grades(self, rows):
        """Write E log rows from :meth:`grade_once`, in the order given."""
        for kind, row in rows:
            if kind == "evaluation":
                with JsonlWriter(self.root / "evaluations.jsonl") as writer:
                    writer.write(row)
                self._evaluations[row.text_id] = row
            else:
                with JsonlWriter(self.root / "e_errors.jsonl") as writer:
                    writer.write(row)

    def e_record(self, qid, candidate):
        return self._evaluations[self._grade_key(qid, candidate)]

    def _grade_after_loop(self, qid, candidate):
        """E for one Exp 2 text after the loop (M.8); an E error is retried twice (II.9.1).

        M.9 in-place revision 2026-10-08: if all three attempts time out on the Math-Verify
        worker wall clock, the text is UNRESOLVED -- a task failure counted in M.8's
        unresolved rate and sensitivity analysis (c) -- and its E record says so, so a
        rerun reads the same label. Any other E error still stops the step (II.9.1). The
        initial answers keep M.2.3 (``math_initial``): there an E error excludes the question.
        """
        attempts = []
        for _ in range(3):
            result = self.grade_once(qid, candidate)
            if result.status in {"correct", "incorrect", "unresolved"}:
                return self._loop_grade(result)
            attempts.append(result)
        if all(r.status == "timeout" and r.detail == WALL_CLOCK_TIMEOUT for r in attempts):
            return self._loop_grade(self._record_timeout_unresolved(qid, candidate, attempts[-1]))
        raise IntegrityError(f"E_error after three attempts: {qid}: {attempts[-1].status}")

    def _record_timeout_unresolved(self, qid, candidate, timeout):
        key = self._grade_key(qid, candidate)
        result = GradeResult(
            status="unresolved",
            detail=E_TIMEOUT_UNRESOLVED_REASON,
            prediction_answer=timeout.prediction_answer,
            contract_version=timeout.contract_version,
            cache_key=key,
            math_verify_version=timeout.math_verify_version,
            n_boxes=timeout.n_boxes,
        )
        reference = self.bundle.private_references[qid].reference_answer
        self.commit_grades(
            [("evaluation", self._eval_record(qid, key, candidate, reference, result))]
        )
        return result

    def _loop_grade(self, result):
        return {
            "status": result.status,
            "is_correct": result.is_correct,
            "task_success": result.status == "correct",
            "prediction_answer": result.prediction_answer,
            "math_verify_version": self.config.math_verify_version,
            "n_boxes": result.n_boxes,
            "boxes_all_equivalent": result.boxes_all_equivalent,
            "e_timeout": result.detail == E_TIMEOUT_UNRESOLVED_REASON,
        }

    def generate(self, *, workers=1):
        from pipelines.math_initial import generate_all

        _ = self.gate_attempts_path  # Reject incomplete reserve transfers before any generation.
        self._bind("generation")
        return generate_all(self, workers=workers)

    def split(self):
        from pipelines.math_initial import split_completed

        _ = self.gate_attempts_path
        self._bind("splits")
        return split_completed(self)

    def _states(self):
        from pipelines.math_initial import require_initial

        require_initial(self.root)
        return {
            row["x0_id"]: State.model_validate(row)
            for row in read_jsonl(self.root / "states.jsonl")
        }

    def _split_ids(self, split):
        split = Split(split)
        info = read_json(self.root / "splits_complete.json")
        for name, expected in info["files"].items():
            if file_hash(self.root / name) != expected:
                raise IntegrityError("split files changed")
        return load_split(split, self.root / "splits")

    def _labels(self, split, purpose):
        from pipelines.math_initial import require_initial

        # Verify the population before opening its labels, not after downstream
        # scoring discovers a changed state/split file.
        require_initial(self.root)
        self._split_ids(split)
        return label_access.read_labels(
            split,
            purpose,
            labels_dir=self.root / "labels",
            split_dir=self.root / "splits",
            gate_attempts_path=self.gate_attempts_path,
        )

    def _verifier_identity(self, index):
        spec = self.config.verifiers[index]
        return str(spec.get("model_id", f"fixture-v{index + 1}")), str(
            spec.get("revision", "fixture")
        )

    def _require_verifier_pair(self):
        if len(self.verifiers) != 2:
            raise IntegrityError("scoring requires the two selected verifier implementations")
        self._bind("scoring")

    def _record_score(self, index, state_id, detail):
        """Validate one verifier output and append its Score and detail rows."""
        verifier_id, version = self._verifier_identity(index)
        row = score_record(verifier_id, version, state_id, detail)
        with JsonlWriter(self.root / "scores.jsonl") as writer:
            writer.write(row)
        with JsonlWriter(self.root / "verifier_details.jsonl") as writer:
            writer.write({"state_id": state_id, "verifier_id": verifier_id, **detail})
        self._scores[(state_id, verifier_id)] = row
        return row

    def _score(self, qid, candidate, state_id):
        records = self._score_items([(qid, state_id, candidate)])[state_id]
        return tuple(row.score for row in records), list(records)

    @staticmethod
    def _batch_size(verifier):
        """States per call: the adapter's fixed batch if it scores in batches, else one."""
        if callable(getattr(verifier, "score_batch", None)):
            return verifier.config.batch_size
        return 1

    def _score_items(self, items):
        """Score (qid, key, text) items, each verifier in its own batches; return key -> rows.

        Every output is computed before any row is written, then rows are appended in
        exactly the order one-by-one scoring writes them: item by item, V1 then V2,
        skipping pairs already on disk (imported or scored before). Fixed-shape batches
        make each score independent of its batch, so the rows are identical as well.
        With nothing pending no verifier is needed (scores imported from the pod).
        """
        identities = [self._verifier_identity(i)[0] for i in range(2)]
        if any((key, vid) not in self._scores for _, key, _ in items for vid in identities):
            self._require_verifier_pair()
        else:
            self._bind("scoring")
        details = {}
        for i, verifier in enumerate(self.verifiers):
            verifier_id = identities[i]
            pending = {}
            for qid, key, text in items:
                if (key, verifier_id) not in self._scores:
                    pending.setdefault(key, (qid, text))
            if not pending:
                continue
            # The only V inputs are public text; no reference-bearing mapping is passed.
            inputs = [(self.questions[qid].problem, text) for qid, text in pending.values()]
            if callable(getattr(verifier, "score_batch", None)):
                outputs = verifier.score_batch(inputs)
                if len(outputs) != len(inputs):
                    raise IntegrityError("V returned a different number of scores than states")
            else:
                outputs = [verifier.score(problem, text) for problem, text in inputs]
            details.update(((key, i), detail) for key, detail in zip(pending, outputs, strict=True))
        for _, key, _ in items:
            for i in range(len(self.verifiers)):
                detail = details.pop((key, i), None)
                if detail is not None:
                    self._record_score(i, key, detail)
        return {key: tuple(self._scores[(key, vid)] for vid in identities) for _, key, _ in items}

    def _score_chunk(self, chunk):
        """Score consecutive (qid, state) pairs (see :meth:`_score_items`)."""
        self._score_items([(qid, state.state_id, state.text) for qid, state in chunk])

    def _score_texts(self, items):
        """The Exp 2 loop's scorer (``math_exp2_loop.ScoreFn``): (x0_id, problem, text, key).

        A text a generative V refuses for its length is returned as overlength, not scored
        (2026-10-13: the trajectory ends there)."""
        for qid, problem, _, _ in items:
            if problem != self.questions[qid].problem:
                raise IntegrityError(f"{qid}: the loop's problem text differs from the bundle")
        overlength = {}
        for i, verifier in enumerate(self.verifiers):
            if not callable(getattr(verifier, "_row", None)):
                continue
            verifier_id = self._verifier_identity(i)[0]
            todo = [
                item
                for item in items
                if item[3] not in overlength and (item[3], verifier_id) not in self._scores
            ]
            if todo and callable(getattr(verifier, "load", None)):
                verifier.load()
            for _, problem, text, key in todo:
                found = input_overlength(verifier, problem, text)
                if found is not None:
                    overlength[key] = {"verifier_id": verifier_id, **found}
        scored = self._score_items(
            [(qid, key, text) for qid, _, text, key in items if key not in overlength]
        )
        return scored, overlength

    def score(self, split):
        split = Split(split)
        label_access.check_split_usable(
            split,
            label_access.Operation.SCORE,
            gate_attempts_path=self.gate_attempts_path,
        )
        states = self._states()
        ids = self._split_ids(split)
        # Chunks are a whole number of every verifier's batches, so no batch is filled
        # with copies except at the end of the split.
        size = math.lcm(*(self._batch_size(verifier) for verifier in self.verifiers))
        for start in range(0, len(ids), size):
            self._score_chunk([(qid, states[qid]) for qid in ids[start : start + size]])
        return {"split": split.value, "n_scored": len(ids)}

    def _initial_pairs(self, ids):
        states = self._states()
        result = []
        for qid in ids:
            try:
                result.append(
                    tuple(
                        self._scores[states[qid].state_id, self._verifier_identity(i)[0]].score
                        for i in range(2)
                    )
                )
            except KeyError as exc:
                raise IntegrityError("complete the required initial verifier scores first") from exc
        return result

    def diagnostics(self, *, stability=False):
        from pipelines.math_verifier_diagnostics import run_diagnostics

        return run_diagnostics(self, stability=stability)

    def require_diagnostics(self):
        """II.4.2 is a required descriptive deliverable, not an effect-size gate."""
        from pipelines.math_verifier_diagnostics import diagnostic_prompt_identity
        from rafr.domains.math.generation import frozen_models

        out = self.root / "verifier_diagnostics"
        marker = self._read_fixed(out / "COMPLETE.json")
        for name, checksum in marker["files"].items():
            if file_hash(out / name) != checksum:
                raise IntegrityError("completed verifier diagnostic files changed")
        report = self._read_fixed(out / "report.json")
        identity = self._read_fixed(out / "identity.json")
        channel = self._channel()
        taus = [channel.tau1, channel.tau2]
        if (
            marker.get("n_per_class") != 150
            or marker.get("n_sampled") != 300
            or marker.get("execution_mode") != "live"
            or marker.get("report_sha256") != file_hash(out / "report.json")
            or marker.get("taus") != taus
            or report.get("tau") != taus
            or report.get("stability_status") != "COMPLETE"
            or report.get("execution_mode") != "live"
            or report.get("sample", {}).get("n") != 300
            or identity.get("models") != frozen_models()["controller"]
            or identity.get("verifiers") != self.config.verifiers
            or identity.get("math_verify_version") != self.config.math_verify_version
            or self._read_fixed(out / "prompts.json") != diagnostic_prompt_identity()
        ):
            raise IntegrityError("formal runs require matching complete 300-state diagnostics")
        return report

    def fit(self):
        from pipelines.math_diagnostics import verifier_diagnostics

        ys = self._labels(Split.D_FIT, label_access.LabelPurpose.TAU_FIT)
        ids = [row.x0_id for row in ys]
        errors = [int(row.label is MathLabel.INCORRECT) for row in ys]
        if set(errors) != {0, 1}:
            raise IntegrityError("D_fit needs both correct and incorrect states")
        incorrect_count = incorrect_count_report(Split.D_FIT, errors)
        pairs = self._initial_pairs(ids)
        pool_ids = self._split_ids(Split.UNLABELED_POOL)
        if set(pool_ids) & set(self._split_ids(Split.D_EXP1)):
            raise IntegrityError("D_Exp1 must be removed before estimating q")
        pool = self._initial_pairs(pool_ids)
        if not pool:
            raise IntegrityError("the post-D_Exp1 unlabeled pool is empty")
        fitted = [fit_tau([pair[i] for pair in pairs], errors, f"math-v{i + 1}") for i in range(2)]
        taus = tuple(item.tau for item in fitted)
        if any(not 0 <= value <= 1 for value in taus):
            raise IntegrityError("fitted threshold outside [0,1]")
        margins = {
            f"v{i + 1}": fit_margin_model([pair[i] for pair in pool], taus[i], f"math-v{i + 1}")
            for i in range(2)
        }
        channel = DegradeChannel(
            {level: LevelParams(p, p) for level, p in zip(LADDER, (0.0, 0.5, 1.0), strict=True)},
            margins,
            *taus,
            component_seed(self.config.seed, "channel"),
        )
        report = finite_json(
            {
                "e_incorrect_count": incorrect_count,
                "tau": [asdict(item) for item in fitted],
                "channel": channel.to_dict(),
                "q_population": "unlabeled_pool after D_Exp1 removal",
                "q_question_ids": pool_ids,
                "q_reads_labels": False,
                "verifier_diagnostics": verifier_diagnostics(pairs, errors, taus),
                "cue_diagnostics": self._cues(ids, pairs, errors, channel, taus),
            }
        )
        self._save_fixed(self.root / "fit.json", report)
        return report

    def _save_fixed(self, path, value):
        value = finite_json(value)
        if path.exists() and read_json(path) != value:
            raise IntegrityError(f"completed {path.name} differs; use a new reviewed run")
        if not path.exists():
            atomic_json(path, value)
        snapshot_path = self.root / "snapshot.json"
        snapshot = read_json(snapshot_path)
        artifacts = snapshot.setdefault("math_artifact_hashes", {})
        key, checksum = str(path.relative_to(self.root)), file_hash(path)
        if key in artifacts and artifacts[key] != checksum:
            raise IntegrityError(f"frozen {path.name} changed")
        artifacts[key] = checksum
        atomic_json(snapshot_path, snapshot)

    def _read_fixed(self, path):
        expected = (
            read_json(self.root / "snapshot.json")
            .get("math_artifact_hashes", {})
            .get(str(path.relative_to(self.root)))
        )
        if expected is None or file_hash(path) != expected:
            raise IntegrityError(
                f"frozen {path.name} changed or is not recorded in the Core snapshot"
            )
        return read_json(path)

    def _channel(self):
        data = self._read_fixed(self.root / "fit.json")["channel"]
        from rafr.channel.margin_fit import MarginModel

        return DegradeChannel(
            {
                Lambda(key): LevelParams(**{f"p_{k}": v for k, v in value.items()})
                for key, value in data["levels"].items()
            },
            {key: MarginModel.from_dict(value) for key, value in data["q"].items()},
            data["tau1"],
            data["tau2"],
            data["seed"],
        )

    def _cues(self, ids, pairs, errors, channel, taus):
        report = {}
        for level in (Lambda.NONE, *LADDER):
            observed = [channel(qid, *pair, level) for qid, pair in zip(ids, pairs, strict=True)]
            groups = {
                "incorrect": [i for i, y in enumerate(errors) if y],
                "all": list(range(len(ids))),
            }
            report[level.value] = {}
            for population, chosen in groups.items():
                report[level.value][population] = (
                    asdict(
                        compute_cues(
                            CueInputs(
                                [observed[i][0] for i in chosen],
                                [observed[i][1] for i in chosen],
                                *taus,
                                [errors[i] for i in chosen],
                            ),
                            high_margin_quantile=0.75,
                        )
                    )
                    if chosen
                    else {"n": 0}
                )
        return finite_json(report)

    def channel_check(self):
        """Step 8: inspect D_fit only, before consuming any held-out gate labels.

        the authors may revise the proposed delta_R using this report before the gate
        consumes it. After that point the existing phase/artifact bindings apply.
        """
        from pipelines.math_diagnostics import channel_descriptive_diagnostics, channel_diagnostics
        from rafr.validity.gate import GAP_TOLERANCE

        if self.config.delta_r is None:
            raise IntegrityError("M.12: provide the proposed delta_R for the D_fit channel check")
        frozen = "gate" in read_json(self.root / "snapshot.json").get("math_phase_bindings", {})
        if frozen:
            self._bind("gate")
        labels = self._labels(Split.D_FIT, label_access.LabelPurpose.CHANNEL_CHECK)
        ids = [row.x0_id for row in labels]
        errors = [int(row.label is MathLabel.INCORRECT) for row in labels]
        channel = self._channel()
        pairs = self._initial_pairs(ids)
        report = channel_diagnostics(channel, pairs, errors, ids, self.config.delta_r)
        reachability = report.get("per_verifier", {})
        passed = len(reachability) == 2 and all(
            row["expected_target_shortfall"] <= GAP_TOLERANCE for row in reachability.values()
        )
        observed = {}
        for level in LADDER:
            values = [channel(qid, *pair, level) for qid, pair in zip(ids, pairs, strict=True)]
            observed[level] = ([v[0] for v in values], [v[1] for v in values])
        vectors = detection_vectors(errors, observed, channel.tau1, channel.tau2)
        descriptive = channel_descriptive_diagnostics(
            errors, observed, (channel.tau1, channel.tau2)
        )
        report.update(
            status="READY_FOR_GATE" if passed else "CHANNEL_CHECK_FAILED",
            passed=passed,
            held_out_labels_read=False,
            gate_attempt_consumed=False,
            paired_correctness_correlation=descriptive["paired_correctness_correlation"],
            realized_ladder=asdict(
                ladder_check(
                    reliability_by_level(vectors),
                    self.config.delta_r,
                    descriptive["p_at_least_one_wrong"],
                    ci=ladder_ci(
                        vectors,
                        n_boot=self.config.bootstrap_n,
                        seed=derive_seed(self.config.seed, "D_fit_channel_check"),
                    ),
                    n_states=sum(errors),
                    descriptive=descriptive["descriptive"],
                )
            ),
        )
        report = finite_json(report)
        path = self.root / "channel_check.json"
        if frozen:
            self._save_fixed(path, report)
        else:
            atomic_json(path, report)
        return report

    def gate(self, attempt=1, *, reserve_decision=None):
        from pipelines.math_diagnostics import channel_descriptive_diagnostics
        from rafr.validity.gate import LadderResult, next_attempt_split, record_attempt

        if attempt not in (1, 2):
            raise IntegrityError("at most two gate attempts")
        path = self.gate_attempts_path
        attempts = label_access.read_gate_attempts(path)
        if len(attempts) >= attempt:
            report = self._read_fixed(self.root / f"gate_{attempt}.json")
            if report["passed"] is not attempts[attempt - 1]["passed"]:
                raise IntegrityError("gate marker contradicts the gate result")
            owner = attempts[attempt - 1].get("run_dir")
            if owner and Path(owner).resolve() != self.root.resolve():
                raise IntegrityError("this gate attempt belongs to another reviewed revision")
            self._bind("gate")
            expected = attempts[attempt - 1].get("gate_report_sha256")
            if expected and expected != file_hash(self.root / f"gate_{attempt}.json"):
                raise IntegrityError("Core gate history contradicts the frozen gate report")
            return report
        if attempt == 2:
            if len(attempts) != 1 or attempts[0]["passed"] is not False:
                raise IntegrityError("reserve requires a recorded failed first gate")
            if reserve_decision is None and (self.root / "reserve_revision.json").exists():
                reserve_decision = self._read_fixed(self.root / "reserve_revision.json")["decision"]
            if not isinstance(reserve_decision, dict) or any(
                not isinstance(reserve_decision.get(key), str) or not reserve_decision[key].strip()
                for key in ("reviewed_by", "decision")
            ):
                raise IntegrityError(
                    "the authors must record their redesign/reserve decision before gate 2"
                )
        split = next_attempt_split(path)
        if split is not (Split.D_GATE if attempt == 1 else Split.D_GATE_RESERVE):
            raise IntegrityError("gate attempt does not match the shared Core history")
        completed_path = self.root / f"gate_{attempt}.json"
        if completed_path.exists():
            # Recover a completed report whose attempt record was interrupted.
            # Its parameters and contents were frozen before publication.
            self._bind("gate")
            report = self._read_fixed(completed_path)
            if report["attempt"] != attempt or report["split"] != split.value:
                raise IntegrityError("completed gate report has another attempt identity")
            checked = LadderResult(
                **{name: report[name] for name in LadderResult.__dataclass_fields__}
            )
            record_attempt(
                path,
                checked,
                split,
                extra={
                    "run_dir": str(self.root.resolve()),
                    "gate_report_sha256": file_hash(completed_path),
                },
            )
            return report
        # The expected-span check must pass BEFORE a held-out label is read or a
        # failed attempt could unlock reserve. A pre-gate failure is not a gate.
        channel_check = self.channel_check()
        if not channel_check["passed"]:
            raise IntegrityError(
                "D_fit channel check failed before held-out gate; stop for the authors' review "
                "without consuming a gate attempt or unlocking reserve"
            )
        self._bind("gate")
        self._save_fixed(self.root / "channel_check.json", channel_check)
        ys = self._labels(split, label_access.LabelPurpose.GATE)
        ids = [row.x0_id for row in ys]
        errors = [int(row.label is MathLabel.INCORRECT) for row in ys]
        incorrect_count = incorrect_count_report(split, errors)
        if not any(errors):
            raise IntegrityError("gate has no E-INCORRECT states")
        channel = self._channel()
        pairs = self._initial_pairs(ids)
        observed = {}
        for level in LADDER:
            values = [channel(qid, *pair, level) for qid, pair in zip(ids, pairs, strict=True)]
            observed[level] = ([v[0] for v in values], [v[1] for v in values])
        vectors = detection_vectors(errors, observed, channel.tau1, channel.tau2)
        descriptive = channel_descriptive_diagnostics(
            errors, observed, (channel.tau1, channel.tau2)
        )
        checked = ladder_check(
            reliability_by_level(vectors),
            self.config.delta_r,
            descriptive["p_at_least_one_wrong"],
            ci=ladder_ci(
                vectors,
                n_boot=self.config.bootstrap_n,
                seed=derive_seed(self.config.seed, "gate", attempt),
            ),
            n_states=sum(errors),
            descriptive=descriptive["descriptive"],
        )
        from analysis.reliability import r_ref

        report = finite_json(
            {
                **asdict(checked),
                "attempt": attempt,
                "split": split.value,
                "e_incorrect_count": incorrect_count,
                "reserve_decision": reserve_decision,
                "paired_correctness_correlation": descriptive["paired_correctness_correlation"],
                "r_ref": {
                    "split": split.value,
                    "channel": "none",
                    "positive_class": "INCORRECT",
                    "use": "offline descriptive reference only, never a design factor",
                    "by_verifier": {
                        f"v{i + 1}": r_ref([pair[i] for pair in pairs], errors, tau)
                        for i, tau in enumerate((channel.tau1, channel.tau2))
                    },
                },
                "cue_diagnostics": self._cues(
                    ids, pairs, errors, channel, (channel.tau1, channel.tau2)
                ),
            }
        )
        self._save_fixed(self.root / f"gate_{attempt}.json", report)
        record_attempt(
            path,
            checked,
            split,
            extra={
                "run_dir": str(self.root.resolve()),
                "gate_report_sha256": file_hash(completed_path),
            },
        )
        return report

    def require_gate(self):
        attempts = label_access.read_gate_attempts(self.gate_attempts_path)
        if not attempts or attempts[-1]["passed"] is not True:
            raise IntegrityError(
                "gate not passed; stop and ask the authors, never auto-enable reserve"
            )
        report = self.gate(attempts[-1]["attempt"])
        if report.get("passed") is not True or report.get("attempt") != attempts[-1]["attempt"]:
            raise IntegrityError("gate marker contradicts the gate result")
        if "cue_diagnostics" not in report:
            raise IntegrityError("Core cue diagnostics must precede any A1-B analysis")

    def _initial_cohort(self, *, smoke=False, purpose=None):
        split = Split.UNLABELED_POOL if smoke else Split.D_EXP1
        purpose = purpose or (
            label_access.LabelPurpose.DRY_RUN if smoke else label_access.LabelPurpose.EXP1_OUTCOME
        )
        ys = self._labels(split, purpose)
        states = self._states()
        ids = [row.x0_id for row in ys if row.label is MathLabel.INCORRECT]
        if smoke:
            if len(ids) < 30:
                raise IntegrityError("M.10: smoke needs 30 E-INCORRECT unlabeled-pool states")
            ids = sorted(ids, key=lambda qid: digest([self.config.seed, "math-smoke", qid]))[:30]
        elif len(ids) != self.config.experiment_n:
            raise IntegrityError("D_Exp1 must contain exactly the selected incorrect states")
        return [states[qid] for qid in ids]

    def exp1(self):
        if self.config.stage == "smoke":
            raise IntegrityError("M.10: smoke cannot run arm comparisons")
        self.require_gate()
        if self.config.stage == "formal":
            self.require_diagnostics()
        if self.config.execution_mode == "live":
            self.require_smoke_for_exp1()
        self._bind("feedback")
        out = self.root / "exp1"
        out.mkdir(exist_ok=True)
        self.record_exp1_outcome(out)  # 0.5 step 11: before any controller call
        states = self._initial_cohort()
        channel = self._channel()
        ids = [state.state_id for state in states]
        donors = derangement(ids, component_seed(self.config.seed, "permuted_b"))
        self._save_fixed(out / "permutation.json", donors)
        pairs = dict(zip(ids, self._initial_pairs([s.x0_id for s in states]), strict=True))
        builder = ObservationBuilder(
            channel,
            channel.tau1,
            channel.tau2,
            self._verifier_identity(0)[0],
            self._verifier_identity(1)[0],
        )
        cfg = build_controller_config(self.config.seed, 1)
        completed = done_keys(out / "decisions.jsonl")
        channel_rows = []
        with JsonlWriter(out / "decisions.jsonl") as log:
            for state in states:
                for level in LADDER:
                    obs = builder.build(state.state_id, *pairs[state.state_id], level)
                    donor = builder.build(
                        donors[state.state_id], *pairs[donors[state.state_id]], level
                    )
                    for arm in (Arm.A1, Arm.B, Arm.PERMUTED_B):
                        channel_rows.append(
                            {
                                "state_id": state.state_id,
                                "lam": level.value,
                                "arm": arm.value,
                                **builder.assignment(state.state_id, level),
                            }
                        )
                        key = state.state_id, arm.value, level.value, 1
                        if key in completed:
                            continue
                        feedback = self.feedback(
                            arm,
                            obs,
                            **(
                                {
                                    "donor_observation": donor,
                                    "permuted_fields": compute_fields(donor),
                                }
                                if arm is Arm.PERMUTED_B
                                else {}
                            ),
                        )
                        decision = controller.decide(
                            self.client,
                            cfg,
                            x0=state.x0,
                            x_t=state.text,
                            observation=feedback,
                            labels=controller.DecisionLabels(
                                state.state_id, arm, level, 1, "math_exp1"
                            ),
                        )
                        log.write(
                            make_log_record(
                                run_id=self.root.name,
                                experiment="exp1",
                                seed=self.config.seed,
                                arm=arm,
                                lam=level,
                                t=1,
                                state_id=state.state_id,
                                x0_id=state.x0_id,
                                x_t=state.text,
                                obs=obs,
                                fields=compute_fields(donor if arm is Arm.PERMUTED_B else obs),
                                decision=decision,
                                permuted_from=donor.state_id if arm is Arm.PERMUTED_B else None,
                            )
                        )
                        completed.add(key)
        expected = {
            (sid, arm.value, level.value, 1)
            for sid in ids
            for arm in (Arm.A1, Arm.B, Arm.PERMUTED_B)
            for level in LADDER
        }
        if completed != expected:
            raise IntegrityError("Exp1 decision coverage does not match nine cells per state")
        # Pure channel rows can be rebuilt after an interruption without a model call.
        write_rows(out / "channel.jsonl", channel_rows)
        self._save_fixed(
            out / "COMPLETE.json",
            {
                "n_questions": len(states),
                "n_decisions": len(completed),
                "decisions_sha256": file_hash(out / "decisions.jsonl"),
                "channel_sha256": file_hash(out / "channel.jsonl"),
            },
        )
        return {"n_questions": len(states), "n_decisions": len(completed)}

    def require_exp1(self):
        """0.5 step 12 consumes a completed, unchanged step 11 in formal runs."""
        out = self.root / "exp1"
        marker = self._read_fixed(out / "COMPLETE.json")
        if (
            marker.get("n_questions") != self.config.experiment_n
            or marker.get("n_decisions") != 9 * self.config.experiment_n
        ):
            raise IntegrityError("formal Exp2 requires complete Exp1 decision coverage")
        for key, filename in (
            ("decisions_sha256", "decisions.jsonl"),
            ("channel_sha256", "channel.jsonl"),
        ):
            if marker.get(key) != file_hash(out / filename):
                raise IntegrityError("formal Exp2 requires unchanged completed Exp1 evidence")
        return marker

    def exp2(self, *, smoke=False):
        if self.config.stage == "smoke" and not smoke:
            raise IntegrityError("M.10: smoke cannot run arm comparisons")
        if not smoke:
            if self.config.stage == "formal":
                self.require_exp1()
                self.require_diagnostics()
            self.require_gate()
            if self.config.execution_mode == "live":
                self.require_smoke()
        self._bind("feedback")
        states = self._initial_cohort(
            smoke=smoke,
            purpose=label_access.LabelPurpose.DRY_RUN
            if smoke
            else label_access.LabelPurpose.EXP2_T1,
        )
        channel = self._channel()
        names = self._loop_names(smoke)
        out = self.root / names["experiment"]
        out.mkdir(exist_ok=True)
        assignment = self._loop_assignment(states, smoke)
        self._save_fixed(out / "assignment.json", assignment)
        trajectories = run_rounds(
            out,
            [Trajectory(s.x0_id, s.x0, assignment[s.x0_id], 1, s.text) for s in states],
            taus=(channel.tau1, channel.tau2),
            verifier_ids=(self._verifier_identity(0)[0], self._verifier_identity(1)[0]),
            score_texts=self._score_texts,
            client=self.client,
            cfg=build_controller_config(self.config.seed, 4),
            feedback=self.feedback,
            namespace=names["namespace"],
            experiment=names["experiment"],
            run_id=self.root.name,
            seed=self.config.seed,
            smoke=smoke,
        )
        self._save_fixed(
            out / "loop_complete.json",
            {
                "k": 4,
                "rounds": complete_rounds(out, 4),
                "n_trajectories": len(trajectories),
                "SMOKE": smoke,
                "execution_mode": self.config.execution_mode,
                "thresholds": [channel.tau1, channel.tau2],
                "verifiers": [self._verifier_identity(i)[0] for i in range(2)],
                "profile_hash": self.config.profile_hash(),
            },
        )
        return {"n_trajectories": len(trajectories), "SMOKE": smoke}

    @staticmethod
    def _loop_names(smoke):
        return {
            "namespace": "math_exp2_smoke" if smoke else "math_exp2",
            "experiment": "exp2_smoke" if smoke else "exp2",
            "arms": (Arm.A1,) if smoke else (Arm.A1, Arm.B),
        }

    def _loop_assignment(self, states, smoke):
        names = self._loop_names(smoke)
        assignment = balanced_assignment(
            [s.x0_id for s in states],
            names["arms"],
            derive_seed(self.config.seed, names["namespace"], "assignment"),
        )
        return {key: arm.value for key, arm in assignment.items()}

    def install_loop(self, src, *, smoke, taus=None):
        """Take a loop the pod ran (``env/math_exp2_remote.py``) into this run, as
        ``exp2`` would have written it, so that ``judge`` and ``report`` run on the laptop
        (2026-10-13: E's judgments back on the laptop, no reference on the pod).

        Checked first: four complete rounds (every file against its round's sha256); the
        pod's ``loop_complete.json`` (SMOKE, live, this configuration's profile hash and
        verifiers; the thresholds -- the step 6 session tau for the smoke, given as
        ``taus``, step 7's frozen tau for the formal run); the first states and the arms
        exactly those ``exp2`` would start from. A second install of the same loop is a
        no-op; another loop into a non-empty directory is refused.
        """
        import shutil

        src = Path(src)
        names = self._loop_names(smoke)
        loop = read_json(src / "loop_complete.json")
        if complete_rounds(src, 4) != [1, 2, 3, 4] or loop.get("rounds") != [1, 2, 3, 4]:
            raise IntegrityError("the pulled loop does not hold four complete, verified rounds")
        expected_taus = list(taus) if smoke else [self._channel().tau1, self._channel().tau2]
        if taus is None and smoke:
            raise IntegrityError("installing the smoke needs the session tau")
        checks = {
            "SMOKE": smoke,
            "execution_mode": self.config.execution_mode,
            "verifiers": [self._verifier_identity(i)[0] for i in range(2)],
            "thresholds": expected_taus,
        }
        for key, value in checks.items():
            if loop.get(key) != value:
                raise IntegrityError(
                    f"the pulled loop's {key} is {loop.get(key)!r}, expected {value!r}"
                )
        # the pod (Linux) and the laptop (Windows) differ only by the prompt keys' separator
        profiles = profile_hash_by_separator(self.config)
        if loop.get("profile_hash") not in profiles.values():
            raise IntegrityError(
                f"the pulled loop's profile_hash is {loop.get('profile_hash')!r}, expected one "
                f"of {profiles!r}"
            )
        states = self._initial_cohort(
            smoke=smoke,
            purpose=label_access.LabelPurpose.DRY_RUN
            if smoke
            else label_access.LabelPurpose.EXP2_T1,
        )
        assignment = self._loop_assignment(states, smoke)
        if read_json(src / "assignment.json") != assignment:
            raise IntegrityError("the pulled loop's arms are not the seeded assignment")
        first = load_round(src, 1)
        if [(tr.x0_id, tr.history[0]["x_t"], tr.arm) for tr in first] != [
            (s.x0_id, s.text, assignment[s.x0_id]) for s in states
        ]:
            raise IntegrityError("the pulled loop did not start from this run's first states")
        out = self.root / names["experiment"]
        if (out / "loop_complete.json").exists():
            if self._read_fixed(out / "loop_complete.json") != loop:
                raise IntegrityError(f"{out} already holds another loop")
            return {"installed": False, "already": str(out)}
        if out.exists() and any(out.iterdir()):
            raise IntegrityError(f"{out} is not empty")
        self._bind("feedback")
        out.mkdir(parents=True, exist_ok=True)
        shutil.copytree(src / "rounds", out / "rounds")
        for name in (
            "ledger.jsonl",
            "transient_retries.jsonl",
            "text_scores.jsonl",
            "text_score_details.jsonl",
        ):
            if (src / name).is_file():
                shutil.copy2(src / name, out / name)
        if complete_rounds(out, 4) != [1, 2, 3, 4]:
            raise IntegrityError("a round changed while it was copied")
        assemble_streams(out, 4)
        self._save_fixed(out / "assignment.json", assignment)
        self._save_fixed(out / "loop_complete.json", loop)
        return {
            "installed": True,
            "into": str(out),
            "n_trajectories": loop["n_trajectories"],
            "SMOKE": smoke,
            "thresholds": loop["thresholds"],
        }

    def judge(self, *, smoke=False):
        out = self.root / ("exp2_smoke" if smoke else "exp2")
        if not (out / "loop_complete.json").exists() or complete_rounds(out, 4) != [1, 2, 3, 4]:
            raise IntegrityError("E can judge only after the entire loop is complete")
        self._read_fixed(out / "loop_complete.json")
        marker = out / "judge_complete.json"
        marker.unlink(missing_ok=True)
        decisions = {(r["x0_id"], r["t"]): r for r in read_jsonl(out / "decisions.jsonl")}
        rows, round_rows = [], []
        for tr in load_round(out, 4):
            rounds = []
            for i, history in enumerate(tr.history):
                text_before = history["x_t"]
                text_after = tr.history[i + 1]["x_t"] if i + 1 < len(tr.history) else tr.x_t
                decision = decisions[tr.x0_id, history["t"]]
                before = self._grade_after_loop(tr.x0_id, text_before)
                after = self._grade_after_loop(tr.x0_id, text_after)
                rounds.append(
                    {
                        "t": history["t"],
                        "action": history["action"],
                        "parse_fail": history["parse_fail"],
                        "input_solution": text_before,
                        "output_solution": text_after,
                        "input_grade": before,
                        "output_grade": after,
                        "input_scores": [decision["s1"], decision["s2"]],
                        "thresholds": [decision["tau1"], decision["tau2"]],
                    }
                )
                round_rows.append({"question_id": tr.x0_id, "arm": tr.arm, **rounds[-1]})
            terminal = self._grade_after_loop(tr.x0_id, tr.x_t)
            # 2026-10-13: a rewrite too long for a V ends the trajectory; it is x_T
            overlength = tr.history[-1].get(OVERLENGTH) if tr.history else None
            rows.append(
                {
                    "domain": "math",
                    "stage": self.config.stage,
                    "execution_mode": self.config.execution_mode,
                    "run_id": self.root.name,
                    "question_id": tr.x0_id,
                    "arm": tr.arm,
                    "initial_status": "incorrect",
                    "rounds": rounds,
                    "terminal_status": terminal["status"],
                    "terminal_solution": tr.x_t,
                    "task_success": terminal["task_success"],
                    "is_correct": terminal["is_correct"],
                    "termination_reason": "accepted"
                    if tr.accepted
                    else OVERLENGTH
                    if overlength
                    else "decision_limit",
                    OVERLENGTH: overlength,
                }
            )
        write_rows(out / "round_judgments.jsonl", round_rows)
        write_rows(out / "trajectories.jsonl", rows)
        atomic_json(
            marker,
            {
                "files": {
                    name: file_hash(out / name)
                    for name in ("round_judgments.jsonl", "trajectories.jsonl", "decisions.jsonl")
                },
                "SMOKE": smoke,
            },
        )
        return {"n_judged": len(rows), "SMOKE": smoke}

    def _require_judged(self, out):
        if complete_rounds(out, 4) != [1, 2, 3, 4]:
            raise IntegrityError("Core round files changed or are incomplete")
        marker = read_json(out / "judge_complete.json")
        for name, expected in marker["files"].items():
            if file_hash(out / name) != expected:
                raise IntegrityError("judged run files changed")
        return list(read_jsonl(out / "trajectories.jsonl"))

    def smoke(self):
        self.exp2(smoke=True)
        self.judge(smoke=True)
        return self.report(smoke=True)

    def record_exp1_outcome(self, out):
        """0.5 step 11 / II.9.1: Exp 1's outcome is D_Exp1's y* from step 5 (E); its hash is
        recorded before any controller call of Exp 1, its single-round smoke included. Frozen."""
        ys = self._labels(Split.D_EXP1, label_access.LabelPurpose.EXP1_OUTCOME)
        rows = [[row.x0_id, row.label.value] for row in ys]
        # Only D_Exp1's rows, read through the label gate: the y* file itself also holds
        # D_gate_reserve's judgments and is never opened here (II.6.1).
        record = {
            "split": Split.D_EXP1.value,
            "source": "step 5's E judgments of D_Exp1 via read_labels (purpose exp1_outcome)",
            "d_exp1_labels_sha256": digest(rows),
            "n": len(rows),
            "n_incorrect": sum(row.label is MathLabel.INCORRECT for row in ys),
        }
        self._save_fixed(out / "outcome_hash.json", record)
        return read_json(out / "outcome_hash.json"), {row.x0_id: row.label for row in ys}

    def exp1_smoke(self):
        """Math Exp 1's single-round live smoke (2026-10-15; II.4.4 / M.10)."""
        from pipelines.math_exp1_smoke import run

        if self.config.stage == "smoke":
            raise IntegrityError("the Exp 1 smoke belongs to the experiment run")
        return run(self)

    def require_smoke_for_exp1(self):
        """II.4.4 / M.10 (2026-10-15): math's Exp 1 may run before the Exp 2 smoke once its
        own single-round live smoke passed; with an Exp 2 smoke report on disk, that one rules."""
        from pipelines.math_exp1_smoke import require

        if (self._exp2_smoke_dir() / "smoke_report.json").exists():
            return self.require_smoke()
        return require(self)

    def _exp2_smoke_dir(self):
        from rafr.core import paths

        return (
            paths.resolve(self.config.live_smoke_run)
            if self.config.live_smoke_run
            else self.root / "exp2_smoke"
        )

    def require_smoke(self):
        out = self._exp2_smoke_dir()
        report = read_json(out / "smoke_report.json")
        if (
            report.get("SMOKE") is not True
            or report.get("execution_mode") != "live"
            or report.get("engineering_defects") != []
            or report.get("n_questions") != 30
            or report.get("profile_hash") != self.config.profile_hash()
        ):
            raise IntegrityError(
                "M.10: require a matching real A1 smoke without engineering defects"
            )
        trajectories = self._require_judged(out)
        producer = read_json(out.parent / "snapshot.json")
        loop = read_json(out / "loop_complete.json")
        if (
            producer.get("domain") != "math"
            or producer.get("execution_mode") != "live"
            or producer.get("math_identity", {}).get("execution_mode") != "live"
            or producer.get("math_identity", {}).get("grader") != self.grader.configuration()
            or producer.get("math_phase_bindings", {}).get("feedback", {}).get("profile_hash")
            != report["profile_hash"]
            or loop.get("execution_mode") != "live"
            or loop.get("SMOKE") is not True
        ):
            raise IntegrityError("smoke report disagrees with its producing run")
        if len(trajectories) != 30 or any(
            row["arm"] != "A1" or row.get("execution_mode") != "live" for row in trajectories
        ):
            raise IntegrityError("smoke trajectory coverage differs")
        audit_info = report.get("prompt_audit", {})
        audit_path = out / "prompt_audit.json"
        if (
            audit_info.get("sha256") != file_hash(audit_path)
            or audit_info.get("n_trajectories") != 3
            or read_json(audit_path).get("engineering_defects") != []
        ):
            raise IntegrityError("smoke requires the actual prompt audit for three trajectories")
        if report.get("judge_complete_sha256") != file_hash(out / "judge_complete.json"):
            raise IntegrityError("smoke report is not bound to its completed judgments")
        current = self._channel()
        if report.get("thresholds") != [current.tau1, current.tau2]:
            raise IntegrityError("smoke thresholds differ from the frozen experiment thresholds")
        smoke_questions = report.get("public_questions", [])
        if len(smoke_questions) != 30:
            raise IntegrityError("smoke exposure list is missing")
        exposed_ids = {q["question_id"] for q in smoke_questions}
        exposed_texts = {problem_fingerprint(q["problem"]) for q in smoke_questions}
        for qid in self._split_ids(Split.D_EXP1):
            if (
                qid in exposed_ids
                or problem_fingerprint(self.questions[qid].problem) in exposed_texts
            ):
                raise IntegrityError("smoke exposure overlaps formal D_Exp1")

    def _exp1_rows(self):
        """Exp 1's completed decisions as the analysis rows (gate, diagnostics and the frozen
        completion marker checked first)."""
        self.require_gate()
        exp1_dir = self.root / "exp1"
        if self.config.stage == "formal":
            self.require_diagnostics()
        marker = self._read_fixed(exp1_dir / "COMPLETE.json")
        if marker["decisions_sha256"] != file_hash(exp1_dir / "decisions.jsonl"):
            raise IntegrityError("Exp1 decisions changed")
        if marker["channel_sha256"] != file_hash(exp1_dir / "channel.jsonl"):
            raise IntegrityError("Exp1 channel changed")
        exp1_rows = []
        channel_rows = {
            (r["state_id"], r["lam"], r["arm"]): r for r in read_jsonl(exp1_dir / "channel.jsonl")
        }
        for row in read_jsonl(exp1_dir / "decisions.jsonl"):
            exp1_rows.append(
                {
                    "question_id": row["x0_id"],
                    "initial_status": "incorrect",
                    "arm": row["arm"],
                    "level": row["lam"],
                    "action": row["action"],
                    "parse_fail": row["parse_fail"],
                    "candidate_hash": digest(row["x_t"]),
                    "input_scores": [row["s1"], row["s2"]],
                    "observed_scores": [row["s1_tilde"], row["s2_tilde"]],
                    "thresholds": [row["tau1"], row["tau2"]],
                    "prompt_hash": row["prompt_hash"],
                    "attempts": row["attempts"],
                    "channel_assignment": channel_rows[row["state_id"], row["lam"], row["arm"]],
                }
            )
        return exp1_rows

    def report_exp1(self):
        """Exp 1's preregistered report before Exp 2 exists (2026-10-15: math's Exp 1 may
        precede the Exp 2 smoke): everything ``report`` computes from Exp 1 alone -- the primary
        and permuted-B contrasts with sensitivities (a) and (b), the Exp 1 diagnostics with the
        gate's r_ref, the drift against the gate, the Exp 1 error boundary and result shape.
        Transport and every Exp 2 table wait for Exp 2 (``report``)."""
        from pipelines.math_analysis import summarize_exp1
        from pipelines.math_diagnostics import exp1_diagnostics, exp1_outcome_diagnostics

        exp1_rows = self._exp1_rows()
        attempts = label_access.read_gate_attempts(self.gate_attempts_path)
        gate_report = self._read_fixed(self.root / f"gate_{attempts[-1]['attempt']}.json")
        report = {
            "domain": "math",
            "protocol": "v1.2 Appendix M",
            "stage": self.config.stage,
            "execution_mode": self.config.execution_mode,
            "is_research_result": self.config.stage == "formal"
            and self.config.execution_mode == "live",
            "exp1": summarize_exp1(exp1_rows, self.config.bootstrap_n, self.config.seed),
            "exp1_diagnostics": {
                **exp1_diagnostics(exp1_rows),
                "h2_reference_r_ref": gate_report["r_ref"],
            },
            **exp1_outcome_diagnostics(
                exp1_rows, gate_report["r"], n_boot=self.config.bootstrap_n, seed=self.config.seed
            ),
            "not_in_this_report": "Exp 2, its reliability diagnostics and the transport "
            "diagnostic (they need Exp 2; MathPipeline.report)",
        }
        atomic_json(self.root / "exp1" / "report_exp1.json", finite_json(report))
        return report

    def report(self, *, smoke=False):
        from pipelines.math_analysis import summarize_exp1, summarize_exp2, summarize_smoke
        from pipelines.math_diagnostics import (
            exp1_diagnostics,
            exp2_reliability_diagnostics,
            protocol_outcome_diagnostics,
        )

        out = self.root / ("exp2_smoke" if smoke else "exp2")
        trajectories = self._require_judged(out)
        if smoke:
            from pipelines.math_smoke import audit_smoke

            report = summarize_smoke(trajectories)
            audit = audit_smoke(self, out, trajectories)
            # M.10 (2026-10-12): the thresholds the loop ran with -- in step 6, the
            # session tau, computed before step 7 freezes the same rule on the same data;
            # require_smoke compares them with the frozen ones.
            loop = self._read_fixed(out / "loop_complete.json")
            thresholds = loop.get("thresholds")
            if thresholds is None:
                channel = self._channel()
                thresholds = [channel.tau1, channel.tau2]
            report.update(
                {
                    "SMOKE": True,
                    "is_research_result": False,
                    "n_questions": len(trajectories),
                    "execution_mode": self.config.execution_mode,
                    "profile_hash": self.config.profile_hash(),
                    **audit,
                    "thresholds": list(thresholds),
                    "judge_complete_sha256": file_hash(out / "judge_complete.json"),
                    "public_questions": [
                        self.questions[row["question_id"]].public_payload() for row in trajectories
                    ],
                }
            )
            atomic_json(out / "smoke_report.json", finite_json(report))
            return report
        exp1_rows = self._exp1_rows()
        report = {
            "domain": "math",
            "protocol": "v1.2 Appendix M",
            "stage": self.config.stage,
            "execution_mode": self.config.execution_mode,
            "is_research_result": self.config.stage == "formal"
            and self.config.execution_mode == "live",
            "exp1": summarize_exp1(exp1_rows, self.config.bootstrap_n, self.config.seed),
            "exp2": summarize_exp2(trajectories, self.config.bootstrap_n, self.config.seed),
            "exp1_diagnostics": exp1_diagnostics(exp1_rows),
            "exp2_reliability_diagnostics": exp2_reliability_diagnostics(trajectories),
        }
        attempts = label_access.read_gate_attempts(self.gate_attempts_path)
        gate_report = self._read_fixed(self.root / f"gate_{attempts[-1]['attempt']}.json")
        report["exp1_diagnostics"]["h2_reference_r_ref"] = gate_report["r_ref"]
        report.update(
            protocol_outcome_diagnostics(
                exp1_rows,
                trajectories,
                gate_report["r"],
                n_boot=self.config.bootstrap_n,
                seed=self.config.seed,
            )
        )
        atomic_json(self.root / "report.json", finite_json(report))
        return report

    def run_all(self, *, workers=1):
        if (self.root / "reserve_revision.json").exists():
            raise IntegrityError(
                "reserve revision requires explicit staged execution; all is disabled"
            )
        self.generate(workers=workers)
        self.split()
        for split in (Split.D_FIT, Split.UNLABELED_POOL):
            self.score(split)
        self.diagnostics(
            stability=self.config.stage == "formal" and self.config.execution_mode == "live"
        )
        self.fit()
        if self.config.stage == "smoke":
            return self.smoke()
        reachability = self.channel_check()
        if not reachability["passed"]:
            return {
                "status": "CHANNEL_CHECK_FAILED_STOPPED",
                "channel_check": reachability,
                "reserve_used": False,
            }
        self.score(Split.D_GATE)
        gate = self.gate()
        if not gate["passed"]:
            return {"status": "GATE_FAILED_STOPPED", "gate": gate, "reserve_used": False}
        smoke = self.smoke()
        if smoke["engineering_defects"]:
            return {"status": "SMOKE_ENGINEERING_DEFECTS", "smoke": smoke}
        self.score(Split.D_EXP1)
        self.exp1()
        self.exp2()
        self.judge()
        return self.report()
