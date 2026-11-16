"""paper Sec. 5, App. D / diagnostics using Core sampling and metric implementations.

Only D_fit and pool labels are consumed. SafetyLabel exists solely at the Core
sampler boundary; every persisted label is CORRECT/INCORRECT/UNRESOLVED. Baseline
reporting reads existing scores (all of D_fit and the whole pool) and makes no model
or E calls. Stability adds neutral prefixes and one single-attempt paraphrase of each
sampled solution.

Stability sample (paper Sec. 5, 2026-10-13): D_fit only, 150 E-INCORRECT and 150
E-CORRECT, as detox. The prefixes follow Core; the paraphrase instruction is
the authors' (paper Sec. 5), its file pinned by sha256 for live runs. The prefixed and paraphrased
texts are scored in batches (``MathPipeline._score_items``). With ``score=False`` the
run stops before scoring and returns those texts: the laptop prepares them (the
controller's paraphrases, E on them), the pod scores them, and once their scores are in
the run (``pipelines/math_step06.py import``) the full run needs no model.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import replace

import numpy as np
from pydantic import ValidationError

from rafr.contracts.schemas import InvalidReason, MathLabel, SafetyLabel, Split
from rafr.core.seeding import attempt_seed, component_seed
from rafr.data.labels import LabelPurpose
from rafr.domains.math import generation
from rafr.domains.math.generation import TRUNCATION_REASONS, MathGenerationOutput
from rafr.domains.math.storage import IntegrityError, atomic_json, digest, file_hash, read_json
from rafr.llm.client import fill_slots
from rafr.logging.writer import JsonlWriter, read_jsonl
from rafr.validity.tau import fit_tau
from rafr.verifiers import diagnostics as core_diagnostics
from rafr.verifiers import perturb

PROMPT_DIR = generation.PROMPT_DIR / "perturb"
N_PER_CLASS = 150
POPULATIONS = (Split.D_FIT, Split.UNLABELED_POOL)
# paper Sec. 5 (2026-10-13): the stability sample is drawn from D_fit alone.
SAMPLE_POPULATION = Split.D_FIT
# LF-normalized sha256 of the two perturbation files (paper Sec. 5: the authors' paraphrase
# instruction, 2026-10-12; the three neutral prefixes of Core). Live runs refuse any other text.
PINNED_PROMPTS = {
    perturb.PARAPHRASE_FILE: "3baf73511556a4afcfe847639f45a38bdb7c90e5ea419733b39046b08aa102d0",
    perturb.PREFIX_FILE: "699d5ea567e654a0af4d5d89e7017841b744a4a5e6874e3f3b0c3edd1d1eba20",
}
_MATH_KEYS = {
    "n_unsafe": "n_incorrect",
    "fnr_unsafe": "fnr_incorrect",
    "fpr_safe": "fpr_correct",
    "disagreement_unsafe": "disagreement_incorrect",
    "both_miss_unsafe": "both_miss_incorrect",
    "flips_toxic_to_clean": "flips_flagged_to_unflagged",
    "flips_clean_to_toxic": "flips_unflagged_to_flagged",
}


def _math_keys(value):
    if isinstance(value, dict):
        return {_MATH_KEYS.get(key, key): _math_keys(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_math_keys(item) for item in value]
    return value


def diagnostic_prompt_identity(*, pinned=False):
    """Validate the mathematical slots and return hashes for the consumed files.

    ``pinned`` (every live run): each file's LF-normalized text must have its pinned sha256.
    """
    perturb.read_prefixes(PROMPT_DIR)
    template = perturb.read_paraphrase_prompt(PROMPT_DIR)
    slots = set(re.findall(r"\{([A-Za-z_][A-Za-z0-9_]*)\}", template))
    if slots != {"problem", "text"}:
        raise IntegrityError("math paraphrase prompt must contain only {problem} and {text} slots")
    if pinned:
        for name, expected in PINNED_PROMPTS.items():
            text = (PROMPT_DIR / name).read_text(encoding="utf-8")
            if perturb.text_sha256(text) != expected:
                raise IntegrityError(f"{name} differs from its pinned sha256 (M.4)")
    return {
        name: file_hash(PROMPT_DIR / name)
        for name in (perturb.PREFIX_FILE, perturb.PARAPHRASE_FILE)
    }


def _populations(pipeline):
    result = {}
    for split in POPULATIONS:
        rows = pipeline._labels(split, LabelPurpose.VERIFIER_DIAGNOSTICS)
        if any(row.label not in {MathLabel.CORRECT, MathLabel.INCORRECT} for row in rows):
            raise IntegrityError("math diagnostics require resolved mathematical initial labels")
        ids = [row.x0_id for row in rows]
        scores = pipeline._initial_pairs(ids)
        result[split.value] = {
            "rows": rows,
            "scores": np.asarray(scores, dtype=float),
            "y": np.asarray([int(row.label is MathLabel.INCORRECT) for row in rows]),
        }
    fit, pool = (result[split.value] for split in POPULATIONS)
    result["combined"] = {
        "rows": fit["rows"] + pool["rows"],
        "scores": np.concatenate((fit["scores"], pool["scores"])),
        "y": np.concatenate((fit["y"], pool["y"])),
    }
    if len({row.x0_id for row in result["combined"]["rows"]}) != len(result["combined"]["rows"]):
        raise IntegrityError("D_fit and pool must be disjoint")
    return result


def _taus(pipeline, populations):
    if (pipeline.root / "fit.json").exists():
        fit = pipeline._read_fixed(pipeline.root / "fit.json")
        return [item["tau"] for item in fit["tau"]], "frozen_D_fit_F1"
    data = populations[Split.D_FIT.value]
    if set(data["y"]) != {0, 1}:
        raise IntegrityError("provisional diagnostics tau requires both D_fit classes")
    return [
        fit_tau(data["scores"][:, i], data["y"], f"math-v{i + 1}").tau for i in range(2)
    ], "provisional_D_fit_F1"


def _baseline(pipeline, populations, taus, source):
    initial = read_json(pipeline.root / "initial_complete.json")
    if "n_incorrect" not in initial or "n_correct" not in initial:
        raise IntegrityError("initial completion record lacks pre-split correct/incorrect totals")
    total = initial["n_incorrect"] + initial["n_correct"]
    if total != initial["n_valid"] or total <= 0:
        raise IntegrityError("pre-split class totals do not match valid initial states")
    p = initial["n_incorrect"] / total
    return {
        "domain": "math",
        "protocol": "v1.1 M.4 / II.4.2",
        "stage": pipeline.config.stage,
        "execution_mode": pipeline.config.execution_mode,
        "run_id": pipeline.root.name,
        "is_research_result": False,
        "tau": taus,
        "tau_source": source,
        "positive_class": "INCORRECT",
        "p_incorrect": p,
        "p_source": "initial_complete pre-split aggregate counts",
        "labels_read": [split.value for split in POPULATIONS],
        "label_purpose": LabelPurpose.VERIFIER_DIAGNOSTICS.value,
        "verifiers": [
            {
                "model_id": pipeline._verifier_identity(i)[0],
                "revision": pipeline._verifier_identity(i)[1],
            }
            for i in range(2)
        ],
        "single": {
            name: {
                f"v{i + 1}": _math_keys(
                    core_diagnostics.single(data["scores"][:, i], data["y"], taus[i], p)
                )
                for i in range(2)
            }
            for name, data in populations.items()
        },
        "pair": {
            name: _math_keys(
                core_diagnostics.pair(
                    data["scores"][:, 0], taus[0], data["scores"][:, 1], taus[1], data["y"]
                )
            )
            for name, data in populations.items()
        },
        "metric_implementation": "rafr.verifiers.diagnostics.single/pair",
        "calibration_note": (
            "ECE uses Core's 15 bins; numeric ECE/Brier do not establish calibrated probabilities."
        ),
        "stability_status": "NOT_RUN",
    }


def _sample(pipeline, n_per_class):
    """Core's sampler on D_fit alone (paper Sec. 5, 2026-10-13); its labels only."""
    states = pipeline._states()
    rows = pipeline._labels(SAMPLE_POPULATION, LabelPurpose.VERIFIER_DIAGNOSTICS)
    if any(row.label not in {MathLabel.CORRECT, MathLabel.INCORRECT} for row in rows):
        raise IntegrityError("math diagnostics require resolved mathematical initial labels")
    selected = perturb.draw_sample(
        [states[row.x0_id] for row in rows],
        {
            row.state_id: SafetyLabel.UNSAFE
            if row.label is MathLabel.INCORRECT
            else SafetyLabel.SAFE
            for row in rows
        },
        {row.state_id: row.split for row in rows},
        component_seed(pipeline.config.seed, perturb.SAMPLE_SEED_COMPONENT),
        n_per_class=n_per_class,
    )
    # The Core sampler needs binary enum identities. This conversion is an adapter
    # boundary only; Core's toxicity label names never enter math artifacts.
    return [
        replace(
            row,
            e_label=MathLabel.INCORRECT.value
            if row.e_label == SafetyLabel.UNSAFE.value
            else MathLabel.CORRECT.value,
        )
        for row in selected
    ]


def _request(pipeline, row, template):
    question = pipeline.questions[row.x0_id]
    base = generation.build_generation_request(
        question.public_payload(), master_seed=pipeline.config.seed, attempt=0
    )
    return replace(
        base,
        purpose="math_verifier_paraphrase",
        messages=[
            {
                "role": "user",
                "content": fill_slots(template, {"problem": question.problem, "text": row.text}),
            }
        ],
        seed=attempt_seed(
            component_seed(pipeline.config.seed, perturb.PARAPHRASE_SEED_COMPONENT),
            row.state_id,
            attempt=0,
        ),
    )


def _paraphrase(pipeline, row, template):
    request = _request(pipeline, row, template)
    response = pipeline.client.complete(request)
    reason, solution = None, None
    if response.refused:
        reason = InvalidReason.REFUSAL
    elif response.stop_reason in TRUNCATION_REASONS:
        reason = InvalidReason.TRUNCATED
    elif not response.text or not response.text.strip():
        reason = InvalidReason.EMPTY
    elif response.error:
        reason = InvalidReason.STRUCTURED_OUTPUT
    else:
        try:
            solution = MathGenerationOutput.model_validate_json(response.text).solution
        except (ValidationError, ValueError):
            reason = InvalidReason.STRUCTURED_OUTPUT
        else:
            if not solution.strip():
                reason = InvalidReason.EMPTY
    # No identical/non-English check and no extra model retries. Final validity
    # is completed after E: The design's UNRESOLVED is an invalid wording perturbation.
    return {
        "state_id": row.state_id,
        "question_id": row.x0_id,
        "valid": reason is None,
        "invalid_reason": reason.value if reason else None,
        "paraphrase": solution if reason is None else None,
        "raw_output": response.text,
        "provider_refused": response.refused,
        "stop_reason": response.stop_reason,
        "model_version": response.model_version,
        "prompt_hash": request.prompt_hash(),
        "seed": request.seed,
        "attempt": 0,
        "cached": response.cached,
        "request": {
            "model": request.model,
            "version": request.version,
            "system": request.system,
            "messages": request.messages,
            "decoding": request.decoding,
            "schema_name": request.schema_name,
        },
    }


def _judge_paraphrase(pipeline, row, text):
    result = None
    for attempt in range(3):
        result = pipeline.grade_once(row.x0_id, text)
        if result.status in {"correct", "incorrect", "unresolved"}:
            return {
                "state_id": row.state_id,
                "question_id": row.x0_id,
                "label": MathLabel(result.status.upper()).value,
                "status": result.status,
                "e_error": None,
                "e_attempts": attempt + 1,
                "candidate_sha256": digest(text),
            }
        if result.status not in {"timeout", "backend_error"}:
            raise IntegrityError("unknown mathematical E status in verifier stability")
    return {
        "state_id": row.state_id,
        "question_id": row.x0_id,
        "label": None,
        "status": result.status,
        "e_error": result.detail,
        "e_attempts": 3,
        "candidate_sha256": digest(text),
    }


def _unique_rows(path):
    result = {}
    for row in read_jsonl(path):
        key = row["state_id"]
        if key in result:
            raise IntegrityError(f"duplicate state in diagnostic stream {path.name}")
        result[key] = row
    return result


def _stability_key(row, kind, text):
    """The score key of a perturbed text (also its pod input's state_id)."""
    return digest(["math_stability", row.state_id, kind, text])


def run_diagnostics(pipeline, *, stability=False, n_per_class=N_PER_CLASS, score=True):
    """Write baseline metrics; optionally run the planned stability sample.

    Smaller samples exist only for explicitly offline fixtures. No CLI may select
    an alternate sample size for a live run. Stability is descriptive, not a
    behavioral gate and never changes tau, q, channel or experiment cohorts.

    ``score=False`` (stability only): sample, paraphrase and judge, then return
    ``{"items": [(question_id, key, text), ...]}`` -- the texts V scores -- instead of
    scoring them; nothing about tau or the baseline is read.
    """
    from pipelines.math_pipeline import finite_json

    if type(n_per_class) is not int or n_per_class < 1:
        raise ValueError("n_per_class must be a positive integer")
    live = not (
        pipeline.config.stage == "stub" and pipeline.config.execution_mode == "offline_stub"
    )
    if n_per_class != N_PER_CLASS and live:
        raise IntegrityError("live stability requires 150 CORRECT and 150 INCORRECT states")
    out = pipeline.root / "verifier_diagnostics"
    if not stability:
        populations = _populations(pipeline)
        taus, tau_source = _taus(pipeline, populations)
        report = _baseline(pipeline, populations, taus, tau_source)
        out.mkdir(exist_ok=True)
        atomic_json(out / "baseline_report.json", finite_json(report))
        return finite_json(report)

    prompts = diagnostic_prompt_identity(pinned=live)
    sample = _sample(pipeline, n_per_class)
    out.mkdir(exist_ok=True)
    pipeline._save_fixed(out / "sample.json", [row.as_dict() for row in sample])
    pipeline._save_fixed(out / "prompts.json", prompts)
    pipeline._save_fixed(
        out / "identity.json",
        {
            "seed": pipeline.config.seed,
            "sample_population": SAMPLE_POPULATION.value,
            "models": generation.frozen_models()["controller"],
            "verifiers": pipeline.config.verifiers,
            "n_per_class": n_per_class,
            "math_verify_version": pipeline.config.math_verify_version,
            "execution_mode": pipeline.config.execution_mode,
        },
    )
    complete_path = out / "COMPLETE.json"
    if complete_path.exists() and score:
        completed = pipeline._read_fixed(complete_path)
        for name, checksum in completed["files"].items():
            if file_hash(out / name) != checksum:
                raise IntegrityError("completed verifier diagnostic files changed")
        return pipeline._read_fixed(out / "report.json")

    paraphrase_path = out / "paraphrases.jsonl"
    judgment_path = out / "paraphrase_judgments.jsonl"
    attempt_path = out / "generation_attempts.jsonl"
    paraphrases = _unique_rows(paraphrase_path)
    judgments = _unique_rows(judgment_path)
    attempts = _unique_rows(attempt_path)
    ids = {row.state_id for row in sample}
    if set(paraphrases) - ids or set(judgments) - ids or set(attempts) - ids:
        raise IntegrityError("perturbation records are outside the frozen sample")
    prefixes = perturb.read_prefixes(PROMPT_DIR)
    template = perturb.read_paraphrase_prompt(PROMPT_DIR)
    with (
        JsonlWriter(attempt_path) as attempt_writer,
        JsonlWriter(paraphrase_path) as para_writer,
        JsonlWriter(judgment_path) as e_writer,
    ):
        for row in sample:
            if row.state_id not in attempts:
                attempt = _paraphrase(pipeline, row, template)
                attempt_writer.write(attempt)
                attempts[row.state_id] = attempt
            attempt = attempts[row.state_id]
            request = _request(pipeline, row, template)
            if (
                attempt["question_id"] != row.x0_id
                or attempt["prompt_hash"] != request.prompt_hash()
                or attempt["seed"] != request.seed
            ):
                raise IntegrityError("saved diagnostic attempt differs from its frozen request")
            finalized = dict(attempt)
            if attempt["valid"]:
                text = attempt["paraphrase"]
                if row.state_id not in judgments:
                    judgment = _judge_paraphrase(pipeline, row, text)
                    e_writer.write(judgment)
                    judgments[row.state_id] = judgment
                judgment = judgments[row.state_id]
                if judgment["question_id"] != row.x0_id or judgment["candidate_sha256"] != digest(
                    text
                ):
                    raise IntegrityError("saved paraphrase judgment belongs to another text")
                if judgment["status"] == "unresolved":
                    finalized["valid"] = False
                    finalized["invalid_reason"] = InvalidReason.UNRESOLVED.value
            elif row.state_id in judgments:
                raise IntegrityError("invalid JSON paraphrase must not have an E judgment")
            if row.state_id not in paraphrases:
                para_writer.write(finalized)
                paraphrases[row.state_id] = finalized
            elif paraphrases[row.state_id] != finalized:
                raise IntegrityError("final paraphrase validity disagrees with its saved attempt/E")
    # The texts V scores: every sampled state with its prefix, then every valid paraphrase
    # (an invalid one, UNRESOLVED included, is dropped before any V scoring).
    texts = {"prefix": {}, "paraphrase": {}}
    for row in sample:
        texts["prefix"][row.state_id] = perturb.prefixed(row, prefixes)
    for row in sample:
        if paraphrases[row.state_id]["valid"]:
            texts["paraphrase"][row.state_id] = paraphrases[row.state_id]["paraphrase"]
    by_id = {row.state_id: row for row in sample}
    items = [
        (by_id[state_id].x0_id, _stability_key(by_id[state_id], kind, text), text)
        for kind in perturb.KINDS
        for state_id, text in texts[kind].items()
    ]
    if not score:
        return {"items": items, "n_sample": len(sample)}
    scored = pipeline._score_items(items)
    changed = {
        kind: {
            f"v{i + 1}": {
                state_id: scored[_stability_key(by_id[state_id], kind, text)][i].score
                for state_id, text in texts[kind].items()
            }
            for i in range(2)
        }
        for kind in perturb.KINDS
    }
    populations = _populations(pipeline)
    taus, tau_source = _taus(pipeline, populations)
    report = _baseline(pipeline, populations, taus, tau_source)
    original = {
        f"v{i + 1}": {
            row.state_id: float(score_pair[i])
            for row, score_pair in zip(
                populations["combined"]["rows"], populations["combined"]["scores"], strict=True
            )
        }
        for i in range(2)
    }
    stability_report = _math_keys(
        core_diagnostics.stability(
            [row.as_dict() for row in sample],
            original,
            changed,
            {f"v{i + 1}": taus[i] for i in range(2)},
            paraphrases,
            judgments,
        )
    )
    stability_report["prefix"]["e_unchanged_note"] = (
        "N/A: fixed neutral prefixes do not alter the solution; prefixed texts are not sent to E."
    )
    stability_report["paraphrase"]["e_status_counts"] = dict(
        Counter(row["status"] for row in judgments.values())
    )
    stability_report["paraphrase"]["e_status_counts_scope"] = (
        "All schema-valid single-attempt paraphrases, including unresolved items "
        "dropped before V scoring."
    )
    report.update(
        stability_status="COMPLETE",
        stability=stability_report,
        stability_metric_implementation="rafr.verifiers.diagnostics.stability",
        sample={
            "n": len(sample),
            "n_per_class": n_per_class,
            "population": SAMPLE_POPULATION.value,
            "sha256": file_hash(out / "sample.json"),
        },
        prompts=prompts,
        thresholds_changed=False,
    )
    report = finite_json(report)
    pipeline._save_fixed(out / "report.json", report)
    pipeline._save_fixed(
        complete_path,
        {
            "report_sha256": file_hash(out / "report.json"),
            "taus": taus,
            "execution_mode": pipeline.config.execution_mode,
            "n_per_class": n_per_class,
            "n_sampled": len(sample),
            "files": {
                name: file_hash(out / name)
                for name in (
                    "sample.json",
                    "prompts.json",
                    "identity.json",
                    "generation_attempts.jsonl",
                    "paraphrases.jsonl",
                    "paraphrase_judgments.jsonl",
                    "report.json",
                )
            },
        },
    )
    return report
