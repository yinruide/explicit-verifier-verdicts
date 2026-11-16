"""Math step 6 on the laptop (docs/math/step06_handoff.md; runbook docs/math/runbook_step06_pod.md).

    python pipelines/math_step06.py inputs                 # D_gate, D_Exp1, rest of the pool
    python pipelines/math_step06.py tau                    # session tau, reused D_fit scores
    python pipelines/math_step06.py x1 --smoke             # the smoke's 30 pool states
    python pipelines/math_step06.py stability-prepare --confirm-live   # paraphrases + E
    python pipelines/math_step06.py lengths                # every pod input against L_in
    python pipelines/math_step06.py check                  # ready for `up`
    python pipelines/math_step06.py import                 # after the pod: tables -> scores
    python pipelines/math_step06.py stability-finish       # II.4.2 report (no model)
    python pipelines/math_step06.py loop-install --run-id R --smoke   # pulled loop -> run
    python pipelines/math_step06.py smoke-report           # E on every round, smoke report
    python pipelines/math_step06.py cost                   # the session's time and cost

The pod gets public text, state ids and the session tau only; every label read here stays on
the laptop, with the purpose Core allows for it:

* ``inputs``: no label. D_gate and D_Exp1 (split order) and the pool states step 2 did not score
  (pool order) -- the II.10 math entry (2026-10-12): the whole pool is scored, so q_i and
  the II.4.2 baseline need no label. D_gate_reserve is never an input. ``state_id``, question
  id, split, problem, solution; sha256 next to each file.
* ``tau``: II.4.1 (Core ``fit_tau``, F1, plateau midpoint) on all 1,500 D_fit states with the
  scores of step 2 (screening 300 + full stage 1,200), y* read for ``tau_fit`` exactly as step 7
  (``MathPipeline.fit``) reads it; step 7 freezes the same rule on the same data (M.10, the authors
  2026-10-12). The pod uses it for the smoke and for the 203-state check's alternative criterion
  (no flip at tau).
* ``x1 --smoke``: ``MathPipeline._initial_cohort(smoke=True)`` (30 pool E-INCORRECT states,
  ``dry_run``), in that order; ``x1`` without ``--smoke`` is the formal Exp 2's D_Exp1.
* ``stability-prepare``: the II.4.2 stability sample (D_fit, 150 / 150 by E label; the authors
  2026-10-13), one paraphrase each by the controller's model and E on it, as
  ``math_verifier_diagnostics`` does it; then the texts the pod scores (each sampled state with
  its prefix, each valid paraphrase), keyed by the diagnostics' own digest ids.
* ``import``: the step 2 tables (D_fit and the pool sample) and the step 6 tables (D_gate,
  D_Exp1, rest of the pool), row by row into the run's ``scores.jsonl``: every table and detail
  file against its recorded sha256, each row's verifier_id / model_version, the detail rows'
  scores and one config hash per V, the 203-state check of this session passed, and coverage =
  D_fit + pool + D_gate + D_Exp1 exactly. The stability texts are a second part. The sources
  are recorded in ``score_imports/<part>.json`` (a frozen artifact of the run).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections.abc import Sequence
from contextlib import contextmanager
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pipelines import math_verifier_diagnostics as md  # noqa: E402
from pipelines.math_pipeline import finite_json  # noqa: E402
from rafr.contracts.schemas import MathLabel, Score, Split  # noqa: E402
from rafr.core import paths  # noqa: E402
from rafr.data.labels import LabelPurpose  # noqa: E402
from rafr.domains.math.storage import (  # noqa: E402
    IntegrityError,
    exclusive_lock,
    file_hash,
    read_json,
)
from rafr.logging.writer import JsonlWriter, read_jsonl  # noqa: E402
from rafr.validity.tau import fit_tau  # noqa: E402

RUN = paths.RUNS / "math_formal_n9000"
FORMAL_CONFIG = ROOT / "configs" / "math" / "formal.json"
ROUND2_CONFIG = ROOT / "configs" / "math" / "step02_round2.json"
STEP06 = paths.DATA / "math" / "step06"
VPROBE2 = paths.DATA / "math" / "vprobe2"
FORMAL_POD = paths.DATA / "math" / "formal_pod"
#: Pod stages of step 6: the scored splits and the II.4.2 stability texts.
SPLIT_STAGES = ("d_gate", "d_exp1", "pool_rest")
STAGES = (*SPLIT_STAGES, "stability")
POOL = Split.UNLABELED_POOL.value
#: II.10 math alternative criterion (2026-10-12): >= 95% of the 203 bitwise identical.
ALT_IDENTICAL_SHARE = 0.95
#: Measured step 2 throughput on the formal pod (states per hour, full stage; plan 10.13).
THROUGHPUT = {"glmz1_9b": 2424 / 4.3, "r1_llama_8b": 2424 / 7.4}
CHECK203_MIN = {"glmz1_9b": 26.0, "r1_llama_8b": 41.0}


class Step06Error(RuntimeError):
    pass


def utc_now() -> str:
    return datetime.now(tz=UTC).isoformat(timespec="seconds")


def sha256_file(path: Path) -> str:
    return file_hash(path)


def write_rows(path: Path, rows: Sequence[dict[str, Any]]) -> str:
    """JSONL (LF) and a ``<name>.sha256`` beside it; returns the sha256."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(path.suffix + ".partial")
    partial.unlink(missing_ok=True)
    with JsonlWriter(partial) as writer:
        writer.write_many(rows)
    partial.replace(path)
    digest = sha256_file(path)
    path.with_name(path.name + ".sha256").write_text(
        f"{digest}  {path.name}\n", encoding="utf-8", newline="\n"
    )
    return digest


def recorded_sha(path: Path) -> str | None:
    sha = Path(path).with_name(Path(path).name + ".sha256")
    return sha.read_text(encoding="utf-8").split()[0] if sha.is_file() else None


def read_checked(path: Path) -> list[dict[str, Any]]:
    """Rows of a file whose sha256 sidecar still matches."""
    want = recorded_sha(path)
    if want is None or want != sha256_file(path):
        raise Step06Error(f"{path}: missing, or changed since its sha256 was written")
    return list(read_jsonl(path))


def write_json(path: Path, value: Any) -> str:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(finite_json(value), indent=1, ensure_ascii=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return sha256_file(path)


def pod_records(value: str | None = None) -> Path:
    """Where the laptop keeps a pod session's records (pod.json, session.json, the pulled
    ``out/`` that ``sync`` replaces whole, and the loops ``follow`` pulls). Step 6's session is
    ``data/math/formal_pod`` (its pulled tables are what the run's score imports point at);
    every later session -- the R1 loop diagnosis, the smoke and Exp 2 -- sets
    ``RAFR_FORMAL_POD_DIR`` so that it never touches them (2026-10-14 / 10-16)."""
    value = os.environ.get("RAFR_FORMAL_POD_DIR") if value is None else value
    if not value:
        return FORMAL_POD
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


class Paths:
    """Where step 6 keeps its laptop files; tests point it at a temporary tree."""

    def __init__(
        self,
        *,
        run: Path = RUN,
        step06: Path = STEP06,
        vprobe2: Path = VPROBE2,
        pod: Path = FORMAL_POD,
        formal_config: Path = FORMAL_CONFIG,
        round2_config: Path = ROUND2_CONFIG,
    ) -> None:
        self.run, self.step06, self.vprobe2, self.pod = (
            Path(run),
            Path(step06),
            Path(vprobe2),
            Path(pod),
        )
        self.formal_config, self.round2_config = Path(formal_config), Path(round2_config)

    def stage_inputs(self, stage: str) -> Path:
        if stage not in STAGES:
            raise Step06Error(f"unknown step 6 stage {stage!r}")
        return self.step06 / f"{stage}_inputs.jsonl"

    @property
    def tau(self) -> Path:
        return self.step06 / "tau.json"

    @property
    def lengths(self) -> Path:
        return self.step06 / "lengths.json"

    def x1(self, smoke: bool) -> Path:
        return self.step06 / ("x1_smoke.jsonl" if smoke else "x1_formal.jsonl")

    # step 2's records (data/math/vprobe2)
    @property
    def screen_inputs(self) -> Path:
        return self.vprobe2 / "screen_inputs.jsonl"

    @property
    def full_inputs(self) -> Path:
        return self.vprobe2 / "full_inputs.jsonl"

    @property
    def check203_ids(self) -> Path:
        return self.vprobe2 / "check203.json"

    @property
    def step2_session(self) -> Path:
        return self.vprobe2 / "session.json"

    def step2(self, stage: str, kind: str, cid: str) -> Path:
        base = self.vprobe2 / "remote" / stage
        if kind == "meta":
            return base / "meta" / f"{cid}.score.json"
        return base / kind / f"{cid}.jsonl"

    def tokenizer(self, cid: str) -> Path:
        return self.vprobe2 / "tokenizers" / cid

    # this session's pod records (data/math/formal_pod; ``sync`` pulls the pod's out/ here)
    @property
    def session(self) -> Path:
        return self.pod / "session.json"

    @property
    def pulled(self) -> Path:
        return self.pod / "remote"

    def step6(self, stage: str, kind: str, cid: str) -> Path:
        base = self.pulled / stage
        if kind == "meta":
            return base / "meta" / f"{cid}.score.json"
        return base / kind / f"{cid}.jsonl"

    def check203(self, cid: str) -> Path:
        return self.pulled / "check203" / f"{cid}.json"

    def loop_pulled(self, run_id: str, smoke: bool) -> Path:
        return self.pod / ("exp2_smoke" if smoke else "exp2") / run_id


# ---------------------------------------------------------------------------------------
# the two V: formal.json must hold step 2's specs unchanged
# ---------------------------------------------------------------------------------------


def selected_verifiers(paths: Paths) -> list[dict[str, Any]]:
    """V1, V2 as step 2 selected them (``stop_point_b``), with their round-2 entries; each
    ``formal.json`` spec must equal the round-2 spec exactly (handoff 5.1: nothing changed)."""
    formal = json.loads(paths.formal_config.read_text(encoding="utf-8"))
    round2 = json.loads(paths.round2_config.read_text(encoding="utf-8"))
    by_id = {c["candidate_id"]: c for c in round2["candidates"]}
    chosen = [round2["stop_point_b"]["v1"], round2["stop_point_b"]["v2"]]
    if len(formal.get("verifiers") or []) != 2:
        raise Step06Error("configs/math/formal.json must hold the two selected verifiers")
    out = []
    for cid, spec in zip(chosen, formal["verifiers"], strict=True):
        if by_id[cid]["spec"] != spec:
            raise Step06Error(
                f"formal.json's verifier for {cid} differs from configs/math/step02_round2.json"
            )
        out.append(by_id[cid])
    return out


# ---------------------------------------------------------------------------------------
# the formal run, opened without any model
# ---------------------------------------------------------------------------------------


def open_pipeline(paths: Paths, *, openai: bool = False, confirm_live: bool = False):
    """The formal run's MathPipeline with no verifier loaded; ``openai`` adds the controller's
    provider (the paraphrases of the stability sample), which needs ``--confirm-live``."""
    from pipelines.math_runtime import build_pipeline
    from rafr.domains.math.config import MathRunConfig

    config = MathRunConfig.load(paths.formal_config)
    pipeline = build_pipeline(config, paths.run, command="diagnostics")
    if openai:
        from rafr.domains.math.generation import frozen_models
        from rafr.llm.cache import LLMCache
        from rafr.llm.client import LLMClient
        from rafr.llm.ledger import Ledger
        from rafr.llm.providers.openai import OpenAIProvider

        config.assert_live_allowed(confirm_live=confirm_live)
        network = frozen_models()["controller"].get("network", {})
        root = pipeline.root
        pipeline.client = LLMClient(
            providers={
                "openai": OpenAIProvider(
                    max_retries=int(network.get("max_retries", 0)),
                    timeout_s=float(network.get("timeout_s", 120)),
                )
            },
            cache=LLMCache(root / "cache"),
            ledger=Ledger(root / "ledger.jsonl"),
            transient_log=root / "transient_retries.jsonl",
        )
    return pipeline


@contextmanager
def run_lock(root: Path):
    """The math CLI's writer mutex for the run (pipelines/math_run.py)."""
    root = Path(root).resolve()
    with exclusive_lock(root.parent / f".{root.name}.writer.lock"):
        yield


# ---------------------------------------------------------------------------------------
# inputs for the pod (public text only)
# ---------------------------------------------------------------------------------------


def state_rows(pipeline, ids: Sequence[str], split: Split) -> list[dict[str, Any]]:
    states = pipeline._states()  # noqa: SLF001
    rows = []
    for qid in ids:
        state = states[qid]
        problem = pipeline.questions[qid].problem
        if state.x0 != problem:
            raise Step06Error(f"{qid}: the state's x0 differs from the bundle problem")
        rows.append(
            {
                "state_id": state.state_id,
                "x0_id": qid,
                "split": split.value,
                "problem": problem,
                "solution": state.text,
            }
        )
    return rows


def step2_state_ids(paths: Paths) -> tuple[list[str], list[str]]:
    """The states step 2 scored: (D_fit, pool sample), from its two input files."""
    screen = read_checked(paths.screen_inputs)
    full = read_checked(paths.full_inputs)
    fit = [r["state_id"] for r in screen] + [
        r["state_id"] for r in full if r["split"] == Split.D_FIT.value
    ]
    pool = [r["state_id"] for r in full if r["split"] == POOL]
    if len(set(fit)) != len(fit) or len(set(pool)) != len(pool):
        raise Step06Error("step 2's inputs hold a state twice")
    return fit, pool


def build_inputs(pipeline, paths: Paths) -> dict[str, Any]:
    """D_gate and D_Exp1 whole, and the pool states step 2 did not score; no label read."""
    split_ids = {split: pipeline._split_ids(split) for split in Split}  # noqa: SLF001
    states = pipeline._states()  # noqa: SLF001
    fit_done, pool_done = step2_state_ids(paths)
    fit_states = {states[q].state_id for q in split_ids[Split.D_FIT]}
    if set(fit_done) != fit_states:
        raise Step06Error("step 2's tables do not cover D_fit exactly")
    pool_states = {states[q].state_id: q for q in split_ids[Split.UNLABELED_POOL]}
    if not set(pool_done) <= set(pool_states):
        raise Step06Error("step 2's pool sample is not inside the pool")
    rest = [q for q in split_ids[Split.UNLABELED_POOL] if states[q].state_id not in set(pool_done)]
    plan = {
        "d_gate": state_rows(pipeline, split_ids[Split.D_GATE], Split.D_GATE),
        "d_exp1": state_rows(pipeline, split_ids[Split.D_EXP1], Split.D_EXP1),
        "pool_rest": state_rows(pipeline, rest, Split.UNLABELED_POOL),
    }
    reserve = {states[q].state_id for q in split_ids[Split.D_GATE_RESERVE]}
    out: dict[str, Any] = {}
    for stage, rows in plan.items():
        if reserve & {r["state_id"] for r in rows}:
            raise Step06Error(f"{stage}: D_gate_reserve is never scored (II.6.1)")
        digest = write_rows(paths.stage_inputs(stage), rows)
        out[stage] = {"n": len(rows), "sha256": digest}
    out["pool"] = {
        "n": len(split_ids[Split.UNLABELED_POOL]),
        "scored_in_step2": len(pool_done),
        "rest": len(rest),
    }
    return out


# ---------------------------------------------------------------------------------------
# step 2's tables
# ---------------------------------------------------------------------------------------


def step2_expected(paths: Paths, cid: str) -> dict[str, dict[str, str]]:
    """sha256 of step 2's screening and full-stage table and detail file of ``cid``: the score
    record written on the pod, the laptop's sync record (and for the full stage, the config's
    ``stop_point_b``) must agree."""
    session = json.loads(paths.step2_session.read_text(encoding="utf-8"))
    round2 = json.loads(paths.round2_config.read_text(encoding="utf-8"))
    out = {}
    for stage in ("screen", "full"):
        meta = json.loads(paths.step2(stage, "meta", cid).read_text(encoding="utf-8"))
        synced = session.get("tables", {}).get(f"{stage}/{cid}", {})
        if not synced.get("ok") or synced.get("sha256") != meta["table_sha256"]:
            raise Step06Error(f"step 2 {stage}/{cid}: the sync record and the score record differ")
        if stage == "full":
            pinned = round2["stop_point_b"]["full_stage_result"]["tables_sha256"].get(cid)
            if pinned != meta["table_sha256"]:
                raise Step06Error(f"step 2 full/{cid}: table sha256 differs from stop_point_b")
        out[stage] = {
            "table": meta["table_sha256"],
            "details": meta["details_sha256"],
            "config_hash": meta["config_hash"],
        }
    return out


def read_table(
    table: Path,
    details: Path,
    *,
    model_id: str,
    revision: str,
    table_sha: str,
    details_sha: str,
) -> tuple[list[Score], str, list[dict[str, Any]]]:
    """Score rows of one pod table, checked against its detail rows; returns the rows, the one
    config hash of their detail rows and a short summary per row (stop reason, truncation)."""
    if sha256_file(table) != table_sha:
        raise Step06Error(f"{table}: sha256 differs from the recorded {table_sha[:16]}")
    if sha256_file(details) != details_sha:
        raise Step06Error(f"{details}: sha256 differs from the recorded {details_sha[:16]}")
    rows = [Score.model_validate(r) for r in read_jsonl(table)]
    summary, hashes = [], set()
    detail_iter = read_jsonl(details)
    for row in rows:
        if row.verifier_id != model_id or row.model_version != revision:
            raise Step06Error(f"{table}: {row.state_id} is not {model_id}@{revision}")
        detail = next(detail_iter, None)
        if detail is None or detail["state_id"] != row.state_id:
            raise Step06Error(f"{details}: detail rows do not follow the score rows")
        if detail["error_score"] != row.score or detail.get("model_id") != model_id:
            raise Step06Error(f"{details}: {row.state_id}'s detail disagrees with its score row")
        hashes.add(detail.get("config_hash"))
        summary.append(
            {"stop_reason": detail.get("stop_reason"), "truncated": detail.get("truncated")}
        )
    if next(detail_iter, None) is not None:
        raise Step06Error(f"{details}: more detail rows than score rows")
    if len(hashes) != 1 or None in hashes:
        raise Step06Error(f"{details}: the rows do not share one config hash")
    return rows, hashes.pop(), summary


def step2_scores(pipeline, paths: Paths, selected: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """state_id -> score per V from step 2's screening and full-stage tables (checked)."""
    result: dict[str, Any] = {"scores": [], "sources": [], "config_hash": []}
    for i, candidate in enumerate(selected):
        cid = candidate["candidate_id"]
        model_id, revision = pipeline._verifier_identity(i)  # noqa: SLF001
        expected = step2_expected(paths, cid)
        merged: dict[str, Score] = {}
        hashes = set()
        for stage in ("screen", "full"):
            rows, config_hash, _ = read_table(
                paths.step2(stage, "scores", cid),
                paths.step2(stage, "details", cid),
                model_id=model_id,
                revision=revision,
                table_sha=expected[stage]["table"],
                details_sha=expected[stage]["details"],
            )
            hashes.add(config_hash)
            for row in rows:
                if row.state_id in merged:
                    raise Step06Error(f"{cid}: {row.state_id} scored twice in step 2")
                merged[row.state_id] = row
            result["sources"].append(
                {
                    "candidate_id": cid,
                    "step": 2,
                    "stage": stage,
                    "table": str(paths.step2(stage, "scores", cid)),
                    "table_sha256": expected[stage]["table"],
                    "details": str(paths.step2(stage, "details", cid)),
                    "details_sha256": expected[stage]["details"],
                    "config_hash": config_hash,
                    "n": len(rows),
                }
            )
        if len(hashes) != 1:
            raise Step06Error(f"{cid}: step 2's two stages were scored in different configurations")
        result["scores"].append(merged)
        result["config_hash"].append(hashes.pop())
    return result


# ---------------------------------------------------------------------------------------
# tau at the session start (M.10: from the reused D_fit scores, II.4.1, as step 7)
# ---------------------------------------------------------------------------------------


def session_tau(pipeline, paths: Paths, selected: Sequence[dict[str, Any]]) -> dict[str, Any]:
    ys = pipeline._labels(Split.D_FIT, LabelPurpose.TAU_FIT)  # noqa: SLF001
    states = pipeline._states()  # noqa: SLF001
    errors = [int(row.label is MathLabel.INCORRECT) for row in ys]
    if set(errors) != {0, 1}:
        raise Step06Error("D_fit needs both correct and incorrect states")
    loaded = step2_scores(pipeline, paths, selected)
    fits = []
    for i, table in enumerate(loaded["scores"]):
        try:
            scores = [table[states[row.x0_id].state_id].score for row in ys]
        except KeyError as exc:
            raise Step06Error(f"step 2 has no score of {exc} for V{i + 1}") from exc
        # the same call, data and order as MathPipeline.fit (step 7)
        fits.append(fit_tau(scores, errors, f"math-v{i + 1}"))
    record = {
        "use": "M.10 (2026-10-12): the smoke's thresholds and the 203-state check's "
        "no-flip criterion, at the step 6 session start; step 7 freezes the same rule on "
        "the same data (MathPipeline.fit)",
        "rule": "II.4.1: F1 maximum on D_fit, midpoint of the widest plateau "
        "(rafr.validity.tau.fit_tau), positive class E-INCORRECT",
        "label_purpose": LabelPurpose.TAU_FIT.value,
        "n": len(errors),
        "n_incorrect": sum(errors),
        "candidates": [c["candidate_id"] for c in selected],
        "verifiers": [
            {
                "model_id": pipeline._verifier_identity(i)[0],
                "revision": pipeline._verifier_identity(i)[1],
            }  # noqa: SLF001
            for i in range(2)
        ],
        "taus": [fit.tau for fit in fits],
        "fits": [asdict(fit) for fit in fits],
        "scores_from": loaded["sources"],
        "config_hash": loaded["config_hash"],
        "computed_at": utc_now(),
    }
    digest = write_json(paths.tau, record)
    paths.tau.with_name(paths.tau.name + ".sha256").write_text(
        f"{digest}  {paths.tau.name}\n", encoding="utf-8", newline="\n"
    )
    return record


def read_tau(paths: Paths) -> dict[str, Any]:
    want = recorded_sha(paths.tau)
    if want is None or want != sha256_file(paths.tau):
        raise Step06Error(f"{paths.tau}: missing or changed; run `math_step06.py tau`")
    return json.loads(paths.tau.read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------------------
# Exp 2's first states (the smoke's 30, or D_Exp1) for the pod
# ---------------------------------------------------------------------------------------


def make_x1(pipeline, paths: Paths, *, smoke: bool) -> dict[str, Any]:
    """The cohort ``MathPipeline.exp2`` would start from, in its order (the arm assignment
    depends on it); public fields only."""
    states = pipeline._initial_cohort(  # noqa: SLF001
        smoke=smoke,
        purpose=LabelPurpose.DRY_RUN if smoke else LabelPurpose.EXP2_T1,
    )
    rows = [{"state_id": s.state_id, "x0_id": s.x0_id, "x0": s.x0, "text": s.text} for s in states]
    for row in rows:
        if row["x0"] != pipeline.questions[row["x0_id"]].problem:
            raise Step06Error(f"{row['x0_id']}: the state's x0 differs from the bundle problem")
    digest = write_rows(paths.x1(smoke), rows)
    return {"x1": str(paths.x1(smoke)), "n": len(rows), "sha256": digest, "SMOKE": smoke}


# ---------------------------------------------------------------------------------------
# the stability texts (II.4.2; M.4, 2026-10-12 / 10-13)
# ---------------------------------------------------------------------------------------


def stability_prepare(pipeline, paths: Paths, *, n_per_class: int = md.N_PER_CLASS) -> dict:
    items = md.run_diagnostics(pipeline, stability=True, score=False, n_per_class=n_per_class)[
        "items"
    ]
    rows = [
        {
            "state_id": key,
            "x0_id": qid,
            "problem": pipeline.questions[qid].problem,
            "solution": text,
        }
        for qid, key, text in items
    ]
    digest = write_rows(paths.stage_inputs("stability"), rows)
    return {"n": len(rows), "sha256": digest}


# ---------------------------------------------------------------------------------------
# input lengths against L_in, on the laptop (tokenizer and template; no weights)
# ---------------------------------------------------------------------------------------


class _NoModel:
    """Enough of a model for building inputs: the dtype and context the adapter checks."""

    def __init__(self, dtype: str) -> None:
        self.dtype = dtype
        self.config = type("C", (), {"max_position_embeddings": 10**9, "_commit_hash": None})()

    def eval(self) -> None:
        return None

    def get_input_embeddings(self) -> Any:
        return type("E", (), {"weight": type("W", (), {"device": "cpu"})()})()


def input_lengths(paths: Paths, selected: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Every pod input built by each V's adapter with its own tokenizer and template (step 2's
    local copies), counted against its L_in. An input longer than L_in cannot be scored
    (truncation is forbidden): such an input is reported and the session waits for the authors."""
    import transformers

    from rafr.domains.math.generative import GENERATIVE_ADAPTERS
    from rafr.domains.math.models import ModelInputError

    files = {stage: paths.stage_inputs(stage) for stage in STAGES}
    files["x1_smoke"] = paths.x1(True)
    files["x1_formal"] = paths.x1(False)
    texts = {}
    for name, path in files.items():
        if path.is_file():
            rows = read_checked(path)
            texts[name] = [
                (r["state_id"], r.get("problem", r.get("x0")), r.get("solution", r.get("text")))
                for r in rows
            ]
    result: dict[str, Any] = {"files": {k: len(v) for k, v in texts.items()}, "verifiers": {}}
    for candidate in selected:
        cid = candidate["candidate_id"]
        spec = candidate["spec"]
        adapter = GENERATIVE_ADAPTERS[spec["implementation"]]
        settings = {
            **spec["settings"],
            "max_input_tokens": 10**6,
            "model_id": spec["model_id"],
            "revision": spec["revision"],
        }
        config = adapter.config_class(**settings)
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            paths.tokenizer(cid), local_files_only=True, **config.tokenizer_kwargs
        )
        verifier = adapter(
            config, tokenizer=tokenizer, model=_NoModel(config.dtype), torch_module=object()
        )
        verifier.load()
        limit = spec["settings"]["max_input_tokens"]
        per_file, over, refused = {}, [], []
        for name, rows in texts.items():
            counts = []
            for state_id, problem, solution in rows:
                try:
                    n = len(verifier._row(problem, solution)[1])  # noqa: SLF001
                except ModelInputError as exc:
                    refused.append({"file": name, "state_id": state_id, "error": str(exc)})
                    continue
                counts.append(n)
                if n > limit:
                    over.append({"file": name, "state_id": state_id, "tokens": n})
            per_file[name] = {"n": len(counts), "max": max(counts, default=None)}
        result["verifiers"][cid] = {
            "L_in": limit,
            "files": per_file,
            "over_L_in": over,
            "refused": refused,
            "ok": not over and not refused,
        }
    result["ok"] = all(v["ok"] for v in result["verifiers"].values())
    result["checked_at"] = utc_now()
    write_json(paths.lengths, result)
    return result


# ---------------------------------------------------------------------------------------
# import of the pod's tables into the run's scores.jsonl
# ---------------------------------------------------------------------------------------


def session_records(paths: Paths) -> dict[str, Any]:
    return json.loads(paths.session.read_text(encoding="utf-8")) if paths.session.is_file() else {}


def check203_passed(paths: Paths, cid: str, config_hash: str) -> dict[str, Any]:
    path = paths.check203(cid)
    if not path.is_file():
        raise Step06Error(f"{cid}: no 203-state check of this session ({path})")
    record = json.loads(path.read_text(encoding="utf-8"))
    if record.get("criterion") not in {"bitwise", "alternative"} or not record.get("passed"):
        raise Step06Error(f"{cid}: the 203-state check of this session did not pass")
    if record.get("config_hash") != config_hash:
        raise Step06Error(f"{cid}: the 203-state check ran in another configuration")
    return record


def step6_sources(paths: Paths, cid: str, stages: Sequence[str]) -> list[dict[str, Any]]:
    """This session's tables of ``cid``: pulled by ``sync`` and recorded there as OK."""
    tables = session_records(paths).get("tables", {})
    out = []
    for stage in stages:
        meta_path = paths.step6(stage, "meta", cid)
        if not meta_path.is_file():
            raise Step06Error(f"{stage}/{cid}: no finished score record ({meta_path})")
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        synced = tables.get(f"{stage}/{cid}", {})
        if not synced.get("ok") or synced.get("sha256") != meta["table_sha256"]:
            raise Step06Error(f"{stage}/{cid}: not pulled and checked by `sync`")
        out.append(
            {
                "stage": stage,
                "table": paths.step6(stage, "scores", cid),
                "details": paths.step6(stage, "details", cid),
                "table_sha256": meta["table_sha256"],
                "details_sha256": meta["details_sha256"],
                "inputs_sha256": meta.get("inputs_sha256"),
            }
        )
    return out


def write_import(
    pipeline,
    part: str,
    tables: Sequence[dict[str, Score]],
    order: Sequence[str],
    provenance: dict[str, Any],
) -> dict[str, Any]:
    """Append the rows of ``tables`` (one per V, state_id -> Score) in ``order``, V1 then V2,
    to the run's ``scores.jsonl``; a row already there must be the same score."""
    pipeline._bind("scoring")  # noqa: SLF001
    known = pipeline._scores  # noqa: SLF001
    added = 0
    identities = [pipeline._verifier_identity(i) for i in range(2)]  # noqa: SLF001
    with JsonlWriter(pipeline.root / "scores.jsonl") as writer:
        for key in order:
            for (verifier_id, version), table in zip(identities, tables, strict=True):
                row = table[key]
                if (row.verifier_id, row.model_version) != (verifier_id, version):
                    raise Step06Error(f"{key}: a row of another verifier")
                present = known.get((key, verifier_id))
                if present is not None:
                    if present.score != row.score or present.model_version != version:
                        raise Step06Error(f"{key}: the run already holds another score")
                    continue
                writer.write(row)
                known[(key, verifier_id)] = row
                added += 1
    record = {"part": part, "rows_added": added, "imported_at": utc_now(), **provenance}
    path = pipeline.root / "score_imports" / f"{part}.json"
    if path.exists():
        previous = read_json(path)
        record = {
            **record,
            "imported_at": previous.get("imported_at"),
            "rows_added": previous.get("rows_added"),
        }
    pipeline._save_fixed(path, record)  # noqa: SLF001
    return record


def import_splits(pipeline, paths: Paths, selected: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Step 2's and this session's tables: D_fit + pool + D_gate + D_Exp1, nothing else."""
    states = pipeline._states()  # noqa: SLF001
    order = [
        states[q].state_id
        for split in (Split.D_FIT, Split.UNLABELED_POOL, Split.D_GATE, Split.D_EXP1)
        for q in pipeline._split_ids(split)  # noqa: SLF001
    ]
    loaded = step2_scores(pipeline, paths, selected)
    tables, sources, checks = [], list(loaded["sources"]), {}
    for i, candidate in enumerate(selected):
        cid = candidate["candidate_id"]
        model_id, revision = pipeline._verifier_identity(i)  # noqa: SLF001
        config_hash = loaded["config_hash"][i]
        checks[cid] = {
            k: v
            for k, v in check203_passed(paths, cid, config_hash).items()
            if k
            in ("criterion", "passed", "n_differing", "share_identical", "flips", "config_hash")
        }
        merged = dict(loaded["scores"][i])
        for source in step6_sources(paths, cid, SPLIT_STAGES):
            rows, hash_, summary = read_table(
                source["table"],
                source["details"],
                model_id=model_id,
                revision=revision,
                table_sha=source["table_sha256"],
                details_sha=source["details_sha256"],
            )
            if hash_ != config_hash:
                raise Step06Error(f"{cid} {source['stage']}: scored in another configuration")
            want = [r["state_id"] for r in read_checked(paths.stage_inputs(source["stage"]))]
            if [r.state_id for r in rows] != want:
                raise Step06Error(f"{cid} {source['stage']}: rows are not the inputs in order")
            for row in rows:
                if row.state_id in merged:
                    raise Step06Error(f"{cid}: {row.state_id} scored in two tables")
                merged[row.state_id] = row
            sources.append(
                {
                    "candidate_id": cid,
                    "step": 6,
                    "stage": source["stage"],
                    "table": str(source["table"]),
                    "table_sha256": source["table_sha256"],
                    "details": str(source["details"]),
                    "details_sha256": source["details_sha256"],
                    "config_hash": hash_,
                    "n": len(rows),
                    "truncated": sum(bool(s["truncated"]) for s in summary),
                }
            )
        if set(merged) != set(order):
            missing = len(set(order) - set(merged))
            extra = len(set(merged) - set(order))
            raise Step06Error(f"{cid}: coverage differs ({missing} missing, {extra} extra)")
        tables.append(merged)
    coverage = {
        split.value: len(pipeline._split_ids(split))  # noqa: SLF001
        for split in (Split.D_FIT, Split.UNLABELED_POOL, Split.D_GATE, Split.D_EXP1)
    }
    return write_import(
        pipeline,
        "splits",
        tables,
        order,
        {
            "coverage": coverage,
            "n_states": len(order),
            "not_scored": [Split.D_GATE_RESERVE.value],
            "sources": sources,
            "check203": checks,
            "basis": "II.10 generative variant and its math entry (step 2's D_fit and pool "
            "scores reused after this session's 203-state check; the rest of the pool, D_gate "
            "and D_Exp1 scored in the step 6 session)",
        },
    )


def import_stability(
    pipeline,
    paths: Paths,
    selected: Sequence[dict[str, Any]],
    *,
    n_per_class: int = md.N_PER_CLASS,
) -> dict[str, Any]:
    rows_in = read_checked(paths.stage_inputs("stability"))
    order = [r["state_id"] for r in rows_in]
    prepared = md.run_diagnostics(pipeline, stability=True, score=False, n_per_class=n_per_class)
    expected = [key for _, key, _ in prepared["items"]]
    if order != expected:
        raise Step06Error("the stability inputs are not the prepared sample's texts")
    tables, sources = [], []
    for i, candidate in enumerate(selected):
        cid = candidate["candidate_id"]
        model_id, revision = pipeline._verifier_identity(i)  # noqa: SLF001
        (source,) = step6_sources(paths, cid, ["stability"])
        rows, config_hash, summary = read_table(
            source["table"],
            source["details"],
            model_id=model_id,
            revision=revision,
            table_sha=source["table_sha256"],
            details_sha=source["details_sha256"],
        )
        check203_passed(paths, cid, config_hash)
        if [r.state_id for r in rows] != order:
            raise Step06Error(f"{cid} stability: rows are not the inputs in order")
        tables.append({r.state_id: r for r in rows})
        sources.append(
            {
                "candidate_id": cid,
                "step": 6,
                "stage": "stability",
                "table": str(source["table"]),
                "table_sha256": source["table_sha256"],
                "details": str(source["details"]),
                "details_sha256": source["details_sha256"],
                "config_hash": config_hash,
                "n": len(rows),
                "truncated": sum(bool(s["truncated"]) for s in summary),
            }
        )
    return write_import(
        pipeline,
        "stability",
        tables,
        order,
        {
            "n_texts": len(order),
            "inputs_sha256": recorded_sha(paths.stage_inputs("stability")),
            "sources": sources,
        },
    )


# ---------------------------------------------------------------------------------------
# time and cost of the session (step 2's measured throughput on the formal pod)
# ---------------------------------------------------------------------------------------


def session_cost(
    counts: dict[str, int],
    *,
    price: float = 1.62,
    smoke_hours: float = 2.0,
    setup_hours: float = 1.0,
    idle_hours: float = 1.0,
    loop_hours: float = 2.5,
) -> dict[str, Any]:
    """Pod hours and dollars: scoring at step 2's measured throughput, both 203-state checks,
    setup (deploy, venv, download, speed, precheck, preflight), the Exp 2 loop's precheck and
    203-state checks at L_in 4,096 (2026-10-13; B unknown until measured: about 32 for
    GLM and 16 for R1-Distill-Llama by memory, so about 7 and 13 batches of the 203), the smoke
    (at those B), and the idle time of pulls and waits."""
    rows = {}
    for cid, rate in THROUGHPUT.items():
        n = sum(counts.values())
        rows[cid] = {
            "states": n,
            "scoring_h": n / rate,
            "check203_h": CHECK203_MIN[cid] / 60,
        }
    gpu = sum(r["scoring_h"] + r["check203_h"] for r in rows.values())
    total = gpu + smoke_hours + setup_hours + idle_hours + loop_hours
    return {
        "per_verifier": rows,
        "counts": counts,
        "setup_h": setup_hours,
        "loop_check_h": loop_hours,
        "smoke_h": smoke_hours,
        "idle_h": idle_hours,
        "total_h": total,
        "price_per_hour": price,
        "total_usd": total * price,
    }


# ---------------------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="command", required=True)
    for name in (
        "inputs",
        "tau",
        "lengths",
        "check",
        "import",
        "stability-finish",
        "smoke-report",
        "cost",
    ):
        sub.add_parser(name)
    p_x1 = sub.add_parser("x1")
    p_x1.add_argument("--smoke", action="store_true")
    p_prep = sub.add_parser("stability-prepare")
    p_prep.add_argument("--confirm-live", action="store_true")
    p_install = sub.add_parser("loop-install")
    p_install.add_argument("--run-id", required=True)
    p_install.add_argument("--smoke", action="store_true")
    args = ap.parse_args(argv)
    # loop-install reads the loop ``follow`` pulled into the session's records directory
    paths = Paths(pod=pod_records()) if args.command == "loop-install" else Paths()
    if args.command == "cost":
        counts = {}
        for stage in STAGES:
            path = paths.stage_inputs(stage)
            counts[stage] = len(read_checked(path)) if path.is_file() else 0
        print(json.dumps(session_cost(counts), indent=1))
        return 0
    selected = selected_verifiers(paths)
    if args.command == "lengths":
        result = input_lengths(paths, selected)
        for cid, v in result["verifiers"].items():
            maxima = "; ".join(f"{k} max {f['max']}" for k, f in v["files"].items())
            over = f"; OVER L_in {len(v['over_L_in'])}, refused {len(v['refused'])}"
            print(f"{cid}: L_in {v['L_in']}; {maxima}" + ("" if v["ok"] else over))
        return 0 if result["ok"] else 8
    if args.command == "check":
        return check(paths, selected)
    pipeline = open_pipeline(
        paths,
        openai=args.command == "stability-prepare",
        confirm_live=getattr(args, "confirm_live", False),
    )
    with run_lock(pipeline.root):
        if args.command == "inputs":
            result = build_inputs(pipeline, paths)
        elif args.command == "tau":
            record = session_tau(pipeline, paths, selected)
            result = {
                "taus": record["taus"],
                "n": record["n"],
                "n_incorrect": record["n_incorrect"],
            }
        elif args.command == "x1":
            result = make_x1(pipeline, paths, smoke=args.smoke)
        elif args.command == "stability-prepare":
            result = stability_prepare(pipeline, paths)
        elif args.command == "import":
            result = {"splits": import_splits(pipeline, paths, selected)}
            if paths.stage_inputs("stability").is_file():
                result["stability"] = import_stability(pipeline, paths, selected)
            result = {
                part: {k: v for k, v in rec.items() if k not in ("sources",)}
                for part, rec in result.items()
            }
        elif args.command == "stability-finish":
            report = md.run_diagnostics(pipeline, stability=True)
            tau = read_tau(paths)
            if report["tau"] != tau["taus"]:
                raise Step06Error(
                    f"the diagnostics tau {report['tau']} differs from the session tau "
                    f"{tau['taus']}"
                )
            result = {"stability_status": report["stability_status"], "tau": report["tau"]}
        elif args.command == "loop-install":
            tau = read_tau(paths)["taus"] if args.smoke else None
            result = pipeline.install_loop(
                paths.loop_pulled(args.run_id, args.smoke), smoke=args.smoke, taus=tau
            )
        else:  # smoke-report
            pipeline.judge(smoke=True)
            report = pipeline.report(smoke=True)
            result = {
                k: report.get(k)
                for k in (
                    "SMOKE",
                    "n_questions",
                    "engineering_defects",
                    "thresholds",
                    "decision_count_distribution",
                )
            }
    print(json.dumps(finite_json(result), ensure_ascii=False))
    return 0


def check(paths: Paths, selected: Sequence[dict[str, Any]]) -> int:
    """Everything the pod needs is on the laptop, final and checked (runbook step 0)."""
    from rafr.domains.math.config import MathRunConfig
    from rafr.formatter.math_feedback import assert_anonymous_descriptions, load_prompts

    problems = []
    config = MathRunConfig.load(paths.formal_config)
    try:
        config.assert_ready("smoke")
        assert_anonymous_descriptions(
            load_prompts(),
            (n for s in config.verifiers for n in (s["model_id"], s["base_model"], s["vendor"])),
        )
        md.diagnostic_prompt_identity(pinned=True)
    except (ValueError, RuntimeError) as exc:
        problems.append(f"prompts / configuration: {exc}")
    v1, v2 = (c["candidate_id"] for c in selected)
    lines = [f"V1 {v1}, V2 {v2}: formal.json = step 2 specs"]
    for stage in STAGES:
        path = paths.stage_inputs(stage)
        if not path.is_file():
            problems.append(f"{path.name} missing")
            continue
        rows = read_checked(path)
        lines.append(f"{stage}: {len(rows)} inputs, sha256 {recorded_sha(path)[:16]}")
    for name, path in (("x1 smoke", paths.x1(True)),):
        if path.is_file():
            lines.append(
                f"{name}: {len(read_checked(path))} states, sha256 {recorded_sha(path)[:16]}"
            )
        else:
            problems.append(f"{path.name} missing")
    try:
        tau = read_tau(paths)
        lines.append(f"session tau: {tau['taus']}")
    except Step06Error as exc:
        problems.append(str(exc))
    if paths.lengths.is_file():
        lengths = json.loads(paths.lengths.read_text(encoding="utf-8"))
        if not lengths.get("ok"):
            problems.append("an input is longer than L_in (lengths.json)")
    else:
        problems.append("lengths.json missing: python pipelines/math_step06.py lengths")
    for candidate in selected:
        cid = candidate["candidate_id"]
        expected = step2_expected(paths, cid)
        details = paths.step2("screen", "details", cid)
        if sha256_file(details) != expected["screen"]["details"]:
            problems.append(f"{cid}: step 2 screening details (the 203 reference) changed")
    for line in lines:
        print(line)
    for problem in problems:
        print(f"NOT READY: {problem}")
    return 0 if not problems else 2


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (Step06Error, IntegrityError, ValueError, OSError) as exc:
        print(f"math step 6 stopped: {exc}", file=sys.stderr)
        sys.exit(1)
