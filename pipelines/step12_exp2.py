"""Step 12 — Exp 2 main experiment (0.5 step 12; 0.2, II.8.3, II.9.3; task 2026-09-24 s. 5).

    python pipelines/step12_exp2.py x1                      # laptop: the x1 file for the pod
    python pipelines/step12_exp2.py loop --run-dir D --x1-file F --x1-sha256 H \\
        --records R1 R2 --tau-file T --cache-dir C --numeric-check N --numeric-check-sha256 S \\
        --key-from-env --confirm                                            # the rented pod
    python pipelines/step12_exp2.py judge  --run-id ID --confirm   # laptop, after the pull
    python pipelines/step12_exp2.py panel  --run-id ID --confirm   # laptop
    python pipelines/step12_exp2.py report --run-id ID [--exp1-run-dir DIR]
    python pipelines/step12_exp2.py cutpoint   # laptop, before step 12: semantic cutpoint on D_fit

The loop runs on the rented pod and only there is anything scored (II.8.3): the controller calls
(OpenAI; the pod's OPENAI_API_KEY, checked for presence only) and V1 / V2 in process through the
one scoring factory (``models.verifiers.factory``, the probe's own code, checked against the
probe's records; cuda records and a CUDA run only). Before loading, the loop checks that it runs
on the fixed formal pod (``models.verifiers.formal_pod``), and after loading it re-scores the 203
D_fit states of the numeric reference and compares them with the probe's scores bit for bit; a
difference stops it before the first controller call (2026-09-24, D-2026-09-24c point 6).
Round-synchronous batching, K = 4, identity channel; every round is written to
its own directory and marked COMPLETE only when it is whole (``rafr.experiments.exp2.run_loop``),
and ``env/exp2_gpu.py`` pulls each round back as soon as it is complete. A resume starts from the
laptop's latest complete round. Each round prints the controller's running cost (ledger x list
price). The run is three segments, formal and smoke alike: ``x1`` on the laptop (the only one
that reads y*, and it writes none into the x1 file) -> ``loop`` on the pod -> ``judge`` / ``panel``
/ ``report`` on the laptop. The loop reads the x1 file, the probe's records, the numeric
reference, the tau file and the shipped code's configs and prompts -- no y*, nothing under data/
-- and refuses to run without ``--key-from-env``, so it reads no .env and builds no Anthropic
client (tests/test_exp2_runtime.py). E takes no part in the loop, and the pod holds no
Anthropic, xAI or Qwen key:

* ``judge`` (laptop): E judges every round text of every trajectory in both arms, deduplicated
  by text (``round_judgments.jsonl``); the t = 1 texts ARE D_Exp1's states, so their judgments
  are step 5's y*, read through the label gate (purpose EXP2_T1) and never re-bought (II.8.3);
  every final also gets quality(x0, x_T) = TextDetox SIM (LaBSE on the CPU,
  ``rafr.domains.detox.semantic``; 2026-09-28), whose preregistered cutpoint ``cutpoint``
  computed beforehand on D_fit's (x0, x1) pairs;
* ``panel`` (laptop): each panel member re-judges both arms' x_T with E's prompt, on its own
  (II.3, III.1.2); ``models.panel`` values are the authors' (App. D, filled 2026-09-24);
* ``report``: TFR contrast (arm-stratified bootstrap), sensitivities, c_t / R_t^(a), decision
  accuracy, rewrite improvement, error boundary alignment, the panel sensitivity when the
  panel files exist, and with ``--exp1-run-dir`` the transport diagnostic.

**Smoke mode** (``--smoke``; protocol II.4.4 as revised 2026-09-25, 0.5 step 10a): one short pod
session after step 10 and before step 11 that only shows the chain runs on the real components.
Its shape is ``configs/exp2_smoke.yaml``: A1 arm only; x1 = the G_fixed states of N = 30 E-UNSAFE
states of the UNLABELED POOL drawn by a fixed seed (their y* read under label purpose DRY_RUN;
``x1 --smoke``), never D_Exp1 / D_fit / D_gate / D_gate_reserve; the loop exactly as II.8.3 (K = 4,
accept allowed from t = 1, V1 / V2 in process through the factory, the formal pod and the 203-state
numeric check as for a formal run); the controller prompt byte for byte the formal one, its hashes
in ``loop_setup.json``; the controller seed namespace and cache key an experiment id of its own
(``exp2_smoke``); every request the controller actually received written to ``prompts_sent.jsonl``
for the report's prompt spot-check. Every output goes under ``runs/exp2_smoke/`` with a SMOKE
marker; the formal commands refuse a smoke run and a smoke run refuses the formal x1 file.
``judge --smoke`` reuses the pool's y* for t = 1 (UNSAFE by construction) and buys E's judgment of
every later round text and x_T; ``report --smoke`` writes ``smoke_report.json`` with the II.4.4
items -- T distribution and non-converged count, per-round E judgments and x_T failures, the
accept rate by observed (y1, y2) at t = 1 and t = 2, parse_fail / refusal_like counts, the prompt
spot-check (O_t injected, budget slot = 4, five significant digits) -- and a list of
``engineering_defects``. No behaviour of the controller is a release condition; the formal Exp 2
(``x1``, ``env/exp2_gpu.py up / start``) requires only that a smoke report with no engineering
defect is on disk (:func:`require_smoke_report`; App. B.1: the smoke never changes the controller
prompt, the formatter, K, the starting population or the controller's inputs).

The earlier smoke-only flags stay usable for engineering pushes but make a run invalid as the
step 10a smoke (``valid_for_release: false`` in the report): ``--v1 / --v2`` (other probe
config_ids), ``--tau-placeholder`` (one tau for both verifiers or two, V1 then V2, D-2026-09-25e)
and ``--always-toxic-stubs`` (V1 / V2 = stubs scoring every text 1.0, 2026-09-24; no factory
verifier is loaded, so no numeric check, recorded as not applicable). Reaching round 4 is no
condition of anything: rounds 3 and 4 and the resume path are covered by the always-rewrite stub
controller in the tests (``rafr.controller.stub``), and a ``stub-`` controller is refused by every
formal step, as a stub verifier is. ``--limit`` exists in smoke mode only and overrides N for an
engineering pass (also making the run invalid for release).
"""

from __future__ import annotations

# A direct ``python pipelines/<step>.py`` puts pipelines/ on sys.path, not the repo root,
# so the repo-root packages (analysis, annotation, pipelines) would not import; rafr is
# installed and unaffected. Covered by tests/test_pipeline_entrypoints.py.
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
import hashlib
import json
import re
from collections import Counter, defaultdict
from collections.abc import Callable, Mapping, Sequence
from datetime import UTC, datetime
from types import SimpleNamespace
from typing import Any

import numpy as np
from analysis import (
    error_boundary,
    evidence_sensitivity,
    exp2_tfr,
    parse_fail,
    reliability,
    transport,
)
from analysis._common import refuse_fake

from pipelines import _validity_io as io
from rafr.channel.identity import IdentityChannel
from rafr.contracts.schemas import (
    BOOTSTRAP_B,
    K_MAX_ROUNDS,
    Arm,
    EvalRecord,
    Observation,
    SafetyLabel,
    Split,
    State,
    text_hash,
)
from rafr.controller import decide as ctrl
from rafr.controller.stub import is_stub_controller
from rafr.core import paths
from rafr.core.config import assert_frozen, get, load_config, master_seed
from rafr.core.runtime import (
    Spend,
    build_client,
    build_panel_client,
    build_pod_client,
    prices_for,
)
from rafr.core.seeding import component_seed, py_rng
from rafr.core.textio import copy_tree_lf
from rafr.data import labels as L
from rafr.domains.detox import semantic
from rafr.evaluator import judge
from rafr.evaluator import panel as panel_mod
from rafr.experiments import exp2
from rafr.experiments.common import InProcessScorer, PairScorer
from rafr.formatter import TEMPLATES_PATH, fixed_significant, format_observation
from rafr.llm.client import LLMRequest, LLMResponse, content_text
from rafr.llm.ledger import Totals
from rafr.logging.snapshot import write_snapshot
from rafr.logging.writer import read_jsonl
from rafr.validity.tau import load_tau
from rafr.verifiers import factory, formal_pod, numeric_check
from rafr.verifiers.stub import always_toxic_pair, is_stub

LOOP_SECTIONS: tuple[str, ...] = (
    "models.controller",
    "models.verifiers",  # v1/v2 config_ids + the factory for in-process scoring
    "generation.g_fixed.validity",  # refusal_phrases_file (controller's refusal_like flag)
    "refusal_phrases",
    "exp2.design",
    "exp2.loop",
    "exp2.x0",
    "exp2.randomization",
    "exp2.controller",
    "exp2.run",
)
"""What the pod's loop reads. E, the panel and the analysis settings are not its business."""

GUARD_SECTIONS: tuple[str, ...] = (
    "models.controller",
    "models.evaluator",
    "models.verifiers",
    "generation.g_fixed.validity",
    "refusal_phrases",
    "exp2",
)
"""Config sections Exp 2 reads as a whole (judge and report check these; 2026-09-18:
a step is blocked only by the values it consumes). The rest of exp1.yaml is Exp 1's business."""

JUDGE_SECTIONS: tuple[str, ...] = ("models.evaluator", "exp2.design", "exp2.x0")
"""What E's post-loop judgments read (laptop)."""

SMOKE_DIR_NAME = "exp2_smoke"
SMOKE_MARKER = "SMOKE"
SMOKE_CONFIG = "exp2_smoke"
"""``configs/exp2_smoke.yaml``: the shape of the step 10a smoke (II.4.4, 2026-09-25)."""
SMOKE_REPORT = "smoke_report.json"
"""The II.4.4 smoke report; the formal Exp 2 needs one with no engineering defect on disk."""
PROMPTS_SENT = "prompts_sent.jsonl"
"""Smoke only: every controller request as actually sent, for the prompt spot-check."""
X1_DIR_NAME = "exp2"
X1_FORMAL = "x1_states.jsonl"
X1_SMOKE = "x1_smoke_states.jsonl"
X1_FIELDS = ("state_id", "x0_id", "x0", "text")

Echo = Callable[[str], None]


def _ts() -> str:
    return datetime.now(tz=UTC).strftime("%Y%m%dT%H%M%SZ")


def _sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def bootstrap_settings(configs) -> tuple[int, int]:
    """(n_boot, seed) for the TFR contrast CI (III.1.2).

    B = 10000 is locked. The seed is this run's seed, derived from the one master seed
    (component ``exp2_run``), and is written into the analysis output and the run snapshot.
    """
    n_boot = get(configs["exp2"], "analysis.bootstrap.n_boot", BOOTSTRAP_B)
    return int(n_boot), component_seed(master_seed(configs), "exp2_run")


def load_verifier(models_cfg, key: str, *, config_id: str | None = None, records=None):
    """V1 / V2 through the one factory (task 2026-09-24 section 3), formal: a cuda record and a
    CUDA run only (D-2026-09-24c point 6), smoke runs included -- they test that path. Stubs and
    the registry guard backend are refused. ``config_id`` overrides the configured id (smoke
    mode only); ``records`` are the probe records' directories (on the pod: ``--records``)."""
    if config_id is None:
        v = factory.from_config(models_cfg, key, records_dirs=records, echo=print)
    else:
        v = factory.make_verifier(
            config_id,
            records_dirs=records or factory.records_dirs_from(models_cfg),
            echo=print,
            formal=True,
        )
    v = factory.refuse_legacy_guard(v)
    if is_stub(v.verifier_id):
        raise SystemExit(f"{v.verifier_id} is a stub; formal steps refuse stubs (II.4.4)")
    return v


def check_pod(models_cfg: Mapping[str, Any], echo: Echo) -> dict[str, Any]:
    """The fixed formal pod (image fingerprint, GPU; D-2026-09-24c point 6), before loading."""
    try:
        return formal_pod.check_here(models_cfg, echo)
    except formal_pod.FormalPodError as exc:
        raise SystemExit(str(exc)) from exc


def check_numbers(
    verifiers: Sequence[Any], reference: Path, reference_sha256: str, echo: Echo
) -> dict[str, Any]:
    """The 203 D_fit states re-scored by each loaded verifier and compared with the probe's
    scores bit for bit (D-2026-09-24c point 6); a difference stops the loop before any call."""
    ref = numeric_check.read_reference(Path(reference), reference_sha256)
    out: dict[str, Any] = {}
    for v in verifiers:
        cid = getattr(v, "config_id", None) or v.verifier_id
        try:
            out[cid] = numeric_check.check_verifier(v, ref.get(cid, []), echo)
        except numeric_check.NumericCheckError as exc:
            raise SystemExit(f"numeric check failed, nothing scored: {exc}") from exc
    return out


def prior_from_y_star(
    states: Sequence[State],
    ecfg: judge.EvaluatorConfig,
    *,
    split: Split = Split.D_EXP1,
    purpose: L.LabelPurpose = L.LabelPurpose.EXP2_T1,
) -> dict[str, EvalRecord]:
    """``text_hash -> EvalRecord``: step 5's judgments of the x1 states, which ARE the t = 1 texts
    (II.8.3, v0.9: t = 1 reuses step 5's E judgment). Formal run: D_Exp1 through the label gate
    under purpose EXP2_T1. Step 10a smoke: the unlabeled pool under purpose DRY_RUN (II.4.4:
    t = 1 is UNSAFE by construction, so a SAFE record means the x1 file is not the smoke's).
    Every record must come from the configured E (II.9.1) and every x1 must be covered --
    re-buying one could label the same text differently in the two experiments."""
    ys = L.read_labels(split, purpose)
    by_state = {y.state_id: y.record for y in ys}
    for rec in by_state.values():
        judge.check_same_e(judge.record_identity(rec), ecfg, where=f"{split.value}'s y* (step05)")
    uncovered = [s.state_id for s in states if s.state_id not in by_state]
    if uncovered:
        raise SystemExit(
            f"{len(uncovered)} x1 state(s) have no step-5 judgment in {split.value} "
            f"(e.g. {uncovered[:5]}); t = 1 reuses step 5's judgment (II.8.3)"
        )
    if purpose is L.LabelPurpose.DRY_RUN:
        safe = [s.state_id for s in states if by_state[s.state_id].label is not SafetyLabel.UNSAFE]
        if safe:
            raise SystemExit(
                f"{len(safe)} smoke x1 state(s) are not E-UNSAFE (e.g. {safe[:5]}); the step 10a "
                "smoke starts from E-UNSAFE pool states only (II.4.4)"
            )
    return {text_hash(s.text): by_state[s.state_id] for s in states}


def refuse_stub_controller(models_cfg: Mapping[str, Any]) -> None:
    """A controller whose model id starts with ``stub-`` is for tests only (II.4.4): every
    formal step and the step 10a smoke refuse it, as they refuse a stub verifier."""
    model = str((models_cfg.get("controller") or {}).get("model", ""))
    if is_stub_controller(model):
        raise SystemExit(
            f"models.controller.model = {model!r} is a stub controller; formal steps refuse "
            "stubs (II.4.4)"
        )


def controller_prompt_hashes(models_cfg: Mapping[str, Any], k: int) -> dict[str, str]:
    """sha256 of the controller's three prompt files and of the system prompt as rendered with
    the decision budget ``k`` (II.4.4: the smoke's prompt is the formal one, byte for byte).

    The file hashes are taken over the LF-normalized bytes (decisions.md D-2026-09-27), like E's,
    the verifiers' and the perturbation prompts': a Windows checkout may hold CRLF where the pod
    holds LF, and the check must not depend on which machine made it."""
    c = models_cfg["controller"]
    out: dict[str, str] = {}
    for key in ("system_prompt_file", "rewrite_prompt_file", "parse_retry_prompt_file"):
        p = paths.resolve(c[key])
        out[key] = hashlib.sha256(p.read_bytes().replace(b"\r\n", b"\n")).hexdigest()
    system = paths.resolve(c["system_prompt_file"]).read_text(encoding="utf-8")
    rendered = ctrl.render_system_prompt(system, k)
    out["system_prompt_rendered_k"] = str(k)
    out["system_prompt_rendered_sha256"] = hashlib.sha256(rendered.encode("utf-8")).hexdigest()
    return out


class PromptRecorder:
    """Smoke only (II.4.4): wraps the LLM client and appends every controller request as the
    provider received it -- system prompt, user turn, attempt index, prompt hash -- to
    ``prompts_sent.jsonl``, so the report can check what was actually sent. Nothing about the
    request or the answer is changed; the ledger is the inner client's."""

    def __init__(self, inner: Any, path: Path) -> None:
        self._inner = inner
        self._path = Path(path)

    @property
    def ledger(self):
        return getattr(self._inner, "ledger", None)

    def complete(self, request: LLMRequest, **kw: Any) -> LLMResponse:
        resp = self._inner.complete(request, **kw)
        user = next((m["content"] for m in reversed(request.messages) if m["role"] == "user"), "")
        row = {
            "prompt_hash": resp.prompt_hash,
            "purpose": request.purpose,
            "attempt": request.attempt,
            "model": request.model,
            "system": request.system,
            "user": content_text(user),
            "cached": resp.cached,
        }
        with self._path.open("a", encoding="utf-8", newline="\n") as fh:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
        return resp


def smoke_cfg(cfgs: Mapping[str, Any]) -> Mapping[str, Any]:
    sc = cfgs.get(SMOKE_CONFIG)
    if not sc:
        raise SystemExit(
            f"configs/{SMOKE_CONFIG}.yaml missing: the step 10a smoke's shape (II.4.4)"
        )
    return sc


def smoke_x1(cfgs: Mapping[str, Any], limit: int | None = None) -> tuple[list[State], dict]:
    """The smoke's starting states (II.4.4, 2026-09-25): N E-UNSAFE states of the unlabeled pool
    by a seeded shuffle, their y* read through the label gate under purpose DRY_RUN. No other
    split is touched. ``limit`` overrides N for an engineering pass (recorded; such a run is not
    valid for release)."""
    sc = smoke_cfg(cfgs)
    if get(sc, "x0.source") != Split.UNLABELED_POOL.value:
        raise SystemExit("exp2_smoke.x0.source must be unlabeled_pool (II.4.4)")
    if get(sc, "x0.label_purpose") != L.LabelPurpose.DRY_RUN.value:
        raise SystemExit("exp2_smoke.x0.label_purpose must be dry_run (II.2.4, II.4.4)")
    configured_n = int(get(sc, "x0.n", 30))
    n = configured_n if limit is None else int(limit)
    ys = L.read_labels(Split.UNLABELED_POOL, L.LabelPurpose.DRY_RUN)
    unsafe = {y.state_id for y in ys if y.label is SafetyLabel.UNSAFE}
    pool = io.load_states(Split.UNLABELED_POOL)
    candidates = sorted((s for s in pool if s.state_id in unsafe), key=lambda s: s.state_id)
    if len(candidates) < n:
        raise SystemExit(f"only {len(candidates)} E-UNSAFE pool states, {n} wanted (II.4.4)")
    seed = component_seed(master_seed(cfgs), str(get(sc, "x0.seed_component", "exp2_smoke_x0")))
    py_rng(seed, "exp2_smoke_x0").shuffle(candidates)
    info = {
        "source": Split.UNLABELED_POOL.value,
        "label_purpose": L.LabelPurpose.DRY_RUN.value,
        "e_unsafe_only": True,
        "n": n,
        "configured_n": configured_n,
        "seed": seed,
        "n_candidates": len(candidates),
    }
    return candidates[:n], info


def smoke_reports_on_disk() -> list[tuple[Path, dict[str, Any]]]:
    base = paths.RUNS / SMOKE_DIR_NAME
    out: list[tuple[Path, dict[str, Any]]] = []
    if not base.is_dir():
        return out
    for p in sorted(base.glob(f"*/{SMOKE_REPORT}")):
        try:
            out.append((p, json.loads(p.read_text(encoding="utf-8"))))
        except (OSError, json.JSONDecodeError):
            continue
    return out


def require_smoke_report() -> Path:
    """The one release condition of the formal Exp 2 (II.4.4, 2026-09-25): a step 10a smoke
    report is on disk and its engineering-defect row was not triggered (or the smoke was rerun
    after the fix and passes). No behaviour of the controller enters this check."""
    reports = smoke_reports_on_disk()
    valid = [p for p, r in reports if r.get("valid_for_release") is True]
    if valid:
        return valid[-1]
    if not reports:
        raise SystemExit(
            f"no step 10a smoke report ({SMOKE_REPORT}) under {paths.RUNS / SMOKE_DIR_NAME}: run "
            "the Exp 2 smoke first (II.4.4; step12 x1 --smoke, env/exp2_gpu.py ... --smoke, "
            "judge --smoke, report --smoke)"
        )
    reasons = {str(p.parent.name): r.get("not_valid_because") for p, r in reports}
    raise SystemExit(
        "no step 10a smoke report is valid for release (II.4.4: engineering defects must be "
        f"fixed and the smoke rerun): {json.dumps(reasons, ensure_ascii=False)}"
    )


# ---------------------------------------------------------------------------------------
# run directories and the x1 file
# ---------------------------------------------------------------------------------------


def run_dir_for(run_id: str, *, smoke: bool) -> Path:
    if smoke:
        d = paths.RUNS / SMOKE_DIR_NAME / run_id
        d.mkdir(parents=True, exist_ok=True)
        return d
    return paths.run_dir(run_id)


def existing_run_dir(run_id: str, *, smoke: bool) -> Path:
    """The run directory of a finished loop (judge / panel / report never create one)."""
    rd = paths.RUNS / SMOKE_DIR_NAME / run_id if smoke else paths.RUNS / run_id
    if not rd.is_dir():
        if not smoke and (paths.RUNS / SMOKE_DIR_NAME / run_id).is_dir():
            raise SystemExit(f"{run_id} is a smoke run; the formal pipeline does not read it")
        raise SystemExit(f"no run directory {rd}")
    check_mode(rd, smoke=smoke)
    return rd


def mark_smoke(rd: Path, note: str | None = None) -> None:
    (Path(rd) / SMOKE_MARKER).write_text(
        "smoke run (protocol II.4.4, 0.5 step 10a): x0 from the unlabeled pool; the formal "
        "pipeline never reads this directory; nothing here enters the proposal or the paper\n"
        + (f"{note}\n" if note else ""),
        encoding="utf-8",
        newline="\n",
    )


ALWAYS_TOXIC_NOTE = (
    "always-toxic stub verifiers (2026-09-24): every text scores 1.0 on both checkers, to "
    "push the trajectories through all K rounds; smoke only, never formal"
)


def check_mode(rd: Path, *, smoke: bool) -> None:
    """A formal command refuses a smoke run, and a smoke command a formal one. Neither reads
    step 10's fake data (the FAKE marker, analysis._common)."""
    refuse_fake(rd, what="step12")
    is_smoke = (Path(rd) / SMOKE_MARKER).exists() or SMOKE_DIR_NAME in Path(rd).parts
    if is_smoke and not smoke:
        raise SystemExit(f"{rd} is a smoke run; the formal pipeline does not read it")
    if smoke and not is_smoke and (Path(rd) / "loop_complete.json").exists():
        raise SystemExit(f"{rd} is a formal run; --smoke may not touch it")


def x1_path(*, smoke: bool) -> Path:
    return paths.DATA / X1_DIR_NAME / (X1_SMOKE if smoke else X1_FORMAL)


def write_x1(states: Sequence[State], path: Path) -> str:
    """x1 for the pod: state_id, x0_id, x0 and text only -- never a label (y* stays here)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as fh:
        for s in states:
            fh.write(json.dumps({k: getattr(s, k) for k in X1_FIELDS}, ensure_ascii=False) + "\n")
    digest = _sha256(path)
    path.with_name(path.name + ".sha256").write_text(
        f"{digest}  {path.name}\n", "utf-8", newline="\n"
    )
    return digest


def read_x1(path: Path, expected_sha256: str) -> list[State]:
    if not Path(path).is_file():
        raise SystemExit(f"{path} not found: python pipelines/step12_exp2.py x1")
    if _sha256(path) != expected_sha256.strip().lower():
        raise SystemExit(f"{path}: sha256 does not match {expected_sha256}")
    out: list[State] = []
    for i, row in enumerate(read_jsonl(path), start=1):
        if tuple(sorted(row)) != tuple(sorted(X1_FIELDS)):
            raise SystemExit(f"{path}:{i}: fields {sorted(row)}; only {X1_FIELDS}")
        out.append(State(**row))
    return out


# ---------------------------------------------------------------------------------------
# cost per round
# ---------------------------------------------------------------------------------------


def cumulative_cost(rd: Path, model: str, k: int = K_MAX_ROUNDS) -> dict[str, Any]:
    """Controller tokens and dollars over every COMPLETE round of the run (live calls only;
    an answer served from the LLM cache bought nothing), from each round's ledger rows."""
    totals = Totals()
    for t in exp2.complete_rounds(rd, k):
        for row in read_jsonl(exp2.round_dir(rd, t) / "ledger.jsonl"):
            totals.add(row)
    spend = Spend(SimpleNamespace(totals=totals), prices_for(model), "controller")  # type: ignore[arg-type]
    return {"usd": spend.usd, "tokens": spend.tokens, "live_calls": spend.live_calls}


def snapshot(rd: Path, *, step: str, cfgs: Mapping[str, Any], extra: dict[str, Any]) -> None:
    """II.10 run snapshot. On the pod the code is a `git archive` without .git: the commit is
    then the RAFR_GIT_HEAD the setup script exports, recorded as such."""
    try:
        write_snapshot(rd, step=step, configs=dict(cfgs), extra=extra)
    except RuntimeError:
        import os
        import shutil

        dst = Path(rd) / "configs"
        if dst.exists():
            shutil.rmtree(dst)
        copy_tree_lf(paths.CONFIGS, dst)  # LF whatever the checkout holds (D-2026-09-27)
        (Path(rd) / "snapshot.json").write_text(
            json.dumps(
                {
                    "step": step,
                    "started_at": datetime.now(tz=UTC).isoformat(timespec="seconds"),
                    "git_head": os.environ.get("RAFR_GIT_HEAD"),
                    "git_head_source": "RAFR_GIT_HEAD (code shipped as git archive, no .git)",
                    "argv": sys.argv,
                    "configs": cfgs,
                    **extra,
                },
                ensure_ascii=False,
                indent=1,
                default=str,
            ),
            encoding="utf-8",
            newline="\n",
        )


def round_printer(rd: Path, model: str, echo: Echo) -> Callable[[int, dict[str, Any]], None]:
    def hook(t: int, info: dict[str, Any]) -> None:
        c = cumulative_cost(rd, model)
        money = "n/a" if c["usd"] is None else f"${c['usd']:.4f}"
        echo(
            f"[exp2] round {t} complete: {info['n_decisions']} decisions, "
            f"{info['n_accepted_total']} trajectories accepted so far; controller cumulative "
            f"{money} ({c['live_calls']} live calls)"
        )

    return hook


# ---------------------------------------------------------------------------------------
# commands
# ---------------------------------------------------------------------------------------


def cmd_x1(args: argparse.Namespace, cfgs: Mapping[str, Any], echo: Echo) -> int:
    if args.smoke:
        states, info = smoke_x1(cfgs, args.limit)
        echo(
            f"[step12] SMOKE x1: {info['n']} E-UNSAFE pool states (configured "
            f"{info['configured_n']}, seed {info['seed']}, label purpose {info['label_purpose']})"
        )
    else:
        if args.limit:
            raise SystemExit("--limit is for smoke tests only: Exp 2 runs on all of D_Exp1")
        if get(cfgs["exp2"], "x0.source") != "d_exp1":
            raise SystemExit("exp2.x0.source must be d_exp1 (group 6d decision)")
        # II.4.4 (2026-09-25): the formal Exp 2 starts only after a valid step 10a smoke report
        echo(f"[step12] step 10a smoke report: {require_smoke_report()}")
        states = io.load_states(Split.D_EXP1)
    path = x1_path(smoke=args.smoke)
    digest = write_x1(states, path)
    echo(f"[step12] {len(states)} x1 states -> {path} (sha256 {digest})")
    return 0


def build_setup(
    *,
    cfgs: Mapping[str, Any],
    run_id: str,
    rd: Path,
    states: Sequence[State],
    scorer: PairScorer,
    tau1: float,
    tau2: float,
    judge_fn: Callable[[str], EvalRecord],
    prior: Mapping[str, EvalRecord] | None = None,
    smoke: bool = False,
    quality_fn: Callable[[str, str], float | None] | None = None,
) -> exp2.Exp2Setup:
    """The formal setup reads ``configs/exp2.yaml``; the step 10a smoke reads
    ``configs/exp2_smoke.yaml`` (A1 only, its own seed components and controller seed
    namespace, II.4.4). Everything else -- K, the identity channel, the scorer, E's role -- is
    the same code path. ``quality_fn`` is quality(x0, x_T) for the finals: only ``judge`` (the
    laptop) passes one (:func:`make_quality_fn`); the pod's loop never builds it."""
    if smoke:
        ex = smoke_cfg(cfgs)
        assign_comp = str(get(ex, "run.randomization_seed_component", "exp2_smoke_randomization"))
        run_comp = str(get(ex, "run.seed_component", "exp2_smoke_run"))
        namespace = str(get(ex, "run.experiment", "exp2_smoke"))
        experiment_name = "exp2_smoke"
    else:
        ex = cfgs["exp2"]
        assign_comp, run_comp = "exp2_randomization", "exp2_run"
        namespace, experiment_name = exp2.SEED_NAMESPACE, "exp2"
    return exp2.Exp2Setup(
        run_id=run_id,
        run_dir=rd,
        x1_states=list(states),
        scorer=scorer,
        tau1=tau1,
        tau2=tau2,
        channel=IdentityChannel(),
        arms=[Arm(x) for x in get(ex, "design.arms")],
        k=int(get(ex, "design.k", K_MAX_ROUNDS)),
        assignment_seed=component_seed(master_seed(cfgs), assign_comp),
        run_seed=component_seed(master_seed(cfgs), run_comp),
        judge_fn=judge_fn,
        prior_judgments=prior,
        # quality(x0, x_T) for the "low preservation among SAFE" report (2026-09-18)
        quality_fn=quality_fn,
        experiment_name=experiment_name,
        seed_namespace=namespace,
    )


def placeholder_taus(values: float | Sequence[float]) -> tuple[float, float]:
    """``--tau-placeholder``: one value for both verifiers (the always-toxic smoke: 0.5), or
    one per verifier, V1 then V2 (the real V1 / V2 smoke: the probe's D_fit F1 taus, the authors
    2026-09-25, D-2026-09-25e). Placeholders either way: smoke runs only."""
    vals = [float(v) for v in (values if isinstance(values, list | tuple) else [values])]
    if len(vals) == 1:
        return vals[0], vals[0]
    if len(vals) == 2:
        return vals[0], vals[1]
    raise SystemExit(f"--tau-placeholder takes one value or two (V1 V2), got {len(vals)}")


def _no_e(text: str) -> EvalRecord:
    raise exp2.Exp2Error("E is not part of the loop (II.8.3)")


def cmd_loop(args: argparse.Namespace, cfgs: Mapping[str, Any], echo: Echo) -> int:
    models = cfgs["models"]
    stubs = bool(args.always_toxic_stubs)
    if not args.smoke:
        if args.v1 or args.v2 or args.tau_placeholder is not None or stubs:
            raise SystemExit(
                "--v1 / --v2 / --tau-placeholder / --always-toxic-stubs are for smoke tests only"
            )
        assert_frozen(cfgs, extra_files=[TEMPLATES_PATH], only_sections=LOOP_SECTIONS)
    elif stubs and (args.v1 or args.v2 or args.tau_placeholder is None):
        raise SystemExit(
            "--always-toxic-stubs replaces --v1 / --v2 and needs --tau-placeholder (the stubs "
            "score 1.0, above any tau)"
        )
    else:
        # II.4.4 (2026-09-25): the smoke's controller prompt and formatter templates are the
        # formal ones, byte for byte -- no placeholder may be left in either
        assert_frozen(
            cfgs,
            extra_files=[TEMPLATES_PATH],
            only_sections=["models.controller", "models.verifiers"],
        )
    refuse_stub_controller(models)  # a stub- controller is for tests only (II.4.4)
    if not args.confirm:
        raise SystemExit("the loop spends money on controller calls: add --confirm")
    if get(cfgs["exp2"], "design.channel") != "identity":
        raise SystemExit("Exp 2 has no channel (II.1.3)")
    if not args.x1_file or not args.x1_sha256:
        raise SystemExit("loop needs --x1-file and --x1-sha256 (python step12_exp2.py x1)")
    if (Path(args.x1_file).name == X1_SMOKE) != bool(args.smoke):
        raise SystemExit("a smoke run takes the smoke x1 file and a formal run the formal one")
    states = read_x1(Path(args.x1_file), args.x1_sha256)
    if not args.key_from_env:
        # the loop is the pod segment (select on the laptop -> loop on the pod -> judge / report
        # on the laptop): OpenAI only, its key from the pod's environment; no .env is read and
        # no Anthropic client is built (E runs on the laptop afterwards)
        raise SystemExit(
            "loop runs on the rented pod: add --key-from-env (the controller's OPENAI_API_KEY "
            "from the pod's environment; the loop builds no Anthropic client and reads no .env)"
        )
    run_id = args.run_id or f"{get(cfgs['exp2'], 'run.run_id_prefix', 'exp2')}_{_ts()}"
    rd = Path(args.run_dir) if args.run_dir else run_dir_for(run_id, smoke=args.smoke)
    rd.mkdir(parents=True, exist_ok=True)
    if args.smoke:
        mark_smoke(rd, ALWAYS_TOXIC_NOTE if stubs else None)
    check_mode(rd, smoke=args.smoke)
    snapshot(
        rd, step="step12_loop", cfgs=cfgs, extra={"smoke": args.smoke, "x1_sha256": args.x1_sha256}
    )
    if args.tau_placeholder is not None:
        tau1, tau2 = placeholder_taus(args.tau_placeholder)
    else:
        taus = load_tau(Path(args.tau_file) if args.tau_file else paths.FROZEN_TAU)
        tau1, tau2 = taus["v1"].tau, taus["v2"].tau
    if not stubs and not (args.numeric_check and args.numeric_check_sha256):
        raise SystemExit(
            "loop needs --numeric-check and --numeric-check-sha256: the 203 D_fit states are "
            "re-scored against the probe before the first call (D-2026-09-24c point 6)"
        )
    pod_env = check_pod(models, echo)
    if stubs:
        v1, v2 = always_toxic_pair()
        numbers: dict[str, Any] = {}
        echo(f"[step12] SMOKE: {ALWAYS_TOXIC_NOTE}")
    else:
        records = list(args.records) if args.records else None
        v1 = load_verifier(models, "v1", config_id=args.v1, records=records)
        v2 = load_verifier(models, "v2", config_id=args.v2, records=records)
        try:
            numbers = check_numbers([v1, v2], args.numeric_check, args.numeric_check_sha256, echo)
        except BaseException:
            v1.close()
            v2.close()
            raise
    client = build_pod_client(models, run_dir=rd, cache_dir=Path(args.cache_dir or rd / "cache"))
    if args.smoke:
        # II.4.4: every request the controller actually received, for the prompt spot-check
        client = PromptRecorder(client, rd / PROMPTS_SENT)
    ccfg = ctrl.build_config(
        models,
        smoke_cfg(cfgs) if args.smoke else cfgs["exp2"],
        base_seed=component_seed(master_seed(cfgs), "controller"),
        refusal_phrases_file=get(cfgs["generation"], "g_fixed.validity.refusal_phrases_file"),
    )
    setup = build_setup(
        cfgs=cfgs, run_id=run_id, rd=rd, states=states, scorer=InProcessScorer(v1, v2),
        tau1=tau1, tau2=tau2, judge_fn=_no_e, smoke=args.smoke,
    )  # fmt: skip
    smoke_profile: dict[str, Any] | None = None
    if args.smoke:
        sc = smoke_cfg(cfgs)
        smoke_profile = {
            "arms": [a.value for a in setup.arms],
            "k": setup.k,
            "seed_namespace": setup.seed_namespace,
            "configured_n": int(get(sc, "x0.n", 30)),
            "verifier_override": bool(args.v1 or args.v2),
            # II.4.4: the controller prompt is the formal one, byte for byte
            "controller_prompt": controller_prompt_hashes(models, setup.k),
            "prompts_sent_file": PROMPTS_SENT,
        }
    (rd / "loop_setup.json").write_text(
        json.dumps(
            {
                "run_id": run_id,
                "smoke": args.smoke,
                "n_x1": len(states),
                "x1_sha256": args.x1_sha256,
                "k": setup.k,
                "arms": [a.value for a in setup.arms],
                "tau": [tau1, tau2],
                "tau_placeholder": args.tau_placeholder is not None,
                "tau_placeholder_note": (
                    "placeholder taus (smoke run), not frozen/tau.json"
                    if args.tau_placeholder is not None
                    else None
                ),
                "v1": v1.describe(),
                "v2": v2.describe(),
                "formal_pod": pod_env,
                "always_toxic_stubs": stubs,
                "numeric_check": (
                    {"not_applicable": "always-toxic stubs: no factory verifier is loaded"}
                    if stubs
                    else {
                        cid: {k: r.get(k) for k in ("n", "n_different", "max_abs_diff")}
                        for cid, r in numbers.items()
                    }
                ),
                "smoke_profile": smoke_profile,
            },
            ensure_ascii=False,
            indent=1,
            default=str,
        ),
        encoding="utf-8",
        newline="\n",
    )
    per_round: dict[int, int] = {}
    printer = round_printer(rd, ccfg.model, echo)

    def on_round(t: int, info: dict[str, Any]) -> None:
        per_round[t] = int(info["n_decisions"])
        printer(t, info)

    try:
        exp2.run_loop(client, ccfg, setup, on_round=on_round)
    finally:
        v1.close()
        v2.close()
    echo(f"[step12] loop complete: {rd}")
    if args.smoke:
        # information only (II.4.4, 2026-09-25): how far the trajectories went is a result to
        # report, never a condition -- no round has to be reached for the smoke to count
        echo(f"[step12] SMOKE decisions per round (this run): {dict(sorted(per_round.items()))}")
    return 0


def _local_states(rd: Path, smoke: bool) -> list[State]:
    """The x1 the loop ran on, read back from the laptop's x1 file (checked by sha256)."""
    setup_info = json.loads((rd / "loop_setup.json").read_text(encoding="utf-8"))
    path = x1_path(smoke=smoke)
    return read_x1(path, setup_info["x1_sha256"])


def make_quality_fn(cfgs: Mapping[str, Any]) -> semantic.QualityScorer:
    """quality(x0, x_T) = TextDetox SIM (2026-09-28), configured in exp2.yaml. The
    pinned sentence-transformers is checked here, before E is paid for; LaBSE loads on the
    first score."""
    scfg = semantic.build_config(cfgs["exp2"])
    try:
        semantic.check_versions([scfg.sim_library])
    except semantic.SemanticError as exc:
        raise SystemExit(f"quality(x0, x_T) cannot be computed: {exc}") from exc
    return semantic.QualityScorer(scfg)


def cmd_judge(args: argparse.Namespace, cfgs: Mapping[str, Any], echo: Echo) -> int:
    if not args.smoke:
        assert_frozen(cfgs, only_sections=JUDGE_SECTIONS)
    if not args.confirm:
        raise SystemExit("E's round judgments cost money: add --confirm")
    rd = existing_run_dir(args.run_id, smoke=args.smoke)
    states = _local_states(rd, args.smoke)
    quality = make_quality_fn(cfgs)
    ecfg = judge.build_config(cfgs["models"])
    client = build_client(cfgs["models"], run_dir=rd)
    # t = 1 reuses step 5's y* (II.8.3): D_Exp1 under EXP2_T1 for the formal run; for the
    # step 10a smoke the pool's y* under DRY_RUN, UNSAFE by construction (II.4.4)
    if args.smoke:
        prior = prior_from_y_star(
            states, ecfg, split=Split.UNLABELED_POOL, purpose=L.LabelPurpose.DRY_RUN
        )
    else:
        prior = prior_from_y_star(states, ecfg)
    setup = build_setup(
        cfgs=cfgs, run_id=args.run_id, rd=rd, states=states, scorer=_NoScorer(),
        tau1=0.0, tau2=0.0, judge_fn=lambda text: judge.judge_text(client, ecfg, text),
        prior=prior, smoke=args.smoke, quality_fn=quality,
    )  # fmt: skip
    summary = exp2.finish(setup)
    (rd / "summary.json").write_text(
        json.dumps(summary, indent=1, default=str), "utf-8", newline="\n"
    )
    echo(f"[step12] E judged the rounds of {rd}: {json.dumps(summary['arms'])}")
    return 0


class _NoScorer:
    """judge / report never score: the loop is over."""

    v1_id = v2_id = v1_version = v2_version = "none"

    def score_pair(self, texts):
        raise exp2.Exp2Error("nothing is scored after the loop")


def cmd_panel(args: argparse.Namespace, cfgs: Mapping[str, Any], echo: Echo) -> int:
    if not args.confirm:
        raise SystemExit("the panel's calls cost money: add --confirm")
    rd = existing_run_dir(args.run_id, smoke=args.smoke)
    members = panel_mod.build_members(cfgs["models"], allow_tbd=False)
    rule = panel_mod.on_failure_rule(cfgs["models"])
    finals = list(read_jsonl(rd / "trajectories.jsonl"))
    if not finals:
        raise SystemExit(f"{rd}/trajectories.jsonl missing: run `judge` first")
    client = build_panel_client(cfgs["models"], run_dir=rd)
    out = panel_mod.rejudge_x_t(client, members, finals, rd, on_failure=rule, echo=echo)
    (rd / "panel_summary.json").write_text(
        json.dumps(
            {k: {kk: vv for kk, vv in v.items() if kk != "labels"} for k, v in out.items()},
            indent=1,
        ),
        encoding="utf-8",
        newline="\n",
    )
    return 0


def panel_labels(rd: Path) -> dict[str, dict[str, str | None]]:
    out: dict[str, dict[str, str | None]] = {}
    for f in sorted(Path(rd).glob("panel_*.jsonl")):
        member = f.stem.removeprefix("panel_")
        labels: dict[str, str | None] = {}
        for row in read_jsonl(f):
            labels[row["text_id"]] = row["label"]
        out[member] = labels
    return out


def _observation_shown(row: Mapping[str, Any]) -> str:
    """The O_t the formatter rendered for this decision row, rebuilt from the row's own
    post-channel scores and thresholds (the same pure function, II.1.4)."""
    obs = Observation(
        state_id=row["state_id"],
        s1_obs=float(row["s1_tilde"]),
        s2_obs=float(row["s2_tilde"]),
        tau1=float(row["tau1"]),
        tau2=float(row["tau2"]),
        lam=row["lam"],
        raw_s1=row.get("s1"),
        raw_s2=row.get("s2"),
    )
    return format_observation(Arm(row["arm"]), obs)


def prompt_spot_check(
    rows: Sequence[Mapping[str, Any]],
    prompts: Mapping[str, Mapping[str, Any]],
    *,
    system_template: str,
    k: int,
    min_trajectories: int,
) -> dict[str, Any]:
    """II.4.4: for at least ``min_trajectories`` trajectories (the ones with the most rounds,
    then by x0_id), every round's request as actually sent: O_t injected (the formatter's string
    is in the user turn), the decision budget slot rendered with ``k`` (the system prompt is the
    template rendered with k, byte for byte), and every score / threshold shown with five
    significant digits (II.7.1). Each failure is an engineering defect."""
    by_x0: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for r in rows:
        by_x0[r["x0_id"]].append(r)
    order = sorted(by_x0, key=lambda x: (-len(by_x0[x]), x))
    chosen = order[:min_trajectories]
    expected_system = ctrl.render_system_prompt(system_template, k)
    checks: list[dict[str, Any]] = []
    defects: list[str] = []
    for x0 in chosen:
        for r in sorted(by_x0[x0], key=lambda r: int(r["t"])):
            attempts = r.get("attempts") or []
            first = attempts[0].get("prompt_hash") if attempts else None
            sent = prompts.get(first) if first else None
            item: dict[str, Any] = {"x0_id": x0, "t": int(r["t"]), "prompt_hash": first}
            if sent is None:
                item["recorded"] = False
                defects.append(f"{x0} t={r['t']}: request not found in {PROMPTS_SENT}")
                checks.append(item)
                continue
            user, system = str(sent.get("user") or ""), str(sent.get("system") or "")
            o_t = _observation_shown(r)
            numbers = [fixed_significant(float(r[key])) for key in ("s1_tilde", "s2_tilde")]
            numbers += [fixed_significant(float(r[key])) for key in ("tau1", "tau2")]
            item.update(
                {
                    "recorded": True,
                    "o_t_injected": o_t in user,
                    "budget_slot_is_k": system == expected_system,
                    "budget_phrase": f"at most {k} decision" in system,
                    "five_significant_digits": all(n in user for n in numbers),
                    "numbers_shown": numbers,
                    "n_attempts": len(attempts),
                }
            )
            for key, msg in (
                ("o_t_injected", "O_t not injected into the user turn"),
                ("budget_slot_is_k", f"system prompt is not the template rendered with k = {k}"),
                ("five_significant_digits", "a score or threshold is not shown with 5 sig. digits"),
            ):
                if not item[key]:
                    defects.append(f"{x0} t={r['t']}: {msg}")
            checks.append(item)
    return {
        "min_trajectories": min_trajectories,
        "trajectories_checked": chosen,
        "n_requests_checked": len(checks),
        "checks": checks,
        "defects": defects,
    }


def smoke_report(
    rd: Path,
    cfgs: Mapping[str, Any],
    finals: Sequence[Mapping[str, Any]],
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """The II.4.4 (2026-09-25) smoke report. Everything in it is reported, nothing decides
    anything about the chain except the ``engineering_defects`` list; ``valid_for_release`` is
    the one release condition of the formal Exp 2 (:func:`require_smoke_report`)."""
    sc = smoke_cfg(cfgs)
    setup = json.loads((rd / "loop_setup.json").read_text(encoding="utf-8"))
    k = int(setup.get("k", K_MAX_ROUNDS))
    profile = setup.get("smoke_profile") or {}
    # T: the round of the terminal text -- accepted_at, or K + 1 for x_{K+1} (II.8.3)
    t_of = [int(f["accepted_at"]) if f["accepted"] else k + 1 for f in finals]
    t_distribution = {str(t): t_of.count(t) for t in sorted(set(t_of))}
    rj = list(read_jsonl(rd / exp2.ROUND_JUDGMENTS)) if (rd / exp2.ROUND_JUDGMENTS).exists() else []
    per_round: dict[str, dict[str, int]] = {}
    for r in rj:
        cell = per_round.setdefault(str(r["t"]), {"n": 0, "UNSAFE": 0, "SAFE": 0})
        cell["n"] += 1
        cell[str(r["e_label"])] = cell.get(str(r["e_label"]), 0) + 1
    rounds = [int(t) for t in get(sc, "report.evidence_rounds", [1, 2])]
    evidence = evidence_sensitivity.exp2([r for r in rows if int(r["t"]) in rounds])
    reasons: Counter[str] = Counter()
    for r in rows:
        for a in r.get("attempts") or []:
            if a.get("parse_fail_reason"):
                reasons[str(a["parse_fail_reason"])] += 1
    prompts: dict[str, Mapping[str, Any]] = {}
    if (rd / PROMPTS_SENT).exists():
        for p in read_jsonl(rd / PROMPTS_SENT):
            prompts.setdefault(str(p["prompt_hash"]), p)
    system_template = paths.resolve(cfgs["models"]["controller"]["system_prompt_file"]).read_text(
        encoding="utf-8"
    )
    spot = prompt_spot_check(
        rows,
        prompts,
        system_template=system_template,
        k=k,
        min_trajectories=int(get(sc, "report.prompt_spot_check_min_trajectories", 3)),
    )
    defects = list(spot["defects"])
    if len(spot["trajectories_checked"]) < spot["min_trajectories"]:
        defects.append(
            f"only {len(spot['trajectories_checked'])} trajectories to spot-check "
            f"(II.4.4 asks for at least {spot['min_trajectories']})"
        )
    if not (rd / exp2.LOOP_COMPLETE).is_file() or exp2.complete_rounds(rd, k) != list(
        range(1, k + 1)
    ):
        defects.append("the loop did not complete all rounds (resume / round writing)")
    if not rj:
        defects.append(f"{exp2.ROUND_JUDGMENTS} missing: E's round judgments were not made")
    # what makes this run the step 10a smoke of II.4.4, and what would not
    not_valid: list[str] = []
    if setup.get("arms") != ["A1"]:
        not_valid.append(f"arms {setup.get('arms')} (II.4.4: A1 only)")
    if setup.get("always_toxic_stubs"):
        not_valid.append("always-toxic stub verifiers, not the real V1 / V2")
    if setup.get("tau_placeholder"):
        not_valid.append("placeholder taus, not frozen/tau.json")
    if profile.get("verifier_override"):
        not_valid.append("--v1 / --v2 override the configured verifiers")
    configured_n = profile.get("configured_n", int(get(sc, "x0.n", 30)))
    if int(setup.get("n_x1", -1)) != int(configured_n):
        not_valid.append(f"n_x1 = {setup.get('n_x1')} != configured N = {configured_n}")
    numbers = setup.get("numeric_check") or {}
    if not numbers or "not_applicable" in numbers:
        not_valid.append("no numeric check recorded (II.10: 203 states, bit for bit)")
    elif any((v or {}).get("n_different") not in (0, None) for v in numbers.values()):
        not_valid.append("numeric check recorded a difference")
    if defects:
        not_valid.append("engineering defects (fix, then rerun the smoke)")
    return {
        "protocol": "II.4.4 (2026-09-25 revision), 0.5 step 10a",
        "run_id": setup.get("run_id"),
        "smoke_profile": profile,
        "n_trajectories": len(finals),
        "T_distribution": t_distribution,
        "n_non_converged": sum(1 for f in finals if not f["accepted"]),
        "e_judgments_by_round": per_round,
        "n_x_T_failures": sum(1 for f in finals if not f["task_success"]),
        "evidence_sensitivity_rounds": rounds,
        "evidence_sensitivity": evidence,
        "parse_fail": {
            "n_decisions": len(rows),
            "n_parse_fail": sum(1 for r in rows if r["parse_fail"]),
            "attempt_failure_reasons": dict(reasons),
            "note": (
                "II.4.4: classify each -- parsing / injection code = engineering defect; the "
                "model's own refusal or malformed answer = subject behaviour, accounted per II.8.2"
            ),
        },
        "refusal_like": {
            "n_rewrites_flagged": sum(1 for r in rows if r.get("refusal_like")),
            "n_x_T_flagged": sum(1 for f in finals if f.get("x_T_refusal_like")),
        },
        "prompt_spot_check": spot,
        "engineering_defects": defects,
        "not_valid_because": not_valid,
        "valid_for_release": not not_valid,
        "note": (
            "smoke results never enter the proposal or the paper and never change the controller "
            "prompt, the formatter, K, the starting population or the controller's inputs "
            "(App. B.1); the formal Exp 2 needs only valid_for_release = true (II.4.4)"
        ),
    }


def cmd_report(args: argparse.Namespace, cfgs: Mapping[str, Any], echo: Echo) -> int:
    if not args.smoke:
        assert_frozen(cfgs, extra_files=[TEMPLATES_PATH], only_sections=GUARD_SECTIONS)
    rd = existing_run_dir(args.run_id, smoke=args.smoke)
    try:  # II.9.1: no report over a missing or unfinished judge pass
        exp2.require_judge_complete(rd)
    except exp2.JudgeNotCompleteError as exc:
        smoke = " --smoke" if args.smoke else ""
        raise SystemExit(
            f"{rd}: {exc}; run `python pipelines/step12_exp2.py judge --run-id {args.run_id}"
            f"{smoke} --confirm` to the end first"
        ) from exc
    finals = list(read_jsonl(rd / exp2.TRAJECTORIES))
    rows = list(read_jsonl(rd / "decisions.jsonl"))
    refuse_fake(rd, rows, what="step12 report")
    truth = reliability.truth_from_round_judgments(list(read_jsonl(rd / exp2.ROUND_JUDGMENTS)))
    n_filled = reliability.fill_c_t(rows, truth)  # raw s: Exp 2 has no channel
    n_boot, boot_seed = bootstrap_settings(cfgs)
    cut = get(cfgs["exp2"], "analysis.semantic_preservation.cutpoint")
    analysis = exp2_tfr.analyze(
        finals,
        k=K_MAX_ROUNDS,
        seed=boot_seed,
        n_boot=n_boot,
        semantic_cutpoint=None if cut in (None, "TBD") else float(cut),
    )
    report: dict[str, Any] = {
        "smoke": args.smoke,
        "parse_fail": parse_fail.report(rows, by=parse_fail.BY_ROUND),
        "analysis": analysis,
        "reliability": {
            "n_rows_with_c_t": n_filled,
            "n_rows": len(rows),
            "r_t_by_arm": reliability.r_t_by_arm(rows),
        },
        "mechanism": exp2_tfr.mechanism(rows, truth),
        "error_boundary_alignment": error_boundary.exp2(rows, truth),
        # III.1.2 (v1.0): accept rate by arm x round x observed (y1, y2), descriptive
        "evidence_sensitivity": evidence_sensitivity.exp2(rows),
        "controller_cost": cumulative_cost(rd, get(cfgs["models"], "controller.model")),
    }
    labels = panel_labels(rd)
    if labels:
        report["panel_sensitivity"] = exp2_tfr.panel_sensitivity(
            finals,
            labels,
            main_estimate=analysis["primary_contrast"]["estimate"],
            seed=boot_seed,
            n_boot=n_boot,
        )
    if args.exp1_run_dir is not None:
        rows1 = list(read_jsonl(Path(args.exp1_run_dir) / "decisions.jsonl"))
        refuse_fake(Path(args.exp1_run_dir), rows1, what="step12 report --exp1-run-dir")
        report["transport"] = transport.compare(rows1, rows, truth)
    (rd / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=1, default=str),
        encoding="utf-8",
        newline="\n",
    )
    echo(parse_fail.render(report["parse_fail"]))
    echo(error_boundary.render(report["error_boundary_alignment"]))
    echo(evidence_sensitivity.render(report["evidence_sensitivity"]))
    echo(f"[step12] TFR: {json.dumps(analysis['primary_TFR'])}")
    echo(f"[step12] report -> {rd / 'report.json'}")
    if args.smoke:
        sr = smoke_report(rd, cfgs, finals, rows)
        (rd / SMOKE_REPORT).write_text(
            json.dumps(sr, ensure_ascii=False, indent=1, default=str),
            encoding="utf-8",
            newline="\n",
        )
        echo(
            f"[step12] SMOKE report (II.4.4): T distribution {json.dumps(sr['T_distribution'])}, "
            f"non-converged {sr['n_non_converged']}, x_T failures {sr['n_x_T_failures']}, "
            f"parse_fail {sr['parse_fail']['n_parse_fail']}, refusal_like rewrites "
            f"{sr['refusal_like']['n_rewrites_flagged']}; prompt spot-check "
            f"{sr['prompt_spot_check']['n_requests_checked']} requests of "
            f"{len(sr['prompt_spot_check']['trajectories_checked'])} trajectories"
        )
        echo(
            f"[step12] SMOKE engineering defects: {sr['engineering_defects'] or 'none'}; "
            f"valid for release: {sr['valid_for_release']}"
            + ("" if sr["valid_for_release"] else f" ({sr['not_valid_because']})")
        )
        echo(f"[step12] smoke report -> {rd / SMOKE_REPORT}")
    return 0


# ---------------------------------------------------------------------------------------
# the semantic-preservation cutpoint (before step 12)
# ---------------------------------------------------------------------------------------

CUTPOINT_REPORT = "semantic_cutpoint.json"
CUTPOINT_SCORES = "semantic_cutpoint_d_fit.jsonl"
CUTPOINT_LINE = re.compile(r"^(    cutpoint: )TBD$", re.M)


def load_similarity_encoder(scfg: semantic.SemanticConfig) -> semantic.Encoder:
    """LaBSE for the cutpoint (a seam for the tests)."""
    return semantic.load_similarity_model(scfg)


def five_number(values: Sequence[float]) -> dict[str, float]:
    q = np.quantile(np.asarray(values, dtype=float), [0.0, 0.25, 0.5, 0.75, 1.0])
    return dict(zip(("min", "q1", "median", "q3", "max"), (float(x) for x in q), strict=True))


def write_cutpoint(path: Path, value: float) -> None:
    """Replace the one ``    cutpoint: TBD`` line of exp2.yaml with the computed value, keeping
    every comment; the file is read back and must give exactly this float."""
    text = path.read_text(encoding="utf-8")
    new, n = CUTPOINT_LINE.subn(lambda m: m.group(1) + repr(float(value)), text)
    if n != 1:
        raise SystemExit(f"{path}: expected one '    cutpoint: TBD' line, found {n}")
    path.write_text(new, encoding="utf-8", newline="\n")
    back = get(
        load_config("exp2", config_dir=path.parent), "analysis.semantic_preservation.cutpoint"
    )
    if not isinstance(back, float) or back != float(value):
        raise SystemExit(f"{path}: cutpoint reads back as {back!r}, wrote {value!r}")


def cmd_cutpoint(args: argparse.Namespace, cfgs: Mapping[str, Any], echo: Echo) -> int:
    """The preregistered cutpoint of the "low preservation among SAFE" report (the authors
    2026-09-28): the 10% quantile of quality(x0, x1) over the D_fit pairs, x1 = the G_fixed
    state. Reads only D_fit's state file (x0 and x1 text, no label) -- never y*, never D_Exp1
    or an Exp 2 text -- writes the value into configs/exp2.yaml and the report and the
    per-pair scores into data/diagnostics/. Run again, it recomputes and must agree."""
    sp = get(cfgs["exp2"], "analysis.semantic_preservation")
    rule = sp["cutpoint_rule"]
    if rule.get("population") != Split.D_FIT.value:
        raise SystemExit(f"cutpoint_rule.population must be d_fit, got {rule.get('population')!r}")
    q, method = float(rule["quantile"]), str(rule["method"])
    scfg = semantic.build_config(cfgs["exp2"])
    from rafr.logging.snapshot import git_describe, git_dirty, git_head

    # the code state the value comes from, taken before this command edits exp2.yaml
    git = {"git_head": git_head(), "git_dirty": git_dirty(), "git_describe": git_describe()}
    states_file = paths.DATA_POOLS / f"states_{Split.D_FIT.value}.jsonl"
    states = io.load_states(Split.D_FIT)
    echo(f"[step12] cutpoint: {len(states)} D_fit pairs (x0, x1) from {states_file}")
    encoder = load_similarity_encoder(scfg)
    scores = [semantic.pair_quality(encoder, s.x0, s.text) for s in states]
    # the reference's own batching (32 pairs per encode call), as a numerical cross-check
    batched = semantic.similarity(encoder, [s.x0 for s in states], [s.text for s in states])
    max_diff = max((abs(a - b) for a, b in zip(scores, batched, strict=True)), default=0.0)
    cut = float(np.quantile(np.asarray(scores, dtype=float), q, method=method))
    summary = five_number(scores)
    existing = sp.get("cutpoint")
    if existing not in (None, "TBD"):
        if float(existing) != cut:
            raise SystemExit(
                f"configs/exp2.yaml already holds cutpoint {existing!r}; recomputed {cut!r}"
            )
        written = "already in configs/exp2.yaml, recomputed identical"
    else:
        write_cutpoint(paths.CONFIGS / "exp2.yaml", cut)
        written = "written to configs/exp2.yaml"
    out_dir = paths.DATA / "diagnostics"
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / CUTPOINT_SCORES).open("w", encoding="utf-8", newline="\n") as fh:
        for s, v in zip(states, scores, strict=True):
            fh.write(json.dumps({"state_id": s.state_id, "x0_id": s.x0_id, "quality": v}) + "\n")
    report = {
        "rule": "2026-09-28: 10% quantile of quality(x0, x1) over the D_fit pairs",
        "population": Split.D_FIT.value,
        "quantile": q,
        "method": method,
        "n": len(scores),
        "cutpoint": cut,
        "five_number_summary": summary,
        "mean": float(np.mean(scores)) if scores else None,
        "states_file": str(states_file),
        "states_file_sha256": _sha256(states_file),
        "scores_file": str(out_dir / CUTPOINT_SCORES),
        "quality": "SIM, one (x0, x1) pair per encode call (semantic.pair_quality)",
        "reference_batching_max_abs_diff": max_diff,
        "protocol": scfg.describe(),
        "versions": {
            name: semantic.installed_version(name)
            for name in ("sentence-transformers", "transformers", "torch", "scipy", "numpy")
        },
        "labels_read": False,
        **git,
        "created_at": datetime.now(tz=UTC).isoformat(timespec="seconds"),
        "config": written,
    }
    (out_dir / CUTPOINT_REPORT).write_text(
        json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8", newline="\n"
    )
    fmt = ", ".join(f"{k} {v:.5g}" for k, v in summary.items())
    echo(f"[step12] quality over D_fit (n = {len(scores)}): {fmt}")
    echo(f"[step12] cutpoint = {q:g} quantile ({method}) = {cut!r} ({cut:.5g}); {written}")
    echo(
        "[step12] reference batching (32 pairs per call) vs one pair per call: "
        f"max |diff| {max_diff:.3e}"
    )
    echo(f"[step12] report -> {out_dir / CUTPOINT_REPORT}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("command", choices=("x1", "loop", "judge", "panel", "report", "cutpoint"))
    ap.add_argument("--run-id", default=None)
    ap.add_argument("--run-dir", default=None, help="loop: explicit run directory (the pod)")
    ap.add_argument("--confirm", action="store_true", help="required: real API calls cost money")
    ap.add_argument(
        "--smoke", action="store_true", help="x0 from the unlabeled pool; runs/exp2_smoke"
    )
    ap.add_argument("--limit", type=int, default=None, help="x1 --smoke: how many x0")
    ap.add_argument("--x1-file", default=None)
    ap.add_argument("--x1-sha256", default=None)
    ap.add_argument("--records", type=Path, nargs="+", help="the probe's records directories")
    ap.add_argument("--tau-file", default=None, help="loop: frozen/tau.json (as pushed)")
    ap.add_argument(
        "--tau-placeholder",
        type=float,
        nargs="+",
        default=None,
        help="loop --smoke only: one tau for both verifiers, or two (V1 V2)",
    )
    ap.add_argument("--v1", default=None, help="loop --smoke only: a probe config_id")
    ap.add_argument("--v2", default=None, help="loop --smoke only: a probe config_id")
    ap.add_argument(
        "--always-toxic-stubs",
        action="store_true",
        help="loop --smoke only: V1 / V2 = stubs scoring every text 1.0, to push the "
        "trajectories through all K rounds (never formal)",
    )
    ap.add_argument("--cache-dir", default=None, help="loop on the pod: the LLM cache")
    ap.add_argument("--numeric-check", default=None, help="loop: the numeric-ref file (pushed)")
    ap.add_argument("--numeric-check-sha256", default=None, help="loop: its recorded sha256")
    ap.add_argument(
        "--key-from-env",
        action="store_true",
        help="loop (required): OPENAI_API_KEY from the pod's environment (checked for presence "
        "only); the loop builds no Anthropic client",
    )
    ap.add_argument("--exp1-run-dir", type=Path, default=None, help="report: transport")
    return ap


def main(argv: list[str] | None = None, echo: Echo | None = None) -> int:
    args = build_parser().parse_args(argv)
    say = echo or (lambda s: print(s, flush=True))
    cfgs = load_config()
    if args.command == "x1":
        return cmd_x1(args, cfgs, say)
    if args.command == "loop":
        return cmd_loop(args, cfgs, say)
    if args.command == "cutpoint":
        return cmd_cutpoint(args, cfgs, say)
    if not args.run_id:
        raise SystemExit(f"{args.command} needs --run-id")
    if args.command == "judge":
        return cmd_judge(args, cfgs, say)
    if args.command == "panel":
        return cmd_panel(args, cfgs, say)
    return cmd_report(args, cfgs, say)


if __name__ == "__main__":
    sys.exit(main())
