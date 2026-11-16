"""Exp 2's round loop for math (paper Sec. 4.4, App. E), shared by the laptop pipeline and the pod.

One round, as before: the Core round barrier scores every active trajectory before any
decision, then the controller decides for each in turn, then the round is written atomically.
What changed (math entry, 2026-10-12): the barrier scores all active trajectories
together, each V in fixed-shape batches (``score_batch``), instead of one full batch per
trajectory; fixed shape and batch invariance are unchanged, so every score is the one a single
call would give. Score reuse by state is kept: a text already scored for its question
(``digest([x0_id, text])``) is not scored again.

``run_rounds`` takes the scorer as a function, so the same loop runs

* on the laptop inside :class:`pipelines.math_pipeline.MathPipeline` (fixtures and stubs; the
  scores go to the run's ``scores.jsonl``), and
* on the pod (``scripts/gpu/math_exp2_remote.py``; the smoke of step 6 and the formal Exp 2, the
authors
  2026-10-13), where :class:`TextScoreStore` keeps the scores in the loop's own directory and
  loads one V at a time: the two generative V do not fit on one GPU together at their fixed
  shapes, so each round loads V1, scores, frees it, then V2.

E never runs here (paper App. E: after the whole loop, on the laptop; no reference answer on the
pod).

A text longer than a V's L_in (2026-10-13; the loop's L_in is 4,096): truncation is
forbidden, so the trajectory ends there. Its current text -- the rewrite of the previous
round -- is its final text for E; the previous round's history entry carries
``v_input_overlength`` (the round, the V, the token count, L_in), the trajectory makes no
further decision, and the analysis reports it by arm and drops it from the complete-case
sensitivity analysis as it does a parse_fail (paper App. E). Any other text a V refuses
(control-token text) still stops the loop.
"""

from __future__ import annotations

import math
import re
import time
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

from rafr.channel import ObservationBuilder
from rafr.channel.identity import IdentityChannel
from rafr.contracts.schemas import Action, Arm, Lambda, Score
from rafr.controller import decide as controller
from rafr.domains.math.models import ModelInputError
from rafr.domains.math.storage import IntegrityError, digest
from rafr.experiments.common import make_log_record
from rafr.experiments.exp2 import (
    Trajectory,
    assemble_streams,
    complete_rounds,
    load_round,
    round_dir,
    write_round,
)
from rafr.formatter.fields import compute_fields
from rafr.logging.writer import JsonlWriter, read_jsonl

K = 4
OVERLENGTH = "v_input_overlength"
#: The adapter's own refusal of an input longer than L_in (generative.py ``_row``).
_LENGTH = re.compile(r"input has (\d+) tokens; L_in is (\d+)")
#: One item to score: (x0_id, problem, text, key); the key is ``text_key(x0_id, text)``.
ScoreItem = tuple[str, str, str, str]
#: (key -> (V1 row, V2 row), key -> overlength record) for the requested items: every item is
#: in exactly one of the two.
ScoreFn = Callable[
    [Sequence[ScoreItem]],
    tuple[Mapping[str, tuple[Score, Score]], Mapping[str, dict[str, Any]]],
]


def text_key(x0_id: str, text: str) -> str:
    """Durable score reuse is keyed by public text and its question."""
    return digest([x0_id, text])


def input_overlength(verifier: Any, problem: str, text: str) -> dict[str, int] | None:
    """``{"tokens", "L_in"}`` when the (loaded) adapter refuses the text for its length, else
    None. A verifier without the generative input builder never refuses; any other refusal of
    the adapter is raised."""
    build = getattr(verifier, "_row", None)
    if not callable(build):
        return None
    try:
        build(problem, text)
    except ModelInputError as exc:
        found = _LENGTH.search(str(exc))
        if found is None:
            raise
        return {"tokens": int(found[1]), "L_in": int(found[2])}
    return None


def ended(tr: Trajectory) -> bool:
    """The trajectory's current text could not be scored (``v_input_overlength``)."""
    return bool(tr.history) and OVERLENGTH in tr.history[-1]


def score_record(verifier_id: str, version: str, state_id: str, detail: Mapping[str, Any]) -> Score:
    """Validate one verifier output and make its Core Score row."""
    value = detail["error_score"]
    if isinstance(value, bool) or not math.isfinite(value) or not 0 <= value <= 1:
        raise IntegrityError("V must return a finite [0,1] error-direction score")
    steps = detail.get("step_scores")
    return Score(
        state_id=state_id,
        verifier_id=verifier_id,
        model_version=version,
        score=value,
        sub_scores={str(j): score for j, score in enumerate(steps)} if steps else None,
    )


def run_rounds(
    out: Path,
    initial: Sequence[Trajectory],
    *,
    taus: tuple[float, float],
    verifier_ids: tuple[str, str],
    score_texts: ScoreFn,
    client: Any,
    cfg: Any,
    feedback: Callable[..., str],
    namespace: str,
    experiment: str,
    run_id: str,
    seed: int | None,
    smoke: bool,
    k: int = K,
    on_round: Callable[[int], None] | None = None,
) -> list[Trajectory]:
    """Rounds 1..k into ``out/rounds``, resuming after the last complete round; then the
    run's assembled streams. ``initial`` is round 1's trajectories (used only when no round
    is complete yet)."""
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    done = complete_rounds(out, k)
    for t in range(1, k + 1):
        if round_dir(out, t).exists() and t not in done:
            raise IntegrityError("incomplete or changed Core round: inspect before resume")
    trajectories = load_round(out, done[-1]) if done else list(initial)
    builder = ObservationBuilder(IdentityChannel(), taus[0], taus[1], *verifier_ids)
    ledger = getattr(client, "ledger", None)
    for t in range(len(done) + 1, k + 1):
        n_ledger = len(ledger.rows()) if ledger is not None else 0
        decisions, scores = [], []
        active = [tr for tr in trajectories if not tr.accepted and not ended(tr)]
        # Core round barrier: every active trajectory is scored, in batches, before any decision.
        items = [(tr.x0_id, tr.x0, tr.x_t, text_key(tr.x0_id, tr.x_t)) for tr in active]
        scored, overlength = score_texts(items) if items else ({}, {})
        observations = {}
        for tr, (_, _, _, key) in zip(active, items, strict=True):
            if key in overlength:
                # 2026-10-13: the trajectory ends; its current text is its final text
                if not tr.history:
                    raise IntegrityError("a first state is longer than L_in")
                tr.history[-1][OVERLENGTH] = {"t": t, **overlength[key]}
                continue
            rows = scored[key]
            if [row.verifier_id for row in rows] != list(verifier_ids):
                raise IntegrityError("a score row belongs to another verifier")
            # Round streams use the decision's state id so scores and decisions join exactly.
            scores.extend(row.model_copy(update={"state_id": tr.state_id}) for row in rows)
            observations[tr.x0_id] = builder.build(
                tr.state_id, rows[0].score, rows[1].score, Lambda.NONE
            )
        for tr in (tr for tr in active if tr.x0_id in observations):
            obs = observations[tr.x0_id]
            arm = Arm(tr.arm)
            decision = controller.decide(
                client,
                cfg,
                x0=tr.x0,
                x_t=tr.x_t,
                observation=feedback(arm, obs),
                labels=controller.DecisionLabels(tr.x0_id, arm, Lambda.NONE, t, namespace),
            )
            decisions.append(
                make_log_record(
                    run_id=run_id,
                    experiment=experiment,
                    seed=seed,
                    arm=arm,
                    lam=Lambda.NONE,
                    t=t,
                    state_id=tr.state_id,
                    x0_id=tr.x0_id,
                    x_t=tr.x_t,
                    obs=obs,
                    fields=compute_fields(obs),
                    decision=decision,
                )
            )
            tr.history.append(
                {
                    "t": t,
                    "x_t": tr.x_t,
                    "action": decision.action.value if decision.action else None,
                    "parse_fail": decision.parse_fail,
                    "refusal_like": False,
                    "empty_rewrite": decision.empty_rewrite,
                }
            )
            if decision.parse_fail:
                tr.n_parse_fail += 1
                tr.t = t + 1
            elif decision.action is Action.ACCEPT:
                tr.accepted, tr.accepted_at = True, t
            else:
                tr.x_t = decision.revised_text
                tr.t = t + 1
                tr.n_rewrites += 1
        write_round(
            out,
            t,
            trajectories,
            decisions,
            scores,
            ledger.rows()[n_ledger:] if ledger is not None else [],
            meta={"SMOKE": smoke, "domain": "math"},
        )
        if on_round is not None:
            on_round(t)
    assemble_streams(out, k)
    return trajectories


class TextScoreStore:
    """The pod's V scores of loop texts, kept in the loop's directory (``ScoreFn``).

    ``make_verifier(i)`` builds V_i's adapter; it is loaded for one batched call and closed
    straight after (``release`` frees the GPU between the two V). Score rows go to
    ``text_scores.jsonl`` and the full adapter records (tokens, readouts) to
    ``text_score_details.jsonl``, details first; a resume reads the score rows, so a detail row
    without its score row is scored again (the later detail row is the one that counts).
    """

    SCORES = "text_scores.jsonl"
    DETAILS = "text_score_details.jsonl"

    def __init__(
        self,
        out: Path,
        identities: Sequence[tuple[str, str]],
        make_verifier: Callable[[int], Any],
        *,
        release: Callable[[], None] | None = None,
        on_batch: Callable[[int, int, float], None] | None = None,
    ) -> None:
        self.out = Path(out)
        self.out.mkdir(parents=True, exist_ok=True)
        self.identities = [tuple(pair) for pair in identities]
        if len(self.identities) != 2:
            raise IntegrityError("the loop scores with exactly two verifiers")
        self.make_verifier = make_verifier
        self.release = release
        self.on_batch = on_batch
        self.known: dict[tuple[str, str], Score] = {}
        for row in read_jsonl(self.out / self.SCORES):
            score = Score.model_validate(row)
            key = (score.state_id, score.verifier_id)
            if key in self.known:
                raise IntegrityError(f"duplicate loop score row for {key}")
            self.known[key] = score

    def __call__(
        self, items: Sequence[ScoreItem]
    ) -> tuple[dict[str, tuple[Score, Score]], dict[str, dict[str, Any]]]:
        overlength: dict[str, dict[str, Any]] = {}
        for i, (verifier_id, version) in enumerate(self.identities):
            pending: dict[str, tuple[str, str]] = {}
            for _, problem, text, key in items:
                if (key, verifier_id) not in self.known and key not in overlength:
                    pending.setdefault(key, (problem, text))
            if not pending:
                continue
            verifier = self.make_verifier(i)
            outputs: list[dict[str, Any]] = []
            elapsed = 0.0
            try:
                for key, (problem, text) in list(pending.items()):
                    found = input_overlength(verifier, problem, text)
                    if found is not None:
                        overlength[key] = {"verifier_id": verifier_id, **found}
                        del pending[key]
                if pending:
                    started = time.monotonic()
                    outputs = verifier.score_batch(list(pending.values()))
                    elapsed = time.monotonic() - started
            finally:
                close = getattr(verifier, "close", None)
                if callable(close):
                    close()
                if self.release is not None:
                    self.release()
            if len(outputs) != len(pending):
                raise IntegrityError("V returned a different number of scores than texts")
            if not pending:
                continue
            rows = [
                score_record(verifier_id, version, key, detail)
                for key, detail in zip(pending, outputs, strict=True)
            ]
            with JsonlWriter(self.out / self.DETAILS) as writer:
                for key, detail in zip(pending, outputs, strict=True):
                    writer.write({"state_id": key, "verifier_id": verifier_id, **detail})
            with JsonlWriter(self.out / self.SCORES) as writer:
                writer.write_many(rows)
            for row in rows:
                self.known[(row.state_id, verifier_id)] = row
            if self.on_batch is not None:
                self.on_batch(i, len(rows), elapsed)
        scored = {
            key: tuple(self.known[(key, vid)] for vid, _ in self.identities)
            for _, _, _, key in items
            if key not in overlength
        }
        return scored, overlength  # type: ignore[return-value]
