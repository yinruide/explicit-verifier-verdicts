"""Fake data for the analysis dry run (step 10; the analysis code was then frozen, before Exp 1).

NO number produced here is an experimental result. Nothing here reads a real label, score,
threshold, channel file or controller output: every value is drawn from the seeds below.

Two parts:

* :func:`write_fake_runs` -- the step 10 generator. It writes a complete fake Exp 1 run
  directory (D_Exp1 = 300 states x 3 arms x 3 lambda) and a complete fake Exp 2 run directory
  (300 trajectories, 150 per arm, K = 4) with the file layout step11 / step12 write, every row
  validated by the real schemas (``LogRecord``, ``Score``, ``EvalRecord``, ``PanelJudgment``)
  and built with the real pure functions: the channel C_lambda (independent resampling, nested
  coupling, fixed p = 0 / 0.5 / 1) over a FAKE score distribution and FAKE thresholds, the
  formatter, the permuted-B derangement, the Exp 2 balanced assignment, round directories with
  COMPLETE markers, and step12's post-loop judging (``exp2.finish``) with a FAKE E. The
  controller's decisions come from an arm-blind model of the observed pair (y1_obs, y2_obs);
  in Exp 1 the three outcomes of each (state, lambda) are then shuffled across the arm labels,
  in Exp 2 the arm is the seeded balanced permutation -- so every true arm effect is zero by
  construction. parse_fail (all four categories, incl. malformed_json), retries that succeed,
  refusal_like rewrites and panel failures appear on purpose. Every run directory carries the
  FAKE marker and every row a ``FAKE-`` run_id (``analysis._common``); all seeds go to
  ``fake_manifest.json``.
* ``fake_exp1_rows`` / ``fake_exp2_finals`` / ``fake_exp2_rows`` / ``fake_exp2_round_truth``
  -- small in-memory helpers the unit tests use.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from analysis._common import FAKE_MARKER, FAKE_RUN_PREFIX
from rafr.channel import ObservationBuilder
from rafr.channel.degrade import FIXED_P, DegradeChannel, LevelParams
from rafr.channel.identity import IdentityChannel
from rafr.channel.margin_fit import MarginModel
from rafr.contracts.schemas import (  # noqa: F401 (re-export for callers)
    K_MAX_ROUNDS,
    PARSE_MAX_RETRIES,
    Action,
    Arm,
    ConflictDirection,
    ControllerAttempt,
    Decision,
    EvalRecord,
    Lambda,
    LogRecord,
    Observation,
    ParseFailReason,
    SafetyLabel,
    State,
    UncertaintyFields,
    text_hash,
)
from rafr.core.seeding import derive_seed, py_rng
from rafr.core.seeding import rng as seeded_rng
from rafr.evaluator.panel import PanelJudgment, panel_path
from rafr.experiments import exp2
from rafr.experiments.common import make_log_record, score_rows
from rafr.formatter import format_observation
from rafr.formatter.fields import compute_fields
from rafr.formatter.permute import derangement, save_permutation
from rafr.logging.schema import RecordKind, stream_filename
from rafr.logging.writer import JsonlWriter


def fake_exp1_rows(
    n_states: int = 60, seed: int = 0, parse_fail_rate: float = 0.03
) -> list[dict[str, Any]]:
    r = py_rng(seed, "fake_exp1")
    rows: list[dict[str, Any]] = []
    for i in range(n_states):
        sid, x0 = f"fake-{i}-g1", f"fake-{i}"
        e_label = "UNSAFE"  # D_Exp1 holds E-UNSAFE states only
        s1, s2 = r.random(), r.random()
        for lam in ("low", "mid", "high"):
            d = {"low": 0.0, "mid": 0.1, "high": 0.25}[lam]
            t1 = min(1.0, max(0.0, s1 + r.uniform(-d, d)))
            t2 = min(1.0, max(0.0, s2 + r.uniform(-d, d)))
            obs = Observation(
                state_id=sid,
                s1_obs=t1,
                s2_obs=t2,
                tau1=0.5,
                tau2=0.5,
                lam=Lambda(lam),
                raw_s1=s1,
                raw_s2=s2,
            )
            fields: UncertaintyFields = compute_fields(obs)
            for arm in ("A1", "B", "permuted_B"):  # arm labels are meaningless here
                pf = r.random() < parse_fail_rate
                action = None if pf else ("accept" if r.random() < 0.5 else "rewrite")
                rec = LogRecord(
                    run_id="fake", experiment="exp1", seed=seed, arm=arm, lam=lam, t=1,
                    state_id=sid, x0_id=x0, x_t=f"fake text {i}", v1_id="fake-v1", v2_id="fake-v2",
                    s1=s1, s2=s2, s1_tilde=t1, s2_tilde=t2, tau1=0.5, tau2=0.5, fields=fields,
                    attempts=[{"attempt": 0, "raw_output": "{}", "prompt_hash": "h", "seed": seed,
                               "parse_fail_reason": ("action_not_in_accept_rewrite" if pf else None)}],  # noqa: E501
                    action=action, revised_text=None if action != "rewrite" else "fake rewrite",
                    n_parse_retries=2 if pf else 0, parse_fail=pf, refusal_like=False,
                    e_label=e_label, e_score=None, prompt_hash="h", model="fake", model_version="fake-0", decoding={},  # noqa: E501
                )  # fmt: skip
                rows.append(rec.model_dump(mode="json"))
    return rows


def fake_exp2_finals(n_x0: int = 80, seed: int = 0, k: int = 4) -> list[dict[str, Any]]:
    r = py_rng(seed, "fake_exp2")
    finals = []
    for i in range(n_x0):
        arm = "A1" if i % 2 == 0 else "B"
        accepted = r.random() < 0.7
        n_rewrites = r.randint(0, k - 1) if accepted else k
        finals.append(
            {
                "x0_id": f"fake-{i}",
                "x0": f"fake x0 {i}",
                "arm": arm,
                "x_T": f"final {i}",
                "semantic_score": r.random(),  # stands in for quality(x0, x_T)
                "accepted": accepted,
                "accepted_at": n_rewrites + 1 if accepted else None,
                "n_rewrites": n_rewrites,
                "n_parse_fail": int(r.random() < 0.05),
                "first_parse_fail_t": None,
                "refusal_like_rewrites": int(r.random() < 0.03),
                "x_T_refusal_like": r.random() < 0.02,
                "e_label": "SAFE" if r.random() < 0.6 else "UNSAFE",
                "e_score": None,
                "task_success": None,
                "round_e_labels": {},
            }  # fmt: skip
        )
        f = finals[-1]
        f["task_success"] = f["e_label"] == "SAFE"
        n_rounds = (n_rewrites + 1) if accepted else k
        if f["n_parse_fail"]:
            f["first_parse_fail_t"] = r.randint(1, n_rounds)
        # per-round E labels (group 6d): random for t < T, the terminal one matches e_label
        last_t = n_rounds if accepted else k + 1
        for t in range(1, last_t + 1):
            f["round_e_labels"][str(t)] = (
                f["e_label"] if t == last_t else ("SAFE" if r.random() < 0.5 else "UNSAFE")
            )
    return finals


def fake_exp2_round_truth(rows: list[dict[str, Any]], seed: int = 0) -> dict[str, str]:
    """A text -> label map over the decision rows' x_t and revised_text, standing in for
    ``round_judgments.jsonl`` so that c_t / R_t / mechanism code runs on fake data."""
    r = py_rng(seed, "fake_exp2_truth")
    truth: dict[str, str] = {}
    for row in rows:
        for text in (row["x_t"], row.get("revised_text")):
            if text and text_hash(text) not in truth:
                truth[text_hash(text)] = "SAFE" if r.random() < 0.5 else "UNSAFE"
    return truth


def fake_exp2_rows(finals: list[dict[str, Any]], seed: int = 0) -> list[dict[str, Any]]:
    """Decision rows consistent with the finals (for transport / reliability scripts)."""
    r = py_rng(seed, "fake_exp2_rows")
    rows = []
    for f in finals:
        n_rounds = (f["n_rewrites"] + 1) if f["accepted"] else f["n_rewrites"]
        for t in range(1, n_rounds + 1):
            s1, s2 = r.random(), r.random()
            obs = Observation(
                state_id=f"{f['x0_id']}-{f['arm']}-t{t}",
                s1_obs=s1,
                s2_obs=s2,
                tau1=0.5,
                tau2=0.5,
                lam=Lambda.NONE,
                raw_s1=s1,
                raw_s2=s2,
            )
            last = t == n_rounds and f["accepted"]
            rec = LogRecord(
                run_id="fake", experiment="exp2", seed=seed, arm=f["arm"], lam="none", t=t,
                state_id=obs.state_id, x0_id=f["x0_id"], x_t=f"fake x_{t} of {f['x0_id']}", v1_id="fake-v1", v2_id="fake-v2",  # noqa: E501
                s1=s1, s2=s2, s1_tilde=s1, s2_tilde=s2, tau1=0.5, tau2=0.5, fields=compute_fields(obs),  # noqa: E501
                attempts=[{"attempt": 0, "raw_output": "{}", "prompt_hash": "h", "seed": seed}],
                action="accept" if last else "rewrite", revised_text=None if last else "fake",
                n_parse_retries=0, parse_fail=False, refusal_like=False, e_label=None, e_score=None,
                prompt_hash="h", model="fake", model_version="fake-0", decoding={},
            )  # fmt: skip
            rows.append(rec.model_dump(mode="json"))
    return rows


# ======================================================================================
# step 10: complete fake run directories
# ======================================================================================

FAKE_TAU1 = 0.30
FAKE_TAU2 = 0.45
"""Fake thresholds: round numbers on purpose; ``frozen/tau.json`` is never read here."""

FAKE_V1_ID, FAKE_V2_ID = "FAKE-v1", "FAKE-v2"
FAKE_E_SHA256 = "FAKE" + "0" * 60
FAKE_E_VERSION = "FAKE-E"
FAKE_CONTROLLER = "FAKE-controller"
FAKE_PANEL_MEMBERS: tuple[str, ...] = ("fake_member_a", "fake_member_b")
FAKE_POOL_SIZE = 2000
"""Size of the fake 'unlabeled pool' score sample the fake q is built from."""

FAKE_GATE_R: dict[str, dict[str, float]] = {
    "v1": {"low": 0.70, "mid": 0.50, "high": 0.30},
    "v2": {"low": 0.65, "mid": 0.48, "high": 0.31},
}
"""Fake D_gate ladder for the drift table (paper App. C.1, D.1). Invented, not step 9's numbers."""

P_ACCEPT_BY_PAIR: dict[tuple[int, int], float] = {
    (0, 0): 0.60,
    (0, 1): 0.35,
    (1, 0): 0.35,
    (1, 1): 0.15,
}
"""The fake controller: P(accept) given the observed pair (y1_obs, y2_obs) it was shown. The arm
never enters, so every arm effect is zero by construction; the pair does, so the evidence
sensitivity table has something to show."""

P_ACCEPT_PER_ROUND = 0.05
"""Exp 2 only: the fake controller's accept probability rises by this much per round."""
P_PARSE_FAIL = 0.03
P_RETRY_THEN_OK = 0.04
P_REFUSAL_LIKE = 0.05
P_E_SAFE_AFTER_REWRITE = 0.55
P_PANEL_AGREES = 0.85
P_PANEL_NO_VERDICT = 0.02
PARSE_FAIL_REASONS: tuple[ParseFailReason, ...] = tuple(ParseFailReason)
"""All four categories of paper Sec. 4.6, malformed_json (truncation) included."""

SEED_COMPONENTS: tuple[str, ...] = (
    "scores",
    "pool",
    "channel",
    "permutation",
    "arm_shuffle",
    "exp1_decisions",
    "exp1_run",
    "exp2_assignment",
    "exp2_decisions",
    "exp2_run",
    "e",
    "panel",
    "semantic",
    "controller",
    "bootstrap_exp1",
    "bootstrap_exp2",
)


@dataclass(frozen=True)
class FakeSeeds:
    """Every fake component's seed, derived from one base seed (never the formal master)."""

    base: int

    def of(self, component: str) -> int:
        if component not in SEED_COMPONENTS:
            raise ValueError(f"unknown fake seed component {component!r}")
        return derive_seed(self.base, "fake", component)

    def as_dict(self) -> dict[str, int]:
        return {"base": self.base, **{c: self.of(c) for c in SEED_COMPONENTS}}


@dataclass(frozen=True)
class FakeOutcome:
    kind: str  # "accept" | "rewrite" | "parse_fail"
    reason: ParseFailReason | None = None  # parse_fail: the category of every attempt
    retry_reason: ParseFailReason | None = None  # a first attempt that failed, then one parsed
    refusal_like: bool = False


@dataclass
class FakeRuns:
    root: Path
    exp1_dir: Path
    exp2_dir: Path
    gate_report: Path
    manifest: Path
    seeds: FakeSeeds


class _FakeScorer:
    v1_id, v2_id = FAKE_V1_ID, FAKE_V2_ID
    v1_version = v2_version = "FAKE-0"

    def __init__(self, seeds: FakeSeeds) -> None:
        self.seeds = seeds

    def score_pair(self, texts):
        return [fake_pair(self.seeds, t) for t in texts]


def mark_fake(d: Path) -> None:
    d.mkdir(parents=True, exist_ok=True)
    (d / FAKE_MARKER).write_text(
        "FAKE data (protocol 0.5 step 10): written by analysis.fake_data from seeds only; no real "
        "label, score, threshold or controller output. The formal pipeline refuses to read it; "
        "no number here is a result.\n",
        encoding="utf-8",
        newline="\n",
    )


def _fresh(d: Path) -> Path:
    if d.exists() and any(d.iterdir()):
        raise FileExistsError(f"{d} is not empty; fake runs are written once, never appended to")
    mark_fake(d)
    return d


def fake_states(n: int) -> list[State]:
    """Fake D_Exp1: one G_fixed state per fake x0 (paper Sec. 4.2: one analysis state per x0)."""
    return [
        State(
            state_id=f"{FAKE_RUN_PREFIX}s{i:04d}",
            x0_id=f"{FAKE_RUN_PREFIX}x0-{i:04d}",
            x0=f"FAKE original sentence {i:04d}",
            text=f"FAKE first-round rewrite {i:04d}",
            t=1,
        )
        for i in range(n)
    ]


def fake_pair(seeds: FakeSeeds, text: str) -> tuple[float, float]:
    """The fake V1 / V2 scores of a text: a function of the text, as a real verifier's is."""
    r = seeded_rng(seeds.of("scores"), text_hash(text))
    return float(r.beta(2.0, 4.0)), float(r.beta(2.0, 3.0))


def fake_channel(seeds: FakeSeeds) -> DegradeChannel:
    """The real C_lambda (paper Sec. 4.3) with p fixed at 0 / 0.5 / 1 and q from a FAKE pool
    sample."""
    r = seeded_rng(seeds.of("pool"))
    pool1 = sorted(float(x) for x in r.beta(2.0, 4.0, FAKE_POOL_SIZE))
    pool2 = sorted(float(x) for x in r.beta(2.0, 3.0, FAKE_POOL_SIZE))
    q = {
        "v1": MarginModel(FAKE_V1_ID, FAKE_TAU1, pool1),
        "v2": MarginModel(FAKE_V2_ID, FAKE_TAU2, pool2),
    }
    levels = {lam: LevelParams(p, p) for lam, p in FIXED_P.items()}
    return DegradeChannel(levels, q, FAKE_TAU1, FAKE_TAU2, seeds.of("channel"))


def draw_outcome(r, pair: tuple[int, int], shift: float = 0.0) -> FakeOutcome:
    """One arm-blind fake controller decision given the observed pair."""
    if r.random() < P_PARSE_FAIL:
        reason = PARSE_FAIL_REASONS[r.randrange(len(PARSE_FAIL_REASONS))]
        return FakeOutcome("parse_fail", reason=reason)
    retry = None
    if r.random() < P_RETRY_THEN_OK:
        retry = (ParseFailReason.BAD_ACTION, ParseFailReason.MALFORMED_JSON)[r.randrange(2)]
    if r.random() < min(1.0, P_ACCEPT_BY_PAIR[pair] + shift):
        return FakeOutcome("accept", retry_reason=retry)
    return FakeOutcome("rewrite", retry_reason=retry, refusal_like=r.random() < P_REFUSAL_LIKE)


def fake_prompt_hash(experiment: str, x0: str, x_t: str, observation: str, attempt: int) -> str:
    """Stands in for the controller request's hash: equal iff the rendered request would be (the
    system prompt and template are constant, so the request is x0, x_t, O_t and the attempt)."""
    key = "\x1f".join((experiment, x0, x_t, observation, str(attempt)))
    return "FAKE" + hashlib.sha256(key.encode("utf-8")).hexdigest()[:28]


_RAW_FAILURE: dict[ParseFailReason, tuple[str | None, dict[str, Any] | None]] = {
    ParseFailReason.REFUSAL: (None, None),
    ParseFailReason.BAD_ACTION: (
        json.dumps({"action": "maybe", "revised_text": ""}),
        {"action": "maybe", "revised_text": ""},
    ),
    ParseFailReason.EMPTY_REVISED_TEXT: (
        json.dumps({"action": "rewrite", "revised_text": ""}),
        {"action": "rewrite", "revised_text": ""},
    ),
    # a structured output cut off at max_output_tokens
    ParseFailReason.MALFORMED_JSON: (json.dumps({"action": "rewrite"})[:-1] + ', "revi', None),
}


def fake_decision(
    o: FakeOutcome,
    *,
    seeds: FakeSeeds,
    experiment: str,
    seed_key: str,
    arm: Arm,
    lam: Lambda,
    t: int,
    x0: str,
    x_t: str,
    observation: str,
    revised: str | None,
) -> Decision:
    """A ``Decision`` with the attempts paper Sec. 4.6 logs: up to three, each with its own seed."""
    base = derive_seed(seeds.of("controller"), "controller", experiment, seed_key, t)
    attempts: list[ControllerAttempt] = []

    def attempt(a: int, reason: ParseFailReason | None, raw, parsed) -> ControllerAttempt:
        return ControllerAttempt(
            attempt=a,
            raw_output=raw,
            provider_refused=reason is ParseFailReason.REFUSAL,
            parsed=parsed,
            parse_fail_reason=reason,
            prompt_hash=fake_prompt_hash(experiment, x0, x_t, observation, a),
            seed=base if a == 0 else derive_seed(base, "attempt", a),
        )

    if o.kind == "parse_fail":
        assert o.reason is not None
        for a in range(PARSE_MAX_RETRIES + 1):
            attempts.append(attempt(a, o.reason, *_RAW_FAILURE[o.reason]))
        action, text = None, None
    else:
        if o.retry_reason is not None:
            attempts.append(attempt(0, o.retry_reason, *_RAW_FAILURE[o.retry_reason]))
        text = revised if o.kind == "rewrite" else ""
        payload = {"action": o.kind, "revised_text": text}
        attempts.append(attempt(len(attempts), None, json.dumps(payload), payload))
        action = Action(o.kind)
    return Decision(
        state_id=seed_key,
        arm=arm,
        lam=lam,
        t=t,
        action=action,
        revised_text=text,
        parse_fail=o.kind == "parse_fail",
        n_parse_retries=len(attempts) - 1,
        refusal_like=o.kind == "rewrite" and o.refusal_like,
        attempts=attempts,
        model=FAKE_CONTROLLER,
        model_version="FAKE-0",
        decoding={},
    )


def fake_eval_record(text: str, label: SafetyLabel) -> EvalRecord:
    return EvalRecord(
        text_id=text_hash(text),
        text=text,
        label=label,
        reason="FAKE",
        model="FAKE-E",
        model_version="FAKE-E",
        prompt_hash="FAKE" + text_hash(text),
        e_prompt_sha256=FAKE_E_SHA256,
        e_version=FAKE_E_VERSION,
        decoding={},
    )


def _write_json(path: Path, obj: Any) -> None:
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=1), encoding="utf-8", newline="\n")


def _rewrite_text(o: FakeOutcome, who: str) -> str | None:
    if o.kind != "rewrite":
        return None
    if o.refusal_like:
        return f"FAKE refusal-like reply ({who}): I can't help with rewriting this."
    return f"FAKE rewrite ({who})"


def write_fake_exp1_run(rd: Path, seeds: FakeSeeds, states: list[State]) -> dict[str, EvalRecord]:
    """A fake step 11 run directory; returns the fake outcome (E's judgment of every state)."""
    _fresh(rd)
    run_id, run_seed = f"{FAKE_RUN_PREFIX}exp1-{seeds.base}", seeds.of("exp1_run")
    scorer = _FakeScorer(seeds)
    raw = {s.state_id: fake_pair(seeds, s.text) for s in states}
    with JsonlWriter(rd / stream_filename(RecordKind.SCORE)) as w:
        w.write_many(score_rows(scorer, list(raw), list(raw.values())))
    # the outcome: D_Exp1 is E-UNSAFE by construction (paper Sec. 4.2), so the fake E says UNSAFE
    outcome = {s.state_id: fake_eval_record(s.text, SafetyLabel.UNSAFE) for s in states}
    jpath = rd / "e_judgments.jsonl"
    with JsonlWriter(jpath) as w:
        w.write_many(outcome[s.state_id] for s in states)
    _write_json(
        rd / "e_judgments_hash.json",
        {
            "file": str(jpath),
            "sha256": hashlib.sha256(jpath.read_bytes()).hexdigest(),
            "n": len(outcome),
            "source": "FAKE E (step 10): every fake D_Exp1 state UNSAFE by construction",
            "e_identity": {"prompt_sha256": FAKE_E_SHA256, "version": FAKE_E_VERSION},
        },
    )
    channel = fake_channel(seeds)
    builder = ObservationBuilder(channel, FAKE_TAU1, FAKE_TAU2, FAKE_V1_ID, FAKE_V2_ID)
    mapping = derangement([s.state_id for s in states], seeds.of("permutation"))
    save_permutation(mapping, seeds.of("permutation"), rd / stream_filename(RecordKind.PERMUTATION))
    arms = [Arm.A1, Arm.B, Arm.PERMUTED_B]
    with (
        JsonlWriter(rd / stream_filename(RecordKind.CHANNEL)) as ch,
        JsonlWriter(rd / stream_filename(RecordKind.DECISION)) as dec,
    ):
        for lam in (Lambda.LOW, Lambda.MID, Lambda.HIGH):
            obs, fields = {}, {}
            for s in states:
                sid = s.state_id
                obs[sid] = builder.build(sid, *raw[sid], lam)
                fields[sid] = compute_fields(obs[sid])
                ch.write(
                    {
                        "state_id": sid,
                        "lam": lam.value,
                        "s1": raw[sid][0],
                        "s2": raw[sid][1],
                        "s1_tilde": obs[sid].s1_obs,
                        "s2_tilde": obs[sid].s2_obs,
                        **builder.assignment(sid, lam),
                    }
                )
            for s in states:
                sid = s.state_id
                pair = (fields[sid].y1, fields[sid].y2)
                r = py_rng(seeds.of("exp1_decisions"), sid, lam.value)
                outcomes = [draw_outcome(r, pair) for _ in arms]
                # the arm labels, shuffled: which of the three arm-blind outcomes each arm gets
                order = list(arms)
                py_rng(seeds.of("arm_shuffle"), sid, lam.value).shuffle(order)
                by_arm = dict(zip(order, outcomes, strict=True))
                for arm in arms:
                    donor = mapping[sid] if arm is Arm.PERMUTED_B else None
                    observation = format_observation(
                        arm, obs[sid], permuted_fields=fields[donor] if donor else None
                    )
                    o = by_arm[arm]
                    decision = fake_decision(
                        o, seeds=seeds, experiment="exp1", seed_key=sid, arm=arm, lam=lam,
                        t=s.t, x0=s.x0, x_t=s.text, observation=observation,
                        revised=_rewrite_text(o, f"{sid}, {arm.value}, {lam.value}"),
                    )  # fmt: skip
                    dec.write(
                        make_log_record(
                            run_id=run_id, experiment="exp1", seed=run_seed, arm=arm, lam=lam,
                            t=s.t, state_id=sid, x0_id=s.x0_id, x_t=s.text, obs=obs[sid],
                            fields=fields[sid], decision=decision, permuted_from=donor,
                            e_label=outcome[sid].label, e_score=None,
                        )
                    )  # fmt: skip
    _write_json(
        rd / "fake_setup.json",
        {
            "FAKE": True,
            "run_id": run_id,
            "run_seed": run_seed,
            "n_states": len(states),
            "arms": [a.value for a in arms],
            "lambdas": ["low", "mid", "high"],
            "tau": {"v1": FAKE_TAU1, "v2": FAKE_TAU2},
            "p": {lam.value: p for lam, p in FIXED_P.items()},
            "channel_seed": seeds.of("channel"),
            "permutation_seed": seeds.of("permutation"),
            "arm_shuffle": {
                "seed": seeds.of("arm_shuffle"),
                "rule": (
                    "per (state, lambda): py_rng(seed, state_id, lambda).shuffle([A1, B, "
                    "permuted_B]) assigns the three arm-blind outcomes to the arm labels"
                ),
            },
        },
    )
    return outcome


def _trajectory_step(tr: exp2.Trajectory, t: int, decision: Decision) -> None:
    """The trajectory update of ``rafr.experiments.exp2.run_loop`` (paper Sec. 4.6, 4.4)."""
    tr.history.append(
        {
            "t": t,
            "x_t": tr.x_t,
            "action": decision.action.value if decision.action else None,
            "parse_fail": decision.parse_fail,
            "refusal_like": decision.refusal_like,
            "empty_rewrite": decision.empty_rewrite,
        }
    )
    if decision.parse_fail:
        tr.n_parse_fail += 1
        tr.t = t + 1  # x_{t+1} = x_t; the round is consumed
    elif decision.action is Action.ACCEPT:
        tr.accepted, tr.accepted_at = True, t
    else:
        assert decision.revised_text is not None
        tr.n_rewrites += 1
        tr.refusal_like_rewrites += int(decision.refusal_like)
        tr.x_t_refusal_like = bool(decision.refusal_like)
        tr.x_t = decision.revised_text
        tr.t = t + 1


def _fake_panel(rd: Path, seeds: FakeSeeds, finals: list[dict[str, Any]]) -> None:
    """``panel_<member>.jsonl`` as step12 ``panel`` writes it: one row per distinct x_T."""
    x_t_label = {text_hash(f["x_T"]): (f["x_T"], f["e_label"]) for f in finals}
    for member in FAKE_PANEL_MEMBERS:
        with JsonlWriter(panel_path(rd, member)) as w:
            for tid, (text, e_label) in x_t_label.items():
                r = py_rng(seeds.of("panel"), member, tid)
                label: SafetyLabel | None = SafetyLabel(e_label)
                if r.random() >= P_PANEL_AGREES:
                    label = SafetyLabel.SAFE if label is SafetyLabel.UNSAFE else SafetyLabel.UNSAFE
                failures: list[str] = []
                if r.random() < P_PANEL_NO_VERDICT:
                    label, failures = None, ["FAKE: no verdict"] * (PARSE_MAX_RETRIES + 1)
                w.write(
                    PanelJudgment(
                        member=member, text_id=tid, text=text, label=label, reason="FAKE",
                        failures=failures, model=f"FAKE-{member}", version="FAKE",
                        model_version="FAKE", prompt_sha256=FAKE_E_SHA256,
                        prompt_hash="FAKE" + tid, decoding={},
                    )
                )  # fmt: skip


def write_fake_exp2_run(
    rd: Path, seeds: FakeSeeds, states: list[State], prior: dict[str, EvalRecord]
) -> dict[str, Any]:
    """A fake step 12 run directory: loop, post-loop E judgments, finals and panel files."""
    _fresh(rd)
    run_id, run_seed = f"{FAKE_RUN_PREFIX}exp2-{seeds.base}", seeds.of("exp2_run")
    arms = [Arm.A1, Arm.B]
    k = K_MAX_ROUNDS
    assign_seed = seeds.of("exp2_assignment")
    assignment = exp2.balanced_assignment([s.x0_id for s in states], arms, assign_seed)
    with JsonlWriter(rd / "assignment.jsonl") as w:
        for x0_id, arm in assignment.items():
            w.write({"x0_id": x0_id, "arm": arm.value, "seed": assign_seed})
    trajs = [
        exp2.Trajectory(x0_id=s.x0_id, x0=s.x0, arm=assignment[s.x0_id].value, t=1, x_t=s.text)
        for s in states
    ]
    scorer = _FakeScorer(seeds)
    builder = ObservationBuilder(IdentityChannel(), FAKE_TAU1, FAKE_TAU2, FAKE_V1_ID, FAKE_V2_ID)
    for t in range(1, k + 1):
        active = [tr for tr in trajs if not tr.accepted]
        decisions: list[LogRecord] = []
        pairs = scorer.score_pair([tr.x_t for tr in active])
        scores = score_rows(scorer, [tr.state_id for tr in active], pairs)
        for tr, (s1, s2) in zip(active, pairs, strict=True):
            obs = builder.build(tr.state_id, s1, s2, Lambda.NONE)
            fields = compute_fields(obs)
            arm = Arm(tr.arm)
            observation = format_observation(arm, obs)
            # arm-blind: the draw is keyed by the x0 and the round, never by the arm
            r = py_rng(seeds.of("exp2_decisions"), tr.x0_id, t)
            o = draw_outcome(r, (fields.y1, fields.y2), shift=P_ACCEPT_PER_ROUND * (t - 1))
            decision = fake_decision(
                o, seeds=seeds, experiment="exp2", seed_key=tr.x0_id, arm=arm, lam=Lambda.NONE,
                t=t, x0=tr.x0, x_t=tr.x_t, observation=observation,
                revised=_rewrite_text(o, f"{tr.x0_id}, round {t}"),
            )  # fmt: skip
            decisions.append(
                make_log_record(
                    run_id=run_id, experiment="exp2", seed=run_seed, arm=arm, lam=Lambda.NONE,
                    t=t, state_id=tr.state_id, x0_id=tr.x0_id, x_t=tr.x_t, obs=obs,
                    fields=fields, decision=decision,
                )
            )  # fmt: skip
            _trajectory_step(tr, t, decision)
        info = {"n_active": len(active), "n_accepted_total": sum(tr.accepted for tr in trajs)}
        exp2.write_round(rd, t, trajs, decisions, scores, (), info)
    exp2.assemble_streams(rd, k)
    _write_json(
        rd / exp2.LOOP_COMPLETE,
        {"k": k, "n_trajectories": len(trajs), "rounds": exp2.complete_rounds(rd, k)},
    )

    def fake_e(text: str) -> EvalRecord:
        u = py_rng(seeds.of("e"), text_hash(text)).random()
        safe = u < P_E_SAFE_AFTER_REWRITE
        return fake_eval_record(text, SafetyLabel.SAFE if safe else SafetyLabel.UNSAFE)

    def fake_quality(x0: str, x_t: str) -> float:
        return py_rng(seeds.of("semantic"), text_hash(x0), text_hash(x_t)).random()

    setup = exp2.Exp2Setup(
        run_id=run_id, run_dir=rd, x1_states=states, scorer=scorer, tau1=FAKE_TAU1,
        tau2=FAKE_TAU2, channel=IdentityChannel(), arms=arms, k=k, assignment_seed=assign_seed,
        run_seed=run_seed, judge_fn=fake_e,
        # t = 1 reuses the (fake) outcome of Exp 1, as step 12 reuses step 5's y* (paper Sec. 4.4)
        prior_judgments={text_hash(s.text): prior[s.state_id] for s in states},
        quality_fn=fake_quality,
    )  # fmt: skip
    summary = exp2.finish(setup, trajs)
    finals = [json.loads(x) for x in (rd / "trajectories.jsonl").read_text("utf-8").splitlines()]
    _fake_panel(rd, seeds, finals)
    _write_json(
        rd / "loop_setup.json",
        {
            "FAKE": True,
            "run_id": run_id,
            "arms": [a.value for a in arms],
            "k": k,
            "n_x1": len(states),
            "tau1": FAKE_TAU1,
            "tau2": FAKE_TAU2,
            "assignment_seed": assign_seed,
            "run_seed": run_seed,
            "panel_members": list(FAKE_PANEL_MEMBERS),
        },
    )
    return summary


def write_fake_gate_report(path: Path) -> Path:
    """A fake step 9 report with the keys the analysis reads: ``r`` and ``r_ref``."""
    _write_json(
        path,
        {
            "FAKE": True,
            "split": "d_gate",
            "population": "e_unsafe",
            "passed": True,
            "r": FAKE_GATE_R,
            "r_ref": {
                "split": "d_gate",
                "lambda": "none",
                "population": "e_unsafe",
                "by_verifier": {"v1": 0.70, "v2": 0.65},
                "n": 100,
                "accuracy_whole_split": {"v1": 0.72, "v2": 0.74},
            },
        },
    )
    return path


def write_fake_runs(root: Path, *, seed: int, n_states: int = 300) -> FakeRuns:
    """The whole step 10 input: fake Exp 1 + Exp 2 run directories, a fake gate report and the
    manifest of every seed. ``root`` gets the FAKE marker too."""
    root = Path(root)
    mark_fake(root)
    seeds = FakeSeeds(int(seed))
    states = fake_states(n_states)
    exp1_dir, exp2_dir = root / "FAKE_exp1_run", root / "FAKE_exp2_run"
    outcome = write_fake_exp1_run(exp1_dir, seeds, states)
    summary2 = write_fake_exp2_run(exp2_dir, seeds, states, outcome)
    gate = write_fake_gate_report(root / "FAKE_gate_report.json")
    manifest = root / "fake_manifest.json"
    pairs = {f"({a},{b})": p for (a, b), p in P_ACCEPT_BY_PAIR.items()}
    _write_json(
        manifest,
        {
            "FAKE": True,
            "note": (
                "fake data for the step 10 analysis dry run; drawn from these seeds only, no real "
                "label, score, threshold or controller output was read; arm effects are zero by "
                "construction"
            ),
            "seeds": seeds.as_dict(),
            "n_states_exp1": n_states,
            "n_trajectories_exp2": n_states,
            "exp2_arm_sizes": {a: v["n"] for a, v in summary2["arms"].items()},
            "exp1_arm_labels": "shuffled per (state, lambda) with seeds.arm_shuffle",
            "exp2_arm_labels": "balanced permutation with seeds.exp2_assignment (assignment.jsonl)",
            "tau": {"v1": FAKE_TAU1, "v2": FAKE_TAU2},
            "decision_model": {
                "p_accept_by_observed_pair": pairs,
                "exp2_accept_shift_per_round": P_ACCEPT_PER_ROUND,
                "p_parse_fail": P_PARSE_FAIL,
                "p_retry_then_ok": P_RETRY_THEN_OK,
                "p_refusal_like_among_rewrites": P_REFUSAL_LIKE,
            },
            "fake_e": {"p_safe_after_rewrite": P_E_SAFE_AFTER_REWRITE, "t1": "UNSAFE (reused)"},
            "fake_panel": {
                "members": list(FAKE_PANEL_MEMBERS),
                "p_agrees_with_e": P_PANEL_AGREES,
                "p_no_verdict": P_PANEL_NO_VERDICT,
            },
            "fake_gate_r": FAKE_GATE_R,
            "files": {"exp1": str(exp1_dir), "exp2": str(exp2_dir), "gate_report": str(gate)},
        },
    )
    return FakeRuns(root, exp1_dir, exp2_dir, gate, manifest, seeds)
