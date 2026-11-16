"""Math step 10 (step 10): Exp 1's inputs and the controller calls they
make, and the primary analysis run end to end on FAKE data whose arm labels are shuffled.
No API call, no GPU, nothing written into the run (``runs/math_formal_n9000`` is read only).

    python pipelines/math_step10.py inputs   # D_Exp1 x {A1, B, permuted-B} x {low, mid, high}
    python pipelines/math_step10.py cost     # Exp 1's controller calls and cost, from `inputs`
    python pipelines/math_step10.py fake     # report.json's analyses on FAKE, arms shuffled

``inputs`` builds every Exp 1 cell as ``MathPipeline.exp1`` does: the D_Exp1 states (300, all
E-INCORRECT by construction of the split; no label is read here), their step 6 scores
(``scores.jsonl``), the frozen channel (``fit.json``: tau, q, p = 0 / 0.5 / 1), the permuted-B
donors (seed component ``permuted_b``), the observation each arm shows and the controller request
(``build_controller_config``, K = 1). The controller is answered by a recording fake provider,
so the prompts and their hashes are the formal ones and nothing leaves the laptop; byte-identical
requests (same per-state seed, same prompt) hit the local cache as they will in the formal run
so the uncached requests are exactly the live controller calls Exp 1 makes at its first
attempts (a parse retry adds one, paper Sec. 4.6). The cells go to
``data/math/step10/exp1_inputs.jsonl``.

``fake`` takes those cells, gives each a FAKE decision, shuffles the three decisions of every
(state, lambda) over the arms A1 / B / permuted-B, and builds FAKE Exp 2 trajectories (D_Exp1's 300
x0, 150 per arm, arms shuffled; parse_fail, unresolved and v_input_overlength included); then it
runs what ``MathPipeline.report`` runs -- ``summarize_exp1`` (primary, sensitivities (a) and (b)),
``summarize_exp2`` (primary, (a), (b), (c), overlength, unresolved rates), ``exp1_diagnostics``,
``exp2_reliability_diagnostics`` and ``protocol_outcome_diagnostics`` against the real gate's R --
and writes ``data/math/step10/fake_report.json`` marked FAKE. With shuffled arms every arm
contrast is null by construction; the run shows that each primary analysis runs on data of
the formal shape, not any result.
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
import sys
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pipelines import math_step06 as s6  # noqa: E402
from rafr.channel import ObservationBuilder  # noqa: E402
from rafr.channel.degrade import LADDER  # noqa: E402
from rafr.contracts.schemas import Arm, Split  # noqa: E402
from rafr.controller import decide as controller  # noqa: E402
from rafr.core import paths  # noqa: E402
from rafr.core.seeding import component_seed, derive_seed  # noqa: E402
from rafr.domains.math.generation import build_controller_config  # noqa: E402
from rafr.domains.math.storage import digest, read_json  # noqa: E402
from rafr.formatter.fields import compute_fields  # noqa: E402
from rafr.formatter.permute import derangement  # noqa: E402

ARMS = (Arm.A1, Arm.B, Arm.PERMUTED_B)
EXPERIMENT = "math_exp1"
#: Fixed seed component of the FAKE data (never a seed of a formal step).
FAKE_COMPONENT = "math_step10_fake"
CONTROLLER_MODEL = "gpt-5.6-terra"


STEP10 = paths.DATA / "math" / "step10"


class Step10Paths(s6.Paths):
    """Step 6's paths plus this step's folder; tests point both at a temporary tree."""

    def __init__(self, *, step10: Path = STEP10, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.step10 = Path(step10)

    @property
    def exp1_inputs(self) -> Path:
        return self.step10 / "exp1_inputs.jsonl"

    @property
    def summary(self) -> Path:
        return self.step10 / "exp1_inputs_summary.json"

    @property
    def fake_report(self) -> Path:
        return self.step10 / "fake_report.json"


def _request_text(request: Any) -> str:
    """The prompt bytes of one controller request (system and messages)."""
    from rafr.llm.client import content_text

    parts = [request.system or ""]
    parts += [content_text(m["content"]) for m in request.messages]
    return "\n".join(parts)


# ---------------------------------------------------------------------------------------
# inputs: every Exp 1 cell, built as MathPipeline.exp1 builds it
# ---------------------------------------------------------------------------------------


def exp1_cells(pipeline: Any, work: Path, count_tokens: Any = None) -> list[dict[str, Any]]:
    from rafr.llm.cache import LLMCache
    from rafr.llm.client import LLMClient
    from rafr.llm.ledger import Ledger
    from rafr.llm.providers.fake import FakeProvider

    config = pipeline.config
    by_question = pipeline._states()  # noqa: SLF001
    qids = pipeline._split_ids(Split.D_EXP1)  # noqa: SLF001
    if len(qids) != config.experiment_n:
        raise SystemExit(f"D_Exp1 holds {len(qids)} states, not {config.experiment_n}")
    states = [by_question[q] for q in qids]
    channel = pipeline._channel()  # noqa: SLF001
    ids = [s.state_id for s in states]
    donors = derangement(ids, component_seed(config.seed, "permuted_b"))
    pairs = dict(zip(ids, pipeline._initial_pairs(qids), strict=True))  # noqa: SLF001
    builder = ObservationBuilder(
        channel,
        channel.tau1,
        channel.tau2,
        pipeline._verifier_identity(0)[0],  # noqa: SLF001
        pipeline._verifier_identity(1)[0],  # noqa: SLF001
    )
    cfg = build_controller_config(config.seed, 1)
    recorder = FakeProvider()
    client = LLMClient(
        providers={"openai": recorder},
        cache=LLMCache(work / "cache"),
        ledger=Ledger(work / "ledger.jsonl"),
    )
    rows = []
    for state in states:
        sid = state.state_id
        for level in LADDER:
            obs = builder.build(sid, *pairs[sid], level)
            donor = builder.build(donors[sid], *pairs[donors[sid]], level)
            for arm in ARMS:
                permuted = arm is Arm.PERMUTED_B
                feedback = pipeline.feedback(
                    arm,
                    obs,
                    **(
                        {"donor_observation": donor, "permuted_fields": compute_fields(donor)}
                        if permuted
                        else {}
                    ),
                )
                before = len(recorder.calls)
                decision = controller.decide(
                    client,
                    cfg,
                    x0=state.x0,
                    x_t=state.text,
                    observation=feedback,
                    labels=controller.DecisionLabels(sid, arm, level, 1, EXPERIMENT),
                )
                live = len(recorder.calls) > before
                text = _request_text(recorder.calls[-1]) if live else None
                rows.append(
                    {
                        "state_id": sid,
                        "x0_id": state.x0_id,
                        "lam": level.value,
                        "arm": arm.value,
                        "candidate_hash": digest(state.text),
                        "s1": pairs[sid][0],
                        "s2": pairs[sid][1],
                        "s1_tilde": obs.s1_obs,
                        "s2_tilde": obs.s2_obs,
                        "tau1": channel.tau1,
                        "tau2": channel.tau2,
                        "permuted_from": donor.state_id if permuted else None,
                        "channel_assignment": {
                            "state_id": sid,
                            "lam": level.value,
                            "arm": arm.value,
                            **builder.assignment(sid, level),
                        },
                        "prompt_hash": decision.attempts[-1].prompt_hash,  # as make_log_record
                        "observation": feedback,
                        "live": live,
                        "prompt_chars": len(text) if text is not None else None,
                        "prompt_tokens_approx": count_tokens(text)
                        if count_tokens and text is not None
                        else None,
                    }
                )
    return rows


def token_counter(paths: s6.Paths) -> Any:
    """Approximate input tokens: R1-Distill-Llama's tokenizer (Llama 3's 128k BPE), the closest
    local stand-in for the controller's own; no OpenAI tokenizer on the laptop."""
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(str(paths.tokenizer("r1_llama_8b")))
    return lambda text: len(tok(text, add_special_tokens=False)["input_ids"])


def inputs(paths: Step10Paths) -> dict[str, Any]:
    pipeline = s6.open_pipeline(paths)
    with tempfile.TemporaryDirectory() as tmp:
        rows = exp1_cells(pipeline, Path(tmp), token_counter(paths))
    digest_ = s6.write_rows(paths.exp1_inputs, rows)
    summary = summarize_inputs(rows)
    summary["exp1_inputs_sha256"] = digest_
    s6.write_json(paths.summary, summary)
    return summary


def summarize_inputs(rows: list[dict[str, Any]]) -> dict[str, Any]:
    cells = Counter((r["arm"], r["lam"]) for r in rows)
    live = [r for r in rows if r["live"]]
    states = sorted({r["state_id"] for r in rows})
    by_cell: dict[tuple[str, str], dict[str, Any]] = {}
    for r in rows:
        by_cell.setdefault((r["state_id"], r["lam"]), {})[r["arm"]] = r
    shared = all(
        len({(c["s1_tilde"], c["s2_tilde"], c["s1"], c["s2"]) for c in arms.values()}) == 1
        for arms in by_cell.values()
    )
    perturbed = Counter()
    for r in rows:
        if r["arm"] == Arm.A1.value:
            a = r["channel_assignment"]
            perturbed[(r["lam"], "v1")] += bool(a["perturbed_v1"])
            perturbed[(r["lam"], "v2")] += bool(a["perturbed_v2"])
    tokens = [r["prompt_tokens_approx"] for r in live if r["prompt_tokens_approx"] is not None]
    return {
        "n_states": len(states),
        "n_cells": len(rows),
        "cells_per_arm_level": {f"{a}/{lam}": n for (a, lam), n in sorted(cells.items())},
        "observed_scores_shared_across_arms_within_level": shared,
        "donors_never_self": all(
            r["permuted_from"] not in (None, r["state_id"])
            for r in rows
            if r["arm"] == Arm.PERMUTED_B.value
        ),
        "perturbed_states": {f"{lam}/{v}": n for (lam, v), n in sorted(perturbed.items())},
        "distinct_prompts": len({r["prompt_hash"] for r in rows}),
        "live_first_attempts": len(live),
        "cached_first_attempts": len(rows) - len(live),
        "live_by_arm_level": {
            f"{a}/{lam}": sum(1 for r in live if (r["arm"], r["lam"]) == (a, lam))
            for (a, lam) in sorted(cells)
        },
        "prompt_tokens_approx": {
            "tokenizer": "R1-Distill-Llama (Llama 3 BPE), a stand-in for the controller's",
            "sum": sum(tokens),
            "mean": statistics.fmean(tokens) if tokens else None,
            "max": max(tokens) if tokens else None,
        },
    }


# ---------------------------------------------------------------------------------------
# cost: Exp 1's controller calls
# ---------------------------------------------------------------------------------------


def g_fixed_outputs(paths: s6.Paths) -> dict[str, float]:
    """Live G_fixed output tokens of the formal run (same backbone, effort and output cap): the
    proxy for a controller decision's output (reasoning plus JSON; a rewrite carries a solution)."""
    rows = [
        json.loads(line)
        for line in (paths.run / "ledger.jsonl").read_text(encoding="utf-8").splitlines()
        if line
    ]
    out = sorted(
        int(r.get("output_tokens") or 0)
        for r in rows
        if r.get("purpose") == "g_fixed" and not r.get("cached")
    )
    if not out:
        raise SystemExit("no live G_fixed call in the run's ledger")
    return {
        "n": len(out),
        "mean": statistics.fmean(out),
        "p90": out[int(0.9 * (len(out) - 1))],
        "max": out[-1],
    }


def cost(paths: Step10Paths) -> dict[str, Any]:
    from rafr.core.runtime import prices_for

    summary = read_json(paths.summary)
    price = prices_for(CONTROLLER_MODEL)
    calls = summary["live_first_attempts"]
    tokens_in = summary["prompt_tokens_approx"]["sum"]
    outputs = g_fixed_outputs(paths)
    scenarios = {}
    for name, per_call in (("g_fixed_mean", outputs["mean"]), ("g_fixed_p90", outputs["p90"])):
        usd = (tokens_in * price.input_ + calls * per_call * price.output) / 1e6
        scenarios[name] = {"output_tokens_per_call": round(per_call), "usd": round(usd, 2)}
    retry = {
        "parse_retries_at_2pct": round(0.02 * calls),
        "note": "a parse retry resends the request plus a fixed prompt (II.8.2); detox's Exp 1 "
        "had none",
    }
    result = {
        "controller": CONTROLLER_MODEL,
        "prices_per_mtok": {"input": price.input_, "output": price.output},
        "decisions": summary["n_cells"],
        "live_calls_first_attempts": calls,
        "cached_by_identical_prompt": summary["cached_first_attempts"],
        "input_tokens_approx": tokens_in,
        "g_fixed_outputs": outputs,
        "scenarios": scenarios,
        "retries": retry,
        "pod": "none: Exp 1 reads the step 6 scores and the frozen channel; no V is loaded",
    }
    s6.write_json(paths.step10 / "exp1_cost.json", result)
    return result


# ---------------------------------------------------------------------------------------
# fake: the primary analyses on FAKE data with shuffled arm labels
# ---------------------------------------------------------------------------------------


def fake_exp1_rows(cells: list[dict[str, Any]], rng: random.Random) -> list[dict[str, Any]]:
    """One FAKE decision per cell; the three decisions of each (state, lambda) shuffled over
    the arms, so no arm differs from another but by chance. Row shape as MathPipeline.report."""
    by_slot: dict[tuple[str, str], dict[str, dict[str, Any]]] = {}
    for c in cells:
        by_slot.setdefault((c["state_id"], c["lam"]), {})[c["arm"]] = c
    rows = []
    for (_, lam), arms in sorted(by_slot.items()):
        flagged = [
            arms[Arm.A1.value][f"s{i}_tilde"] >= arms[Arm.A1.value][f"tau{i}"] for i in (1, 2)
        ]
        p_accept = 0.7 - 0.25 * sum(flagged) + {"low": 0.0, "mid": 0.05, "high": 0.1}[lam]
        decisions = []
        for _ in ARMS:
            if rng.random() < 0.01:
                decisions.append((None, True))
            else:
                decisions.append(("accept" if rng.random() < p_accept else "rewrite", False))
        rng.shuffle(decisions)
        for arm, (action, parse_fail) in zip(ARMS, decisions, strict=True):
            c = arms[arm.value]
            attempt = {"prompt_hash": c["prompt_hash"]}
            rows.append(
                {
                    "domain": "math",
                    "stage": "development",
                    "execution_mode": "fake",
                    "run_id": "math_step10_FAKE",
                    "question_id": c["x0_id"],
                    "initial_status": "incorrect",
                    "arm": arm.value,
                    "level": lam,
                    "action": action,
                    "parse_fail": parse_fail,
                    "candidate_hash": c["candidate_hash"],
                    "input_scores": [c["s1"], c["s2"]],
                    "observed_scores": [c["s1_tilde"], c["s2_tilde"]],
                    "thresholds": [c["tau1"], c["tau2"]],
                    "prompt_hash": c["prompt_hash"],
                    "attempts": [attempt] * (3 if parse_fail else 1),
                    "channel_assignment": c["channel_assignment"],
                }
            )
    return rows


def fake_trajectory(
    qid: str, arm: str, taus: tuple[float, float], rng: random.Random
) -> dict[str, Any]:
    """A FAKE Exp 2 trajectory satisfying the analysis's invariants (paper Sec. 4.4, App. E)."""
    status, rounds = "incorrect", []
    for t in range(1, 5):
        scores = [rng.random() * 2 * taus[0], rng.random() * 2 * taus[1]]
        roll = rng.random()
        if roll < 0.01:
            action, parse_fail, after = None, True, status
        elif roll < 0.4:
            action, parse_fail, after = "accept", False, status
        else:
            action, parse_fail = "rewrite", False
            draw = rng.random()
            after = "unresolved" if draw < 0.03 else "correct" if draw < 0.35 else "incorrect"
        rounds.append(
            {
                "t": t,
                "action": action,
                "parse_fail": parse_fail,
                "input_grade": {"status": status},
                "output_grade": {"status": after},
                "input_scores": scores,
                "thresholds": list(taus),
            }
        )
        status = after
        if action == "accept":
            reason = "accepted"
            break
        if action == "rewrite" and t < 4 and rng.random() < 0.02:
            reason = "v_input_overlength"  # 2026-10-13: the rewrite is x_T
            break
    else:
        reason = "decision_limit"
    row = {
        "domain": "math",
        "stage": "development",
        "execution_mode": "fake",
        "run_id": "math_step10_FAKE",
        "question_id": qid,
        "initial_status": "incorrect",
        "arm": arm,
        "terminal_status": status,
        "termination_reason": reason,
        "is_correct": status == "correct" if status in ("correct", "incorrect") else None,
        "rounds": rounds,
    }
    if reason == "v_input_overlength":
        row["v_input_overlength"] = {"t": len(rounds) + 1, "tokens": 4200, "L_in": 4096}
    return row


def fake_trajectories(
    cells: list[dict[str, Any]], taus: tuple[float, float], rng: random.Random
) -> list[dict[str, Any]]:
    qids = sorted({c["x0_id"] for c in cells})
    arms = ["A1", "B"] * (len(qids) // 2) + ["A1"] * (len(qids) % 2)
    rng.shuffle(arms)
    return [fake_trajectory(q, a, taus, rng) for q, a in zip(qids, arms, strict=True)]


def fake(paths: Step10Paths, n_boot: int | None = None) -> dict[str, Any]:
    from pipelines.math_analysis import summarize_exp1, summarize_exp2
    from pipelines.math_diagnostics import (
        exp1_diagnostics,
        exp2_reliability_diagnostics,
        protocol_outcome_diagnostics,
    )
    from rafr.domains.math.config import MathRunConfig

    config = MathRunConfig.load(paths.formal_config)
    n_boot = n_boot or config.bootstrap_n
    cells = s6.read_checked(paths.exp1_inputs)
    rng = random.Random(derive_seed(config.seed, FAKE_COMPONENT))  # noqa: S311 - fake data
    exp1_rows = fake_exp1_rows(cells, rng)
    taus = (cells[0]["tau1"], cells[0]["tau2"])
    trajectories = fake_trajectories(cells, taus, rng)
    gate = read_json(paths.run / "gate_1.json")
    report = {
        "FAKE": True,
        "is_research_result": False,
        "what": "math step 10: every preregistered analysis of MathPipeline.report on FAKE "
        "decisions and trajectories of the formal shape; arm labels shuffled within each "
        "(state, lambda) in Exp 1 and over the 300 x0 in Exp 2 -- any arm contrast is null by "
        "construction",
        "inputs_sha256": s6.recorded_sha(paths.exp1_inputs),
        "fake_seed_component": FAKE_COMPONENT,
        "exp1": summarize_exp1(exp1_rows, n_boot, config.seed),
        "exp2": summarize_exp2(trajectories, n_boot, config.seed),
        "exp1_diagnostics": exp1_diagnostics(exp1_rows),
        "exp2_reliability_diagnostics": exp2_reliability_diagnostics(trajectories),
    }
    report["exp1_diagnostics"]["h2_reference_r_ref"] = gate["r_ref"]
    report.update(
        protocol_outcome_diagnostics(
            exp1_rows, trajectories, gate["r"], n_boot=n_boot, seed=config.seed
        )
    )
    s6.write_json(paths.fake_report, report)
    return report


def fake_summary(report: dict[str, Any]) -> dict[str, Any]:
    exp1, exp2 = report["exp1"], report["exp2"]
    return {
        "FAKE": report["FAKE"],
        "exp1": {
            "n_questions": exp1["n_questions"],
            "n_rows": exp1["n_rows"],
            "sections": sorted(exp1),
        },
        "exp2": {"sections": sorted(exp2)},
        "diagnostics": sorted(k for k in report if k not in ("exp1", "exp2")),
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("command", choices=("inputs", "cost", "fake"))
    ap.add_argument("--n-boot", type=int, help="fake: bootstrap draws (default: formal.json's)")
    args = ap.parse_args(argv)
    paths = Step10Paths()
    if args.command == "inputs":
        summary = inputs(paths)
        print(json.dumps({k: v for k, v in summary.items() if k != "live_by_arm_level"}))
    elif args.command == "cost":
        print(json.dumps(cost(paths)))
    else:
        print(json.dumps(fake_summary(fake(paths, args.n_boot))))
    return 0


if __name__ == "__main__":
    sys.exit(main())
