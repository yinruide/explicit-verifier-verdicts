"""Regression of the extension against the preregistered detox runs (read only; no API call).

Post-pilot extension, stop point B (2026-10-31). Two checks; neither writes anything into
the original run directories:

1. **LogRecord.** Every decision row of the original detox Exp 1 and Exp 2 runs (the run-level
   ``decisions.jsonl`` and every round's) validates against the current schema, whose ``Arm``
   has four more members.
2. **Prompts.** The extension runners rebuild the first-round prompts of the original A1 / B /
   permuted-B decisions on D_Exp1 with a fake controller that always accepts -- Exp 1's 2,700
   (state, arm, lambda) through ``exp1_ext.run_exp1_ext``, Exp 2's 300 round-1 decisions through
   ``exp2_ext.run_loop_ext`` with the original assignment and the reused t = 1 scores -- and every
   prompt hash and seed must equal the original run's, byte for byte. On the way the Exp 1 runner
   recomputes the channel and checks it bit for bit against the original ``channel.jsonl``, and
   permuted-B's table is redrawn and compared with the original ``permutation.jsonl``.

The rebuilt decisions go to a scratch directory (``--work``; default a temporary one) and are
thrown away; the report goes to ``runs/ext_prompt_regression_<timestamp>/report.json``.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
import json
import tempfile
from collections.abc import Mapping
from dataclasses import replace
from datetime import UTC, datetime
from typing import Any

from pipelines import _validity_io as io
from pipelines.step11_exp1 import TableScorer
from rafr.channel.degrade import DegradeChannel
from rafr.contracts.schemas import Arm, EvalRecord, Lambda, LogRecord, SafetyLabel, Split, State
from rafr.controller import decide as ctrl
from rafr.core import paths
from rafr.core.config import get, load_config, master_seed
from rafr.core.seeding import component_seed
from rafr.experiments import exp1_ext, exp2_ext
from rafr.experiments.ext_config import checked, load_ext
from rafr.formatter.extended import detox_wording
from rafr.formatter.permute import load_permutation
from rafr.llm.client import LLMClient
from rafr.llm.providers.fake import FakeProvider
from rafr.logging.writer import read_jsonl
from rafr.validity.tau import load_tau


def _accept(req: Any) -> str:
    return json.dumps({"action": "accept", "revised_text": ""})


def _fake_client() -> LLMClient:
    return LLMClient(providers={"fake": FakeProvider(responder=_accept)})


def _ccfg(cfgs: Mapping[str, Any], exp: str) -> ctrl.ControllerConfig:
    """The formal controller config, sent to the fake provider (the prompt hash covers system
    prompt, messages and schema only; provider and seed are not part of it)."""
    ccfg = ctrl.build_config(
        cfgs["models"],
        cfgs[exp],
        base_seed=component_seed(master_seed(cfgs), "controller"),
        refusal_phrases_file=get(cfgs["generation"], "g_fixed.validity.refusal_phrases_file"),
    )
    return replace(ccfg, provider="fake")


def revalidate_logs(run_dirs: list[Path]) -> dict[str, Any]:
    """Every decision row of these runs against the current LogRecord schema."""
    out: dict[str, Any] = {}
    for rd in run_dirs:
        files = [rd / "decisions.jsonl", *sorted(rd.glob("rounds/round_*/decisions.jsonl"))]
        for f in files:
            if not f.is_file():
                continue
            n = bad = 0
            first_error = None
            for row in read_jsonl(f):
                n += 1
                try:
                    LogRecord.model_validate(row)
                except ValueError as exc:  # pydantic.ValidationError is a ValueError
                    bad += 1
                    first_error = first_error or str(exc)[:300]
            out[paths.shown(f)] = {
                "rows": n,
                "invalid": bad,
                "first_error": first_error,
            }
    return out


def _label(text: str) -> EvalRecord:
    return EvalRecord(
        text_id="regression",
        text=text,
        label=SafetyLabel.UNSAFE,
        model="regression",
        model_version=None,
        prompt_hash="regression",
        e_prompt_sha256="regression",
        e_version=None,
        decoding={},
    )


def _compare(
    original: list[dict[str, Any]], rebuilt: list[dict[str, Any]], key: tuple[str, ...]
) -> dict[str, Any]:
    orig = {tuple(r[k] for k in key): r for r in original}
    new = {tuple(r[k] for k in key): r for r in rebuilt}
    by_arm: dict[str, dict[str, int]] = {}
    mismatched: list[str] = []
    for k, o in orig.items():
        c = by_arm.setdefault(o["arm"], {"n": 0, "prompt_identical": 0, "seed_identical": 0})
        c["n"] += 1
        r = new.get(k)
        if r is None:
            mismatched.append(f"{k}: not rebuilt")
            continue
        same_prompt = r["attempts"][0]["prompt_hash"] == o["attempts"][0]["prompt_hash"]
        c["prompt_identical"] += int(same_prompt)
        c["seed_identical"] += int(r["attempts"][0]["seed"] == o["attempts"][0]["seed"])
        if not same_prompt:
            mismatched.append(f"{k}: prompt hash differs")
    return {
        "n_original": len(orig),
        "n_rebuilt": len(new),
        "by_arm": by_arm,
        "all_identical": not mismatched and len(orig) == len(new),
        "mismatched": mismatched[:20],
    }


def exp1_prompts(cfgs: Mapping[str, Any], ext: Mapping[str, Any], work: Path) -> dict[str, Any]:
    """The 2,700 original Exp 1 decisions rebuilt through the extension's Exp 1 runner."""
    spec = get(ext, "exp1.e_incorrect")
    run = paths.resolve(spec["original_run"])
    sha = spec["original_sha256"]
    decisions = checked(run / "decisions.jsonl", sha["decisions.jsonl"], what="Exp 1 decisions")
    channel_file = checked(run / "channel.jsonl", sha["channel.jsonl"], what="Exp 1 channel")
    perm_file = checked(run / "permutation.jsonl", sha["permutation.jsonl"], what="permuted-B")
    original = list(read_jsonl(decisions))
    states: list[State] = io.load_states(Split.D_EXP1)
    if {s.state_id for s in states} != {r["state_id"] for r in original}:
        raise SystemExit("D_Exp1's states are not the original Exp 1's")
    v1, v2 = io.verifier_ids(cfgs["models"])
    s1, s2 = io.load_scores_for(states, v1), io.load_scores_for(states, v2)
    taus = load_tau(paths.FROZEN_TAU)
    master = master_seed(cfgs)
    setup = exp1_ext.Exp1ExtSetup(
        run_id="ext-prompt-regression",
        run_dir=work / "exp1",
        cohort=exp1_ext.E_CORRECT,  # the registry that runs A1 / B / permuted-B
        states=states,
        scorer=TableScorer(v1, v2, s1, s2, {s.text: s.state_id for s in states}),
        tau1=taus["v1"].tau,
        tau2=taus["v2"].tau,
        channel=DegradeChannel.load(paths.FROZEN_CHANNEL),
        lambdas=[Lambda(x) for x in get(cfgs["exp1"], "design.lambdas")],
        arms=list(exp1_ext.ORIGINAL_ARMS),
        wording=detox_wording(),
        e_labels={s.state_id: _label(s.text) for s in states},
        expected_label=SafetyLabel.UNSAFE,
        perm_seed=component_seed(master, "permuted_b"),
        original_channel=list(read_jsonl(channel_file)),
        seed_namespace="exp1",  # the original's: seeds must match too
    )
    exp1_ext.run_exp1_ext(_fake_client(), _ccfg(cfgs, "exp1"), None, setup)
    rebuilt = list(read_jsonl(work / "exp1" / "decisions.jsonl"))
    out = _compare(original, rebuilt, ("state_id", "arm", "lam"))
    seed_o, table_o = load_permutation(perm_file)
    seed_n, table_n = load_permutation(work / "exp1" / "permutation.jsonl")
    out["permutation_identical"] = seed_o == seed_n and table_o == table_n
    out["channel_bitwise_identical"] = True  # run_exp1_ext raises otherwise
    return out


class _NoScorer:
    """Round 1 takes the reused scores; with an accept-only controller there is no round 2."""

    def __init__(self, v1_id: str, v2_id: str) -> None:
        self.v1_id, self.v2_id = v1_id, v2_id
        self.v1_version = self.v2_version = "reused"

    def score_pair(self, texts: Any) -> Any:
        raise RuntimeError("nothing is scored in the regression")


def t1_scores_from(path: Path, v1_id: str, v2_id: str) -> dict[str, tuple[float, float]]:
    """x0 -> (s1, s2) from an Exp 2 round-1 score file (round ids ``{x0}-{arm}-t1``)."""
    by: dict[str, dict[str, float]] = {}
    for r in read_jsonl(path):
        x0 = str(r["state_id"]).rsplit("-", 2)[0]
        slot = by.setdefault(x0, {})
        if r["verifier_id"] in slot and slot[r["verifier_id"]] != r["score"]:
            raise SystemExit(f"{path}: {x0} has two {r['verifier_id']} scores")
        slot[r["verifier_id"]] = float(r["score"])
    out = {}
    for x0, slot in by.items():
        if set(slot) != {v1_id, v2_id}:
            raise SystemExit(f"{path}: {x0} has scores for {sorted(slot)}")
        out[x0] = (slot[v1_id], slot[v2_id])
    return out


def exp2_prompts(cfgs: Mapping[str, Any], ext: Mapping[str, Any], work: Path) -> dict[str, Any]:
    """The 300 original Exp 2 round-1 decisions rebuilt through the extension's Exp 2 loop."""
    spec = get(ext, "exp2.e_incorrect")
    run = paths.resolve(spec["original_run"])
    assignment_file = checked(
        run / "assignment.jsonl", spec["original_sha256"]["assignment.jsonl"], what="assignment"
    )
    x1_file = checked(spec["x1_file"], spec["x1_sha256"], what="Exp 2 x1")
    t1 = spec["t1_scores"]
    t1_file = checked(t1["source"], t1["source_sha256"], what="t = 1 scores")
    starts = [State.model_validate(r) for r in read_jsonl(x1_file)]
    assignment = {r["x0_id"]: Arm(r["arm"]) for r in read_jsonl(assignment_file)}
    v1, v2 = io.verifier_ids(cfgs["models"])
    taus = load_tau(paths.FROZEN_TAU)
    setup = exp2_ext.Exp2ExtSetup(
        run_id="ext-prompt-regression",
        run_dir=work / "exp2",
        cohort=exp2_ext.E_INCORRECT,
        starts=starts,
        plan={s.x0_id: [assignment[s.x0_id]] for s in starts},  # the ORIGINAL arm, to compare
        scorer=_NoScorer(v1, v2),
        tau1=taus["v1"].tau,
        tau2=taus["v2"].tau,
        k=int(get(ext, "exp2.k")),
        wording=detox_wording(),
        judge_fn=lambda text: _label(text),
        t1_scores=t1_scores_from(t1_file, v1, v2),
    )
    exp2_ext.run_loop_ext(_fake_client(), _ccfg(cfgs, "exp2"), None, setup)
    original = list(read_jsonl(run / "rounds" / "round_1" / "decisions.jsonl"))
    rebuilt = [r for r in read_jsonl(work / "exp2" / "decisions.jsonl") if int(r["t"]) == 1]
    return _compare(original, rebuilt, ("state_id", "arm", "t"))


def regression(work: Path) -> dict[str, Any]:
    cfgs = load_config()
    ext = load_ext()
    runs = [
        paths.resolve(get(ext, "exp1.e_incorrect.original_run")),
        paths.resolve(get(ext, "exp2.e_incorrect.original_run")),
    ]
    return {
        "log_records": revalidate_logs(runs),
        "exp1_prompts": exp1_prompts(cfgs, ext, Path(work)),
        "exp2_round1_prompts": exp2_prompts(cfgs, ext, Path(work)),
    }


def passed(report: Mapping[str, Any]) -> bool:
    logs_ok = all(v["invalid"] == 0 and v["rows"] > 0 for v in report["log_records"].values())
    e1, e2 = report["exp1_prompts"], report["exp2_round1_prompts"]
    return bool(
        logs_ok and e1["all_identical"] and e1["permutation_identical"] and e2["all_identical"]
    )


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--work", default=None, help="scratch directory (default: a temporary one)")
    args = ap.parse_args(argv)
    ts = datetime.now(tz=UTC).strftime("%Y%m%dT%H%M%SZ")
    if args.work:
        report = regression(Path(args.work))
    else:
        with tempfile.TemporaryDirectory() as tmp:
            report = regression(Path(tmp))
    report["passed"] = passed(report)
    out = paths.run_dir(f"ext_prompt_regression_{ts}")
    out.mkdir(parents=True, exist_ok=True)
    (out / "report.json").write_text(
        json.dumps(report, indent=1, ensure_ascii=False), encoding="utf-8", newline="\n"
    )
    for f, v in report["log_records"].items():
        print(f"[regression] LogRecord {f}: {v['rows']} rows, {v['invalid']} invalid")
    for name in ("exp1_prompts", "exp2_round1_prompts"):
        r = report[name]
        print(f"[regression] {name}: {json.dumps(r['by_arm'])}; all identical {r['all_identical']}")
    same_table = report["exp1_prompts"]["permutation_identical"]
    print(f"[regression] permuted-B table identical: {same_table}")
    print(f"[regression] passed: {report['passed']} -> {out / 'report.json'}")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
