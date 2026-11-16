"""Step 11 — Exp 1 main experiment (step 11; 0.2, paper Sec. 4.3, 3, paper Sec. 4.6, 3).

Preconditions (all checked, run refused otherwise):

* every config value frozen (no TBD), no PLACEHOLDER in prompts or formatter templates;
* ``frozen/tau.json`` and ``frozen/channel.json`` exist and the gate attempts file records a
  PASSED attempt (paper Sec. 4.3); ``data/diagnostics/gate_diagnostics.json`` exists (paper App. A:
  diagnostics recorded before A1 - B is looked at);
* score tables of both formal verifiers cover every D_Exp1 state (step06).

Order of operations (paper Sec. 3): the Exp 1 outcome IS D_Exp1's y* -- the
judgments step05 made before the split, with the same E. They are read through the label gate
(purpose EXP1_OUTCOME), checked to come from the configured E, copied to
``exp1.outcome.judgments_file`` and hashed, THEN the controller runs. No E call is made here.
Outputs under ``runs/<run_id>/``.

The run report carries, besides the primary difference-in-differences (paper Sec. 4.6: lambda_high
vs lambda_low only, lambda_mid as a secondary three-level curve):

* the parse_fail table by arm x lambda, any cell above 2% flagged (paper Sec. 4.6);
* the N of the primary metric: D_Exp1 holds E-UNSAFE states only (paper Sec. 4.2), so
  the false accept rate's denominator is D_Exp1 in full (paper Sec. 4.6); a state that is not
  E-UNSAFE stops the analysis (``analysis.exp1_did.Exp1DataError``);
* the H2 strata (paper Sec. 4.6), computed from ``channel.jsonl``'s perturbation assignment;
* r_hat_ref, copied from the gate diagnostics next to those strata — offline reference only,
  never a stratifier and never a prompt input (paper Sec. 3);
* error boundary alignment per lambda and arm (paper Sec. 6.2).
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
from datetime import UTC, datetime

from analysis import error_boundary, evidence_sensitivity, exp1_did, parse_fail
from analysis._common import refuse_fake

from pipelines import _validity_io as io
from rafr.channel.degrade import DegradeChannel
from rafr.contracts.schemas import BOOTSTRAP_B, Arm, Lambda, Split
from rafr.controller import decide as ctrl
from rafr.controller.stub import is_stub_controller
from rafr.core import paths
from rafr.core.config import assert_frozen, get, load_config, master_seed
from rafr.core.runtime import build_client
from rafr.core.seeding import component_seed
from rafr.data import labels as L
from rafr.evaluator import judge
from rafr.experiments import exp1
from rafr.formatter import TEMPLATES_PATH
from rafr.logging.snapshot import write_snapshot
from rafr.logging.writer import read_jsonl
from rafr.validity import gate as G
from rafr.validity.tau import load_tau

GUARD_SECTIONS: tuple[str, ...] = (
    "models.controller",
    "models.evaluator",
    "models.verifiers",  # verifier ids key the score tables
    "generation.g_fixed.validity",  # refusal_phrases_file (controller's refusal_like flag)
    "refusal_phrases",
    "exp1",
)
"""Config sections Exp 1 actually reads; the formal-run guard checks these and nothing else
(2026-09-18: a step is blocked only by the values it consumes -- an Exp 2-only TBD
must not stop Exp 1). models.g_fixed is out: the states come frozen from step04."""


class TableScorer:
    """PairScorer backed by the formal verifiers' score tables (states are fixed in Exp 1)."""

    def __init__(
        self,
        v1_id,
        v2_id,
        s1: dict[str, float],
        s2: dict[str, float],
        text_to_state: dict[str, str],
    ):
        self.v1_id, self.v2_id = v1_id, v2_id
        self.v1_version = self.v2_version = "scores_table"
        self._s1, self._s2, self._map = s1, s2, text_to_state

    def score_pair(self, texts):
        return [(self._s1[self._map[t]], self._s2[self._map[t]]) for t in texts]


def _ts() -> str:
    return datetime.now(tz=UTC).strftime("%Y%m%dT%H%M%SZ")


def _sha256(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def read_r_ref(diagnostics_path: Path) -> dict:
    """r_hat_ref from the gate diagnostics (paper Sec. 3): the offline reference reliability, the
    same value for every arm. step09 computed it on the labelled gate split at lambda = none;
    this step only copies it into the report, so no split's labels are read here and nothing
    downstream can condition on it."""
    try:
        diag = json.loads(diagnostics_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as e:
        return {"available": False, "reason": str(e)}
    r_ref = diag.get("r_ref")
    if not r_ref:
        return {
            "available": False,
            "reason": f"no r_ref block in {diagnostics_path} (re-run step09)",
        }
    return {
        "available": True,
        **r_ref,
        "source": str(diagnostics_path),
        "role": "offline reference for H2 (II.1.1): reported alongside, never a stratifier, "
        "never in a prompt, never in the gate",
    }


def bootstrap_settings(configs) -> tuple[int, int]:
    """(n_boot, seed) for the CIs of this analysis (paper Sec. 4.6).

    B = 10000 is locked. The seed is this run's seed, derived from the one master seed
    (component ``exp1_run``), and is written into the analysis output and the run snapshot.
    """
    n_boot = get(configs["exp1"], "analysis.bootstrap.n_boot", BOOTSTRAP_B)
    return int(n_boot), component_seed(master_seed(configs), "exp1_run")


def outcome_from_y_star(states, ecfg) -> dict:
    """state_id -> EvalRecord: D_Exp1's y*, the Exp 1 outcome (paper Sec. 3).

    Read through the label gate (EXP1_OUTCOME) and refused if any record was made by an E other
    than the configured one; every state of the run must have one.
    """
    out = {}
    for y in L.read_labels(Split.D_EXP1, L.LabelPurpose.EXP1_OUTCOME):
        judge.check_same_e(judge.record_identity(y.record), ecfg, where="D_Exp1's y* (step05)")
        out[y.state_id] = y.record
    missing = [s.state_id for s in states if s.state_id not in out]
    if missing:
        raise SystemExit(f"{len(missing)} Exp 1 states have no y* (e.g. {missing[:3]})")
    return out


def preconditions() -> None:
    if not paths.FROZEN_TAU.exists():
        raise SystemExit("frozen/tau.json missing (step07)")
    if not paths.FROZEN_CHANNEL.exists():
        raise SystemExit("frozen/channel.json missing (step09 must pass first)")
    attempts = G.read_attempts(paths.FROZEN_GATE_ATTEMPTS)
    if not attempts or attempts[-1].get("passed") is not True:
        raise SystemExit("gate has not passed (II.6.1); Exp 1 refused")
    if not (paths.DATA / "diagnostics" / "gate_diagnostics.json").exists():
        raise SystemExit(
            "gate_diagnostics.json missing: II.6.2 diagnostics must be recorded before Exp 1"
        )


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--run-id", default=None, help="resume an existing run directory")
    ap.add_argument("--confirm", action="store_true", help="required: real API calls cost money")
    args = ap.parse_args(argv)
    cfgs = load_config()
    assert_frozen(cfgs, extra_files=[TEMPLATES_PATH], only_sections=GUARD_SECTIONS)
    # a controller whose model id starts with stub- is for tests only (the verifiers'
    # rule applied to the controller); a formal step refuses it before anything else happens
    model = str(get(cfgs["models"], "controller.model", ""))
    if is_stub_controller(model):
        raise SystemExit(
            f"models.controller.model = {model!r} is a stub controller; formal steps refuse "
            "stubs (II.4.4)"
        )
    preconditions()
    if not args.confirm:
        raise SystemExit(
            "Exp 1 spends money on E and controller calls: "
            "re-run with --confirm after the authors' go-ahead"
        )

    run_id = args.run_id or f"{get(cfgs['exp1'], 'run.run_id_prefix', 'exp1')}_{_ts()}"
    rd = paths.run_dir(run_id)
    refuse_fake(rd, what="step11")  # a resume never lands in step 10's fake data
    write_snapshot(
        rd,
        step="step11",
        configs=cfgs,
        extra={
            "tau_sha256": _sha256(paths.FROZEN_TAU),
            "channel_sha256": _sha256(paths.FROZEN_CHANNEL),
        },
    )
    client = build_client(cfgs["models"], run_dir=rd)

    v1, v2 = io.verifier_ids(cfgs["models"])
    states = io.load_states(Split(get(cfgs["exp1"], "design.states_split", "d_exp1")))
    s1, s2 = io.load_scores_for(states, v1), io.load_scores_for(states, v2)
    taus = load_tau(paths.FROZEN_TAU)
    channel = DegradeChannel.load(paths.FROZEN_CHANNEL)

    # 1) the outcome: D_Exp1's y* from step05, recorded before any controller call (paper Sec. 3)
    ecfg = judge.build_config(cfgs["models"])  # the one E; re-checks the prompt sha256
    e_labels = outcome_from_y_star(states, ecfg)
    jpath = paths.resolve(get(cfgs["exp1"], "outcome.judgments_file"))
    jpath.parent.mkdir(parents=True, exist_ok=True)
    with jpath.open("w", encoding="utf-8", newline="\n") as fh:
        for s in states:
            fh.write(e_labels[s.state_id].model_dump_json() + "\n")
    (rd / "e_judgments_hash.json").write_text(
        json.dumps(
            {
                "file": str(jpath),
                "sha256": _sha256(jpath),
                "n": len(e_labels),
                "source": "D_Exp1 y* (step05)",
                "e_identity": ecfg.identity(),
            }
        ),
        encoding="utf-8",
        newline="\n",
    )
    print(
        f"[step11] outcome = D_Exp1's y*: {len(e_labels)} states, sha256 "
        f"{_sha256(jpath)[:12]} (recorded before any controller call)"
    )

    # 2) controller
    ccfg = ctrl.build_config(
        cfgs["models"],
        cfgs["exp1"],
        base_seed=component_seed(master_seed(cfgs), "controller"),
        refusal_phrases_file=get(cfgs["generation"], "g_fixed.validity.refusal_phrases_file"),
    )
    setup = exp1.Exp1Setup(
        run_id=run_id,
        run_dir=rd,
        states=states,
        scorer=TableScorer(v1, v2, s1, s2, {s.text: s.state_id for s in states}),
        tau1=taus["v1"].tau,
        tau2=taus["v2"].tau,
        channel=channel,
        lambdas=[Lambda(x) for x in get(cfgs["exp1"], "design.lambdas")],
        arms=[Arm(x) for x in get(cfgs["exp1"], "design.arms")],
        perm_seed=component_seed(master_seed(cfgs), "permuted_b"),
        run_seed=component_seed(master_seed(cfgs), "exp1_run"),
        e_labels=e_labels,
    )
    summary = exp1.run_exp1(client, ccfg, setup)
    rows = list(read_jsonl(rd / "decisions.jsonl"))
    refuse_fake(rd, rows, what="step11 analysis")
    channel_rows = list(read_jsonl(rd / "channel.jsonl"))
    n_boot, boot_seed = bootstrap_settings(cfgs)
    analysis = exp1_did.analyze(rows, seed=boot_seed, n_boot=n_boot, channel_rows=channel_rows)
    # paper Sec. 3: the offline reference sits NEXT TO the H2 strata, never inside them
    analysis["h2_reference_r_ref"] = read_r_ref(
        paths.DATA / "diagnostics" / "gate_diagnostics.json"
    )
    report = {
        "run_summary": summary,
        # how often a lambda level sent the controller exactly
        # lambda_low's prompt (main) and showed exactly lambda_low's s_obs (auxiliary)
        "identical_to_low": exp1.identity_to_low_by_lambda(rows),
        "parse_fail": parse_fail.report(rows, by=parse_fail.BY_LAMBDA),
        "analysis": analysis,
        "error_boundary_alignment": error_boundary.exp1(rows),
        # paper Sec. 4.6: accept rate by arm x lambda x observed (y1, y2), descriptive
        "evidence_sensitivity": evidence_sensitivity.exp1(rows),
    }
    (rd / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=1, default=str),
        encoding="utf-8",
        newline="\n",
    )
    print(parse_fail.render(report["parse_fail"]))
    print(evidence_sensitivity.render(report["evidence_sensitivity"]))
    eff = analysis["effective_n"]
    print(
        f"[step11] N of the primary metric: all {eff['n_states']} D_Exp1 states (E-UNSAFE by "
        f"construction, III.1.2); {eff['n_did_units']} contribute a D_i."
    )
    print(error_boundary.render(report["error_boundary_alignment"]))
    ident = report["identical_to_low"]["by_lambda"]
    print(
        "[step11] (state, arm) whose prompt is byte for byte lambda_low's: "
        + ", ".join(f"{lam} {c['prompt_identical']}/{c['n']}" for lam, c in ident.items())
        + "; auxiliary, s_obs bit for bit lambda_low's: "
        + ", ".join(f"{lam} {c['s_obs_identical']}/{c['n']}" for lam, c in ident.items())
    )
    print(f"[step11] ledger: {json.dumps(client._ledger.summary() if client._ledger else {})}")
    print(f"[step11] report -> {rd / 'report.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
