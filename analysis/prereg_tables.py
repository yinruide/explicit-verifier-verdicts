"""The preregistered analysis plan (``prereg/analysis_plan.md``), section by section: each
section's callable functions and output tables (protocol 0.5 step 10, tag
``analysis-preregistered``).

:data:`SECTIONS` is the map -- plan section -> the functions that compute it -> the tables it
outputs -- and says, for every part of a section that has no function, why. :func:`build_tables`
computes every table from the logs of one Exp 1 run and one Exp 2 run (plus step 9's report,
for the gate-side reference values); :func:`write_tables` writes each table as JSON and all of
them as one Markdown file.

Fake vs formal. :func:`load_inputs` takes ``fake``. With ``fake=True`` (step 10) it refuses every
input that lacks the FAKE marker, so the dry run reads no real label, score or controller
output, and every file :func:`write_tables` writes is named ``FAKE_*`` and says FAKE inside.
With ``fake=False`` it refuses everything marked FAKE, so the formal analysis never reads step
10's data. The formal entry point::

    python -m analysis.prereg_tables --exp1-run-dir runs/exp1_... --exp2-run-dir runs/exp2_... \\
        [--gate-report data/diagnostics/step09_report.json] [--out-dir DIR]

Every interval is a 95% percentile bootstrap, B = 10000, seed recorded (III.1.2); the pilot makes
no significance claims (III.1.1), so no table carries a p-value or a verdict.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import sys
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from analysis import (
    design_checks,
    drift,
    error_boundary,
    evidence_sensitivity,
    exp1_did,
    exp2_tfr,
    parse_fail,
    reliability,
    transport,
)
from analysis._common import (
    ACCOUNTINGS,
    FAKE_FILE_PREFIX,
    FakeDataError,
    refuse_fake,
    require_fake,
    same_sign,
)
from rafr.contracts.schemas import BOOTSTRAP_B, K_MAX_ROUNDS
from rafr.experiments.exp1 import identity_to_low_by_lambda
from rafr.logging.writer import read_jsonl

PILOT_NOTE = (
    "pilot: point estimates and 95% percentile bootstrap CIs; no significance claims (III.1.1)"
)


# ---- the map ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SectionSpec:
    section: str
    title: str
    functions: tuple[str, ...]
    tables: tuple[str, ...]
    not_implemented: tuple[str, ...] = ()


SECTIONS: tuple[SectionSpec, ...] = (
    SectionSpec(
        "1",
        "hypotheses and sign convention (I.2)",
        ("analysis.exp1_did.did",),
        ("T01_h1_direction",),
    ),
    SectionSpec(
        "2",
        "the pilot's standing (III.1.1)",
        (),
        (),
        ("no computation: every table carries the pilot note; no test, no p-value",),
    ),
    SectionSpec(
        "3",
        "construction of the analysis objects (II.2.1-II.2.4, II.3)",
        ("analysis.design_checks.exp1", "analysis.exp1_did.check_d_exp1_all_unsafe"),
        ("T03_exp1_construction",),
        (
            "the construction itself (prefilter, G_fixed states, y*, splits) is steps 0-5, "
            "done and frozen; step 10 checks what the Exp 1 log shows of it",
        ),
    ),
    SectionSpec(
        "4",
        "metric definitions (II.9.3)",
        (
            "analysis.exp1_did.cell_rates",
            "analysis.exp2_tfr.tfr_by_arm",
            "analysis.exp2_tfr.secondary",
            "analysis.exp2_tfr.mechanism",
            "analysis.exp2_tfr.low_preservation_among_safe",
        ),
        ("T04_metrics",),
        (
            "semantic preservation SCORE (TextDetox official protocol) not computed: "
            "rafr/domains/detox/semantic.py does not exist yet and needs the TextDetox models; "
            "the finals' semantic_score is None in a formal run until it does",
        ),
    ),
    SectionSpec(
        "5",
        "E and y* (II.3, II.9.1)",
        ("analysis.design_checks.e_identity",),
        ("T05_e_identity",),
    ),
    SectionSpec(
        "6",
        "Exp 1 primary analysis (III.1.2, II.8.2 / C.24)",
        (
            "analysis.exp1_did.effective_n",
            "analysis.exp1_did.cell_rates",
            "analysis.exp1_did.did",
            "analysis.exp1_did.delta_by_lambda",
            "rafr.experiments.exp1.identity_to_low_by_lambda",
        ),
        (
            "T06_exp1_effective_n",
            "T06_exp1_cells",
            "T06_exp1_did",
            "T06_exp1_delta_by_lambda",
            "T06_prompt_identity",
        ),
    ),
    SectionSpec(
        "6.1",
        "point estimates and intervals (III.1.2)",
        ("analysis._common.bootstrap_ci", "analysis._common.stratified_bootstrap_ci"),
        ("T061_bootstrap",),
    ),
    SectionSpec(
        "6.2",
        "H2 strata (III.1.2); r_hat_ref alongside (II.1.1)",
        ("analysis.exp1_did.h2_strata",),
        ("T062_h2_strata", "T062_r_ref"),
    ),
    SectionSpec(
        "6.3",
        "evidence sensitivity (III.1.2)",
        ("analysis.evidence_sensitivity.exp1", "analysis.evidence_sensitivity.exp2"),
        ("T063_evidence_exp1", "T063_evidence_exp2"),
    ),
    SectionSpec(
        "7",
        "Exp 2 start, loop and final output (II.8.3)",
        ("analysis.design_checks.exp2",),
        ("T07_exp2_construction",),
    ),
    SectionSpec(
        "8",
        "Exp 2 analyses (III.1.2)",
        (
            "analysis.exp2_tfr.tfr_by_arm",
            "analysis.exp2_tfr.contrast",
            "analysis.exp2_tfr.secondary",
            "analysis.reliability.fill_c_t",
            "analysis.reliability.r_t_by_arm",
            "analysis.exp2_tfr.mechanism",
        ),
        (
            "T08_tfr_by_arm",
            "T08_tfr_contrast",
            "T08_secondary",
            "T08_rewrite_rounds",
            "T08_r_t",
            "T08_mechanism",
        ),
        (
            "Rep x R_t interaction (exploratory mechanism only): the plan gives no estimator, "
            "and R_t is post-treatment (a collider); defining one now would add analysis the "
            "preregistration does not contain",
        ),
    ),
    SectionSpec(
        "8.1",
        "panel sensitivity (III.1.2, Exp 2 only)",
        ("analysis.exp2_tfr.panel_sensitivity",),
        ("T081_panel",),
    ),
    SectionSpec(
        "8.2",
        "semantic preservation, descriptive",
        ("analysis.exp2_tfr.low_preservation_among_safe",),
        ("T082_semantic",),
        (
            "the share needs the cutpoint (configs/exp2.yaml, still TBD) and the scores "
            "(semantic.py, not written): until both exist only the counts are reported",
        ),
    ),
    SectionSpec(
        "9",
        "parse_fail and refusal_like accounting and sensitivities (II.8.2)",
        (
            "analysis._common.accepted",
            "analysis.exp1_did.did",
            "analysis.exp2_tfr.contrast",
            "analysis.parse_fail.report",
        ),
        ("T09_parse_fail_exp1", "T09_parse_fail_exp2", "T09_robustness"),
    ),
    SectionSpec(
        "10",
        "drift on Exp 1 (II.6.3)",
        ("analysis.drift.exp1_ladder", "analysis.drift.compare_to_gate"),
        ("T10_drift",),
        ("no 'clear drift' verdict: II.6.3 names no threshold",),
    ),
    SectionSpec(
        "11",
        "transport diagnostic (I.5.4, II.10)",
        ("analysis.transport.compare",),
        ("T11_transport", "T11_transport_by_round"),
    ),
    SectionSpec(
        "12",
        "result shapes (III.2.3)",
        ("analysis.exp1_did.result_shape",),
        ("T12_shape",),
        ("no flag for 'Delta ~ 0' or 'permuted-B ~ B': III.2.3 gives no tolerance",),
    ),
    SectionSpec(
        "12.1",
        "error boundary alignment (III.3)",
        ("analysis.error_boundary.exp1", "analysis.error_boundary.exp2"),
        ("T121_error_boundary_exp1", "T121_error_boundary_exp2"),
    ),
    SectionSpec(
        "13",
        "feasibility (III.1.3)",
        (),
        (),
        (
            "the power re-run with measured parameters is for after the pilot (its target N "
            "comes from the pilot's effect size) and its scripts (feas/) are not in the repo",
        ),
    ),
    SectionSpec(
        "13.1",
        "the gate (II.6.1)",
        ("rafr.validity.gate.ladder_check", "rafr.validity.gate.ladder_ci"),
        (),
        ("done at step 9 (data/diagnostics/step09_report.json); not re-run here",),
    ),
    SectionSpec(
        "14",
        "freezing and reproduction (II.10)",
        ("analysis._common.refuse_fake", "analysis._common.require_fake"),
        ("T14_provenance",),
        ("the three git tags are the authors' (App. A.1)",),
    ),
    SectionSpec(
        "15",
        "values filled in (App. D)",
        (),
        (),
        ("no computation: the values are in the plan itself",),
    ),
)


# ---- tables ----------------------------------------------------------------------------


@dataclass
class Table:
    id: str
    section: str
    title: str
    columns: list[str]
    rows: list[list[Any]]
    notes: list[str] = field(default_factory=list)


@dataclass
class Inputs:
    exp1_dir: Path
    exp2_dir: Path
    exp1_rows: list[dict[str, Any]]
    exp1_channel: list[dict[str, Any]]
    exp1_outcome: list[dict[str, Any]]
    exp2_rows: list[dict[str, Any]]
    exp2_finals: list[dict[str, Any]]
    exp2_assignment: list[dict[str, Any]]
    exp2_round_judgments: list[dict[str, Any]]
    exp2_evaluations: list[dict[str, Any]]
    panel_labels: dict[str, dict[str, str | None]]
    gate_report: dict[str, Any] | None
    gate_report_path: Path | None
    fake: bool


def _jsonl(path: Path) -> list[dict[str, Any]]:
    return list(read_jsonl(path))


def load_inputs(
    exp1_dir: Path, exp2_dir: Path, *, gate_report: Path | None = None, fake: bool
) -> Inputs:
    """Read both run directories; ``fake`` decides which kind of data is refused."""
    exp1_dir, exp2_dir = Path(exp1_dir), Path(exp2_dir)
    guard = require_fake if fake else refuse_fake
    rows1 = _jsonl(exp1_dir / "decisions.jsonl")
    rows2 = _jsonl(exp2_dir / "decisions.jsonl")
    guard(exp1_dir, rows1, what="Exp 1 run")
    guard(exp2_dir, rows2, what="Exp 2 run")
    if not rows1 or not rows2:
        raise FileNotFoundError(f"decisions.jsonl missing or empty in {exp1_dir} / {exp2_dir}")
    outcome: list[dict[str, Any]] = []
    meta = exp1_dir / "e_judgments_hash.json"
    if meta.is_file():
        opath = Path(json.loads(meta.read_text(encoding="utf-8"))["file"])
        guard(opath, what="Exp 1 outcome file")
        outcome = _jsonl(opath)
    gate = None
    if gate_report is not None:
        guard(Path(gate_report), what="gate report")
        gate = json.loads(Path(gate_report).read_text(encoding="utf-8"))
        if bool(gate.get("FAKE")) != fake:
            raise FakeDataError(f"gate report {gate_report}: FAKE={gate.get('FAKE')}, fake={fake}")
    panel: dict[str, dict[str, str | None]] = {}
    for f in sorted(exp2_dir.glob("panel_*.jsonl")):
        panel[f.stem.removeprefix("panel_")] = {r["text_id"]: r["label"] for r in _jsonl(f)}
    return Inputs(
        exp1_dir=exp1_dir,
        exp2_dir=exp2_dir,
        exp1_rows=rows1,
        exp1_channel=_jsonl(exp1_dir / "channel.jsonl"),
        exp1_outcome=outcome,
        exp2_rows=rows2,
        exp2_finals=_jsonl(exp2_dir / "trajectories.jsonl"),
        exp2_assignment=_jsonl(exp2_dir / "assignment.jsonl"),
        exp2_round_judgments=_jsonl(exp2_dir / "round_judgments.jsonl"),
        exp2_evaluations=_jsonl(exp2_dir / "evaluations.jsonl"),
        panel_labels=panel,
        gate_report=gate,
        gate_report_path=None if gate_report is None else Path(gate_report),
        fake=fake,
    )


def _split(key: str) -> tuple[str, str]:
    a, b = key.split("|", 1)
    return a, b


def _level_order(g: Any) -> tuple[int, Any]:
    """Rows in lambda order (low, mid, high) or round order, not alphabetical."""
    g = str(g)
    if g in exp1_did.LAMBDAS:
        return (0, exp1_did.LAMBDAS.index(g))
    return (1, int(g)) if g.isdigit() else (2, g)


def _pf_order(key: str) -> tuple[str, tuple[int, Any]]:
    arm, g = _split(key)
    return arm, _level_order(g)


def _sign(x: float | None) -> str:
    if x is None or x != x:
        return "undefined"
    return "positive" if x > 0 else "negative" if x < 0 else "zero"


def _sha256(path: Path) -> str | None:
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None


def compute(
    inp: Inputs,
    *,
    seed_exp1: int,
    seed_exp2: int,
    n_boot: int = BOOTSTRAP_B,
    semantic_cutpoint: float | None = None,
    n_exp1: int = 300,
    n_per_arm: int = 150,
) -> dict[str, Any]:
    """Every preregistered analysis on the inputs, as nested results (the tables' source).
    ``n_exp1`` / ``n_per_arm`` are the sizes the construction checks expect (II.2.4, II.8.3)."""
    rows1 = copy.deepcopy(inp.exp1_rows)
    rows2 = copy.deepcopy(inp.exp2_rows)
    truth2 = reliability.truth_from_round_judgments(inp.exp2_round_judgments)
    n_c = reliability.fill_c_t(rows2, truth2)  # raw s: Exp 2 has no channel
    a1 = exp1_did.analyze(rows1, seed=seed_exp1, n_boot=n_boot, channel_rows=inp.exp1_channel)
    a2 = exp2_tfr.analyze(
        inp.exp2_finals,
        k=K_MAX_ROUNDS,
        seed=seed_exp2,
        n_boot=n_boot,
        semantic_cutpoint=semantic_cutpoint,
    )
    gate_r = (inp.gate_report or {}).get("r")
    ladder = drift.exp1_ladder(rows1, seed=seed_exp1, n_boot=n_boot)
    return {
        "exp1": a1,
        "exp1_prompt_identity": identity_to_low_by_lambda(rows1),
        "exp1_parse_fail": parse_fail.report(rows1, by=parse_fail.BY_LAMBDA),
        "exp1_error_boundary": error_boundary.exp1(rows1),
        "exp1_evidence_sensitivity": evidence_sensitivity.exp1(rows1),
        "exp1_checks": design_checks.exp1(rows1, n_expected=n_exp1),
        "exp1_drift": {"ladder": ladder, "vs_gate": drift.compare_to_gate(ladder, gate_r)},
        "r_ref": (inp.gate_report or {}).get("r_ref"),
        "exp2": a2,
        "exp2_reliability": {"n_rows_with_c_t": n_c, "r_t_by_arm": reliability.r_t_by_arm(rows2)},
        "exp2_mechanism": exp2_tfr.mechanism(rows2, truth2),
        "exp2_parse_fail": parse_fail.report(rows2, by=parse_fail.BY_ROUND),
        "exp2_error_boundary": error_boundary.exp2(rows2, truth2),
        "exp2_evidence_sensitivity": evidence_sensitivity.exp2(rows2),
        "exp2_panel": exp2_tfr.panel_sensitivity(
            inp.exp2_finals,
            inp.panel_labels,
            main_estimate=a2["primary_contrast"]["estimate"],
            seed=seed_exp2,
            n_boot=n_boot,
        ),
        "exp2_checks": design_checks.exp2(
            inp.exp2_finals,
            rows2,
            inp.exp2_assignment,
            exp1_rows=rows1,
            round_judgments=inp.exp2_round_judgments,
            n_per_arm=n_per_arm,
        ),
        "e_identity": design_checks.e_identity(
            {"exp1_outcome": inp.exp1_outcome, "exp2_evaluations": inp.exp2_evaluations}
        ),
        "transport": transport.compare(rows1, rows2, truth2),
        "settings": {
            "seed_exp1": seed_exp1,
            "seed_exp2": seed_exp2,
            "n_boot": n_boot,
            "semantic_cutpoint": semantic_cutpoint,
            "n_exp1": n_exp1,
            "n_per_arm": n_per_arm,
        },
    }


# ---- one builder per table ------------------------------------------------------------

Builder = Callable[[dict[str, Any], Inputs], Table]


def t01_h1_direction(r: dict[str, Any], inp: Inputs) -> Table:
    rows = []
    for acc in ACCOUNTINGS:
        d = r["exp1"]["primary_did"][acc]
        rows.append(
            [acc, d["contrast"], d["estimate"], *d["ci95"], d["n_states"], _sign(d["estimate"])]
        )
    return Table(
        "T01_h1_direction",
        "1",
        "H1: Delta_FA(high) - Delta_FA(low) = D-bar, A1 - B (positive = B better, H1's direction)",
        ["accounting", "contrast", "D_bar", "ci95_lo", "ci95_hi", "n_states", "sign"],
        rows,
        [
            "H1 is stated as Delta_FA(lambda_high) > Delta_FA(lambda_low), i.e. D-bar > 0 (I.2)",
            "same sign under all three accountings: "
            f"{r['exp1']['primary_robust_across_accountings']}",
            PILOT_NOTE,
        ],
    )


def _checks(tid: str, section: str, title: str, checks: list[dict[str, Any]]) -> Table:
    rows = [[c["check"], c["value"], c["expected"], c["ok"], c["ref"]] for c in checks]
    return Table(tid, section, title, ["check", "value", "expected", "ok", "ref"], rows)


def t03(r: dict[str, Any], inp: Inputs) -> Table:
    return _checks(
        "T03_exp1_construction",
        "3",
        "Exp 1: what the log shows of the construction",
        r["exp1_checks"],
    )


def t04_metrics(r: dict[str, Any], inp: Inputs) -> Table:
    rows: list[list[Any]] = []
    for key, c in r["exp1"]["cells_main"].items():
        arm, lam = _split(key)
        rows.append(
            [
                "Exp 1 primary",
                "decision-level false accept rate",
                arm,
                f"lambda={lam}",
                c["false_accept_rate"],
                c["n"],
            ]
        )
    for arm, c in r["exp2"]["primary_TFR"].items():
        rows.append(["Exp 2 primary", "TFR", arm, "all randomized trajectories", c["TFR"], c["n"]])
    for arm, c in r["exp2"]["secondary"].items():
        rows.append(
            [
                "Exp 2 secondary",
                "conditional FAR",
                arm,
                "accepted trajectories",
                c["conditional_FAR"],
                c["n_accepted"],
            ]
        )
        rows.append(
            [
                "Exp 2 secondary",
                "rewrite rounds (median; right-censored at K)",
                arm,
                "",
                c["rewrite_rounds_median"],
                c["n"],
            ]
        )
        rows.append(
            ["Exp 2 secondary", "non-convergence rate", arm, "", c["non_convergence_rate"], c["n"]]
        )
    for arm, c in r["exp2"]["low_preservation_among_safe"]["by_arm"].items():
        rows.append(
            [
                "Exp 2 secondary",
                "semantic preservation: share below cutpoint among E-SAFE x_T",
                arm,
                "",
                c["share_below_cutpoint"],
                c["n_scored"],
            ]
        )
    for arm, c in r["exp2_mechanism"].items():
        rows.append(
            [
                "mechanism",
                "decision accuracy",
                arm,
                "all decisions",
                c["decision_accuracy"],
                c["n_decisions"],
            ]
        )
        rows.append(
            [
                "mechanism",
                "rewrite improvement rate",
                arm,
                "rewrites of E-UNSAFE text",
                c["rewrite_improvement_rate"],
                c["n_rewrites"],
            ]
        )
    rows.append(["mechanism", "R_t^(a)", "all", "see T08_r_t", None, None])
    return Table(
        "T04_metrics",
        "4",
        "the metrics of II.9.3 at a glance (main accounting)",
        ["layer", "metric", "arm", "cell", "value", "n"],
        rows,
        [
            "TFR and conditional FAR are different quantities (C.10); Exp 1 FA and Exp 2 TFR "
            "are not comparable",
            PILOT_NOTE,
        ],
    )


def t05(r: dict[str, Any], inp: Inputs) -> Table:
    rows = [
        [x["source"], x["e_prompt_sha256"], x["e_version"], x["n_records"], x["single_e"]]
        for x in r["e_identity"]
    ]
    return Table(
        "T05_e_identity",
        "5",
        "every E record names one E (II.9.1)",
        ["source", "e_prompt_sha256", "e_version", "n_records", "single_e"],
        rows,
    )


def t06_effective_n(r: dict[str, Any], inp: Inputs) -> Table:
    e = r["exp1"]["effective_n"]
    return Table(
        "T06_exp1_effective_n",
        "6",
        "N of the Exp 1 primary metric (III.1.2)",
        ["n_states", "n_did_units"],
        [[e["n_states"], e["n_did_units"]]],
        [e["note"]],
    )


def t06_cells(r: dict[str, Any], inp: Inputs) -> Table:
    rows = []
    for key, c in r["exp1"]["cells_main"].items():
        arm, lam = _split(key)
        rows.append([arm, lam, c["n"], c["false_accept_rate"]])
    return Table(
        "T06_exp1_cells",
        "6",
        "FA(arm, lambda) = P(accept) over D_Exp1, main accounting",
        ["arm", "lambda", "n", "false_accept_rate"],
        rows,
    )


def t06_did(r: dict[str, Any], inp: Inputs) -> Table:
    rows = []
    for block in ("primary_did", "secondary_permutedB_did"):
        for acc in ACCOUNTINGS:
            d = r["exp1"][block][acc]
            role = "primary" if block == "primary_did" else "secondary"
            rows.append(
                [role, d["contrast"], acc, "high vs low", d["estimate"], *d["ci95"], d["n_states"]]
            )
    b = r["exp1"]["bootstrap"]
    return Table(
        "T06_exp1_did",
        "6",
        "D-bar: (Y_A1 - Y_B at high) - (at low), per x0; permuted-B - B secondary",
        ["role", "contrast", "accounting", "levels", "D_bar", "ci95_lo", "ci95_hi", "n_states"],
        rows,
        [
            f"95% percentile bootstrap over x0, B = {b['n_boot']}, seed {b['seed']} (III.1.2)",
            PILOT_NOTE,
        ],
    )


def t06_delta(r: dict[str, Any], inp: Inputs) -> Table:
    a = r["exp1"]["secondary_delta_fa_by_lambda_main"]["values"]
    p = r["exp1"]["secondary_permutedB_delta_fa_by_lambda_main"]
    rows = [
        ["A1 - B", a.get("low"), a.get("mid"), a.get("high")],
        ["permuted_B - B", p.get("low"), p.get("mid"), p.get("high")],
    ]
    return Table(
        "T06_exp1_delta_by_lambda",
        "6",
        "Delta_FA(lambda) at three levels (secondary, main accounting)",
        ["contrast", "low", "mid", "high"],
        rows,
        ["lambda_mid is not in the primary D-bar (III.1.2); read for the shape of III.2.3"],
    )


def t06_prompt_identity(r: dict[str, Any], inp: Inputs) -> Table:
    rows = []
    for pair, c in r["exp1_prompt_identity"]["pairwise"].items():
        for arm, x in [("all", c), *sorted(c["by_arm"].items())]:
            rows.append(
                [
                    pair,
                    arm,
                    x["n"],
                    x["prompt_identical"],
                    x["prompt_share"],
                    x["s_obs_identical"],
                    x["s_obs_share"],
                ]
            )
    return Table(
        "T06_prompt_identity",
        "6",
        "C.24: per arm and pair of levels, share of (state, arm) whose controller prompt is "
        "byte for byte the same (main) and whose s_obs is bit for bit the same (auxiliary)",
        [
            "levels",
            "arm",
            "n",
            "prompt_identical",
            "prompt_share",
            "s_obs_identical",
            "s_obs_share",
        ],
        rows,
        ["identical prompts share one controller answer (II.8.2, common random numbers)"],
    )


def t061(r: dict[str, Any], inp: Inputs) -> Table:
    rows = []
    for name, b in (
        ("Exp 1 D-bar and drift ladder", r["exp1"]["bootstrap"]),
        ("Exp 2 TFR contrast and panel", r["exp2"]["bootstrap"]),
    ):
        rows.append([name, b["unit"], b["method"], b["level"], b["n_boot"], b["seed"]])
    return Table(
        "T061_bootstrap",
        "6.1",
        "interval settings (III.1.2)",
        ["analysis", "resampling unit", "method", "level", "B", "seed"],
        rows,
        [PILOT_NOTE],
    )


def t062_h2(r: dict[str, Any], inp: Inputs) -> Table:
    h = r["exp1"]["h2_strata"]
    rows = []
    for name in ("any_verifier_perturbed", "observed_disagreement"):
        for value, s in h[name]["strata"].items():
            for lam in ("low", "mid", "high"):
                n_a1 = (s["cells"].get(f"A1|{lam}") or {}).get("n", 0)
                n_b = (s["cells"].get(f"B|{lam}") or {}).get("n", 0)
                rows.append([name, value, lam, n_a1, n_b, s["delta_fa_by_lambda"].get(lam)])
    notes = [h["note"], f"source of stratifier (i): {h['source']}"]
    notes += [
        f"{n}: cells whose stratum differs across arms = "
        f"{h[n]['cells_differing_across_arms']} (must be 0)"
        for n in ("any_verifier_perturbed", "observed_disagreement")
    ]
    notes.append("with p fixed at 0 / 0.5 / 1, stratifier (i) varies only at lambda_mid (C.27)")
    return Table(
        "T062_h2_strata",
        "6.2",
        "Delta_FA(lambda), A1 - B, within the two arm-independent strata",
        ["stratifier", "stratum", "lambda", "n_A1", "n_B", "delta_fa"],
        rows,
        notes,
    )


def t062_r_ref(r: dict[str, Any], inp: Inputs) -> Table:
    ref = r["r_ref"] or {}
    by_v = ref.get("by_verifier") or {}
    acc = ref.get("accuracy_whole_split") or {}
    rows = [
        [v, by_v.get(v), ref.get("n"), acc.get(v), ref.get("split"), ref.get("lambda")]
        for v in ("v1", "v2")
    ]
    src = str(inp.gate_report_path) if inp.gate_report_path else "no gate report given"
    return Table(
        "T062_r_ref",
        "6.2",
        "r_hat_ref: offline reference, reported alongside H2, never a stratifier (II.1.1)",
        [
            "verifier",
            "detection_rate (E-UNSAFE)",
            "n_e_unsafe",
            "accuracy_whole_split",
            "split",
            "lambda",
        ],
        rows,
        [f"source: {src}"],
    )


def _evidence(tid: str, rep: dict[str, Any], group: str) -> Table:
    rows = []
    for g, arms in sorted(rep["cells"].items(), key=lambda kv: _level_order(kv[0])):
        for arm, by_pair in arms.items():
            for pair in rep["pairs"]:
                c = by_pair[pair]
                rows.append([g, arm, pair, c["n"], c["n_accepted"], c["accept_rate"]])
    notes = [rep["note"]]
    if "cells_differing_across_arms" in rep:
        notes.append(
            "(state, lambda) cells whose observed pair differs across arms: "
            f"{rep['cells_differing_across_arms']} (must be 0)"
        )
    return Table(
        tid,
        "6.3",
        f"accept rate by arm x {group} x observed (y1, y2), main accounting",
        [group, "arm", "(y1,y2)", "n", "n_accepted", "accept_rate"],
        rows,
        notes,
    )


def t063_exp1(r: dict[str, Any], inp: Inputs) -> Table:
    return _evidence("T063_evidence_exp1", r["exp1_evidence_sensitivity"], "lambda")


def t063_exp2(r: dict[str, Any], inp: Inputs) -> Table:
    return _evidence("T063_evidence_exp2", r["exp2_evidence_sensitivity"], "t")


def t07(r: dict[str, Any], inp: Inputs) -> Table:
    return _checks(
        "T07_exp2_construction",
        "7",
        "Exp 2: start, randomization, loop and terminal text",
        r["exp2_checks"],
    )


def t08_tfr(r: dict[str, Any], inp: Inputs) -> Table:
    rows = [[arm, c["n"], c["TFR"]] for arm, c in r["exp2"]["primary_TFR"].items()]
    return Table(
        "T08_tfr_by_arm",
        "8",
        "TFR = P(task_success(x_T) = False) over all randomized trajectories",
        ["arm", "n", "TFR"],
        rows,
    )


def t08_contrast(r: dict[str, Any], inp: Inputs) -> Table:
    c = r["exp2"]["primary_contrast"]
    b = c["bootstrap"]
    rows = [[c["rule"], c["estimate"], *c["ci95"], c["n_by_arm"].get("A1"), c["n_by_arm"].get("B")]]
    return Table(
        "T08_tfr_contrast",
        "8",
        "primary causal claim: TFR(A1) - TFR(B) (positive = B better)",
        ["rule", "estimate", "ci95_lo", "ci95_hi", "n_A1", "n_B"],
        rows,
        [
            "95% percentile bootstrap, trajectories resampled within arm, "
            f"B = {b['n_boot']}, seed {b['seed']}",
            PILOT_NOTE,
        ],
    )


def t08_secondary(r: dict[str, Any], inp: Inputs) -> Table:
    rows = [
        [
            arm,
            c["n"],
            c["n_accepted"],
            c["conditional_FAR"],
            c["non_convergence_rate"],
            c["rewrite_rounds_median"],
        ]
        for arm, c in r["exp2"]["secondary"].items()
    ]
    return Table(
        "T08_secondary",
        "8",
        "secondary: conditional FAR (post-treatment selection), non-convergence, rewrite rounds",
        [
            "arm",
            "n",
            "n_accepted",
            "conditional_FAR",
            "non_convergence_rate",
            "rewrite_rounds_median",
        ],
        rows,
        ["conditional FAR = P(failure | accepted): secondary / mechanism only (II.9.3)"],
    )


def t08_rounds(r: dict[str, Any], inp: Inputs) -> Table:
    rows = []
    for arm, c in r["exp2"]["secondary"].items():
        for n_rw, count in c["rewrite_rounds_distribution"].items():
            rows.append([arm, n_rw, count])
    return Table(
        "T08_rewrite_rounds",
        "8",
        f"distribution of rewrite rounds per trajectory (right-censored at K = {K_MAX_ROUNDS})",
        ["arm", "n_rewrites", "n_trajectories"],
        rows,
    )


def t08_r_t(r: dict[str, Any], inp: Inputs) -> Table:
    rows = []
    for arm, by_t in r["exp2_reliability"]["r_t_by_arm"].items():
        for t, c in by_t.items():
            rows.append([arm, t, c["n"], c["v1"], c["v2"]])
    return Table(
        "T08_r_t",
        "8",
        "R_t^(a): realized verifier reliability per round and arm (an outcome)",
        ["arm", "t", "n", "R_v1", "R_v2"],
        rows,
        [
            "post-treatment: reported as an outcome only, never a stratifier or design factor "
            "(II.1.1)",
            f"rows with c_t: {r['exp2_reliability']['n_rows_with_c_t']}",
        ],
    )


def t08_mechanism(r: dict[str, Any], inp: Inputs) -> Table:
    rows = [
        [
            arm,
            c["n_decisions"],
            c["decision_accuracy"],
            c["n_accepts"],
            c["accept_decision_accuracy"],
            c["n_rewrites"],
            c["rewrite_improvement_rate"],
            c["rewrite_degradation_rate"],
            c["n_rows_without_judgment"],
        ]
        for arm, c in r["exp2_mechanism"].items()
    ]
    return Table(
        "T08_mechanism",
        "8",
        "decision accuracy and rewrite improvement rate (E's round judgments)",
        [
            "arm",
            "n_decisions",
            "decision_accuracy",
            "n_accepts",
            "accept_decision_accuracy",
            "n_rewrites",
            "rewrite_improvement_rate",
            "rewrite_degradation_rate",
            "n_rows_without_judgment",
        ],
        rows,
    )


def t081(r: dict[str, Any], inp: Inputs) -> Table:
    rows = []
    for m, x in r["exp2_panel"]["members"].items():
        c, tfr, miss = x["contrast"], x["TFR_by_arm"], x["n_without_verdict_by_arm"]
        rows.append(
            [
                m,
                (tfr.get("A1") or {}).get("TFR"),
                (tfr.get("B") or {}).get("TFR"),
                c["estimate"],
                *c["ci95"],
                x["same_direction_as_main"],
                miss.get("A1", 0),
                miss.get("B", 0),
            ]
        )
    return Table(
        "T081_panel",
        "8.1",
        "each panel member as its own judge of x_T: TFR(A1) - TFR(B) and its direction",
        [
            "member",
            "TFR_A1",
            "TFR_B",
            "estimate",
            "ci95_lo",
            "ci95_hi",
            "same_direction_as_main",
            "n_without_verdict_A1",
            "n_without_verdict_B",
        ],
        rows,
        [r["exp2_panel"]["note"], f"main estimate: {_cell(r['exp2_panel']['main_estimate'])}"],
    )


def t082(r: dict[str, Any], inp: Inputs) -> Table:
    s = r["exp2"]["low_preservation_among_safe"]
    rows = [
        [arm, c["n_safe"], c["n_scored"], c["n_below_cutpoint"], c["share_below_cutpoint"]]
        for arm, c in s["by_arm"].items()
    ]
    return Table(
        "T082_semantic",
        "8.2",
        "E-SAFE x_T whose semantic preservation is below the cutpoint (descriptive)",
        ["arm", "n_safe", "n_scored", "n_below_cutpoint", "share_below_cutpoint"],
        rows,
        [
            s["definition"],
            f"cutpoint: {s['cutpoint']} (None = still TBD: counts only, share missing, not 0)",
        ],
    )


def _pf(tid: str, rep: dict[str, Any], group: str) -> Table:
    rows = []
    for key, c in sorted(rep["cells"].items(), key=lambda kv: _pf_order(kv[0])):
        arm, g = _split(key)
        rows.append(
            [
                arm,
                g,
                c["n"],
                c["n_parse_fail"],
                c["parse_fail_rate"],
                c["ALERT"],
                c["n_parse_fail_malformed_json"],
                c["parse_fail_by_final_reason"],
            ]
        )
    return Table(
        tid,
        "9",
        f"parse_fail rate by arm x {group}; > {rep['threshold']:.0%} flagged (II.8.2)",
        [
            "arm",
            group,
            "n",
            "n_parse_fail",
            "parse_fail_rate",
            "ALERT",
            "n_malformed_json",
            "by_final_reason",
        ],
        rows,
        rep["alerts"] or ["no cell above the threshold"],
    )


def t09_pf1(r: dict[str, Any], inp: Inputs) -> Table:
    return _pf("T09_parse_fail_exp1", r["exp1_parse_fail"], "lambda")


def t09_pf2(r: dict[str, Any], inp: Inputs) -> Table:
    return _pf("T09_parse_fail_exp2", r["exp2_parse_fail"], "t")


def t09_robustness(r: dict[str, Any], inp: Inputs) -> Table:
    rows = []
    for acc in ACCOUNTINGS:
        d = r["exp1"]["primary_did"][acc]
        rows.append(
            [
                "Exp 1",
                "D-bar (A1 - B)",
                acc,
                d["estimate"],
                *d["ci95"],
                d["n_states"],
                _sign(d["estimate"]),
            ]
        )
    e2 = r["exp2"]
    contrasts = {"main": e2["primary_contrast"], **e2["sensitivity"]}
    for rule in ("main", "parse_fail_as_accept", "complete_case", "refusal_like_as_failure"):
        c = contrasts.get(rule)
        if c is None:
            rows.append(
                ["Exp 2", "TFR(A1) - TFR(B)", rule, None, None, None, None, "not computable"]
            )
            continue
        rows.append(
            [
                "Exp 2",
                "TFR(A1) - TFR(B)",
                rule,
                c["estimate"],
                *c["ci95"],
                c["n"],
                _sign(c["estimate"]),
            ]
        )
    exp2_robust = same_sign(
        [contrasts[x]["estimate"] for x in exp2_tfr.ACCOUNTING_RULES if contrasts.get(x)]
    )
    return Table(
        "T09_robustness",
        "9",
        "main accounting and the II.8.2 sensitivities",
        ["experiment", "estimate", "rule", "value", "ci95_lo", "ci95_hi", "n", "sign"],
        rows,
        [
            "Exp 1 robust (main, (a), (b) same sign): "
            f"{r['exp1']['primary_robust_across_accountings']}",
            f"Exp 2 robust (main, (a), (b) same sign): {e2['robust_across_accountings']} "
            f"(recomputed: {exp2_robust})",
            "refusal_like_as_failure is Exp 2's extra sensitivity, not part of the robustness rule",
        ],
    )


def t10(r: dict[str, Any], inp: Inputs) -> Table:
    rows = [
        [
            x["verifier"],
            x["lambda"],
            x["R_exp1"],
            x["ci95_lo"],
            x["ci95_hi"],
            x["n"],
            x["R_gate"],
            x["exp1_minus_gate"],
        ]
        for x in r["exp1_drift"]["vs_gate"]
    ]
    b = r["exp1_drift"]["ladder"]["bootstrap"]
    return Table(
        "T10_drift",
        "10",
        "R_lambda^(i) on D_Exp1 (observed judgments, all states E-UNSAFE) vs the gate's on D_gate",
        ["verifier", "lambda", "R_exp1", "ci95_lo", "ci95_hi", "n", "R_gate", "exp1_minus_gate"],
        rows,
        [
            f"bootstrap over x0, B = {b['n_boot']}, seed {b['seed']}",
            "reported as a transport diagnostic; no condition is removed (II.6.3)",
        ],
    )


def t11(r: dict[str, Any], inp: Inputs) -> Table:
    tr = r["transport"]
    rows = []
    for k in ("s1", "s2", "m1", "m2", "unsafe"):
        for pop, ks in (
            ("exp1_states", None),
            ("exp2_decision_points", tr[k]["ks"]),
            ("exp2_t_ge_2", tr[k]["ks_t_ge_2"]),
        ):
            s = tr[k][pop]
            rows.append(
                [
                    k,
                    pop,
                    s.get("n"),
                    s.get("mean"),
                    s.get("sd"),
                    s.get("median"),
                    s.get("q10"),
                    s.get("q90"),
                    ks,
                ]
            )
    return Table(
        "T11_transport",
        "11",
        "toxicity (raw scores), margins and UNSAFE share: Exp 1 states vs Exp 2 decision points",
        ["quantity", "population", "n", "mean", "sd", "median", "q10", "q90", "ks_vs_exp1"],
        rows,
        [
            tr["note"],
            f"Exp 2 rows t = 1: {tr['n_exp2_rows_t1']}, t >= 2: {tr['n_exp2_rows_t_ge_2']}",
        ],
    )


def t11_by_round(r: dict[str, Any], inp: Inputs) -> Table:
    rows = []
    for t, qs in r["transport"]["by_round_exp2"].items():
        for k, s in qs.items():
            rows.append([t, k, s.get("n"), s.get("mean"), s.get("median")])
    return Table(
        "T11_transport_by_round",
        "11",
        "Exp 2 decision points by round",
        ["t", "quantity", "n", "mean", "median"],
        rows,
    )


def t12(r: dict[str, Any], inp: Inputs) -> Table:
    s = r["exp1"]["result_shape"]
    rows = []
    for contrast in ("A1 - B", "permuted_B - B"):
        x = s[contrast]
        d = x["delta_fa_by_lambda"]
        rows.append(
            [
                contrast,
                d.get("low"),
                d.get("mid"),
                d.get("high"),
                x["rising_in_lambda"],
                x["negative_at_low"],
            ]
        )
    return Table(
        "T12_shape",
        "12",
        "III.2.3 shapes read off the point estimates",
        ["contrast", "delta_low", "delta_mid", "delta_high", "rising_in_lambda", "negative_at_low"],
        rows,
        [
            s["note"],
            "null result A/B: decision accuracy by arm (T08_mechanism) and error boundary "
            "alignment (T121_*)",
        ],
    )


def _eb(tid: str, rep: dict[str, Any], group: str) -> Table:
    rows = []
    for g, arms in sorted(rep["cells"].items(), key=lambda kv: _level_order(kv[0])):
        for arm, c in arms.items():
            rows.append(
                [
                    g,
                    arm,
                    c["n"],
                    c["n_misaligned"],
                    c["misaligned_share"],
                    c["decision_error_rate_on_misaligned"],
                    c["decision_error_rate_overall"],
                ]
            )
    return Table(
        tid,
        "12.1",
        f"error boundary alignment by arm x {group} (III.3, main accounting)",
        [
            group,
            "arm",
            "n",
            "n_misaligned",
            "misaligned_share",
            "error_rate_on_misaligned",
            "error_rate_overall",
        ],
        rows,
        [rep["definition"]],
    )


def t121_exp1(r: dict[str, Any], inp: Inputs) -> Table:
    return _eb("T121_error_boundary_exp1", r["exp1_error_boundary"], "lambda")


def t121_exp2(r: dict[str, Any], inp: Inputs) -> Table:
    return _eb("T121_error_boundary_exp2", r["exp2_error_boundary"], "t")


def t14(r: dict[str, Any], inp: Inputs) -> Table:
    s = r["settings"]
    rows = [
        ["FAKE", inp.fake],
        ["exp1 run dir", str(inp.exp1_dir)],
        ["exp1 decisions sha256", _sha256(inp.exp1_dir / "decisions.jsonl")],
        ["exp2 run dir", str(inp.exp2_dir)],
        ["exp2 decisions sha256", _sha256(inp.exp2_dir / "decisions.jsonl")],
        ["exp2 trajectories sha256", _sha256(inp.exp2_dir / "trajectories.jsonl")],
        ["gate report", str(inp.gate_report_path) if inp.gate_report_path else None],
        ["bootstrap seed Exp 1", s["seed_exp1"]],
        ["bootstrap seed Exp 2", s["seed_exp2"]],
        ["bootstrap B", s["n_boot"]],
        ["semantic cutpoint", s["semantic_cutpoint"]],
    ]
    return Table(
        "T14_provenance", "14", "what these tables were computed from", ["item", "value"], rows
    )


BUILDERS: dict[str, Builder] = {
    "T01_h1_direction": t01_h1_direction,
    "T03_exp1_construction": t03,
    "T04_metrics": t04_metrics,
    "T05_e_identity": t05,
    "T06_exp1_effective_n": t06_effective_n,
    "T06_exp1_cells": t06_cells,
    "T06_exp1_did": t06_did,
    "T06_exp1_delta_by_lambda": t06_delta,
    "T06_prompt_identity": t06_prompt_identity,
    "T061_bootstrap": t061,
    "T062_h2_strata": t062_h2,
    "T062_r_ref": t062_r_ref,
    "T063_evidence_exp1": t063_exp1,
    "T063_evidence_exp2": t063_exp2,
    "T07_exp2_construction": t07,
    "T08_tfr_by_arm": t08_tfr,
    "T08_tfr_contrast": t08_contrast,
    "T08_secondary": t08_secondary,
    "T08_rewrite_rounds": t08_rounds,
    "T08_r_t": t08_r_t,
    "T08_mechanism": t08_mechanism,
    "T081_panel": t081,
    "T082_semantic": t082,
    "T09_parse_fail_exp1": t09_pf1,
    "T09_parse_fail_exp2": t09_pf2,
    "T09_robustness": t09_robustness,
    "T10_drift": t10,
    "T11_transport": t11,
    "T11_transport_by_round": t11_by_round,
    "T12_shape": t12,
    "T121_error_boundary_exp1": t121_exp1,
    "T121_error_boundary_exp2": t121_exp2,
    "T14_provenance": t14,
}


def build_tables(results: dict[str, Any], inp: Inputs) -> list[Table]:
    """Every table of :data:`SECTIONS`, in section order."""
    listed = [t for s in SECTIONS for t in s.tables]
    missing = sorted(set(listed) ^ set(BUILDERS))
    if missing:
        raise RuntimeError(f"SECTIONS and BUILDERS disagree on {missing}")
    return [BUILDERS[t](results, inp) for t in listed]


# ---- output ----------------------------------------------------------------------------


def _cell(x: Any) -> str:
    if x is None:
        return "--"
    if isinstance(x, bool):
        return "yes" if x else "no"
    if isinstance(x, float):
        return "nan" if x != x else f"{x:.4f}"
    if isinstance(x, (dict, list)):
        return json.dumps(x, ensure_ascii=False, default=str)
    return str(x).replace("|", "\\|")


def render_markdown(tables: Sequence[Table], *, fake: bool) -> str:
    out: list[str] = []
    title = (
        "FAKE -- preregistered tables on FAKE data (step 10)" if fake else "Preregistered tables"
    )
    out += [f"# {title}", ""]
    if fake:
        out += [
            "**FAKE.** Every number below comes from step 10's fake logs (analysis.fake_data): "
            "arm labels shuffled, no real label, score or controller output. None of it is a "
            "result. The formal pipeline refuses to read these files.",
            "",
        ]
    out += [PILOT_NOTE, ""]
    by_section: dict[str, list[Table]] = {}
    for t in tables:
        by_section.setdefault(t.section, []).append(t)
    for spec in SECTIONS:
        out += [f"## Section {spec.section}: {spec.title}", ""]
        if spec.functions:
            out += ["Functions: " + ", ".join(f"`{f}`" for f in spec.functions), ""]
        for reason in spec.not_implemented:
            out += [f"Not computed here: {reason}.", ""]
        for t in by_section.get(spec.section, []):
            out += [f"### {'FAKE ' if fake else ''}{t.id}: {t.title}", ""]
            out.append("| " + " | ".join(t.columns) + " |")
            out.append("|" + "---|" * len(t.columns))
            for row in t.rows:
                out.append("| " + " | ".join(_cell(x) for x in row) + " |")
            out.append("")
            for n in t.notes:
                out.append(f"* {n}")
            if t.notes:
                out.append("")
    return "\n".join(out).rstrip() + "\n"


def write_tables(
    tables: Sequence[Table], results: dict[str, Any], out_dir: Path, *, fake: bool
) -> list[Path]:
    """One JSON per table, the Markdown of all of them, and the full nested results. Fake
    output is named ``FAKE_*`` and carries ``"FAKE": true``."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    prefix = FAKE_FILE_PREFIX if fake else ""
    written = []

    def dump(name: str, obj: Any) -> None:
        p = out_dir / f"{prefix}{name}"
        p.write_text(
            json.dumps(obj, ensure_ascii=False, indent=1, default=str),
            encoding="utf-8",
            newline="\n",
        )
        written.append(p)

    for t in tables:
        dump(f"{t.id}.json", {"FAKE": fake, **asdict(t)})
    dump("analysis_full.json", {"FAKE": fake, **results})
    dump(
        "sections.json",
        {"FAKE": fake, "sections": [asdict(s) for s in SECTIONS]},
    )
    md = out_dir / f"{prefix}tables.md"
    md.write_text(render_markdown(tables, fake=fake), encoding="utf-8", newline="\n")
    written.append(md)
    return written


# ---- formal entry point ----------------------------------------------------------------


def main(argv: Sequence[str] | None = None) -> int:  # pragma: no cover - thin CLI
    from datetime import UTC, datetime

    from rafr.core import paths
    from rafr.core.config import get, load_config, master_seed
    from rafr.core.seeding import component_seed

    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--exp1-run-dir", type=Path, required=True)
    ap.add_argument("--exp2-run-dir", type=Path, required=True)
    ap.add_argument(
        "--gate-report", type=Path, default=paths.DATA / "diagnostics" / "step09_report.json"
    )
    ap.add_argument("--out-dir", type=Path, default=None)
    args = ap.parse_args(argv)
    cfgs = load_config()
    inp = load_inputs(
        args.exp1_run_dir, args.exp2_run_dir, gate_report=args.gate_report, fake=False
    )
    cut = get(cfgs["exp2"], "analysis.semantic_preservation.cutpoint")
    n_exp1 = int(get(cfgs["data"], "splits.sizes.d_exp1"))
    results = compute(
        inp,
        # the seeds step11 / step12 use for their own reports (III.1.2: recorded, on disk)
        seed_exp1=component_seed(master_seed(cfgs), "exp1_run"),
        seed_exp2=component_seed(master_seed(cfgs), "exp2_run"),
        n_boot=int(get(cfgs["exp1"], "analysis.bootstrap.n_boot", BOOTSTRAP_B)),
        semantic_cutpoint=None if cut in (None, "TBD") else float(cut),
        n_exp1=n_exp1,
        n_per_arm=n_exp1 // 2,
    )
    tables = build_tables(results, inp)
    out = args.out_dir or paths.run_dir(
        f"analysis_{datetime.now(tz=UTC).strftime('%Y%m%dT%H%M%SZ')}"
    )
    for p in write_tables(tables, results, out, fake=False):
        print(f"wrote {p}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
