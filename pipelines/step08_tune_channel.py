"""Step 08 — check C_lambda on D_fit (protocol II.5, II.6.1, II.6.2 baseline; 0.5 step 8).

The three levels are FIXED ex ante (protocol II.5.1, C.27; 2026-09-26): p = 0 / 0.5 / 1
at low / mid / high, the same for both verifiers, from ``configs/channel.yaml: degrade.levels``
(``degrade.levels_from_config`` refuses anything else). This step solves nothing: it checks.

Gate metric (d) (decisions.md D-2026-09-24 section 1): R_lambda^(i) is computed on the E-UNSAFE
states, so every reliability number this step uses is taken on D_fit's E-UNSAFE subset:

1. fit q (independent resampling, group 6a) on the UNLABELED pool (scores only, no labels);
2. reachability, per verifier, on D_fit's E-UNSAFE subset (label purpose CHANNEL_CHECK): the
   baseline TPR_i (no perturbation); the all-resampled floor r_i = P_{s~q_i}(s >= tau_i),
   computed exactly from the pool's score distribution and no label
   (``MarginModel.prob_at_least``); the span TPR_i - r_i and whether it is >= 2 delta_R, i.e.
   whether the expected adjacent gap (TPR_i - r_i)/2 is >= delta_R (II.5.2, II.5.3). If either
   span is short the step STOPS: it prints the numbers, records them in the run directory and
   writes nothing to frozen/ or data/diagnostics/ (II.6.1: do not enter the gate; the authors decide
   between another verifier and another delta_R);
3. with the fixed levels: the realized R_lambda^(i) on the same subset, adjacent gaps vs delta_R
   and their CI (resampling E-UNSAFE states only); accuracy and balanced accuracy on ALL of D_fit,
   descriptively; the five cue diagnostics per lambda (lambda = 0 is the baseline, II.6.2) on the
   population(s) named by ``channel.diagnostics.diagnostics_population``;
4. formal run: write ``frozen/channel.candidate.json`` and ``data/diagnostics/step08_report.json``.

delta_R is read from ``channel.gate.delta_r``. Its formal value is the authors', written at step 8,
and is TBD until he sets it; the formal run refuses while it is (II.10). ``--reachability-only``
runs steps 1-3 against the formal delta_R if set, else the PROVISIONAL one
(``channel.gate.delta_r_provisional``, labelled as such), and writes only
``runs/step08_<ts>/reachability.json`` -- nothing in frozen/ or data/diagnostics/.
"""

from __future__ import annotations

# A direct ``python pipelines/<step>.py`` puts pipelines/ on sys.path, not the repo root,
# so the repo-root packages (analysis, annotation, pipelines) would not import; rafr is
# installed and unaffected. Covered by tests/test_pipeline_entrypoints.py.
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
import json
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from typing import Any

from pipelines import _validity_io as io
from rafr.channel.degrade import (
    DegradeChannel,
    DegradeConfigError,
    LevelParams,
    expected_ladder,
    levels_from_config,
)
from rafr.channel.margin_fit import REPORT_BINS, MarginModel, fit_margin_model
from rafr.contracts.schemas import BOOTSTRAP_B, TBD, Lambda, Split
from rafr.core import paths
from rafr.core.config import assert_frozen, get, load_config, master_seed
from rafr.core.seeding import component_seed
from rafr.core.textio import crlf_rewrite_notice
from rafr.data.labels import LabelPurpose
from rafr.logging.snapshot import git_head, write_snapshot
from rafr.validity import cue_diagnostics as CD
from rafr.validity import gate as G
from rafr.validity import metrics as mt
from rafr.validity.tau import load_tau

CANDIDATE = paths.FROZEN / "channel.candidate.json"

LEVELS = (Lambda.NONE, Lambda.LOW, Lambda.MID, Lambda.HIGH)

DIAGNOSTICS_POPULATIONS: dict[str, tuple[str, ...]] = {
    "all": ("all",),
    "unsafe": ("unsafe",),
    "both": ("unsafe", "all"),
}
"""``channel.diagnostics.diagnostics_population`` -> the population(s) the five II.6.2
diagnostics (and their lambda = 0 baseline on D_fit) are computed on, primary first. ``all`` =
every state of the split; ``unsafe`` = its E-UNSAFE states, the population R is computed on (gate
metric (d)). With ``both`` the E-UNSAFE subset is primary and the whole split secondary (the authors
2026-09-24, decisions.md D-2026-09-24c)."""


def diagnostics_roles(populations: tuple[str, ...]) -> dict[str, str]:
    """``{population: "primary" | "secondary"}``, written next to the diagnostics."""
    return {p: ("primary" if i == 0 else "secondary") for i, p in enumerate(populations)}


FORMAL_SECTIONS: tuple[str, ...] = (
    "models.verifiers",
    "channel.degrade",
    "channel.gate",
    "channel.diagnostics",
)
REACHABILITY_SECTIONS: tuple[str, ...] = (
    "models.verifiers",
    "channel.degrade",
    "channel.diagnostics",
)
"""``--reachability-only`` measures everything the formal run does but needs no formal delta_R,
so ``channel.gate`` (whose ``delta_r`` stays TBD until the authors write it at step 8) is not
checked."""


def fixed_levels(channel_cfg) -> dict[Lambda, LevelParams]:
    """The three levels, fixed ex ante (II.5.1, C.27): ``channel.degrade.levels`` must hold
    p = 0 / 0.5 / 1 for both verifiers. Nothing measured here can change them."""
    try:
        return levels_from_config(get(channel_cfg, "degrade", {}) or {})
    except DegradeConfigError as e:
        raise SystemExit(f"[step08] {e}") from e


def levels_record(levels: dict[Lambda, LevelParams]) -> dict[str, Any]:
    """``levels_p`` in the report: the fixed values and where they come from (no solve)."""
    return {
        "fixed": True,
        "rule": "fixed ex ante, the same for both verifiers: p = 0 / 0.5 / 1 at low / mid / "
        "high (protocol II.5.1, C.27, 2026-09-26); not solved from data, not hand-filled",
        "source": "configs/channel.yaml: degrade.levels",
        "values": {lam.value: p.to_dict() for lam, p in levels.items()},
    }


def ci_n_boot(channel_cfg) -> int:
    """``channel.gate.ci.n_boot``; B = 10000 is locked by III.1.2 and only lowered in tests."""
    return int(get(channel_cfg, "gate.ci.n_boot", BOOTSTRAP_B))


def ci_seed(configs) -> int:
    """The gate CI's seed, derived from the one master seed (component ``gate_ci``)."""
    return component_seed(master_seed(configs), "gate_ci")


def diagnostics_populations(channel_cfg) -> tuple[str, ...]:
    """The population(s) of the II.6.2 diagnostics. No default: TBD (or anything else) refuses."""
    value = get(channel_cfg, "diagnostics.diagnostics_population", TBD)
    if value not in DIAGNOSTICS_POPULATIONS:
        raise SystemExit(
            "channel.diagnostics.diagnostics_population must be one of "
            f"{sorted(DIAGNOSTICS_POPULATIONS)}, got {value!r} (not decided yet: the authors)"
        )
    return DIAGNOSTICS_POPULATIONS[value]


def delta_r_for(channel_cfg, *, reachability_only: bool) -> tuple[float, str]:
    """``(delta_R, "formal" | "provisional")``. The formal value is required for everything but
    ``--reachability-only``, which falls back to the provisional one and says so."""
    formal = get(channel_cfg, "gate.delta_r", TBD)
    if formal not in (None, TBD):
        return float(formal), "formal"
    if not reachability_only:
        raise SystemExit("channel.gate.delta_r is TBD: the formal delta_R is the authors' (App. D)")
    prov = get(channel_cfg, "gate.delta_r_provisional")
    if prov in (None, TBD):
        raise SystemExit("channel.gate.delta_r_provisional is not set")
    return float(prov), "provisional"


def apply_channel(
    channel, states, s1: dict, s2: dict, lam: Lambda
) -> tuple[list[float], list[float]]:
    a, b = [], []
    for s in states:
        t1, t2 = channel(s.state_id, s1[s.state_id], s2[s.state_id], lam)
        a.append(t1)
        b.append(t2)
    return a, b


@dataclass
class LevelEvaluation:
    """Everything step08 and step09 read off one split for one channel.

    ``r`` / ``vectors`` are gate metric (d): E-UNSAFE states only, the vectors in split order
    (what ``gate.ladder_ci`` resamples). ``r_none`` is the same quantity with no perturbation
    (r_ref in step09). ``descriptive`` is accuracy / balanced accuracy on ALL states.
    ``diagnostics[population][lambda]`` are the five II.6.2 items.
    """

    n_states: int
    n_unsafe: int
    r: dict[str, dict[Lambda, float]]
    vectors: dict[str, dict[Lambda, list[int]]]
    r_none: dict[str, float]
    p_at_least_one_missed: dict[Lambda, float]
    descriptive: dict[str, dict[str, dict[str, float]]]
    diagnostics: dict[str, dict[str, CD.CueReport]] = field(default_factory=dict)


def evaluate_levels(
    channel,
    states,
    s1,
    s2,
    labels,
    tau1,
    tau2,
    *,
    high_margin_quantile: float,
    n_bins: int,
    populations: tuple[str, ...] = (),
) -> LevelEvaluation:
    """R per lambda per verifier on the E-UNSAFE subset (gate metric (d)), the whole-split
    descriptive accuracy, and the cue diagnostics on each requested population."""
    y = [labels[s.state_id] for s in states]
    observed = {lam: apply_channel(channel, states, s1, s2, lam) for lam in LEVELS}
    vectors = G.detection_vectors(y, observed, tau1, tau2)
    r_all = G.reliability_by_level(vectors)
    ladder_vectors = {v: {lam: vectors[v][lam] for lam in G.LADDER} for v in G.VERIFIER_KEYS}
    missed = {
        lam: 1.0
        - sum(a * b for a, b in zip(vectors["v1"][lam], vectors["v2"][lam], strict=True))
        / len(vectors["v1"][lam])
        for lam in G.LADDER
    }
    diags: dict[str, dict[str, CD.CueReport]] = {}
    for pop in populations:
        idx = [i for i, lab in enumerate(y) if pop == "all" or lab == 1]
        diags[pop] = {}
        for lam, (t1, t2) in observed.items():
            inputs = CD.CueInputs(
                [t1[i] for i in idx], [t2[i] for i in idx], tau1, tau2, [y[i] for i in idx]
            )
            diags[pop][lam.value] = CD.compute(
                inputs, high_margin_quantile=high_margin_quantile, n_bins=n_bins
            )
    return LevelEvaluation(
        n_states=len(states),
        n_unsafe=sum(y),
        r={v: {lam: r_all[v][lam] for lam in G.LADDER} for v in G.VERIFIER_KEYS},
        vectors=ladder_vectors,
        r_none={v: r_all[v][Lambda.NONE] for v in G.VERIFIER_KEYS},
        p_at_least_one_missed=missed,
        descriptive=G.descriptive_accuracy(y, observed, tau1, tau2),
        diagnostics=diags,
    )


def reachability(
    q: dict[str, MarginModel], states, s1, s2, labels, tau1, tau2, delta_r: float
) -> dict[str, dict[str, Any]]:
    """Per verifier on D_fit's E-UNSAFE subset (gate metric (d); group 6b report):

    * ``baseline_tpr`` -- TPR_i, 1[s_i >= tau_i] over the E-UNSAFE states, no perturbation;
    * ``all_resampled_floor`` -- r_i = P_{s~q_i}(s >= tau_i): what TPR becomes when EVERY state
      is resampled, from the pool's score distribution alone (no label);
    * ``span`` = TPR_i - r_i and whether it is >= 2 * delta_R (II.5.2, II.5.3): with p fixed at
      0 / 0.5 / 1 the expected adjacent gap is span / 2, so this is "expected gap >= delta_R";
    * ``descriptive_whole_split`` -- the same three quantities in accuracy and balanced-accuracy
      units over ALL of D_fit (II.5.2, v1.0: reported descriptively, next to the detection-rate
      ones the ladder is checked in). With every state resampled the judgment is independent of
      the label, so the accuracy floor is ``r_i * pi + (1 - r_i) * (1 - pi)`` (pi = D_fit's
      E-UNSAFE share) and the balanced-accuracy floor is 1/2 (TPR r_i, TNR 1 - r_i).
    """
    unsafe = [s.state_id for s in states if labels[s.state_id] == 1]
    if not unsafe:
        raise SystemExit("D_fit has no E-UNSAFE state: R (gate metric (d)) is undefined")
    y = [int(labels[s.state_id]) for s in states]
    pi = len(unsafe) / len(states)
    out: dict[str, dict[str, Any]] = {}
    for v, s, tau in (("v1", s1, tau1), ("v2", s2, tau2)):
        tpr = sum(int(s[sid] >= tau) for sid in unsafe) / len(unsafe)
        floor = q[v].prob_at_least(tau)
        span = tpr - floor
        pred = [int(s[st.state_id] >= tau) for st in states]
        base_acc, base_ba = mt.accuracy(y, pred), mt.balanced_accuracy(y, pred)
        floor_acc, floor_ba = floor * pi + (1 - floor) * (1 - pi), 0.5
        out[v] = {
            "population": G.GATE_POPULATION,
            "n_states": len(unsafe),
            "tau": tau,
            "baseline_tpr": tpr,
            "all_resampled_floor": floor,
            "floor_definition": "r_i = P_{s~q_i}(s >= tau_i), q_i = the unlabeled pool's scores",
            "span": span,
            "expected_adjacent_gap": span / 2,
            "two_gaps_of_delta_r_reachable": span >= 2 * delta_r - G.GAP_TOLERANCE,
            "descriptive_whole_split": {
                "n_states": len(states),
                "e_unsafe_share": pi,
                "accuracy": {
                    "baseline": base_acc,
                    "all_resampled_floor": floor_acc,
                    "span": base_acc - floor_acc,
                },
                "balanced_accuracy": {
                    "baseline": base_ba,
                    "all_resampled_floor": floor_ba,
                    "span": base_ba - floor_ba,
                },
                "note": (
                    "II.5.2 (v1.0): descriptive only; the ladder is checked in detection-rate "
                    "units on the E-UNSAFE subset"
                ),
            },
        }
    return out


def _print_reachability(reach: dict[str, dict[str, Any]], delta_r: float, status: str) -> None:
    for v, rc in reach.items():
        print(
            f"[step08] {v}: E-UNSAFE n={rc['n_states']}  TPR {rc['baseline_tpr']:.3f}  "
            f"floor r {rc['all_resampled_floor']:.3f}  span {rc['span']:.3f}  "
            f"(2*delta_R = {2 * delta_r:.3f}, {status}: "
            f"{'reachable' if rc['two_gaps_of_delta_r_reachable'] else 'NOT reachable'})"
        )
        d = rc["descriptive_whole_split"]
        print(
            f"[step08] {v}: descriptive, all {d['n_states']} states: accuracy "
            f"{d['accuracy']['baseline']:.3f} -> floor {d['accuracy']['all_resampled_floor']:.3f} "
            f"(span {d['accuracy']['span']:.3f}); balanced accuracy "
            f"{d['balanced_accuracy']['baseline']:.3f} -> floor 0.500 "
            f"(span {d['balanced_accuracy']['span']:.3f})"
        )


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--reachability-only",
        action="store_true",
        help="the whole check (TPR, floor, span; the realized ladder and the cue diagnostics under "
        "the fixed p) against the formal delta_R if set, else the provisional one; writes "
        "runs/step08_<ts>/reachability.json only, nothing in frozen/ or data/diagnostics/",
    )
    args = ap.parse_args(argv)
    cfgs = load_config()
    only = args.reachability_only
    assert_frozen(cfgs, only_sections=list(REACHABILITY_SECTIONS if only else FORMAL_SECTIONS))
    ch_cfg = cfgs["channel"]
    delta_r, delta_r_status = delta_r_for(ch_cfg, reachability_only=only)
    populations = diagnostics_populations(ch_cfg)
    levels = fixed_levels(ch_cfg)
    if not paths.FROZEN_TAU.exists():
        raise SystemExit("frozen/tau.json missing; run step07 first")
    run_dir = paths.run_dir(f"step08_{datetime.now(tz=UTC).strftime('%Y%m%dT%H%M%SZ')}")
    write_snapshot(run_dir, step="step08", configs=cfgs, extra={"reachability_only": only})

    v1, v2 = io.verifier_ids(cfgs["models"])
    taus = load_tau(paths.FROZEN_TAU)
    tau1, tau2 = taus["v1"].tau, taus["v2"].tau

    # 1) q on the unlabeled pool (no labels involved)
    pool = io.load_states(Split(get(ch_cfg, "degrade.q.source_split", "unlabeled_pool")))
    q = {
        "v1": fit_margin_model(list(io.load_scores_for(pool, v1).values()), tau1, v1, REPORT_BINS),
        "v2": fit_margin_model(list(io.load_scores_for(pool, v2).values()), tau2, v2, REPORT_BINS),
    }

    # 2) reachability on D_fit's E-UNSAFE subset (gate metric (d)); a short span stops here
    states = io.load_states(Split.D_FIT)
    labels = io.load_labels_for(states, Split.D_FIT, LabelPurpose.CHANNEL_CHECK)
    s1 = io.load_scores_for(states, v1)
    s2 = io.load_scores_for(states, v2)
    reach = reachability(q, states, s1, s2, labels, tau1, tau2, delta_r)
    tpr = {v: reach[v]["baseline_tpr"] for v in ("v1", "v2")}
    floor = {v: reach[v]["all_resampled_floor"] for v in ("v1", "v2")}
    head = {
        "split": Split.D_FIT.value,
        "gate_population": G.GATE_POPULATION,
        "delta_r": delta_r,
        "delta_r_status": delta_r_status,
        "levels_p": levels_record(levels),
        "expected_ladder": expected_ladder(tpr, floor, levels),
        "reachability": reach,
    }
    _print_reachability(reach, delta_r, delta_r_status)
    short = [v for v in ("v1", "v2") if not reach[v]["two_gaps_of_delta_r_reachable"]]
    if short:
        stop = {**head, "stopped": True, "short_span": short, "git_head": git_head()}
        (run_dir / "reachability.json").write_text(
            json.dumps({**stop, "run_dir": str(run_dir)}, ensure_ascii=False, indent=1),
            encoding="utf-8",
            newline="\n",
        )
        raise SystemExit(
            f"[step08] span TPR_i - r_i < 2 * delta_R = {2 * delta_r:.3f} ({delta_r_status}) for "
            f"{short}: the expected adjacent gap misses delta_R (II.5.3). Do not enter the gate "
            "(II.6.1); p stays fixed. Nothing written to frozen/ or data/diagnostics/; the authors "
            f"decides between another verifier and another delta_R. Numbers: {run_dir}"
        )

    # 3) the realized ladder, descriptive accuracy and diagnostics under the fixed levels
    chan_seed = component_seed(master_seed(cfgs), "channel")
    channel = DegradeChannel(levels, q, tau1, tau2, chan_seed)
    ev = evaluate_levels(
        channel, states, s1, s2, labels, tau1, tau2,
        high_margin_quantile=float(get(ch_cfg, "diagnostics.high_margin_quantile")),
        n_bins=int(get(ch_cfg, "diagnostics.n_bins", 10)),
        populations=populations,
    )  # fmt: skip
    # CI first, verdict second: ladder_check gets the interval already formed and only
    # carries it (II.6.1, v0.8 - the step 8 read is point estimates as well).
    ci = G.ladder_ci(ev.vectors, n_boot=ci_n_boot(ch_cfg), seed=ci_seed(cfgs))
    ladder = G.ladder_check(
        ev.r,
        delta_r,
        ev.p_at_least_one_missed,
        ci=ci,
        n_states=ev.n_unsafe,
        descriptive=ev.descriptive,
    )
    report = {
        **head,
        "n_states": ev.n_states,
        "n_e_unsafe": ev.n_unsafe,
        "derived_p_z": {lam.value: p.derived_mode_probs() for lam, p in channel.levels.items()},
        "ladder_on_d_fit": G.ladder_result_to_dict(ladder),
        "descriptive_whole_split": ev.descriptive,
        "diagnostics_population": list(populations),
        "diagnostics_roles": diagnostics_roles(populations),
        "cue_diagnostics": {
            pop: {k: asdict(rep) for k, rep in by_lam.items()}
            for pop, by_lam in ev.diagnostics.items()
        },
        "git_head": git_head(),
        "run_dir": str(run_dir),
    }
    print(f"[step08] fixed p by level (II.5.1, C.27): {json.dumps(head['levels_p']['values'])}")
    print(f"[step08] expected ladder: {json.dumps(head['expected_ladder'])}")
    print(f"[step08] derived P(z) by level: {json.dumps(report['derived_p_z'])}")
    print(
        f"[step08] R on D_fit's {ev.n_unsafe} E-UNSAFE states: {json.dumps(ladder.r)}  "
        f"gaps: {json.dumps(ladder.gaps)}"
    )
    print(
        f"[step08] ladder spacing >= delta_R on D_fit: {ladder.passed}"
        + ("" if ladder.passed else f"  -> {ladder.failed_conditions}")
    )
    for pop, by_lam in ev.diagnostics.items():
        dis = {k: round(v.disagreement_rate, 3) for k, v in by_lam.items()}
        print(f"[step08] disagreement rate by lambda ({pop}): {dis}")
    if not ladder.passed:
        print(
            "[step08] realized spacing misses delta_R although the span carries it in "
            "expectation: the per-state draws are finite. Check n and the diagnostics before "
            "entering the gate (II.6.1)."
        )

    # 4) outputs: the run directory only for --reachability-only, frozen/ + data/ otherwise
    if only:
        (run_dir / "reachability.json").write_text(
            json.dumps({**report, "stopped": False}, ensure_ascii=False, indent=1),
            encoding="utf-8",
            newline="\n",
        )
        print(f"[step08] reachability only -> {run_dir / 'reachability.json'}; nothing frozen")
        return 0
    channel.save(
        CANDIDATE, extra={"status": "candidate", "git_head": git_head(), "run_dir": str(run_dir)}
    )
    out_dir = io.diagnostics_dir(ch_cfg)
    out_dir.mkdir(parents=True, exist_ok=True)
    report_path = out_dir / "step08_report.json"
    with crlf_rewrite_notice(report_path):  # the 2026-09-26 report was written CRLF
        report_path.write_text(
            json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8", newline="\n"
        )
    print(f"[step08] candidate channel -> {CANDIDATE}; report -> {out_dir / 'step08_report.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
