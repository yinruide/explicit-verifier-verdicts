"""Step 09 — one-shot manipulation-validity gate (paper Sec. 4.3, App. A; step 9).

* attempt 1 uses D_gate; attempt 2 (only after a recorded failure) uses D_gate_reserve;
  a third attempt is refused (``frozen/gate_attempts.json``);
* the five descriptive diagnostics are computed and WRITTEN before the ladder verdict is
  even printed (paper App. A: recorded before anyone looks at A1 - B);
* R_lambda^(i) is gate metric (d) (section 1): the share of the
  split's E-UNSAFE states with 1[s~_i >= tau_i]. Hard check: the four adjacent-gap POINT
  estimates must all be >= delta_R; the 95% percentile bootstrap CIs of the six R and the four
  gaps resample E-UNSAFE states only (unit = state; B and seed written to the attempt record),
  are reported alongside and take no part in the verdict, and no multiplicity correction is
  applied (paper Sec. 4.3, App. B). Accuracy and balanced accuracy on the WHOLE split are reported
  next to it, descriptively;
* the five diagnostics are computed on the population(s) ``channel.diagnostics.
  diagnostics_population`` names (``all`` / ``unsafe`` / ``both``; TBD refuses);
* on pass, the candidate channel becomes ``frozen/channel.json``; the authors freeze it (commit and
git tag
  ``channel-frozen``) before Exp 1;
* before any label is read the attempt prints what it reads (2026-09-27): the split and its
  size, the score tables (file, raw-byte sha256, rows, one identity), the label purpose (GATE,
  one read), the reserve's lock, tau, the channel file and seed, delta_R; afterwards the paired
  correctness correlation across levels (paper App. B, descriptive) and one report,
  ``data/diagnostics/step09_report.json``, pass or fail.

the labels are E's own judgments (y* = E, paper Sec. 3), so the old ``--e-judgments`` check of E
against y* is gone, and so is the module that computed E's agreement with human labels
(retired that use; deleted 2026-09-24).
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
import shutil
from dataclasses import asdict
from datetime import UTC, datetime
from typing import Any

from pipelines import _validity_io as io
from pipelines.step08_tune_channel import (
    CANDIDATE,
    FORMAL_SECTIONS,
    ci_n_boot,
    ci_seed,
    delta_r_for,
    diagnostics_populations,
    diagnostics_roles,
    evaluate_levels,
)
from rafr.channel.degrade import DegradeChannel
from rafr.contracts.schemas import Lambda, Split
from rafr.core import paths
from rafr.core.config import assert_frozen, find_tbd, get, load_config
from rafr.data import labels as L
from rafr.logging.snapshot import git_head, write_snapshot
from rafr.logging.writer import read_jsonl
from rafr.validity import gate as G
from rafr.validity.tau import load_tau
from rafr.verifiers import scores_table_path


def score_table_identity(states, verifier_id: str) -> dict[str, Any]:
    """Where this split's scores come from: the table file, its raw-byte sha256, how many of the
    split's states it holds and the (model_version, prompt_hash) of those rows -- one identity."""
    path = scores_table_path(verifier_id, paths.DATA_SCORES)
    want = {s.state_id for s in states}
    idents: set[tuple[str, str | None]] = set()
    n = 0
    for r in read_jsonl(path):
        if r["state_id"] in want:
            n += 1
            idents.add((str(r.get("model_version")), r.get("prompt_hash")))
    if len(idents) != 1:
        raise SystemExit(f"{path}: {len(idents)} identities among this split's rows; want one")
    ((mv, ph),) = idents
    return {
        "table": str(path),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "rows_for_split": n,
        "model_version": mv,
        "prompt_hash": ph,
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.parse_args(argv)
    cfgs = load_config()
    assert_frozen(cfgs, only_sections=list(FORMAL_SECTIONS))
    if not CANDIDATE.exists():
        raise SystemExit(f"{CANDIDATE} missing; run step08 first")
    ch_cfg = cfgs["channel"]
    delta_r, _ = delta_r_for(ch_cfg, reachability_only=False)
    populations = diagnostics_populations(ch_cfg)
    split = G.next_attempt_split(paths.FROZEN_GATE_ATTEMPTS)  # refuses a third attempt
    L.check_split_usable(split, L.Operation.FIT, gate_attempts_path=paths.FROZEN_GATE_ATTEMPTS)
    run_dir = paths.run_dir(f"step09_{datetime.now(tz=UTC).strftime('%Y%m%dT%H%M%SZ')}")
    write_snapshot(run_dir, step="step09", configs=cfgs, extra={"split": split.value})

    v1, v2 = io.verifier_ids(cfgs["models"])
    taus = load_tau(paths.FROZEN_TAU)
    channel = DegradeChannel.load(CANDIDATE)
    states = io.load_states(split)

    # 0) preflight (2026-09-27): what this attempt reads, printed and checked before any
    # label is read. The reserve is neither read nor scored unless it is this attempt's split.
    want_n = get(cfgs["data"], f"splits.sizes.{split.value}")
    if want_n is not None and len(states) != int(want_n):
        raise SystemExit(f"{split.value}: {len(states)} states, configured {want_n}")
    tables = {"v1": score_table_identity(states, v1), "v2": score_table_identity(states, v2)}
    reserve_locked = not L.reserve_unlocked(paths.FROZEN_GATE_ATTEMPTS)
    preflight = {
        "config_guard": {
            "sections": list(FORMAL_SECTIONS),
            "passed": True,
            "tbd_elsewhere": find_tbd(cfgs),  # e.g. Exp 2's cut point: not a step09 section
        },
        "attempt_split": split.value,
        "n_states": len(states),
        "configured_n": want_n,
        "score_tables": tables,
        "labels": {"split": split.value, "purpose": L.LabelPurpose.GATE.value, "reads": 1},
        "d_gate_reserve": {
            "locked": reserve_locked,
            "labels_read": split is Split.D_GATE_RESERVE,
            "scored": split is Split.D_GATE_RESERVE,
        },
        "tau": {"v1": taus["v1"].tau, "v2": taus["v2"].tau, "file": str(paths.FROZEN_TAU)},
        "channel": {
            "file": str(CANDIDATE),
            "seed": channel.seed,
            "levels": {lam.value: p.to_dict() for lam, p in channel.levels.items()},
        },
        "delta_r": delta_r,
    }
    print(
        f"[step09] preflight: attempt on {split.value}, {len(states)} states (configured {want_n})"
    )
    print(
        f"[step09] preflight: config guard passed on {list(FORMAL_SECTIONS)}; TBD outside them: "
        f"{preflight['config_guard']['tbd_elsewhere']}"
    )
    for v, t in tables.items():
        print(
            f"[step09] preflight: {v} scores {t['table']} (sha256 {t['sha256'][:12]}), "
            f"{t['rows_for_split']} rows for {split.value}, {t['model_version']}"
        )
    print(
        f"[step09] preflight: y* of {split.value} read once, purpose {L.LabelPurpose.GATE.value}; "
        f"D_gate_reserve locked={reserve_locked}, its labels and scores untouched"
    )
    print(
        f"[step09] preflight: tau {preflight['tau']['v1']} / {preflight['tau']['v2']}; channel "
        f"seed {channel.seed}, levels {preflight['channel']['levels']}; delta_R {delta_r}"
    )

    labels = io.load_labels_for(states, split, L.LabelPurpose.GATE)
    s1, s2 = io.load_scores_for(states, v1), io.load_scores_for(states, v2)
    ev = evaluate_levels(
        channel, states, s1, s2, labels, taus["v1"].tau, taus["v2"].tau,
        high_margin_quantile=float(get(ch_cfg, "diagnostics.high_margin_quantile")),
        n_bins=int(get(ch_cfg, "diagnostics.n_bins", 10)),
        populations=populations,
    )  # fmt: skip

    # 1) diagnostics on disk FIRST (paper App. A)
    out_dir = io.diagnostics_dir(ch_cfg)
    out_dir.mkdir(parents=True, exist_ok=True)
    diag_path = out_dir / "gate_diagnostics.json"
    diag = {
        "split": split.value,
        "n_states": ev.n_states,
        "diagnostics_population": list(populations),
        "diagnostics_roles": diagnostics_roles(populations),
        "cue_diagnostics": {
            pop: {k: asdict(v) for k, v in by_lam.items()} for pop, by_lam in ev.diagnostics.items()
        },
        # r_ref (paper Sec. 3): the offline reference reliability - R of each verifier on this
        # labelled split with NO perturbation, identical for every arm, in the gate's own units
        # (gate metric (d): the share of E-UNSAFE states detected). It is recorded here because
        # this is the step that may read these labels; it is not an input to the ladder verdict
        # or to any cue diagnostic, and step11 only copies it into the Exp 1 report next to the
        # H2 strata (paper Sec. 4.6: reported alongside, never a stratifier).
        "r_ref": {
            "split": split.value,
            "lambda": Lambda.NONE.value,
            "population": G.GATE_POPULATION,
            "by_verifier": ev.r_none,
            "n": ev.n_unsafe,
            "accuracy_whole_split": {
                v: ev.descriptive[v][Lambda.NONE.value] for v in G.VERIFIER_KEYS
            },
        },
        "git_head": git_head(),
        "run_dir": str(run_dir),
        "recorded_at": datetime.now(tz=UTC).isoformat(timespec="seconds"),
    }
    diag_path.write_text(
        json.dumps(diag, ensure_ascii=False, indent=1), encoding="utf-8", newline="\n"
    )

    # 2) hard check + attempt record
    # the interval is formed first and only carried; the verdict below reads the point
    # estimates alone (paper Sec. 4.3). No multiplicity correction (paper App. B). The CI resamples
    # the E-UNSAFE states only: they are the population R is defined on (gate metric (d)).
    ci = G.ladder_ci(ev.vectors, n_boot=ci_n_boot(ch_cfg), seed=ci_seed(cfgs))
    ladder = G.ladder_check(
        ev.r,
        delta_r,
        ev.p_at_least_one_missed,
        ci=ci,
        n_states=ev.n_unsafe,
        descriptive=ev.descriptive,
    )
    entry = G.record_attempt(
        paths.FROZEN_GATE_ATTEMPTS,
        ladder,
        split,
        extra={"diagnostics_file": str(diag_path), "git_head": git_head(), "run_dir": str(run_dir)},
    )
    print(f"[step09] attempt {entry['attempt']} on {split.value}: passed={ladder.passed}")
    print(
        f"[step09] R on the {ev.n_unsafe} E-UNSAFE states (gate metric (d)): "
        f"{json.dumps(ladder.r)} gaps: {json.dumps(ladder.gaps)} delta_R={delta_r}"
    )
    print(
        f"[step09] 95% CI (B={ci['n_boot']}, seed={ci['seed']}, unit=E-UNSAFE state, reporting "
        f"only) R: {json.dumps(ci['r'])} gaps: {json.dumps(ci['gaps'])}"
    )
    print(
        f"[step09] descriptive, all {ev.n_states} states (not part of the verdict): "
        f"{json.dumps(ev.descriptive)}"
    )
    print(
        f"[step09] r_ref (offline reference, lambda=none, E-UNSAFE states of {split.value}): "
        f"{json.dumps(diag['r_ref']['by_verifier'])}"
    )
    paired = G.paired_correctness_correlation(ev.vectors)
    phis = {v: {k: x["phi"] for k, x in d.items()} for v, d in paired.items()}
    print(
        "[step09] paired correctness correlation across levels (phi, E-UNSAFE states, "
        f"descriptive): {json.dumps(phis)}"
    )
    for c in ladder.failed_conditions:
        print(f"[step09]   FAILED: {c}")
    if ladder.passed:
        shutil.copyfile(CANDIDATE, paths.FROZEN_CHANNEL)
        print(f"[step09] channel frozen -> {paths.FROZEN_CHANNEL}.")
        print("[step09] Next: the authors commit and tag 'channel-frozen'.")
    elif entry["attempt"] < G.GATE_MAX_ATTEMPTS:
        print(
            "[step09] ladder not established. Redesign the channel (step08); the second "
            "attempt must use D_gate_reserve, whose y* and scores are now unlocked."
        )
    else:
        print(
            "[step09] ladder not established on the second attempt: report it as is "
            "(II.6.1). No further attempts."
        )

    # 3) the attempt's report (2026-09-27): everything above in one file, pass or fail
    report = {
        "split": split.value,
        "attempt": entry["attempt"],
        "passed": ladder.passed,
        "failed_conditions": ladder.failed_conditions,
        "delta_r": delta_r,
        "population": G.GATE_POPULATION,
        "n_states": ev.n_states,
        "n_e_unsafe": ev.n_unsafe,
        "r": ladder.r,
        "gaps": ladder.gaps,
        "ci": ci,
        "p_at_least_one_missed": ladder.p_at_least_one_wrong,
        "descriptive_whole_split": ev.descriptive,
        "paired_correctness_correlation": paired,
        "diagnostics_population": list(populations),
        "diagnostics_roles": diagnostics_roles(populations),
        "cue_diagnostics": diag["cue_diagnostics"],
        "r_ref": diag["r_ref"],
        "preflight": preflight,
        "frozen_channel": str(paths.FROZEN_CHANNEL) if ladder.passed else None,
        "gate_attempts_file": str(paths.FROZEN_GATE_ATTEMPTS),
        "diagnostics_file": str(diag_path),
        "git_head": git_head(),
        "run_dir": str(run_dir),
    }
    report_path = out_dir / "step09_report.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8", newline="\n"
    )
    print(f"[step09] report -> {report_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
