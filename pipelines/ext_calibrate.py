"""Fit the pi-hat calibrators (post-pilot extension; local compute, no API call).

Per verifier, logit-scale Platt on the unlabeled pool, diagnosed on D_gate (2026-10-31;
settings in ``configs/ext.yaml: calibration``). Order of operations:

1. the pool is checked to share no x0 with D_fit, D_gate, D_gate_reserve, D_Exp1 (split files);
2. the clipped-score counts per verifier x split (pool, D_gate, D_Exp1; scores only) -- any cell
   above ``max_clip_share`` stops the step before anything is fitted, and the table is printed;
3. the fit on the pool, y* read with purpose CALIBRATION_FIT;
4. the prior correction (2026-10-31;
   ``calibration.prior_correction``): its two premises checked first
   (``rafr.calibration.fit.check_prior_premises``: pool + D_Exp1 = all states - D_fit - D_gate -
   D_gate_reserve, and the recorded split is the procedure's seeded draw, so D_Exp1 is a uniform
   draw among the remaining E-UNSAFE), then b += logit(p_R) - logit(p_pool) from the pool's
   counts and |D_Exp1| -- no D_gate or D_Exp1 label is read;
5. the frozen file is written once (``calibration.frozen_file``; an existing file is never
   overwritten) with both (a, b), Platt's b_fit, p_pool / p_R / delta, eps, the target rule, the
   id list of the fitting data and its sha256, and the superseded file it replaces;
6. the diagnostics on D_gate, y* read with purpose CALIBRATION_DIAG: equal-frequency ECE,
   Brier, the reliability table, n and the number of task failures, bootstrap intervals --
   g_i against task failure and pi-hat_i against "verdict wrong". There is no quality threshold.

Outputs: the frozen file (to be committed; the authors tag it) and
``runs/ext_calibration_<domain>_<timestamp>/diagnostics.json`` (+ ``clip_report.json``).
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
import json
import subprocess
from datetime import UTC, datetime
from typing import Any

from pipelines import _validity_io as io
from rafr.calibration import fit as F
from rafr.calibration import platt
from rafr.calibration.apply import Calibrator, file_sha256, write_frozen
from rafr.contracts.schemas import SafetyLabel, Split
from rafr.core import paths
from rafr.core.config import load_config, master_seed
from rafr.core.seeding import component_seed
from rafr.data import labels as L
from rafr.data import splits as S
from rafr.experiments.ext_config import checked, load_ext
from rafr.logging.writer import read_jsonl
from rafr.validity.tau import load_tau
from rafr.verifiers import scores_table_path


def _git_head() -> str | None:
    try:
        out = subprocess.run(  # noqa: S603 - fixed argv, no shell
            ["git", "rev-parse", "HEAD"],  # noqa: S607
            cwd=paths.ROOT,
            capture_output=True,
            text=True,
            check=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return out.stdout.strip()


def _scores(split: Split, v1: str, v2: str) -> tuple[list[str], dict[str, dict[str, float]]]:
    states = io.load_states(split)
    return [s.state_id for s in states], {
        "v1": io.load_scores_for(states, v1),
        "v2": io.load_scores_for(states, v2),
    }


def _prior_premises(cfgs: dict[str, Any], pool_y: dict[str, int]) -> dict[str, Any]:
    """The prior correction's premises from the state list (data/pools/states.jsonl, minus the
    x0 dropped before the split), the split files, the manifest's split plan and the pool's own
    y*; the D_Exp1 states are failures by design (E-UNSAFE)."""
    manifest = json.loads(paths.MANIFEST.read_text(encoding="utf-8"))["splits"]
    dropped = {d["x0_id"] for d in manifest.get("dropped_before_split") or []}
    all_x0 = [r["x0_id"] for r in read_jsonl(paths.DATA_POOLS / "states.jsonl")]
    all_x0 = [x for x in all_x0 if x not in dropped]
    if len(all_x0) != int(manifest["n_x0_with_y_star"]):
        raise SystemExit(
            f"{len(all_x0)} states, the manifest records {manifest['n_x0_with_y_star']}"
        )
    pool_states = io.load_states(Split.UNLABELED_POOL)
    x0_of = {s.state_id: s.x0_id for s in pool_states}
    splits = S.load_all_splits(paths.DATA_SPLITS)
    plan = manifest["plan"]
    master = master_seed(cfgs)
    if int(plan["master_seed"]) != master:
        raise SystemExit(f"split plan master seed {plan['master_seed']}, configs {master}")
    return F.check_prior_premises(
        all_x0=all_x0,
        splits=splits,
        plan=plan,
        expected_seed=component_seed(master, S.SEED_COMPONENT),
        pool_failure_x0=[x0_of[sid] for sid, y in pool_y.items() if y],
    )


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.parse_args(argv)
    cfgs = load_config()
    ext = load_ext()
    cal = ext["calibration"]
    if ext.get("domain") != "detox":
        raise SystemExit("this step wires detox only (math is the authors')")
    frozen = paths.resolve(cal["frozen_file"])
    if frozen.exists():
        raise SystemExit(f"{frozen} exists: the calibrator is fitted once and never refitted")
    eps = float(cal["epsilon"])
    v1, v2 = io.verifier_ids(cfgs["models"])

    # 1) no leakage: the pool is disjoint from every other split, by x0
    disjoint = F.assert_pool_disjoint(paths.DATA_SPLITS)
    print(f"[ext_calibrate] pool disjoint from the other splits by x0: {disjoint['sizes']}")

    # 2) clipped counts per verifier x split (scores only), before anything is fitted
    pool_ids, pool_scores = _scores(Split.UNLABELED_POOL, v1, v2)
    gate_ids, gate_scores = _scores(Split.D_GATE, v1, v2)
    exp1_ids, exp1_scores = _scores(Split.D_EXP1, v1, v2)
    ts = datetime.now(tz=UTC).strftime("%Y%m%dT%H%M%SZ")
    out = paths.resolve(cal.get("diagnostics_dir", "runs")) / f"ext_calibration_detox_{ts}"
    out.mkdir(parents=True, exist_ok=False)
    by_split = {
        sp.value: {k: [sc[k][i] for i in ids] for k in ("v1", "v2")}
        for sp, ids, sc in (
            (Split.UNLABELED_POOL, pool_ids, pool_scores),
            (Split.D_GATE, gate_ids, gate_scores),
            (Split.D_EXP1, exp1_ids, exp1_scores),
        )
    }
    try:
        clip = F.clip_report(by_split, eps, float(cal["max_clip_share"]))
    except F.ClipShareError as exc:
        (out / "clip_report.json").write_text(
            json.dumps(exc.report, indent=1), encoding="utf-8", newline="\n"
        )
        print(json.dumps(exc.report, indent=1))
        raise SystemExit(f"[ext_calibrate] STOP: {exc}") from exc
    (out / "clip_report.json").write_text(
        json.dumps(clip, indent=1), encoding="utf-8", newline="\n"
    )
    for cell, c in clip["cells"].items():
        print(f"[ext_calibrate] clipped {cell}: {c['clipped']}/{c['n']} ({c['share']:.4%})")

    # 3) the pool's y* (CALIBRATION_FIT)
    pool_y = F.task_failures(
        Split.UNLABELED_POOL, L.LabelPurpose.CALIBRATION_FIT, SafetyLabel.UNSAFE
    )
    if set(pool_y) != set(pool_ids):
        raise SystemExit("the pool's y* and its states do not cover the same states")

    # 4) the prior correction: premises first, then p_pool, p_R, delta (no D_gate / D_Exp1 y*)
    pc = cal["prior_correction"]
    if pc.get("added_failures_split") != Split.D_EXP1.value:
        raise SystemExit("the prior correction is wired for D_Exp1's failures only")
    try:
        premises = _prior_premises(cfgs, pool_y)
    except F.PriorPremiseError as exc:
        raise SystemExit(
            f"[ext_calibrate] STOP: a premise of the prior correction fails: {exc}"
        ) from exc
    n_exp1 = premises["sizes"][Split.D_EXP1.value]
    shift = platt.prior_shift(len(pool_ids), sum(pool_y.values()), n_exp1)
    print(
        f"[ext_calibrate] prior correction: premises hold; p_pool {shift['p_pool']!r}, "
        f"p_R {shift['p_R']!r}, delta {shift['delta']!r}"
    )
    sup = cal["superseded"]
    superseded = checked(sup["file"], sup["sha256"], what="superseded calibrator")

    # 5) the fit on the pool, frozen once
    record = F.fit_record(
        domain="detox",
        ids=pool_ids,
        failures=pool_y,
        scores=pool_scores,
        verifier_ids={"v1": v1, "v2": v2},
        eps=eps,
        provenance={
            "states_file": "data/pools/states_unlabeled_pool.jsonl",
            "states_sha256": file_sha256(paths.DATA_POOLS / "states_unlabeled_pool.jsonl"),
            "scores_sha256": {
                cid: file_sha256(scores_table_path(cid, paths.DATA_SCORES)) for cid in (v1, v2)
            },
            "labels": "E's y* (step05) through the label gate, purpose calibration_fit",
            "pool_disjoint": disjoint,
            "supersedes": {
                "file": paths.shown(superseded),
                "sha256": sup["sha256"],
                "diagnostics": sup.get("diagnostics"),
                "reason": sup.get("reason"),
            },
            "code_git_head": _git_head(),
            "fitted_at": datetime.now(tz=UTC).isoformat(timespec="seconds"),
        },
        prior_correction={**shift, "added_split": Split.D_EXP1.value, "premises": premises},
    )
    sha = write_frozen(frozen, record)
    print(f"[ext_calibrate] frozen calibrator -> {frozen} (sha256 {sha})")
    for k in ("v1", "v2"):
        f = record["verifiers"][k]
        print(
            f"[ext_calibrate] {k} {f['verifier_id']}: a {f['a']!r}, b_fit {f['b_fit']!r}, "
            f"b {f['b']!r}, {f['iterations']} Newton steps, max |grad| "
            f"{f['max_abs_gradient']:.3g}, n {f['n']} ({f['n_positive']} task failures), "
            f"clipped {f['n_clipped']}"
        )

    # 6) the diagnostics on D_gate (y*: CALIBRATION_DIAG), from the frozen file as written
    calibrator = Calibrator.load(frozen, expected_sha256=sha)
    gate_y = F.task_failures(Split.D_GATE, L.LabelPurpose.CALIBRATION_DIAG, SafetyLabel.UNSAFE)
    taus = load_tau(paths.FROZEN_TAU)
    boot = cal["bootstrap"]
    diag = F.diagnose(
        calibrator,
        ids=gate_ids,
        failures=gate_y,
        scores=gate_scores,
        taus={"v1": taus["v1"].tau, "v2": taus["v2"].tau},
        n_bins=int(cal["ece_bins"]),
        n_boot=int(boot["n_boot"]),
        seed=component_seed(master_seed(cfgs), str(boot["seed_component"])),
        alpha=float(boot["alpha"]),
    )
    diag_out: dict[str, Any] = {
        "frozen_file": paths.shown(frozen),
        "frozen_sha256": sha,
        "split": Split.D_GATE.value,
        "score": "raw (lambda = none)",
        "tau_sha256": file_sha256(paths.FROZEN_TAU),
        "prior_correction": record["prior_correction"],
        "params": {
            k: {x: record["verifiers"][k][x] for x in ("verifier_id", "a", "b_fit", "b")}
            for k in ("v1", "v2")
        },
        **diag,
    }
    (out / "diagnostics.json").write_text(
        json.dumps(diag_out, indent=1), encoding="utf-8", newline="\n"
    )
    for k, d in diag["verifiers"].items():
        for name in ("g_vs_failure", "pihat_vs_verdict_wrong"):
            m = d[name]
            print(
                f"[ext_calibrate] D_gate {k} {name}: n {m['n']} ({m['n_positive']} positive), "
                f"ECE {m['ece']:.4f} {[round(x, 4) for x in m['ece_ci']]}, "
                f"Brier {m['brier']:.4f} {[round(x, 4) for x in m['brier_ci']]}"
            )
    print(f"[ext_calibrate] diagnostics -> {out / 'diagnostics.json'}")
    print(
        "[ext_calibrate] next: commit the frozen file, put its sha256 into configs/ext.yaml "
        f"(calibration.frozen_sha256 = {sha})"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
