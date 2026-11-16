"""Step 10 — every preregistered analysis on FAKE logs (0.5 step 10, tag analysis-preregistered).

    python pipelines/step10_fake_data_test.py [--seed S] [--n-states 300] [--n-boot 10000]

1. ``analysis.fake_data.write_fake_runs`` writes, under ``runs/step10_<ts>/``, a fake Exp 1 run
   (300 states x 3 arms x 3 lambda) and a fake Exp 2 run (300 trajectories, 150 per arm, K = 4)
   in the layout step11 / step12 write and with the real row schemas; parse_fail (all four
   categories) and refusal_like cases included; arm labels shuffled (Exp 1: per (state, lambda);
   Exp 2: the balanced permutation), every seed in ``fake_manifest.json``. Every directory
   carries the FAKE marker and every row a ``FAKE-`` run_id. Nothing is read from ``data/`` or
   ``frozen/``: no real label, score, threshold or controller output.
2. ``analysis.prereg_tables`` in fake mode -- it refuses any input without the FAKE marker --
   computes every table of ``prereg/analysis_plan.md`` section by section and writes them to
   ``runs/step10_<ts>/tables/`` as ``FAKE_<table>.json`` plus ``FAKE_tables.md``. The formal
   readers (step11, step12, ``analysis.prereg_tables`` without fake) refuse these files.

Every arm effect is zero by construction; any number here is a property of the fake generator,
not a result. When the authors are satisfied they tag ``analysis-preregistered``.
"""

from __future__ import annotations

# A direct ``python pipelines/<step>.py`` puts pipelines/ on sys.path, not the repo root,
# so the repo-root packages (analysis, annotation, pipelines) would not import; rafr is
# installed and unaffected. Covered by tests/test_pipeline_entrypoints.py.
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
from datetime import UTC, datetime

from analysis import fake_data, prereg_tables

from rafr.contracts.schemas import BOOTSTRAP_B
from rafr.core import paths
from rafr.core.config import get, load_config
from rafr.logging.snapshot import write_snapshot

DEFAULT_SEED = 20260927
"""Base seed of the fake data. Not the formal master seed: fake and real draws never meet."""


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED, help="base seed of the fake data")
    ap.add_argument("--n-states", type=int, default=300, help="fake D_Exp1 size (II.2.4: 300)")
    ap.add_argument("--n-boot", type=int, default=BOOTSTRAP_B, help="bootstrap B (III.1.2: 10000)")
    args = ap.parse_args(argv)
    cfgs = load_config()
    rd = paths.run_dir(f"step10_{datetime.now(tz=UTC).strftime('%Y%m%dT%H%M%SZ')}")
    runs = fake_data.write_fake_runs(rd, seed=args.seed, n_states=args.n_states)
    write_snapshot(
        rd,
        step="step10",
        configs=cfgs,
        extra={"FAKE": True, "seed": args.seed, "n_states": args.n_states, "n_boot": args.n_boot},
    )
    inp = prereg_tables.load_inputs(
        runs.exp1_dir, runs.exp2_dir, gate_report=runs.gate_report, fake=True
    )
    # the preregistered cutpoint as configured; TBD today, so the semantic table has counts only
    cut = get(cfgs["exp2"], "analysis.semantic_preservation.cutpoint")
    results = prereg_tables.compute(
        inp,
        seed_exp1=runs.seeds.of("bootstrap_exp1"),
        seed_exp2=runs.seeds.of("bootstrap_exp2"),
        n_boot=args.n_boot,
        semantic_cutpoint=None if cut in (None, "TBD") else float(cut),
        n_exp1=args.n_states,
        n_per_arm=args.n_states // 2,
    )
    tables = prereg_tables.build_tables(results, inp)
    written = prereg_tables.write_tables(tables, results, rd / "tables", fake=True)
    did = results["exp1"]["primary_did"]["main"]
    tfr = results["exp2"]["primary_contrast"]
    failed = [c["check"] for c in results["exp1_checks"] + results["exp2_checks"] if not c["ok"]]
    print(f"[step10] FAKE runs -> {runs.exp1_dir.name}, {runs.exp2_dir.name} under {rd}")
    print(f"[step10] {len(tables)} FAKE tables, {len(written)} files -> {rd / 'tables'}")
    print(
        f"[step10] FAKE Exp 1 D-bar {did['estimate']:+.4f} [{did['ci95'][0]:+.4f}, "
        f"{did['ci95'][1]:+.4f}] (n={did['n_states']}); FAKE Exp 2 TFR(A1)-TFR(B) "
        f"{tfr['estimate']:+.4f} [{tfr['ci95'][0]:+.4f}, {tfr['ci95'][1]:+.4f}]; arm effects are "
        "zero by construction"
    )
    print(f"[step10] construction checks failed: {failed or 'none'}")
    print(f"[step10] review {rd / 'tables' / 'FAKE_tables.md'}; FAKE numbers are not results.")
    print("[step10] Next: the authors review and tags 'analysis-preregistered'.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
