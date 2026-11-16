"""Step 07 — fit tau_1, tau_2 on D_fit (paper Sec. 5; step 7).

Criterion (locked): maximize F1 on D_fit; on a plateau take the midpoint. F2,
PR-AUC, precision, recall, balanced accuracy are reported alongside and never take part in the
selection.

Inputs: states_d_fit.jsonl, D_fit's y* (E's judgments, purpose TAU_FIT), the two formal
verifiers' score tables. Output: ``frozen/tau.json``. The authors then freeze it (commit and git
tag) before the channel is checked.

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
from datetime import UTC, datetime

from pipelines import _validity_io as io
from rafr.contracts.schemas import Split
from rafr.core import paths
from rafr.core.config import assert_frozen, load_config
from rafr.data.labels import LabelPurpose
from rafr.logging.snapshot import git_head, write_snapshot
from rafr.validity import tau as T


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--force", action="store_true", help="overwrite an existing frozen/tau.json")
    args = ap.parse_args(argv)

    cfgs = load_config()
    assert_frozen(cfgs, only_sections=["models.verifiers"])
    if paths.FROZEN_TAU.exists() and not args.force:
        raise SystemExit(
            f"{paths.FROZEN_TAU} exists; tau is frozen. Use --force only before tagging."
        )
    run_dir = paths.run_dir(f"step07_{datetime.now(tz=UTC).strftime('%Y%m%dT%H%M%SZ')}")
    write_snapshot(run_dir, step="step07", configs=cfgs)

    v1, v2 = io.verifier_ids(cfgs["models"])
    states = io.load_states(Split.D_FIT)
    labels = io.load_labels_for(states, Split.D_FIT, LabelPurpose.TAU_FIT)
    fits = {}
    for key, vid in (("v1", v1), ("v2", v2)):
        scores = io.load_scores_for(states, vid)
        s, y = io.aligned(states, scores, labels)
        fits[key] = T.fit_tau(s, y, vid)
        f = fits[key]
        print(
            f"[step07] {key}={vid}: tau={f.tau:.4f} F1={f.f1_max:.4f} "
            f"plateau=[{f.plateau_lo:.4f}, {f.plateau_hi:.4f}] n_plateaus={f.n_plateaus}"
        )
        if f.n_plateaus > 1:
            print(
                f"[step07] WARNING {key}: {f.n_plateaus} disjoint F1-maximal plateaus; "
                "widest one used. the authors to confirm."
            )
    T.save_tau(
        fits,
        paths.FROZEN_TAU,
        extra={
            "split": Split.D_FIT.value,
            "n_states": len(states),
            "git_head": git_head(),
            "run_dir": str(run_dir),
        },
    )
    print(json.dumps({k: v.aux for k, v in fits.items()}, indent=1))
    print(f"[step07] wrote {paths.FROZEN_TAU}. Next: the authors commit and tag 'tau-frozen'.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
