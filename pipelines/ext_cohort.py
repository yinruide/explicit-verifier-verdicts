"""Draw the E-correct cohort (post-pilot extension, stop point C; no API call).

The D_gate states E judged correct (detox: SAFE), read through the label gate with purpose
ECORRECT_COHORT (2026-10-31). Their number is reported; ``cohorts.e_correct.n`` (300) are
drawn without replacement with the seed component ``ext_ecorrect_cohort`` -- all of them if there
are no more than that. The cohort file (``cohorts.e_correct.file``) is written once; its sha256
then goes into ``configs/ext.yaml`` (``cohorts.e_correct.file_sha256``).
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse

from rafr.contracts.schemas import SafetyLabel, Split
from rafr.core import paths
from rafr.core.config import get, load_config, master_seed
from rafr.core.seeding import component_seed
from rafr.data import labels as L
from rafr.experiments.ext_cohort import draw_cohort, write_cohort
from rafr.experiments.ext_config import load_ext


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.parse_args(argv)
    cfgs = load_config()
    ext = load_ext()
    spec = get(ext, "cohorts.e_correct")
    if ext.get("domain") != "detox" or spec["source_split"] != Split.D_GATE.value:
        raise SystemExit("this step draws detox's E-correct cohort from D_gate only")
    path = paths.resolve(spec["file"])
    if path.exists():
        raise SystemExit(f"{path} exists: the cohort is drawn once")
    correct = SafetyLabel(spec["correct_label"])
    ys = L.read_labels(Split.D_GATE, L.LabelPurpose.ECORRECT_COHORT)
    candidates = [y.state_id for y in ys if y.label is correct]
    n = int(spec["n"])
    component = str(spec["seed_component"])
    seed = component_seed(master_seed(cfgs), component)
    ids = draw_cohort(candidates, n, seed)
    sha = write_cohort(
        path,
        ids,
        {
            "domain": "detox",
            "source_split": Split.D_GATE.value,
            "correct_label": correct.value,
            "n_gate": len(ys),
            "n_candidates": len(candidates),
            "n_requested": n,
            "all_candidates": len(candidates) <= n,
            "seed_component": component,
            "seed": seed,
        },
    )
    how = "all of them" if len(candidates) <= n else "drawn without replacement"
    print(
        f"[ext_cohort] D_gate: {len(ys)} states, {len(candidates)} judged {correct.value} by E; "
        f"cohort {len(ids)} ({how})"
    )
    print(f"[ext_cohort] -> {path} (sha256 {sha}); put it into cohorts.e_correct.file_sha256")
    return 0


if __name__ == "__main__":
    sys.exit(main())
