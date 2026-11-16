"""The extension contrasts per domain (2026-11-01): a wrapper of ``ext_contrasts``.

``analysis/ext_contrasts.py`` is covered by the tag ``detox-ext-frozen`` and is not changed. Its
entry reads detox's ``configs/ext.yaml`` and detox's master seed; this wrapper imports its
functions as they are and passes the domain's extension config and master seed in their place.
Every rule of the analysis is ``ext_contrasts``'s (its module docstring): extension run
directories only, the preregistered files read by the sha256 the config records, III.1.2's
intervals with the seed component ``analysis.bootstrap.seed_component``.

Domains (:data:`DOMAINS`, shared with ``analysis/ext_decision_quality.py``):

* ``detox``: ``configs/ext.yaml``; master seed ``configs/data.yaml: seeds.master``. On the detox
  runs the wrapper's ``report.json`` and ``tables.md`` are byte for byte ``ext_contrasts.py``'s
  (``runs/ext_analysis_20261101T004507Z``);
* ``math``: ``configs/math/ext.yaml``; master seed ``configs/math/formal.json: seed``
  (``MathRunConfig.seed``); no panel (M.3), so a panel label in a math run is an error. Math's
  original Exp 2 finals name the start ``question_id`` and record ``termination_reason``; they are
  given ``ext_contrasts``'s ``x0_id`` and ``accepted`` (:func:`original_finals`, 2026-11-05);

Outputs: ``runs/ext_analysis_<domain>_<ts>/`` with ``report.json`` and ``tables.md`` written as
``ext_contrasts.main`` writes them, and ``inputs.json`` (the config, the seed's source, the run
directories). ``--fake`` runs the whole wrapper on ``ext_contrasts``'s fake data with the
domain's panel vocabulary (FAKE marker, FAKE_ file names, seed 0) and never reads a real file.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
import json
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

from analysis import ext_contrasts as X
from analysis._common import FAKE_FILE_PREFIX, FAKE_MARKER, refuse_fake, require_fake
from rafr.calibration.apply import file_sha256
from rafr.contracts.schemas import BOOTSTRAP_ALPHA, BOOTSTRAP_B
from rafr.core import paths
from rafr.core.config import find_tbd, get, load_config, load_yaml, master_seed
from rafr.core.seeding import component_seed
from rafr.experiments.ext_config import checked
from rafr.logging.writer import read_jsonl

E_INCORRECT, E_CORRECT = X.E_INCORRECT, X.E_CORRECT
OUT_PREFIX = "ext_analysis"
MATH_RUN_CONFIG = "configs/math/formal.json"
FAKE_SEED = 0


class ExtDomainError(RuntimeError):
    pass


def _detox_master_seed() -> int:
    return master_seed(load_config())


def _math_master_seed() -> int:
    from rafr.domains.math.config import MathRunConfig

    seed = MathRunConfig.load(paths.resolve(MATH_RUN_CONFIG)).seed
    if seed is None:
        raise ExtDomainError(f"{MATH_RUN_CONFIG} has no seed")
    return seed


@dataclass(frozen=True)
class Domain:
    name: str
    ext_config: str  # repo-relative
    seed_source: str  # where the domain's master seed is read, for the record
    master_seed: Callable[[], int] | None  # None = not wired yet
    # a panel label -> the SAFE / UNSAFE that ext_contrasts.panel_block counts (anything but
    # SAFE is a task failure); None = the domain has no panel
    panel_labels: Mapping[str, str] | None


DOMAINS: dict[str, Domain] = {
    "detox": Domain(
        "detox",
        "configs/ext.yaml",
        "configs/data.yaml: seeds.master",
        _detox_master_seed,
        {"SAFE": "SAFE", "UNSAFE": "UNSAFE"},
    ),
    "math": Domain(
        "math",
        "configs/math/ext.yaml",
        f"{MATH_RUN_CONFIG}: seed",
        _math_master_seed,
        None,
    ),
}


def domain(name: str) -> Domain:
    if name not in DOMAINS:
        raise ExtDomainError(f"unknown domain {name!r}; known: {sorted(DOMAINS)}")
    return DOMAINS[name]


def domain_master_seed(name: str) -> int:
    """The master seed the domain's extension components derive from."""
    d = domain(name)
    if d.master_seed is None:
        raise ExtDomainError(f"{name}'s master seed is {d.seed_source}")
    return d.master_seed()


def load_domain_ext(name: str, path: str | Path | None = None) -> dict[str, Any]:
    """The domain's extension config (``path`` overrides the registry's), whose ``domain`` must
    be ``name``."""
    p = paths.resolve(path or domain(name).ext_config)
    if not p.is_file():
        raise ExtDomainError(f"no extension config for domain {name!r}: {p} does not exist")
    ext = load_yaml(p)
    if ext.get("domain") != name:
        raise ExtDomainError(f"{p} is for {ext.get('domain')!r}, not {name!r}")
    return ext


def map_panel(rows: Iterable[Mapping[str, Any]], name: str) -> list[dict[str, Any]]:
    """``exp2_finals`` rows with every panel label in ``ext_contrasts``'s SAFE / UNSAFE (None,
    no verdict, stays None). A domain without a panel must carry no panel label."""
    labels = domain(name).panel_labels
    out = []
    for r in rows:
        panel = r.get("_panel") or {}
        if labels is None:
            if panel:
                raise ExtDomainError(
                    f"{name} has no panel, but trajectory {(r['x0_id'], r['arm'])} carries "
                    f"panel labels of {sorted(panel)}"
                )
            out.append(dict(r))
            continue
        mapped: dict[str, str | None] = {}
        for member, lab in panel.items():
            if lab is not None and lab not in labels:
                raise ExtDomainError(
                    f"panel {member}: label {lab!r} of {(r['x0_id'], r['arm'])} is not one of "
                    f"{sorted(labels)} ({name})"
                )
            mapped[member] = None if lab is None else labels[lab]
        out.append({**dict(r), "_panel": mapped})
    return out


# Math's original Exp 2 finals (runs/math_formal_n9000/exp2/trajectories.jsonl, written by
# pipelines/math_pipeline.py) name the start ``question_id`` and end in ``termination_reason``:
# "accepted" exactly when the trajectory was accepted (``tr.accepted``), else "decision_limit" or
# "v_input_overlength". ext_contrasts reads ``x0_id`` and ``accepted`` (2026-11-05).
MATH_TERMINATION_REASONS = ("accepted", "decision_limit", "v_input_overlength")


def original_finals(rows: Iterable[Mapping[str, Any]], name: str) -> list[dict[str, Any]]:
    """The original Exp 2 finals in ``ext_contrasts``'s fields; only math's need mapping (the
    math extension runs already write ``x0_id`` and ``accepted``)."""
    rows = [dict(r) for r in rows]
    if name != "math":
        return rows
    out = []
    for r in rows:
        if "x0_id" in r or "accepted" in r:
            raise ExtDomainError(
                f"a math original Exp 2 final already has x0_id / accepted: {r.get('question_id')}"
            )
        reason = r.get("termination_reason")
        if not r.get("question_id") or reason not in MATH_TERMINATION_REASONS:
            raise ExtDomainError(
                f"math original Exp 2 final {r.get('question_id')!r}: termination_reason "
                f"{reason!r} is not one of {MATH_TERMINATION_REASONS}"
            )
        out.append({**r, "x0_id": r["question_id"], "accepted": reason == "accepted"})
    return out


def gather(
    name: str,
    ext: Mapping[str, Any],
    *,
    exp1_e_incorrect: Sequence[Path] = (),
    exp1_e_correct: Sequence[Path] = (),
    exp2_e_incorrect: Sequence[Path] = (),
    exp2_e_correct: Sequence[Path] = (),
    base: Path | None = None,
    fake: bool = False,
) -> tuple[X.Plan, dict[str, Any], dict[str, Any], dict[str, Any]]:
    """(plan, exp1, exp2, cited) as ``ext_contrasts.main`` builds them, from the domain's config;
    Exp 2's panel labels mapped per :func:`map_panel`."""
    if ext.get("domain") != name:
        raise ExtDomainError(f"the extension config is for {ext.get('domain')!r}, not {name!r}")
    acfg = get(ext, "analysis")
    tbd = find_tbd(acfg, "ext.analysis")
    if tbd:
        raise ExtDomainError(f"the contrast list is not confirmed yet: {tbd}")
    plan = X.plan_from_config(acfg)
    pre = {e: str(get(ext, f"{e}.run_id_prefix")) for e in X.EXPERIMENTS}
    s1, s2 = get(ext, "exp1.e_incorrect"), get(ext, "exp2.e_incorrect")
    run1, run2 = paths.resolve(s1["original_run"]), paths.resolve(s2["original_run"])
    f1 = checked(run1 / "decisions.jsonl", s1["original_sha256"]["decisions.jsonl"],
                 what="original Exp 1 decisions")  # fmt: skip
    f2 = checked(run2 / "trajectories.jsonl", s2["original_sha256"]["trajectories.jsonl"],
                 what="original Exp 2 finals")  # fmt: skip
    guard = require_fake if fake else refuse_fake
    for rd in [*exp1_e_incorrect, *exp1_e_correct, *exp2_e_incorrect, *exp2_e_correct]:
        guard(rd, what="ext_analysis")

    def original(f: Path) -> list[dict[str, Any]]:
        rows = list(read_jsonl(f))
        if fake:
            require_fake(f, rows, what="ext_analysis --fake")
        return rows

    exp1: dict[str, Any] = {}
    if exp1_e_incorrect:
        exp1[E_INCORRECT] = X.exp1_rows(E_INCORRECT, exp1_e_incorrect, original(f1),
                                        X._original_arms(plan), prefix=pre["exp1"],
                                        base=base)  # fmt: skip
    if exp1_e_correct:
        exp1[E_CORRECT] = X.exp1_rows(E_CORRECT, exp1_e_correct, prefix=pre["exp1"], base=base)
    exp2: dict[str, Any] = {}
    if exp2_e_incorrect:
        original_panel = {
            fname.removeprefix("panel_").removesuffix(".jsonl"): {
                r["text_id"]: r["label"]
                for r in read_jsonl(checked(run2 / fname, sha, what=f"original Exp 2 {fname}"))
            }
            for fname, sha in s2["original_sha256"].items()
            if fname.startswith("panel_")
        }
        finals = X.exp2_finals(E_INCORRECT, exp2_e_incorrect, original_finals(original(f2), name),
                               original_panel=original_panel, prefix=pre["exp2"],
                               base=base)  # fmt: skip
        exp2[E_INCORRECT] = map_panel(finals, name)
    if exp2_e_correct:
        finals = X.exp2_finals(E_CORRECT, exp2_e_correct, prefix=pre["exp2"], base=base)
        exp2[E_CORRECT] = map_panel(finals, name)
    cited: dict[str, dict[str, Any]] = {}
    for exp, by_cohort in (acfg.get("cited_not_recomputed") or {}).items():
        for cohort, spec in by_cohort.items():
            cited.setdefault(exp, {})[cohort] = X.cite_prereg(
                paths.resolve(spec["tables_dir"]),
                spec["sha256"],
                [tuple(c) for c in spec["contrasts"]],
                fake=fake,
            )
    return plan, exp1, exp2, cited


def analyze(
    name: str,
    ext: Mapping[str, Any],
    *,
    seed: int,
    n_boot: int | None = None,
    **runs: Any,
) -> dict[str, Any]:
    """The report ``ext_contrasts.analyze`` makes of the domain's runs (``runs``: the run lists
    and ``base`` / ``fake`` of :func:`gather`); ``n_boot`` defaults to the config's."""
    plan, exp1, exp2, cited = gather(name, ext, **runs)
    if n_boot is None:
        n_boot = int(get(ext, "analysis.bootstrap.n_boot"))
    return X.analyze(exp1=exp1, exp2=exp2, plan=plan, cited=cited, n_boot=n_boot, seed=seed)


def write_outputs(report: Mapping[str, Any], out: Path, *, prefix: str = "") -> list[Path]:
    """``report.json`` and ``tables.md`` exactly as ``ext_contrasts.main`` writes them."""
    out.mkdir(parents=True, exist_ok=True)
    rj, tm = out / f"{prefix}report.json", out / f"{prefix}tables.md"
    rj.write_text(json.dumps(X._clean(report), indent=1), encoding="utf-8", newline="\n")
    tm.write_text(X.render(report), encoding="utf-8", newline="\n")
    return [rj, tm]


# ---- fake data -------------------------------------------------------------------------------


def fake_inputs(root: Path, name: str) -> tuple[dict[str, Any], dict[str, Any]]:
    """``ext_contrasts.write_fake``'s data with the panel in the domain's vocabulary (its files
    deleted for a domain without a panel), and an extension config naming it."""
    spec = X.write_fake(root)
    labels = domain(name).panel_labels
    back = {v: k for k, v in (labels or {}).items()}
    for f in sorted(Path(root).rglob("panel_*.jsonl")):
        if labels is None:
            f.unlink()
            continue
        rows = [{**r, "label": None if r["label"] is None else back[r["label"]]}
                for r in read_jsonl(f)]  # fmt: skip
        X._jsonl(f, rows)
    o1, o2 = Path(spec["exp1_original"]), Path(spec["exp2_original"])
    if name == "math":  # math's original finals in math's own layout (see original_finals)
        finals = o2 / "trajectories.jsonl"
        X._jsonl(finals, [
            {**{k: v for k, v in r.items() if k not in ("x0_id", "accepted")},
             "question_id": r["x0_id"],
             "termination_reason": "accepted" if r["accepted"] else "decision_limit"}
            for r in read_jsonl(finals)
        ])  # fmt: skip
    sha2 = {"trajectories.jsonl": file_sha256(o2 / "trajectories.jsonl")}
    sha2 |= {f.name: file_sha256(f) for f in sorted(o2.glob("panel_*.jsonl"))}
    extra = {
        e: {c: [list(p) for p in v] for c, v in d.items()} for e, d in X.EXTRA_CONTRASTS.items()
    }
    ext = {
        "domain": name,
        "exp1": {
            "run_id_prefix": X.RUN_PREFIXES["exp1"],
            "e_incorrect": {
                "original_run": str(o1),
                "original_sha256": {"decisions.jsonl": file_sha256(o1 / "decisions.jsonl")},
            },
        },
        "exp2": {
            "run_id_prefix": X.RUN_PREFIXES["exp2"],
            "e_incorrect": {"original_run": str(o2), "original_sha256": sha2},
        },
        "analysis": {
            "contrasts": [list(c) for c in X.COMMON_CONTRASTS],
            "extra_contrasts": extra,
            "cited_not_recomputed": {
                "exp1": {
                    E_INCORRECT: {
                        "contrasts": [list(c) for c in X.CITED_CONTRASTS["exp1"][E_INCORRECT]],
                        "tables_dir": str(spec["cited_tables"]),
                        "sha256": dict.fromkeys(X.FAKE_CITED),
                    }
                }
            },
            "bootstrap": {"n_boot": BOOTSTRAP_B, "alpha": BOOTSTRAP_ALPHA},
        },
    }
    return spec, ext


def fake_runs(spec: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "exp1_e_incorrect": spec["exp1"][E_INCORRECT],
        "exp1_e_correct": spec["exp1"][E_CORRECT],
        "exp2_e_incorrect": spec["exp2"][E_INCORRECT],
        "exp2_e_correct": spec["exp2"][E_CORRECT],
        "base": spec["base"],
        "fake": True,
    }


# ---- entry -----------------------------------------------------------------------------------


def _shown(p: Path) -> str:
    try:
        return paths.shown(p)
    except ValueError:
        return Path(p).as_posix()


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--domain", required=True, choices=sorted(DOMAINS))
    ap.add_argument("--fake", action="store_true", help="end-to-end run on shuffled fake data")
    ap.add_argument("--exp1-e-incorrect", type=Path, nargs="*", default=[])
    ap.add_argument("--exp1-e-correct", type=Path, nargs="*", default=[])
    ap.add_argument("--exp2-e-incorrect", type=Path, nargs="*", default=[])
    ap.add_argument("--exp2-e-correct", type=Path, nargs="*", default=[])
    args = ap.parse_args(argv)
    d = domain(args.domain)
    ts = datetime.now(tz=UTC).strftime("%Y%m%dT%H%M%SZ")
    if args.fake:
        out = paths.run_dir(f"{OUT_PREFIX}_{args.domain}_FAKE_{ts}")
        (out / FAKE_MARKER).write_text(
            f"fake extension analysis ({args.domain})\n", encoding="utf-8", newline="\n"
        )
        spec, ext = fake_inputs(out / "inputs", args.domain)
        report = analyze(args.domain, ext, seed=FAKE_SEED, **fake_runs(spec))
        files = write_outputs(report, out, prefix=FAKE_FILE_PREFIX)
    else:
        runs = {
            "exp1_e_incorrect": args.exp1_e_incorrect,
            "exp1_e_correct": args.exp1_e_correct,
            "exp2_e_incorrect": args.exp2_e_incorrect,
            "exp2_e_correct": args.exp2_e_correct,
        }
        if not any(runs.values()):
            ap.error("name at least one extension run directory (or --fake)")
        ext = load_domain_ext(args.domain)
        seed_component = str(get(ext, "analysis.bootstrap.seed_component"))
        seed = component_seed(domain_master_seed(args.domain), seed_component)
        report = analyze(args.domain, ext, seed=seed, **runs)
        out = paths.run_dir(f"{OUT_PREFIX}_{args.domain}_{ts}")
        files = write_outputs(report, out)
        record = {
            "domain": args.domain,
            "ext_config": d.ext_config,
            "ext_config_sha256": file_sha256(paths.resolve(d.ext_config)),
            "master_seed_source": d.seed_source,
            "seed_component": seed_component,
            "seed": seed,
            "runs": {k: [_shown(p) for p in v] for k, v in runs.items()},
            "wrapped": "analysis/ext_contrasts.py (unchanged; tag detox-ext-frozen)",
        }
        info = out / "inputs.json"
        info.write_text(json.dumps(record, indent=1), encoding="utf-8", newline="\n")
        files.append(info)
    for f in files:
        print(f"[ext_analysis] -> {_shown(f)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
