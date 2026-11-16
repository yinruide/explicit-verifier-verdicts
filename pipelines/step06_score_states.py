"""Step 06 — the formal scores table of V1 and V2 (0.5 step 6; App. A.2; task 2026-09-24 s. 4).

    python pipelines/step06_score_states.py reuse      # laptop: D_fit + pool from the probe
    python pipelines/step06_score_states.py inputs     # laptop: D_gate + D_Exp1 -> to_score
    python pipelines/step06_score_states.py numeric-ref [--config-ids A B]   # laptop: 203 D_fit
    python pipelines/step06_score_states.py download --records R1 R2          # pod
    python pipelines/step06_score_states.py score --input F --input-sha256 H --records R1 R2 \\
        --numeric-check REF --numeric-check-sha256 H2 --splits-dir S --gate-attempts G --out OUT
    python pipelines/step06_score_states.py install --pulled DIR [--pod-sha256 FILE]  # laptop
    python pipelines/step06_score_states.py verify     # laptop: every table complete, one scale

II.4.2's diagnostics (decisions.md D-2026-09-25c; the perturbations are built on the laptop
before the pod, the pod only scores them):

    python pipelines/step06_score_states.py perturb-sample     # 150 E-UNSAFE + 150 E-SAFE
    python pipelines/step06_score_states.py perturb-lock       # hash the prefix / paraphrase files
    python pipelines/step06_score_states.py paraphrase --budget 2 [--limit N] --confirm   # OpenAI
    python pipelines/step06_score_states.py judge-paraphrases --budget 2 [--limit N] --confirm
    python pipelines/step06_score_states.py perturb-inputs     # the pod's perturbed input
    python pipelines/step06_score_states.py numeric-check ...  # pod: the 203 states only
    python pipelines/step06_score_states.py score-perturbed ...   # pod: as score, other input
    python pipelines/step06_score_states.py install --perturbed --pulled DIR ...   # laptop
    python pipelines/step06_score_states.py diagnostics        # runs/step06_diagnostics/<run_id>

Which verifiers: the probe config_ids in ``models.verifiers.v1.id`` / ``v2.id``; TBD refuses.
Every score comes from the ONE factory (``rafr.verifiers.factory``), i.e. from the step 2 probe's
own scoring code, so the table is on the probe's scale:

* **D_fit and the unlabeled pool** were already scored by the probe. ``reuse`` copies those rows
  from ``data/scores/probe/<config_id>.jsonl`` -- after checking that the table is the one the
  configuration's **cuda** probe record lists (its sha256), and row by row that
  ``model_version`` (repo@commit) and ``prompt_hash`` are what the factory would produce now
  (from the probe's records and the locked prompt files), and that the recorded dtype, batch
  size, text budget and revision are the factory's; one difference stops the step. The probe
  file itself is only read.
* **D_gate and D_Exp1** are scored by the factory in formal mode through batch_score (the
  D_gate_reserve lock, II.6.1, is checked on the input and needs the split file on the machine;
  the reserve is scored only once a gate attempt has failed, with ``inputs --include-reserve``).

**GPU only** (2026-09-24, decisions.md D-2026-09-24c point 6): every formal score comes
from the fixed pod (``models.verifiers.formal_pod``: image and GPU, checked before anything is
loaded), from a cuda probe record, on CUDA -- the encoder included. A configuration the probe ran
on the laptop CPU only has no cuda record and is refused until the probe has run it on the GPU.
Before a configuration scores a single state, ``score`` re-scores the 203 D_fit states of
``numeric-ref`` (drawn with a fixed seed) and compares them with the probe's scores bit for bit;
a difference stops the job and names the largest difference and how many states differ
(:mod:`rafr.verifiers.numeric_check`). Every forward pass has the probe's batch size, the short
last batch filled up with filler rows whose scores are dropped.

Output, per configuration: ``data/scores/<config_id>.jsonl``, one ``Score`` row per state --
state_id, verifier_id, model_version, score, timestamp (A.2), plus prompt_hash and sub_scores.
``install`` checks pulled tables (row count, state_id set equal to the input's, identity,
sha256 against the pod's copy) before merging; ``verify`` checks that every table covers
D_fit + pool + D_gate + D_Exp1 exactly (no reserve state while the reserve is locked) with one
model_version and one prompt_hash.
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
import time
from collections import deque
from collections.abc import Callable, Iterable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from typing import Any

import numpy as np

from pipelines import _validity_io as vio
from rafr.contracts.schemas import Score, Split, State
from rafr.core import paths
from rafr.core.config import assert_frozen, load_config, master_seed
from rafr.core.runtime import Spend, build_client, prices_for
from rafr.core.seeding import component_seed
from rafr.core.textio import crlf_rewrite_notice
from rafr.data.labels import LabelPurpose, read_labels, reserve_unlocked
from rafr.evaluator import judge
from rafr.llm.client import LLMClient, TransientRetriesExhaustedError
from rafr.logging.snapshot import git_head, write_snapshot
from rafr.logging.writer import read_jsonl
from rafr.validity import tau as tau_mod
from rafr.verifiers import (
    batch_score,
    diagnostics,
    factory,
    formal_pod,
    numeric_check,
    perturb,
    reserve_scores_table_path,
)
from rafr.verifiers.candidates import probe

REUSED_SPLITS: tuple[Split, ...] = (Split.D_FIT, Split.UNLABELED_POOL)
SCORED_SPLITS: tuple[Split, ...] = (Split.D_GATE, Split.D_EXP1)
STEP_DIR_NAME = "step06"
INPUT_NAME = "to_score.jsonl"
FORMAL_KIND = "cuda"  # formal scores come from the GPU only (D-2026-09-24c point 6)

Echo = Callable[[str], None]


class Step06Error(RuntimeError):
    pass


def step_dir() -> Path:
    return paths.DATA / STEP_DIR_NAME


def formal_table(config_id: str) -> Path:
    return paths.DATA_SCORES / f"{config_id}.jsonl"


def reserve_table(config_id: str) -> Path:
    """D_gate_reserve's scores, if it is ever scored (after a failed first gate): a table of its
    own, never merged into :func:`formal_table` (decisions.md D-2026-09-27)."""
    return reserve_scores_table_path(config_id, paths.DATA_SCORES)


def probe_table(config_id: str) -> Path:
    return paths.DATA_SCORES / "probe" / f"{config_id}.jsonl"


def numeric_ref_path() -> Path:
    """The 203 D_fit states and the probe's scores the pod must reproduce (step 6 and Exp 2)."""
    return paths.DATA / "numeric_check" / "reference.jsonl"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as fh:
        while block := fh.read(1 << 20):
            h.update(block)
    return h.hexdigest()


def load_states(split: Split) -> list[State]:
    p = paths.DATA_POOLS / f"states_{split.value}.jsonl"
    if not p.exists():
        raise Step06Error(f"no states for {split.value} at {p}; run step05 split")
    return [State.model_validate(r) for r in read_jsonl(p)]


def split_ids(splits: Sequence[Split]) -> set[str]:
    return {s.state_id for sp in splits for s in load_states(sp)}


# ---------------------------------------------------------------------------------------
# what the factory would produce now, without loading weights
# ---------------------------------------------------------------------------------------


def expected_identity(
    config_id: str,
    *,
    records_dirs: Sequence[Path] | None = None,
    prompt_dir: Path | None = None,
) -> dict[str, Any]:
    """``model_version`` / ``prompt_hash`` / ``dtype`` the factory gives ``config_id``: from the
    probe's records and the locked prompt files, with the factory's own parameter checks."""
    cfg = factory.config_for(config_id, prompt_dir)
    m = cfg.model
    factory.check_source(m)
    if m.is_lm:
        probe.check_prompt_lock(prompt_dir)
    rec = factory.find_probe_record(m.model_id, records_dirs, device_kind=FORMAL_KIND)
    settings = factory.static_checks(cfg, rec)
    now = probe.prompt_hash(cfg, prompt_dir)
    if m.is_lm and now != settings["prompt_hash"]:
        raise factory.ProbeMismatchError(
            f"{config_id}: prompt hash now {now}, the probe recorded {settings['prompt_hash']}"
        )
    return {
        "config_id": config_id,
        "model_version": f"{m.repo_id}@{rec.revision}",
        "prompt_hash": now,
        "dtype": rec.meta.get("dtype"),
        "device_kind": rec.device_kind,
        "probe_table_sha256": rec.config_meta(config_id).get("table_sha256"),
    }


def check_rows(
    rows: Sequence[Mapping[str, Any]], want_ids: set[str], ident: Mapping[str, Any], where: str
) -> list[Score]:
    """Every row a valid Score of this configuration with the expected identity, one per
    wanted state and nothing else. Raises :class:`Step06Error` on the first problem."""
    out: list[Score] = []
    seen: set[str] = set()
    for i, r in enumerate(rows):
        s = Score.model_validate(r)
        if s.verifier_id != ident["config_id"]:
            raise Step06Error(f"{where} row {i}: verifier_id {s.verifier_id!r}")
        if s.model_version != ident["model_version"]:
            raise Step06Error(
                f"{where} row {i}: model_version {s.model_version} != {ident['model_version']}"
            )
        if s.prompt_hash != ident["prompt_hash"]:
            raise Step06Error(
                f"{where} row {i}: prompt_hash {s.prompt_hash} != {ident['prompt_hash']}"
            )
        if s.state_id in seen:
            raise Step06Error(f"{where}: state {s.state_id} appears twice")
        seen.add(s.state_id)
        out.append(s)
    if seen != want_ids:
        raise Step06Error(
            f"{where}: {len(seen)} rows; {len(seen - want_ids)} state(s) not wanted, "
            f"{len(want_ids - seen)} missing (expected {len(want_ids)})"
        )
    return out


def write_table(path: Path, rows: Sequence[Score]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".jsonl.tmp")
    with tmp.open("w", encoding="utf-8", newline="\n") as fh:
        for r in rows:
            fh.write(r.model_dump_json() + "\n")
    with crlf_rewrite_notice(path):  # the formal tables written before 2026-09-27 are CRLF
        tmp.replace(path)


def merge_into_formal(
    config_id: str, rows: Sequence[Score], ident: Mapping[str, Any], path: Path | None = None
) -> int:
    """Replace ``rows``' states in the formal table (or ``path``), keep every other row. Rows
    already there must carry the same identity: one table is one scale."""
    path = path or formal_table(config_id)
    kept: dict[str, Score] = {}
    for r in read_jsonl(path):
        s = Score.model_validate(r)
        if (s.model_version, s.prompt_hash) != (ident["model_version"], ident["prompt_hash"]):
            raise Step06Error(
                f"{path} already holds rows of another identity ({s.model_version}, "
                f"{s.prompt_hash}); move it away rather than mixing scales"
            )
        kept[s.state_id] = s
    for r in rows:
        kept[r.state_id] = r
    write_table(path, list(kept.values()))
    return len(kept)


def configured(models_cfg: Mapping[str, Any]) -> tuple[str, str]:
    factory.check_factory_path(models_cfg)
    return factory.configured_ids(models_cfg)


# ---------------------------------------------------------------------------------------
# commands
# ---------------------------------------------------------------------------------------


def cmd_reuse(models_cfg: Mapping[str, Any], echo: Echo) -> dict[str, Any]:
    want = split_ids(REUSED_SPLITS)
    out: dict[str, Any] = {}
    for cid in configured(models_cfg):
        ident = expected_identity(cid, records_dirs=factory.records_dirs_from(models_cfg))
        src = probe_table(cid)
        if not src.is_file():
            raise Step06Error(f"{src} not found: the probe has not scored {cid}")
        if sha256_file(src) != ident["probe_table_sha256"]:
            raise Step06Error(
                f"{src} is not the table {cid}'s cuda probe record lists (sha256 "
                f"{ident['probe_table_sha256']}): formal scores come from the GPU only "
                "(D-2026-09-24c); a laptop CPU run's table is not reused"
            )
        rows = check_rows(list(read_jsonl(src)), want, ident, str(src))
        n = merge_into_formal(cid, rows, ident)
        out[cid] = {"reused": len(rows), "table_rows": n, "probe_sha256": sha256_file(src)}
        echo(f"[step06] {cid}: reused {len(rows)} probe rows (D_fit + pool) -> {formal_table(cid)}")
    return out


def cmd_inputs(include_reserve: bool, echo: Echo) -> dict[str, Any]:
    splits = list(SCORED_SPLITS)
    if include_reserve:
        if not reserve_unlocked(paths.FROZEN_GATE_ATTEMPTS):
            raise Step06Error("D_gate_reserve is locked until a gate attempt has failed (II.6.1)")
        splits.append(Split.D_GATE_RESERVE)
    states = [s for sp in splits for s in load_states(sp)]
    items = [(s.state_id, s.text) for s in states]
    batch_score.refuse_locked_reserve(
        items, split_dir=paths.DATA_SPLITS, gate_attempts_path=paths.FROZEN_GATE_ATTEMPTS
    )
    d = step_dir()
    d.mkdir(parents=True, exist_ok=True)
    path = d / INPUT_NAME
    with path.open("w", encoding="utf-8", newline="\n") as fh:
        for sid, text in items:
            fh.write(json.dumps({"state_id": sid, "text": text}, ensure_ascii=False) + "\n")
    digest = sha256_file(path)
    (d / (INPUT_NAME + ".sha256")).write_text(
        f"{digest}  {INPUT_NAME}\n", encoding="utf-8", newline="\n"
    )
    echo(f"[step06] {len(items)} states ({', '.join(s.value for s in splits)}) -> {path}")
    echo(f"[step06] sha256 {digest}")
    return {"n": len(items), "splits": [s.value for s in splits], "sha256": digest}


def probe_table_for(cid: str, records_dirs: Sequence[Path] | None) -> tuple[Path, str]:
    """The probe table of ``cid`` and the sha256 its cuda record lists."""
    model_id, _ = factory.parse_config_id(cid)
    rec = factory.find_probe_record(model_id, records_dirs, device_kind=FORMAL_KIND)
    sha = rec.config_meta(cid).get("table_sha256")
    if not sha:
        raise Step06Error(f"{cid}: its cuda probe record lists no table sha256")
    return probe_table(cid), str(sha)


def cmd_numeric_ref(
    cfgs: Mapping[str, Any],
    records_dirs: Sequence[Path] | None,
    config_ids: Sequence[str] | None,
    echo: Echo,
) -> dict[str, Any]:
    """Laptop: the 203 D_fit states (seed component ``pod_numeric_check``), their text and each
    configuration's probe score from its cuda record's table -- what the pod must reproduce."""
    ids = list(config_ids) if config_ids else list(configured(cfgs["models"]))
    for cid in ids:
        factory.parse_config_id(cid)
    seed = component_seed(master_seed(cfgs), numeric_check.SEED_COMPONENT)
    states = numeric_check.draw_states(load_states(Split.D_FIT), seed)
    rows = numeric_check.build_reference(ids, states, lambda c: probe_table_for(c, records_dirs))
    path = numeric_ref_path()
    digest = numeric_check.write_reference(path, rows)
    echo(f"[step06] numeric check reference: {len(states)} D_fit states x {ids} -> {path}")
    echo(f"[step06] sha256 {digest} (seed {seed})")
    return {"config_ids": ids, "n": len(states), "seed": seed, "sha256": digest, "path": str(path)}


def check_pod(models_cfg: Mapping[str, Any], echo: Echo) -> dict[str, Any]:
    """The fixed formal pod (image and GPU), checked before any model is loaded."""
    try:
        return formal_pod.check_here(models_cfg, echo)
    except formal_pod.FormalPodError as exc:
        raise Step06Error(str(exc)) from exc


def cmd_download(
    models_cfg, records_dirs: Sequence[Path], echo: Echo, config_ids: Sequence[str] | None = None
) -> dict[str, Any]:
    """Pod: fetch the pinned commit of every GPU configuration and check each weight file's
    sha256 against the probe's download manifest. ``config_ids`` (an Exp 2 smoke run) replaces
    the configured V1 / V2."""
    from huggingface_hub import snapshot_download

    from rafr.verifiers import probe_run

    out: dict[str, Any] = {}
    for cid in list(config_ids) if config_ids else configured(models_cfg):
        model_id, _ = factory.parse_config_id(cid)
        rec = factory.find_probe_record(model_id, records_dirs, device_kind=FORMAL_KIND)
        m = probe.model(model_id)
        want = {w["path"]: w["sha256"] for w in rec.manifest.get("weight_files") or []}
        allow = list(probe_run.ALLOW_PATTERNS)
        if any(p.endswith(".bin") for p in want):
            allow.append("*.bin")
        local = Path(
            snapshot_download(
                m.repo_id,
                revision=rec.revision,
                allow_patterns=allow,
                ignore_patterns=probe_run.IGNORE_PATTERNS,
            )
        )
        bad = [p for p, digest in want.items() if probe_run.sha256_file(local / p) != digest]
        if bad or not want:
            raise Step06Error(f"{cid}: weight files differ from the probe's manifest: {bad}")
        out[cid] = {"repo_id": m.repo_id, "revision": rec.revision, "files_checked": len(want)}
        echo(f"[step06] {cid}: {m.repo_id}@{rec.revision}, {len(want)} weight file(s) verified")
    return out


def write_numeric_result(out_dir: Path, cid: str, result: Mapping[str, Any]) -> None:
    path = out_dir / "numeric_check" / f"{cid}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(result, ensure_ascii=False, indent=1), encoding="utf-8", newline="\n"
    )


def cmd_score(
    models_cfg: Mapping[str, Any],
    *,
    input_path: Path,
    input_sha256: str,
    records_dirs: Sequence[Path],
    numeric_ref: Path,
    numeric_ref_sha256: str,
    splits_dir: Path,
    gate_attempts: Path,
    out_dir: Path,
    echo: Echo,
    perturbed: bool = False,
) -> dict[str, Any]:
    """Pod: check the pod, then score the input with every configured configuration, formally
    (cuda record, CUDA run), one model at a time: the 203-state numeric check first, then the
    table, resumable and written as it goes, and its meta when it is done.

    ``perturbed``: the input is the II.4.2 perturbation items (``<state_id>#prefix`` /
    ``<state_id>#paraphrase``); the reserve lock is checked on their states (``perturb-inputs``
    has already checked on the laptop that every one is a D_fit or pool state)."""
    from rafr.verifiers import probe_run

    pod_env = check_pod(models_cfg, echo)
    items = probe_run.load_probe_inputs(input_path, input_sha256)
    base = [(perturb.split_item_id(i)[0], t) for i, t in items] if perturbed else items
    batch_score.refuse_locked_reserve(base, split_dir=splits_dir, gate_attempts_path=gate_attempts)
    reference = numeric_check.read_reference(numeric_ref, numeric_ref_sha256)
    out: dict[str, Any] = {}
    for cid in configured(models_cfg):
        started = time.monotonic()
        v = factory.refuse_legacy_guard(
            factory.make_verifier(cid, records_dirs=records_dirs, echo=echo, formal=True)
        )
        try:
            try:
                numbers = numeric_check.check_verifier(v, reference.get(cid, []), echo)
            except numeric_check.NumericCheckError as exc:
                write_numeric_result(out_dir, cid, {"error": str(exc), **exc.result})
                raise
            write_numeric_result(out_dir, cid, numbers)
            path = out_dir / "scores" / f"{cid}.jsonl"
            written = batch_score.run(
                verifier=v, items=items, out_path=path, resume=True, echo=echo
            )
            info = {
                "config_id": cid,
                "rows_written_this_run": written,
                "n_input": len(items),
                "input_sha256": input_sha256,
                "table_sha256": sha256_file(path),
                "verifier": v.describe(),
                "numeric_check": {k: numbers[k] for k in ("n", "n_different", "max_abs_diff")},
                "formal_pod": pod_env,
                "elapsed_s": round(time.monotonic() - started, 1),
                "finished_at": datetime.now(tz=UTC).isoformat(timespec="seconds"),
                "git_head": probe_run.environment().get("git_head"),
                "environment": probe_run.environment(),
            }
        finally:
            v.close()
            probe_run.release_gpu()
        meta = out_dir / "meta" / f"{cid}.json"
        meta.parent.mkdir(parents=True, exist_ok=True)
        meta.write_text(
            json.dumps(info, ensure_ascii=False, indent=1), encoding="utf-8", newline="\n"
        )
        out[cid] = info
        echo(f"[step06] {cid}: {len(items)} {'items' if perturbed else 'states'} -> {path}")
    return out


def cmd_numeric_check(
    models_cfg: Mapping[str, Any],
    *,
    records_dirs: Sequence[Path],
    numeric_ref: Path,
    numeric_ref_sha256: str,
    out_dir: Path,
    echo: Echo,
) -> dict[str, Any]:
    """Pod: the 203-state check alone, for every configured configuration (runbook step 3:
    nothing is scored until it has passed). ``score`` repeats it before it scores anything."""
    from rafr.verifiers import probe_run

    pod_env = check_pod(models_cfg, echo)
    reference = numeric_check.read_reference(numeric_ref, numeric_ref_sha256)
    out: dict[str, Any] = {"formal_pod": pod_env, "configs": {}}
    for cid in configured(models_cfg):
        v = factory.refuse_legacy_guard(
            factory.make_verifier(cid, records_dirs=records_dirs, echo=echo, formal=True)
        )
        try:
            try:
                numbers = numeric_check.check_verifier(v, reference.get(cid, []), echo)
            except numeric_check.NumericCheckError as exc:
                write_numeric_result(out_dir, cid, {"error": str(exc), **exc.result})
                raise
            write_numeric_result(out_dir, cid, numbers)
        finally:
            v.close()
            probe_run.release_gpu()
        out["configs"][cid] = {k: numbers[k] for k in ("n", "n_different", "max_abs_diff")}
        echo(
            f"[step06] {cid}: numeric check PASSED -- {numbers['n']} D_fit states equal the "
            "probe's scores bit for bit"
        )
    echo("[step06] numeric check: every configuration PASSED")
    return out


def refuse_outside_sample_splits(state_ids: Iterable[str], splits_dir: Path) -> None:
    """The perturbation items come from D_fit + the pool only: a state of D_gate, D_gate_reserve
    or D_Exp1 is refused (the split files list x0 ids; both id forms are checked)."""
    forbidden = batch_score.selection_forbidden_ids(splits_dir)
    hit = [sid for sid in state_ids if any(sid in ids for ids in forbidden.values())]
    if hit:
        raise Step06Error(
            f"{len(hit)} perturbation item(s) belong to D_gate / D_gate_reserve / D_Exp1 "
            f"(e.g. {hit[:3]}); the II.4.2 diagnostics use D_fit + pool only"
        )


def cmd_install(
    models_cfg: Mapping[str, Any],
    pulled: Path,
    input_path: Path,
    pod_sha256: Mapping[str, str] | None,
    echo: Echo,
    perturbed: bool = False,
) -> dict[str, Any]:
    """Check pulled tables against the input and the probe identity, then merge them.

    ``perturbed``: the tables of the II.4.2 perturbation items. They are checked the same way
    and written whole to ``data/step06/perturb/scores/<config_id>.jsonl``, never into the formal
    tables (they are not states)."""
    want = {str(r["state_id"]) for r in read_jsonl(input_path)}
    out: dict[str, Any] = {}
    for cid in configured(models_cfg):
        src = Path(pulled) / "scores" / f"{cid}.jsonl"
        if not src.is_file():
            continue
        if pod_sha256 is not None and pod_sha256.get(src.name) != sha256_file(src):
            raise Step06Error(f"{src}: sha256 differs from the pod's copy")
        ident = expected_identity(cid, records_dirs=factory.records_dirs_from(models_cfg))
        rows = check_rows(list(read_jsonl(src)), want, ident, str(src))
        if perturbed:
            dst = perturb.scores_dir() / f"{cid}.jsonl"
            write_table(dst, rows)
            out[cid] = {"installed": len(rows), "path": str(dst), "sha256": sha256_file(dst)}
            echo(f"[step06] {cid}: {len(rows)} perturbation rows checked -> {dst}")
            continue
        reserve = split_ids([Split.D_GATE_RESERVE])
        n_reserve = sum(r.state_id in reserve for r in rows)
        if n_reserve and n_reserve != len(rows):
            raise Step06Error(
                f"{src}: {n_reserve} D_gate_reserve rows mixed with {len(rows) - n_reserve} "
                "others; the reserve is installed on its own (D-2026-09-27)"
            )
        if n_reserve:
            if not reserve_unlocked(paths.FROZEN_GATE_ATTEMPTS):
                raise Step06Error("D_gate_reserve is locked until a gate attempt has failed")
            dst = reserve_table(cid)  # its own file; the formal table is not touched
        else:
            dst = formal_table(cid)
        n = merge_into_formal(cid, rows, ident, dst)
        out[cid] = {"installed": len(rows), "table_rows": n, "path": str(dst)}
        echo(f"[step06] {cid}: {len(rows)} rows checked and merged -> {dst}")
    if not out:
        raise Step06Error(f"no table of a configured verifier under {pulled}/scores")
    return out


def cmd_verify(models_cfg: Mapping[str, Any], echo: Echo) -> dict[str, Any]:
    splits = [*REUSED_SPLITS, *SCORED_SPLITS]
    unlocked = reserve_unlocked(paths.FROZEN_GATE_ATTEMPTS)
    reserve = split_ids([Split.D_GATE_RESERVE])
    out: dict[str, Any] = {}
    for cid in configured(models_cfg):
        path = formal_table(cid)
        ids = {str(r["state_id"]) for r in read_jsonl(path)}
        if ids & reserve:
            raise Step06Error(
                f"{path} holds D_gate_reserve states; the reserve has a table of its own "
                f"({reserve_table(cid)}, D-2026-09-27)"
            )
        ident = expected_identity(cid, records_dirs=factory.records_dirs_from(models_cfg))
        check_rows(list(read_jsonl(path)), split_ids(splits), ident, str(path))
        out[cid] = {"rows": len(ids), "sha256": sha256_file(path), **ident}
        echo(f"[step06] {cid}: {len(ids)} rows, complete, one identity ({ident['model_version']})")
        rpath = reserve_table(cid)
        if rpath.is_file():
            if not unlocked:
                raise Step06Error(f"{rpath} exists while the reserve is locked (II.6.1)")
            check_rows(list(read_jsonl(rpath)), reserve, ident, str(rpath))
            out[cid]["reserve"] = {"rows": len(reserve), "sha256": sha256_file(rpath)}
            echo(f"[step06] {cid}: reserve table {rpath}, {len(reserve)} rows, complete")
    return out


# ---------------------------------------------------------------------------------------
# II.4.2: the wording perturbations (laptop, before the pod; D-2026-09-25c)
# ---------------------------------------------------------------------------------------

PARAPHRASE_BUDGET_USD = 2.0  # OpenAI, all paraphrase runs together (2026-09-25)
JUDGE_BUDGET_USD = 2.0  # Anthropic, all E runs on the paraphrases together (D-2026-09-25d)
PARAPHRASE_RUN_PREFIX = "step06_paraphrase_"
JUDGE_RUN_PREFIX = "step06_judge_paraphrases_"
_END = object()


def cmd_perturb_sample(cfgs: Mapping[str, Any], echo: Echo) -> dict[str, Any]:
    """Draw the 150 E-UNSAFE + 150 E-SAFE states of D_fit + pool and their prefixes. Reads y*
    of those two splits only (purpose VERIFIER_DIAGNOSTICS). Re-running reproduces the file
    byte for byte; a different draw over an existing file is refused."""
    states: list[State] = []
    labels: dict[str, Any] = {}
    splits: dict[str, Split] = {}
    for sp in perturb.SAMPLE_SPLITS:
        ys = {y.state_id: y.label for y in read_labels(sp, LabelPurpose.VERIFIER_DIAGNOSTICS)}
        for s in load_states(sp):
            states.append(s)
            labels[s.state_id] = ys[s.state_id]
            splits[s.state_id] = sp
    seed = component_seed(master_seed(cfgs), perturb.SAMPLE_SEED_COMPONENT)
    rows = [r.as_dict() for r in perturb.draw_sample(states, labels, splits, seed)]
    path = perturb.sample_path()
    if path.exists() and perturb.read_jsonl(path) != rows:
        raise Step06Error(f"{path} exists and holds a different sample; move it away first")
    digest = perturb.write_jsonl(path, rows)
    n = {lab: sum(r["e_label"] == lab for r in rows) for lab in ("UNSAFE", "SAFE")}
    echo(f"[step06] perturbation sample: {len(rows)} states {n} -> {path} (sha256 {digest})")
    return {"n": len(rows), "by_label": n, "seed": seed, "sha256": digest, "path": str(path)}


def cmd_perturb_lock(echo: Echo) -> dict[str, Any]:
    """Hash the prefix table and the paraphrase prompt into their lock file. Refused once a
    paraphrase has been bought under a lock the files no longer match."""
    lock = perturb.prompts_dir() / perturb.LOCK_FILE
    if perturb.paraphrases_path().exists() and lock.is_file():
        perturb.check_lock()  # raises if the files changed after paraphrases were bought
    payload = perturb.write_lock()
    echo(f"[step06] locked {payload['files']} -> {lock}")
    return payload


def run_ordered(
    items: Sequence[Any],
    work: Callable[[Any], Any],
    *,
    workers: int,
    on_done: Callable[[Any, Any], None],
    should_stop: Callable[[], str | None],
) -> str | None:
    """``work(item)`` on ``workers`` threads; results reach ``on_done`` in input order, so the
    output file is that of a serial run. After each result ``should_stop()`` may give a reason:
    nothing new starts, the calls already in flight finish and are recorded (they are paid
    for), and the reason is returned. A request whose transient retries ran out stops it too."""
    it = iter(items)
    pending: deque = deque()
    n = max(1, int(workers))
    reason: str | None = None
    with ThreadPoolExecutor(max_workers=n) as pool:
        while True:
            while reason is None and len(pending) < n:
                item = next(it, _END)
                if item is _END:
                    break
                pending.append((item, pool.submit(work, item)))
            if not pending:
                break
            item, fut = pending.popleft()
            try:
                result = fut.result()
            except TransientRetriesExhaustedError as exc:
                reason = reason or f"transient_errors: {exc}"
                continue
            on_done(item, result)
            if reason is None:
                reason = should_stop()
    return reason


def earlier_spend(prefix: str, purpose: str, model: str, current: Path | None) -> float:
    """Dollars bought by earlier runs of the same command (their ledgers, live calls only), so
    that ``--budget`` caps the whole task and not one run."""
    prices = prices_for(model)
    if prices is None:
        raise Step06Error(f"no list price for {model!r}: the budget cannot be enforced")
    usd = 0.0
    for ledger in sorted(paths.RUNS.glob(f"{prefix}*/ledger.jsonl")):
        if current is not None and ledger.parent == current:
            continue
        for row in read_jsonl(ledger):
            if row.get("purpose") == purpose and not row.get("cached"):
                usd += prices.cost(row)
    return usd


def _budget_stop(spend: Spend, before: float, budget: float) -> Callable[[], str | None]:
    def check() -> str | None:
        total = before + float(spend.usd or 0.0)
        return f"budget: ${total:.4f} >= ${budget:.2f}" if total >= budget else None

    return check


def _projection(spend: Spend, n_new: int, n_left: int) -> dict[str, Any]:
    usd = float(spend.usd or 0.0)
    live = spend.live_calls
    unit = usd / live if live else None
    return {
        "items_this_run": n_new,
        "live_calls_this_run": live,
        "usd_this_run": round(usd, 6),
        "usd_per_live_call": unit,
        "items_left": n_left,
        "projected_remaining_usd": None if unit is None else unit * n_left,
    }


def cmd_paraphrase(
    cfgs: Mapping[str, Any],
    client: LLMClient,
    *,
    run_dir: Path,
    budget: float,
    limit: int | None,
    workers: int,
    echo: Echo,
) -> dict[str, Any]:
    """One paraphrase per sampled state (G_fixed's model and call configuration, the locked
    paraphrase prompt); an invalid one is recorded with its reason and not regenerated.
    Resumable: a state with a record is skipped. Stops once all runs together reach
    ``budget`` dollars (list price, live calls)."""
    perturb.check_lock()
    sample = perturb.read_sample()
    pcfg = perturb.paraphrase_config(
        cfgs["models"],
        cfgs["generation"],
        base_seed=component_seed(master_seed(cfgs), perturb.PARAPHRASE_SEED_COMPONENT),
    )
    before = earlier_spend(PARAPHRASE_RUN_PREFIX, perturb.PURPOSE, pcfg.model, run_dir)
    spend = Spend(client.ledger, prices_for(pcfg.model), perturb.PURPOSE)
    path = perturb.paraphrases_path()
    done = perturb.read_rows_by_state(path)
    todo = [r for r in sample if r.state_id not in done]
    if limit is not None:
        todo = todo[:limit]
    echo(
        f"[step06] paraphrase: {len(done)} of {len(sample)} done, {len(todo)} to buy now "
        f"({pcfg.model}, decoding {pcfg.decoding}); earlier runs ${before:.4f}, cap ${budget:.2f}"
    )
    if before >= budget:
        raise Step06Error(f"the paraphrase budget ${budget:.2f} is spent (${before:.4f})")

    def work(row: perturb.SampleRow) -> dict[str, Any]:
        req = perturb.paraphrase_request(pcfg, row)
        return perturb.paraphrase_row(pcfg, row, req, client.complete(req))

    path.parent.mkdir(parents=True, exist_ok=True)
    new: list[dict[str, Any]] = []
    with path.open("a", encoding="utf-8", newline="\n") as fh:

        def on_done(_row: Any, rec: dict[str, Any]) -> None:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
            fh.flush()
            new.append(rec)

        reason = run_ordered(
            todo,
            work,
            workers=workers,
            on_done=on_done,
            should_stop=_budget_stop(spend, before, budget),
        )
    rows = perturb.read_rows_by_state(path)
    invalid: dict[str, int] = {}
    for r in rows.values():
        if not r["valid"]:
            invalid[r["invalid_reason"]] = invalid.get(r["invalid_reason"], 0) + 1
    out = {
        "n_sampled": len(sample),
        "n_done": len(rows),
        "n_valid": sum(r["valid"] for r in rows.values()),
        "invalid_by_reason": invalid,
        "truncated": sum(r.get("stop_reason") in ("length", "max_tokens") for r in rows.values()),
        "complete": len(rows) == len(sample),
        "stop_reason": reason,
        "earlier_runs_usd": round(before, 6),
        "budget_usd": budget,
        **_projection(spend, len(new), len(sample) - len(rows)),
    }
    echo(f"[step06] paraphrase: {json.dumps(out)}")
    return out


def judgment_row(state_id: str, rec: Any, failures: list[str], ecfg: Any) -> dict[str, Any]:
    return {
        "state_id": state_id,
        "label": None if rec is None else rec.label.value,
        "reason": None if rec is None else rec.reason,
        "e_error": failures[-1] if rec is None and failures else None,
        "parse_failures": failures,
        "e_identity": ecfg.identity(),
        "record": None if rec is None else rec.model_dump(mode="json"),
    }


def cmd_judge_paraphrases(
    cfgs: Mapping[str, Any],
    client: LLMClient,
    *,
    run_dir: Path,
    budget: float,
    limit: int | None,
    workers: int,
    echo: Echo,
) -> dict[str, Any]:
    """E (the formal one: prompt, version, decoding, calling convention) on every VALID
    paraphrase, one judgment each. E failing after its parse retries is recorded (label None,
    the reason) and the state leaves the "E label unchanged" column; it does not stop the run.
    Resumable; stops once all runs together reach ``budget`` dollars."""
    ecfg = judge.build_config(cfgs["models"])
    sample = perturb.read_sample()
    paras = perturb.read_rows_by_state(perturb.paraphrases_path())
    if len(paras) != len(sample):
        raise Step06Error(f"{len(sample) - len(paras)} state(s) have no paraphrase record yet")
    path = perturb.judgments_path()
    done = perturb.read_rows_by_state(path)
    for sid, r in done.items():
        judge.check_same_e(r["e_identity"], ecfg, where=f"{path} ({sid})")
    valid = [r for r in sample if paras[r.state_id]["valid"]]
    todo = [r for r in valid if r.state_id not in done]
    if limit is not None:
        todo = todo[:limit]
    before = earlier_spend(JUDGE_RUN_PREFIX, judge.PURPOSE, ecfg.model, run_dir)
    spend = Spend(client.ledger, prices_for(ecfg.model), judge.PURPOSE)
    echo(
        f"[step06] E on paraphrases: {len(done)} of {len(valid)} valid done, {len(todo)} to buy "
        f"now; earlier runs ${before:.4f}, cap ${budget:.2f}"
    )
    if before >= budget:
        raise Step06Error(f"the E budget ${budget:.2f} is spent (${before:.4f})")

    def work(row: perturb.SampleRow) -> dict[str, Any]:
        text = paras[row.state_id]["paraphrase"]
        try:
            rec, failures = judge.judge_with_failures(client, ecfg, text)
        except judge.EvaluatorError as exc:
            return judgment_row(row.state_id, None, list(exc.failures or [str(exc)]), ecfg)
        return judgment_row(row.state_id, rec, failures, ecfg)

    path.parent.mkdir(parents=True, exist_ok=True)
    new: list[dict[str, Any]] = []
    with path.open("a", encoding="utf-8", newline="\n") as fh:

        def on_done(_row: Any, rec: dict[str, Any]) -> None:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
            fh.flush()
            new.append(rec)

        reason = run_ordered(
            todo,
            work,
            workers=workers,
            on_done=on_done,
            should_stop=_budget_stop(spend, before, budget),
        )
    rows = perturb.read_rows_by_state(path)
    y0 = {r.state_id: r.e_label for r in sample}
    judged = [r for r in rows.values() if r["label"] is not None]
    out = {
        "n_valid_paraphrases": len(valid),
        "n_done": len(rows),
        "n_judged": len(judged),
        "n_e_failed": len(rows) - len(judged),
        "n_label_unchanged": sum(r["label"] == y0[r["state_id"]] for r in judged),
        "complete": len(rows) == len(valid),
        "stop_reason": reason,
        "earlier_runs_usd": round(before, 6),
        "budget_usd": budget,
        **_projection(spend, len(new), len(valid) - len(rows)),
    }
    echo(f"[step06] E on paraphrases: {json.dumps(out)}")
    return out


def cmd_perturb_inputs(echo: Echo) -> dict[str, Any]:
    """The pod's perturbed input: every sampled state with its prefix and every valid
    paraphrase, ``{state_id: <item id>, text}`` only; checked to hold D_fit + pool states only."""
    perturb.check_lock()
    sample = perturb.read_sample()
    refuse_outside_sample_splits([r.state_id for r in sample], paths.DATA_SPLITS)
    rows = perturb.build_pod_input(
        sample, perturb.read_prefixes(), perturb.read_rows_by_state(perturb.paraphrases_path())
    )
    path = perturb.pod_input_path()
    digest = perturb.write_jsonl(path, rows)
    n = {k: sum(r["state_id"].endswith(perturb.ITEM_SEP + k) for r in rows) for k in perturb.KINDS}
    echo(f"[step06] perturbed input: {len(rows)} items {n} -> {path} (sha256 {digest})")
    return {"n": len(rows), "by_kind": n, "sha256": digest, "path": str(path)}


def load_perturbed_scores(cid: str, ident: Mapping[str, Any]) -> dict[str, dict[str, float]]:
    """``{kind: {state_id: score}}`` of one configuration's perturbation table, rows checked
    against the formal identity (one scale with the states' scores)."""
    path = perturb.scores_dir() / f"{cid}.jsonl"
    if not path.is_file():
        raise Step06Error(f"{path} not found: the pod has not scored the perturbed texts")
    out: dict[str, dict[str, float]] = {k: {} for k in perturb.KINDS}
    for r in read_jsonl(path):
        s = Score.model_validate(r)
        if (s.model_version, s.prompt_hash) != (ident["model_version"], ident["prompt_hash"]):
            raise Step06Error(f"{path}: a row of another identity ({s.model_version})")
        sid, kind = perturb.split_item_id(s.state_id)
        out[kind][sid] = s.score
    return out


def fit_or_load_tau(
    ids: Mapping[str, str], d_fit: list[State], scores: Mapping[str, Mapping[str, float]]
) -> tuple[dict[str, float], str]:
    """tau_i: ``frozen/tau.json`` when step 7 has run; otherwise fitted here on D_fit with step
    7's own rule and reading (same scores, same labels, so the same number), marked as such."""
    if paths.FROZEN_TAU.exists():
        fits = tau_mod.load_tau(paths.FROZEN_TAU)
        return {v: float(fits[v].tau) for v in diagnostics.VERIFIERS}, "frozen/tau.json"
    labels = vio.load_labels_for(d_fit, Split.D_FIT, LabelPurpose.TAU_FIT)
    taus = {}
    for v in diagnostics.VERIFIERS:
        s, y = vio.aligned(d_fit, scores[v], labels)
        taus[v] = float(tau_mod.fit_tau(s, y, ids[v]).tau)
    return taus, "fitted here on D_fit with step 7's rule (F1-max, plateau midpoint); not frozen"


def cmd_diagnostics(cfgs: Mapping[str, Any], run_dir: Path, echo: Echo) -> dict[str, Any]:
    """II.4.2's full table on D_fit + pool (runs/step06_diagnostics/<run_id>/)."""
    models = cfgs["models"]
    ids = dict(zip(diagnostics.VERIFIERS, configured(models), strict=True))
    idents = {
        v: expected_identity(cid, records_dirs=factory.records_dirs_from(models))
        for v, cid in ids.items()
    }
    by_split = {sp: load_states(sp) for sp in perturb.SAMPLE_SPLITS}
    y = {}
    for sp, states in by_split.items():
        got = {r.state_id: r.label for r in read_labels(sp, LabelPurpose.VERIFIER_DIAGNOSTICS)}
        y.update({s.state_id: int(got[s.state_id].value == "UNSAFE") for s in states})
    all_states = [s for states in by_split.values() for s in states]
    scores = {v: vio.load_scores_for(all_states, cid) for v, cid in ids.items()}
    taus, tau_source = fit_or_load_tau(ids, by_split[Split.D_FIT], scores)
    manifest = json.loads(paths.MANIFEST.read_text(encoding="utf-8"))["labels"]
    p = int(manifest["n_unsafe"]) / int(manifest["n_labelled"])
    pops = {
        "D_fit + pool": all_states,
        "D_fit (in-sample for tau)": by_split[Split.D_FIT],
        "pool": by_split[Split.UNLABELED_POOL],
    }
    single: dict[str, Any] = {}
    pair: dict[str, Any] = {}
    for name, states in pops.items():
        yy = np.array([y[s.state_id] for s in states])
        arr = {v: np.array([scores[v][s.state_id] for s in states]) for v in diagnostics.VERIFIERS}
        single[name] = {v: diagnostics.single(arr[v], yy, taus[v], p) for v in arr}
        pair[name] = diagnostics.pair(arr["v1"], taus["v1"], arr["v2"], taus["v2"], yy)
    sample = [r.as_dict() for r in perturb.read_sample()]
    perturbed = {k: {} for k in perturb.KINDS}
    for v, cid in ids.items():
        per = load_perturbed_scores(cid, idents[v])
        for k in perturb.KINDS:
            perturbed[k][v] = per[k]
    stab = diagnostics.stability(
        sample,
        scores,
        perturbed,
        taus,
        perturb.read_rows_by_state(perturb.paraphrases_path()),
        perturb.read_rows_by_state(perturb.judgments_path()),
    )
    mx = {
        "run_id": run_dir.name,
        "git_head": git_head(),
        "verifiers": {
            v: {"config_id": ids[v], "model_version": idents[v]["model_version"]}
            for v in diagnostics.VERIFIERS
        },
        "tau": taus,
        "tau_source": tau_source,
        "p": p,
        "single": single,
        "pair": pair,
        "stability": stab,
        "sample": {
            "path": str(perturb.sample_path()),
            "sha256": sha256_file(perturb.sample_path()),
        },
        "prompts": perturb.check_lock()["files"],
    }
    (run_dir / "metrics.json").write_text(
        json.dumps(mx, ensure_ascii=False, indent=1, default=str), encoding="utf-8", newline="\n"
    )
    (run_dir / "report.md").write_text(
        diagnostics.render_report(mx), encoding="utf-8", newline="\n"
    )
    echo(f"[step06] diagnostics -> {run_dir / 'report.md'}")
    return {"run_dir": str(run_dir), "tau": taus, "tau_source": tau_source}


LAPTOP_COMMANDS = (
    "reuse", "inputs", "numeric-ref", "install", "verify", "perturb-sample", "perturb-lock",
    "paraphrase", "judge-paraphrases", "perturb-inputs", "diagnostics",
)  # fmt: skip
POD_COMMANDS = ("download", "numeric-check", "score", "score-perturbed")
PAID_COMMANDS = ("paraphrase", "judge-paraphrases")
PARAPHRASE_SECTIONS = ("models.controller", "models.g_fixed", "generation", "refusal_phrases")


def laptop_run_dir(command: str, run_id: str | None) -> Path:
    ts = f"{datetime.now(tz=UTC):%Y%m%dT%H%M%SZ}"
    if command == "diagnostics":  # the task's location: runs/step06_diagnostics/<run_id>/
        d = paths.RUNS / "step06_diagnostics" / (run_id or ts)
        d.mkdir(parents=True, exist_ok=True)
        return d
    return paths.run_dir(f"step06_{command.replace('-', '_')}_{ts}")


def main(  # noqa: C901 - one dispatch over the commands
    argv: Sequence[str] | None = None,
    echo: Echo | None = None,
    client: LLMClient | None = None,
) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("command", choices=(*LAPTOP_COMMANDS, *POD_COMMANDS))
    ap.add_argument("--include-reserve", action="store_true", help="inputs: after a failed gate")
    ap.add_argument("--records", type=Path, nargs="+", help="the probe's records directories")
    ap.add_argument(
        "--config-ids", nargs="+", help="download / numeric-ref: instead of V1 / V2 (Exp 2 smoke)"
    )
    ap.add_argument("--input", type=Path, help="score / install: the to_score input")
    ap.add_argument("--input-sha256", help="score: sha256 recorded when the input was built")
    ap.add_argument("--splits-dir", type=Path, default=None)
    ap.add_argument("--gate-attempts", type=Path, default=None)
    ap.add_argument("--out", type=Path, default=None, help="score / numeric-check: output dir")
    ap.add_argument("--pulled", type=Path, help="install: directory with scores/ (and meta/)")
    ap.add_argument("--pod-sha256", type=Path, help="install: JSON {file name: sha256} from pod")
    ap.add_argument("--perturbed", action="store_true", help="install: the perturbation tables")
    ap.add_argument("--numeric-check", type=Path, help="score: the numeric-ref file (as pushed)")
    ap.add_argument("--numeric-check-sha256", help="score: sha256 numeric-ref recorded")
    ap.add_argument("--budget", type=float, help="paraphrase / judge-paraphrases: USD, all runs")
    ap.add_argument("--limit", type=int, help="paraphrase / judge-paraphrases: a dry run of N")
    ap.add_argument("--workers", type=int, default=4, help="paraphrase / judge-paraphrases")
    ap.add_argument("--confirm", action="store_true", help="paraphrase / judge: they cost money")
    ap.add_argument("--run-id", help="diagnostics: the run directory's name")
    args = ap.parse_args(argv)
    say = echo or (lambda s: print(s, flush=True))
    cfgs = load_config()
    if not (args.command in ("download", "numeric-ref") and args.config_ids):
        assert_frozen(cfgs, only_sections=["models.verifiers"])
    models = cfgs["models"]
    records = list(args.records) if args.records else factory.records_dirs_from(models)

    run_dir: Path | None = None
    if args.command in PAID_COMMANDS:
        if not args.confirm:
            raise SystemExit(f"{args.command} buys API calls: add --confirm")
        cap = PARAPHRASE_BUDGET_USD if args.command == "paraphrase" else JUDGE_BUDGET_USD
        if args.budget is None or not (0 < args.budget <= cap):
            raise SystemExit(f"{args.command} needs --budget in (0, {cap}] (2026-09-25)")
        if args.command == "paraphrase":
            assert_frozen(
                cfgs,
                only_sections=list(PARAPHRASE_SECTIONS),
                extra_files=[
                    perturb.prompts_dir() / perturb.PREFIX_FILE,
                    perturb.prompts_dir() / perturb.PARAPHRASE_FILE,
                ],
            )
        else:
            assert_frozen(cfgs, only_sections=["models.evaluator"])
    if args.command in LAPTOP_COMMANDS:
        run_dir = laptop_run_dir(args.command, args.run_id)
        write_snapshot(run_dir, step="step06", configs=cfgs, extra={"command": args.command})
    if args.command in PAID_COMMANDS:
        assert run_dir is not None
        client = client or build_client(models, run_dir=run_dir)
    if args.command == "reuse":
        result = cmd_reuse(models, say)
    elif args.command == "inputs":
        result = cmd_inputs(args.include_reserve, say)
    elif args.command == "numeric-ref":
        result = cmd_numeric_ref(cfgs, records, args.config_ids, say)
    elif args.command == "download":
        result = cmd_download(models, records, say, args.config_ids)
    elif args.command == "numeric-check":
        if not (args.numeric_check and args.numeric_check_sha256 and args.out):
            raise SystemExit("numeric-check needs --numeric-check, --numeric-check-sha256, --out")
        result = cmd_numeric_check(
            models,
            records_dirs=records,
            numeric_ref=args.numeric_check,
            numeric_ref_sha256=args.numeric_check_sha256,
            out_dir=args.out,
            echo=say,
        )
    elif args.command in ("score", "score-perturbed"):
        need = (args.input, args.input_sha256, args.out, args.numeric_check)
        if not all(need) or not args.numeric_check_sha256:
            raise SystemExit(
                f"{args.command} needs --input, --input-sha256, --out, --numeric-check and "
                "--numeric-check-sha256 (the 203 D_fit states re-scored first, D-2026-09-24c)"
            )
        result = cmd_score(
            models,
            input_path=args.input,
            input_sha256=args.input_sha256,
            records_dirs=records,
            numeric_ref=args.numeric_check,
            numeric_ref_sha256=args.numeric_check_sha256,
            splits_dir=args.splits_dir or paths.DATA_SPLITS,
            gate_attempts=args.gate_attempts or paths.FROZEN_GATE_ATTEMPTS,
            out_dir=args.out,
            echo=say,
            perturbed=args.command == "score-perturbed",
        )
    elif args.command == "install":
        if not (args.pulled and args.input):
            raise SystemExit("install needs --pulled and --input")
        pod = json.loads(args.pod_sha256.read_text(encoding="utf-8")) if args.pod_sha256 else None
        result = cmd_install(models, args.pulled, args.input, pod, say, perturbed=args.perturbed)
    elif args.command == "verify":
        result = cmd_verify(models, say)
    elif args.command == "perturb-sample":
        result = cmd_perturb_sample(cfgs, say)
    elif args.command == "perturb-lock":
        result = cmd_perturb_lock(say)
    elif args.command in PAID_COMMANDS:
        assert client is not None and run_dir is not None
        fn = cmd_paraphrase if args.command == "paraphrase" else cmd_judge_paraphrases
        result = fn(
            cfgs,
            client,
            run_dir=run_dir,
            budget=float(args.budget),
            limit=args.limit,
            workers=args.workers,
            echo=say,
        )
    elif args.command == "perturb-inputs":
        result = cmd_perturb_inputs(say)
    else:
        assert run_dir is not None
        result = cmd_diagnostics(cfgs, run_dir, say)
    if run_dir is not None:
        (run_dir / "result.json").write_text(
            json.dumps({"result": result, "git_head": git_head()}, indent=1, default=str),
            encoding="utf-8",
            newline="\n",
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
