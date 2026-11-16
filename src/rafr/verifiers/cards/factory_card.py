"""verifier_card.md for a configuration of the one scoring factory (App. A.2; task 2026-09-24 s. 3).

    python -m rafr.verifiers.cards.factory_card --config-id sg_2b.native --config-id enc_paradetox
    python -m rafr.verifiers.cards.factory_card --from-config          # V1 and V2 as configured

Writes ``src/rafr/verifiers/cards/<config_id>.md``. Everything in a card is copied from the step
2 probe's records (download manifest, meta, environment) and from the probe code the factory
imports -- nothing is typed in by hand and no weights are loaded:

* model identity; HF repo, commit hash, sha256 of every weight file;
* the prompt files the configuration's wording comes from, their sha256, and the prompt hash
  every scores-table row carries;
* how tau_i is determined: pending step 7 (``tau-frozen``) until that git tag exists, then the
  value and fit record read from ``frozen/tau.json`` AS COMMITTED at the tagged commit (not the
  working tree); the rule itself (F1-max on D_fit, plateau midpoint, II.4.1 / C.17) with the
  candidate threshold set and the plateau rule of ``rafr.validity.tau``;
* score direction and range, what the score is read from, the batch cap and the inference batch
  size, the text budget and truncation;
* the determinism guarantee;
* the pod: the fixed image and GPU (``models.verifiers.formal_pod``), and the versions of the
  probe's cuda run and -- once step 6 has run on the pod and ``sync`` has pulled its meta
  (``data/step06/pulled/meta/<config_id>.json``) -- of the step 6 scoring run.

``generate.py`` next to it writes the older registry candidates' cards; those candidates are not
on the formal scoring path.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from rafr.core import paths
from rafr.validity.tau import PLATEAU_TOL
from rafr.verifiers import factory, formal_pod
from rafr.verifiers.candidates import probe
from rafr.verifiers.candidates.base import DEFAULT_BATCH_LIMIT

DEFAULT_OUT_DIR = Path(__file__).resolve().parent
TAU_TAG = "tau-frozen"
TAU_POINTER = "frozen/tau.json"
VERSION_KEYS = ("python", "torch", "cuda", "transformers", "tokenizers", "huggingface_hub",
                "accelerate", "gpu", "driver")  # fmt: skip


def step06_meta(config_id: str) -> dict[str, Any] | None:
    """The step 6 scoring run's meta for ``config_id`` once ``env/step06_gpu.py sync`` has
    pulled it, else None."""
    path = paths.DATA / "step06" / "pulled" / "meta" / f"{config_id}.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None


def tag_commit(tag: str) -> str | None:
    """The commit a git tag points at, or None when it does not exist (yet)."""
    try:
        out = subprocess.run(  # noqa: S603 - fixed argv
            ["git", "rev-list", "-n", "1", tag],
            cwd=paths.ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
    except OSError:
        return None
    if out.returncode != 0:
        return None
    return out.stdout.strip() or None


def tau_payload(commit: str) -> dict[str, Any] | None:
    """``frozen/tau.json`` as committed at ``commit`` -- the tagged commit, never the working
    tree -- or None when that commit has no such file."""
    try:
        out = subprocess.run(  # noqa: S603 - fixed argv
            ["git", "show", f"{commit}:{TAU_POINTER}"],
            cwd=paths.ROOT,
            capture_output=True,
            text=True,
            encoding="utf-8",
            check=False,
        )
    except OSError:
        return None
    if out.returncode != 0:
        return None
    return json.loads(out.stdout)


def fit_for(config_id: str, payload: dict[str, Any] | None) -> dict[str, Any] | None:
    """The fit record of ``config_id`` (matched on ``verifier_id``) in a tau.json payload."""
    for fit in ((payload or {}).get("fits") or {}).values():
        if fit.get("verifier_id") == config_id:
            return fit
    return None


def tau_rule_lines() -> list[str]:
    """How tau_i is determined (II.4.1, C.17), as ``rafr.validity.tau`` implements it."""
    return [
        "How tau_i is determined (II.4.1, C.17; `rafr.validity.tau.fit_tau`):",
        "",
        "* fitted on D_fit against y*; the judgment is `y = 1[s >= tau_i]`;",
        "* candidate thresholds: the distinct D_fit scores in ascending order, the midpoint of",
        "  every adjacent pair of them, plus the two ends -- the smallest distinct score (every",
        "  state flagged) and the largest plus half the last gap (no state flagged); a single",
        "  distinct score is the only candidate;",
        "* criterion: F1 over the candidates. The candidates whose F1 lies within "
        f"{PLATEAU_TOL:g} of the",
        "  maximum, taken in ascending order, form runs of adjacent candidates; each maximal run",
        "  is one plateau. tau_i is the midpoint of the widest plateau (width = its threshold",
        "  span; on a tie, the lowest); a one-candidate plateau gives that candidate itself;",
        "* F2, PR-AUC, precision, recall and balanced accuracy are reported alongside and take",
        "  no part in the choice.",
        "",
        "Not applied here: `score()` returns the raw score.",
    ]


def tau_section(config_id: str) -> list[str]:
    out = ["## tau_i", ""]
    tau_commit = tag_commit(TAU_TAG)
    if not tau_commit:
        out += [f"**Pending step 7 (`{TAU_TAG}`).** The value is not fixed yet.", ""]
        out += tau_rule_lines()
        out += [
            "",
            f"Step 7 writes the fitted value to `{TAU_POINTER}` and the git tag `{TAU_TAG}` "
            "freezes it.",
        ]
        return out
    fit = fit_for(config_id, tau_payload(tau_commit))
    out.append(f"Frozen at step 7: `{TAU_POINTER}` at git tag `{TAU_TAG}` (commit `{tau_commit}`).")
    out.append("")
    if fit is None:
        out += [f"That commit's `{TAU_POINTER}` holds no fit for `{config_id}`.", ""]
        out += tau_rule_lines()
        return out
    out += [
        "| field | value |",
        "|---|---|",
        f"| tau_i | {fit['tau']:.5g} (full precision `{fit['tau']!r}`) |",
        f"| fitted on | D_fit, {fit['n']} states, {fit['n_positive']} of them E-UNSAFE |",
        f"| criterion | `{fit['criterion']}` |",
        f"| F1 at tau_i (the maximum) | {fit['f1_max']:.5g} |",
        f"| candidate thresholds | {fit['n_candidates']} |",
        f"| plateaus attaining the maximum | {fit['n_plateaus']} |",
        f"| widest plateau [lo, hi] | [{fit['plateau_lo']!r}, {fit['plateau_hi']!r}] |",
        "",
    ]
    out += tau_rule_lines()
    return out


def read_from(cfg: probe.ProbeConfig, meta: dict[str, Any]) -> str:
    m = cfg.model
    if not m.is_lm:
        return (
            f"softmax over the classification head, class {meta.get('toxic_index')!r} = 'toxic' "
            "resolved by name through config.id2label (P(toxic))"
        )
    v = meta.get("verdict") or {}
    toxic, safe = v.get("toxic") or [None, None], v.get("safe") or [None, None]
    text = (
        f"one forward pass; logits at the verdict position (after the fixed prefix "
        f"{v.get('prefix_text')!r}, token ids {v.get('prefix_ids')}); softmax over the two "
        f"verdict tokens only -- toxic {toxic[1]!r} (id {toxic[0]}), safe {safe[1]!r} "
        f"(id {safe[0]}); score = P(toxic)"
    )
    if cfg.aggregates:
        subs = ", ".join(c.sub_id for c in cfg.calls)
        text += f"; one call per policy ({subs}), score = the larger, both sub-scores kept"
    return text


def render_card(
    config_id: str,
    *,
    records_dirs: Sequence[Path] | None = None,
    prompt_dir: Path | None = None,
    models_cfg: dict[str, Any] | None = None,
) -> str:
    if models_cfg is None:
        from rafr.core.config import load_config

        models_cfg = load_config("models")
    pod = formal_pod.expected(models_cfg)
    cfg = factory.config_for(config_id, prompt_dir)
    m = cfg.model
    # the card describes the formal verifier: formal scores come from a cuda record only
    # (D-2026-09-24c point 6)
    rec = factory.find_probe_record(m.model_id, records_dirs, device_kind="cuda")
    info = rec.config_meta(config_id)
    meta, env = rec.meta, rec.environment
    out: list[str] = []
    add = out.append
    add(f"# verifier_card: {config_id}")
    add("")
    add("Generated by `python -m rafr.verifiers.cards.factory_card` from the step 2 probe's")
    add(f"records (`{rec.records_dir.as_posix()}`) and the probe code the scoring factory")
    add("imports. Do not edit by hand. Protocol App. A.2.")
    add("")
    add("## Model identity")
    add("")
    add("| field | value |")
    add("|---|---|")
    add(f"| config_id (= verifier_id in the scores table) | `{config_id}` |")
    add(f"| model | `{m.model_id}` -- {m.vendor}, {m.base_model}, {m.params_b} B parameters |")
    add(f"| policy variant | {cfg.variant or 'n/a (encoder)'} |")
    add(f"| HF repo | `{m.repo_id}` |")
    add(f"| commit hash | `{rec.revision}` |")
    add(f"| model_version (scores table) | `{m.repo_id}@{rec.revision}` |")
    add(f"| weights dtype | {m.dtype} |")
    add(f"| license / gated | {m.license} / {'yes' if m.gated else 'no'} |")
    add("| scoring code | `rafr.verifiers.factory:make_verifier` (the probe's own path) |")
    add("")
    add("## Weight files (sha256)")
    add("")
    add("| file | bytes | sha256 |")
    add("|---|---|---|")
    for w in rec.manifest.get("weight_files") or []:
        add(f"| `{w.get('path')}` | {w.get('bytes')} | `{w.get('sha256')}` |")
    add("")
    add("## Prompt")
    add("")
    files = factory.prompt_files_for(cfg)
    if files:
        hashes = probe.prompt_file_hashes(prompt_dir)
        add("| prompt file (`configs/prompts/`) | sha256 (LF-normalised) |")
        add("|---|---|")
        for name in files:
            add(f"| `{name}` | `{hashes[name]}` |")
        add("")
    elif m.is_lm:
        add("No prompt file: the chat template's own default categories, unchanged.")
        add("")
    if m.is_lm:
        add(f"Prompt hash (every scores-table row): `{info.get('prompt_hash')}`.")
        add("Rendered by the model's own chat template at the pinned revision "
            f"(template source: {meta.get('template_source')}); the state is the one user "
            "message; locked at stop point A (`verifier_probe_prompts.lock.json`).")  # fmt: skip
        if meta.get("categories_mechanism"):
            add(f"Llama Guard categories applied via: {meta.get('categories_mechanism')}.")
    else:
        add("None: the encoder reads the text itself.")
    add("")
    add("## Score")
    add("")
    add("| field | value |")
    add("|---|---|")
    add("| range | [0, 1] |")
    add("| direction | higher = more toxic |")
    add(f"| read from | {read_from(cfg, meta)} |")
    add(f"| batch cap per `score()` call | {DEFAULT_BATCH_LIMIT} |")
    add(f"| inference batch size | {meta.get('batch_size')} |")
    if m.is_lm:
        add(
            f"| text budget | {meta.get('text_max_tokens')} tokens of this model's tokenizer; the "
            "text is cut before it is templated, never the template |"
        )
    else:
        add(f"| truncation | {meta.get('truncation')} |")
    add("")
    add("Scales are not comparable between V1 and V2 and nothing downstream subtracts one")
    add("from the other (II.7.1, C.4).")
    add("")
    out.extend(tau_section(config_id))
    add("")
    add("## Determinism")
    add("")
    add("Same `(model_version, prompt hash, text)` gives the same score bit for bit:")
    add("")
    add("* `model.eval()` and `torch.no_grad()`; nothing is sampled on the scoring path")
    add("  (greedy decoding happens only in the preflight that settles the verdict tokens);")
    if m.is_lm:
        fixed = meta.get("fixed_shape")
        add(f"* attention implementation `{meta.get('attn_implementation')}`, recorded per model;")
        shape = (
            "every batch padded to a fixed number of rows and to one fixed length per call "
            "type, so a score cannot depend on its neighbours;"
            if fixed
            else "DYNAMIC padding (the probe ran so; not bit-reproducible across batchings);"
        )
        add("* left padding with position ids rebuilt from the mask; " + shape)
    else:
        add("* eager attention; every batch padded to max_length and to a fixed number of rows;")
    add("* TF32 off, cuDNN deterministic and not benchmarking,")
    add("  `torch.use_deterministic_algorithms(True, warn_only=True)`, `CUBLAS_WORKSPACE_CONFIG`;")
    add("* on every load the two invented direction-check sentences are re-scored and must equal")
    add("  the probe's recorded scores exactly, or the factory refuses to score.")
    add("")
    add("## Pod and versions")
    add("")
    add(f"Every formal score comes from one fixed pod (`models.verifiers.formal_pod`; "
        f"D-2026-09-24c): image `{pod['image']}`, GPU {pod['gpu']}. Checked over ssh and again "
        "on the pod before any model is loaded: " + ", ".join(
            f"{k} {v}" for k, v in pod["expect"].items()) + ".")  # fmt: skip
    add("")
    run6 = step06_meta(config_id)
    env6 = (run6 or {}).get("environment") or {}
    add("| | the probe's cuda run | the step 6 scoring run |")
    add("|---|---|---|")
    add(f"| git HEAD | `{env.get('git_head')}` | `{(run6 or {}).get('git_head')}` |")
    for key in VERSION_KEYS:
        add(f"| {key} | {env.get(key)} | {env6.get(key) if run6 else 'not run yet'} |")
    if run6:
        n = run6.get("numeric_check") or {}
        add("")
        add(f"Step 6 run finished {run6.get('finished_at')}; the 203-state check before it: "
            f"{n.get('n_different')} of {n.get('n')} differ (max |diff| {n.get('max_abs_diff')})."
            )  # fmt: skip
    add("")
    return "\n".join(out).rstrip() + "\n"


def write_card(
    config_id: str,
    out_dir: Path = DEFAULT_OUT_DIR,
    *,
    records_dirs: Sequence[Path] | None = None,
    prompt_dir: Path | None = None,
) -> Path:
    path = Path(out_dir) / f"{config_id}.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        render_card(config_id, records_dirs=records_dirs, prompt_dir=prompt_dir),
        encoding="utf-8",
        newline="\n",
    )
    return path


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--config-id", action="append", default=[], help="repeatable")
    ap.add_argument("--from-config", action="store_true", help="V1 and V2 of configs/models.yaml")
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    args = ap.parse_args(argv)
    from rafr.core.config import load_config

    models = load_config("models")
    ids = list(args.config_id)
    if args.from_config:
        ids += list(factory.configured_ids(models))
    if not ids:
        raise SystemExit("give --config-id (repeatable) or --from-config")
    for cid in ids:
        path = write_card(cid, args.out_dir, records_dirs=factory.records_dirs_from(models))
        print(f"wrote {path}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
