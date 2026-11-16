"""Pod side of the math step 2 probe. Never on the laptop.

    python scripts/gpu/math_vprobe_remote.py fetch     --base B --candidate C [--no-weights]
    python scripts/gpu/math_vprobe_remote.py precheck  --base B --candidate C
    python scripts/gpu/math_vprobe_remote.py preflight --base B --candidate C
    python scripts/gpu/math_vprobe_remote.py score     --base B --candidate C

Runs in the venv that the RunPod setup script (not released) creates inside ``<base>/rafr`` (``git
archive`` of one commit), and is started by the RunPod driver (not released) on the laptop.
Candidates and every setting come from ``configs/math/step02_vprobe.json`` as committed; a candidate
whose
``license_reviewed`` is still false is refused before anything is downloaded or loaded.

* fetch: the pinned revision only, the configured file patterns only, no token sent. With the
  weights it checks every LFS file against the Hub's sha256 and writes the
  ``rafr_model_manifest.json`` the math adapters require of a local checkpoint.
* precheck: before any weight is fetched, the candidate's own config (shrunk to 2 layers; for a
  multimodal config such as Gemma 4 its text model, rebuilt), tokenizer and code with RANDOM
  weights, through our adapter (``build_verifier``): it loads, scores the toy items, twice bit
  for bit, and a judge scores each item alone as in a batch.
* preflight: the real weights; the verifier_card with the reviewed source (vendor, base,
  licence, attribution notes); for a judge the five most likely tokens at
  the verdict position for the toy items (greedy decoding must give the configured verdict
  token) and each toy item scored alone as in a batch; the direction check (a wrong toy answer
  scores higher than the right one); every probe input tokenized and length-checked without a
  forward pass. Any failed check exits non-zero (PREFLIGHT_EXIT) and nothing is scored.
* score: the probe inputs in their order, rows written as ``pipelines/math_pipeline.py`` writes
  them (Core ``Score`` + one detail row); resumes after the last complete row.

The probe inputs carry state_id and public text; nothing here reads a label or a reference.
"""

from __future__ import annotations

import argparse
import fnmatch
import hashlib
import json
import math
import os
import platform
import shutil
import subprocess
import sys
import tempfile
import time
import traceback
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from rafr.contracts.schemas import Score  # noqa: E402
from rafr.domains.math.models import (  # noqa: E402
    VerifierSpec,
    build_verifier,
    fill_legacy_config_defaults,
    split_solution_steps,
)
from rafr.logging.writer import JsonlWriter, read_jsonl  # noqa: E402

CONFIG = ROOT / "configs" / "math" / "step02_vprobe.json"
MANIFEST = "rafr_model_manifest.json"
WEIGHT_SUFFIXES = (".safetensors", ".bin")
#: Invented items (not data): the same problem with a right and a wrong final answer.
TOY = [
    ("What is 2 + 3?", "We add the two numbers.\n\n2 + 3 = 5.\n\nThe answer is \\boxed{5}."),
    ("What is 2 + 3?", "We add the two numbers.\n\n2 + 3 = 6.\n\nThe answer is \\boxed{6}."),
    ("Compute 7 times 8.", "Multiply.\n\n7 \\times 8 = 56.\n\nThe answer is \\boxed{56}."),
    ("Compute 7 times 8.", "Multiply.\n\n7 \\times 8 = 54.\n\nThe answer is \\boxed{54}."),
]
TOY_PAIRS = [(0, 1), (2, 3)]  # (right, wrong)
#: Tiny dimensions for the precheck; only the keys a config has are changed.
SHRINK = {
    "num_hidden_layers": 2,
    "hidden_size": 64,
    "intermediate_size": 128,
    "num_attention_heads": 4,
    "num_key_value_heads": 2,
    "max_window_layers": 2,
    # DeepSeek-V2 (multi-head latent attention, mixture of experts)
    "kv_lora_rank": 16,
    "q_lora_rank": 16,  # None (V2-Lite) stays None
    "qk_nope_head_dim": 16,
    "qk_rope_head_dim": 8,
    "v_head_dim": 16,
    "n_routed_experts": 4,
    "num_experts_per_tok": 2,
    "moe_intermediate_size": 32,
    "n_shared_experts": 1,
    "first_k_dense_replace": 1,
    "n_group": 1,
    "topk_group": 1,
}
#: Composite (multimodal) configs keep the language model in ``text_config``. Gemma 4 derives
#: each layer's attention sizes from ``layer_types`` when the config is built, so the shrunk
#: config is rebuilt from its dict; a key changes only where the text config sets it (non-zero).
TEXT_SHRINK = {
    **{key: SHRINK[key] for key in ("num_hidden_layers", "hidden_size", "intermediate_size",
                                    "num_attention_heads", "num_key_value_heads")},
    "head_dim": 16,
    "hidden_size_per_layer_input": 16,
    "num_kv_shared_layers": 0,
    "num_experts": 4,
    "top_k_experts": 2,
    "moe_intermediate_size": 32,
}  # fmt: skip
#: Gemma 4's full-attention layers override two sizes (``per_layer_config``); the config is
#: rebuilt from these arguments instead.
GLOBAL_LAYER_SHRINK = {"head_dim": ("global_head_dim", 32),
                       "num_key_value_heads": ("num_global_key_value_heads", 1)}  # fmt: skip
#: Sub-config fields that must equal the language model's hidden size (Gemma 4 12B's
#: encoder-free image projection).
PROJECTION_DIMS = ("mm_embed_dim", "output_proj_dims")


def utc_now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def die(msg: str, code: int = 3) -> None:
    print(f"ERROR: {msg}", file=sys.stderr, flush=True)
    raise SystemExit(code)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=1, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n"
    )


# ---------------------------------------------------------------------------------------
# configuration and layout
# ---------------------------------------------------------------------------------------


def load_candidate(candidate_id: str, config: Path | None = None) -> dict[str, Any]:
    config = Path(config or CONFIG)
    cfg = json.loads(config.read_text(encoding="utf-8"))
    for c in cfg["candidates"]:
        if c["candidate_id"] == candidate_id:
            if c["spec"]["settings"].get("license_reviewed") is not True:
                die(
                    f"{candidate_id}: license_reviewed is false in {config.name}; "
                    "the authors decide"
                )
            return c
    die(f"unknown candidate {candidate_id!r}")
    raise AssertionError  # unreachable


def layout(base: Path) -> dict[str, Path]:
    base = Path(base)
    out = base / "out"
    return {
        "models": base / "models",
        "tmp": base / "tmp",
        "inputs": base / "data" / "probe_inputs.jsonl",
        "scores": out / "scores",
        "details": out / "details",
        "meta": out / "meta",
        "precheck": out / "precheck",
        "preflight": out / "preflight",
    }


def source_review(candidate: dict[str, Any]) -> dict[str, Any]:
    """The reviewed source beside the adapter's own card, for the verifier_card (paper Sec. 5):
    vendor and base as paper Sec. 5 compares them, the licence, and attribution notes such as a
    vendor overlap
    (a note only; it removes no pair)."""
    spec = candidate["spec"]
    return {
        "vendor": spec["vendor"],
        "base_model": spec["base_model"],
        "license": candidate.get("license"),
        "attribution": candidate.get("attribution"),
    }


def spec_for(candidate: dict[str, Any], local_path: Path) -> VerifierSpec:
    spec = dict(candidate["spec"])
    spec["settings"] = {**spec["settings"], "local_path": str(local_path)}
    return VerifierSpec(**spec)


def manifest_files(root: Path) -> dict[str, str]:
    """sha256 of every file under ``root`` except the Hub's local cache and the manifest."""
    return {
        path.relative_to(root).as_posix(): sha256_file(path)
        for path in sorted(root.rglob("*"))
        if path.is_file() and path.name != MANIFEST and ".cache" not in path.relative_to(root).parts
    }


def write_manifest(root: Path, spec: VerifierSpec) -> dict[str, str]:
    files = manifest_files(root)
    write_json(
        root / MANIFEST,
        {
            "model_id": spec.model_id,
            "revision": spec.revision,
            "tokenizer_revision": spec.settings["tokenizer_revision"],
            "files": files,
        },
    )
    return files


def environment() -> dict[str, Any]:
    import importlib.metadata as md

    versions = {}
    for package in ("torch", "transformers", "tokenizers", "huggingface_hub", "accelerate"):
        try:
            versions[package] = md.version(package)
        except md.PackageNotFoundError:
            versions[package] = None
    gpu = driver = None
    try:
        import torch

        if torch.cuda.is_available():
            gpu = torch.cuda.get_device_name(0)
    except ImportError:
        pass
    try:
        proc = subprocess.run(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            check=False,
        )
    except OSError:  # no nvidia-smi: not a GPU host
        proc = None
    if proc is not None and proc.returncode == 0 and proc.stdout.strip():
        driver = proc.stdout.strip().splitlines()[0]
    return {
        "python": platform.python_version(),
        **versions,
        "gpu": gpu,
        "driver": driver,
        "git_head": os.environ.get("RAFR_GIT_HEAD"),
    }


# ---------------------------------------------------------------------------------------
# fetch
# ---------------------------------------------------------------------------------------


def expected_files(candidate: dict[str, Any], *, weights: bool) -> dict[str, Any]:
    """The pinned revision's files that the configured patterns select, with their Hub sizes
    and (for LFS files) sha256."""
    from huggingface_hub import HfApi

    spec = candidate["spec"]
    info = HfApi().model_info(
        spec["model_id"], revision=spec["revision"], files_metadata=True, token=False
    )
    if info.sha != spec["revision"]:
        die(f"{spec['model_id']}: the Hub resolved {info.sha}, not {spec['revision']}")
    patterns = candidate["download"]["allow_patterns"]
    out = {}
    for s in info.siblings:
        if not any(fnmatch.fnmatch(s.rfilename, p) for p in patterns):
            continue
        if not weights and s.rfilename.endswith(WEIGHT_SUFFIXES):
            continue
        out[s.rfilename] = {"size": s.size, "lfs_sha256": s.lfs.sha256 if s.lfs else None}
    return out


def fetch(base: Path, candidate_id: str, *, weights: bool) -> dict[str, Any]:
    from huggingface_hub import snapshot_download

    c = load_candidate(candidate_id)
    spec = c["spec"]
    paths = layout(base)
    dest = paths["models"] / candidate_id
    want = expected_files(c, weights=weights)
    patterns = [
        p for p in c["download"]["allow_patterns"] if weights or not p.endswith(WEIGHT_SUFFIXES)
    ]
    started = time.monotonic()
    snapshot_download(
        repo_id=spec["model_id"],
        revision=spec["revision"],
        allow_patterns=patterns,
        local_dir=dest,
        token=False,
    )
    elapsed = time.monotonic() - started
    have = {n: h for n, h in manifest_files(dest).items() if weights or n in want}
    missing = sorted(set(want) - set(have))
    if missing:
        die(f"{candidate_id}: not downloaded: {missing}")
    bad = []
    for name, meta in want.items():
        size = (dest / name).stat().st_size
        if meta["size"] is not None and size != meta["size"]:
            bad.append(f"{name}: size {size} != Hub {meta['size']}")
        if meta["lfs_sha256"] and have[name] != meta["lfs_sha256"]:
            bad.append(f"{name}: sha256 differs from the Hub's LFS sha256")
    if bad:
        die(f"{candidate_id}: " + "; ".join(bad))
    record = {
        "candidate_id": candidate_id,
        "model_id": spec["model_id"],
        "revision": spec["revision"],
        "weights": weights,
        "files": {n: {"sha256": have[n], **want[n]} for n in sorted(want)},
        "bytes": sum((dest / n).stat().st_size for n in want),
        "elapsed_s": round(elapsed, 1),
        "fetched_at": utc_now(),
    }
    if weights:
        if set(have) != set(want):
            die(f"{candidate_id}: unexpected files {sorted(set(have) - set(want))}")
        write_manifest(dest, VerifierSpec(**spec))
        record["manifest_sha256"] = sha256_file(dest / MANIFEST)
    write_json(paths["meta"] / f"{candidate_id}.fetch{'' if weights else '_small'}.json", record)
    print(f"{candidate_id}: {len(want)} files, {record['bytes'] / 1e9:.2f} GB in {elapsed:.0f} s")
    return record


# ---------------------------------------------------------------------------------------
# precheck: the candidate's own architecture, tokenizer and code with tiny random weights
# ---------------------------------------------------------------------------------------


def shrink(config: Any) -> tuple[Any, dict[str, Any]]:
    """The config with tiny dimensions, and what was changed."""
    if getattr(config, "text_config", None) is not None:
        return shrink_composite(config)
    changed = {}
    for key, value in SHRINK.items():
        if getattr(config, key, None) is not None:
            setattr(config, key, value)
            changed[key] = value
    if getattr(config, "model_type", None) == "deepseek_v2":
        config.num_key_value_heads = config.num_attention_heads  # one KV head per query head
        # derived at __init__ from the shrunk sizes (configuration_deepseek_v2.py)
        config.qk_head_dim = config.qk_nope_head_dim + config.qk_rope_head_dim
        config.head_dim = config.qk_rope_head_dim
        changed.update(
            num_key_value_heads=config.num_key_value_heads,
            qk_head_dim=config.qk_head_dim,
            head_dim=config.head_dim,
        )
    if hasattr(config, "head_dim") and getattr(config, "model_type", None) != "deepseek_v2":
        config.head_dim = config.hidden_size // config.num_attention_heads
        changed["head_dim"] = config.head_dim
    return config, changed


def shrink_composite(config: Any) -> tuple[Any, dict[str, Any]]:
    """A multimodal config: the text model shrunk, rebuilt so derived per-layer sizes follow."""
    data = config.to_dict()
    text = data["text_config"]
    changed = {}
    for key, value in TEXT_SHRINK.items():
        if text.get(key):
            text[key] = changed[key] = value
    if text.get("layer_types"):
        kinds = text["layer_types"]
        first = next((k for k in kinds if k != "full_attention"), "full_attention")
        text["layer_types"] = changed["layer_types"] = [first, "full_attention"]
    # derived from layer_types and these two arguments when the config is built
    overrides = next(iter((text.pop("per_layer_config", None) or {}).values()), {})
    for key, (argument, value) in GLOBAL_LAYER_SHRINK.items():
        if key in overrides:
            text[argument] = changed[argument] = value
    for name in ("vision_config", "audio_config"):
        sub = data.get(name) or {}
        for key in PROJECTION_DIMS:
            if sub.get(key):
                sub[key] = text["hidden_size"]
                changed[f"{name}.{key}"] = text["hidden_size"]
    return type(config).from_dict(data), changed


def build_tiny(src: Path, dst: Path, candidate: dict[str, Any]) -> dict[str, Any]:
    import torch
    import transformers as tf

    spec = candidate["spec"]
    trust = bool(spec["settings"].get("trust_remote_code"))
    config = tf.AutoConfig.from_pretrained(src, trust_remote_code=trust, local_files_only=True)
    # repository code written for transformers 4.x: the adapter's load fills the same defaults
    legacy = fill_legacy_config_defaults(config) if trust else {}
    config, changed = shrink(config)
    if spec["implementation"] == "qwen_prm":
        model = tf.AutoModel.from_config(config, trust_remote_code=True)
    else:
        model = tf.AutoModelForCausalLM.from_config(config, trust_remote_code=trust)
    model = model.to(getattr(torch, spec["settings"]["dtype"]))
    dst.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(dst)
    for f in src.iterdir():  # the real tokenizer and code files, byte for byte
        if f.is_file() and f.name != "config.json" and not f.name.endswith(".index.json"):
            if not f.name.endswith(WEIGHT_SUFFIXES):
                shutil.copy2(f, dst / f.name)
    return {"model_class": type(model).__name__, "shrunk": changed, "legacy_defaults": legacy}


def score_items(verifier: Any, form: str, items: list[tuple[str, str]]) -> list[dict[str, Any]]:
    if form == "prm":
        return [verifier.score(p, s) for p, s in items]
    return verifier.score_batch(items)


def precheck(base: Path, candidate_id: str) -> dict[str, Any]:
    c = load_candidate(candidate_id)
    paths = layout(base)
    src = paths["models"] / candidate_id
    form = c["spec"]["form"]
    record: dict[str, Any] = {"candidate_id": candidate_id, "environment": environment()}
    paths["tmp"].mkdir(parents=True, exist_ok=True)
    try:
        with tempfile.TemporaryDirectory(dir=paths["tmp"]) as tmp:
            dst = Path(tmp) / candidate_id
            record["tiny"] = build_tiny(src, dst, c)
            write_manifest(dst, VerifierSpec(**c["spec"]))
            verifier = build_verifier(spec_for(c, dst))
            first = score_items(verifier, form, TOY)
            second = score_items(verifier, form, TOY)
            a = [r["error_score"] for r in first]
            b = [r["error_score"] for r in second]
            record["scores"] = a
            record["finite_unit"] = all(math.isfinite(x) and 0 <= x <= 1 for x in a)
            record["repeat_identical"] = a == b
            if form != "prm":
                alone = [verifier.score(p, s)["error_score"] for p, s in TOY]
                record["alone_equals_batch"] = alone == a
            verifier.close()
        record["compatible"] = record["finite_unit"] and record["repeat_identical"]
        record["batch_invariant"] = record.get("alone_equals_batch", True)
    except Exception as exc:  # report every failure; the run stops on it
        record.update(compatible=False, error=f"{type(exc).__name__}: {exc}")
        record["traceback"] = traceback.format_exc()
    write_json(paths["precheck"] / f"{candidate_id}.json", record)
    print(json.dumps({k: record.get(k) for k in ("candidate_id", "compatible", "batch_invariant",
                                                    "error")}))  # fmt: skip
    if not record["compatible"]:
        raise SystemExit(4)
    if not record["batch_invariant"]:
        raise SystemExit(5)
    return record


# ---------------------------------------------------------------------------------------
# preflight: real weights
# ---------------------------------------------------------------------------------------


def input_lengths(verifier: Any, form: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Token length of every probe input as the adapter builds it; no forward pass."""
    lengths, errors = [], []
    for r in rows:
        try:
            if form == "prm":
                marker = "<extra_0>" if verifier.config.kind == "qwen_prm" else "ки"
                if marker in r["problem"] or marker in r["solution"]:
                    raise ValueError("literal scoring marker")
                steps = split_solution_steps(r["solution"])
                serialized = verifier._serialize(r["problem"], steps)  # noqa: SLF001
                ids = verifier._encode(serialized, add_special_tokens=True)  # noqa: SLF001
            else:
                ids = verifier._row(r["problem"], r["solution"])[1]  # noqa: SLF001
            lengths.append(len(ids))
        except Exception as exc:
            errors.append({"state_id": r["state_id"], "error": f"{type(exc).__name__}: {exc}"})
    limit = verifier.config.max_tokens if form == "prm" else verifier.config.max_input_tokens
    ordered = sorted(lengths)
    return {
        "n": len(rows),
        "limit": limit,
        "max": ordered[-1] if ordered else None,
        "p99": ordered[int(0.99 * (len(ordered) - 1))] if ordered else None,
        "mean": sum(ordered) / len(ordered) if ordered else None,
        "n_over_limit": sum(n > limit for n in lengths),
        "errors": errors,
    }


def verdict_tokens(verifier: Any, items: list[tuple[str, str]]) -> list[dict[str, Any]]:
    """The five most likely next tokens at the verdict position of each toy item."""
    torch = verifier._torch  # noqa: SLF001
    out = []
    for problem, solution in items:
        ids = verifier._row(problem, solution)[1]  # noqa: SLF001
        tensor = torch.tensor([ids], dtype=torch.long, device=verifier._input_device())  # noqa: SLF001
        with torch.inference_mode():
            logits = verifier._model(input_ids=tensor, use_cache=False).logits[0, -1].float()  # noqa: SLF001
        probs = torch.softmax(logits, dim=-1)
        top = torch.topk(probs, 5)
        yes, no = verifier._verdict_ids  # noqa: SLF001
        out.append(
            {
                "top5": [
                    [int(i), verifier._decode([int(i)]), float(p)]  # noqa: SLF001
                    for p, i in zip(top.values.tolist(), top.indices.tolist(), strict=True)
                ],
                "p_yes": float(probs[yes]),
                "p_no": float(probs[no]),
                "top1_is_verdict": int(top.indices[0]) in (yes, no),
            }
        )
    return out


# Exit codes of a failed preflight (step 5).
PREFLIGHT_EXIT = {"verdict": 6, "direction": 7, "input_length": 8, "batch_invariance": 9}


def preflight_failures(record: dict[str, Any], form: str) -> list[tuple[str, int]]:
    """Failed preflight checks, in order. The verdict check: greedy decoding at the verdict
    position gives one of the configured tokens for every toy item (2026-10-08: no
    manual stop when it agrees with the proposed " Yes" / " No"; stop and report when not)."""
    lengths = record["input_lengths"]
    failed = []
    if form != "prm" and not record.get("verdict_ok"):
        failed.append("verdict")
    if not record["direction_ok"]:
        failed.append("direction")
    if lengths["n_over_limit"] or lengths["errors"]:
        failed.append("input_length")
    if form != "prm" and not record.get("alone_equals_batch"):
        failed.append("batch_invariance")
    return [(name, PREFLIGHT_EXIT[name]) for name in failed]


def preflight(base: Path, candidate_id: str) -> dict[str, Any]:
    c = load_candidate(candidate_id)
    paths = layout(base)
    form = c["spec"]["form"]
    verifier = build_verifier(spec_for(c, paths["models"] / candidate_id))
    started = time.monotonic()
    verifier.load()
    record: dict[str, Any] = {
        "candidate_id": candidate_id,
        "environment": environment(),
        "load_s": round(time.monotonic() - started, 1),
    }
    toy = score_items(verifier, form, TOY)
    record["toy"] = [
        {k: r.get(k) for k in ("error_score", "step_scores", "p_no", "input_tokens")} for r in toy
    ]
    record["direction"] = [
        {"right": toy[i]["error_score"], "wrong": toy[j]["error_score"]} for i, j in TOY_PAIRS
    ]
    record["direction_ok"] = all(d["wrong"] > d["right"] for d in record["direction"])
    record["rendered_toy_input"] = toy[1]["serialized_input"]
    record["card"] = {**verifier.verifier_card(), "source": source_review(c)}
    if form != "prm":
        record["verdict"] = verdict_tokens(verifier, TOY)
        record["verdict_ok"] = all(v["top1_is_verdict"] for v in record["verdict"])
        # a score must not depend on its batch neighbours (step 6's 203-state check)
        alone = [verifier.score(p, s)["error_score"] for p, s in TOY]
        record["alone_equals_batch"] = alone == [r["error_score"] for r in toy]
    rows = list(read_jsonl(paths["inputs"]))
    record["input_lengths"] = input_lengths(verifier, form, rows)
    verifier.close()
    record["failures"] = [name for name, _ in preflight_failures(record, form)]
    write_json(paths["preflight"] / f"{candidate_id}.json", record)
    lengths = record["input_lengths"]
    print(
        json.dumps(
            {
                "candidate_id": candidate_id,
                "direction_ok": record["direction_ok"],
                "verdict_ok": record.get("verdict_ok"),
                "alone_equals_batch": record.get("alone_equals_batch"),
                "max_tokens": lengths["max"],
                "limit": lengths["limit"],
                "n_over_limit": lengths["n_over_limit"],
                "n_input_errors": len(lengths["errors"]),
                "failures": record["failures"],
            }
        )
    )
    failures = preflight_failures(record, form)
    if failures:  # the run stops here and reports; it does not score
        raise SystemExit(failures[0][1])
    return record


# ---------------------------------------------------------------------------------------
# score
# ---------------------------------------------------------------------------------------


def check_inputs(path: Path) -> str:
    recorded = path.with_name(path.name + ".sha256").read_text(encoding="utf-8").split()[0]
    if sha256_file(path) != recorded:
        die(f"{path} changed since its sha256 was written")
    return recorded


def resume_point(scores: Path, details: Path, ids: list[str]) -> int:
    done = [r["state_id"] for r in read_jsonl(scores)] if scores.is_file() else []
    if done != ids[: len(done)]:
        die(f"{scores}: the rows are not a prefix of the probe inputs; start a new table")
    kept = list(read_jsonl(details))[: len(done)] if details.is_file() else []
    if [r["state_id"] for r in kept] != done:
        die(f"{details}: fewer detail rows than score rows")
    details.unlink(missing_ok=True)
    with JsonlWriter(details) as writer:  # drop a detail row whose score row was not written
        writer.write_many(kept)
    return len(done)


def score(base: Path, candidate_id: str, *, every: int = 200) -> dict[str, Any]:
    c = load_candidate(candidate_id)
    paths = layout(base)
    form = c["spec"]["form"]
    inputs_sha = check_inputs(paths["inputs"])
    rows = list(read_jsonl(paths["inputs"]))
    ids = [r["state_id"] for r in rows]
    for key in ("scores", "details", "meta"):
        paths[key].mkdir(parents=True, exist_ok=True)
    table = paths["scores"] / f"{candidate_id}.jsonl"
    detail_file = paths["details"] / f"{candidate_id}.jsonl"
    start = resume_point(table, detail_file, ids)
    spec = spec_for(c, paths["models"] / candidate_id)
    verifier = build_verifier(spec)
    step = 1 if form == "prm" else verifier.config.batch_size
    started_at, t0 = utc_now(), time.monotonic()
    verifier.load()
    for begin in range(start, len(rows), step):
        chunk = rows[begin : begin + step]
        outputs = score_items(verifier, form, [(r["problem"], r["solution"]) for r in chunk])
        if len(outputs) != len(chunk):
            die("the verifier returned a different number of scores than inputs")
        # details first: resume_point drops a detail row whose score row is missing
        with JsonlWriter(detail_file) as writer:
            for r, detail in zip(chunk, outputs, strict=True):
                writer.write({"state_id": r["state_id"], "verifier_id": spec.model_id, **detail})
        with JsonlWriter(table) as writer:
            for r, detail in zip(chunk, outputs, strict=True):
                value = detail["error_score"]
                if isinstance(value, bool) or not math.isfinite(value) or not 0 <= value <= 1:
                    die(f"{r['state_id']}: score {value!r} outside [0, 1]")
                steps = detail.get("step_scores")
                writer.write(
                    Score(
                        state_id=r["state_id"],
                        verifier_id=spec.model_id,
                        model_version=spec.revision,
                        score=value,
                        sub_scores={str(j): s for j, s in enumerate(steps)} if steps else None,
                    )
                )
        done = begin + len(chunk)
        if done % every < step or done == len(rows):
            rate = (done - start) / max(time.monotonic() - t0, 1e-9)
            eta = (len(rows) - done) / rate if rate else float("nan")
            print(f"{candidate_id}: {done}/{len(rows)} ({rate:.2f}/s, ETA {eta / 60:.1f} min)",
                  flush=True)  # fmt: skip
    verifier.close()
    record = {
        "candidate_id": candidate_id,
        "model_id": spec.model_id,
        "revision": spec.revision,
        "n": len(rows),
        "resumed_from": start,
        "started_at": started_at,
        "finished_at": utc_now(),
        "elapsed_s": round(time.monotonic() - t0, 1),
        "inputs_sha256": inputs_sha,
        "table_sha256": sha256_file(table),
        "details_sha256": sha256_file(detail_file),
        "config_hash": next(iter(read_jsonl(detail_file)))["config_hash"],
        "environment": environment(),
    }
    write_json(paths["meta"] / f"{candidate_id}.score.json", record)
    print(json.dumps({k: record[k] for k in ("candidate_id", "n", "elapsed_s", "table_sha256")}))
    return record


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("command", choices=("fetch", "precheck", "preflight", "score"))
    ap.add_argument("--base", type=Path, required=True)
    ap.add_argument("--candidate", required=True)
    ap.add_argument("--no-weights", action="store_true", help="fetch: config, tokenizer, code")
    args = ap.parse_args(argv)
    os.environ.setdefault("HF_HUB_DISABLE_IMPLICIT_TOKEN", "1")
    if args.command == "fetch":
        fetch(args.base, args.candidate, weights=not args.no_weights)
    elif args.command == "precheck":
        precheck(args.base, args.candidate)
    elif args.command == "preflight":
        preflight(args.base, args.candidate)
    else:
        score(args.base, args.candidate)
    return 0


if __name__ == "__main__":
    sys.exit(main())
