"""Pod side of math step 2, round 2. Never on the laptop.

    python scripts/gpu/math_vprobe2_remote.py fetch     --base B --candidate C [--no-weights]
    python scripts/gpu/math_vprobe2_remote.py speed     --base B --candidate C
    python scripts/gpu/math_vprobe2_remote.py precheck  --base B --candidate C
    python scripts/gpu/math_vprobe2_remote.py preflight --base B --candidate C --stage {screen,full}
    python scripts/gpu/math_vprobe2_remote.py check203  --base B --candidate C
    python scripts/gpu/math_vprobe2_remote.py score     --base B --candidate C --stage {screen,full}

Candidates and settings come from ``configs/math/step02_round2.json`` as committed; a candidate
whose ``license_reviewed`` is false, or with a setting still ``TBD``, is refused before anything
is downloaded or loaded. Scoring is the generative variant (``generative.py``).

* fetch: as in round 1 (pinned revision, configured patterns, no token, LFS sha256 checked,
  ``rafr_model_manifest.json`` written).
* precheck (before the weights): (1) the candidate's own config shrunk to 2 layers (sliding
  window 16, so a window is crossed), its tokenizer and template, RANDOM weights, through
  ``build_verifier``'s adapter at a small shape: the toy items twice and each alone; token ids,
  verdict logits and scores must agree bit for bit (exit 4 / 5). (2) on a GPU, the full-size
  model with random weights at the configured fixed shape [B, L_in + L_gen + L_forced]: one
  batch runs to L_gen; the peak memory must leave the configured margin (31B / 32B
  candidates enter only if this fits; exit 13), and the decoding step time is recorded.
* preflight (real weights): the verifier_card with the reviewed source; the toy items scored
  in a batch, twice, and each alone (exit 9 / 12 when they differ); the greedy token at each
  verdict readout (exit 6 when not one of the two verdict tokens); the direction check (a wrong
  toy answer scores higher; exit 7); the step side's generated verdict count on the toy items
  (exit 11 when no step verdict at all); every input of the stage built and length-checked
  (exit 8); the peak memory.
* check203: the 203 states of ``check203.json`` scored again and compared with their screening
  rows: generated token ids, every verdict readout's logits, error_score. Any difference: exit
  10, nothing scored (generative variant). Precondition of ``score --stage full``.
* score: the stage's inputs in their order; Core ``Score`` rows plus one detail row each with
  the generated tokens, stop reason, truncation and readouts; resumes after the last complete
  batch.

Nothing here reads a label or a reference answer: the inputs carry state_id and public text.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import sys
import tempfile
import time
import traceback
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts" / "gpu"))

import math_vprobe_remote as r1  # noqa: E402 - round 1 helpers, unchanged
from rafr.contracts.schemas import Score  # noqa: E402
from rafr.domains.math.generative import (  # noqa: E402
    GENERATIVE_ADAPTERS,
    GROUPED_DECODE_ATTENTION,
    register_attention,
)
from rafr.domains.math.models import VerifierSpec, build_verifier  # noqa: E402
from rafr.logging.writer import JsonlWriter, read_jsonl  # noqa: E402

CONFIG = ROOT / "configs" / "math" / "step02_round2.json"
MANIFEST = r1.MANIFEST
TOY = r1.TOY
TOY_PAIRS = r1.TOY_PAIRS
STAGES = ("screen", "full")
# Exit codes.
EXIT = {
    "precheck_incompatible": 4,
    "precheck_batch_invariance": 5,
    "verdict": 6,
    "direction": 7,
    "input_length": 8,
    "batch_invariance": 9,
    "check203": 10,
    "step_format": 11,
    "repeat": 12,
    "fit": 13,
    "padding": 14,
}
#: The tiny precheck shape: B rows, L_gen tokens (random weights never stop by themselves).
TINY = {"batch_size": 3, "max_new_tokens": 24, "sliding_window": 16}
SUBCONFIG_SHRINK = {
    "num_hidden_layers": 1,
    "hidden_size": 64,
    "intermediate_size": 128,
    "num_attention_heads": 4,
    "num_key_value_heads": 2,
}

die = r1.die
utc_now = r1.utc_now
sha256_file = r1.sha256_file
write_json = r1.write_json
environment = r1.environment
manifest_files = r1.manifest_files


# ---------------------------------------------------------------------------------------
# configuration and layout
# ---------------------------------------------------------------------------------------


def load_cfg(config: Path | None = None) -> dict[str, Any]:
    return json.loads(Path(config or CONFIG).read_text(encoding="utf-8"))


def load_candidate(candidate_id: str, config: Path | None = None) -> dict[str, Any]:
    cfg = load_cfg(config)
    for c in cfg["candidates"]:
        if c["candidate_id"] == candidate_id:
            settings = c["spec"]["settings"]
            if settings.get("license_reviewed") is not True:
                die(f"{candidate_id}: license_reviewed is false; the authors decide")
            pending = sorted(k for k, v in settings.items() if v == "TBD")
            if pending:
                die(f"{candidate_id}: still TBD (the authors decide): {', '.join(pending)}")
            return c
    die(f"unknown candidate {candidate_id!r}")
    raise AssertionError  # unreachable


def layout(base: Path) -> dict[str, Path]:
    base = Path(base)
    out = base / "out"
    return {
        "models": base / "models",
        "tmp": base / "tmp",
        "probe_inputs": base / "data" / "probe_inputs.jsonl",
        "screen_inputs": base / "data" / "screen_inputs.jsonl",
        "full_inputs": base / "data" / "full_inputs.jsonl",
        "check203": base / "data" / "check203.json",
        "out": out,
        "precheck": out / "precheck",
        "preflight": out / "preflight",
        "check203_out": out / "check203",
        "meta": out / "meta",
    }


def stage_dirs(paths: dict[str, Path], stage: str) -> dict[str, Path]:
    return {key: paths["out"] / stage / key for key in ("scores", "details", "meta")}


def write_manifest(root: Path, spec: dict[str, Any]) -> dict[str, str]:
    files = manifest_files(root)
    write_json(
        root / MANIFEST,
        {
            "model_id": spec["model_id"],
            "revision": spec["revision"],
            "tokenizer_revision": spec["settings"]["tokenizer_revision"],
            "files": files,
        },
    )
    return files


def spec_for(candidate: dict[str, Any], local_path: Path, **overrides: Any) -> VerifierSpec:
    spec = dict(candidate["spec"])
    spec["settings"] = {**spec["settings"], "local_path": str(local_path), **overrides}
    return VerifierSpec(**spec)


def source_review(candidate: dict[str, Any]) -> dict[str, Any]:
    """The reviewed source for the verifier_card (paper Sec. 5): vendor, base, lineage sets,
    licence."""
    spec = candidate["spec"]
    return {
        "vendor": spec["vendor"],
        "base_model": spec["base_model"],
        "lineage": spec.get("lineage"),
        "license": candidate.get("license"),
        "format_source": candidate.get("format_source"),
        "loading_note": candidate.get("loading_note"),
        "literature_flags": candidate.get("literature_flags"),
    }


def read_rows(path: Path) -> list[dict[str, Any]]:
    recorded = path.with_name(path.name + ".sha256").read_text(encoding="utf-8").split()[0]
    if sha256_file(path) != recorded:
        die(f"{path} changed since its sha256 was written")
    return list(read_jsonl(path))


def stage_rows(paths: dict[str, Path], stage: str) -> list[dict[str, Any]]:
    """Screening: the 300. Full: the other 1,200 D_fit states and the stratified pool sample
    (2026-10-11); the screening rows are reused once the 203-state check passes."""
    return read_rows(paths["screen_inputs" if stage == "screen" else "full_inputs"])


# ---------------------------------------------------------------------------------------
# fetch (round 1's, with this configuration)
# ---------------------------------------------------------------------------------------


def fetch(base: Path, candidate_id: str, *, weights: bool) -> dict[str, Any]:
    from huggingface_hub import snapshot_download

    c = load_candidate(candidate_id)
    spec = c["spec"]
    paths = layout(base)
    dest = paths["models"] / candidate_id
    want = r1.expected_files(c, weights=weights)
    patterns = [
        p for p in c["download"]["allow_patterns"] if weights or not p.endswith(r1.WEIGHT_SUFFIXES)
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
        write_manifest(dest, spec)
        record["manifest_sha256"] = sha256_file(dest / MANIFEST)
    write_json(paths["meta"] / f"{candidate_id}.fetch{'' if weights else '_small'}.json", record)
    print(f"{candidate_id}: {len(want)} files, {record['bytes'] / 1e9:.2f} GB in {elapsed:.0f} s")
    return record


# ---------------------------------------------------------------------------------------
# comparison of two scorings of the same items
# ---------------------------------------------------------------------------------------


def fingerprint(result: dict[str, Any]) -> dict[str, Any]:
    """What the design's generative variant compares: tokens, verdict logits, score."""
    return {
        "generated_token_ids": result["generated_token_ids"],
        "forced_token_ids": result["forced_token_ids"],
        "readout_logits": [r["logits"] for r in result["readouts"]],
        "error_score": result["error_score"],
    }


def first_divergence(a: list[int], b: list[int]) -> int | None:
    for i, (x, y) in enumerate(zip(a, b, strict=False)):
        if x != y:
            return i
    return None if len(a) == len(b) else min(len(a), len(b))


def compare(reference: list[dict[str, Any]], again: list[dict[str, Any]]) -> dict[str, Any]:
    """Per item: identical or not, the first diverging generated position, the score gap."""
    differing = []
    max_gap = 0.0
    for i, (a, b) in enumerate(zip(reference, again, strict=True)):
        fa, fb = fingerprint(a), fingerprint(b)
        gap = abs(fa["error_score"] - fb["error_score"])
        max_gap = max(max_gap, gap)
        if fa != fb:
            differing.append(
                {
                    "index": i,
                    "first_divergent_token": first_divergence(
                        fa["generated_token_ids"], fb["generated_token_ids"]
                    ),
                    "score_gap": gap,
                }
            )
    return {
        "n": len(reference),
        "n_differing": len(differing),
        "max_score_gap": max_gap,
        "differing": differing,
    }


# ---------------------------------------------------------------------------------------
# precheck: tiny random weights (determinism), full-size random weights (fit)
# ---------------------------------------------------------------------------------------


def shrink(config: Any) -> tuple[Any, dict[str, Any]]:
    """Round 1's shrink, then a small sliding window and tiny vision / audio towers."""
    config, changed = r1.shrink(config)
    text = config.get_text_config() if hasattr(config, "get_text_config") else config
    kinds = getattr(text, "layer_types", None)
    if kinds and len(kinds) != text.num_hidden_layers:  # one per layer (Qwen2, Olmo 3)
        first = next((k for k in kinds if k != "full_attention"), "full_attention")
        text.layer_types = changed["layer_types"] = [first, "full_attention"]
    if getattr(text, "sliding_window", None):
        text.sliding_window = changed["sliding_window"] = TINY["sliding_window"]
    for name in ("vision_config", "audio_config"):
        sub = getattr(config, name, None)
        if sub is None:
            continue
        for key, value in SUBCONFIG_SHRINK.items():
            if getattr(sub, key, None):
                setattr(sub, key, value)
                changed[f"{name}.{key}"] = value
        if getattr(sub, "head_dim", None):
            sub.head_dim = changed[f"{name}.head_dim"] = 16
    return config, changed


def model_class(candidate: dict[str, Any]) -> Any:
    import transformers as tf

    return getattr(tf, candidate["spec"]["settings"]["model_class"])


def build_tiny(src: Path, dst: Path, candidate: dict[str, Any]) -> dict[str, Any]:
    import torch
    import transformers as tf

    config = tf.AutoConfig.from_pretrained(src, local_files_only=True)
    config, changed = shrink(config)
    torch.manual_seed(20261011)
    model = model_class(candidate).from_config(config)
    model = model.to(getattr(torch, candidate["spec"]["settings"]["dtype"]))
    dst.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(dst)
    for f in src.iterdir():  # the real tokenizer and template files, byte for byte
        if (
            f.is_file()
            and f.name not in ("config.json", MANIFEST)
            and not f.name.endswith((".index.json", *r1.WEIGHT_SUFFIXES))
        ):
            shutil.copy2(f, dst / f.name)
    return {"model_class": type(model).__name__, "shrunk": changed}


def toy_shape(candidate: dict[str, Any], verifier_factory) -> int:
    """L_in for the tiny run: the longest toy input under this tokenizer plus one, rounded up to
    16, so every toy row is left padded, as every scored row is (L_in exceeds every input). A
    toy row exactly L_in long, scored alone among copies of itself, made a batch without any
    padding, where transformers drops the mask in the prefill: another kernel, so the row
    differed from its batched copy (deepcritic_7b, 2026-10-11)."""
    longest = max(len(verifier_factory()._row(p, s)[1]) for p, s in TOY)  # noqa: SLF001
    return 16 * math.ceil((longest + 1) / 16)


#: Extra left padding of the padding-transparency check; its logit tolerance (float32).
PAD_EXTRA, PAD_TOLERANCE = 37, 1e-4


def padding_check(snug: list[dict[str, Any]], wide: list[dict[str, Any]]) -> dict[str, Any]:
    """Left padding must be invisible to a row: the same generated and forced tokens and
    verdict logits within ``PAD_TOLERANCE`` (the shapes differ, so not bit for bit)."""
    same_tokens = all(
        a["generated_token_ids"] == b["generated_token_ids"]
        and a["forced_token_ids"] == b["forced_token_ids"]
        for a, b in zip(snug, wide, strict=True)
    )
    gap = max(
        (
            abs(x - y)
            for a, b in zip(snug, wide, strict=True)
            for ra, rb in zip(a["readouts"], b["readouts"], strict=True)
            for x, y in zip(ra["logits"], rb["logits"], strict=True)
        ),
        default=0.0,
    )
    return {
        "extra_padding": PAD_EXTRA,
        "same_tokens": same_tokens,
        "max_logit_gap": gap,
        "tolerance": PAD_TOLERANCE,
        "transparent": same_tokens and gap <= PAD_TOLERANCE,
    }


def tiny_checks(verifier: Any) -> dict[str, Any]:
    batch = verifier.score_batch(TOY)
    again = verifier.score_batch(TOY)
    alone = [verifier.score(p, s) for p, s in TOY]
    return {
        "scores": [r["error_score"] for r in batch],
        "stop_reasons": [r["stop_reason"] for r in batch],
        "finite_unit": all(
            math.isfinite(r["error_score"]) and 0 <= r["error_score"] <= 1 for r in batch
        ),
        "repeat": compare(batch, again),
        "alone_vs_batch": compare(batch, alone),
    }


def fit_check(c: dict[str, Any], src: Path) -> dict[str, Any]:
    """The full-size model with random weights at the configured fixed shape, one batch to L_gen."""
    import torch
    import transformers as tf

    settings = c["spec"]["settings"]
    if not torch.cuda.is_available():
        return {"skipped": "no GPU"}
    if settings["attn_implementation"] == GROUPED_DECODE_ATTENTION:
        register_attention()  # before the model is built with it
    config = tf.AutoConfig.from_pretrained(src, local_files_only=True)
    torch.cuda.reset_peak_memory_stats()
    started = time.monotonic()
    with torch.device("cuda"):
        torch.manual_seed(0)
        model = model_class(c).from_config(
            config,
            dtype=getattr(torch, settings["dtype"]),
            attn_implementation=settings["attn_implementation"],
        )
    model.eval()
    built = time.monotonic() - started
    tokenizer = tf.AutoTokenizer.from_pretrained(src, **settings.get("tokenizer_kwargs", {}))
    spec = spec_for(c, src)
    adapter = GENERATIVE_ADAPTERS[spec.implementation]
    cfg = adapter.config_class(
        **{**spec.settings, "model_id": spec.model_id, "revision": spec.revision}
    )
    verifier = adapter(cfg, tokenizer=tokenizer, model=model, torch_module=torch)
    verifier.load()
    torch.cuda.synchronize()
    started = time.monotonic()
    result = verifier.score_batch(TOY[:1])[0]
    torch.cuda.synchronize()
    elapsed = time.monotonic() - started
    steps = result["generated_tokens"] + len(result["forced_token_ids"])
    total = torch.cuda.get_device_properties(0).total_memory
    record = {
        "fixed_shape": result["fixed_shape"],
        "build_s": round(built, 1),
        "batch_s": round(elapsed, 1),
        "decode_steps": steps,
        "step_ms": round(1000 * elapsed / max(steps, 1), 2),
        "stop_reason": result["stop_reason"],
        "peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
        "peak_reserved_gib": torch.cuda.max_memory_reserved() / 2**30,
        "device_total_gib": total / 2**30,
    }
    margin = load_cfg()["pod"]["fit_margin_gib"]
    record["fits"] = record["peak_reserved_gib"] + margin <= record["device_total_gib"]
    verifier.close()
    del model
    torch.cuda.empty_cache()
    return record


def precheck(base: Path, candidate_id: str) -> dict[str, Any]:
    import torch

    c = load_candidate(candidate_id)
    paths = layout(base)
    src = paths["models"] / candidate_id
    record: dict[str, Any] = {"candidate_id": candidate_id, "environment": environment()}
    paths["tmp"].mkdir(parents=True, exist_ok=True)
    try:
        with tempfile.TemporaryDirectory(dir=paths["tmp"]) as tmp:
            dst = Path(tmp) / candidate_id
            record["tiny"] = build_tiny(src, dst, c)
            write_manifest(dst, c["spec"])
            device = "cuda" if torch.cuda.is_available() else "cpu"

            def make(length: int = 4096, **changes: Any) -> Any:
                return build_verifier(
                    spec_for(
                        c,
                        dst,
                        batch_size=TINY["batch_size"],
                        max_input_tokens=length,
                        max_new_tokens=TINY["max_new_tokens"],
                        device_map=device,
                        **changes,
                    )
                )

            import transformers as tf

            text = tf.AutoConfig.from_pretrained(dst, local_files_only=True)
            text = text.get_text_config() if hasattr(text, "get_text_config") else text
            room = text.max_position_embeddings - TINY["max_new_tokens"]
            probe = make(min(4096, room - c["spec"]["settings"]["max_forced_tokens"]))
            probe.load()
            length = toy_shape(c, lambda: probe)
            probe.close()
            verifier = make(length)
            record["tiny_shape"] = {"L_in": length, **TINY}
            record.update(tiny_checks(verifier))
            verifier.close()
            # padding transparency: the same rows with PAD_EXTRA more left padding generate the
            # same tokens (float32, so a rounding tie cannot flip a greedy choice)
            snug = make(length, dtype="float32")
            wide = make(length + PAD_EXTRA, dtype="float32")
            record["padding"] = padding_check(snug.score_batch(TOY), wide.score_batch(TOY))
            snug.close()
            wide.close()
        record["compatible"] = record["finite_unit"] and not record["repeat"]["n_differing"]
        record["batch_invariant"] = not record["alone_vs_batch"]["n_differing"]
        record["padding_transparent"] = record["padding"]["transparent"]
        if record["compatible"] and record["batch_invariant"] and record["padding_transparent"]:
            try:
                record["fit"] = fit_check(c, src)
            except torch.OutOfMemoryError as exc:  # the fixed shape does not fit: exit 13, not 4
                record["fit"] = {"fits": False, "error": f"{type(exc).__name__}: {exc}"}
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    except Exception as exc:  # report every failure; the run stops on it
        record.update(compatible=False, error=f"{type(exc).__name__}: {exc}")
        record["traceback"] = traceback.format_exc()
    write_json(paths["precheck"] / f"{candidate_id}.json", record)
    fit = record.get("fit") or {}
    print(
        json.dumps(
            {
                "candidate_id": candidate_id,
                "compatible": record.get("compatible"),
                "batch_invariant": record.get("batch_invariant"),
                "padding_transparent": record.get("padding_transparent"),
                "fits": fit.get("fits"),
                "peak_reserved_gib": fit.get("peak_reserved_gib"),
                "step_ms": fit.get("step_ms"),
                "error": record.get("error"),
            }
        )
    )
    if not record["compatible"]:
        raise SystemExit(EXIT["precheck_incompatible"])
    if not record["batch_invariant"]:
        raise SystemExit(EXIT["precheck_batch_invariance"])
    if not record["padding_transparent"]:
        raise SystemExit(EXIT["padding"])
    if fit.get("fits") is False:
        raise SystemExit(EXIT["fit"])
    return record


def speed(base: Path, candidate_id: str) -> dict[str, Any]:
    """Decoding speed alone (before any precheck): the precheck's
    full-size run with random weights at the configured fixed shape, to L_gen. No tiny checks,
    no real weights, nothing scored; the laptop compares the step time with the cost model."""
    import torch

    c = load_candidate(candidate_id)
    paths = layout(base)
    record: dict[str, Any] = {
        "candidate_id": candidate_id,
        "environment": environment(),
        "attn_implementation": c["spec"]["settings"]["attn_implementation"],
    }
    try:
        record["fit"] = fit_check(c, paths["models"] / candidate_id)
    except torch.OutOfMemoryError as exc:
        record["fit"] = {"fits": False, "error": f"{type(exc).__name__}: {exc}"}
    record["measured_at"] = utc_now()
    write_json(paths["out"] / "speed" / f"{candidate_id}.json", record)
    fit = record["fit"]
    print(
        json.dumps(
            {
                "candidate_id": candidate_id,
                "attn_implementation": record["attn_implementation"],
                "fits": fit.get("fits"),
                "peak_reserved_gib": fit.get("peak_reserved_gib"),
                "step_ms": fit.get("step_ms"),
                "decode_steps": fit.get("decode_steps"),
            }
        )
    )
    if fit.get("fits") is False:
        raise SystemExit(EXIT["fit"])
    return record


# ---------------------------------------------------------------------------------------
# preflight: real weights
# ---------------------------------------------------------------------------------------


def input_lengths(verifier: Any, rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Every input of the stage built as the adapter builds it; no forward pass."""
    lengths, errors = [], []
    for r in rows:
        try:
            lengths.append(len(verifier._row(r["problem"], r["solution"])[1]))  # noqa: SLF001
        except Exception as exc:
            errors.append({"state_id": r["state_id"], "error": f"{type(exc).__name__}: {exc}"})
    ordered = sorted(lengths)
    return {
        "n": len(rows),
        "L_in": verifier.config.max_input_tokens,
        "max": ordered[-1] if ordered else None,
        "p99": ordered[int(0.99 * (len(ordered) - 1))] if ordered else None,
        "errors": errors,
    }


def preflight_failures(record: dict[str, Any], granularity: str) -> list[str]:
    failed = []
    if not record["verdict_ok"]:
        failed.append("verdict")
    if not record["direction_ok"]:
        failed.append("direction")
    if record["input_lengths"]["errors"]:
        failed.append("input_length")
    if record["alone_vs_batch"]["n_differing"]:
        failed.append("batch_invariance")
    if granularity == "step" and not any(record["step_verdicts_generated"]):
        failed.append("step_format")
    if record["repeat"]["n_differing"]:
        failed.append("repeat")
    return failed


#: Preflight failures the authors may accept for a candidate (scored anyway and recorded, as round
#: 1's ds_llm_chat): the direction check only; every other failure stops.
ACCEPTABLE_FAILURES = ("direction",)


def accepted_failures(candidate: dict[str, Any]) -> dict[str, str]:
    """The candidate's ``accepted_preflight_failures`` (failure -> the authors' decision)."""
    accepted = candidate.get("accepted_preflight_failures") or {}
    wrong = sorted(set(accepted) - set(ACCEPTABLE_FAILURES))
    if wrong or not all(isinstance(v, str) and v.strip() for v in accepted.values()):
        die(
            f"{candidate['candidate_id']}: only {ACCEPTABLE_FAILURES} can be accepted, with reasons"
        )
    return dict(accepted)


def preflight(base: Path, candidate_id: str, stage: str) -> dict[str, Any]:
    c = load_candidate(candidate_id)
    paths = layout(base)
    verifier = build_verifier(spec_for(c, paths["models"] / candidate_id))
    record = preflight_record(c, verifier, stage_rows(paths, stage), stage=stage)
    write_json(paths["preflight"] / f"{candidate_id}.json", record)
    return preflight_verdict(record)


def preflight_record(
    c: dict[str, Any], verifier: Any, rows: list[dict[str, Any]], *, stage: str
) -> dict[str, Any]:
    """Load the real weights, run the toy checks, build every input of ``rows``; close."""
    import torch

    candidate_id = c["candidate_id"]
    started = time.monotonic()
    verifier.load()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    record: dict[str, Any] = {
        "candidate_id": candidate_id,
        "stage": stage,
        "environment": environment(),
        "load_s": round(time.monotonic() - started, 1),
    }
    started = time.monotonic()
    toy = verifier.score_batch(TOY)
    record["toy_batch_s"] = round(time.monotonic() - started, 1)
    record["repeat"] = compare(toy, verifier.score_batch(TOY))
    record["alone_vs_batch"] = compare(toy, [verifier.score(p, s) for p, s in TOY])
    record["toy"] = [
        {
            key: r.get(key)
            for key in (
                "error_score",
                "stop_reason",
                "generated_tokens",
                "step_scores",
                "missing_steps",
                "readouts",
            )
        }
        for r in toy
    ]
    record["direction"] = [
        {"right": toy[i]["error_score"], "wrong": toy[j]["error_score"]} for i, j in TOY_PAIRS
    ]
    record["direction_ok"] = all(d["wrong"] > d["right"] for d in record["direction"])
    # greedy decoding at every readout picks one of the two verdict tokens
    verdict_ids = set(verifier._verdict_ids)  # noqa: SLF001
    record["verdict_ok"] = all(ro["greedy"] in verdict_ids for r in toy for ro in r["readouts"])
    record["step_verdicts_generated"] = [r.get("n_steps_judged_generated", 0) for r in toy]
    record["rendered_toy_input"] = toy[1]["serialized_input"]
    record["toy_generated_text"] = [r["generated_text"] for r in toy]
    record["card"] = {**verifier.verifier_card(), "source": source_review(c)}
    record["input_lengths"] = input_lengths(verifier, rows)
    if torch.cuda.is_available():
        record["peak_reserved_gib"] = torch.cuda.max_memory_reserved() / 2**30
    verifier.close()
    record["failures"] = preflight_failures(record, c["granularity"])
    accepted = accepted_failures(c)
    record["accepted_failures"] = {f: accepted[f] for f in record["failures"] if f in accepted}
    return record


def preflight_verdict(record: dict[str, Any]) -> dict[str, Any]:
    """Print the summary line; exit with the first failure the authors have not accepted."""
    print(
        json.dumps(
            {
                "candidate_id": record["candidate_id"],
                "failures": record["failures"],
                "accepted": sorted(record["accepted_failures"]),
                "direction": record["direction"],
                "max_input": record["input_lengths"]["max"],
                "L_in": record["input_lengths"]["L_in"],
                "toy_stop_reasons": [r["stop_reason"] for r in record["toy"]],
                "peak_reserved_gib": record.get("peak_reserved_gib"),
            }
        )
    )
    blocking = [f for f in record["failures"] if f not in record["accepted_failures"]]
    if blocking:  # the run stops here and reports; nothing is scored
        raise SystemExit(EXIT[blocking[0]])
    return record


# ---------------------------------------------------------------------------------------
# score and the 203-state check
# ---------------------------------------------------------------------------------------


def resume_point(scores: Path, details: Path, ids: list[str]) -> int:
    return r1.resume_point(scores, details, ids)


def score_rows(
    verifier: Any,
    spec: VerifierSpec,
    rows: list[dict[str, Any]],
    dirs: dict[str, Path],
    candidate_id: str,
    *,
    every: int = 50,
) -> tuple[Path, Path, int]:
    ids = [r["state_id"] for r in rows]
    for key in ("scores", "details", "meta"):
        dirs[key].mkdir(parents=True, exist_ok=True)
    table = dirs["scores"] / f"{candidate_id}.jsonl"
    detail_file = dirs["details"] / f"{candidate_id}.jsonl"
    done = resume_point(table, detail_file, ids)
    step = verifier.config.batch_size
    start = done - done % step if done < len(rows) else done
    if start < done:  # whole batches only: a resumed batch is the original batch
        for path in (table, detail_file):
            kept = list(read_jsonl(path))[:start]
            path.unlink()
            with JsonlWriter(path) as writer:
                writer.write_many(kept)
    t0 = time.monotonic()
    for begin in range(start, len(rows), step):
        chunk = rows[begin : begin + step]
        outputs = verifier.score_batch([(r["problem"], r["solution"]) for r in chunk])
        if len(outputs) != len(chunk):
            die("the verifier returned a different number of scores than inputs")
        with JsonlWriter(detail_file) as writer:  # details first (resume_point)
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
                        sub_scores=(
                            {str(k): s for k, s in enumerate(steps, 1) if s is not None}
                            if steps
                            else None
                        ),
                    )
                )
        done = begin + len(chunk)
        if done % every < step or done == len(rows):
            rate = (done - start) / max(time.monotonic() - t0, 1e-9)
            eta = (len(rows) - done) / rate if rate else float("nan")
            truncated = sum(o["truncated"] for o in outputs)
            print(
                f"{candidate_id}: {done}/{len(rows)} ({rate * 3600:.0f}/h, ETA "
                f"{eta / 3600:.2f} h; this batch {truncated}/{len(chunk)} truncated)",
                flush=True,
            )
    return table, detail_file, start


def check203(base: Path, candidate_id: str) -> dict[str, Any]:
    c = load_candidate(candidate_id)
    paths = layout(base)
    ids = json.loads(paths["check203"].read_text(encoding="utf-8"))["state_ids"]
    screen = {r["state_id"]: r for r in read_rows(paths["screen_inputs"])}
    reference_file = stage_dirs(paths, "screen")["details"] / f"{candidate_id}.jsonl"
    if not reference_file.is_file():
        die(f"{candidate_id}: no screening rows to check against ({reference_file})")
    reference = {r["state_id"]: r for r in read_jsonl(reference_file)}
    if any(sid not in reference or sid not in screen for sid in ids):
        die(f"{candidate_id}: the screening rows do not cover the 203 states")
    spec = spec_for(c, paths["models"] / candidate_id)
    verifier = build_verifier(spec)
    verifier.load()
    started = time.monotonic()
    again = verifier.score_batch([(screen[s]["problem"], screen[s]["solution"]) for s in ids])
    elapsed = time.monotonic() - started
    config_hash = verifier.config_hash
    verifier.close()
    result = compare([reference[s] for s in ids], again)
    result["differing"] = [{**d, "state_id": ids[d["index"]]} for d in result["differing"]]
    record = {
        "candidate_id": candidate_id,
        "environment": environment(),
        "config_hash": config_hash,
        "reference_config_hash": reference[ids[0]]["config_hash"],
        "elapsed_s": round(elapsed, 1),
        "identical": result["n_differing"] == 0 and config_hash == reference[ids[0]]["config_hash"],
        **result,
        "checked_at": utc_now(),
    }
    write_json(paths["check203_out"] / f"{candidate_id}.json", record)
    print(
        json.dumps(
            {
                k: record[k]
                for k in ("candidate_id", "identical", "n_differing", "max_score_gap", "elapsed_s")
            }
        )
    )
    if not record["identical"]:
        raise SystemExit(EXIT["check203"])
    return record


def score(base: Path, candidate_id: str, stage: str) -> dict[str, Any]:
    c = load_candidate(candidate_id)
    paths = layout(base)
    rows = stage_rows(paths, stage)
    spec = spec_for(c, paths["models"] / candidate_id)
    verifier = build_verifier(spec)
    verifier.load()
    if stage == "full":
        check = paths["check203_out"] / f"{candidate_id}.json"
        record = json.loads(check.read_text(encoding="utf-8")) if check.is_file() else {}
        if not record.get("identical") or record.get("config_hash") != verifier.config_hash:
            die(f"{candidate_id}: the 203-state check has not passed in this configuration")
    dirs = stage_dirs(paths, stage)
    started_at, t0 = utc_now(), time.monotonic()
    table, detail_file, start = score_rows(verifier, spec, rows, dirs, candidate_id)
    config_hash = verifier.config_hash
    verifier.close()
    details = list(read_jsonl(detail_file))
    record = {
        "candidate_id": candidate_id,
        "stage": stage,
        "model_id": spec.model_id,
        "revision": spec.revision,
        "n": len(rows),
        "resumed_from": start,
        "started_at": started_at,
        "finished_at": utc_now(),
        "elapsed_s": round(time.monotonic() - t0, 1),
        "truncated": sum(d["truncated"] for d in details),
        "stop_reasons": {
            reason: sum(d["stop_reason"] == reason for d in details)
            for reason in ("stop", "eos", "length")
        },
        "table_sha256": sha256_file(table),
        "details_sha256": sha256_file(detail_file),
        "config_hash": config_hash,
        "environment": environment(),
    }
    write_json(dirs["meta"] / f"{candidate_id}.score.json", record)
    print(
        json.dumps(
            {
                k: record[k]
                for k in ("candidate_id", "stage", "n", "elapsed_s", "truncated", "table_sha256")
            }
        )
    )
    return record


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument(
        "command", choices=("fetch", "speed", "precheck", "preflight", "check203", "score")
    )
    ap.add_argument("--base", type=Path, required=True)
    ap.add_argument("--candidate", required=True)
    ap.add_argument("--stage", choices=STAGES)
    ap.add_argument("--no-weights", action="store_true", help="fetch: config, tokenizer, template")
    args = ap.parse_args(argv)
    os.environ.setdefault("HF_HUB_DISABLE_IMPLICIT_TOKEN", "1")
    if args.command in ("preflight", "score") and not args.stage:
        ap.error(f"{args.command} needs --stage")
    if args.command == "fetch":
        fetch(args.base, args.candidate, weights=not args.no_weights)
    elif args.command == "speed":
        speed(args.base, args.candidate)
    elif args.command == "precheck":
        precheck(args.base, args.candidate)
    elif args.command == "preflight":
        preflight(args.base, args.candidate, args.stage)
    elif args.command == "check203":
        check203(args.base, args.candidate)
    else:
        score(args.base, args.candidate, args.stage)
    return 0


if __name__ == "__main__":
    sys.exit(main())
