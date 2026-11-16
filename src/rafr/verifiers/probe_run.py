"""Step 2 probe runner: download, preflight, score -- one model at a time (decisions.md).

    python -m rafr.verifiers.probe_run download  --models all-gpu --out OUT
    python -m rafr.verifiers.probe_run preflight --models lg_1b --out OUT
    python -m rafr.verifiers.probe_run score     --models all-gpu --out OUT \\
        --input probe_inputs.jsonl --input-sha256 <hex>
    python -m rafr.verifiers.probe_run render    --models lg_12b --out OUT   # templates only
    python -m rafr.verifiers.probe_run lock                                   # after stop A

The guard LMs run on the rented GPU; ``enc_paradetox`` runs the same ``score`` command on the
local CPU. Layout of OUT::

    download_manifest.json    per model: repo, resolved commit, sha256 of every weight file
    environment.json          python / torch / transformers / GPU of the last command
    preflight/<model>.json    renders, greedy continuations, verdict tokens, direction check
    scores/<verifier_id>.jsonl  the scores table (rafr.contracts.schemas.Score rows)
    meta/<model>.json         batch size, lengths, truncations, timings, versions

Each model is loaded, run and released before the next one, and each configuration's table
is written (atomically) as soon as it is complete, so an interrupted run loses at most the
model it was on; a rerun skips every table that is already complete.

The input may hold ``state_id`` and ``text`` only and must match the sha256 recorded when it
was built locally (y* never leaves the laptop; App. D / decisions.md). Scoring refuses to
start until the probe prompts are locked (stop point A, ``probe.check_prompt_lock``), and a
guard LM is scored only after its preflight has settled the verdict prefix and tokens and its
direction check has passed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import sys
import time
from collections.abc import Callable, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from rafr.contracts.schemas import Score
from rafr.core.textio import crlf_rewrite_notice
from rafr.verifiers.candidates import probe as P

INPUT_FIELDS = frozenset({"state_id", "text"})
WEIGHT_SUFFIXES = (".safetensors", ".bin")
#: Weights, tokenizer, config and templates only; never Meta's duplicate ``original/``
#: checkpoint, pictures, or other formats of the same weights.
ALLOW_PATTERNS = ["*.json", "*.model", "*.txt", "*.jinja", "tokenizer*", "*.safetensors"]
IGNORE_PATTERNS = ["original/*", "*.msgpack", "*.h5", "*.onnx", "*.gguf", "*.png", "*.pth"]
MANIFEST_NAME = "download_manifest.json"
SCORE_CHUNK = 256
GREEDY_TOKENS = 24
CHUNK = 1 << 20

Echo = Callable[[str], None]


class PreflightError(RuntimeError):
    """The model does not behave as the probe assumes; stop and report, score nothing.

    ``report`` is the preflight record up to the failure, so it can be written out and read.
    """

    def __init__(self, message: str, report: dict[str, Any] | None = None) -> None:
        super().__init__(message)
        self.report = report or {}


def utc_now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as fh:
        while block := fh.read(CHUNK):
            h.update(block)
    return h.hexdigest()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(
        json.dumps(payload, indent=1, ensure_ascii=False), encoding="utf-8", newline="\n"
    )
    with crlf_rewrite_notice(path):
        tmp.replace(path)


def read_json(path: Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def select_models(spec: str) -> list[str]:
    """``all``, ``all-gpu``, or a comma list of probe model ids."""
    if spec == "all":
        return ["enc_paradetox", *P.GPU_ORDER]
    if spec == "all-gpu":
        return list(P.GPU_ORDER)
    ids = [s.strip() for s in spec.split(",") if s.strip()]
    for i in ids:
        P.model(i)
    return ids


# ---------------------------------------------------------------------------------------
# input
# ---------------------------------------------------------------------------------------


def load_probe_inputs(path: Path, expected_sha256: str) -> list[tuple[str, str]]:
    """``[(state_id, text)]`` in file order, after the sha256 and the two-field check."""
    actual = sha256_file(path)
    if actual != expected_sha256.strip().lower():
        raise SystemExit(f"{path}: sha256 {actual} does not match the expected {expected_sha256}")
    out: list[tuple[str, str]] = []
    seen: set[str] = set()
    with Path(path).open("r", encoding="utf-8") as fh:
        for i, line in enumerate(fh, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if set(row) != INPUT_FIELDS:
                raise SystemExit(
                    f"{path}:{i}: fields {sorted(row)}; only state_id and text allowed"
                )
            sid, text = str(row["state_id"]), row["text"]
            if not isinstance(text, str) or sid in seen:
                raise SystemExit(f"{path}:{i}: bad text or duplicate state_id {sid!r}")
            seen.add(sid)
            out.append((sid, text))
    return out


# ---------------------------------------------------------------------------------------
# environment and download
# ---------------------------------------------------------------------------------------


def environment() -> dict[str, Any]:
    """Library versions and the device, recorded with every command."""
    env: dict[str, Any] = {
        "recorded_at": utc_now(),
        "git_head": os.environ.get("RAFR_GIT_HEAD") or _git_head(),
        "python": platform.python_version(),
        "platform": platform.platform(),
    }
    for name in ("torch", "transformers", "tokenizers", "huggingface_hub", "accelerate"):
        try:
            mod = __import__(name)
            env[name] = str(getattr(mod, "__version__", "?"))
        except ImportError:
            env[name] = None
    try:
        import torch

        env["cuda"] = torch.version.cuda
        if torch.cuda.is_available():
            env["gpu"] = torch.cuda.get_device_name(0)
            env["gpu_memory_gb"] = round(torch.cuda.get_device_properties(0).total_memory / 2**30)
            env["driver"] = _driver_version()
            # what the formal pod check reads (image CUDA_VERSION, Ubuntu, torch, GPU name)
            from rafr.verifiers.formal_pod import observe_here

            env["pod"] = observe_here()
        else:
            env["gpu"] = None
    except ImportError:
        pass
    return env


def _driver_version() -> str | None:
    """The NVIDIA driver version as nvidia-smi reports it (torch does not expose it)."""
    import subprocess

    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            check=False,
            timeout=30,
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return None
    lines = [x.strip() for x in out.splitlines() if x.strip()]
    return lines[0] if lines else None


def _git_head() -> str | None:
    """HEAD of the checkout this runs from; on the pod the code is a ``git archive`` without
    ``.git``, and env/probe_remote_setup.sh exports RAFR_GIT_HEAD instead."""
    try:
        from rafr.logging.snapshot import git_head

        return git_head()
    except (RuntimeError, OSError):
        return None


def load_manifest(out_dir: Path) -> dict[str, Any]:
    path = Path(out_dir) / MANIFEST_NAME
    return read_json(path) if path.is_file() else {"models": {}}


def pinned_revision(out_dir: Path, model_id: str) -> str:
    entry = (load_manifest(out_dir).get("models") or {}).get(model_id)
    if not entry or not entry.get("revision"):
        raise SystemExit(f"{model_id}: not in {Path(out_dir) / MANIFEST_NAME}; run download first")
    return str(entry["revision"])


def download(model_ids: Sequence[str], out_dir: Path, echo: Echo) -> dict[str, Any]:
    """Resolve each repo's head commit, fetch that commit, hash every weight file.

    HF_TOKEN is read by huggingface_hub from the environment; this code never touches it.
    """
    from huggingface_hub import HfApi, snapshot_download

    api = HfApi()
    manifest = load_manifest(out_dir)
    models = dict(manifest.get("models") or {})
    for model_id in model_ids:
        m = P.model(model_id)
        info = api.model_info(m.repo_id)
        revision = str(info.sha)
        files = [s.rfilename for s in (info.siblings or [])]
        allow = list(ALLOW_PATTERNS)
        if not any(f.endswith(".safetensors") for f in files):
            allow.append("*.bin")  # s-nlp/roberta_toxicity_classifier ships pytorch_model.bin
        echo(f"{model_id}: {m.repo_id}@{revision}")
        prev = models.get(model_id) or {}
        if prev.get("revision") == revision and prev.get("weight_files"):
            echo("  already downloaded and hashed at this revision")
            continue
        started = time.monotonic()
        local = Path(
            snapshot_download(
                m.repo_id, revision=revision, allow_patterns=allow, ignore_patterns=IGNORE_PATTERNS
            )
        )
        weights = []
        for path in sorted(local.rglob("*")):
            if path.is_file() and path.suffix in WEIGHT_SUFFIXES:
                weights.append(
                    {
                        "path": str(path.relative_to(local)).replace("\\", "/"),
                        "bytes": path.stat().st_size,
                        "sha256": sha256_file(path),
                    }
                )
        if not weights:
            raise SystemExit(f"{model_id}: no weight files after download")
        models[model_id] = {
            "repo_id": m.repo_id,
            "revision": revision,
            "hub_sha_at_check": m.hub_sha_at_check,
            "dtype": m.dtype,
            "gated": m.gated,
            "license": m.license,
            "weight_files": weights,
            "downloaded_at": utc_now(),
            "download_s": round(time.monotonic() - started, 1),
        }
        write_json(Path(out_dir) / MANIFEST_NAME, {"models": models})
        echo(f"  {len(weights)} weight file(s), {sum(w['bytes'] for w in weights) / 2**30:.1f} GiB")
    write_json(Path(out_dir) / MANIFEST_NAME, {"models": models})
    return {"models": models}


# ---------------------------------------------------------------------------------------
# preflight (guard LMs)
# ---------------------------------------------------------------------------------------

_PLACEHOLDER = P.Verdict((), "", -1, -1, "", "")


def _verdict_forms(
    backend: Any, m: P.ProbeModel, observed: list[tuple[int, str]]
) -> tuple[P.Verdict | None, dict[str, Any]]:
    """Both verdict token ids, in the spelling the model actually emits, each one token."""
    assert m.toxic_word is not None and m.safe_word is not None
    lead = {piece.startswith(" ") for _, piece in observed}
    checks: dict[str, Any] = {
        "bare_word_single_token": {
            w: backend.single_token_id(w) is not None for w in (m.toxic_word, m.safe_word)
        }
    }
    if len(lead) != 1:
        checks["error"] = f"inconsistent leading space across verdict tokens: {observed}"
        return None, checks
    space = " " if lead.pop() else ""
    forms = {"toxic": space + m.toxic_word, "safe": space + m.safe_word}
    ids = {k: backend.single_token_id(f) for k, f in forms.items()}
    checks["forms"] = forms
    checks["form_ids"] = ids
    if None in ids.values():
        checks["error"] = f"a verdict form is not a single token: {forms} -> {ids}"
        return None, checks
    for tid, piece in observed:
        want = ids["toxic"] if piece.strip() == m.toxic_word else ids["safe"]
        if tid != want:
            checks["error"] = f"emitted id {tid} ({piece!r}) is not the encoded form id {want}"
            return None, checks
    return (
        P.Verdict((), "", int(ids["toxic"]), int(ids["safe"]), forms["toxic"], forms["safe"]),
        checks,
    )


def preflight_lm(
    backend: Any, m: P.ProbeModel, prompt_dir: Path | None, echo: Echo
) -> tuple[dict[str, Any], P.Verdict, bool, dict[str, dict[str, Any]]]:
    """Settle the verdict prefix and tokens on the invented sentences, then check direction.

    Raises :class:`PreflightError` when the model does not emit a verdict word, emits it
    after different prefixes, emits a prefix other than the expected one, or spells a verdict
    word as more than one token (decisions.md section 4: stop and report).
    """
    configs = P.configs_for(m.model_id, prompt_dir)
    report: dict[str, Any] = {
        "model_id": m.model_id,
        "repo_id": m.repo_id,
        "revision": backend.revision,
        "template_source": getattr(backend, "template_source", None),
        "attn_implementation": getattr(backend, "attn_implementation", None),
        "expected_prefix": m.label_prefix,
        "example_texts": {"neutral": P.NEUTRAL_EXAMPLE, "abusive": P.ABUSIVE_EXAMPLE},
        "configs": {},
    }
    lg_manual = False
    scorers: dict[str, P.GuardProbeScorer] = {}
    for cfg in configs:
        scorer = P.GuardProbeScorer(cfg, backend, _PLACEHOLDER, prompt_dir=prompt_dir)
        for call in cfg.calls:
            cats = call.template_kwargs.get("categories")
            if cats is not None:
                applied = P.categories_applied(scorer.render(call, P.NEUTRAL_EXAMPLE), cats)
                lg_manual = not applied
                report["categories_mechanism"] = "chat_template" if applied else "manual"
        scorer.lg_manual = lg_manual
        scorers[cfg.verifier_id] = scorer

    splits: list[dict[str, Any]] = []
    for cfg in configs:
        scorer = scorers[cfg.verifier_id]
        entry: dict[str, Any] = {"calls": {}, "sources": list(cfg.sources)}
        for call in cfg.calls:
            renders = {}
            for which, text in (("neutral", P.NEUTRAL_EXAMPLE), ("abusive", P.ABUSIVE_EXAMPLE)):
                prompt = scorer.render(call, text)
                gen = backend.greedy(prompt, GREEDY_TOKENS)
                found = P.split_prefix(gen, m.toxic_word or "", m.safe_word or "")
                renders[which] = {
                    "prompt": prompt,
                    "greedy": "".join(piece for _, piece in gen),
                    "greedy_ids": [t for t, _ in gen],
                }
                splits.append({"where": f"{cfg.verifier_id}/{call.sub_id}/{which}", "found": found})
            entry["calls"][call.sub_id] = renders
        report["configs"][cfg.verifier_id] = entry

    missing = [s["where"] for s in splits if s["found"] is None]
    if missing:
        write_report_error(report, f"no verdict word in the greedy continuation of {missing}")
    prefixes = {tuple(s["found"][0]) for s in splits}
    if len(prefixes) != 1:
        write_report_error(report, f"the verdict prefix differs across prompts: {prefixes}")
    prefix_ids = tuple(prefixes.pop())
    prefix_text = "".join(backend.token_piece(t) for t in prefix_ids)
    report["observed_prefix"] = {"ids": list(prefix_ids), "text": prefix_text}
    if prefix_text != m.label_prefix:
        write_report_error(
            report,
            f"observed verdict prefix {prefix_text!r} differs from the expected "
            f"{m.label_prefix!r} (probe.MODELS[{m.model_id!r}].label_prefix)",
        )
    observed = [(s["found"][1], s["found"][2]) for s in splits]
    base_verdict, checks = _verdict_forms(backend, m, observed)
    report["verdict_tokens"] = checks
    if base_verdict is None:
        write_report_error(report, str(checks.get("error")))
    assert base_verdict is not None
    verdict = P.Verdict(
        prefix_ids,
        prefix_text,
        base_verdict.toxic_id,
        base_verdict.safe_id,
        base_verdict.toxic_form,
        base_verdict.safe_form,
    )

    directions: dict[str, dict[str, Any]] = {}
    for cfg in configs:
        scorer = scorers[cfg.verifier_id]
        scorer.verdict = verdict
        neutral, abusive = scorer.score_detailed([P.NEUTRAL_EXAMPLE, P.ABUSIVE_EXAMPLE])
        ok = P.direction_ok(neutral.score, abusive.score)
        directions[cfg.verifier_id] = {
            "neutral": neutral.score,
            "abusive": abusive.score,
            "neutral_sub": neutral.sub_scores,
            "abusive_sub": abusive.sub_scores,
            "ok": ok,
        }
        report["configs"][cfg.verifier_id]["direction"] = directions[cfg.verifier_id]
        report["configs"][cfg.verifier_id]["prompt_hash"] = scorer.prompt_hash
        echo(
            f"  {cfg.verifier_id}: neutral {neutral.score:.4f}  abusive {abusive.score:.4f}  "
            f"direction {'ok' if ok else 'WRONG'}"
        )
    report["passed"] = True
    return report, verdict, lg_manual, directions


def write_report_error(report: dict[str, Any], message: str) -> None:
    report["passed"] = False
    report["error"] = message
    raise PreflightError(message, report)


# ---------------------------------------------------------------------------------------
# scoring
# ---------------------------------------------------------------------------------------


def table_complete(path: Path, n: int, model_version: str, prompt_hash: str | None) -> bool:
    if not path.is_file():
        return False
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    return (
        len(rows) == n
        and all(r.get("model_version") == model_version for r in rows)
        and all(r.get("prompt_hash") == prompt_hash for r in rows)
    )


def write_table(path: Path, rows: list[Score]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".jsonl.tmp")
    with tmp.open("w", encoding="utf-8", newline="\n") as fh:
        for r in rows:
            fh.write(r.model_dump_json() + "\n")
    with crlf_rewrite_notice(path):
        tmp.replace(path)


def score_lm(
    m: P.ProbeModel,
    items: Sequence[tuple[str, str]],
    out_dir: Path,
    *,
    prompt_dir: Path | None,
    batch_size: int,
    fixed_shape: bool,
    echo: Echo,
    scores_dir: Path | None = None,
) -> dict[str, Any]:
    from rafr.verifiers.candidates.torch_probe import TorchProbeBackend

    scores_dir = Path(scores_dir or out_dir / "scores")

    revision = pinned_revision(out_dir, m.model_id)
    meta: dict[str, Any] = {"model_id": m.model_id, "started_at": utc_now(), "configs": {}}
    t0 = time.monotonic()
    backend = TorchProbeBackend(
        model=m, revision=revision, batch_size=batch_size, fixed_shape=fixed_shape
    )
    meta["load_s"] = round(time.monotonic() - t0, 1)
    try:
        try:
            report, verdict, lg_manual, directions = preflight_lm(backend, m, prompt_dir, echo)
        except PreflightError as exc:
            write_json(out_dir / "preflight" / f"{m.model_id}.json", exc.report)
            raise
        write_json(out_dir / "preflight" / f"{m.model_id}.json", report)
        meta.update(
            {
                "repo_id": m.repo_id,
                "revision": backend.revision,
                "model_version": f"{m.repo_id}@{backend.revision}",
                "dtype": m.dtype,
                "batch_size": batch_size,
                "fixed_shape": fixed_shape,
                "text_max_tokens": P.TEXT_MAX_TOKENS,
                "verdict": {
                    "prefix_ids": list(verdict.prefix_ids),
                    "prefix_text": verdict.prefix_text,
                    "toxic": [verdict.toxic_id, verdict.toxic_form],
                    "safe": [verdict.safe_id, verdict.safe_form],
                },
                "categories_mechanism": report.get("categories_mechanism"),
                "attn_implementation": getattr(backend, "attn_implementation", None),
                "template_source": getattr(backend, "template_source", None),
            }
        )
        texts = [t for _, t in items]
        for cfg in P.configs_for(m.model_id, prompt_dir):
            vid = cfg.verifier_id
            scorer = P.GuardProbeScorer(
                cfg, backend, verdict, prompt_dir=prompt_dir, lg_manual=lg_manual
            )
            info: dict[str, Any] = {"prompt_hash": scorer.prompt_hash, "direction": directions[vid]}
            meta["configs"][vid] = info
            if not directions[vid]["ok"]:
                info["status"] = "skipped: direction check failed"
                echo(f"{vid}: SKIPPED, direction check failed")
                continue
            path = scores_dir / f"{vid}.jsonl"
            if table_complete(path, len(items), scorer.model_version, scorer.prompt_hash):
                info["status"] = "already complete"
                echo(f"{vid}: already complete")
                continue
            info["pad_to"] = {c.sub_id: scorer.pad_length(c) for c in cfg.calls}
            started = time.monotonic()
            rows: list[Score] = []
            truncated: list[dict[str, Any]] = []
            for i in range(0, len(texts), SCORE_CHUNK):
                chunk = items[i : i + SCORE_CHUNK]
                for (sid, _), r in zip(
                    chunk, scorer.score_detailed(texts[i : i + SCORE_CHUNK]), strict=True
                ):
                    rows.append(
                        Score(
                            state_id=sid,
                            verifier_id=vid,
                            model_version=scorer.model_version,
                            prompt_hash=scorer.prompt_hash,
                            score=r.score,
                            sub_scores=r.sub_scores,
                        )
                    )
                    if r.truncated:
                        truncated.append({"state_id": sid, "n_text_tokens": r.n_text_tokens})
                done = min(i + SCORE_CHUNK, len(texts))
                echo(f"  {vid}: {done}/{len(texts)} ({time.monotonic() - started:.0f}s)")
            write_table(path, rows)
            info.update(
                {
                    "status": "complete",
                    "n_rows": len(rows),
                    "truncated": truncated,
                    "elapsed_s": round(time.monotonic() - started, 1),
                    "table_sha256": sha256_file(path),
                }
            )
            write_json(out_dir / "meta" / f"{m.model_id}.json", meta)
    finally:
        backend.close()
    meta["finished_at"] = utc_now()
    meta["elapsed_s"] = round(time.monotonic() - t0, 1)
    write_json(out_dir / "meta" / f"{m.model_id}.json", meta)
    return meta


def score_encoder(
    m: P.ProbeModel,
    items: Sequence[tuple[str, str]],
    out_dir: Path,
    *,
    echo: Echo,
    scores_dir: Path | None = None,
) -> dict[str, Any]:
    """``enc_paradetox`` on the local CPU: P(toxic), the class resolved by name via id2label."""
    scores_dir = Path(scores_dir or out_dir / "scores")
    from rafr.verifiers.candidates.registry import build

    revision = pinned_revision(out_dir, m.model_id)
    t0 = time.monotonic()
    v = build(
        "roberta-toxicity",
        verifier_id=m.model_id,
        revision=revision,
        inference_batch_size=P.ENCODER_BATCH_SIZE,
    )
    neutral, abusive = v.score([P.NEUTRAL_EXAMPLE, P.ABUSIVE_EXAMPLE])
    ok = P.direction_ok(neutral, abusive)
    echo(f"  {m.model_id}: neutral {neutral:.4f}  abusive {abusive:.4f}  direction {ok}")
    backend = v.backend
    meta: dict[str, Any] = {
        "model_id": m.model_id,
        "repo_id": m.repo_id,
        "revision": backend.revision,
        "model_version": v.model_version,
        "dtype": m.dtype,
        "batch_size": v.inference_batch_size,
        "truncation": v.truncation.describe(),
        "id2label": {str(k): val for k, val in backend.id2label.items()},
        "toxic_index": v.toxic_index,
        "started_at": utc_now(),
        "configs": {},
    }
    info: dict[str, Any] = {
        "prompt_hash": None,
        "direction": {"neutral": neutral, "abusive": abusive, "ok": ok},
    }
    meta["configs"][m.model_id] = info
    if not ok:
        info["status"] = "skipped: direction check failed"
        write_json(out_dir / "meta" / f"{m.model_id}.json", meta)
        return meta
    path = scores_dir / f"{m.model_id}.jsonl"
    if table_complete(path, len(items), v.model_version, None):
        info["status"] = "already complete"
    else:
        tok = backend.tokenizer
        rows: list[Score] = []
        truncated: list[dict[str, Any]] = []
        started = time.monotonic()
        for i in range(0, len(items), SCORE_CHUNK):
            chunk = list(items[i : i + SCORE_CHUNK])
            scores = v.score([t for _, t in chunk])
            for (sid, text), s in zip(chunk, scores, strict=True):
                rows.append(
                    Score(
                        state_id=sid, verifier_id=m.model_id, model_version=v.model_version, score=s
                    )
                )
                n = len(tok(text)["input_ids"])
                if n > P.TEXT_MAX_TOKENS:
                    truncated.append({"state_id": sid, "n_tokens_with_special": n})
            echo(f"  {m.model_id}: {min(i + SCORE_CHUNK, len(items))}/{len(items)}")
        write_table(path, rows)
        info.update(
            {
                "status": "complete",
                "n_rows": len(rows),
                "truncated": truncated,
                "elapsed_s": round(time.monotonic() - started, 1),
                "table_sha256": sha256_file(path),
            }
        )
    meta["finished_at"] = utc_now()
    meta["elapsed_s"] = round(time.monotonic() - t0, 1)
    write_json(out_dir / "meta" / f"{m.model_id}.json", meta)
    return meta


def release_gpu() -> None:
    import gc

    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass


# ---------------------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("command", choices=("download", "preflight", "score", "render", "lock"))
    p.add_argument("--models", default="all-gpu", help="all | all-gpu | comma list of ids")
    p.add_argument("--out", type=Path, default=Path("data/probe/out"))
    p.add_argument("--scores-dir", type=Path, help="where the tables go (default: OUT/scores)")
    p.add_argument("--input", type=Path, help="probe_inputs.jsonl (state_id, text)")
    p.add_argument("--input-sha256", help="sha256 recorded when the input was built")
    p.add_argument("--prompt-dir", type=Path, default=None, help="default: configs/prompts")
    p.add_argument("--batch-size", type=int, default=P.LM_BATCH_SIZE)
    p.add_argument(
        "--dynamic-padding",
        action="store_true",
        help="pad each batch to its longest prompt (faster, not bit-reproducible across runs)",
    )
    return p


def main(argv: Sequence[str] | None = None, echo: Echo | None = None) -> int:
    args = build_parser().parse_args(argv)
    say = echo or (lambda s: print(s, flush=True))
    out = Path(args.out)

    if args.command == "lock":
        d = Path(args.prompt_dir or P.prompts_dir())
        payload = {"locked_at": utc_now(), **P.lock_payload(d)}
        write_json(d / P.PROMPT_LOCK_FILE, payload)
        say(f"locked {len(payload['files'])} prompt file(s) -> {d / P.PROMPT_LOCK_FILE}")
        return 0

    out.mkdir(parents=True, exist_ok=True)
    write_json(out / "environment.json", environment())
    models = select_models(args.models)

    if args.command == "download":
        download(models, out, say)
        return 0

    if args.command == "render":
        from rafr.verifiers.candidates.torch_probe import TorchProbeBackend

        for model_id in models:
            m = P.model(model_id)
            if not m.is_lm:
                continue
            backend = TorchProbeBackend.templates_only(
                model=m, revision=pinned_revision(out, model_id)
            )
            for cfg in P.configs_for(model_id, args.prompt_dir):
                scorer = P.GuardProbeScorer(cfg, backend, _PLACEHOLDER, prompt_dir=args.prompt_dir)
                for call in cfg.calls:
                    cats = call.template_kwargs.get("categories")
                    if cats is not None:
                        rendered = scorer.render(call, P.NEUTRAL_EXAMPLE)
                        scorer.lg_manual = not P.categories_applied(rendered, cats)
                    src = backend.template_source
                    say(f"===== {cfg.verifier_id} / {call.sub_id} (template: {src})")
                    say(scorer.render(call, P.NEUTRAL_EXAMPLE))
        return 0

    fixed = not args.dynamic_padding
    if args.command == "preflight":
        from rafr.verifiers.candidates.torch_probe import TorchProbeBackend

        status = 0
        for model_id in models:
            m = P.model(model_id)
            if not m.is_lm:
                continue
            say(f"=== preflight {model_id}")
            backend = TorchProbeBackend(
                model=m,
                revision=pinned_revision(out, model_id),
                batch_size=args.batch_size,
                fixed_shape=fixed,
            )
            try:
                report, *_ = preflight_lm(backend, m, args.prompt_dir, say)
            except PreflightError as exc:
                write_json(out / "preflight" / f"{model_id}.json", exc.report)
                say(f"PREFLIGHT FAILED {model_id}: {exc}")
                status = 3
                continue
            finally:
                backend.close()
                release_gpu()
            write_json(out / "preflight" / f"{model_id}.json", report)
        return status

    # score
    if args.input is None or not args.input_sha256:
        raise SystemExit("score needs --input and --input-sha256")
    if any(P.model(m).is_lm for m in models):
        P.check_prompt_lock(args.prompt_dir)  # the encoder has no prompt to lock
    items = load_probe_inputs(args.input, args.input_sha256)
    say(f"{len(items)} input rows, sha256 verified; models: {models}")
    for model_id in models:
        m = P.model(model_id)
        say(f"=== {model_id} ({m.repo_id})")
        try:
            if m.is_lm:
                score_lm(
                    m,
                    items,
                    out,
                    prompt_dir=args.prompt_dir,
                    batch_size=args.batch_size,
                    fixed_shape=fixed,
                    echo=say,
                    scores_dir=args.scores_dir,
                )
            else:
                score_encoder(m, items, out, echo=say, scores_dir=args.scores_dir)
        except PreflightError as exc:
            say(f"STOP {model_id}: preflight failed: {exc}")
            return 3
        finally:
            release_gpu()
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
