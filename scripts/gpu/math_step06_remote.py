"""Pod side of math step 6. Never on the laptop.

    python scripts/gpu/math_step06_remote.py fetch     --base B --candidate C [--no-weights]
    python scripts/gpu/math_step06_remote.py speed     --base B --candidate C
    python scripts/gpu/math_step06_remote.py precheck  --base B --candidate C
    python scripts/gpu/math_step06_remote.py preflight --base B --candidate C
    python scripts/gpu/math_step06_remote.py check203  --base B --candidate C [--loop]
    python scripts/gpu/math_step06_remote.py score     --base B --candidate C --stage S
    python scripts/gpu/math_step06_remote.py loopfit   --base B --candidate C [--l-in N]
    python scripts/gpu/math_step06_remote.py loopselect --base B --candidate C
    python scripts/gpu/math_step06_remote.py loopdiag  --base B --candidate C --l-in N \
        --batch-size B
    python scripts/gpu/math_step06_remote.py loopbisect --base B --candidate C --lo L --hi H \
        --batch-size B [--prefill-kernel efficient|flash|cudnn|math]

The two V are step 2's (``configs/math/step02_round2.json``, ``stop_point_b``) and every
command first checks that ``configs/math/formal.json`` holds the candidate's spec unchanged.
``fetch``, ``speed`` and ``precheck`` are step 2's own (``scripts/gpu/math_vprobe2_remote.py``, same
layout under ``--base``). Scoring is the generative variant (``generative.py``), with
B, L_in, L_gen, the prompts, stop string and attention exactly as in step 2.

The Exp 2 loop (the smoke at the end of the session, and the formal Exp 2) runs with L_in
4,096 and its own B (``configs/math/exp2_loop.json``; 2026-10-13,: B and L_in may
be adjusted by stage once the precheck's padding transparency passes and the 203-state check
agrees). ``loopfit`` is that precheck: the tiny model's determinism checks and padding
transparency at the loop L_in, then the full-size random-weight run at the loop shape from the
V's step 2 B down the grid until one fits -- the loop's B (exit 13 when none fits).
``check203 --loop`` is the 203-state check in the loop configuration against the same step 2
rows; only B and L_in may differ from step 2's configuration (``<id>.loop.json``).
``loopselect`` (2026-10-13, after R1-Distill-Llama's loop check at L_in 4,096 / B 24
failed): for a V with several loop L_in values (``max_input_tokens_order``), the largest at
which ``loopfit`` fits a B >= 8 and ``check203 --loop`` agrees; the loop reads each V's B and
L_in from its ``out/loopfit/<id>.json``.

* preflight: step 2's toy checks on the real weights (direction failure accepted for
  R1-Distill-Llama only, as in step 2) and every input of this session -- the stage files and
  Exp 2's first states when on the pod -- built and checked against L_in (exit 8).
* check203: the 203 D_fit states of ``check203.json`` scored again and compared with their
  step 2 screening rows (``data/reference/<id>.jsonl``): generated and forced token ids, every
  verdict readout's logits, the score. All identical: passed (``bitwise``). Otherwise the math
  alternative criterion (2026-10-12 / 10-13): no flip at the session tau and at
  least 95% of the 203 with bitwise identical token sequences -- passed (``alternative``),
  the job continues and the record keeps the identical share, the flips and each differing
  item's score gap. Neither: exit 15, nothing scored. A different configuration hash: exit 10.
* score: a stage's inputs in order (``d_gate``, ``d_exp1``, ``pool_rest``, ``stability``), only
  after this session's check passed in this configuration; Core ``Score`` rows plus one detail
  row each; resumes after the last complete batch.

Nothing here reads a label or a reference answer: the inputs carry state ids and public text.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import subprocess
import sys
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts" / "gpu"))

import math_vprobe2_remote as r2  # noqa: E402 - step 2's pod worker
from rafr.domains.math.models import build_verifier  # noqa: E402
from rafr.logging.writer import read_jsonl  # noqa: E402

FORMAL_CONFIG = ROOT / "configs" / "math" / "formal.json"
LOOP_CONFIG = ROOT / "configs" / "math" / "exp2_loop.json"
# The settings the Exp 2 loop may change from step 2's (2026-10-13).
LOOP_ADJUSTABLE = ("batch_size", "max_input_tokens")
STAGES = ("d_gate", "d_exp1", "pool_rest", "stability")
X1_FILES = ("x1_smoke.jsonl", "x1_formal.jsonl")
# math alternative criterion (2026-10-12): at least this share bitwise identical.
ALT_IDENTICAL_SHARE = 0.95
EXIT = {**r2.EXIT, "check203_configuration": 10, "check203_alternative": 15, "no_check": 16}

die = r2.die
utc_now = r2.utc_now
write_json = r2.write_json
sha256_file = r2.sha256_file


def layout(base: Path) -> dict[str, Any]:
    paths: dict[str, Any] = r2.layout(base)
    data = Path(base) / "data"
    paths["stage_inputs"] = {stage: data / f"{stage}_inputs.jsonl" for stage in STAGES}
    paths["x1"] = [data / name for name in X1_FILES]
    paths["reference"] = data / "reference"
    paths["tau"] = data / "tau.json"
    return paths


def load_v(candidate_id: str, formal: Path | None = None) -> dict[str, Any]:
    """Step 2's candidate entry, refused unless formal.json holds its spec unchanged."""
    c = r2.load_candidate(candidate_id)
    specs = json.loads(Path(formal or FORMAL_CONFIG).read_text(encoding="utf-8"))["verifiers"]
    if c["spec"] not in specs:
        die(f"{candidate_id}: configs/math/formal.json does not hold step 2's spec unchanged")
    return c


def v_index(candidate: dict[str, Any], formal: Path | None = None) -> int:
    specs = json.loads(Path(formal or FORMAL_CONFIG).read_text(encoding="utf-8"))["verifiers"]
    return specs.index(candidate["spec"])


def read_tau(paths: dict[str, Any], candidate: dict[str, Any]) -> float:
    path = paths["tau"]
    if not path.is_file():
        die(f"{path} not found: the laptop pushes the session tau with `up`")
    recorded = path.with_name(path.name + ".sha256").read_text(encoding="utf-8").split()[0]
    if sha256_file(path) != recorded:
        die(f"{path} changed since its sha256 was written")
    record = json.loads(path.read_text(encoding="utf-8"))
    cid = candidate["candidate_id"]
    if cid not in record["candidates"]:
        die(f"{path} has no tau for {cid}")
    return float(record["taus"][record["candidates"].index(cid)])


def session_rows(paths: dict[str, Any]) -> list[dict[str, Any]]:
    """Every input the pod holds this session, as (state_id, problem, solution) rows."""
    rows = []
    for stage in STAGES:
        path = paths["stage_inputs"][stage]
        if path.is_file():
            rows += r2.read_rows(path)
    for path in paths["x1"]:
        if path.is_file():
            rows += [
                {"state_id": r["state_id"], "problem": r["x0"], "solution": r["text"]}
                for r in r2.read_rows(path)
            ]
    return rows


# ---------------------------------------------------------------------------------------
# the Exp 2 loop's configuration: L_in 4,096 and the B the loop precheck measures
# ---------------------------------------------------------------------------------------


def load_loop(path: Path | None = None) -> dict[str, Any]:
    loop = json.loads(Path(path or LOOP_CONFIG).read_text(encoding="utf-8"))
    if type(loop.get("max_input_tokens")) is not int or not loop.get("batch_grid"):
        die("configs/math/exp2_loop.json needs max_input_tokens and batch_grid")
    return loop


def loop_candidate(
    c: dict[str, Any], batch_size: int, loop: dict[str, Any], length: int | None = None
) -> dict[str, Any]:
    """The candidate with the loop's L_in (``length``, else the configured one) and
    ``batch_size``; nothing else changes."""
    spec = {**c["spec"]}
    spec["settings"] = {
        **spec["settings"],
        "batch_size": batch_size,
        "max_input_tokens": length or loop["max_input_tokens"],
    }
    return {**c, "spec": spec}


def loop_lengths(loop: dict[str, Any], candidate_id: str) -> list[int]:
    """The loop L_in values to try for this V, largest first (2026-10-13: R1-Distill-Llama
    4,096 / 3,072 / 2,048); the configured L_in alone for the others."""
    return list(
        loop.get("max_input_tokens_order", {}).get(candidate_id, [loop["max_input_tokens"]])
    )


def loopfit_path(paths: dict[str, Any], candidate_id: str) -> Path:
    return paths["out"] / "loopfit" / f"{candidate_id}.json"


def loop_shape(paths: dict[str, Any], candidate_id: str, loop: dict[str, Any]) -> tuple[int, int]:
    """The loop's B and L_in of this session (``loopfit``; the configured L_in when the record
    predates per-V L_in), or stop."""
    path = loopfit_path(paths, candidate_id)
    record = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}
    if not record.get("passed") or type(record.get("batch_size")) is not int:
        print(f"{candidate_id}: no passed loop precheck in this session ({path})", file=sys.stderr)
        raise SystemExit(EXIT["no_check"])
    return record["batch_size"], int(record.get("L_in") or loop["max_input_tokens"])


def loop_batch_size(paths: dict[str, Any], candidate_id: str) -> int:
    """The loop's B of this session (``loopfit``), or stop."""
    return loop_shape(paths, candidate_id, {"max_input_tokens": 0})[0]


def loop_spec(paths: dict[str, Any], c: dict[str, Any], loop: dict[str, Any]) -> Any:
    cid = c["candidate_id"]
    size, length = loop_shape(paths, cid, loop)
    return r2.spec_for(loop_candidate(c, size, loop, length), paths["models"] / cid)


def loopfit(
    base: Path, candidate_id: str, loop_path: Path | None = None, length: int | None = None
) -> dict[str, Any]:
    """The precheck at the loop shape (2026-10-13) and the loop's B; ``length``
    is one of the V's loop L_in values (``loopselect``), else the configured one."""
    import tempfile

    import torch

    c = load_v(candidate_id)
    loop = load_loop(loop_path)
    paths = layout(base)
    src = paths["models"] / candidate_id
    length = length or loop["max_input_tokens"]
    if length not in loop_lengths(loop, candidate_id):
        die(f"{candidate_id}: L_in {length} is not one of its loop values")
    record: dict[str, Any] = {
        "candidate_id": candidate_id,
        "environment": r2.environment(),
        "L_in": length,
        "step2_batch_size": c["spec"]["settings"]["batch_size"],
    }
    paths["tmp"].mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=paths["tmp"]) as tmp:
        dst = Path(tmp) / candidate_id
        record["tiny"] = r2.build_tiny(src, dst, c)
        r2.write_manifest(dst, c["spec"])
        device = "cuda" if torch.cuda.is_available() else "cpu"

        def make(input_tokens: int, **changes: Any) -> Any:
            return build_verifier(
                r2.spec_for(
                    c,
                    dst,
                    batch_size=r2.TINY["batch_size"],
                    max_input_tokens=input_tokens,
                    max_new_tokens=r2.TINY["max_new_tokens"],
                    device_map=device,
                    **changes,
                )
            )

        probe = make(length)
        probe.load()
        snug_length = r2.toy_shape(c, lambda: probe)
        probe.close()
        verifier = make(length)
        record.update(r2.tiny_checks(verifier))  # at the loop L_in: repeat, alone vs batch
        verifier.close()
        snug, wide = make(snug_length, dtype="float32"), make(length, dtype="float32")
        record["padding"] = {
            **r2.padding_check(snug.score_batch(r2.TOY), wide.score_batch(r2.TOY)),
            "extra_padding": length - snug_length,
        }
        snug.close()
        wide.close()
    record["batch_invariant"] = (
        record["finite_unit"]
        and not record["repeat"]["n_differing"]
        and not record["alone_vs_batch"]["n_differing"]
    )
    record["padding_transparent"] = record["padding"]["transparent"]
    tries, chosen = [], None
    if record["batch_invariant"] and record["padding_transparent"]:
        smallest = int(loop.get("min_batch_size", 1))
        grid = [b for b in loop["batch_grid"] if smallest <= b <= record["step2_batch_size"]]
        for size in grid:
            try:
                fit = r2.fit_check(loop_candidate(c, size, loop, length), src)
            except torch.OutOfMemoryError as exc:
                fit = {"fits": False, "error": f"{type(exc).__name__}: {exc}"}
            finally:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            tries.append({"batch_size": size, **fit})
            if fit.get("fits") is True:
                chosen = size
                break
            if "skipped" in fit:  # no GPU: nothing can be measured here
                break
    record.update(tries=tries, batch_size=chosen, measured_at=utc_now())
    record["passed"] = chosen is not None
    write_json(loopfit_path(paths, candidate_id), record)
    print(
        json.dumps(
            {
                "candidate_id": candidate_id,
                "L_in": length,
                "batch_invariant": record["batch_invariant"],
                "padding_transparent": record["padding_transparent"],
                "tried": [(t["batch_size"], t.get("fits")) for t in tries],
                "batch_size": chosen,
            }
        ),
        flush=True,
    )
    if not record["batch_invariant"]:
        raise SystemExit(EXIT["precheck_batch_invariance"])
    if not record["padding_transparent"]:
        raise SystemExit(EXIT["padding"])
    if chosen is None:
        raise SystemExit(EXIT["fit"])
    return record


# ---------------------------------------------------------------------------------------
# preflight
# ---------------------------------------------------------------------------------------


def preflight(base: Path, candidate_id: str) -> dict[str, Any]:
    c = load_v(candidate_id)
    paths = layout(base)
    rows = session_rows(paths)
    if not rows:
        die("no step 6 input on the pod: run `up` on the laptop")
    verifier = build_verifier(r2.spec_for(c, paths["models"] / candidate_id))
    record = r2.preflight_record(c, verifier, rows, stage="step06")
    record["inputs"] = {
        p.name: sha256_file(p)
        for p in [*paths["stage_inputs"].values(), *paths["x1"]]
        if p.is_file()
    }
    write_json(paths["preflight"] / f"{candidate_id}.json", record)
    return r2.preflight_verdict(record)


# ---------------------------------------------------------------------------------------
# the 203-state check, bitwise or by the math alternative criterion
# ---------------------------------------------------------------------------------------


def judge203(
    ids: list[str],
    reference: list[dict[str, Any]],
    again: list[dict[str, Any]],
    tau: float,
    *,
    config_hash: str,
    reference_config_hash: str,
    comparable: bool | None = None,
) -> dict[str, Any]:
    """generative variant and its math alternative (2026-10-12, enabled 10-13).

    ``comparable`` (the loop configuration): the configurations differ only where the design lets a
    stage adjust them; by default the two configuration hashes must be equal."""
    compared = r2.compare(reference, again)
    same_tokens = [
        a["generated_token_ids"] == b["generated_token_ids"]
        and a["forced_token_ids"] == b["forced_token_ids"]
        for a, b in zip(reference, again, strict=True)
    ]
    flip = [
        (a["error_score"] >= tau) != (b["error_score"] >= tau)
        for a, b in zip(reference, again, strict=True)
    ]
    differing = []
    for d in compared["differing"]:
        i = d["index"]
        differing.append(
            {
                "state_id": ids[i],
                "first_divergent_token": d["first_divergent_token"],
                "tokens_identical": same_tokens[i],
                "score_reference": reference[i]["error_score"],
                "score_now": again[i]["error_score"],
                "score_gap": d["score_gap"],
                "flip_at_tau": flip[i],
            }
        )
    n = len(ids)
    share = sum(same_tokens) / n if n else 0.0
    same_config = config_hash == reference_config_hash if comparable is None else comparable
    bitwise = same_config and compared["n_differing"] == 0
    alternative = same_config and not any(flip) and share >= ALT_IDENTICAL_SHARE
    criterion = (
        "bitwise"
        if bitwise
        else "alternative"
        if alternative
        else "configuration"
        if not same_config
        else "failed"
    )
    return {
        "n": n,
        "tau": tau,
        "config_hash": config_hash,
        "reference_config_hash": reference_config_hash,
        "n_differing": compared["n_differing"],
        "n_tokens_identical": sum(same_tokens),
        "share_identical": share,
        "flips": sum(flip),
        "max_score_gap": compared["max_score_gap"],
        "differing": differing,
        "criterion": criterion,
        "passed": criterion in ("bitwise", "alternative"),
        "rule": "bitwise: generated and forced token ids, verdict logits and score all identical; "
        "alternative (II.10 math entry, 2026-10-12, enabled 2026-10-13): no flip at "
        f"tau and >= {ALT_IDENTICAL_SHARE:.0%} of the items with identical token sequences",
    }


def check_record_path(paths: dict[str, Any], candidate_id: str, loop: bool) -> Path:
    return paths["check203_out"] / f"{candidate_id}{'.loop' if loop else ''}.json"


def check_reference(
    paths: dict[str, Any], candidate_id: str
) -> tuple[list[str], dict[str, Any], dict[str, Any]]:
    """The 203 state ids, their public texts and step 2's screening rows of this V."""
    ids = json.loads(paths["check203"].read_text(encoding="utf-8"))["state_ids"]
    screen = {r["state_id"]: r for r in r2.read_rows(paths["screen_inputs"])}
    reference_file = paths["reference"] / f"{candidate_id}.jsonl"
    if not reference_file.is_file():
        die(f"{candidate_id}: no step 2 screening rows to check against ({reference_file})")
    reference = {r["state_id"]: r for r in r2.read_rows(reference_file)}
    if any(sid not in reference or sid not in screen for sid in ids):
        die(f"{candidate_id}: the screening rows do not cover the 203 states")
    return ids, screen, reference


def adjusted_against_step2(
    verifier: Any, step2: Any, reference_hash: str
) -> tuple[dict[str, Any], bool]:
    """What differs from step 2 in the settings the design lets a stage adjust, and whether nothing
    else differs (and step 2's own configuration is the reference's)."""
    mine = verifier.config.logical_configuration()
    theirs = step2.config.logical_configuration()
    adjusted = {k: {"step2": theirs[k], "loop": mine[k]} for k in LOOP_ADJUSTABLE}
    comparable = step2.config_hash == reference_hash and all(
        mine[k] == theirs[k] for k in mine if k not in LOOP_ADJUSTABLE
    )
    return adjusted, comparable


def check203(base: Path, candidate_id: str, *, loop: bool = False) -> dict[str, Any]:
    c = load_v(candidate_id)
    paths = layout(base)
    ids, screen, reference = check_reference(paths, candidate_id)
    tau = read_tau(paths, c)
    step2 = build_verifier(r2.spec_for(c, paths["models"] / candidate_id))
    reference_hash = reference[ids[0]]["config_hash"]
    comparable, adjusted = None, None
    if loop:
        verifier = build_verifier(loop_spec(paths, c, load_loop()))
        adjusted, comparable = adjusted_against_step2(verifier, step2, reference_hash)
    else:
        verifier = step2
    verifier.load()
    started = time.monotonic()
    again = verifier.score_batch([(screen[s]["problem"], screen[s]["solution"]) for s in ids])
    elapsed = time.monotonic() - started
    config_hash = verifier.config_hash
    verifier.close()
    result = judge203(
        ids,
        [reference[s] for s in ids],
        again,
        tau,
        config_hash=config_hash,
        reference_config_hash=reference_hash,
        comparable=comparable,
    )
    record = {
        "candidate_id": candidate_id,
        "configuration": "exp2_loop" if loop else "step2",
        "adjusted": adjusted,
        "environment": r2.environment(),
        "elapsed_s": round(elapsed, 1),
        **result,
        "checked_at": utc_now(),
    }
    write_json(check_record_path(paths, candidate_id, loop), record)
    keys = ("criterion", "n_differing", "share_identical", "flips", "max_score_gap", "elapsed_s")
    print(json.dumps({"candidate_id": candidate_id, **{k: record[k] for k in keys}}))
    if record["criterion"] == "alternative":
        print(
            f"{candidate_id}: bitwise identity failed on {record['n_differing']} of 203; the "
            "alternative criterion passed (II.10 math entry): continuing, recorded in "
            f"out/check203/{check_record_path(paths, candidate_id, loop).name}",
            flush=True,
        )
    if record["criterion"] == "configuration":
        raise SystemExit(EXIT["check203_configuration"])
    if record["criterion"] == "failed":
        raise SystemExit(EXIT["check203_alternative"])
    return record


# ---------------------------------------------------------------------------------------
# a V's loop L_in when it has several values to try (2026-10-13: R1-Distill-Llama)
# ---------------------------------------------------------------------------------------


def read_record(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}


def run_self(argv: list[str]) -> int:
    """This worker in its own process (each attempt loads its models afresh)."""
    script = str(Path(__file__).resolve())
    return subprocess.run([sys.executable, "-u", script, *argv], check=False).returncode  # noqa: S603


def loopselect(base: Path, candidate_id: str, *, run: Any = None) -> dict[str, Any]:
    """The V's loop L_in: the largest of its values (``max_input_tokens_order``) at which the
    loop precheck fits a B (>= ``min_batch_size``, from its step 2 B down the grid) and the
    203-state check in that configuration agrees. Each value is tried with ``loopfit
    --l-in`` and ``check203 --loop``, each in its own process; an attempt this session already
    made at that L_in (``out/loopfit``, ``out/check203``) is reused, not repeated (both are
    deterministic). Every attempt's records are kept in ``out/loopselect/<id>/``; the passing
    one is left as the session's loop records, which the loop reads. None passes: exit 15 (13
    when no value fits a B)."""
    run = run or run_self
    load_v(candidate_id)
    loop = load_loop()
    paths = layout(base)
    folder = paths["out"] / "loopselect" / candidate_id
    fit_file = loopfit_path(paths, candidate_id)
    check_file = check_record_path(paths, candidate_id, True)
    common = ["--base", str(base), "--candidate", candidate_id]
    attempts, chosen = [], None
    for length in loop_lengths(loop, candidate_id):
        kept_fit = folder / f"L{length}.loopfit.json"
        kept_check = folder / f"L{length}.check203.json"
        fit, check = read_record(kept_fit), read_record(kept_check)
        if not fit and read_record(fit_file).get("L_in") == length:  # made before loopselect
            fit = read_record(fit_file)
            write_json(kept_fit, fit)
            adjusted = read_record(check_file).get("adjusted") or {}
            if (adjusted.get("max_input_tokens") or {}).get("loop") == length and (
                adjusted.get("batch_size") or {}
            ).get("loop") == fit.get("batch_size"):
                check = read_record(check_file)
                write_json(kept_check, check)
        if not fit:
            code = run(["loopfit", *common, "--l-in", str(length)])
            if code not in (0, EXIT["fit"]):
                raise SystemExit(code)
            fit = read_record(fit_file)
            write_json(kept_fit, fit)
        if fit.get("passed") and not check:
            code = run(["check203", "--loop", *common])
            if code not in (0, EXIT["check203_alternative"]):
                raise SystemExit(code)
            check = read_record(check_file)
            write_json(kept_check, check)
        attempt = {
            "L_in": length,
            "batch_size": fit.get("batch_size"),
            "fits": bool(fit.get("passed")),
            "check203": check.get("criterion"),
            "passed": bool(fit.get("passed") and check.get("passed")),
        }
        attempts.append(attempt)
        print(json.dumps({"candidate_id": candidate_id, **attempt}), flush=True)
        if attempt["passed"]:
            chosen = attempt
            write_json(fit_file, fit)  # the session's loop records: this attempt's
            write_json(check_file, check)
            break
    record = {
        "candidate_id": candidate_id,
        "rule": "the largest loop L_in of max_input_tokens_order at which the loop precheck "
        "fits a B >= min_batch_size and the 203-state check in that configuration agrees "
        "(configs/math/exp2_loop.json, 2026-10-13; II.10)",
        "attempts": attempts,
        "L_in": chosen["L_in"] if chosen else None,
        "batch_size": chosen["batch_size"] if chosen else None,
        "passed": chosen is not None,
        "environment": r2.environment(),
        "selected_at": utc_now(),
    }
    write_json(paths["out"] / "loopselect" / f"{candidate_id}.json", record)
    if chosen is None:
        raise SystemExit(
            EXIT["check203_alternative"] if any(a["fits"] for a in attempts) else EXIT["fit"]
        )
    return record


# ---------------------------------------------------------------------------------------
# R1-Distill-Llama's loop diagnosis (2026-10-14)
# ---------------------------------------------------------------------------------------

#: torch SDPA backends the diagnosis may force on the prefill (``loopdiag --prefill-kernel``).
PREFILL_KERNELS = ("efficient", "flash", "cudnn", "math")


@contextlib.contextmanager
def prefill_kernel(name: str | None) -> Iterator[dict[str, int]]:
    """Run transformers' sdpa path -- every attention call except the one-token grouped decode
    steps of ``generative.grouped_decode_attention``, i.e. the prefill -- under one forced torch
    SDPA backend; the decode and forced steps are untouched. Diagnosis only (``loopdiag``), never
    a formal score: the scoring path itself is not changed."""
    calls = {"prefill_calls": 0}
    if name is None:
        yield calls
        return
    import importlib

    from torch.nn.attention import SDPBackend, sdpa_kernel

    backend = {
        "efficient": SDPBackend.EFFICIENT_ATTENTION,
        "flash": SDPBackend.FLASH_ATTENTION,
        "cudnn": SDPBackend.CUDNN_ATTENTION,
        "math": SDPBackend.MATH,
    }[name]
    sdpa = importlib.import_module("transformers.integrations.sdpa_attention")
    original = sdpa.sdpa_attention_forward

    def forced(*args: Any, **kwargs: Any) -> Any:
        calls["prefill_calls"] += 1
        with sdpa_kernel([backend]):
            return original(*args, **kwargs)

    sdpa.sdpa_attention_forward = forced
    try:
        yield calls
    finally:
        sdpa.sdpa_attention_forward = original


def loopdiag_path(
    paths: dict[str, Any], candidate_id: str, length: int, size: int, kernel: str | None
) -> Path:
    name = f"L{length}_B{size}_{kernel or 'default'}.json"
    return paths["out"] / "loopdiag" / candidate_id / name


def loopdiag(
    base: Path,
    candidate_id: str,
    length: int,
    size: int,
    *,
    kernel: str | None = None,
) -> dict[str, Any]:
    """The 203-state check at a given L_in and B (L_gen, prompts, stop string and every other
    setting as in step 2), optionally with the prefill under one forced SDPA backend, compared
    with step 2's screening rows by the session check's own rule. A diagnosis: it never stands in
    for a session's check (its record lives in ``out/loopdiag``) and exits 0 whatever the
    comparison shows. L_in = step 2's and B = step 2's is step 2's configuration through the
    loop's code path."""
    if kernel is not None and kernel not in PREFILL_KERNELS:
        die(f"unknown prefill kernel {kernel!r}")
    c = load_v(candidate_id)
    paths = layout(base)
    ids, screen, reference = check_reference(paths, candidate_id)
    tau = read_tau(paths, c)
    step2 = build_verifier(r2.spec_for(c, paths["models"] / candidate_id))
    reference_hash = reference[ids[0]]["config_hash"]
    spec = r2.spec_for(loop_candidate(c, size, load_loop(), length), paths["models"] / candidate_id)
    verifier = build_verifier(spec)
    adjusted, comparable = adjusted_against_step2(verifier, step2, reference_hash)
    settings = spec.settings
    verifier.load()
    started = time.monotonic()
    with prefill_kernel(kernel) as calls:
        again = verifier.score_batch([(screen[s]["problem"], screen[s]["solution"]) for s in ids])
    elapsed = time.monotonic() - started
    config_hash = verifier.config_hash
    verifier.close()
    if kernel is not None and not calls["prefill_calls"]:
        die(f"--prefill-kernel {kernel}: no prefill went through it (attention implementation?)")
    result = judge203(
        ids,
        [reference[s] for s in ids],
        again,
        tau,
        config_hash=config_hash,
        reference_config_hash=reference_hash,
        comparable=comparable,
    )
    first = sorted(
        d["first_divergent_token"]
        for d in result["differing"]
        if d["first_divergent_token"] is not None
    )
    record = {
        "candidate_id": candidate_id,
        "configuration": "loop_diagnosis",
        "L_in": length,
        "batch_size": size,
        "cache_length": length + settings["max_new_tokens"] + settings["max_forced_tokens"],
        "prefill_kernel": kernel or "default",
        **calls,
        "adjusted": adjusted,
        "environment": r2.environment(),
        "elapsed_s": round(elapsed, 1),
        "first_divergent_token": {
            "n": len(first),
            "min": first[0] if first else None,
            "median": first[len(first) // 2] if first else None,
        },
        **result,
        "checked_at": utc_now(),
    }
    write_json(loopdiag_path(paths, candidate_id, length, size, kernel), record)
    keys = ("L_in", "batch_size", "cache_length", "prefill_kernel", "criterion", "n_differing")
    keys += ("share_identical", "flips", "max_score_gap", "elapsed_s")
    print(json.dumps({"candidate_id": candidate_id, **{k: record[k] for k in keys}}), flush=True)
    return record


def loopbisect(
    base: Path,
    candidate_id: str,
    lo: int,
    hi: int,
    step: int,
    size: int,
    *,
    run: Any = None,
) -> dict[str, Any]:
    """The largest loop L_in on the ``step`` grid between ``lo`` and ``hi`` whose 203-state check
    is bitwise equal to step 2 (2026-10-15). ``lo``
    is known to pass and ``hi`` known to fail (the diagnosis of 2026-10-15: 1,344 and 1,920 at
    B 32), so only the grid strictly between is probed, by bisection; each probe is a full
    ``loopdiag`` at (L_in, B) in its own process, its record in ``out/loopdiag``. A diagnosis:
    exit 0 whatever the probes show; an error of a probe stops it with that probe's code."""
    if not lo < hi or (hi - lo) % step:
        die(f"loopbisect needs lo < hi on the {step} grid, got {lo}, {hi}")
    run = run or run_self
    load_v(candidate_id)
    paths = layout(base)
    probes = []
    low, high = lo, hi
    while high - low > step:
        mid = low + step * ((high - low) // step // 2)
        path = loopdiag_path(paths, candidate_id, mid, size, None)
        record = read_record(path)
        if not record:
            argv = ["loopdiag", "--base", str(base), "--candidate", candidate_id]
            code = run([*argv, "--l-in", str(mid), "--batch-size", str(size)])
            if code != 0:
                raise SystemExit(code)
            record = read_record(path)
        bitwise = record.get("criterion") == "bitwise"
        probes.append(
            {
                "L_in": mid,
                "criterion": record.get("criterion"),
                "n_differing": record.get("n_differing"),
                "flips": record.get("flips"),
                "bitwise": bitwise,
            }
        )
        print(json.dumps({"candidate_id": candidate_id, "bisect": probes[-1]}), flush=True)
        low, high = (mid, high) if bitwise else (low, mid)
    result = {
        "candidate_id": candidate_id,
        "rule": "largest L_in on the grid whose 203-state check is bitwise equal to step 2 "
        "(2026-10-15); the endpoints are the diagnosis's: lo passes, hi fails",
        "grid": {"lo": lo, "hi": hi, "step": step, "batch_size": size},
        "probes": probes,
        "largest_bitwise_L_in": low,
        "smallest_failing_L_in": high,
        "above_step_2": low > lo,
        "environment": r2.environment(),
        "finished_at": utc_now(),
    }
    write_json(paths["out"] / "loopbisect" / f"{candidate_id}.json", result)
    keys = ("largest_bitwise_L_in", "smallest_failing_L_in", "above_step_2")
    print(json.dumps({"candidate_id": candidate_id, **{k: result[k] for k in keys}}), flush=True)
    return result


def require_check(
    paths: dict[str, Any], candidate_id: str, config_hash: str, *, loop: bool = False
) -> dict[str, Any]:
    path = check_record_path(paths, candidate_id, loop)
    record = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}
    if not record.get("passed") or record.get("config_hash") != config_hash:
        print(
            f"{candidate_id}: the 203-state check has not passed in this configuration",
            file=sys.stderr,
        )
        raise SystemExit(EXIT["no_check"])
    return record


# ---------------------------------------------------------------------------------------
# scoring a stage
# ---------------------------------------------------------------------------------------


def score(base: Path, candidate_id: str, stage: str) -> dict[str, Any]:
    if stage not in STAGES:
        die(f"unknown stage {stage!r}")
    c = load_v(candidate_id)
    paths = layout(base)
    inputs = paths["stage_inputs"][stage]
    if not inputs.is_file():
        die(f"{inputs} not on the pod")
    rows = r2.read_rows(inputs)
    spec = r2.spec_for(c, paths["models"] / candidate_id)
    verifier = build_verifier(spec)
    verifier.load()
    check = require_check(paths, candidate_id, verifier.config_hash)
    dirs = r2.stage_dirs(paths, stage)
    started_at, t0 = utc_now(), time.monotonic()
    table, detail_file, start = r2.score_rows(verifier, spec, rows, dirs, candidate_id)
    config_hash = verifier.config_hash
    verifier.close()
    details = list(read_jsonl(detail_file))
    record = {
        "candidate_id": candidate_id,
        "stage": stage,
        "model_id": spec.model_id,
        "revision": spec.revision,
        "n": len(rows),
        "inputs_sha256": sha256_file(inputs),
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
        "check203_criterion": check["criterion"],
        "environment": r2.environment(),
    }
    write_json(dirs["meta"] / f"{candidate_id}.score.json", record)
    keys = ("candidate_id", "stage", "n", "elapsed_s", "truncated", "table_sha256")
    print(json.dumps({k: record[k] for k in keys}), flush=True)
    return record


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument(
        "command",
        choices=(
            "fetch",
            "speed",
            "precheck",
            "preflight",
            "check203",
            "score",
            "loopfit",
            "loopselect",
            "loopdiag",
            "loopbisect",
        ),
    )
    ap.add_argument("--l-in", type=int, help="loopfit: one of the V's loop L_in values")
    ap.add_argument("--batch-size", type=int, help="loopdiag / loopbisect: B")
    ap.add_argument("--lo", type=int, help="loopbisect: L_in known to pass")
    ap.add_argument("--hi", type=int, help="loopbisect: L_in known to fail")
    ap.add_argument("--step", type=int, default=64, help="loopbisect: the L_in grid")
    ap.add_argument("--prefill-kernel", choices=PREFILL_KERNELS, help="loopdiag: forced SDPA")
    ap.add_argument("--base", type=Path, required=True)
    ap.add_argument("--candidate", required=True)
    ap.add_argument("--stage", choices=STAGES)
    ap.add_argument("--no-weights", action="store_true", help="fetch: config, tokenizer, template")
    ap.add_argument(
        "--loop", action="store_true", help="check203 in the Exp 2 loop's configuration"
    )
    args = ap.parse_args(argv)
    os.environ.setdefault("HF_HUB_DISABLE_IMPLICIT_TOKEN", "1")
    load_v(args.candidate)
    if args.command == "score" and not args.stage:
        ap.error("score needs --stage")
    if args.command == "fetch":
        r2.fetch(args.base, args.candidate, weights=not args.no_weights)
    elif args.command == "speed":
        r2.speed(args.base, args.candidate)
    elif args.command == "precheck":
        r2.precheck(args.base, args.candidate)
    elif args.command == "preflight":
        preflight(args.base, args.candidate)
    elif args.command == "check203":
        check203(args.base, args.candidate, loop=args.loop)
    elif args.command == "loopfit":
        loopfit(args.base, args.candidate, length=args.l_in)
    elif args.command == "loopselect":
        loopselect(args.base, args.candidate)
    elif args.command == "loopbisect":
        if not (args.lo and args.hi and args.batch_size):
            ap.error("loopbisect needs --lo, --hi and --batch-size")
        loopbisect(args.base, args.candidate, args.lo, args.hi, args.step, args.batch_size)
    elif args.command == "loopdiag":
        if not args.l_in or not args.batch_size:
            ap.error("loopdiag needs --l-in and --batch-size")
        loopdiag(args.base, args.candidate, args.l_in, args.batch_size, kernel=args.prefill_kernel)
    else:
        score(args.base, args.candidate, args.stage)
    return 0


if __name__ == "__main__":
    sys.exit(main())
