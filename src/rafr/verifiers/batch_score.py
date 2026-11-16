"""Batch scoring on the rented GPU: ``to_score.jsonl`` -> the scores table.

    python -m rafr.verifiers.batch_score \\
        --input data/to_score/d_gate.jsonl \\
        --config-id sg_2b.native \\
        --scores-dir data/scores

Input is one JSON object per line with ``state_id`` and ``text``; output is the scores table
this package already owns (:func:`rafr.verifiers.write_scores_table`), one
:class:`rafr.contracts.schemas.Score` row per state. Nothing about the format is new here.

Resumable: the table is append-only and already-scored ``state_id``s are skipped, so a pod
that dies halfway (or an hourly budget that runs out) costs only the states it had not
reached. ``--no-resume`` rescores everything, appending fresh rows; the loader keeps the
last row per state.

**Formal scoring goes through ``--config-id``**: the one factory,
:func:`rafr.verifiers.factory.make_verifier`, i.e. the step 2 probe's own scoring code checked
against the probe's records; rows then carry ``prompt_hash`` and ``sub_scores`` like the probe's
table. ``--candidate`` still builds the registry ENCODERS for smoke tests, but refuses every
registry guard LM: that backend tokenises a chat-template rendering with a second BOS
(point 4) and is off the formal path. ``--impl`` objects that turn
out to be the registry guard wrapper are refused too.

Formal scoring refuses a verifier whose weights were never pinned by
the RunPod driver (not released), because ``model_version`` would then not identify what produced
the numbers. ``--allow-unpinned`` is for smoke tests only.

D_gate_reserve is not scored before the first gate failure (paper Sec. 4.3): any input that holds a
reserve state raises :class:`rafr.data.labels.LabelAccessError` until
``frozen/gate_attempts.json`` records a failed attempt. To know which states are the reserve's,
the split file ``data/splits/d_gate_reserve.jsonl`` has to be on this machine; without it the
run refuses rather than scoring blind (``--splits-dir`` / ``--gate-attempts`` point elsewhere).
The check applies to ``--self-check --input`` too: that scores the input as well.

``--selection`` is for step 2 (verifier selection, 2026-09-24): selection
may score D_fit and the unlabeled pool only, so with the flag any state of D_gate,
D_gate_reserve or D_Exp1 raises :class:`rafr.data.labels.LabelAccessError` -- whatever the
gate record says. It needs all three split files and refuses to run without them. Without the
flag the formal step-6 rules apply: D_gate and D_Exp1 are scored there.
"""

from __future__ import annotations

import argparse
import importlib
import sys
import time
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import Any

from rafr.contracts.schemas import Score, Split
from rafr.core import paths
from rafr.data.labels import LabelAccessError, reserve_unlocked
from rafr.data.splits import load_split
from rafr.logging.writer import read_jsonl
from rafr.verifiers import load_scores_table, scores_table_path, write_scores_table
from rafr.verifiers.candidates.base import (
    UNPINNED,
    PaddingComparison,
    VerifierConfigError,
    compare_scores,
    padding_mode,
)
from rafr.verifiers.candidates.registry import CANDIDATES, GUARD_LM, build, names
from rafr.verifiers.stub import is_stub

DEFAULT_SCORES_DIR = Path("data/scores")
#: Rows written per flush. Independent of the inference batch size; only bounds how much a
#: crash can lose.
WRITE_CHUNK = 64


def load_items(path: Path, limit: int | None = None) -> list[tuple[str, str]]:
    """``[(state_id, text), ...]`` in file order. Duplicated state_ids are an input error."""
    out: list[tuple[str, str]] = []
    seen: set[str] = set()
    for i, row in enumerate(read_jsonl(path), start=1):
        try:
            state_id = str(row["state_id"])
            text = row["text"]
        except KeyError as exc:
            raise SystemExit(f"{path}:{i}: missing field {exc}") from None
        if not isinstance(text, str):
            raise SystemExit(f"{path}:{i}: 'text' must be a string")
        if state_id in seen:
            raise SystemExit(f"{path}:{i}: duplicate state_id {state_id!r}")
        seen.add(state_id)
        out.append((state_id, text))
        if limit is not None and len(out) >= limit:
            break
    return out


def reserve_ids(split_dir: Path) -> set[str] | None:
    """State ids (and x0 ids) of D_gate_reserve, or None when its split file is absent.

    One state per x0 whose id is ``<x0_id>-g1`` (rafr.generation.g_fixed.state_id_for); both
    forms are returned, so an input keyed either way is caught.
    """
    if not (Path(split_dir) / f"{Split.D_GATE_RESERVE.value}.jsonl").exists():
        return None
    x0s = load_split(Split.D_GATE_RESERVE, Path(split_dir))
    return set(x0s) | {f"{x}-g1" for x in x0s}


def refuse_locked_reserve(
    items: Sequence[tuple[str, str]], *, split_dir: Path, gate_attempts_path: Path
) -> None:
    """paper Sec. 4.3: raise if ``items`` hold a D_gate_reserve state and the reserve is still
    locked."""
    reserve = reserve_ids(split_dir)
    if reserve is None:
        raise SystemExit(
            f"cannot tell whether the input holds D_gate_reserve states: "
            f"{Path(split_dir) / (Split.D_GATE_RESERVE.value + '.jsonl')} not found. Copy "
            "data/splits/ and frozen/gate_attempts.json to this machine, or point --splits-dir "
            "/ --gate-attempts at them. The reserve may not be scored before the first gate "
            "failure (II.6.1), so this run does not score blind."
        )
    hit = [sid for sid, _ in items if sid in reserve]
    if hit and not reserve_unlocked(gate_attempts_path):
        raise LabelAccessError(
            f"{len(hit)} input state(s) belong to D_gate_reserve (e.g. {hit[:3]}), which may "
            "not be scored before the first gate attempt has been recorded as failed "
            f"(II.6.1; {gate_attempts_path})"
        )


# Splits step 2 (verifier selection) may neither score nor read: allows D_fit
#: and the unlabeled pool only. D_Exp1 matters as much as the two gate splits: it holds
#: E-UNSAFE states only, so scoring it for selection would leak its y* into the choice.
SELECTION_FORBIDDEN: tuple[Split, ...] = (Split.D_GATE, Split.D_GATE_RESERVE, Split.D_EXP1)


def selection_forbidden_ids(split_dir: Path) -> dict[Split, set[str]]:
    """State ids (and x0 ids) of every split selection may not touch, per split.

    Raises SystemExit when a split file is missing: without it the run cannot know what it
    may not score, and it does not score blind.
    """
    out: dict[Split, set[str]] = {}
    for sp in SELECTION_FORBIDDEN:
        path = Path(split_dir) / f"{sp.value}.jsonl"
        if not path.exists():
            raise SystemExit(
                f"--selection: cannot tell whether the input holds {sp.value} states, {path} "
                "not found. Step 2 may score D_fit and the unlabeled pool only (App. D), so "
                "this run does not score blind; point --splits-dir at the split files."
            )
        x0s = load_split(sp, Path(split_dir))
        out[sp] = set(x0s) | {f"{x}-g1" for x in x0s}
    return out


def refuse_selection_splits(items: Sequence[tuple[str, str]], *, split_dir: Path) -> None:
    """Step 2: raise if ``items`` hold any D_gate, D_gate_reserve or D_Exp1 state."""
    forbidden = selection_forbidden_ids(split_dir)
    for sp, ids in forbidden.items():
        hit = [sid for sid, _ in items if sid in ids]
        if hit:
            raise LabelAccessError(
                f"{len(hit)} input state(s) belong to {sp.value} (e.g. {hit[:3]}); verifier "
                "selection (step 2) may score D_fit and the unlabeled pool only (App. D, v0.9)"
            )


def make_verifier(
    *,
    candidate: str | None,
    impl: str | None,
    batch_size: int | None,
    verifier_id: str | None,
    config_id: str | None = None,
    tolerate_drift: bool = False,
) -> Any:
    """The factory (``config_id``, formal), a registry ENCODER candidate, or any
    ``"package.module:factory"`` (score_spec.md). Registry guard LMs are refused."""
    from rafr.verifiers import factory

    if sum(bool(x) for x in (candidate, impl, config_id)) != 1:
        raise SystemExit("give exactly one of --config-id, --candidate or --impl")
    if config_id:
        if batch_size is not None or verifier_id is not None:
            raise SystemExit(
                "--config-id takes the probe's batch size and id; --batch-size / --verifier-id "
                "would make it score differently from the probe"
            )
        # formal (a cuda record and a CUDA run, point 6) unless a drift is
        # explicitly tolerated, which makes it an engineering run on the record's own device
        return factory.make_verifier(
            config_id, tolerate_drift=tolerate_drift, formal=not tolerate_drift, echo=print
        )
    if candidate:
        if candidate not in CANDIDATES:
            raise SystemExit(f"unknown candidate {candidate!r}; known: {names()}")
        if CANDIDATES[candidate].kind == GUARD_LM:
            raise SystemExit(
                f"{candidate} is a registry guard LM: that backend adds a second BOS to the chat "
                "template's rendering and is off the formal path (decisions.md D-2026-09-24b "
                "point 4). Score guard LMs with --config-id (rafr.verifiers.factory)."
            )
        kwargs: dict[str, Any] = {}
        if batch_size is not None:
            kwargs["inference_batch_size"] = batch_size
        return build(candidate, verifier_id=verifier_id, **kwargs)

    mod, _, fn = str(impl).partition(":")
    if not mod or not fn:
        raise SystemExit(f"--impl must look like 'package.module:factory', got {impl!r}")
    obj = getattr(importlib.import_module(mod), fn)
    return factory.refuse_legacy_guard(obj() if callable(obj) else obj)


def batches(items: Sequence[tuple[str, str]], size: int) -> Iterator[list[tuple[str, str]]]:
    for i in range(0, len(items), size):
        yield list(items[i : i + size])


def run(
    *,
    verifier: Any,
    items: Sequence[tuple[str, str]],
    out_path: Path,
    resume: bool,
    echo: Any,
) -> int:
    """Score every item not already in the table. Returns how many rows were written."""
    done: set[str] = set()
    if resume:
        done = set(load_scores_table(out_path))
        if done:
            echo(f"resume: {len(done)} state(s) already in {out_path}")

    todo = [(sid, text) for sid, text in items if sid not in done]
    echo(f"to score: {len(todo)} of {len(items)}")
    if not todo:
        return 0

    per_call = min(verifier.batch_limit, max(WRITE_CHUNK, verifier.inference_batch_size))
    written = 0
    started = time.monotonic()
    detailed = hasattr(verifier, "score_detailed")  # the factory: prompt hash + sub-scores
    for chunk in batches(todo, per_call):
        texts = [text for _, text in chunk]
        results = verifier.score_detailed(texts) if detailed else verifier.score(texts)
        if len(results) != len(chunk):
            raise VerifierConfigError(
                f"{verifier.verifier_id}: {len(results)} scores for {len(chunk)} texts"
            )
        rows = [
            Score(
                state_id=sid,
                verifier_id=verifier.verifier_id,
                model_version=verifier.model_version,
                prompt_hash=verifier.prompt_hash if detailed else None,
                score=r.score if detailed else r,
                sub_scores=r.sub_scores if detailed else None,
            )
            for (sid, _), r in zip(chunk, results, strict=True)
        ]
        write_scores_table(rows, out_path)
        written += len(rows)
        echo(f"  {written}/{len(todo)} ({time.monotonic() - started:.1f}s)")
    return written


def frozen_threshold(verifier_id: str, path: Path | None = None) -> float | None:
    """tau_i for this verifier from ``frozen/tau.json``, if step 7 has already run.

    At selection time it has not, and that is fine: the padding comparison reports a
    threshold-free worst case as well.
    """
    from rafr.core import paths
    from rafr.validity.tau import load_tau

    p = Path(path) if path is not None else paths.FROZEN_TAU
    if not p.is_file():
        return None
    for fit in load_tau(p).values():
        if fit.verifier_id == verifier_id:
            return float(fit.tau)
    return None


def compare_padding_modes(
    verifier: Any, texts: Sequence[str], threshold: float | None = None
) -> tuple[PaddingComparison, list[float], list[float]]:
    """Score the same texts with fixed shapes and with dynamic padding, on one loaded model.

    Decision (2026-09-19): selection runs with ``RAFR_VERIFIER_DYNAMIC_PADDING=1``
    for speed and the formal steps 6 and 12 run with fixed shapes, so the two modes must be
    shown to agree on the *judgments* before a selection conclusion can be trusted. Any
    flip here means the selection may have to be redone -- report it, do not absorb it.
    """
    items = list(texts)
    with padding_mode(verifier, fixed=True):
        fixed = verifier.score(items)
    with padding_mode(verifier, fixed=False):
        dynamic = verifier.score(items)
    return compare_scores(fixed, dynamic, threshold), fixed, dynamic


def self_check(
    verifier: Any, texts: Sequence[str], echo: Any, threshold: float | None = None
) -> PaddingComparison:
    """What only a real GPU and real weights can settle.

    Two things, neither of which an offline test can falsify:

    1. the label prefix and the label spellings in the registry were read off the model
       cards, not off the weights. The greedy continuation below is decoded *without* the
       teacher-forced prefix, so it shows what the model emits on its own;
    2. whether fixed-shape padding and dynamic padding agree on the judgments (see
       :func:`compare_padding_modes`).

    Nothing here is part of a run.
    """
    backend = verifier.backend
    echo(f"verifier_id     {verifier.verifier_id}")
    echo(f"model_version   {verifier.model_version}")
    echo(f"truncation      {verifier.truncation.describe()}")
    echo(f"score           {verifier.score_semantics}")
    echo(f"normalization   {verifier.normalization}")
    if hasattr(backend, "label_token_ids"):
        for group, ids in zip(verifier.labels, backend.label_token_ids, strict=True):
            decoded = [backend.tokenizer.decode([i]) for i in ids]
            echo(f"  label {group.name:<14} token ids {ids} -> {decoded}")

    echo("")
    echo(f"padding modes compared on {len(texts)} text(s)")
    cmp, fixed, dynamic = compare_padding_modes(verifier, texts, threshold)
    echo(f"  max |fixed - dynamic|      {cmp.max_abs_diff:.3e}")
    echo(f"  mean |fixed - dynamic|     {cmp.mean_abs_diff:.3e}")
    echo(f"  scores that differ at all  {cmp.n_differing}/{cmp.n}")
    if cmp.threshold is None:
        echo("  tau                        not frozen yet (step 7); no per-tau flip count")
    else:
        echo(f"  tau                        {cmp.threshold:.6f}")
        echo(f"  judgments that flip at tau {cmp.n_flips_at_threshold}")
    echo(f"  worst case over ALL tau    {cmp.worst_case_flips} flip(s)")
    if cmp.clean():
        echo("  -> the two modes agree on every judgment for every possible threshold")
    else:
        echo(
            f"  -> NOT CLEAN: some tau near {cmp.worst_case_threshold:.6f} flips "
            f"{cmp.worst_case_flips} judgment(s). Report this: a selection run under "
            "dynamic padding may have to be redone."
        )

    echo("")
    for i, text in enumerate(texts[:SELF_CHECK_SHOW]):
        echo(f"  fixed={fixed[i]:.6f} dynamic={dynamic[i]:.6f}  {text!r}")
        if hasattr(backend, "greedy_head"):
            head = backend.greedy_head(verifier.prepare([text])[0])
            echo(f"    greedy continuation: {head!r}")
    return cmp


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--input", type=Path, help="jsonl with state_id and text")
    p.add_argument(
        "--config-id",
        help="formal: a probe configuration (enc_paradetox, sg_2b.native, ...) through "
        "rafr.verifiers.factory",
    )
    p.add_argument(
        "--tolerate-drift",
        action="store_true",
        help="--config-id: accept direction-check scores that differ from the probe's (recorded)",
    )
    p.add_argument("--candidate", help=f"smoke tests, encoders only: one of {', '.join(names())}")
    p.add_argument("--impl", help="'package.module:factory' (score_spec.md)")
    p.add_argument("--verifier-id", help="override the verifier's own id (names the table)")
    p.add_argument("--scores-dir", type=Path, default=DEFAULT_SCORES_DIR)
    p.add_argument(
        "--out", type=Path, help="explicit output path (default: <scores-dir>/<id>.jsonl)"
    )
    p.add_argument("--batch-size", type=int, help="inference batch size; cannot change a score")
    p.add_argument("--limit", type=int, help="score only the first N input rows (smoke test)")
    p.add_argument("--no-resume", action="store_true", help="rescore states already in the table")
    p.add_argument(
        "--allow-unpinned",
        action="store_true",
        help="allow weights whose revision env/download_models.sh never pinned (smoke tests)",
    )
    p.add_argument(
        "--self-check",
        action="store_true",
        help="label tokens, a greedy continuation, and the fixed-vs-dynamic padding "
        "comparison, then exit. Uses --input when given (first --limit rows, default "
        f"{SELF_CHECK_ROWS}), otherwise two built-in texts",
    )
    p.add_argument(
        "--threshold",
        type=float,
        help="tau for the padding flip count; default: this verifier's entry in "
        "frozen/tau.json if step 7 has run. The threshold-free worst case is reported "
        "either way",
    )
    p.add_argument(
        "--splits-dir",
        type=Path,
        default=None,
        help="where d_gate_reserve.jsonl is (default: data/splits); needed for --input",
    )
    p.add_argument(
        "--gate-attempts",
        type=Path,
        default=None,
        help="the gate attempts record (default: frozen/gate_attempts.json)",
    )
    p.add_argument(
        "--selection",
        action="store_true",
        help="step 2 verifier selection: refuse any D_gate / D_gate_reserve / D_Exp1 state "
        "(needs their split files)",
    )
    p.add_argument("--allow-stub", action="store_true", help=argparse.SUPPRESS)
    return p


def main(argv: Sequence[str] | None = None, echo: Any = None) -> int:
    args = build_parser().parse_args(argv)
    say = echo or print
    if args.config_id and args.self_check:
        raise SystemExit(
            "--self-check is for the registry candidates; --config-id already runs the probe's "
            "preflight (verdict tokens, direction re-score) when it loads"
        )

    verifier = make_verifier(
        candidate=args.candidate,
        impl=args.impl,
        batch_size=args.batch_size,
        verifier_id=args.verifier_id,
        config_id=args.config_id,
        tolerate_drift=args.tolerate_drift,
    )
    if is_stub(verifier.verifier_id) and not args.allow_stub:
        raise SystemExit(f"{verifier.verifier_id} is a stub; formal scoring refuses stubs (II.4.4)")
    if verifier.model_version.endswith(f"@{UNPINNED}") and not args.allow_unpinned:
        raise SystemExit(
            f"{verifier.verifier_id}: weights are not pinned to a commit "
            f"(model_version={verifier.model_version}). Run env/download_models.sh first, or "
            "pass --allow-unpinned for a smoke test."
        )

    guard = dict(
        split_dir=args.splits_dir or paths.DATA_SPLITS,
        gate_attempts_path=args.gate_attempts or paths.FROZEN_GATE_ATTEMPTS,
    )

    def check(rows: Sequence[tuple[str, str]]) -> None:
        if args.selection:
            refuse_selection_splits(rows, split_dir=guard["split_dir"])
        refuse_locked_reserve(rows, **guard)

    if args.self_check:
        # Real texts when there are any: the padding comparison is only informative on the
        # kind of sentences the run will actually score.
        if args.input is not None:
            rows = load_items(args.input, limit=args.limit or SELF_CHECK_ROWS)
            check(rows)
            texts = [t for _, t in rows]
        else:
            texts = list(SELF_CHECK_TEXTS)
        threshold = args.threshold
        if threshold is None:
            threshold = frozen_threshold(verifier.verifier_id)
        cmp = self_check(verifier, texts, say, threshold)
        return 0 if cmp.clean() else 2

    if args.input is None:
        raise SystemExit("--input is required unless --self-check is given")
    items = load_items(args.input, limit=args.limit)
    check(items)
    out_path = args.out or scores_table_path(verifier.verifier_id, args.scores_dir)
    say(f"{verifier.verifier_id} ({verifier.model_version}) -> {out_path}")
    written = run(
        verifier=verifier,
        items=items,
        out_path=Path(out_path),
        resume=not args.no_resume,
        echo=say,
    )
    say(f"wrote {written} row(s)")
    return 0


#: Deliberately mild, and never real experiment data: the self-check is about the plumbing.
#: Used only when --self-check is given without --input.
SELF_CHECK_TEXTS = (
    "You are a complete idiot and everyone knows it.",
    "Thanks for the detailed write-up, this was really helpful.",
)
#: How many input rows --self-check reads when --limit is not given. Enough for the padding
#: comparison to mean something, small enough to be cheap on an 8B.
SELF_CHECK_ROWS = 64
#: How many of them get their scores and greedy continuation printed one by one.
SELF_CHECK_SHOW = 4


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
