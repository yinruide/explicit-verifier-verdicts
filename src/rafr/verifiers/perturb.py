"""Wording perturbations for the design's stability row (step 6).

The design asks for V1 / V2's "intrinsic stability": how often the verdict ``1[s >= tau]`` survives
a change of wording that leaves the meaning alone -- a paraphrase, and an irrelevant prefix. The
texts made here serve that one descriptive number; they never enter tau, the channel, the gate,
Exp 1 or Exp 2.

* **Sample**: 300 states of D_fit + the unlabeled pool, 150 E-UNSAFE and 150 E-SAFE, drawn with
  the seed component ``perturb_sample`` (the draw reads y* of D_fit and the pool, purpose
  ``VERIFIER_DIAGNOSTICS``; the sample file stays on the laptop, labels and all).
* **Prefix**: a fixed table of three neutral, content-free openings
  (``configs/prompts/perturb_prefixes.txt``); each state gets one, by seed, joined to its text by
  one space. Not sent to E: by construction a prefix does not change toxicity.
* **Paraphrase**: one call per state to the controller's backbone with G_fixed's call
  configuration (same model, version, decoding -- reasoning_effort medium, temperature not sent
  -- and the same structured output), prompt ``configs/prompts/perturb_paraphrase.txt``: keep the
  meaning and the tone, change only the wording (2026-09-25: in Exp 2 the verifiers read
  exactly this model's wording). A paraphrase is invalid on G_fixed's rules
  (``generation/validity.py``: refusal, empty, identical to the original state, non-English);
  an invalid one is dropped with its reason and never regenerated. Valid paraphrases go to E.

Both prompt files are hashed into ``configs/prompts/perturb_prompts.lock.json`` before the first
paraphrase is bought; generation refuses to run while a file differs from the lock.

What goes to the pod is :func:`build_pod_input`: ``{state_id, text}`` rows only, the item id
being ``<state_id>#prefix`` / ``<state_id>#paraphrase`` -- never a label.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from rafr.contracts.schemas import PLACEHOLDER, GFixedOutput, SafetyLabel, Split, State
from rafr.core import paths
from rafr.core.seeding import attempt_seed, py_rng
from rafr.generation import g_fixed
from rafr.generation.validity import judge_validity
from rafr.llm.client import LLMRequest, fill_slots

N_PER_CLASS = 150
SAMPLE_SPLITS: tuple[Split, ...] = (Split.D_FIT, Split.UNLABELED_POOL)
SAMPLE_SEED_COMPONENT = "perturb_sample"
PARAPHRASE_SEED_COMPONENT = "perturb_paraphrase"
PURPOSE = "perturb_paraphrase"

PREFIX_FILE = "perturb_prefixes.txt"
PARAPHRASE_FILE = "perturb_paraphrase.txt"
LOCK_FILE = "perturb_prompts.lock.json"
N_PREFIXES = 3
TEXT_SLOT = "text"
PREFIX_JOIN = " "

PREFIX = "prefix"
PARAPHRASE = "paraphrase"
KINDS: tuple[str, ...] = (PREFIX, PARAPHRASE)
ITEM_SEP = "#"


class PerturbError(RuntimeError):
    pass


# ---------------------------------------------------------------------------------------
# files
# ---------------------------------------------------------------------------------------


def data_dir() -> Path:
    return paths.DATA / "step06" / "perturb"


def sample_path() -> Path:
    return data_dir() / "sample.jsonl"


def paraphrases_path() -> Path:
    return data_dir() / "paraphrases.jsonl"


def judgments_path() -> Path:
    return data_dir() / "paraphrase_judgments.jsonl"


def pod_input_path() -> Path:
    return data_dir() / "to_score.jsonl"


def scores_dir() -> Path:
    """The perturbed texts' scores, installed from the pod (never part of the formal tables)."""
    return data_dir() / "scores"


def prompts_dir() -> Path:
    return paths.CONFIGS / "prompts"


def sha256_file(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def text_sha256(text: str) -> str:
    return hashlib.sha256(text.replace("\r\n", "\n").encode("utf-8")).hexdigest()


def write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> str:
    """Write ``rows`` (LF, UTF-8) and a ``<name>.sha256`` next to them; returns the sha256."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8", newline="\n") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    tmp.replace(path)
    digest = sha256_file(path)
    path.with_name(path.name + ".sha256").write_text(
        f"{digest}  {path.name}\n", encoding="utf-8", newline="\n"
    )
    return digest


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def recorded_sha256(path: Path) -> str | None:
    sha = Path(path).with_name(Path(path).name + ".sha256")
    return sha.read_text(encoding="utf-8").split()[0] if sha.is_file() else None


# ---------------------------------------------------------------------------------------
# prompt files and their lock
# ---------------------------------------------------------------------------------------


def read_prefixes(directory: Path | None = None) -> list[str]:
    """The prefix table: exactly three non-empty, distinct lines of plain ASCII."""
    path = Path(directory or prompts_dir()) / PREFIX_FILE
    lines = [ln.strip() for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]
    if PLACEHOLDER in path.read_text(encoding="utf-8"):
        raise PerturbError(f"{path} still holds the placeholder marker")
    if len(lines) != N_PREFIXES or len(set(lines)) != N_PREFIXES:
        raise PerturbError(f"{path}: expected {N_PREFIXES} distinct prefixes, got {lines}")
    if not all(ln.isascii() for ln in lines):
        raise PerturbError(f"{path}: prefixes must be plain ASCII")
    return lines


def read_paraphrase_prompt(directory: Path | None = None) -> str:
    path = Path(directory or prompts_dir()) / PARAPHRASE_FILE
    text = path.read_text(encoding="utf-8")
    if PLACEHOLDER in text:
        raise PerturbError(f"{path} still holds the placeholder marker")
    if "{" + TEXT_SLOT + "}" not in text:
        raise PerturbError(f"{path} has no {{{TEXT_SLOT}}} slot")
    return text


def lock_payload(directory: Path | None = None) -> dict[str, Any]:
    d = Path(directory or prompts_dir())
    read_prefixes(d)
    read_paraphrase_prompt(d)
    return {
        "files": {
            name: text_sha256((d / name).read_text(encoding="utf-8"))
            for name in (PREFIX_FILE, PARAPHRASE_FILE)
        }
    }


def write_lock(directory: Path | None = None) -> dict[str, Any]:
    d = Path(directory or prompts_dir())
    payload = {"locked_at": datetime.now(tz=UTC).isoformat(timespec="seconds"), **lock_payload(d)}
    (d / LOCK_FILE).write_text(json.dumps(payload, indent=1) + "\n", encoding="utf-8", newline="\n")
    return payload


def check_lock(directory: Path | None = None) -> dict[str, Any]:
    """Refuse while the two prompt files are not locked or differ from their lock."""
    d = Path(directory or prompts_dir())
    path = d / LOCK_FILE
    if not path.is_file():
        raise PerturbError(f"{path} not found: lock the perturbation prompts first")
    locked = json.loads(path.read_text(encoding="utf-8"))
    now = lock_payload(d)
    if locked.get("files") != now["files"]:
        raise PerturbError(
            f"the perturbation prompt files differ from {path}: {now['files']} vs "
            f"{locked.get('files')}"
        )
    return locked


# ---------------------------------------------------------------------------------------
# the sample
# ---------------------------------------------------------------------------------------


@dataclass(frozen=True)
class SampleRow:
    state_id: str
    x0_id: str
    split: str
    text: str
    e_label: str  # y* of the state (laptop only)
    prefix_index: int

    def as_dict(self) -> dict[str, Any]:
        return dict(self.__dict__)


def draw_sample(
    states: Sequence[State],
    labels: Mapping[str, SafetyLabel],
    splits: Mapping[str, Split],
    seed: int,
    *,
    n_per_class: int | None = None,
) -> list[SampleRow]:
    """``n_per_class`` (default :data:`N_PER_CLASS`) E-UNSAFE and as many E-SAFE states, and a
    prefix for each.

    Deterministic in (the states, their y*, seed): each class is sorted by state_id, shuffled
    with ``py_rng(seed, "sample", class)`` and cut; the prefix of a state is drawn from
    ``py_rng(seed, "prefix", state_id)``. Output order: UNSAFE first, then SAFE, each in draw
    order. Raises when a class is too small.
    """
    n_per_class = N_PER_CLASS if n_per_class is None else n_per_class
    out: list[SampleRow] = []
    for label in (SafetyLabel.UNSAFE, SafetyLabel.SAFE):
        members = sorted(
            (s for s in states if labels[s.state_id] is label), key=lambda s: s.state_id
        )
        if len(members) < n_per_class:
            raise PerturbError(f"only {len(members)} {label.value} states; need {n_per_class}")
        py_rng(seed, "sample", label.value).shuffle(members)
        for s in members[:n_per_class]:
            out.append(
                SampleRow(
                    state_id=s.state_id,
                    x0_id=s.x0_id,
                    split=splits[s.state_id].value,
                    text=s.text,
                    e_label=label.value,
                    prefix_index=py_rng(seed, "prefix", s.state_id).randrange(N_PREFIXES),
                )
            )
    return out


def read_sample(path: Path | None = None) -> list[SampleRow]:
    p = Path(path or sample_path())
    want = recorded_sha256(p)
    if want is None or want != sha256_file(p):
        raise PerturbError(f"{p}: missing, or its sha256 differs from the one recorded")
    return [SampleRow(**r) for r in read_jsonl(p)]


def prefixed(row: SampleRow, prefixes: Sequence[str]) -> str:
    return prefixes[row.prefix_index] + PREFIX_JOIN + row.text


def item_id(state_id: str, kind: str) -> str:
    if kind not in KINDS:
        raise PerturbError(f"unknown perturbation {kind!r}")
    return f"{state_id}{ITEM_SEP}{kind}"


def split_item_id(item: str) -> tuple[str, str]:
    state_id, sep, kind = item.rpartition(ITEM_SEP)
    if not sep or kind not in KINDS:
        raise PerturbError(f"not a perturbation item id: {item!r}")
    return state_id, kind


# ---------------------------------------------------------------------------------------
# paraphrases
# ---------------------------------------------------------------------------------------


def paraphrase_config(
    models_cfg: Mapping[str, Any],
    generation_cfg: Mapping[str, Any],
    *,
    base_seed: int,
    prompt_dir: Path | None = None,
) -> g_fixed.GFixedConfig:
    """G_fixed's configuration (backbone, version, decoding, network, validity rules) with the
    paraphrase prompt in place of G_fixed's. ``max_attempts`` is not used: one call per state."""
    return g_fixed.build_config(
        models_cfg,
        generation_cfg,
        prompt_text=read_paraphrase_prompt(prompt_dir),
        base_seed=base_seed,
    )


def paraphrase_request(cfg: g_fixed.GFixedConfig, row: SampleRow) -> LLMRequest:
    """G_fixed's request shape (one user message, the same structured output) with the state's
    text in the ``{text}`` slot; the seed is derived per state."""
    return LLMRequest(
        purpose=PURPOSE,
        provider=cfg.provider,
        api=cfg.api,
        model=cfg.model,
        version=cfg.version,
        system=None,
        messages=[
            {"role": "user", "content": fill_slots(cfg.prompt_template, {TEXT_SLOT: row.text})}
        ],
        schema=GFixedOutput,
        decoding=cfg.decoding,
        seed=attempt_seed(cfg.base_seed, row.state_id, attempt=0)
        if cfg.base_seed is not None
        else None,
        attempt=0,
        network=cfg.network,
    )


def paraphrase_row(cfg: g_fixed.GFixedConfig, row: SampleRow, req: LLMRequest, resp) -> dict:
    """One paraphrase record: the call, the raw output, and G_fixed's validity verdict against
    the original state text."""
    output = (resp.parsed or {}).get("revised_text") if resp.parsed else None
    reason = judge_validity(
        output,
        row.text,
        provider_refused=resp.refused,
        phrases=cfg.refusal_phrases,
        non_english_threshold=cfg.non_english_threshold,
    )
    return {
        "state_id": row.state_id,
        "paraphrase": output if reason is None else None,
        "valid": reason is None,
        "invalid_reason": None if reason is None else reason.value,
        "raw_output": resp.text,
        "provider_refused": resp.refused,
        "stop_reason": resp.stop_reason,
        "seed": req.seed,
        "model": cfg.model,
        "model_version": resp.model_version,
        "prompt_hash": resp.prompt_hash,
        "decoding": cfg.decoding,
        "cached": resp.cached,
    }


def read_rows_by_state(path: Path) -> dict[str, dict[str, Any]]:
    """Latest row per state_id of a JSONL file (empty when it does not exist)."""
    if not Path(path).is_file():
        return {}
    return {r["state_id"]: r for r in read_jsonl(path)}


# ---------------------------------------------------------------------------------------
# the pod's input
# ---------------------------------------------------------------------------------------


def build_pod_input(
    sample: Sequence[SampleRow],
    prefixes: Sequence[str],
    paraphrases: Mapping[str, Mapping[str, Any]],
) -> list[dict[str, str]]:
    """``{state_id, text}`` rows (``state_id`` = the item id): every sampled state with its
    prefix, then every VALID paraphrase. Every sampled state must have a paraphrase record."""
    missing = [r.state_id for r in sample if r.state_id not in paraphrases]
    if missing:
        raise PerturbError(f"{len(missing)} sampled state(s) have no paraphrase yet: {missing[:3]}")
    rows = [
        {"state_id": item_id(r.state_id, PREFIX), "text": prefixed(r, prefixes)} for r in sample
    ]
    for r in sample:
        p = paraphrases[r.state_id]
        if p["valid"]:
            rows.append({"state_id": item_id(r.state_id, PARAPHRASE), "text": p["paraphrase"]})
    return rows


__all__ = [
    "KINDS",
    "N_PER_CLASS",
    "PARAPHRASE",
    "PREFIX",
    "PerturbError",
    "SampleRow",
    "build_pod_input",
    "check_lock",
    "draw_sample",
    "item_id",
    "paraphrase_config",
    "paraphrase_request",
    "paraphrase_row",
    "prefixed",
    "read_paraphrase_prompt",
    "read_prefixes",
    "read_sample",
    "split_item_id",
    "write_lock",
]
