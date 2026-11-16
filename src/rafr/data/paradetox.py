"""ParaDetox EN (paper Sec. 4.2): download, clean, deduplicate the x0 pool.

Only the toxic source sentence is used; gold neutral references are never used as states
(paper Sec. 4.2). Cleaning = whitespace normalization + exact dedupe on the normalized
text + drop empties. Nothing else is altered.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

from rafr.contracts.schemas import text_hash
from rafr.generation.validity import normalize_ws

DEFAULT_REPO = "s-nlp/paradetox"
DEFAULT_TEXT_FIELD = "en_toxic_comment"
_DATA_SUFFIXES = (".parquet", ".csv", ".tsv", ".jsonl", ".json")


def resolve_revision(repo_id: str = DEFAULT_REPO, token: str | None = None) -> str:
    """The dataset repo's current commit sha (decision, 2026-09-17).

    step00 calls this once, pins the answer in the manifest and passes that pin to
    :func:`download` on every later run, so the pool never silently follows the upstream head
    and nobody has to copy a hash into a config file by hand.
    """
    from huggingface_hub import HfApi

    return str(HfApi().dataset_info(repo_id, token=token).sha)


def download(
    repo_id: str = DEFAULT_REPO,
    revision: str | None = None,
    dest_dir: Path | None = None,
    token: str | None = None,
) -> Path:
    """Snapshot the HF dataset repo into ``dest_dir`` (default ``data/raw/paradetox``).

    ``revision`` should be the commit hash that :func:`resolve_revision` pinned on the first
    run (``data/manifest.json: source.hf_revision``) so that the pool is reproducible; the
    manifest records the files' sha256 regardless.
    """
    from huggingface_hub import snapshot_download

    from rafr.core import paths

    dest = Path(dest_dir) if dest_dir else paths.DATA_RAW / "paradetox"
    dest.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id=repo_id,
        repo_type="dataset",
        revision=revision,
        local_dir=str(dest),
        token=token,
    )
    return dest


def find_data_files(root: Path) -> list[Path]:
    files = [p for p in Path(root).rglob("*") if p.is_file() and p.suffix.lower() in _DATA_SUFFIXES]
    return sorted(p for p in files if ".cache" not in p.parts and not p.name.startswith("."))


def read_table(path: Path) -> pd.DataFrame:
    suf = path.suffix.lower()
    if suf == ".parquet":
        return pd.read_parquet(path)
    if suf == ".csv":
        return pd.read_csv(path)
    if suf == ".tsv":
        return pd.read_csv(path, sep="\t")
    if suf in (".jsonl", ".json"):
        return pd.read_json(path, lines=(suf == ".jsonl"))
    raise ValueError(f"unsupported data file: {path}")


def load_raw(root: Path, text_field: str = DEFAULT_TEXT_FIELD) -> list[dict[str, Any]]:
    """Concatenate every data file under ``root`` and return rows as dicts."""
    frames = [read_table(p) for p in find_data_files(root)]
    if not frames:
        raise FileNotFoundError(f"no data files under {root}")
    df = pd.concat(frames, ignore_index=True)
    if text_field not in df.columns:
        raise KeyError(f"column {text_field!r} not in {list(df.columns)}")
    return df.to_dict(orient="records")


@dataclass
class CleanResult:
    x0s: list[dict[str, str]]  # {"x0_id", "text"}
    stats: dict[str, int] = field(default_factory=dict)


def x0_id_for(normalized_text: str) -> str:
    return "x0-" + text_hash(normalized_text)


def clean(
    rows: list[dict[str, Any]], text_field: str = DEFAULT_TEXT_FIELD, min_chars: int = 1
) -> CleanResult:
    """Normalize whitespace, drop empties / too-short, dedupe exactly on normalized text.

    Order is preserved (first occurrence wins) so that the result is deterministic given the
    same raw files.
    """
    seen: set[str] = set()
    out: list[dict[str, str]] = []
    n_empty = n_dupe = 0
    for r in rows:
        raw = r.get(text_field)
        text = normalize_ws(str(raw)) if raw is not None and not _is_nan(raw) else ""
        if len(text) < max(1, min_chars):
            n_empty += 1
            continue
        if text in seen:
            n_dupe += 1
            continue
        seen.add(text)
        out.append({"x0_id": x0_id_for(text), "text": text})
    stats = {"n_raw": len(rows), "n_empty": n_empty, "n_duplicates": n_dupe, "n_kept": len(out)}
    return CleanResult(x0s=out, stats=stats)


def _is_nan(v: Any) -> bool:
    return isinstance(v, float) and v != v


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()
