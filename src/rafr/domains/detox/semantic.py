"""Semantic preservation for detox: the TextDetox 2024 official evaluation (I.6, II.2.1, II.9.3).

Reference implementation: pan-webis-de/pan-code, ``clef24/text-detoxification/evaluate.py`` at
commit 88037f4602d013698c31985764a17f086baa451f (2024-03-13). Its three sub-metrics, each
reproduced here from that file:

* **SIM** (:func:`similarity`): ``SentenceTransformer("sentence-transformers/LaBSE")``; per
  batch of ``min(32, n)`` pairs, ``model.encode(originals + rewrites)`` and ``1 - cosine`` of
  each (original, rewrite) pair with ``scipy.spatial.distance.cosine``.
* **STA** (:func:`style_accuracy`): ``textdetox/xlmr-large-toxicity-classifier``; batches of
  32, ``padding=True, truncation=True, max_length=512``; ``softmax(logits, -1)[:, target]``
  with target 0 -- the reference's ``main()`` passes ``style_target_label=0`` and the model's
  ``id2label`` is ``{0: neutral, 1: toxic}``, so STA is P(neutral). The reference swallows any
  exception and scores the batch 0; here it raises, because a silent 0 would read as "toxic".
* **FL** (:func:`fluency`): ``sacrebleu.CHRF()`` with its defaults,
  ``sentence_score(hypothesis=rewrite, references=[reference]).score / 100``. It needs a human
  reference; the reference computes it only when one is given.

``quality(x0, x_T)`` is SIM (2026-09-28). A quality score is always computed the same
way -- one (x0, x) pair per ``encode`` call -- so the Exp 2 scores and the D_fit scores behind
the preregistered cutpoint come from one code path; :func:`similarity` with the reference's
batch of 32 is kept for the diagnostic that compares the two.

Model ids, revisions and library versions come from ``configs/exp2.yaml:
analysis.semantic_preservation.protocol``; :func:`check_versions` refuses a different
sentence-transformers / sacrebleu. Everything heavy (torch, transformers, sentence-transformers,
sacrebleu, scipy) is imported only when a model is loaded or a metric computed, never at module
import: the pod's loop and the test suite never load them.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Protocol

REFERENCE = "pan-webis-de/pan-code clef24/text-detoxification/evaluate.py"
REFERENCE_BATCH_SIZE = 32
"""evaluate.py's default batch size for SIM and STA."""


class SemanticError(RuntimeError):
    pass


@dataclass(frozen=True)
class SemanticConfig:
    reference: str
    quality: str
    device: str
    sim_model: str
    sim_revision: str
    sim_library: str
    sta_model: str
    sta_revision: str
    sta_target_label: int
    sta_batch_size: int
    sta_max_length: int
    fl_metric: str
    fl_library: str

    def describe(self) -> dict[str, Any]:
        return {
            "reference": self.reference,
            "quality": self.quality,
            "device": self.device,
            "similarity": {
                "model": self.sim_model,
                "revision": self.sim_revision,
                "library": self.sim_library,
            },
            "toxicity": {
                "model": self.sta_model,
                "revision": self.sta_revision,
                "target_label": self.sta_target_label,
                "batch_size": self.sta_batch_size,
                "max_length": self.sta_max_length,
            },
            "fluency": {"metric": self.fl_metric, "library": self.fl_library},
        }


def build_config(exp2_cfg: Mapping[str, Any]) -> SemanticConfig:
    """``configs/exp2.yaml: analysis.semantic_preservation.protocol``, checked."""
    try:
        p = exp2_cfg["analysis"]["semantic_preservation"]["protocol"]
        sim, sta, fl = p["similarity"], p["toxicity"], p["fluency"]
        cfg = SemanticConfig(
            reference=str(p["reference"]),
            quality=str(p["quality"]),
            device=str(p["device"]),
            sim_model=str(sim["model"]),
            sim_revision=str(sim["revision"]),
            sim_library=str(sim["library"]),
            sta_model=str(sta["model"]),
            sta_revision=str(sta["revision"]),
            sta_target_label=int(sta["target_label"]),
            sta_batch_size=int(sta["batch_size"]),
            sta_max_length=int(sta["max_length"]),
            fl_metric=str(fl["metric"]),
            fl_library=str(fl["library"]),
        )
    except (KeyError, TypeError) as exc:
        raise SemanticError(
            f"configs/exp2.yaml: analysis.semantic_preservation.protocol incomplete ({exc})"
        ) from exc
    if cfg.quality != "similarity":
        raise SemanticError(f"quality must be 'similarity' (2026-09-28), got {cfg.quality!r}")
    if cfg.device != "cpu":
        raise SemanticError(f"the semantic metrics run on the CPU, got device {cfg.device!r}")
    for name, rev in (("similarity", cfg.sim_revision), ("toxicity", cfg.sta_revision)):
        if len(rev) != 40 or any(c not in "0123456789abcdef" for c in rev):
            raise SemanticError(f"{name}.revision must be a full 40-hex commit, got {rev!r}")
    return cfg


def _pin(spec: str) -> tuple[str, str]:
    name, sep, version = spec.partition("==")
    if not sep or not name or not version:
        raise SemanticError(f"library pin must read 'name==version', got {spec!r}")
    return name, version


def installed_version(dist: str) -> str | None:
    from importlib import metadata

    try:
        return metadata.version(dist)
    except metadata.PackageNotFoundError:
        return None


def check_versions(pins: Sequence[str]) -> dict[str, str]:
    """Every ``name==version`` pin must be the installed version (pip install -e ".[semantic]")."""
    out: dict[str, str] = {}
    for spec in pins:
        name, want = _pin(spec)
        have = installed_version(name)
        if have != want:
            raise SemanticError(
                f"{name} {have or 'not installed'} != pinned {want} "
                f'(configs/exp2.yaml; pip install -e ".[semantic]")'
            )
        out[name] = have
    return out


# ------------------------------------------------------------------------------------------
# SIM
# ------------------------------------------------------------------------------------------


class Encoder(Protocol):
    def encode(self, sentences: list[str]) -> Any: ...


def similarity(
    encoder: Encoder,
    originals: Sequence[str],
    rewrites: Sequence[str],
    batch_size: int = REFERENCE_BATCH_SIZE,
) -> list[float]:
    """evaluate.py's ``evaluate_sim`` (non-efficient branch): per batch, encode the originals
    followed by the rewrites in ONE call, then ``1 - scipy cosine`` pair by pair."""
    from scipy.spatial.distance import cosine

    if len(originals) != len(rewrites):
        raise SemanticError(f"{len(originals)} originals but {len(rewrites)} rewrites")
    if not originals:
        return []
    originals, rewrites = list(originals), list(rewrites)
    batch_size = min(batch_size, len(originals))
    out: list[float] = []
    for i in range(0, len(originals), batch_size):
        o, r = originals[i : i + batch_size], rewrites[i : i + batch_size]
        emb = encoder.encode(o + r)
        out.extend(
            float(1 - cosine(a, b)) for a, b in zip(emb[: len(o)], emb[len(o) :], strict=True)
        )
    return out


def pair_quality(encoder: Encoder, x0: str, x: str) -> float:
    """quality(x0, x): SIM of one pair, encoded on its own ([x0, x] in one call). The D_fit
    cutpoint and every Exp 2 score are computed this way."""
    return similarity(encoder, [x0], [x], batch_size=1)[0]


def load_similarity_model(cfg: SemanticConfig):
    """LaBSE at the pinned revision on the CPU (the reference: ``SentenceTransformer(id)``)."""
    check_versions([cfg.sim_library])
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(cfg.sim_model, revision=cfg.sim_revision, device=cfg.device)


class QualityScorer:
    """``quality(x0, x_T)`` for Exp 2's finals (``exp2.Exp2Setup.quality_fn``). LaBSE is loaded
    on the first call, so building the scorer costs nothing; ``loader`` is for tests."""

    def __init__(
        self, cfg: SemanticConfig, loader: Callable[[SemanticConfig], Encoder] | None = None
    ) -> None:
        self.cfg = cfg
        self._loader = loader or load_similarity_model
        self._encoder: Encoder | None = None

    def __call__(self, x0: str, x_t: str) -> float:
        if self._encoder is None:
            self._encoder = self._loader(self.cfg)
        return pair_quality(self._encoder, x0, x_t)


# ------------------------------------------------------------------------------------------
# STA
# ------------------------------------------------------------------------------------------


def load_toxicity_model(cfg: SemanticConfig):
    """(model, tokenizer) of the TextDetox toxicity classifier at the pinned revision, CPU."""
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    model = AutoModelForSequenceClassification.from_pretrained(
        cfg.sta_model, revision=cfg.sta_revision
    )
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(cfg.sta_model, revision=cfg.sta_revision)
    label = model.config.id2label.get(cfg.sta_target_label)
    if label != "neutral":
        raise SemanticError(
            f"{cfg.sta_model}: label {cfg.sta_target_label} is {label!r}, expected 'neutral'"
        )
    return model, tokenizer


def style_accuracy(
    model: Any,
    tokenizer: Any,
    texts: Sequence[str],
    *,
    target_label: int = 0,
    batch_size: int = REFERENCE_BATCH_SIZE,
    max_length: int = 512,
) -> list[float]:
    """evaluate.py's ``classify_texts`` as ``main()`` calls it: softmax probability of
    ``target_label`` (0 = neutral) per text, batches of 32, truncated at 512 tokens."""
    import torch

    out: list[float] = []
    for i in range(0, len(texts), batch_size):
        inputs = tokenizer(
            list(texts[i : i + batch_size]),
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        ).to(model.device)
        with torch.no_grad():
            logits = model(**inputs).logits
            if logits.shape[-1] > 1:
                preds = torch.softmax(logits, -1)[:, target_label]
            else:
                preds = torch.sigmoid(logits)[:, 0]
        out.extend(float(p) for p in preds.cpu().numpy())
    return out


# ------------------------------------------------------------------------------------------
# FL
# ------------------------------------------------------------------------------------------


def fluency(rewrites: Sequence[str], references: Sequence[str]) -> list[float]:
    """evaluate.py's CHRF block: ``CHRF()`` defaults, sentence-level, divided by 100."""
    if len(rewrites) != len(references):
        raise SemanticError(f"{len(rewrites)} rewrites but {len(references)} references")
    from sacrebleu import CHRF

    chrf = CHRF()
    return [
        float(chrf.sentence_score(hypothesis=h, references=[r]).score / 100)
        for h, r in zip(rewrites, references, strict=True)
    ]


def joint(sta: Sequence[float], sim: Sequence[float], fl: Sequence[float]) -> list[float]:
    """evaluate.py's ``J = STA * SIM * CHRF``, element-wise."""
    return [a * b * c for a, b, c in zip(sta, sim, fl, strict=True)]
