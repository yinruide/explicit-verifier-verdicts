"""The one scoring path for V1 / V2.

    make_verifier(config_id) -> verifier;   verifier.score(texts: list[str]) -> list[float]

Step 6's scores table, Exp 2's in-process scoring and the step 2 probe must score with the SAME
code, so that the probe's D_fit and pool scores can be reused as they are and tau keeps one
scale. This module therefore writes no scoring code of its own. It imports the probe's:

* guard LMs: :class:`rafr.verifiers.candidates.torch_probe.TorchProbeBackend` (left padding,
  fixed shapes, ``logits_to_keep=1``, the template's own BOS), the verdict settled by
  :func:`rafr.verifiers.probe_run.preflight_lm` (greedy decode of the invented sentences, single
  verdict tokens, direction check) and :class:`rafr.verifiers.candidates.probe.GuardProbeScorer`
  (prompt rendering, the two-token softmax at the verdict position, ShieldGemma native's max
  over its two policies);
* the encoder: exactly the call ``probe_run.score_encoder`` makes
  (``registry.build("roberta-toxicity", ...)``: P(toxic) by name through id2label).

Every parameter is taken from, and checked against, what the probe recorded for that model in
its output directory (``download_manifest.json``, ``meta/<model>.json``,
``environment.json``; ``models.verifiers.probe_records`` says where they are): the resolved
commit, dtype, batch size, fixed-shape padding, the text budget (max length), the attention
implementation, the template source, the verdict prefix and token ids, how Llama Guard's
categories were applied, the prompt hash, and the transformers version. Any difference raises
:class:`ProbeMismatchError`; nothing is silently used. The two direction-check sentences are
re-scored on load and must reproduce the probe's recorded scores bit for bit, which catches a
numerical drift (another GPU, another torch) before a single state is scored; only an explicit
``tolerate_drift=True`` lets a drift through (never in formal mode), and it is then recorded in
:meth:`describe`.

**Device kinds** (2026-09-24, point 6). A probe record is a
``cpu`` or a ``cuda`` record, by where the probe ran (its ``environment.json``), and the direction
sentences' reference scores live in that record: they are kept per device kind and a run is only
ever compared with a record of its own kind. Formal scores come from the GPU only:
``make_verifier(..., formal=True)`` accepts a cuda record and a cuda run and nothing else, and
never tolerates a drift. A cpu record serves the laptop's dry run only.

**Batches** are the probe's: every forward pass has exactly the probe's batch size, the last,
short batch is filled up with filler rows (copies of its last text, as the probe's own backends
do) whose scores are dropped, and the sequence length is fixed per call type. The factory checks
that the probe ran that way and that the loaded backend runs that way; nothing else is allowed.

``config_id`` is the probe's configuration name, one to one: ``enc_paradetox``,
``sg_2b.native``, ``lg_1b.task``, ... (:func:`rafr.verifiers.candidates.probe.all_verifier_ids`).
It is also the ``verifier_id`` of every scores-table row.

The older registry guard backend (``torch_backend.TorchGuardBackend`` behind
``GuardLMVerifier``) tokenises a chat-template rendering with a second BOS and is NOT on this
path: :func:`from_config` accepts only :data:`FACTORY_PATH`, and :func:`refuse_legacy_guard`
rejects a ``GuardLMVerifier`` wherever a formal step receives a verifier object.
"""

from __future__ import annotations

import importlib
import json
import os
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from rafr.contracts.schemas import TBD
from rafr.core import paths
from rafr.verifiers.candidates import probe
from rafr.verifiers.candidates.base import DEFAULT_BATCH_LIMIT, VerifierConfigError

FACTORY_PATH = "rafr.verifiers.factory:make_verifier"
"""The only ``models.verifiers.factory`` value a formal step accepts."""

DEFAULT_RECORDS: tuple[str, ...] = ("data/probe/local", "data/probe/remote")
"""Where the probe's records live on the laptop: the local CPU run (the encoder) and the GPU
run pulled back by the RunPod driver (not released). ``models.verifiers.probe_records``
overrides it."""

FORBIDDEN_SOURCES: tuple[str, ...] = ("anthropic", "openai", "qwen", "xai", "x-ai", "grok")
"""paper Sec. 3: V1 and V2 may not share a source with E (Anthropic), the controller (OpenAI)
or the panel (Qwen, xAI)."""

ENCODER_CANDIDATE: dict[str, str] = {"enc_paradetox": "roberta-toxicity"}
"""The registry entry ``probe_run.score_encoder`` builds for each encoder configuration."""

DEVICE_KINDS: tuple[str, ...] = ("cpu", "cuda")
"""Where a probe record was made and where a run happens; records are kept per kind."""

MANIFEST_NAME = "download_manifest.json"
ENVIRONMENT_NAME = "environment.json"
DEVICE_ENV = "RAFR_VERIFIER_DEVICE"  # read by the probe's backends when they are built


class ProbeMismatchError(VerifierConfigError):
    """The factory would score differently from the probe; it refuses instead."""


# ---------------------------------------------------------------------------------------
# configuration ids and the probe's records
# ---------------------------------------------------------------------------------------


def parse_config_id(config_id: str) -> tuple[str, str | None]:
    """``"sg_2b.native"`` -> ``("sg_2b", "native")``; ``"enc_paradetox"`` -> ``(.., None)``."""
    model_id, _, variant = str(config_id).partition(".")
    m = probe.model(model_id)
    if m.is_lm and variant not in probe.VARIANTS:
        raise VerifierConfigError(
            f"{config_id!r}: a guard LM configuration is '<model>.<variant>', variant in "
            f"{probe.VARIANTS}"
        )
    if not m.is_lm and variant:
        raise VerifierConfigError(f"{config_id!r}: the encoder has no variants")
    return model_id, (variant or None)


def check_source(model: probe.ProbeModel) -> None:
    """paper Sec. 3: refuse a verifier from E's, the controller's or a panel member's source."""
    hay = f"{model.repo_id} {model.vendor}".lower()
    hit = [s for s in FORBIDDEN_SOURCES if s in hay]
    if hit:
        raise VerifierConfigError(
            f"{model.repo_id} ({model.vendor}) shares a source with E / the controller / the "
            f"panel ({hit}); II.9.1 forbids it"
        )


def config_for(config_id: str, prompt_dir: Path | None = None) -> probe.ProbeConfig:
    model_id, _ = parse_config_id(config_id)
    for cfg in probe.configs_for(model_id, prompt_dir):
        if cfg.verifier_id == config_id:
            return cfg
    raise VerifierConfigError(f"{config_id!r} is not a probe configuration")


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


@dataclass(frozen=True)
class ProbeRecord:
    """What the probe wrote for one model: download entry, meta, environment, preflight."""

    model_id: str
    records_dir: Path
    manifest: dict[str, Any]
    meta: dict[str, Any]
    environment: dict[str, Any]
    preflight: dict[str, Any] | None = None

    @property
    def revision(self) -> str:
        return str(self.manifest["revision"])

    @property
    def device_kind(self) -> str:
        """``cuda`` if the probe's environment record names a GPU, else ``cpu``."""
        return "cuda" if self.environment.get("gpu") else "cpu"

    def config_meta(self, config_id: str) -> dict[str, Any]:
        info = (self.meta.get("configs") or {}).get(config_id)
        if not info:
            raise ProbeMismatchError(
                f"{config_id}: the probe's meta for {self.model_id} has no entry for it"
            )
        return info


def records_dirs_from(models_cfg: Mapping[str, Any] | None) -> list[Path]:
    raw = ((models_cfg or {}).get("verifiers") or {}).get("probe_records") or DEFAULT_RECORDS
    return [paths.resolve(p) for p in raw]


def find_probe_record(
    model_id: str, records_dirs: Sequence[Path] | None = None, device_kind: str | None = None
) -> ProbeRecord:
    """The one records directory holding both a download entry and a meta file for the model,
    made on ``device_kind`` (``cpu`` / ``cuda``). Without a kind there must be exactly one
    record; with one, a record of the other kind never answers (records are per device kind)."""
    if device_kind is not None and device_kind not in DEVICE_KINDS:
        raise VerifierConfigError(f"device kind {device_kind!r} is not one of {DEVICE_KINDS}")
    dirs = [Path(d) for d in (records_dirs or [paths.resolve(p) for p in DEFAULT_RECORDS])]
    found: list[ProbeRecord] = []
    for d in dirs:
        manifest_path, meta_path = d / MANIFEST_NAME, d / "meta" / f"{model_id}.json"
        if not (manifest_path.is_file() and meta_path.is_file()):
            continue
        entry = (_read_json(manifest_path).get("models") or {}).get(model_id)
        if not entry:
            continue
        pre = d / "preflight" / f"{model_id}.json"
        env = d / ENVIRONMENT_NAME
        found.append(
            ProbeRecord(
                model_id=model_id,
                records_dir=d,
                manifest=entry,
                meta=_read_json(meta_path),
                environment=_read_json(env) if env.is_file() else {},
                preflight=_read_json(pre) if pre.is_file() else None,
            )
        )
    if not found:
        raise ProbeMismatchError(
            f"{model_id}: no probe record (download manifest + meta) under {[str(d) for d in dirs]}"
            " -- the factory only scores configurations the probe has scored"
        )
    kinds = sorted({r.device_kind for r in found})
    if device_kind is not None:
        found = [r for r in found if r.device_kind == device_kind]
        if not found:
            raise ProbeMismatchError(
                f"{model_id}: no {device_kind} probe record (the records there are {kinds}); "
                f"records are kept per device kind and a {'/'.join(kinds)} record does not serve "
                f"a {device_kind} run"
            )
    if len(found) > 1:
        raise ProbeMismatchError(
            f"{model_id}: {device_kind or ''} probe records in more than one place "
            f"{[str(r.records_dir) for r in found]}"
            + ("" if device_kind else f" (kinds {kinds}: name the device kind)")
        )
    return found[0]


def _require_equal(what: str, factory: Any, recorded: Any, where: str) -> None:
    if factory != recorded:
        raise ProbeMismatchError(
            f"{where}: {what} differs from the probe's -- factory {factory!r}, probe {recorded!r}"
        )


def prompt_files_for(cfg: probe.ProbeConfig) -> tuple[str, ...]:
    """The configs/prompts files a configuration's wording comes from (empty for the encoder
    and for Llama Guard native, whose categories are the template's own)."""
    fam, var = cfg.model.family, cfg.variant
    if fam == probe.ENC or (fam == probe.LG and var == probe.NATIVE):
        return ()
    if var == probe.TASK:
        return (
            (probe.TASK_DEF_FILE, probe.GG_BLOCK_FILE)
            if fam == probe.GG
            else (probe.TASK_DEF_FILE,)
        )
    if fam == probe.SG:
        return (probe.SG_HARASSMENT_FILE, probe.SG_HATE_SPEECH_FILE)
    return (probe.GG_HARM_FILE, probe.GG_BLOCK_FILE)


def static_checks(cfg: probe.ProbeConfig, rec: ProbeRecord) -> dict[str, Any]:
    """Every parameter that can be compared without loading weights. Returns the settings the
    factory will use (all of them the probe's)."""
    m, where = cfg.model, cfg.verifier_id
    info = rec.config_meta(where)
    if info.get("status") not in ("complete", "already complete"):
        raise ProbeMismatchError(
            f"{where}: the probe did not score this configuration ({info.get('status')!r})"
        )
    _require_equal("repo", m.repo_id, rec.manifest.get("repo_id"), where)
    _require_equal("dtype", m.dtype, rec.meta.get("dtype"), where)
    _require_equal("dtype (download manifest)", m.dtype, rec.manifest.get("dtype"), where)
    _require_equal("revision (meta)", rec.revision, rec.meta.get("revision"), where)
    settings: dict[str, Any] = {"batch_size": int(rec.meta["batch_size"])}
    if m.is_lm:
        _require_equal(
            "text budget (max tokens)",
            probe.TEXT_MAX_TOKENS,
            rec.meta.get("text_max_tokens"),
            where,
        )
        # every batch at the probe's batch size, the short last one filled up
        _require_equal("fixed batch shape", True, rec.meta.get("fixed_shape"), where)
        settings["fixed_shape"] = True
        settings["prompt_hash"] = info.get("prompt_hash")
    else:
        settings["truncation"] = rec.meta.get("truncation")
        settings["toxic_index"] = rec.meta.get("toxic_index")
    return settings


def _current_versions() -> dict[str, str | None]:
    out: dict[str, str | None] = {}
    for name in ("torch", "transformers"):
        try:
            out[name] = str(importlib.import_module(name).__version__)
        except ImportError:
            out[name] = None
    return out


def probe_device(rec: ProbeRecord) -> str:
    """Where the probe ran this model (:attr:`ProbeRecord.device_kind`). The factory runs it on
    the same kind of device: a CUDA forward pass of a CPU-probed model would not reproduce the
    probe's numbers."""
    return rec.device_kind


def require_cuda() -> None:
    """Formal scoring runs on a CUDA device or not at all."""
    try:
        import torch
    except ImportError as exc:
        raise VerifierConfigError("formal scoring needs torch with CUDA; torch is missing") from exc
    if not torch.cuda.is_available():
        raise VerifierConfigError(
            "formal scores come from the GPU only (2026-09-24): no CUDA device here"
        )


@contextmanager
def device_env(device: str) -> Iterator[None]:
    """``RAFR_VERIFIER_DEVICE`` set while the probe's backend is built (it reads it once)."""
    old = os.environ.get(DEVICE_ENV)
    os.environ[DEVICE_ENV] = device
    try:
        yield
    finally:
        if old is None:
            os.environ.pop(DEVICE_ENV, None)
        else:
            os.environ[DEVICE_ENV] = old


def check_environment(rec: ProbeRecord, where: str) -> dict[str, Any]:
    """transformers is pinned (pyproject [gpu]) and must equal the probe's; torch comes with the
    pod image and is recorded next to the probe's (a numerical consequence of a different torch
    or GPU is caught by the direction re-score)."""
    now = _current_versions()
    probe = {k: rec.environment.get(k) for k in ("torch", "transformers", "gpu", "cuda")}
    _require_equal("transformers version", now["transformers"], probe["transformers"], where)
    return {"probe": probe, "now": now}


# ---------------------------------------------------------------------------------------
# the verifier object
# ---------------------------------------------------------------------------------------


@dataclass
class FactoryVerifier:
    """score_spec.md verifier (``verifier_id``, ``model_version``, ``batch_limit``, ``score``)
    around the probe's own scorer. ``score_detailed`` adds the sub-scores and truncation the
    scores table records (the probe's ``ScoredText``)."""

    config_id: str
    config: probe.ProbeConfig
    record: ProbeRecord
    settings: dict[str, Any]
    scorer: Any = None  # probe.GuardProbeScorer for a guard LM
    encoder: Any = None  # the registry EncoderToxicityVerifier the probe builds
    drift: dict[str, Any] = field(default_factory=dict)
    environment: dict[str, Any] = field(default_factory=dict)
    batch_limit: int = DEFAULT_BATCH_LIMIT

    @property
    def verifier_id(self) -> str:
        return self.config_id

    @property
    def model_version(self) -> str:
        if self.scorer is not None:
            return str(self.scorer.model_version)
        return str(self.encoder.model_version)

    @property
    def prompt_hash(self) -> str | None:
        return None if self.scorer is None else self.scorer.prompt_hash

    @property
    def inference_batch_size(self) -> int:
        return int(self.settings["batch_size"])

    def score_detailed(self, texts: Sequence[str]) -> list[probe.ScoredText]:
        items = list(texts)
        if self.scorer is not None:
            return self.scorer.score_detailed(items)
        scores = self.encoder.score(items)
        tok = self.encoder.backend.tokenizer
        out = []
        for text, s in zip(items, scores, strict=True):
            # the probe's own truncation count for the encoder (special tokens included)
            n = len(tok(text)["input_ids"])
            out.append(probe.ScoredText(s, None, n, n > probe.TEXT_MAX_TOKENS))
        return out

    def score(self, texts: list[str]) -> list[float]:
        if isinstance(texts, str):
            raise TypeError("score() takes a list of texts, not a single string")
        if len(texts) > self.batch_limit:
            raise ValueError(f"batch of {len(texts)} exceeds batch_limit={self.batch_limit}")
        if not texts:
            return []
        if self.scorer is not None:
            return self.scorer.score(list(texts))
        return self.encoder.score(list(texts))

    def describe(self) -> dict[str, Any]:
        """What was loaded and how -- the step-6 table checks, the run snapshot and the card."""
        m = self.config.model
        out: dict[str, Any] = {
            "config_id": self.config_id,
            "model_id": m.model_id,
            "repo_id": m.repo_id,
            "revision": self.record.revision,
            "model_version": self.model_version,
            "dtype": m.dtype,
            "batch_size": self.inference_batch_size,
            "batch_limit": self.batch_limit,
            "prompt_hash": self.prompt_hash,
            "records_dir": str(self.record.records_dir),
            "record_device_kind": self.record.device_kind,
            "device": self.settings.get("device"),
            "formal": bool(self.settings.get("formal")),
            "environment": self.environment,
            "drift": self.drift,
        }
        if self.scorer is not None:
            b = self.scorer.backend
            v = self.scorer.verdict
            out.update(
                fixed_shape=b.fixed_shape,
                text_max_tokens=probe.TEXT_MAX_TOKENS,
                attn_implementation=getattr(b, "attn_implementation", None),
                template_source=getattr(b, "template_source", None),
                lg_manual=self.scorer.lg_manual,
                verdict={
                    "prefix_ids": list(v.prefix_ids),
                    "toxic": [v.toxic_id, v.toxic_form],
                    "safe": [v.safe_id, v.safe_form],
                },
            )
        else:
            out.update(
                truncation=self.encoder.truncation.describe(),
                toxic_index=self.encoder.toxic_index,
            )
        return out

    def close(self) -> None:
        if self.scorer is not None and hasattr(self.scorer.backend, "close"):
            self.scorer.backend.close()


# ---------------------------------------------------------------------------------------
# building it
# ---------------------------------------------------------------------------------------


def _check_direction(
    where: str, recorded: Mapping[str, Any], now: Mapping[str, Any], tolerate: bool
) -> dict[str, Any]:
    """The invented neutral / abusive sentences must score exactly as they did in the probe."""
    diff = {
        k: (recorded.get(k), now.get(k))
        for k in ("neutral", "abusive")
        if recorded.get(k) != now.get(k)
    }
    if diff and not tolerate:
        raise ProbeMismatchError(
            f"{where}: the direction-check sentences no longer score as in the probe {diff} -- "
            "another GPU, torch or kernel changed the numbers. Scores from here would not be "
            "on the probe's scale; pass tolerate_drift=True only if the authors accept that."
        )
    return {"direction": {k: {"probe": a, "now": b} for k, (a, b) in diff.items()}} if diff else {}


def _build_lm(
    cfg: probe.ProbeConfig,
    rec: ProbeRecord,
    settings: dict[str, Any],
    *,
    prompt_dir: Path | None,
    echo: Callable[[str], None],
    tolerate_drift: bool,
) -> tuple[Any, dict[str, Any]]:
    from rafr.verifiers import probe_run
    from rafr.verifiers.candidates.torch_probe import TorchProbeBackend

    m, where = cfg.model, cfg.verifier_id
    with device_env(settings["device"]):
        backend = TorchProbeBackend(
            model=m,
            revision=rec.revision,
            batch_size=settings["batch_size"],
            fixed_shape=settings["fixed_shape"],
        )
    try:
        _require_equal("loaded revision", backend.revision, rec.revision, where)
        _require_equal("batch size (backend)", backend.batch_size, settings["batch_size"], where)
        _require_equal("fixed batch shape (backend)", backend.fixed_shape, True, where)
        _require_equal(
            "attention implementation",
            getattr(backend, "attn_implementation", None),
            rec.meta.get("attn_implementation"),
            where,
        )
        _require_equal(
            "template source",
            getattr(backend, "template_source", None),
            rec.meta.get("template_source"),
            where,
        )
        report, verdict, lg_manual, directions = probe_run.preflight_lm(
            backend, m, prompt_dir, echo
        )
        recorded = rec.meta.get("verdict") or {}
        _require_equal(
            "verdict prefix ids", list(verdict.prefix_ids), recorded.get("prefix_ids"), where
        )
        _require_equal(
            "toxic verdict token",
            [verdict.toxic_id, verdict.toxic_form],
            recorded.get("toxic"),
            where,
        )
        _require_equal(
            "safe verdict token", [verdict.safe_id, verdict.safe_form], recorded.get("safe"), where
        )
        _require_equal(
            "category mechanism",
            report.get("categories_mechanism"),
            rec.meta.get("categories_mechanism"),
            where,
        )
        drift = _check_direction(
            where, rec.config_meta(where).get("direction") or {}, directions[where], tolerate_drift
        )
        scorer = probe.GuardProbeScorer(
            cfg, backend, verdict, prompt_dir=prompt_dir, lg_manual=lg_manual
        )
        _require_equal("prompt hash", scorer.prompt_hash, settings["prompt_hash"], where)
    except Exception:
        backend.close()
        raise
    return scorer, drift


def _build_encoder(
    cfg: probe.ProbeConfig, rec: ProbeRecord, settings: dict[str, Any], *, tolerate_drift: bool
) -> tuple[Any, dict[str, Any]]:
    from rafr.verifiers.candidates import registry

    m, where = cfg.model, cfg.verifier_id
    name = ENCODER_CANDIDATE.get(m.model_id)
    if name is None:
        raise VerifierConfigError(f"{where}: no encoder candidate mapped for {m.model_id}")
    _require_equal("repo (registry)", registry.spec(name).repo_id, m.repo_id, where)
    v = registry.build(
        name,
        verifier_id=m.model_id,
        revision=rec.revision,
        inference_batch_size=settings["batch_size"],
    )
    _require_equal("truncation and padding", v.truncation.describe(), settings["truncation"], where)
    _require_equal("fixed batch shape", v.truncation.fixed_batch_shape, True, where)
    _require_equal("batch size (encoder)", v.inference_batch_size, settings["batch_size"], where)
    _require_equal("dtype (encoder)", v.dtype, m.dtype, where)
    with device_env(settings["device"]):
        backend = v.backend  # loads the weights, on the probe's kind of device
    _require_equal("loaded revision", backend.revision, rec.revision, where)
    _require_equal("toxic class index", v.toxic_index, settings["toxic_index"], where)
    _require_equal(
        "id2label",
        {str(k): val for k, val in backend.id2label.items()},
        rec.meta.get("id2label"),
        where,
    )
    neutral, abusive = v.score([probe.NEUTRAL_EXAMPLE, probe.ABUSIVE_EXAMPLE])
    drift = _check_direction(
        where,
        rec.config_meta(where).get("direction") or {},
        {"neutral": neutral, "abusive": abusive},
        tolerate_drift,
    )
    return v, drift


def make_verifier(
    config_id: str,
    *,
    records_dirs: Sequence[Path] | None = None,
    prompt_dir: Path | None = None,
    echo: Callable[[str], None] | None = None,
    tolerate_drift: bool = False,
    device: str | None = None,
    formal: bool = False,
) -> FactoryVerifier:
    """Load ``config_id`` exactly as the probe scored it; raise on any difference.

    ``device`` (``cpu`` / ``cuda``) picks the probe record of that kind and runs there; without
    it the one record there is decides. ``formal=True`` (step 6, Exp 2): cuda only -- a cuda
    record, a CUDA device present, no tolerated drift (point 6).
    """
    say = echo or (lambda _s: None)
    if device is not None and device not in DEVICE_KINDS:
        raise VerifierConfigError(f"device {device!r} is not one of {DEVICE_KINDS}")
    if formal:
        if tolerate_drift:
            raise VerifierConfigError("formal scoring never tolerates a numerical drift")
        if device not in (None, "cuda"):
            raise VerifierConfigError(
                f"formal scores come from the GPU only (2026-09-24), not {device!r}"
            )
        device = "cuda"
        require_cuda()
    model_id, _ = parse_config_id(config_id)
    m = probe.model(model_id)
    check_source(m)
    cfg = config_for(config_id, prompt_dir)
    if m.is_lm:
        probe.check_prompt_lock(prompt_dir)  # prompts are the ones confirmed in the prompt review
    rec = find_probe_record(model_id, records_dirs, device_kind=device)
    settings = static_checks(cfg, rec)
    settings["device"] = rec.device_kind
    settings["formal"] = formal
    env = check_environment(rec, config_id)
    if m.is_lm:
        scorer, drift = _build_lm(
            cfg, rec, settings, prompt_dir=prompt_dir, echo=say, tolerate_drift=tolerate_drift
        )
        return FactoryVerifier(
            config_id, cfg, rec, settings, scorer=scorer, drift=drift, environment=env
        )
    encoder, drift = _build_encoder(cfg, rec, settings, tolerate_drift=tolerate_drift)
    return FactoryVerifier(
        config_id, cfg, rec, settings, encoder=encoder, drift=drift, environment=env
    )


# ---------------------------------------------------------------------------------------
# the formal entry points
# ---------------------------------------------------------------------------------------


def configured_ids(models_cfg: Mapping[str, Any]) -> tuple[str, str]:
    """``models.verifiers.v1.id`` / ``v2.id`` -- probe config_ids; TBD refuses."""
    ver = models_cfg.get("verifiers") or {}
    ids = []
    for key in ("v1", "v2"):
        vid = (ver.get(key) or {}).get("id")
        if vid in (None, TBD):
            raise VerifierConfigError(
                f"models.verifiers.{key}.id is TBD (the authors, after step 2)"
            )
        parse_config_id(str(vid))
        ids.append(str(vid))
    if ids[0] == ids[1]:
        raise VerifierConfigError("V1 and V2 are the same configuration")
    return ids[0], ids[1]


def check_factory_path(models_cfg: Mapping[str, Any]) -> None:
    spec = (models_cfg.get("verifiers") or {}).get("factory")
    if spec != FACTORY_PATH:
        raise VerifierConfigError(
            f"models.verifiers.factory is {spec!r}; the formal scoring path is {FACTORY_PATH} "
            "and nothing else (task 2026-09-24 section 3; the registry guard backend is off it)"
        )


def from_config(
    models_cfg: Mapping[str, Any],
    key: str,
    *,
    records_dirs: Sequence[Path] | None = None,
    echo: Any = None,
) -> FactoryVerifier:
    """V1 or V2 (``key`` = ``"v1"`` / ``"v2"``) through the one factory, formal: cuda only."""
    check_factory_path(models_cfg)
    ids = dict(zip(("v1", "v2"), configured_ids(models_cfg), strict=True))
    return make_verifier(
        ids[key],
        records_dirs=records_dirs or records_dirs_from(models_cfg),
        echo=echo,
        formal=True,
    )


def refuse_legacy_guard(obj: Any) -> Any:
    """Formal steps call this on every verifier object they are handed: the registry's guard
    LM wrapper (double BOS; point 4) never scores formally."""
    from rafr.verifiers.candidates.guard_lm import GuardLMVerifier

    if isinstance(obj, GuardLMVerifier):
        raise VerifierConfigError(
            f"{getattr(obj, 'verifier_id', obj)!r} is the deprecated registry guard backend "
            "(double BOS); formal scoring goes through rafr.verifiers.factory"
        )
    return obj
