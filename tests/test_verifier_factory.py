"""The one scoring factory (task 2026-09-24 section 3; decisions.md D-2026-09-24b point 4).

Two layers:

* offline checks that need no torch: config ids, the source rule, the probe records and every
  parameter comparison, the formal entry points, the legacy guard backend kept off the path, the
  verifier card;
* consistency with tiny RANDOM weights (needs torch + transformers, skipped otherwise, e.g. in
  CI): for one configuration of each architecture -- Gemma 2 (ShieldGemma), Llama (Llama Guard
  3), Llama 4 (Llama Guard 4), Granite (Granite Guardian) -- and the encoder, the same texts are
  scored once through the probe's own runner (``probe_run.score_lm`` / ``score_encoder``, which
  writes the table and the records) and once through ``make_verifier``; the two must be
  identical bit for bit.

The probe's preflight reads the verdict tokens off a greedy decode of the real weights, which
random weights cannot produce; both paths therefore get the same stand-in preflight (same
verdict, direction scores computed by the probe's own scorer). Everything else is the real code.
"""

from __future__ import annotations

import copy
import dataclasses
import json
import shutil
from pathlib import Path

import pytest

from rafr.core.config import load_config
from rafr.verifiers import factory as F
from rafr.verifiers.candidates import probe as P
from rafr.verifiers.candidates.base import VerifierConfigError

ROOT = Path(__file__).resolve().parents[1]


# ---------------------------------------------------------------------------------------
# offline: ids, sources, records, parameters
# ---------------------------------------------------------------------------------------


def test_every_probe_configuration_is_a_config_id():
    ids = P.all_verifier_ids()
    assert "enc_paradetox" in ids and "sg_2b.native" in ids and "lg_1b.task" in ids
    for cid in ids:
        model_id, variant = F.parse_config_id(cid)
        assert F.config_for(cid).verifier_id == cid
        assert (variant is None) == (not P.model(model_id).is_lm)
    for bad in ("sg_2b", "sg_2b.other", "enc_paradetox.native", "nope.native"):
        with pytest.raises(VerifierConfigError):
            F.parse_config_id(bad)


def test_no_verifier_from_e_the_controller_or_the_panel():
    """II.9.1 (v0.9): V1, V2 not from Anthropic, OpenAI, Qwen or xAI."""
    for m in P.MODELS.values():
        F.check_source(m)  # the probe's candidates are Google, Meta, IBM, s-nlp
    for vendor, repo in (("Alibaba", "Qwen/Qwen3Guard-Gen-8B"), ("xAI", "xai-org/grok-guard")):
        fake = dataclasses.replace(P.MODELS["sg_2b"], vendor=vendor, repo_id=repo)
        with pytest.raises(VerifierConfigError, match="II.9.1"):
            F.check_source(fake)


def _records(tmp: Path, model_id: str, config_id: str, **meta_over) -> Path:
    m = P.model(model_id)
    d = tmp / "records"
    (d / "meta").mkdir(parents=True, exist_ok=True)
    manifest = {
        "models": {
            model_id: {
                "repo_id": m.repo_id,
                "revision": "c0ffee",
                "dtype": m.dtype,
                "weight_files": [{"path": "model.safetensors", "bytes": 3, "sha256": "ab" * 32}],
            }
        }
    }
    (d / F.MANIFEST_NAME).write_text(json.dumps(manifest), encoding="utf-8")
    meta = {
        "model_id": model_id,
        "revision": "c0ffee",
        "dtype": m.dtype,
        "batch_size": 8 if m.is_lm else 16,
        "fixed_shape": True,
        "text_max_tokens": P.TEXT_MAX_TOKENS,
        "attn_implementation": "eager",
        "template_source": "tokenizer",
        "verdict": {"prefix_ids": [1], "prefix_text": "\n", "toxic": [5, "Yes"], "safe": [6, "No"]},
        "categories_mechanism": None,
        "truncation": "t",
        "toxic_index": 1,
        "id2label": {"0": "neutral", "1": "toxic"},
        "configs": {
            config_id: {
                "prompt_hash": "ph",
                "direction": {"neutral": 0.1, "abusive": 0.9, "ok": True},
                "status": "complete",
            }
        },
    }
    meta.update(meta_over)
    (d / "meta" / f"{model_id}.json").write_text(json.dumps(meta), encoding="utf-8")
    (d / F.ENVIRONMENT_NAME).write_text(
        json.dumps({"torch": "2.x", "transformers": "5.16.1", "gpu": "A6000", "cuda": "12"}),
        encoding="utf-8",
    )
    return d


def test_static_checks_take_every_setting_from_the_probe_and_refuse_a_difference(tmp_path):
    d = _records(tmp_path, "sg_2b", "sg_2b.native")
    cfg = F.config_for("sg_2b.native")
    rec = F.find_probe_record("sg_2b", [d])
    assert rec.revision == "c0ffee"
    settings = F.static_checks(cfg, rec)
    assert settings == {"batch_size": 8, "fixed_shape": True, "prompt_hash": "ph"}
    for over, what in (
        ({"dtype": "float32"}, "dtype"),
        ({"text_max_tokens": 256}, "text budget"),
        ({"revision": "other"}, "revision"),
        ({"configs": {"sg_2b.native": {"status": "skipped: direction check failed"}}}, "did not"),
    ):
        d2 = _records(tmp_path / what.replace(" ", "_"), "sg_2b", "sg_2b.native", **over)
        with pytest.raises(F.ProbeMismatchError, match=what):
            F.static_checks(cfg, F.find_probe_record("sg_2b", [d2]))
    with pytest.raises(F.ProbeMismatchError, match="no probe record"):
        F.find_probe_record("sg_9b", [d])
    with pytest.raises(F.ProbeMismatchError, match="more than one place"):
        F.find_probe_record("sg_2b", [d, d])
    enc = _records(tmp_path / "enc", "enc_paradetox", "enc_paradetox")
    s = F.static_checks(F.config_for("enc_paradetox"), F.find_probe_record("enc_paradetox", [enc]))
    assert s == {"batch_size": 16, "truncation": "t", "toxic_index": 1}


def test_transformers_must_be_the_probe_version(tmp_path):
    d = _records(tmp_path, "sg_2b", "sg_2b.native")
    rec = F.find_probe_record("sg_2b", [d])
    env = json.loads((d / F.ENVIRONMENT_NAME).read_text(encoding="utf-8"))
    env["transformers"] = "0.0.1"
    (d / F.ENVIRONMENT_NAME).write_text(json.dumps(env), encoding="utf-8")
    rec = F.find_probe_record("sg_2b", [d])
    with pytest.raises(F.ProbeMismatchError, match="transformers"):
        F.check_environment(rec, "sg_2b.native")


def test_the_formal_entry_accepts_only_the_factory_and_two_distinct_config_ids():
    models = copy.deepcopy(load_config("models"))
    assert models["verifiers"]["factory"] == F.FACTORY_PATH
    # 2026-09-25 (decisions.md D-2026-09-25)
    assert F.configured_ids(models) == ("enc_paradetox", "sg_9b.native")
    models["verifiers"]["v2"]["id"] = "TBD"
    with pytest.raises(VerifierConfigError, match="TBD"):
        F.configured_ids(models)
    models["verifiers"]["v1"]["id"] = "enc_paradetox"
    models["verifiers"]["v2"]["id"] = "enc_paradetox"
    with pytest.raises(VerifierConfigError, match="same"):
        F.configured_ids(models)
    models["verifiers"]["v2"]["id"] = "lg_1b.native"
    assert F.configured_ids(models) == ("enc_paradetox", "lg_1b.native")
    for spec in ("rafr.verifiers.candidates.registry:llama_guard_3_1b", None, "x:y"):
        models["verifiers"]["factory"] = spec
        with pytest.raises(VerifierConfigError, match="formal scoring path"):
            F.from_config(models, "v1")
    assert [str(p) for p in F.records_dirs_from(load_config("models"))] == [
        str(ROOT / "data" / "probe" / "local"),
        str(ROOT / "data" / "probe" / "remote"),
    ]


def test_the_legacy_guard_backend_is_off_the_formal_path(tmp_path):
    """decisions.md D-2026-09-24b point 4: the registry guard backend (double BOS) never scores
    formally: the factory refuses it as an object, batch_score refuses it as a candidate."""
    from rafr.verifiers import batch_score
    from rafr.verifiers.candidates import registry

    legacy = registry.build("llama-guard-3-1b")  # lazy: no weights
    with pytest.raises(VerifierConfigError, match="deprecated"):
        F.refuse_legacy_guard(legacy)
    for name, spec in registry.CANDIDATES.items():
        if spec.kind == registry.GUARD_LM:
            with pytest.raises(SystemExit, match="second BOS"):
                batch_score.make_verifier(
                    candidate=name, impl=None, batch_size=None, verifier_id=None
                )
    src = tmp_path / "in.jsonl"
    src.write_text(json.dumps({"state_id": "a", "text": "t"}) + "\n", encoding="utf-8")
    with pytest.raises(SystemExit, match="second BOS"):
        batch_score.main(["--input", str(src), "--candidate", "shieldgemma-2b"], echo=print)
    with pytest.raises(SystemExit, match="exactly one"):
        batch_score.make_verifier(
            candidate="toxic-bert", impl=None, batch_size=None, verifier_id=None,
            config_id="enc_paradetox",
        )  # fmt: skip
    # nothing on the formal path imports the legacy classes
    for rel in ("src/rafr/verifiers/factory.py", "pipelines/step12_exp2.py"):
        text = (ROOT / rel).read_text(encoding="utf-8")
        assert "import TorchGuardBackend" not in text
        assert "GuardLMVerifier(" not in text


def test_the_card_has_every_appendix_a2_field(tmp_path, monkeypatch):
    from rafr.verifiers.cards import factory_card

    # the pending branch, whatever tags the checkout running the test happens to have
    monkeypatch.setattr(factory_card, "tag_commit", lambda tag: None)
    d = _records(tmp_path, "sg_2b", "sg_2b.native")
    card = factory_card.render_card("sg_2b.native", records_dirs=[d])
    for needle in (
        "`sg_2b.native`",
        "`google/shieldgemma-2b`",
        "`c0ffee`",
        "ab" * 32,
        "verifier_sg_harassment.txt",
        "verifier_sg_hate_speech.txt",
        "Prompt hash (every scores-table row): `ph`",
        "F1",
        "tau-frozen",
        "[0, 1]",
        "higher = more toxic",
        "batch cap",
        "inference batch size | 8",
        "the larger, both sub-scores kept",
        "direction-check sentences",
        "| transformers | 5.16.1 | not run yet |",
        "| torch | 2.x | not run yet |",
        # step 6 preparation (task section 5): dtype, the pod, tau pending step 7
        "weights dtype | bfloat16",
        "runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404",
        "A100 SXM 80GB",
        "**Pending step 7 (`tau-frozen`).**",
    ):
        assert needle in card, needle
    hashes = P.prompt_file_hashes()
    assert hashes["verifier_sg_harassment.txt"] in card
    enc = factory_card.render_card(
        "enc_paradetox", records_dirs=[_records(tmp_path / "e", "enc_paradetox", "enc_paradetox")]
    )
    assert "None: the encoder reads the text itself." in enc and "P(toxic)" in enc
    out = factory_card.write_card("sg_2b.native", tmp_path / "cards", records_dirs=[d])
    assert out.name == "sg_2b.native.md" and out.read_text(encoding="utf-8") == card
    # the rule is written out even before step 7
    assert "midpoint of" in card and "1e-12" in card and "widest plateau" in card


def test_the_card_reads_tau_from_the_tagged_commit_once_tau_frozen_exists(tmp_path, monkeypatch):
    from rafr.verifiers.cards import factory_card

    seen: list[str] = []
    fit = {
        "verifier_id": "sg_2b.native", "tau": 0.21225668054284383, "f1_max": 0.46206896551724136,
        "plateau_lo": 0.2, "plateau_hi": 0.225, "n_plateaus": 2,
        "n_candidates": 85, "n": 600, "n_positive": 96, "criterion": "f1_max_plateau_midpoint",
    }  # fmt: skip
    monkeypatch.setattr(
        factory_card, "tag_commit", lambda tag: "c0mm1t" if tag == "tau-frozen" else None
    )

    def payload(commit):
        seen.append(commit)
        return {"fits": {"v2": fit, "v1": {**fit, "verifier_id": "enc_paradetox", "tau": 0.5}}}

    monkeypatch.setattr(factory_card, "tau_payload", payload)
    d = _records(tmp_path, "sg_2b", "sg_2b.native")
    card = factory_card.render_card("sg_2b.native", records_dirs=[d])
    assert seen == ["c0mm1t"]  # the tagged commit's file, not the working tree's
    for needle in (
        "Frozen at step 7: `frozen/tau.json` at git tag `tau-frozen` (commit `c0mm1t`).",
        "| tau_i | 0.21226 (full precision `0.21225668054284383`) |",
        "| fitted on | D_fit, 600 states, 96 of them E-UNSAFE |",
        "| F1 at tau_i (the maximum) | 0.46207 |",
        "| candidate thresholds | 85 |",
        "| plateaus attaining the maximum | 2 |",
        "| widest plateau [lo, hi] | [0.2, 0.225] |",
        "the largest plus half the last gap",
        "within 1e-12 of the",
    ):
        assert needle in card, needle
    assert "Pending step 7" not in card
    # a tag whose tau.json has no fit for this configuration says so instead of inventing one
    other = factory_card.render_card(
        "enc_paradetox", records_dirs=[_records(tmp_path / "e", "enc_paradetox", "enc_paradetox")]
    )
    assert "| tau_i | 0.5 (full precision `0.5`) |" in other
    monkeypatch.setattr(factory_card, "tau_payload", lambda commit: {"fits": {}})
    none = factory_card.render_card("sg_2b.native", records_dirs=[d])
    assert "holds no fit for `sg_2b.native`" in none and "| tau_i |" not in none


# ---------------------------------------------------------------------------------------
# consistency with tiny random weights: the probe path == the factory, bit for bit
# ---------------------------------------------------------------------------------------

TEXTS = [
    "hello world",
    "the quick brown fox is not safe",
    "Yes no yes no",
    "",
    " ".join(["fox"] * 700),  # longer than the 512-token text budget: truncated
]


def _tiny_tokenizer(tf):
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

    tok = Tokenizer(models.BPE())
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=400,
        special_tokens=["<pad>", "<bos>", "<eos>"],
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
    )
    tok.train_from_iterator(
        ["hello world the quick brown fox safe unsafe Yes No yes no"] * 5, trainer
    )
    fast = tf.PreTrainedTokenizerFast(
        tokenizer_object=tok, bos_token="<bos>", eos_token="<eos>", pad_token="<pad>"
    )
    # stands in for each vendor's template: it takes their messages and keyword arguments
    fast.chat_template = (
        "{{ bos_token }}{% for m in messages %}<{{ m['role'] }}>"
        "{% if m['content'] is string %}{{ m['content'] }}{% else %}"
        "{% for p in m['content'] %}{{ p['text'] }}{% endfor %}{% endif %}{% endfor %}"
        "{% if guideline is defined %}[{{ guideline }}]{% endif %}"
        "{% if categories is defined %}{% for k, v in categories.items() %}{{ k }}: {{ v }};"
        "{% endfor %}{% endif %}<assistant>"
    )
    return fast


@pytest.fixture(scope="module")
def tiny_models(tmp_path_factory):
    torch = pytest.importorskip("torch")
    tf = pytest.importorskip("transformers")
    root = tmp_path_factory.mktemp("tiny")
    fast = _tiny_tokenizer(tf)
    common = dict(
        vocab_size=len(fast), hidden_size=16, intermediate_size=32, num_hidden_layers=2,
        num_attention_heads=2, num_key_value_heads=1, max_position_embeddings=4096,
        pad_token_id=fast.pad_token_id, bos_token_id=fast.bos_token_id,
        eos_token_id=fast.eos_token_id,
    )  # fmt: skip
    configs = {
        "sg_2b": tf.Gemma2Config(head_dim=8, **common),
        "lg_1b": tf.LlamaConfig(head_dim=8, **common),
        "gg_8b": tf.GraniteConfig(**common),
    }
    dirs = {}
    for model_id, cfg in configs.items():
        torch.manual_seed(0)
        d = root / model_id
        tf.AutoModelForCausalLM.from_config(cfg).save_pretrained(d)
        fast.save_pretrained(d)
        dirs[model_id] = d
    text = tf.Llama4TextConfig(head_dim=8, moe_layers=[], intermediate_size_mlp=32, **common)
    vision = tf.Llama4VisionConfig(
        hidden_size=16, intermediate_size=32, num_hidden_layers=1, num_attention_heads=2,
        image_size=28, patch_size=14, vision_output_dim=16, projector_input_dim=16,
        projector_output_dim=16,
    )  # fmt: skip
    torch.manual_seed(0)
    d = root / "lg_12b"
    tf.Llama4ForConditionalGeneration(
        tf.Llama4Config(text_config=text.to_dict(), vision_config=vision.to_dict())
    ).save_pretrained(d)
    fast.save_pretrained(d)
    dirs["lg_12b"] = d
    torch.manual_seed(0)
    d = root / "enc_paradetox"
    tf.AutoModelForSequenceClassification.from_config(
        tf.RobertaConfig(
            vocab_size=len(fast), hidden_size=16, intermediate_size=32, num_hidden_layers=1,
            num_attention_heads=2, max_position_embeddings=520, type_vocab_size=1,
            pad_token_id=fast.pad_token_id, id2label={0: "neutral", 1: "toxic"},
            label2id={"neutral": 0, "toxic": 1},
        )
    ).save_pretrained(d)  # fmt: skip
    fast.save_pretrained(d)
    dirs["enc_paradetox"] = d
    return dirs


def _stand_in_preflight(backend, m, prompt_dir, echo):
    """The probe's preflight minus the greedy decode random weights cannot pass: the verdict
    tokens are read off the tokenizer, the direction scores come from the probe's scorer."""
    toxic = backend.encode(m.toxic_word)[0]
    safe = backend.encode(m.safe_word)[0]
    if safe == toxic:
        safe = toxic + 1
    verdict = P.Verdict(
        tuple(backend.encode(m.label_prefix or "")), m.label_prefix or "", toxic, safe,
        m.toxic_word, m.safe_word,
    )  # fmt: skip
    report = {"passed": True, "categories_mechanism": "chat_template" if m.family == P.LG else None}
    directions = {}
    for cfg in P.configs_for(m.model_id, prompt_dir):
        sc = P.GuardProbeScorer(cfg, backend, verdict, prompt_dir=prompt_dir)
        n, a = sc.score_detailed([P.NEUTRAL_EXAMPLE, P.ABUSIVE_EXAMPLE])
        directions[cfg.verifier_id] = {
            "neutral": n.score, "abusive": a.score, "neutral_sub": n.sub_scores,
            "abusive_sub": a.sub_scores, "ok": True,
        }  # fmt: skip
    return report, verdict, False, directions


@pytest.fixture
def probe_world(tiny_models, tmp_path, monkeypatch):
    """Point the probe's candidates at the tiny weights, lock a copy of the prompts, stand in
    for the greedy preflight, and forbid the legacy guard backend outright."""
    from rafr.verifiers import probe_run
    from rafr.verifiers.candidates import guard_lm, registry, torch_backend

    models = dict(P.MODELS)
    for model_id, d in tiny_models.items():
        models[model_id] = dataclasses.replace(models[model_id], repo_id=str(d))
    monkeypatch.setattr(P, "MODELS", models)
    cands = dict(registry.CANDIDATES)
    cands["roberta-toxicity"] = dataclasses.replace(
        cands["roberta-toxicity"], repo_id=str(tiny_models["enc_paradetox"])
    )
    monkeypatch.setattr(registry, "CANDIDATES", cands)
    monkeypatch.setattr(probe_run, "preflight_lm", _stand_in_preflight)
    monkeypatch.setattr(P, "direction_ok", lambda a, b: True)  # random weights: no direction
    monkeypatch.setenv("RAFR_VERIFIER_DEVICE", "cpu")
    monkeypatch.delenv("RAFR_VERIFIER_DYNAMIC_PADDING", raising=False)

    def forbidden(*a, **k):
        raise AssertionError("the legacy registry guard backend was reached")

    monkeypatch.setattr(torch_backend.TorchGuardBackend, "__init__", forbidden)
    monkeypatch.setattr(guard_lm.GuardLMVerifier, "__init__", forbidden)
    prompts = tmp_path / "prompts"
    prompts.mkdir()
    for name in P.PROMPT_FILES:
        shutil.copyfile(P.prompts_dir() / name, prompts / name)
    (prompts / P.PROMPT_LOCK_FILE).write_text(json.dumps(P.lock_payload(prompts)), "utf-8")
    out = tmp_path / "out"
    out.mkdir()
    manifest = {
        "models": {
            mid: {
                "repo_id": str(d),
                "revision": "local",
                "dtype": P.MODELS[mid].dtype,
                "weight_files": [{"path": "model.safetensors", "bytes": 1, "sha256": "0" * 64}],
            }
            for mid, d in tiny_models.items()
        }
    }
    (out / probe_run.MANIFEST_NAME).write_text(json.dumps(manifest), encoding="utf-8")
    (out / F.ENVIRONMENT_NAME).write_text(json.dumps(probe_run.environment()), encoding="utf-8")
    return out, prompts


def _probe_table(out: Path, config_id: str) -> list[float]:
    rows = [json.loads(x) for x in (out / "scores" / f"{config_id}.jsonl").read_text().splitlines()]
    return [r["score"] for r in rows]


@pytest.mark.parametrize(
    "config_id",
    ["sg_2b.native", "lg_1b.task", "lg_12b.native", "gg_8b.task", "enc_paradetox"],
)
def test_factory_scores_bit_for_bit_like_the_probe(probe_world, config_id):
    """Gemma 2, Llama, Llama 4, Granite and the encoder: same texts, probe runner vs factory."""
    from rafr.verifiers import probe_run

    out, prompts = probe_world
    items = [(f"s{i}", t) for i, t in enumerate(TEXTS)]
    model_id, _ = F.parse_config_id(config_id)
    m = P.model(model_id)
    if m.is_lm:
        probe_run.score_lm(
            m, items, out, prompt_dir=prompts, batch_size=P.LM_BATCH_SIZE, fixed_shape=True,
            echo=lambda s: None,
        )  # fmt: skip
    else:
        probe_run.score_encoder(m, items, out, echo=lambda s: None)
    probe_scores = _probe_table(out, config_id)

    v = F.make_verifier(config_id, records_dirs=[out], prompt_dir=prompts)
    try:
        assert v.verifier_id == config_id
        assert v.model_version == f"{m.repo_id}@local"
        assert v.score([t for _, t in items]) == probe_scores  # bit for bit
        detailed = v.score_detailed([t for _, t in items])
        assert [d.score for d in detailed] == probe_scores
        assert detailed[-1].truncated and not detailed[0].truncated
        rows = [
            json.loads(x) for x in (out / "scores" / f"{config_id}.jsonl").read_text().splitlines()
        ]
        assert all(r["prompt_hash"] == v.prompt_hash for r in rows)
        assert [r["sub_scores"] for r in rows] == [d.sub_scores for d in detailed]
        info = v.describe()
        assert info["batch_size"] == (P.LM_BATCH_SIZE if m.is_lm else P.ENCODER_BATCH_SIZE)
        assert info["drift"] == {}
    finally:
        v.close()


def test_a_numerical_drift_is_refused_unless_tolerated(probe_world):
    from rafr.verifiers import probe_run

    out, prompts = probe_world
    m = P.model("lg_1b")
    probe_run.score_lm(
        m, [("a", "hello world")], out, prompt_dir=prompts, batch_size=P.LM_BATCH_SIZE,
        fixed_shape=True, echo=lambda s: None,
    )  # fmt: skip
    meta_path = out / "meta" / "lg_1b.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    meta["configs"]["lg_1b.native"]["direction"]["neutral"] = 0.123456
    meta_path.write_text(json.dumps(meta), encoding="utf-8")
    with pytest.raises(F.ProbeMismatchError, match="direction-check"):
        F.make_verifier("lg_1b.native", records_dirs=[out], prompt_dir=prompts)
    v = F.make_verifier("lg_1b.native", records_dirs=[out], prompt_dir=prompts, tolerate_drift=True)
    assert v.describe()["drift"]["direction"]["neutral"]["probe"] == 0.123456
    v.close()
    meta["configs"]["lg_1b.native"]["direction"]["neutral"] = None
    meta["attn_implementation"] = "sdpa"
    meta_path.write_text(json.dumps(meta), encoding="utf-8")
    with pytest.raises(F.ProbeMismatchError, match="attention"):
        F.make_verifier("lg_1b.native", records_dirs=[out], prompt_dir=prompts)
    # an unlocked or changed prompt is refused before anything loads
    (prompts / P.PROMPT_LOCK_FILE).unlink()
    with pytest.raises(VerifierConfigError, match="stop point A"):
        F.make_verifier("lg_1b.native", records_dirs=[out], prompt_dir=prompts)


# ---------------------------------------------------------------------------------------
# device kinds, formal mode, fixed batch shape (2026-09-24, D-2026-09-24c point 6)
# ---------------------------------------------------------------------------------------


def _set_gpu(d: Path, gpu: str | None) -> None:
    env = json.loads((d / F.ENVIRONMENT_NAME).read_text(encoding="utf-8"))
    env["gpu"] = gpu
    (d / F.ENVIRONMENT_NAME).write_text(json.dumps(env), encoding="utf-8")


def test_records_are_kept_per_device_kind_and_never_serve_the_other_kind(tmp_path):
    cuda = _records(tmp_path / "pod", "enc_paradetox", "enc_paradetox")
    cpu = _records(tmp_path / "laptop", "enc_paradetox", "enc_paradetox")
    _set_gpu(cpu, None)
    assert F.find_probe_record("enc_paradetox", [cuda, cpu], "cuda").records_dir == cuda
    assert F.find_probe_record("enc_paradetox", [cuda, cpu], "cpu").records_dir == cpu
    assert F.find_probe_record("enc_paradetox", [cpu]).device_kind == "cpu"
    with pytest.raises(F.ProbeMismatchError, match="name the device kind"):
        F.find_probe_record("enc_paradetox", [cuda, cpu])
    with pytest.raises(F.ProbeMismatchError, match="no cuda probe record"):
        F.find_probe_record("enc_paradetox", [cpu], "cuda")
    with pytest.raises(F.ProbeMismatchError, match="no cpu probe record"):
        F.find_probe_record("enc_paradetox", [cuda], "cpu")
    with pytest.raises(VerifierConfigError, match="device kind"):
        F.find_probe_record("enc_paradetox", [cuda], "mps")


def test_formal_mode_is_cuda_only_and_never_tolerates_a_drift(tmp_path, monkeypatch):
    d = _records(tmp_path, "enc_paradetox", "enc_paradetox")
    with pytest.raises(VerifierConfigError, match="never tolerates"):
        F.make_verifier("enc_paradetox", records_dirs=[d], formal=True, tolerate_drift=True)
    with pytest.raises(VerifierConfigError, match="GPU only"):
        F.make_verifier("enc_paradetox", records_dirs=[d], formal=True, device="cpu")
    torch = pytest.importorskip("torch")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(VerifierConfigError, match="no CUDA device"):
        F.make_verifier("enc_paradetox", records_dirs=[d], formal=True)
    # with a GPU present, a formal run takes the cuda record and refuses a cpu-only one
    monkeypatch.setattr(F, "require_cuda", lambda: None)
    _set_gpu(d, None)
    with pytest.raises(F.ProbeMismatchError, match="no cuda probe record"):
        F.make_verifier("enc_paradetox", records_dirs=[d], formal=True)


def test_the_probe_must_have_run_with_a_fixed_batch_shape(tmp_path):
    d = _records(tmp_path, "sg_2b", "sg_2b.native", fixed_shape=False)
    with pytest.raises(F.ProbeMismatchError, match="fixed batch shape"):
        F.static_checks(F.config_for("sg_2b.native"), F.find_probe_record("sg_2b", [d]))


@pytest.mark.parametrize("config_id", ["lg_1b.native", "enc_paradetox"])
def test_a_short_batch_is_filled_up_to_the_probe_batch_size_and_the_filler_dropped(
    probe_world, config_id
):
    """Every forward pass has the probe's batch size; the short last batch is filled with filler
    rows whose scores are dropped, so a text scores the same whatever its neighbours."""
    from rafr.verifiers import probe_run

    out, prompts = probe_world
    model_id, _ = F.parse_config_id(config_id)
    m = P.model(model_id)
    items = [(f"s{i}", t) for i, t in enumerate(TEXTS)]
    if m.is_lm:
        probe_run.score_lm(
            m, items, out, prompt_dir=prompts, batch_size=P.LM_BATCH_SIZE, fixed_shape=True,
            echo=lambda s: None,
        )  # fmt: skip
    else:
        probe_run.score_encoder(m, items, out, echo=lambda s: None)
    v = F.make_verifier(config_id, records_dirs=[out], prompt_dir=prompts)
    try:
        model = v.scorer.backend.model if m.is_lm else v.encoder.backend.model
        rows: list[int] = []
        forward = model.forward

        def spy(*a, **k):
            ids = k.get("input_ids", a[0] if a else None)
            rows.append(int(ids.shape[0]))
            return forward(*a, **k)

        model.forward = spy
        full = v.score(list(TEXTS))
        rows.clear()
        three = v.score(list(TEXTS[:3]))
        want = P.LM_BATCH_SIZE if m.is_lm else P.ENCODER_BATCH_SIZE
        assert rows and set(rows) == {want}  # 3 texts -> one forward pass of the full batch
        assert len(three) == 3 and three == full[:3]  # filler dropped; neighbours do not matter
    finally:
        v.close()
