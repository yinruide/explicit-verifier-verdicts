"""core: seeding determinism, config loading, formal-run protection."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from rafr.core import config as C
from rafr.core import paths, seeding


def test_derive_seed_is_stable_and_key_sensitive():
    a = seeding.derive_seed(7, "channel", "state-1", "low")
    b = seeding.derive_seed(7, "channel", "state-1", "low")
    c = seeding.derive_seed(7, "channel", "state-1", "mid")
    d = seeding.derive_seed(8, "channel", "state-1", "low")
    assert a == b
    assert len({a, c, d}) == 3
    assert 0 <= a < 2**63
    # regression pin: cross-platform stability of the derivation
    assert seeding.derive_seed(0, "x") == seeding.derive_seed(0, "x")


def test_attempt_seed_changes_with_attempt():
    s0 = seeding.attempt_seed(1, "x0-1", attempt=0)
    s1 = seeding.attempt_seed(1, "x0-1", attempt=1)
    assert s0 != s1


def test_rng_reproducible():
    r1 = seeding.rng(3, "a").random(3).tolist()
    r2 = seeding.rng(3, "a").random(3).tolist()
    assert r1 == r2


def test_repo_configs_load_and_every_tbd_has_a_comment_owner():
    cfgs = C.load_config()
    assert set(C.CONFIG_FILES) <= set(cfgs)
    # every value is decided since 2026-09-28: the last TBD, the Exp 2 semantic cutpoint, was
    # computed on D_fit by step12_exp2.py cutpoint (D-2026-09-28)
    assert C.find_tbd(cfgs) == []
    # the locked constants are present as literals in configs and agree with contracts
    assert cfgs["exp2"]["design"]["k"] == 4
    assert cfgs["exp1"]["controller"]["parse_max_retries"] == 2
    assert cfgs["exp1"]["controller"]["parse_fail_alert_threshold"] == 0.02
    assert cfgs["channel"]["gate"]["max_attempts"] == 2


def test_find_tbd_paths():
    obj = {"a": "TBD", "b": {"c": "TBD", "d": 1}, "e": ["x", "TBD"], "f": "tbd"}
    assert C.find_tbd(obj) == ["a", "b.c", "e[1]"]


def test_assert_frozen_lists_everything(tmp_path: Path):
    prompt = tmp_path / "p.txt"
    prompt.write_text("hello PLACEHOLDER", encoding="utf-8")
    cfgs = {"models": {"controller": {"model": "TBD", "system_prompt_file": str(prompt)}}}
    with pytest.raises(C.NotFrozenError) as ei:
        C.assert_frozen(cfgs)
    err = ei.value
    assert err.tbd == ["models.controller.model"]
    assert err.placeholders == [str(prompt)]
    assert "models.controller.model" in str(err)
    assert str(prompt) in str(err)


def test_assert_frozen_dry_run_exception(tmp_path: Path):
    prompt = tmp_path / "p.txt"
    prompt.write_text("hello PLACEHOLDER", encoding="utf-8")
    cfgs = {"models": {"controller": {"model": "TBD", "system_prompt_file": str(prompt)}}}
    C.assert_frozen(cfgs, allow_placeholders=True, allow_tbd=True)  # dry run only


def test_assert_frozen_missing_prompt_file_blocks(tmp_path: Path):
    cfgs = {"models": {"controller": {"model": "x", "system_prompt_file": str(tmp_path / "no")}}}
    with pytest.raises(C.NotFrozenError):
        C.assert_frozen(cfgs)


def test_assert_frozen_only_sections(tmp_path: Path):
    ok_prompt = tmp_path / "ok.txt"
    ok_prompt.write_text("real prompt", encoding="utf-8")
    cfgs = {
        "models": {
            "controller": {
                "model": "gpt",
                "system_prompt_file": str(ok_prompt),
                "rewrite_prompt_file": str(ok_prompt),
                "parse_retry_prompt_file": str(ok_prompt),
            },
            "evaluator": {"model": "TBD"},
        }
    }
    C.assert_frozen(cfgs, only_sections=["models.controller"])
    with pytest.raises(C.NotFrozenError):
        C.assert_frozen(cfgs, only_sections=["models.evaluator"])


def test_prompt_files_are_an_explicit_list_not_a_key_suffix(tmp_path: Path):
    """Decision (2026-09-17): the guard checks PROMPT_FILES only. A key that merely ends in
    `_file` (a data file that need not exist yet) is ignored; a listed entry is checked."""
    good = tmp_path / "ok.txt"
    good.write_text("real prompt", encoding="utf-8")
    cfgs = {
        "models": {
            "controller": {
                "model": "x",
                "system_prompt_file": str(good),
                "rewrite_prompt_file": str(good),
                "parse_retry_prompt_file": str(good),
            },
            "g_fixed": {"prompt_file": str(good)},
            "evaluator": {"prompt_file": str(good), "scale_file": str(good)},
        },
        "generation": {"g_fixed": {"validity": {"refusal_phrases_file": str(good)}}},
        "exp1": {"outcome": {"judgments_file": str(tmp_path / "not_yet_written.jsonl")}},
        "channel": {"gate": {"attempts_file": str(tmp_path / "missing.json")}},
    }
    C.assert_frozen(cfgs)  # data-file keys are not prompts
    assert C.prompt_files(cfgs) == [str(good)] * len(C.PROMPT_FILES)
    assert C.prompt_files(cfgs, only_sections=["models.controller"]) == [str(good)] * 3
    assert C.prompt_files(cfgs, only_sections=["channel"]) == []
    # a listed entry pointing at a placeholder / missing file blocks
    cfgs["models"]["evaluator"]["prompt_file"] = str(tmp_path / "absent.txt")
    with pytest.raises(C.NotFrozenError) as ei:
        C.assert_frozen(cfgs)
    assert ei.value.placeholders == [str(tmp_path / "absent.txt")]
    C.assert_frozen(cfgs, only_sections=["models.controller"])  # evaluator not consumed


def test_prompt_files_missing_entry_is_reported_not_skipped(tmp_path: Path):
    """Review (2026-09-18): a PROMPT_FILES entry that the configs do not set at all used to
    be skipped silently, so a config without `rewrite_prompt_file` passed the guard as if
    that prompt had been checked. It is now a blocking item of its own."""
    good = tmp_path / "ok.txt"
    good.write_text("real prompt", encoding="utf-8")
    cfgs = {
        "models": {
            "controller": {"model": "x", "system_prompt_file": str(good)},
            "evaluator": {"prompt_file": str(good)},
        }
    }
    with pytest.raises(C.NotFrozenError) as ei:
        C.prompt_files(cfgs, only_sections=["models.controller"])
    assert ei.value.missing == [
        "models.controller.rewrite_prompt_file",
        "models.controller.parse_retry_prompt_file",
    ]
    with pytest.raises(C.NotFrozenError) as ei2:
        C.assert_frozen(cfgs)
    err = ei2.value
    assert err.tbd == [] and err.placeholders == []
    assert set(err.missing) == set(C.PROMPT_FILES) - {
        "models.controller.system_prompt_file",
        "models.evaluator.prompt_file",
    }
    assert "models.g_fixed.prompt_file" in str(err)
    # the dry run may still pass (placeholders allowed = prompt entries not checked)
    C.assert_frozen(cfgs, allow_placeholders=True, allow_tbd=True)
    # an entry outside the requested sections is not in scope and not reported
    C.assert_frozen(cfgs, only_sections=["models.evaluator.prompt_file"])


def test_prompt_files_list_matches_the_real_configs():
    """Every listed entry exists in configs/ and every `_file` key in configs/ that is a
    prompt / template / phrase file is listed (data files are named in the exclusion)."""
    cfgs = C.load_config()
    for dotted in C.PROMPT_FILES:
        cname, _, rest = dotted.partition(".")
        assert isinstance(C.get(cfgs[cname], rest), str), f"{dotted} missing from configs/"
    data_files = {"exp1.outcome.judgments_file"}  # written by a step, never a prompt

    def walk(obj, prefix):
        if isinstance(obj, dict):
            for k, v in obj.items():
                yield from walk(v, f"{prefix}.{k}")
        elif prefix.endswith("_file"):
            yield prefix

    found = {p for name, cobj in cfgs.items() for p in walk(cobj, name)}
    assert found == set(C.PROMPT_FILES) | data_files
    assert set(C.prompt_files(cfgs)) == {
        C.get(cfgs[d.partition(".")[0]], d.partition(".")[2]) for d in C.PROMPT_FILES
    }


def test_current_repo_state_is_frozen():
    """Every formal step's guard passes on the whole config since 2026-09-28 (the last TBD, the
    Exp 2 semantic cutpoint, computed by step12_exp2.py cutpoint). The seven prompt / phrase
    files of PROMPT_FILES were filled by the authors on 2026-09-18; the formatter templates (checked
    via ``extra_files`` by step03/11/12) since the authors filled in the two checker descriptions on
    2026-09-25 (D-2026-09-25e)."""
    from rafr.formatter import TEMPLATES_PATH

    cfgs = C.load_config()
    C.assert_frozen(cfgs)
    assert C.find_placeholders(C.prompt_files(cfgs)) == []
    C.assert_frozen(cfgs, extra_files=[TEMPLATES_PATH])


def test_paths_resolve_relative_to_root():
    assert paths.resolve("configs/models.yaml") == paths.ROOT / "configs" / "models.yaml"
    assert (paths.CONFIGS / "models.yaml").exists()


def test_get_dotted():
    assert C.get({"a": {"b": {"c": 3}}}, "a.b.c") == 3
    assert C.get({"a": {}}, "a.b.c", default="d") == "d"


def test_yaml_files_are_valid(tmp_path: Path):
    for name in C.CONFIG_FILES:
        with (paths.CONFIGS / f"{name}.yaml").open("r", encoding="utf-8") as fh:
            assert isinstance(yaml.safe_load(fh), dict)
