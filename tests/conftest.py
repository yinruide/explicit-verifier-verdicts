"""Shared fixtures. Tests are offline: fake provider + stub verifier only."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from rafr.core import paths
from rafr.llm.cache import LLMCache
from rafr.llm.client import LLMClient
from rafr.llm.ledger import Ledger
from rafr.llm.providers import MissingAPIKeyError, load_api_key
from rafr.llm.providers import anthropic as anthropic_provider
from rafr.llm.providers import openai as openai_provider
from rafr.llm.providers import openai_compatible as openai_compatible_provider
from rafr.llm.providers.fake import FakeProvider

FIXTURES = Path(__file__).parent / "fixtures"

WATCHED = ("data", "frozen", "archive")
IGNORED = ("data/cache", "data/raw", "archive/round1/data/cache")
"""Everything a run writes and cannot cheaply reproduce, plus the round-1 archive, which
nothing may write at all (protocol v0.9, II.10). The caches are skipped: walking 8K cache
files per test would be slow, and the archived one is checked against its sha256 list in
archive/round1/README.md instead. Watching the whole of data/ rather than a list of
subdirectories is deliberate -- the escape this guard was written for landed in data/_fake/, a
directory no list would have had on it yet."""


def _snapshot_formal_paths() -> dict[str, tuple[int, int]]:
    """(size, mtime) of every file under WATCHED, resolved against the REAL repo root.

    Deliberately independent of ``paths.DATA`` and friends: those are what a test redirects,
    and this has to notice when a redirect is missing.
    """
    root = paths.repo_root()
    skip = {str(root / rel) for rel in IGNORED}
    out: dict[str, tuple[int, int]] = {}
    for rel in WATCHED:
        top = root / rel
        if not top.exists():
            continue
        if top.is_file():
            st = top.stat()
            out[str(top.relative_to(root))] = (st.st_size, st.st_mtime_ns)
            continue
        for dirpath, dirnames, filenames in os.walk(top):
            # prune, do not filter afterwards: data/cache holds one file per LLM call, and
            # walking it 360 times over takes minutes
            dirnames[:] = [d for d in dirnames if os.path.join(dirpath, d) not in skip]
            for name in filenames:
                q = Path(dirpath) / name
                st = q.stat()
                out[str(q.relative_to(root))] = (st.st_size, st.st_mtime_ns)
    return out


@pytest.fixture(autouse=True)
def no_formal_data_writes():
    """Fail a test that writes into the repo's real data/ or frozen/ (2026-09-20).

    A --fake step00 run overwrote a finished formal run's splits that day. The step itself is
    fixed (a fake run is redirected under data/_fake and checks the redirect), but the same
    accident can arrive from a test that forgets the ``sandbox`` fixture, which is exactly
    how nine live API calls happened on 2026-09-20. This is the net under both.
    """
    before = _snapshot_formal_paths()
    yield
    after = _snapshot_formal_paths()
    if before != after:
        changed = sorted(set(before) ^ set(after)) or sorted(
            k for k in before if before[k] != after.get(k)
        )
        raise AssertionError(
            "this test wrote into the repo's real data/ or frozen/: "
            f"{changed[:10]}. Use the `sandbox` fixture (it redirects every path in "
            "paths._REDIRECTED to tmp)."
        )


@pytest.fixture
def sandbox(tmp_path: Path):
    """Redirect every data/runs/frozen path to tmp so pipeline tests never touch the repo.

    Undone at teardown, which also undoes a redirect the step did itself
    (``paths.use_fake_data_root()``), so a --fake run in one test cannot leak into the next.

    It uses its OWN MonkeyPatch rather than the test's (2026-09-20): with the shared
    one, a ``monkeypatch.undo()`` inside a test silently un-sandboxes the rest of it, and the
    next --fake run writes into the real repo. That is how data/_fake appeared in the working
    tree while this very guard was being written.
    """
    with pytest.MonkeyPatch.context() as mp:
        yield _sandbox_paths(mp, tmp_path)


def _sandbox_paths(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    data = tmp_path / "data"
    monkeypatch.setattr(paths, "DATA", data)
    for name in (
        "DATA_RAW",
        "DATA_POOLS",
        "DATA_SPLITS",
        "DATA_LABELS",
        "DATA_SCORES",
        "DATA_CACHE",
    ):
        monkeypatch.setattr(paths, name, data / name.split("_", 1)[1].lower())
    monkeypatch.setattr(paths, "MANIFEST", data / "manifest.json")
    monkeypatch.setattr(paths, "RUNS", tmp_path / "runs")
    monkeypatch.setattr(paths, "FROZEN", tmp_path / "frozen")
    monkeypatch.setattr(paths, "FROZEN_GATE_ATTEMPTS", tmp_path / "gate_attempts.json")
    monkeypatch.setattr(paths, "FROZEN_TAU", tmp_path / "frozen" / "tau.json")
    monkeypatch.setattr(paths, "FROZEN_CHANNEL", tmp_path / "frozen" / "channel.json")
    # every path a fake run redirects must also be sandboxed here, or a leaked redirect
    # would survive the test (2026-09-20)
    patched = {"DATA", "MANIFEST", "RUNS", "FROZEN"} | {
        n for n in paths._REDIRECTED if n.startswith(("DATA_", "FROZEN_"))
    }
    assert set(paths._REDIRECTED) <= patched, set(paths._REDIRECTED) - patched
    return tmp_path


@pytest.fixture(autouse=True)
def no_live_api_keys(monkeypatch):
    """Make the "tests are offline" rule above enforceable rather than a convention.

    A pipeline step run without ``--fake`` builds the real providers and spends money. On
    2026-09-20, filling the last evaluator TBD turned a stale "this step still refuses"
    test into nine live E calls before anyone noticed. Reading the repo ``.env`` now raises
    instead; an explicit ``env_file`` (the load_api_key unit test) still works, and providers
    constructed with ``api_key=`` never reach this path.
    """

    def blocked(var_name: str, env_file: Path | None = None) -> str:
        if env_file is not None:
            return load_api_key(var_name, env_file=env_file)
        raise MissingAPIKeyError(
            f"refusing to read {var_name} from the repo .env inside the test suite: "
            "tests are offline (use --fake / the fake provider)"
        )

    for mod in (anthropic_provider, openai_provider, openai_compatible_provider):
        monkeypatch.setattr(mod, "load_api_key", blocked)


@pytest.fixture
def make_symlink():
    """``make_symlink(link, target, target_is_directory=False)``: create a symlink, or skip.

    A test that builds a symlink to check that the code refuses one cannot run on a Windows
    account without the symlink privilege (WinError 1314). That is the environment, not the
    code under test (no pipeline creates symlinks), so the test is skipped there (2026-11-05).
    """

    def make(link: Path, target: Path, *, target_is_directory: bool = False) -> None:
        try:
            Path(link).symlink_to(target, target_is_directory=target_is_directory)
        except OSError as exc:
            if os.name == "nt" and getattr(exc, "winerror", None) == 1314:
                pytest.skip("environment: this Windows account may not create symlinks (1314)")
            raise

    return make


MATH_EXT_BINDINGS = (
    ("source_manifest",),
    ("calibration", "frozen_sha256"),
    ("cohorts", "e_correct", "file_sha256"),
    ("exp1", "e_incorrect", "original_sha256", "decisions.jsonl"),
    ("exp1", "e_incorrect", "original_sha256", "channel.jsonl"),
    ("exp1", "e_incorrect", "original_sha256", "permutation.json"),
    ("exp2", "e_incorrect", "original_sha256", "assignment.json"),
    ("exp2", "e_incorrect", "original_sha256", "trajectories.jsonl"),
    ("exp2", "e_incorrect", "x1_sha256"),
    ("exp2", "e_incorrect", "t1_scores", "source_sha256"),
    ("analysis", "cited_not_recomputed", "exp1", "e_incorrect", "sha256", "T06_exp1_did.json"),
    (
        "analysis",
        "cited_not_recomputed",
        "exp1",
        "e_incorrect",
        "sha256",
        "T06_exp1_delta_by_lambda.json",
    ),
)


@pytest.fixture
def math_ext_unbound():
    """``math_ext_unbound(bindings=all)``: configs/math/ext.yaml with those artifact bindings
    back to TBD, as before they were filled (2026-11-05). The repository config is bound, so a
    test of the missing-binding guards builds its unbound config here; ``.bindings`` lists them.
    """
    import yaml

    def unbound(bindings=MATH_EXT_BINDINGS):
        cfg = yaml.safe_load((paths.ROOT / "configs/math/ext.yaml").read_text(encoding="utf-8"))
        for *parents, leaf in bindings:
            node = cfg
            for key in parents:
                node = node[key]
            assert leaf in node, (parents, leaf)
            node[leaf] = "TBD"
        return cfg

    unbound.bindings = MATH_EXT_BINDINGS
    return unbound


@pytest.fixture
def fake_provider() -> FakeProvider:
    return FakeProvider()


@pytest.fixture
def llm(tmp_path: Path, fake_provider: FakeProvider) -> LLMClient:
    return LLMClient(
        providers={"fake": fake_provider},
        cache=LLMCache(tmp_path / "cache"),
        ledger=Ledger(tmp_path / "ledger.jsonl"),
    )


@pytest.fixture
def paradetox_sample() -> list[dict]:
    """Small synthetic sample in ParaDetox column layout (never real data)."""
    with (FIXTURES / "paradetox_sample.jsonl").open("r", encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]
