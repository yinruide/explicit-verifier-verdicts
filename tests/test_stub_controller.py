"""The always-rewrite stub controller (2026-09-25 revision): rounds 3 and 4, the
terminal text x_5 when nothing is accepted, x_5 = x_4 when the fourth decision is a parse_fail, a
bit-identical resume from any round, and the formal steps' refusal of a ``stub-`` controller. No
API call is made anywhere here."""

from __future__ import annotations

import shutil
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

import pytest

from rafr.channel.identity import IdentityChannel
from rafr.contracts import schemas
from rafr.contracts.schemas import Arm, EvalRecord, SafetyLabel, State
from rafr.controller import stub as cstub
from rafr.experiments import exp2
from rafr.experiments.common import InProcessScorer
from rafr.llm.client import LLMClient
from rafr.llm.providers.fake import FakeProvider
from rafr.logging.writer import read_jsonl
from rafr.verifiers.stub import stub_pair


def _states(n: int) -> list[State]:
    return [
        State(state_id=f"x0-{i}-g1", x0_id=f"x0-{i}", x0=f"orig {i} idiot", text=f"state text {i}")
        for i in range(n)
    ]


def _eval(text: str, label: SafetyLabel = SafetyLabel.UNSAFE) -> EvalRecord:
    return EvalRecord(
        text_id="h",
        text=text,
        label=label,
        model="e",
        model_version=None,
        prompt_hash="p",
        e_prompt_sha256="h",
        e_version="v",
        decoding={},
    )


def _setup(run_dir: Path, states: list[State]) -> exp2.Exp2Setup:
    v1, v2 = stub_pair()
    return exp2.Exp2Setup(
        run_id="stub",
        run_dir=run_dir,
        x1_states=states,
        scorer=InProcessScorer(v1, v2),
        tau1=0.5,
        tau2=0.5,
        channel=IdentityChannel(),
        arms=[Arm.A1, Arm.B],
        k=4,
        assignment_seed=3,
        run_seed=1,
        judge_fn=lambda text: _eval(text, SafetyLabel.UNSAFE),
    )


def _client(responder) -> LLMClient:
    return LLMClient(providers={"fake": FakeProvider(responder=responder)})


def test_stub_controller_runs_every_trajectory_to_round_4_and_x_t_is_x5(tmp_path: Path):
    """A controller that never accepts: all four rounds run, every trajectory ends unaccepted at
    t = 5 with x_T = x_5, the fourth rewrite, which is judged by E but never scored (paper Sec.
    4.4)."""
    states = _states(5)
    ccfg = cstub.stub_controller_config()
    assert cstub.is_stub_controller(ccfg.model) and ccfg.model.startswith("stub-")
    setup = _setup(tmp_path / "a", states)
    trajs = exp2.run_loop(_client(cstub.always_rewrite_responder()), ccfg, setup)
    assert exp2.complete_rounds(tmp_path / "a", 4) == [1, 2, 3, 4]
    for tr in trajs:
        assert not tr.accepted and tr.accepted_at is None
        assert tr.t == 5 and tr.n_rewrites == 4 and tr.n_parse_fail == 0
        assert [h["t"] for h in tr.history] == [1, 2, 3, 4]
        assert all(h["action"] == "rewrite" and not h["parse_fail"] for h in tr.history)
        assert cstub.rewrites_so_far(tr.x_t) == 4  # x_5 carries four rewrite marks
        assert tr.x_t.startswith(tr.history[0]["x_t"])  # x_1 is the G_fixed state itself
    rows = list(read_jsonl(tmp_path / "a" / "decisions.jsonl"))
    assert Counter(int(r["t"]) for r in rows) == {1: 5, 2: 5, 3: 5, 4: 5}
    assert all(r["model"] == cstub.STUB_CONTROLLER_MODEL for r in rows)
    # x_5 is never scored: the score rows stop at t = 4
    scored_t = {
        int(r["state_id"].rsplit("-t", 1)[1]) for r in read_jsonl(tmp_path / "a" / "scores.jsonl")
    }
    assert scored_t == {1, 2, 3, 4}
    exp2.finish(setup, trajs)  # E's round judgments and the finals (trajectories.jsonl)
    finals = list(read_jsonl(tmp_path / "a" / "trajectories.jsonl"))
    for f, tr in zip(finals, trajs, strict=True):
        assert f["x_T"] == tr.x_t and not f["accepted"] and f["n_rewrites"] == 4
        assert set(f["round_e_labels"]) == {"1", "2", "3", "4", "5"}  # terminal t = K + 1
    rj = list(read_jsonl(tmp_path / "a" / exp2.ROUND_JUDGMENTS))
    assert {r["t"] for r in rj if r["terminal"]} == {5}


def test_stub_controller_parse_fail_at_round_4_leaves_x5_equal_to_x4(tmp_path: Path):
    """paper Sec. 4.4: when the fourth decision is a parse_fail, x_5 = x_4 -- the round is consumed
    (paper Sec. 4.6), no new text appears, and the terminal text is x_4."""
    states = _states(4)
    ccfg = cstub.stub_controller_config()
    setup = _setup(tmp_path / "b", states)
    trajs = exp2.run_loop(
        _client(cstub.always_rewrite_responder(parse_fail_at_round=4)), ccfg, setup
    )
    for tr in trajs:
        assert not tr.accepted and tr.t == 5
        assert tr.n_rewrites == 3 and tr.n_parse_fail == 1
        assert [h["parse_fail"] for h in tr.history] == [False, False, False, True]
        assert tr.x_t == tr.history[3]["x_t"]  # x_5 = x_4
        assert cstub.rewrites_so_far(tr.x_t) == 3
    rows = [r for r in read_jsonl(tmp_path / "b" / "decisions.jsonl") if int(r["t"]) == 4]
    assert rows and all(r["parse_fail"] and r["action"] is None for r in rows)
    assert all(r["n_parse_retries"] == 2 and len(r["attempts"]) == 3 for r in rows)
    assert all(all(a["parse_fail_reason"] for a in r["attempts"]) for r in rows)
    exp2.finish(setup, trajs)
    finals = list(read_jsonl(tmp_path / "b" / "trajectories.jsonl"))
    for f, tr in zip(finals, trajs, strict=True):
        assert f["x_T"] == tr.history[3]["x_t"] and f["first_parse_fail_t"] == 4
        assert f["n_parse_fail"] == 1 and f["n_rewrites"] == 3


def _freeze_time(monkeypatch) -> None:
    """Every timestamp the runners write goes through schemas.utc_now_iso, which reads the
    module's ``datetime``; pin it so two runs can be compared byte for byte."""
    fixed = datetime(2026, 9, 25, 12, 0, 0, tzinfo=UTC)

    class Frozen(datetime):
        @classmethod
        def now(cls, tz=None):  # noqa: ARG003
            return fixed

    monkeypatch.setattr(schemas, "datetime", Frozen)


@pytest.mark.parametrize("resume_after", [1, 2, 3, 4])
def test_resume_from_any_round_is_bit_identical_to_one_pass(
    tmp_path: Path, monkeypatch, resume_after: int
):
    """paper Sec. 4.4: a run cut off after any complete round and resumed produces exactly the
    files of one uninterrupted run -- every round directory, the assembled streams, the loop
    marker -- byte for byte (timestamps pinned; a parse_fail on the way at round 3)."""
    _freeze_time(monkeypatch)
    states = _states(6)
    ccfg = cstub.stub_controller_config()
    responder = cstub.always_rewrite_responder(parse_fail_at_round=3)
    one = tmp_path / "one"
    exp2.run_loop(_client(responder), ccfg, _setup(one, states))
    assert exp2.complete_rounds(one, 4) == [1, 2, 3, 4]

    two = tmp_path / "two"
    exp2.run_loop(_client(responder), ccfg, _setup(two, states))
    # cut run two back to `resume_after` complete rounds, as an interrupted pod leaves it
    for t in range(resume_after + 1, 5):
        shutil.rmtree(exp2.round_dir(two, t))
    for name in ("decisions.jsonl", "scores.jsonl", exp2.LOOP_COMPLETE):
        (two / name).unlink()
    assert exp2.complete_rounds(two, 4) == list(range(1, resume_after + 1))
    exp2.run_loop(_client(responder), ccfg, _setup(two, states))

    files = sorted(p.relative_to(one) for p in one.rglob("*") if p.is_file())
    assert files == sorted(p.relative_to(two) for p in two.rglob("*") if p.is_file())
    assert any(f.as_posix().endswith("round_4/trajectories.jsonl") for f in files)
    for rel in files:
        assert (one / rel).read_bytes() == (two / rel).read_bytes(), rel


def test_formal_steps_refuse_a_stub_controller(sandbox: Path, monkeypatch):
    """a controller whose model id starts with stub- is for tests only; step 11 and both
    of step 12's loops refuse it before any other precondition, as they refuse a stub verifier."""
    from pipelines import step11_exp1 as s11
    from pipelines import step12_exp2 as s12

    from rafr.core.config import load_config

    cfgs = load_config()
    cfgs["models"]["controller"]["model"] = cstub.STUB_CONTROLLER_MODEL
    for mod in (s11, s12):
        monkeypatch.setattr(mod, "load_config", lambda: cfgs)
        monkeypatch.setattr(mod, "assert_frozen", lambda *a, **k: None)
    with pytest.raises(SystemExit, match="stub controller"):
        s11.main(["--confirm"])
    base = ["loop", "--run-id", "f", "--confirm", "--x1-file", "x", "--x1-sha256", "y"]
    with pytest.raises(SystemExit, match="stub controller"):
        s12.main(base)
    with pytest.raises(SystemExit, match="stub controller"):
        s12.main([*base, "--smoke"])
    assert cstub.is_stub_controller("stub-anything") and not cstub.is_stub_controller(
        "gpt-5.6-terra"
    )
    assert not cstub.is_stub_controller(None)
    from rafr.verifiers.stub import STUB_PREFIX

    assert cstub.STUB_PREFIX == STUB_PREFIX  # one rule for verifiers and controller
