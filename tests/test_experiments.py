"""experiments: Exp 1 shared perturbation across arms + E-first + resume; Exp 2 loop rules."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest

from rafr.channel.identity import IdentityChannel
from rafr.contracts.schemas import Arm, EvalRecord, Lambda, SafetyLabel, State
from rafr.controller import decide as ctrl
from rafr.core.config import load_config
from rafr.core.seeding import derive_seed
from rafr.experiments import exp1, exp2
from rafr.experiments.common import InProcessScorer
from rafr.llm.client import LLMClient, canonical_json, sha256_hex
from rafr.llm.providers.fake import FakeProvider
from rafr.logging.writer import read_jsonl
from rafr.verifiers.stub import stub_pair


def _states(n: int) -> list[State]:
    return [
        State(state_id=f"x0-{i}-g1", x0_id=f"x0-{i}", x0=f"orig {i} idiot", text=f"state text {i}")
        for i in range(n)
    ]


def _eval(text: str, label=SafetyLabel.UNSAFE) -> EvalRecord:
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


def _ccfg(scheme: str = "per_state") -> ctrl.ControllerConfig:
    models = {
        "controller": {
            "provider": "fake",
            "model": "ctrl",
            "version": "v",
            "decoding": {"temperature": 0.2, "seed": 1},
            "seed_scheme": scheme,
            "network": {},
        }
    }
    return ctrl.build_config(
        models,
        {},
        base_seed=1,
        system_prompt="S",
        rewrite_template="{x0}|{x_t}|{observation}",
        retry_prompt="R",
    )


class _Noisy:
    """Deterministic stand-in for C_lambda that actually changes the scores."""

    name = "noisy"

    def __call__(self, state_id, s1, s2, lam):
        d = {Lambda.LOW: 0.01, Lambda.MID: 0.05, Lambda.HIGH: 0.2}[lam]
        return (min(1.0, s1 + d), max(0.0, s2 - d))


def _accept_responder(r):
    return json.dumps({"action": "accept", "revised_text": ""})


def _setup1(tmp_path: Path, states, e_labels, **kw) -> exp1.Exp1Setup:
    v1, v2 = stub_pair()
    base = dict(
        run_id="t",
        run_dir=tmp_path / "exp1",
        states=states,
        scorer=InProcessScorer(v1, v2),
        tau1=0.5,
        tau2=0.5,
        channel=_Noisy(),
        lambdas=[Lambda.LOW, Lambda.MID, Lambda.HIGH],
        arms=[Arm.A1, Arm.B, Arm.PERMUTED_B],
        perm_seed=7,
        run_seed=1,
        e_labels=e_labels,
    )
    base.update(kw)
    return exp1.Exp1Setup(**base)


def test_exp1_requires_e_judgments_before_any_controller_call(tmp_path: Path):
    states = _states(3)
    fp = FakeProvider(responder=_accept_responder)
    client = LLMClient(providers={"fake": fp})
    with pytest.raises(exp1.Exp1Error):
        exp1.run_exp1(client, _ccfg(), _setup1(tmp_path, states, {states[0].state_id: _eval("a")}))
    assert fp.calls == []  # nothing was sent


def test_exp1_full_cross_shares_perturbation_and_logs_everything(tmp_path: Path):
    states = _states(4)
    e = {s.state_id: _eval(s.text) for s in states}
    fp = FakeProvider(responder=_accept_responder)
    client = LLMClient(providers={"fake": fp})
    summary = exp1.run_exp1(client, _ccfg(), _setup1(tmp_path, states, e))
    rows = list(read_jsonl(tmp_path / "exp1" / "decisions.jsonl"))
    assert len(rows) == 4 * 3 * 3 and summary["n_new_decisions"] == 36
    # same (state, lambda) -> identical (s1~, s2~) across the three arms; raw also logged
    by = {}
    for r in rows:
        by.setdefault((r["state_id"], r["lam"]), set()).add((r["s1_tilde"], r["s2_tilde"]))
        assert r["s1"] is not None and r["s1_tilde"] != r["s1"]
        assert r["e_label"] == "UNSAFE" and r["tau1"] == 0.5
    assert all(len(v) == 1 for v in by.values())
    # all arms of a cell saw the same A1 prefix; B carries fields; permuted_B carries a donor
    perm = [r for r in rows if r["arm"] == "permuted_B"]
    assert all(r["permuted_from_state_id"] not in (None, r["state_id"]) for r in perm)
    assert (tmp_path / "exp1" / "permutation.jsonl").exists()
    assert len(list(read_jsonl(tmp_path / "exp1" / "channel.jsonl"))) == 4 * 3
    assert len(list(read_jsonl(tmp_path / "exp1" / "scores.jsonl"))) == 4 * 2
    # FA = P(accept | UNSAFE) = 1 here (fake always accepts)
    assert all(c["false_accept_rate"] == 1.0 for c in summary["cells"].values())
    assert len(summary["cells"]) == 9


def test_exp1_permuted_b_keeps_one_donor_across_lambdas_and_shows_its_current_block(
    tmp_path: Path, monkeypatch
):
    """the permuted-B table is keyed by the seed only -- a state has the same
    donor at all three levels, never itself, and at each level it shows the donor's six fields
    OF THAT LEVEL. The table is written once and read back on resume."""
    states = _states(5)
    e = {s.state_id: _eval(s.text) for s in states}
    seen: list[tuple] = []
    real = exp1.format_observation

    def spy(arm, obs, *, permuted_fields=None):
        seen.append((arm, obs, permuted_fields))
        return real(arm, obs, permuted_fields=permuted_fields)

    monkeypatch.setattr(exp1, "format_observation", spy)
    client = LLMClient(providers={"fake": FakeProvider(responder=_accept_responder)})
    exp1.run_exp1(client, _ccfg(), _setup1(tmp_path, states, e))
    rows = list(read_jsonl(tmp_path / "exp1" / "decisions.jsonl"))
    donors: dict[str, set[str]] = {}
    for r in rows:
        if r["arm"] == "permuted_B":
            donors.setdefault(r["state_id"], set()).add(r["permuted_from_state_id"])
    assert len(donors) == 5
    assert all(len(d) == 1 and r not in d for r, d in donors.items())  # one donor, not itself
    own = {(o.state_id, o.lam): o for arm, o, _ in seen if arm is Arm.A1}
    donor_of = {sid: next(iter(d)) for sid, d in donors.items()}
    shown = [(o, f) for arm, o, f in seen if arm is Arm.PERMUTED_B]
    assert len(shown) == 5 * 3
    for obs, fields in shown:
        donor_obs = own[(donor_of[obs.state_id], obs.lam)]
        assert fields == exp1.compute_fields(donor_obs)  # the donor's block at this lambda
    table = list(read_jsonl(tmp_path / "exp1" / "permutation.jsonl"))
    assert len(table) == 5 and all("lam" not in r and r["seed"] == 7 for r in table)
    assert {r["state_id"]: r["donor"] for r in table} == donor_of
    # read back on resume; a table of another seed is refused
    assert exp1.permutation_for(
        tmp_path / "exp1" / "permutation.jsonl", [s.state_id for s in states], 7
    ) == donor_of  # fmt: skip
    with pytest.raises(exp1.Exp1Error, match="seed"):
        exp1.permutation_for(
            tmp_path / "exp1" / "permutation.jsonl", [s.state_id for s in states], 8
        )


def test_exp1_resume_skips_done_cells(tmp_path: Path):
    states = _states(3)
    e = {s.state_id: _eval(s.text) for s in states}
    fp = FakeProvider(responder=_accept_responder)
    client = LLMClient(providers={"fake": fp})
    exp1.run_exp1(client, _ccfg(), _setup1(tmp_path, states, e))
    n1 = len(fp.calls)
    s = exp1.run_exp1(client, _ccfg(), _setup1(tmp_path, states, e))
    assert len(fp.calls) == n1 and s["n_new_decisions"] == 0
    assert len(list(read_jsonl(tmp_path / "exp1" / "decisions.jsonl"))) == 27


def test_exp1_parse_fail_counts_as_not_accepted(tmp_path: Path):
    states = _states(2)
    e = {s.state_id: _eval(s.text) for s in states}
    fp = FakeProvider(responder=lambda r: '{"action": "nope", "revised_text": ""}')
    client = LLMClient(providers={"fake": fp})
    summary = exp1.run_exp1(client, _ccfg(), _setup1(tmp_path, states, e))
    assert all(c["false_accept_rate"] == 0.0 for c in summary["cells"].values())
    assert all(v["ALERT"] and v["parse_fail_rate"] == 1.0 for v in summary["parse_fail"].values())


# ---------------------------------------------------------------- Exp 2


def _setup2(tmp_path: Path, states, judge_fn, **kw) -> exp2.Exp2Setup:
    v1, v2 = stub_pair()
    base = dict(
        run_id="t",
        run_dir=tmp_path / "exp2",
        x1_states=states,
        scorer=InProcessScorer(v1, v2),
        tau1=0.5,
        tau2=0.5,
        channel=IdentityChannel(),
        arms=[Arm.A1, Arm.B],
        k=4,
        assignment_seed=3,
        run_seed=1,
        judge_fn=judge_fn,
    )
    base.update(kw)
    return exp2.Exp2Setup(**base)


def test_exp2_refuses_channel_and_wrong_k(tmp_path: Path):
    states = _states(4)
    client = LLMClient(providers={"fake": FakeProvider(responder=_accept_responder)})
    with pytest.raises(exp2.Exp2Error):
        exp2.run_exp2(client, _ccfg(), _setup2(tmp_path, states, _eval, channel=_Noisy()))
    with pytest.raises(exp2.Exp2Error):
        exp2.run_exp2(client, _ccfg(), _setup2(tmp_path, states, _eval, k=3))


def test_exp2_balanced_assignment_is_seeded():
    ids = [f"x{i}" for i in range(11)]
    a = exp2.balanced_assignment(ids, [Arm.A1, Arm.B], seed=1)
    b = exp2.balanced_assignment(ids, [Arm.A1, Arm.B], seed=1)
    c = exp2.balanced_assignment(ids, [Arm.A1, Arm.B], seed=2)
    assert a == b and a != c
    counts = [list(a.values()).count(x) for x in (Arm.A1, Arm.B)]
    assert abs(counts[0] - counts[1]) <= 1


@pytest.mark.parametrize(
    ("namespace", "arms"),
    [(exp2.SEED_NAMESPACE, [Arm.A1, Arm.B]), ("exp2_smoke", [Arm.A1])],
)
def test_exp2_seed_and_cache_key_carry_no_arm_and_no_lambda(
    tmp_path: Path, monkeypatch, namespace: str, arms: list[Arm]
):
    """paper Sec. 4.6: in Exp 2 and in its step 10a smoke the controller seed is
    derived from (experiment, x0, t, attempt) alone -- the state key is the x0 id, not the round
    id {x0_id}-{arm}-t{t} -- and the cache key from provider, api, model, version, decoding,
    seed, prompt hash and attempt. Neither holds an arm or a lambda: relabelling a decision's
    arm or lambda leaves its seed unchanged, and with the prompt held fixed the same x0 under
    the other arm gets the same seed and the same cache key."""
    scheme = load_config("models")["controller"]["seed_scheme"]
    assert scheme == "per_state"  # the formal scheme; per_cell would key the seed by arm, lambda
    ccfg = _ccfg(scheme)
    monkeypatch.setattr(exp2, "format_observation", lambda arm, obs: "O_t")  # prompt fixed
    seen: list[tuple[ctrl.DecisionLabels, object]] = []
    real = ctrl.decide

    def spy(client, cfg, *, labels, **kw):
        fp = client.providers["fake"]
        n0 = len(fp.calls)
        out = real(client, cfg, labels=labels, **kw)
        seen.extend((labels, req) for req in fp.calls[n0:])
        return out

    monkeypatch.setattr(ctrl, "decide", spy)
    states = _states(6)
    x0_ids = {s.x0_id for s in states}

    def run(assignment_seed: int, sub: str) -> dict[tuple[str, int, int], tuple]:
        seen.clear()
        # never parseable: every decision makes all three attempts, every trajectory 4 rounds
        client = LLMClient(providers={"fake": FakeProvider(responder=lambda r: "not json")})
        setup = _setup2(
            tmp_path / sub, states, _eval, arms=arms, assignment_seed=assignment_seed,
            seed_namespace=namespace,
        )  # fmt: skip
        exp2.run_loop(client, ccfg, setup)
        got: dict[tuple[str, int, int], tuple] = {}
        for labels, req in seen:
            assert labels.state_id in x0_ids and labels.lam is Lambda.NONE
            want = derive_seed(ccfg.base_seed, "controller", namespace, labels.state_id, labels.t)
            assert req.seed == ctrl.attempt_call_seed(want, req.attempt)
            for arm in Arm:
                for lam in Lambda:
                    other = replace(labels, arm=arm, lam=lam)
                    assert ctrl.call_seed(ccfg, other) == ctrl.call_seed(ccfg, labels)
            parts = {
                "provider": req.provider, "api": req.api, "model": req.model,
                "version": req.version, "decoding": req.decoding, "seed": req.seed,
                "prompt_hash": req.prompt_hash(), "attempt": req.attempt,
            }  # fmt: skip
            assert req.cache_key() == sha256_hex(canonical_json(parts))
            got[(labels.state_id, labels.t, req.attempt)] = (labels.arm, req.seed, req.cache_key())
        assert len(got) == len(seen) == 6 * 4 * 3
        return got

    first = run(3, "a")
    if len(arms) > 1:
        order = [s.x0_id for s in states]  # the loop assigns in x1 order
        other_seed = next(
            s
            for s in range(4, 50)
            if exp2.balanced_assignment(order, arms, s) != exp2.balanced_assignment(order, arms, 3)
        )
        second = run(other_seed, "b")
        flipped = {k[0] for k in first if first[k][0] is not second[k][0]}
        assert flipped  # some x0 sits in the other arm this time
        assert {k: v[1:] for k, v in first.items()} == {k: v[1:] for k, v in second.items()}


def test_exp2_loop_rules_and_checkpoints(tmp_path: Path):
    states = _states(6)

    # scripted controller: parse_fail at t=1 (all 3 attempts), rewrite at t=2, accept at t=3
    seen: dict[str, int] = {}

    def responder2(r):
        content = r.messages[0]["content"]
        x0, x_t = content.split("|")[0], content.split("|")[1]
        if r.attempt == 0:
            seen[x0] = seen.get(x0, 0) + 1
        n = seen[x0] - 1  # round index for this x0, stable across parse retries
        if n == 0:
            return '{"action": "bad", "revised_text": ""}'
        if n == 1:
            return json.dumps({"action": "rewrite", "revised_text": "rw " + x_t})
        return json.dumps({"action": "accept", "revised_text": ""})

    fp = FakeProvider(responder=responder2)
    client = LLMClient(providers={"fake": fp})
    summary = exp2.run_exp2(
        client, _ccfg(), _setup2(tmp_path, states, lambda text: _eval(text, SafetyLabel.SAFE))
    )
    rd = tmp_path / "exp2"
    assert exp2.complete_rounds(rd, 4) == [1, 2, 3, 4]
    assert all((rd / "rounds" / f"round_{t}" / "COMPLETE.json").exists() for t in range(1, 5))
    finals = list(read_jsonl(rd / "trajectories.jsonl"))
    assert len(finals) == 6 and all(f["accepted"] and f["accepted_at"] == 3 for f in finals)
    assert all(f["n_parse_fail"] == 1 and f["n_rewrites"] == 1 for f in finals)
    assert all(f["x_T"].startswith("rw state text") for f in finals)
    rows = list(read_jsonl(rd / "decisions.jsonl"))
    assert len(rows) == 6 * 3  # t=1,2,3 for each; none active at t=4
    # parse_fail round: x_{t+1} = x_t
    t1 = {r["x0_id"]: r for r in rows if r["t"] == 1}
    t2 = {r["x0_id"]: r for r in rows if r["t"] == 2}
    assert all(t1[k]["parse_fail"] and t2[k]["x_t"] == t1[k]["x_t"] for k in t1)
    assert all(r["lam"] == "none" and r["s1"] == r["s1_tilde"] for r in rows)  # no channel
    for arm_stats in summary["arms"].values():
        assert arm_stats["TFR"] == 0.0 and arm_stats["non_convergence_rate"] == 0.0
    assert (rd / "assignment.jsonl").exists()
    # no quality function configured -> x0 is recorded, the score is null (semantic.py later)
    assert all(f["semantic_score"] is None and f["x0"] for f in finals)
    # group 6d: E judged every round text of every trajectory after the loop, once per
    # distinct text: x_1 (= x_2 after the parse_fail round) and the t=3 rewrite -> 2 per x0
    assert len(list(read_jsonl(rd / "evaluations.jsonl"))) == 12
    rj = list(read_jsonl(rd / exp2.ROUND_JUDGMENTS))
    assert len(rj) == 6 * 3 and {r["t"] for r in rj} == {1, 2, 3}
    assert all(r["source"] == "exp2" and r["e_label"] == "SAFE" for r in rj)
    assert sum(r["terminal"] for r in rj) == 6 and all(r["t"] == 3 for r in rj if r["terminal"])
    assert all(f["round_e_labels"] == {"1": "SAFE", "2": "SAFE", "3": "SAFE"} for f in finals)
    assert all(f["first_parse_fail_t"] == 1 for f in finals)


def test_exp2_finals_record_quality_of_the_terminal_text(tmp_path: Path):
    """(2026-09-18): the finals carry quality(x0, x_T) as ``semantic_score`` for the
    descriptive "low preservation among SAFE" report. The runner calls the configured
    quality function on (x0, x_T) exactly, and only after the loop."""
    states = _states(4)
    calls: list[tuple[str, str]] = []

    def quality(x0: str, x_t: str) -> float:
        calls.append((x0, x_t))
        return 0.25 if x0.endswith("0") else 0.75

    client = LLMClient(providers={"fake": FakeProvider(responder=_accept_responder)})
    exp2.run_exp2(
        client,
        _ccfg(),
        _setup2(tmp_path, states, lambda text: _eval(text, SafetyLabel.SAFE), quality_fn=quality),
    )
    finals = list(read_jsonl(tmp_path / "exp2" / "trajectories.jsonl"))
    assert len(finals) == 4 and sorted(calls) == sorted((f["x0"], f["x_T"]) for f in finals)
    assert {f["semantic_score"] for f in finals} <= {0.25, 0.75}
    assert all(f["semantic_score"] == quality(f["x0"], f["x_T"]) for f in finals)


def test_exp2_reuses_prior_judgments_for_t1_and_judges_every_round(tmp_path: Path):
    """Decision (group 6d): t = 1 texts are the D_Exp1 states, so step11's judgments are
    reused; every later round text (and the terminal text) is judged after the loop."""
    from rafr.contracts.schemas import text_hash

    states = _states(4)
    fp = FakeProvider(
        responder=lambda r: json.dumps(
            {"action": "rewrite", "revised_text": "again " + r.messages[0]["content"].split("|")[1]}
        )
    )
    client = LLMClient(providers={"fake": fp})
    calls: list[str] = []

    def judge(text: str) -> EvalRecord:
        calls.append(text)
        return _eval(text, SafetyLabel.UNSAFE)

    prior = {text_hash(s.text): _eval(s.text, SafetyLabel.SAFE) for s in states}
    exp2.run_exp2(client, _ccfg(), _setup2(tmp_path, states, judge, prior_judgments=prior))
    # never accepted: rounds 1..4 decided, x_5 is the terminal text -> 5 texts per x0,
    # of which t = 1 came from the prior judgments
    assert len(calls) == 4 * 4 and not any(c.startswith("state text") for c in calls)
    rj = list(read_jsonl(tmp_path / "exp2" / exp2.ROUND_JUDGMENTS))
    assert len(rj) == 4 * 5
    assert all(r["source"] == "reused" and r["e_label"] == "SAFE" for r in rj if r["t"] == 1)
    assert all(r["source"] == "exp2" and r["e_label"] == "UNSAFE" for r in rj if r["t"] > 1)
    assert all(r["terminal"] == (r["t"] == 5) for r in rj)
    finals = list(read_jsonl(tmp_path / "exp2" / "trajectories.jsonl"))
    assert all(
        f["round_e_labels"]["1"] == "SAFE" and f["round_e_labels"]["5"] == "UNSAFE" for f in finals
    )
    assert all(f["task_success"] is False and f["first_parse_fail_t"] is None for f in finals)
    # E is not in the loop: no judgment happened before the last round checkpoint was written
    assert len(list(read_jsonl(tmp_path / "exp2" / "evaluations.jsonl"))) == 4 * 5


def test_exp2_non_convergence_and_tfr(tmp_path: Path):
    states = _states(4)
    fp = FakeProvider(
        responder=lambda r: json.dumps(
            {"action": "rewrite", "revised_text": "again " + r.messages[0]["content"].split("|")[1]}
        )
    )
    client = LLMClient(providers={"fake": fp})
    summary = exp2.run_exp2(
        client, _ccfg(), _setup2(tmp_path, states, lambda text: _eval(text, SafetyLabel.UNSAFE))
    )
    rows = list(read_jsonl(tmp_path / "exp2" / "decisions.jsonl"))
    assert len(rows) == 4 * 4  # K rounds each
    for a in summary["arms"].values():
        assert (
            a["TFR"] == 1.0 and a["non_convergence_rate"] == 1.0 and a["rewrite_rounds_median"] == 4
        )


def test_exp2_resumes_from_checkpoint(tmp_path: Path):
    import shutil

    states = _states(4)
    calls = {"n": 0}

    def responder(r):
        calls["n"] += 1
        return json.dumps(
            {"action": "rewrite", "revised_text": "r " + r.messages[0]["content"].split("|")[1]}
        )

    client = LLMClient(providers={"fake": FakeProvider(responder=responder)})
    exp2.run_exp2(client, _ccfg(), _setup2(tmp_path, states, lambda t: _eval(t, SafetyLabel.SAFE)))
    n_full = calls["n"]
    # drop the last two rounds and resume: only rounds 3-4 are recomputed, nothing is doubled
    rd = tmp_path / "exp2"
    for t in (3, 4):
        shutil.rmtree(rd / "rounds" / f"round_{t}")
    calls["n"] = 0
    exp2.run_exp2(client, _ccfg(), _setup2(tmp_path, states, lambda t: _eval(t, SafetyLabel.SAFE)))
    assert calls["n"] == n_full // 2
    assert len(list(read_jsonl(rd / "decisions.jsonl"))) == 4 * 4


def _strip_times(obj):
    """Drop wall-clock fields: two runs differ in them and in nothing else."""
    if isinstance(obj, dict):
        return {k: _strip_times(v) for k, v in obj.items() if k not in ("created_at", "timestamp")}
    if isinstance(obj, list):
        return [_strip_times(v) for v in obj]
    return obj


class _CrashError(RuntimeError):
    pass


def test_exp2_interrupted_in_round_2_resumes_from_round_2_to_the_same_result(tmp_path: Path):
    """Task 2026-09-24 sections 5 / 7: cut the run off in the middle of round 2, resume, and get
    exactly what one uninterrupted run gives -- round 1 is kept, round 2 is redone whole, no
    decision is logged twice."""
    states = _states(6)

    def responder(r):
        x_t = r.messages[0]["content"].split("|")[1]
        if x_t.startswith("rw rw"):
            return json.dumps({"action": "accept", "revised_text": ""})
        return json.dumps({"action": "rewrite", "revised_text": "rw " + x_t})

    def judge(text):
        return _eval(text, SafetyLabel.SAFE if text.startswith("rw") else SafetyLabel.UNSAFE)

    one = tmp_path / "one"
    client = LLMClient(providers={"fake": FakeProvider(responder=responder)})
    exp2.run_exp2(client, _ccfg(), _setup2(one, states, judge))

    two = tmp_path / "two"
    budget = {"n": 6 + 3}  # round 1 = 6 decisions, then 3 of round 2's 6

    def crashing(r):
        budget["n"] -= 1
        if budget["n"] < 0:
            raise _CrashError("pod lost")
        return responder(r)

    client = LLMClient(providers={"fake": FakeProvider(responder=crashing)})
    with pytest.raises(_CrashError):
        exp2.run_loop(client, _ccfg(), _setup2(two, states, judge))
    rd = two / "exp2"
    assert exp2.complete_rounds(rd, 4) == [1]
    assert not (rd / "rounds" / "round_2").exists()
    round1 = {f.name: f.read_bytes() for f in (rd / "rounds" / "round_1").iterdir()}

    fp = FakeProvider(responder=responder)
    client = LLMClient(providers={"fake": fp})
    exp2.run_exp2(client, _ccfg(), _setup2(two, states, judge))
    assert {f.name: f.read_bytes() for f in (rd / "rounds" / "round_1").iterdir()} == round1
    assert fp.calls and all(
        c.messages[0]["content"].split("|")[1].startswith("rw") for c in fp.calls
    )
    for name in ("decisions.jsonl", "scores.jsonl", "trajectories.jsonl", "round_judgments.jsonl"):
        a = [_strip_times(r) for r in read_jsonl(one / "exp2" / name)]
        b = [_strip_times(r) for r in read_jsonl(rd / name)]
        assert a == b, name
    assert (one / "exp2" / "assignment.jsonl").read_text() == (rd / "assignment.jsonl").read_text()


def test_exp2_judgments_wait_for_the_whole_loop(tmp_path: Path):
    """E never runs before round K is on disk (paper Sec. 4.4): finish refuses an unfinished
    loop."""
    states = _states(2)
    setup = _setup2(tmp_path, states, lambda t: _eval(t))
    with pytest.raises(exp2.LoopNotFinishedError):
        exp2.finish(setup)
    trajs = [exp2.Trajectory(x0_id=s.x0_id, x0=s.x0, arm="A1", t=1, x_t=s.text) for s in states]
    exp2.write_round(setup.run_dir, 1, trajs, [], [])
    with pytest.raises(exp2.LoopNotFinishedError):
        exp2.finish(setup)
    # a tampered round is not COMPLETE
    f = setup.run_dir / "rounds" / "round_1" / "trajectories.jsonl"
    f.write_text(f.read_text() + "\n", encoding="utf-8")
    assert exp2.complete_rounds(setup.run_dir, 4) == []


def test_exp2_terminal_output_is_the_last_rewrite_and_is_never_scored(tmp_path: Path):
    """paper Sec. 4.4: with no accept in 4 decisions, x_T is the OUTPUT of the 4th decision (x_5),
    not the text the controller last looked at, and x_5 never goes to the verifiers."""
    states = _states(3)

    def responder(r):
        x_t = r.messages[0]["content"].split("|")[1]
        return json.dumps({"action": "rewrite", "revised_text": "rw " + x_t})

    client = LLMClient(providers={"fake": FakeProvider(responder=responder)})
    exp2.run_exp2(
        client, _ccfg(), _setup2(tmp_path, states, lambda t: _eval(t, SafetyLabel.UNSAFE))
    )
    rd = tmp_path / "exp2"
    finals = {f["x0_id"]: f for f in read_jsonl(rd / "trajectories.jsonl")}
    rows = list(read_jsonl(rd / "decisions.jsonl"))
    for s in states:
        f = finals[s.x0_id]
        seen = [r["x_t"] for r in rows if r["x0_id"] == s.x0_id]
        assert len(seen) == 4 and f["accepted"] is False
        assert f["x_T"] == "rw " + seen[-1] and f["x_T"] not in seen  # x_5, one rewrite past x_4
    # the scores table stops at the 4th decision's input: no state_id for t = 5
    assert {r["state_id"].rsplit("-t", 1)[1] for r in read_jsonl(rd / "scores.jsonl")} == {
        "1", "2", "3", "4"
    }  # fmt: skip
    # and the terminal text is judged by E as t = K + 1
    rj = [r for r in read_jsonl(rd / exp2.ROUND_JUDGMENTS) if r["terminal"]]
    assert len(rj) == 3 and all(r["t"] == 5 for r in rj)


def test_exp2_terminal_output_when_the_last_decision_is_a_parse_fail(tmp_path: Path):
    """paper Sec. 4.4: a parse_fail at t = 4 produces no new text, so x_5 = x_4."""
    states = _states(2)

    def responder(r):
        x_t = r.messages[0]["content"].split("|")[1]
        if x_t.startswith("rw rw rw "):  # the 4th decision
            return '{"action": "bad", "revised_text": ""}'
        return json.dumps({"action": "rewrite", "revised_text": "rw " + x_t})

    client = LLMClient(providers={"fake": FakeProvider(responder=responder)})
    exp2.run_exp2(
        client, _ccfg(), _setup2(tmp_path, states, lambda t: _eval(t, SafetyLabel.UNSAFE))
    )
    rd = tmp_path / "exp2"
    rows = list(read_jsonl(rd / "decisions.jsonl"))
    for f in read_jsonl(rd / "trajectories.jsonl"):
        x4 = next(r["x_t"] for r in rows if r["x0_id"] == f["x0_id"] and r["t"] == 4)
        assert f["n_parse_fail"] == 1 and f["first_parse_fail_t"] == 4
        assert f["x_T"] == x4 and f["round_e_labels"]["5"] == f["round_e_labels"]["4"]


def test_exp2_refusal_like_flag_travels_with_the_text(tmp_path: Path):
    """paper Sec. 4.6 sensitivity: the flag recorded on the trajectory is x_T's own, so a refusal-
    looking rewrite that is later replaced does not make the trajectory a failure."""
    states = _states(2)
    ccfg = _ccfg()
    ccfg.refusal_phrases = ["i cannot help"]

    def responder(r):
        x_t = r.messages[0]["content"].split("|")[1]
        # t=1 -> refusal-looking rewrite; t=2 -> a normal one; t=3 -> accept
        if x_t.startswith("I cannot help"):
            return json.dumps({"action": "rewrite", "revised_text": "a polite rewrite"})
        if x_t == "a polite rewrite":
            return json.dumps({"action": "accept", "revised_text": ""})
        return json.dumps({"action": "rewrite", "revised_text": "I cannot help with that"})

    client = LLMClient(providers={"fake": FakeProvider(responder=responder)})
    summary = exp2.run_exp2(
        client, ccfg, _setup2(tmp_path, states, lambda t: _eval(t, SafetyLabel.SAFE))
    )
    finals = list(read_jsonl(tmp_path / "exp2" / "trajectories.jsonl"))
    assert all(f["refusal_like_rewrites"] == 1 for f in finals)  # it happened ...
    assert all(f["x_T_refusal_like"] is False for f in finals)  # ... but not in x_T
    assert all(a["TFR_refusal_like_as_failure"] == 0.0 for a in summary["arms"].values())


def test_exp1_channel_log_records_the_perturbation_assignment(tmp_path: Path):
    """the channel log carries the assignment itself — per-verifier flags, derived z
    mode, replacement scores — not only the resulting (s, s~)."""
    from rafr.channel.degrade import DegradeChannel, LevelParams
    from rafr.channel.margin_fit import MarginModel

    states = _states(5)
    e = {s.state_id: _eval(s.text) for s in states}
    q = {v: MarginModel(v, 0.5, [0.05, 0.2, 0.4, 0.6, 0.8, 0.95]) for v in ("v1", "v2")}
    ch = DegradeChannel(
        {
            Lambda.LOW: LevelParams(0.0, 0.0),
            Lambda.MID: LevelParams(1.0, 0.0),
            Lambda.HIGH: LevelParams(1.0, 1.0),
        },
        q,
        tau1=0.5,
        tau2=0.5,
        seed=5,
    )
    client = LLMClient(providers={"fake": FakeProvider(responder=_accept_responder)})
    exp1.run_exp1(client, _ccfg(), _setup1(tmp_path, states, e, channel=ch))
    rows = list(read_jsonl(tmp_path / "exp1" / "channel.jsonl"))
    by_lam = {}
    for r in rows:
        by_lam.setdefault(r["lam"], []).append(r)
    assert {r["z"] for r in by_lam["low"]} == {"none"}
    assert {r["z"] for r in by_lam["mid"]} == {"v1_only"}
    assert {r["z"] for r in by_lam["high"]} == {"both"}
    for r in by_lam["high"]:
        assert r["perturbed_v1"] and r["perturbed_v2"]
        assert (r["replacement_s1"], r["replacement_s2"]) == (r["s1_tilde"], r["s2_tilde"])
    for r in by_lam["low"]:
        assert r["replacement_s1"] is None and (r["s1"], r["s2"]) == (r["s1_tilde"], r["s2_tilde"])


def test_each_experiment_tells_the_controller_its_own_decision_budget(tmp_path: Path):
    """(2026-09-18): the system prompt's ``{k}`` is 1 in Exp 1 (single-shot, paper Sec. 4.6)
    and K_MAX_ROUNDS in Exp 2 (paper Sec. 4.4); every controller call carries the rendered value."""
    from rafr.contracts.schemas import K_MAX_ROUNDS

    ccfg = _ccfg()
    ccfg.system_prompt = "budget={k}"
    states = _states(2)
    e = {s.state_id: _eval(s.text) for s in states}
    fp1 = FakeProvider(responder=_accept_responder)
    exp1.run_exp1(LLMClient(providers={"fake": fp1}), ccfg, _setup1(tmp_path / "a", states, e))
    assert fp1.calls and all(c.system == "budget=1" for c in fp1.calls)
    assert exp1.EXP1_DECISIONS == 1
    fp2 = FakeProvider(responder=_accept_responder)
    exp2.run_exp2(
        LLMClient(providers={"fake": fp2}),
        ccfg,
        _setup2(tmp_path / "b", states, lambda t: _eval(t, SafetyLabel.SAFE)),
    )
    assert fp2.calls and all(c.system == f"budget={K_MAX_ROUNDS}" for c in fp2.calls)
    assert ccfg.system_prompt == "budget={k}"  # the runners do not mutate the shared config


def _row(sid, arm, lam, s1, s2, prompt, donor=None):
    return {
        "state_id": sid, "arm": arm, "lam": lam, "s1_tilde": s1, "s2_tilde": s2,
        "permuted_from_state_id": donor, "attempts": [{"prompt_hash": prompt}],
    }  # fmt: skip


def test_exp1_reports_the_share_of_cells_whose_prompt_is_lambda_lows_byte_for_byte():
    """the main number is how often each level sent exactly
    lambda_low's prompt (same prompt + the per-state seed = the same controller answer); next to
    it, as an auxiliary, how often the s_obs was lambda_low's bit for bit (permuted-B counts the
    donor's s~ at that level too). The prompt can be identical while the s_obs differ below the
    displayed precision."""
    rows = []
    for lam, a, b in (("low", 0.9, 0.2), ("mid", 0.9, 0.2), ("high", 0.4, 0.2)):
        rows.append(_row("s", "A1", lam, a, b, f"pA-{lam if lam == 'high' else 'x'}"))
        rows.append(_row("s", "B", lam, a, b, f"pB-{lam}"))
    for lam in ("low", "mid", "high"):  # donor state d: same s~ at low / mid, changes at high
        rows.append(
            _row(
                "d",
                "A1",
                lam,
                0.1,
                0.1 if lam != "high" else 0.7,
                "d-high" if lam == "high" else "d",
            )
        )
    rows.append(_row("s", "permuted_B", "low", 0.9, 0.2, "q1", donor="d"))
    rows.append(_row("s", "permuted_B", "mid", 0.9, 0.2, "q2", donor="d"))
    rows.append(_row("s", "permuted_B", "high", 0.4, 0.2, "q3", donor="d"))
    # a difference in the last bit counts; -0.0 is not 0.0
    rows.append(_row("e", "A1", "low", 0.0, 0.5, "e"))
    rows.append(_row("e", "A1", "mid", -0.0, 0.5, "e"))
    out = exp1.identity_to_low_by_lambda(rows)
    assert (out["main"], out["auxiliary"]) == ("prompt_share", "s_obs_share")
    low, mid, high = (out["by_lambda"][k] for k in ("low", "mid", "high"))
    assert low["prompt_share"] == low["s_obs_share"] == 1.0 and low["n"] == 5
    assert mid["by_arm"]["A1"] == {"n": 3, "prompt_identical": 3, "prompt_share": 1.0,
                                   "s_obs_identical": 2, "s_obs_share": 2 / 3}  # fmt: skip
    assert mid["by_arm"]["B"]["s_obs_identical"] == 1
    assert mid["by_arm"]["B"]["prompt_identical"] == 0
    assert mid["by_arm"]["permuted_B"]["s_obs_identical"] == 1  # own and donor s~ unchanged
    assert high["by_arm"]["permuted_B"]["s_obs_identical"] == 0 and high["s_obs_identical"] == 0
    assert out["pairwise"]["mid|high"]["n"] == 4
    assert out["pairwise"]["low|mid"]["s_obs_identical"] == 4
    # paper Sec. 4.6: per arm AND per pair of levels; the per-arm counts add up to the total
    mh = out["pairwise"]["mid|high"]
    assert set(mh["by_arm"]) == {"A1", "B", "permuted_B"}
    assert sum(c["n"] for c in mh["by_arm"].values()) == mh["n"] == 4
    assert mh["by_arm"]["A1"] == {"n": 2, "prompt_identical": 0, "prompt_share": 0.0,
                                  "s_obs_identical": 0, "s_obs_share": 0.0}  # fmt: skip
    lm = out["pairwise"]["low|mid"]
    assert lm["by_arm"]["A1"]["s_obs_identical"] == 2 and lm["by_arm"]["A1"]["n"] == 3
    assert lm["by_arm"]["permuted_B"] == {"n": 1, "prompt_identical": 0, "prompt_share": 0.0,
                                          "s_obs_identical": 1, "s_obs_share": 1.0}  # fmt: skip
    summary = exp1.summarize_exp1(
        [{**r, "e_label": "UNSAFE", "parse_fail": False, "action": "accept"} for r in rows]
    )
    assert summary["identical_to_low"] == out
    src = (Path(__file__).resolve().parents[1] / "pipelines" / "step11_exp1.py").read_text("utf-8")
    assert '"identical_to_low": exp1.identity_to_low_by_lambda(rows)' in src
