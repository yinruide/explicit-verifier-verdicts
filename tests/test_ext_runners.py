"""Extension runners (post-pilot, 2026-10-31) with stub verifiers and a fake controller.

Exp 1: the E-incorrect cohort runs the four new arms only and refuses unless the recomputed
channel equals the original run's bit for bit; the E-correct cohort runs all seven; the pi-hat
mismatch table is its own derangement, one donor at every lambda, showing the donor's pi-hat of
that lambda and nothing else of the donor; seeds keep the original per-state derivation.
Exp 2: full pairing, the reused t = 1 scores (no scoring in round 1), the in-process t = 1 of the
E-correct cohort scored once per start and compared with a reference, No Verifier without any
verifier, and E's post-loop pass."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from rafr.calibration import fit as F
from rafr.calibration.apply import Calibrator, write_frozen
from rafr.contracts.schemas import Arm, EvalRecord, Lambda, SafetyLabel, State, text_hash
from rafr.controller import decide as ctrl
from rafr.core.seeding import derive_seed
from rafr.experiments import exp1, exp1_ext, exp2, exp2_ext
from rafr.experiments.common import InProcessScorer
from rafr.formatter.extended import detox_wording
from rafr.formatter.permute import load_permutation
from rafr.llm.client import LLMClient
from rafr.llm.providers.fake import FakeProvider
from rafr.logging.writer import read_jsonl
from rafr.verifiers.stub import stub_pair

NV_SYSTEM = "no verifier; at most {k} decisions in total"
NV_REWRITE = "{x0}|{x_t}"


def _states(n: int, prefix: str = "x0") -> list[State]:
    return [
        State(
            state_id=f"{prefix}-{i}-g1",
            x0_id=f"{prefix}-{i}",
            x0=f"orig {i} idiot",
            text=f"state text {i}",
        )  # fmt: skip
        for i in range(n)
    ]


def _eval(text: str, label: SafetyLabel = SafetyLabel.UNSAFE) -> EvalRecord:
    return EvalRecord(
        text_id=text_hash(text), text=text, label=label, model="e", model_version=None,
        prompt_hash="p", e_prompt_sha256="h", e_version="v", decoding={},
    )  # fmt: skip


def _models() -> dict:
    return {
        "controller": {
            "provider": "fake", "model": "ctrl", "version": "v", "decoding": {},
            "seed_scheme": "per_state", "network": {},
        }
    }  # fmt: skip


def _ccfg() -> ctrl.ControllerConfig:
    return ctrl.build_config(
        _models(), {}, base_seed=1, system_prompt="S {k} decisions",
        rewrite_template="{x0}|{x_t}|{observation}", retry_prompt="R",
    )  # fmt: skip


def _ccfg_nv() -> ctrl.ControllerConfig:
    return ctrl.build_config(
        _models(), {}, base_seed=1, system_prompt=NV_SYSTEM, rewrite_template=NV_REWRITE,
        retry_prompt="R",
    )  # fmt: skip


class _Noisy:
    name = "noisy"

    def __call__(self, state_id, s1, s2, lam):
        d = {Lambda.LOW: 0.0, Lambda.MID: 0.05, Lambda.HIGH: 0.2}[lam]
        return (min(1.0, s1 + d), max(0.0, s2 - d))


def _accept(req):
    return json.dumps({"action": "accept", "revised_text": ""})


def _rewrite(req):
    user = req.messages[0]["content"]
    return json.dumps({"action": "rewrite", "revised_text": f"rw{len(user) % 97}-{req.seed}"})


@pytest.fixture
def calibrator(tmp_path: Path) -> Calibrator:
    r = np.random.default_rng(0)
    s = r.uniform(0.0, 1.0, 400)
    y = (r.random(400) < s).astype(int)
    ids = [f"p{i}" for i in range(400)]
    rec = F.fit_record(
        domain="detox", ids=ids, failures=dict(zip(ids, y.tolist(), strict=True)),
        scores={"v1": dict(zip(ids, s, strict=True)), "v2": dict(zip(ids, s[::-1], strict=True))},
        verifier_ids={"v1": "stub-v1", "v2": "stub-v2"}, eps=1e-7, provenance={},
    )  # fmt: skip
    path = tmp_path / "frozen" / "calibration_detox.json"
    return Calibrator.load(path, expected_sha256=write_frozen(path, rec))


def _setup1(tmp_path: Path, states, cohort: str, arms, label, **kw) -> exp1_ext.Exp1ExtSetup:
    v1, v2 = stub_pair()
    base = dict(
        run_id="t", run_dir=tmp_path / f"ext_{cohort}", cohort=cohort, states=states,
        scorer=InProcessScorer(v1, v2), tau1=0.5, tau2=0.5, channel=_Noisy(),
        lambdas=[Lambda.LOW, Lambda.MID, Lambda.HIGH], arms=arms, wording=detox_wording(),
        e_labels={s.state_id: _eval(s.text, label) for s in states}, expected_label=label,
        perm_seed=7, pihat_perm_seed=11, run_seed=1,
    )  # fmt: skip
    base.update(kw)
    return exp1_ext.Exp1ExtSetup(**base)


def _original_exp1(tmp_path: Path, states) -> Path:
    """An original Exp 1 run (A1 / B / permuted-B) through the preregistered runner."""
    v1, v2 = stub_pair()
    setup = exp1.Exp1Setup(
        run_id="orig", run_dir=tmp_path / "orig", states=states, scorer=InProcessScorer(v1, v2),
        tau1=0.5, tau2=0.5, channel=_Noisy(), lambdas=[Lambda.LOW, Lambda.MID, Lambda.HIGH],
        arms=[Arm.A1, Arm.B, Arm.PERMUTED_B], perm_seed=7, run_seed=1,
        e_labels={s.state_id: _eval(s.text) for s in states},
    )  # fmt: skip
    exp1.run_exp1(LLMClient(providers={"fake": FakeProvider(responder=_accept)}), _ccfg(), setup)
    return tmp_path / "orig"


def test_exp1_e_incorrect_runs_the_new_arms_on_the_original_channel(tmp_path, calibrator):
    states = _states(6)
    orig = _original_exp1(tmp_path, states)
    channel = list(read_jsonl(orig / "channel.jsonl"))
    fp = FakeProvider(responder=_accept)
    setup = _setup1(
        tmp_path, states, exp1_ext.E_INCORRECT, list(exp1_ext.NEW_ARMS), SafetyLabel.UNSAFE,
        calibrator=calibrator, original_channel=channel,
    )  # fmt: skip
    summary = exp1_ext.run_exp1_ext(LLMClient(providers={"fake": fp}), _ccfg(), _ccfg_nv(), setup)
    rows = list(read_jsonl(setup.run_dir / "decisions.jsonl"))
    assert len(rows) == 6 * 4 * 3 and summary["n_new_decisions"] == 72
    assert {r["arm"] for r in rows} == {a.value for a in exp1_ext.NEW_ARMS}
    meta = json.loads((setup.run_dir / exp1_ext.RUN_META).read_text(encoding="utf-8"))
    assert meta["channel_check_against_original"] == {"n_rows": 18, "bitwise_identical": True}
    assert meta["reused_arms"] == ["A1", "B", "permuted_B"] and meta["seed_namespace"] == "exp1"
    # the per-state seed is the original run's: arm and lambda are not in it
    orig_seed = {
        r["state_id"]: r["attempts"][0]["seed"] for r in read_jsonl(orig / "decisions.jsonl")
    }
    assert all(r["attempts"][0]["seed"] == orig_seed[r["state_id"]] for r in rows)
    # every row carries its cohort's label; FA = 1 with an always-accept controller
    assert all(c["false_accept_rate"] == 1.0 for c in summary["cells"].values())


def test_exp1_e_incorrect_refuses_a_channel_that_differs_by_one_bit(tmp_path, calibrator):
    states = _states(4)
    channel = list(read_jsonl(_original_exp1(tmp_path, states) / "channel.jsonl"))
    channel[2] = {**channel[2], "s1_tilde": np.nextafter(channel[2]["s1_tilde"], 2.0)}
    fp = FakeProvider(responder=_accept)
    setup = _setup1(
        tmp_path, states, exp1_ext.E_INCORRECT, list(exp1_ext.NEW_ARMS), SafetyLabel.UNSAFE,
        calibrator=calibrator, original_channel=channel,
    )  # fmt: skip
    with pytest.raises(exp1_ext.Exp1ExtError, match="differ from the original"):
        exp1_ext.run_exp1_ext(LLMClient(providers={"fake": fp}), _ccfg(), _ccfg_nv(), setup)
    assert fp.calls == []  # nothing was sent


def test_exp1_cohort_rules_are_enforced_before_any_call(tmp_path, calibrator):
    states = _states(3)
    fp = FakeProvider(responder=_accept)
    client = LLMClient(providers={"fake": fp})
    bad = [
        # E-incorrect never reruns the original arms
        _setup1(tmp_path, states, exp1_ext.E_INCORRECT, [Arm.A1], SafetyLabel.UNSAFE,
                original_channel=[]),
        # every E-correct state must be SAFE
        _setup1(tmp_path, states, exp1_ext.E_CORRECT, [Arm.A1], SafetyLabel.SAFE,
                e_labels={s.state_id: _eval(s.text) for s in states}),
        # the pi-hat arms need the calibrator; E-incorrect needs the original channel
        _setup1(tmp_path, states, exp1_ext.E_CORRECT, [Arm.B_PIHAT], SafetyLabel.SAFE),
        _setup1(tmp_path, states, exp1_ext.E_INCORRECT, [Arm.VERDICT_ONLY], SafetyLabel.UNSAFE),
        # a smoke's states are outside D_Exp1: it carries no original channel rows
        _setup1(tmp_path, states, exp1_ext.E_INCORRECT, [Arm.VERDICT_ONLY], SafetyLabel.UNSAFE,
                original_channel=[], smoke=True),
    ]  # fmt: skip
    for setup in bad:
        with pytest.raises(exp1_ext.Exp1ExtError):
            exp1_ext.run_exp1_ext(client, _ccfg(), _ccfg_nv(), setup)
    assert fp.calls == []


def test_a_smoke_client_keeps_its_own_cache(tmp_path):
    from rafr.core.runtime import build_client, build_panel_client

    c = build_client({}, run_dir=tmp_path, fake=True, cache_dir=tmp_path / "c")
    assert c._cache.root == tmp_path / "c"
    models = {"panel": {"members": {"grok": {"provider": "xai"}}}}
    p = build_panel_client(models, run_dir=tmp_path, fake=True, cache_dir=tmp_path / "p")
    assert p._cache.root == tmp_path / "p"


def test_ext_exp2_smoke_flag_belongs_to_loop():
    from pipelines import ext_exp2

    for command in ("judge", "panel"):
        with pytest.raises(SystemExit, match="--smoke belongs to starts / loop"):
            ext_exp2.main([command, "--smoke", "3", "--run-id", "exp2ext_x"])


def test_exp1_e_incorrect_smoke_runs_without_the_original_channel(tmp_path, calibrator):
    """Stop point D (2026-10-31): the smoke's E-UNSAFE pool states are not D_Exp1's, so
    there is nothing to match; the run says so and is marked smoke."""
    states = _states(4, prefix="pool")
    fp = FakeProvider(responder=_accept)
    setup = _setup1(
        tmp_path, states, exp1_ext.E_INCORRECT, list(exp1_ext.NEW_ARMS), SafetyLabel.UNSAFE,
        calibrator=calibrator, smoke=True, seed_namespace="exp1_smoke",
    )  # fmt: skip
    exp1_ext.run_exp1_ext(LLMClient(providers={"fake": fp}), _ccfg(), _ccfg_nv(), setup)
    meta = json.loads((setup.run_dir / exp1_ext.RUN_META).read_text(encoding="utf-8"))
    assert meta["smoke"] is True and meta["seed_namespace"] == "exp1_smoke"
    assert meta["channel_check_against_original"].startswith("not applicable")
    rows = list(read_jsonl(setup.run_dir / "decisions.jsonl"))
    assert len(rows) == 4 * 4 * 3
    for r in rows:
        want = derive_seed(1, "controller", "exp1_smoke", r["state_id"], 1)
        assert r["attempts"][0]["seed"] == want


def test_exp1_e_correct_seven_arms_and_the_independent_pihat_mismatch(
    tmp_path, calibrator, monkeypatch
):
    states = _states(6, prefix="g")
    seen: list[tuple] = []
    real = exp1_ext.format_extended

    def spy(arm, obs, wording, *, pihat=None, donor_pihat=None):
        seen.append((arm, obs, pihat, donor_pihat))
        return real(arm, obs, wording, pihat=pihat, donor_pihat=donor_pihat)

    monkeypatch.setattr(exp1_ext, "format_extended", spy)
    fp = FakeProvider(responder=_accept)
    arms = list(exp1_ext.COHORTS[exp1_ext.E_CORRECT].arms)
    setup = _setup1(
        tmp_path, states, exp1_ext.E_CORRECT, arms, SafetyLabel.SAFE, calibrator=calibrator
    )
    summary = exp1_ext.run_exp1_ext(LLMClient(providers={"fake": fp}), _ccfg(), _ccfg_nv(), setup)
    rd = setup.run_dir
    rows = list(read_jsonl(rd / "decisions.jsonl"))
    assert len(rows) == 6 * 7 * 3
    assert all(c["false_reject_rate"] == 0.0 for c in summary["cells"].values())
    # two tables, two seeds, both derangements over this cohort, written apart
    seed_b, perm_b = load_permutation(rd / "permutation.jsonl")
    seed_p, perm_p = load_permutation(rd / exp1_ext.PIHAT_PERMUTATION)
    assert (seed_b, seed_p) == (7, 11) and perm_b != perm_p
    assert set(perm_p) == set(perm_p.values()) == {s.state_id for s in states}
    # one pi-hat donor per state at every lambda; the log names it
    mism = [r for r in rows if r["arm"] == "B_pihat_mismatched"]
    assert all(r["permuted_from_state_id"] == perm_p[r["state_id"]] for r in mism)
    # pi-hat side stream = the calibrator on s_obs; the mismatch arm shows the donor's of that
    # lambda, the B_pihat arm its own
    ph = {(r["state_id"], r["lam"]): r for r in read_jsonl(rd / exp1_ext.PIHAT_STREAM)}
    assert len(ph) == 6 * 3
    for r in ph.values():
        d = calibrator.detail(r["s1_obs"], r["s2_obs"], 0.5, 0.5)
        assert (d.pihat1, d.pihat2) == (r["pihat1"], r["pihat2"])
    for arm, obs, pihat, donor in seen:
        if arm is Arm.B_PIHAT:
            own = ph[(obs.state_id, obs.lam.value)]
            assert (pihat.pihat1, pihat.pihat2) == (own["pihat1"], own["pihat2"])
        if arm is Arm.B_PIHAT_MISMATCHED:
            d = ph[(perm_p[obs.state_id], obs.lam.value)]
            assert (donor.pihat1, donor.pihat2) == (d["pihat1"], d["pihat2"])
    # No Verifier: its own prompts, no checker output, the same request at every lambda
    nv = [c for c in fp.calls if c.system.startswith("no verifier")]
    assert len(nv) == 6 * 3 and all("CHECKER" not in c.messages[0]["content"] for c in nv)
    by_state: dict[str, set[str]] = {}
    for r in rows:
        if r["arm"] == "no_verifier":
            by_state.setdefault(r["state_id"], set()).add(r["prompt_hash"])
    assert all(len(h) == 1 for h in by_state.values())
    # namespace exp1_ecorrect: never the original Exp 1's per-state seed
    for r in rows:
        want = derive_seed(1, "controller", "exp1_ecorrect", r["state_id"], 1)
        assert r["attempts"][0]["seed"] == want


def test_the_formal_no_verifier_request_names_no_checker_score_threshold_or_flag(tmp_path):
    """2026-10-31: the No Verifier request as sent -- the final system prompt and user
    turn, through the formal controller configuration -- has none of these words."""
    from dataclasses import replace

    from rafr.core.config import get, load_config
    from rafr.experiments import ext_config

    cfgs = load_config()
    refusal = get(cfgs["generation"], "g_fixed.validity.refusal_phrases_file")
    nv = {
        exp: ext_config.no_verifier_config(cfgs["models"], cfgs[exp], ext_config.load_ext(),
                                           base_seed=1, refusal_phrases_file=refusal)
        for exp in ("exp1", "exp2")
    }  # fmt: skip
    assert nv["exp2"].system_prompt == nv["exp1"].system_prompt  # one file, {k} filled per call
    fp = FakeProvider(responder=_accept)
    ccfg = replace(nv["exp1"], provider="fake")
    setup = _setup1(tmp_path, _states(3, prefix="nv"), exp1_ext.E_CORRECT, [Arm.NO_VERIFIER],
                    SafetyLabel.SAFE, lambdas=[Lambda.LOW])  # fmt: skip
    exp1_ext.run_exp1_ext(LLMClient(providers={"fake": fp}), ccfg, ccfg, setup)
    assert len(fp.calls) == 3
    for req in fp.calls:
        sent = "\n".join([req.system or "", *(m["content"] for m in req.messages)]).lower()
        for word in ("checker", "score", "threshold", "flag"):
            assert word not in sent, word
        assert "state text" in sent and "you are the controller" in sent


def test_exp1_resume_is_idempotent_and_a_changed_setup_is_refused(tmp_path, calibrator):
    states = _states(3, prefix="g")
    fp = FakeProvider(responder=_accept)
    client = LLMClient(providers={"fake": fp})
    arms = [Arm.NO_VERIFIER, Arm.B_PIHAT_MISMATCHED]
    setup = _setup1(tmp_path, states, exp1_ext.E_CORRECT, arms, SafetyLabel.SAFE,
                    calibrator=calibrator)  # fmt: skip
    exp1_ext.run_exp1_ext(client, _ccfg(), _ccfg_nv(), setup)
    n = len(fp.calls)
    again = exp1_ext.run_exp1_ext(client, _ccfg(), _ccfg_nv(), setup)
    assert again["n_new_decisions"] == 0 and len(fp.calls) == n
    other = _setup1(tmp_path, states, exp1_ext.E_CORRECT, arms, SafetyLabel.SAFE,
                    calibrator=calibrator, pihat_perm_seed=12)  # fmt: skip
    with pytest.raises((exp1_ext.Exp1ExtError, exp1.Exp1Error)):
        exp1_ext.run_exp1_ext(client, _ccfg(), _ccfg_nv(), other)


def test_exp1_smoke_namespace_draws_its_own_seeds(tmp_path, calibrator):
    states = _states(2, prefix="g")
    fp = FakeProvider(responder=_accept)
    setup = _setup1(tmp_path, states, exp1_ext.E_CORRECT, [Arm.VERDICT_ONLY], SafetyLabel.SAFE,
                    seed_namespace="exp1_ecorrect_smoke")  # fmt: skip
    exp1_ext.run_exp1_ext(LLMClient(providers={"fake": fp}), _ccfg(), None, setup)
    for r in read_jsonl(setup.run_dir / "decisions.jsonl"):
        assert r["attempts"][0]["seed"] == derive_seed(
            1, "controller", "exp1_ecorrect_smoke", r["state_id"], 1
        )


# ---- Exp 2 ---------------------------------------------------------------------------------


class _Counting(InProcessScorer):
    """The stub pair, counting what it is asked to score."""

    def __init__(self):
        super().__init__(*stub_pair())
        self.texts: list[str] = []

    def score_pair(self, texts):
        self.texts += list(texts)
        return super().score_pair(texts)


def test_full_pairing_plan_gives_each_start_the_arm_it_did_not_get():
    plan = exp2_ext.full_pairing_plan(
        ["a", "b"], new_arms=[Arm.NO_VERIFIER], fill_arms=[Arm.A1, Arm.B],
        original_assignment={"a": "A1", "b": "B"},
    )  # fmt: skip
    assert plan == {"a": [Arm.NO_VERIFIER, Arm.B], "b": [Arm.NO_VERIFIER, Arm.A1]}
    with pytest.raises(exp2_ext.Exp2ExtError):
        exp2_ext.full_pairing_plan(["a"], fill_arms=[Arm.A1])


def _setup2(tmp_path, starts, cohort, plan, scorer, **kw) -> exp2_ext.Exp2ExtSetup:
    base = dict(
        run_id="t2", run_dir=tmp_path / f"exp2ext_{cohort}", cohort=cohort, starts=starts,
        plan=plan, scorer=scorer, tau1=0.5, tau2=0.5, k=4, wording=detox_wording(),
        judge_fn=lambda text: _eval(text), run_seed=3,
    )  # fmt: skip
    base.update(kw)
    return exp2_ext.Exp2ExtSetup(**base)


def test_exp2_e_incorrect_reuses_t1_scores_and_pairs_every_start(tmp_path, calibrator):
    starts = _states(4)
    scorer = _Counting()
    reused = {s.x0_id: (0.9, 0.1) for s in starts}
    plan = exp2_ext.full_pairing_plan(
        [s.x0_id for s in starts], new_arms=list(exp2_ext.NEW_ARMS), fill_arms=[Arm.A1, Arm.B],
        original_assignment={s.x0_id: ("A1" if i % 2 else "B") for i, s in enumerate(starts)},
    )  # fmt: skip
    fp = FakeProvider(responder=_rewrite)
    setup = _setup2(tmp_path, starts, exp2_ext.E_INCORRECT, plan, scorer,
                    calibrator=calibrator, t1_scores=reused)  # fmt: skip
    trajs = exp2_ext.run_loop_ext(LLMClient(providers={"fake": fp}), _ccfg(), _ccfg_nv(), setup)
    assert len(trajs) == 4 * 4  # three new arms + the one fill arm per start
    rd = setup.run_dir
    r1 = list(read_jsonl(exp2.round_dir(rd, 1) / "scores.jsonl"))
    assert {r["score"] for r in r1} == {0.9, 0.1}  # round 1: the reused scores, nothing scored
    assert all(t not in scorer.texts for t in (s.text for s in starts))
    rows = list(read_jsonl(rd / "decisions.jsonl"))
    assert len(rows) == 16 * 4  # always-rewrite: every trajectory decides in all four rounds
    for r in rows:  # the seed key is (namespace exp2, x0, t): no arm
        want = derive_seed(1, "controller", "exp2", r["x0_id"], r["t"])
        assert r["attempts"][0]["seed"] == want
        if r["arm"] == "no_verifier":
            assert r["s1"] is None and r["tau1"] is None and r["fields"] is None
    ph = exp2_ext.read_pihat(rd)
    assert len(ph) == 4 * 4  # B_pihat: one row per start per round
    meta = json.loads((rd / exp2_ext.RUN_META).read_text(encoding="utf-8"))
    assert meta["t1_scores"] == "reused" and meta["seed_namespace"] == "exp2"
    nv_calls = [c for c in fp.calls if c.system.startswith("no verifier")]
    assert len(nv_calls) == 4 * 4 and all(
        "CHECKER" not in c.messages[0]["content"] for c in nv_calls
    )


def test_exp2_e_correct_scores_each_start_once_and_records_the_comparison(tmp_path):
    starts = _states(5, prefix="g")
    scorer = _Counting()
    v1, v2 = stub_pair()
    reference = {s.x0_id: (v1.score([s.text])[0], v2.score([s.text])[0]) for s in starts}
    reference[starts[0].x0_id] = (0.123, reference[starts[0].x0_id][1])  # one differs
    plan = {s.x0_id: list(exp2_ext.COHORTS[exp2_ext.E_CORRECT].arms) for s in starts}
    prior = {text_hash(s.text): _eval(s.text, SafetyLabel.SAFE) for s in starts}
    cal_free = [a for a in plan[starts[0].x0_id] if a is not Arm.B_PIHAT]
    plan = {x: cal_free for x in plan}
    setup = _setup2(tmp_path, starts, exp2_ext.E_CORRECT, plan, scorer,
                    t1_reference=reference, prior_judgments=prior,
                    judge_fn=lambda text: _eval(text, SafetyLabel.SAFE))  # fmt: skip
    client = LLMClient(providers={"fake": FakeProvider(responder=_accept)})
    summary = exp2_ext.run_exp2_ext(client, _ccfg(), _ccfg_nv(), setup)
    assert scorer.texts == [s.text for s in starts]  # round 1: one text per start, all arms share
    check = json.loads((setup.run_dir / exp2_ext.T1_CHECK).read_text(encoding="utf-8"))
    assert check["n"] == 5 and check["v1_identical"] == 4 and check["v2_identical"] == 5
    assert summary["n_trajectories"] == 5 * 4
    assert all(a["TFR"] == 0.0 for a in summary["arms"].values())
    for r in read_jsonl(setup.run_dir / "decisions.jsonl"):
        assert r["attempts"][0]["seed"] == derive_seed(
            1, "controller", "exp2_ecorrect", r["x0_id"], 1
        )


def test_exp2_no_verifier_alone_needs_no_scorer(tmp_path):
    starts = _states(3)
    setup = _setup2(tmp_path, starts, exp2_ext.E_INCORRECT,
                    {s.x0_id: [Arm.NO_VERIFIER] for s in starts}, None)  # fmt: skip
    client = LLMClient(providers={"fake": FakeProvider(responder=_rewrite)})
    exp2_ext.run_loop_ext(client, None, _ccfg_nv(), setup)
    rows = list(read_jsonl(setup.run_dir / "decisions.jsonl"))
    assert len(rows) == 3 * 4 and all(r["s1"] is None for r in rows)
    assert all(not list(read_jsonl(exp2.round_dir(setup.run_dir, t) / "scores.jsonl"))
               for t in range(1, 5))  # fmt: skip
    with pytest.raises(exp2_ext.Exp2ExtError):  # a verifier arm without a scorer is refused
        exp2_ext.run_loop_ext(client, _ccfg(), _ccfg_nv(), _setup2(
            tmp_path / "b", starts, exp2_ext.E_INCORRECT, {s.x0_id: [Arm.B] for s in starts},
            None))  # fmt: skip


def test_exp2_refusals(tmp_path, calibrator):
    starts = _states(2)
    scorer = _Counting()
    cases = [
        # an arm the cohort does not run in the extension
        _setup2(tmp_path, starts, exp2_ext.E_CORRECT,
                {s.x0_id: [Arm.PERMUTED_B] for s in starts}, scorer),
        # reused t = 1 scores must cover every planned start
        _setup2(tmp_path, starts, exp2_ext.E_INCORRECT, {s.x0_id: [Arm.B] for s in starts},
                scorer, t1_scores={starts[0].x0_id: (0.5, 0.5)}),
        # B + pi-hat without the calibrator
        _setup2(tmp_path, starts, exp2_ext.E_INCORRECT,
                {s.x0_id: [Arm.B_PIHAT] for s in starts}, scorer),
        # K is locked
        _setup2(tmp_path, starts, exp2_ext.E_INCORRECT, {s.x0_id: [Arm.B] for s in starts},
                scorer, k=3),
    ]  # fmt: skip
    for setup in cases:
        with pytest.raises(exp2_ext.Exp2ExtError):
            exp2_ext.run_loop_ext(
                LLMClient(providers={"fake": FakeProvider()}), _ccfg(), _ccfg_nv(), setup
            )


def test_runners_and_calibrator_never_reach_the_label_gate():
    """Only calibration.fit and the pipelines read y*; the runners and the frozen calibrator do
    not (the decision path rule, extended to the extension's modules)."""
    import ast

    from rafr.core import paths

    src = paths.ROOT / "src" / "rafr"
    for rel in (
        "calibration/apply.py",
        "calibration/platt.py",
        "experiments/exp1_ext.py",
        "experiments/exp2_ext.py",
        "experiments/ext_cohort.py",
        "formatter/extended.py",
    ):
        tree = ast.parse((src / rel).read_text(encoding="utf-8"))
        mods = {n.module for n in ast.walk(tree) if isinstance(n, ast.ImportFrom) and n.module} | {
            a.name for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names
        }
        assert not any(m.startswith(("rafr.data.labels", "rafr.evaluator", "rafr.calibration.fit"))
                       for m in mods), rel  # fmt: skip
