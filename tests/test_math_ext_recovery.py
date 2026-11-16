"""Interrupted Exp1 streams are validated before resuming or declaring completion."""

import copy
import json

import pytest
from pipelines.math_ext_exp1 import BoundScoreIdentity
from pipelines.math_ext_exp1_runner import (
    COHORTS,
    E_CORRECT,
    Exp1ExtError,
    Exp1ExtSetup,
    run_exp1_ext,
)
from tests.test_math_ext_exp2 import calibration

from rafr.contracts.schemas import EvalRecord, Lambda, MathLabel, State, text_hash
from rafr.domains.math.generation import build_controller_config
from rafr.formatter.math_extended import (
    build_no_verifier_config,
    math_wording,
    original_feedback,
)
from rafr.llm.client import LLMClient
from rafr.llm.providers.fake import FakeProvider
from rafr.logging.writer import read_jsonl


class Identity:
    name = "identity"

    def __call__(self, state_id, a, b, lam):
        return a, b


def harness(tmp_path, responder=None):
    states = [
        State(state_id=f"s{i}", x0_id=f"q{i}", x0=f"Compute {i}+1", text=f"Answer {i + 1}")
        for i in range(3)
    ]
    setup = Exp1ExtSetup(
        run_id="recovery-fixture",
        run_dir=tmp_path / "exp1",
        cohort=E_CORRECT,
        states=states,
        scorer=BoundScoreIdentity("v1", "v2", "1", "1"),
        tau1=0.5,
        tau2=0.5,
        channel=Identity(),
        lambdas=(Lambda.LOW, Lambda.MID, Lambda.HIGH),
        arms=COHORTS[E_CORRECT].arms,
        wording=math_wording(),
        e_labels={
            s.state_id: EvalRecord(
                text_id=text_hash(s.text),
                text=s.text,
                label=MathLabel.CORRECT,
                model="fixture-E",
                model_version=None,
                prompt_hash="fixture",
                e_prompt_sha256="fixture",
                e_version="fixture",
                decoding={},
            )
            for s in states
        },
        expected_label=MathLabel.CORRECT,
        calibrator=calibration(),
        perm_seed=9,
        pihat_perm_seed=11,
        run_seed=17,
        score_states=lambda ss: [(0.1 + int(s.state_id[1:]) * 0.2, 0.8) for s in ss],
        original_feedback=original_feedback,
    )
    provider = FakeProvider(responder or (lambda req: '{"action":"accept","revised_text":""}'))
    client = LLMClient({"openai": provider})

    def run():
        return run_exp1_ext(
            client,
            build_controller_config(17, 1),
            build_no_verifier_config(17, 1),
            setup,
        )

    run()
    return setup.run_dir / "decisions.jsonl", provider, run


def write_rows(path, rows):
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


@pytest.mark.parametrize(
    "mutation",
    [
        "duplicate",
        "unexpected_cell",
        "run_id",
        "x0_id",
        "x_t",
        "seed",
        "model",
        "decoding",
        "score",
        "field",
        "donor",
        "e_label",
        "e_score",
        "posthoc",
        "attempt_seed",
        "attempt_hash",
        "attempt_parse",
        "attempt_failure",
        "raw_output",
        "retries",
        "prompt_hash",
        "extra_field",
    ],
)
def test_corrupt_unsealed_stream_stops_before_calls(tmp_path, mutation):
    path, provider, run = harness(tmp_path)
    rows = list(read_jsonl(path))[:10]  # A legitimate interrupted prefix has no completion seal.
    row = rows[0]
    if mutation == "duplicate":
        rows.append(copy.deepcopy(row))
    elif mutation == "unexpected_cell":
        row["state_id"] = "not-in-cohort"
    elif mutation in {"run_id", "x0_id", "x_t", "model"}:
        row[mutation] = "changed"
    elif mutation == "seed":
        row["seed"] += 1
    elif mutation == "decoding":
        row["decoding"]["max_output_tokens"] += 1
    elif mutation == "score":
        row["s1_tilde"] += 0.000000000001
    elif mutation == "field":
        row["fields"]["m1"] += 0.000000000001
    elif mutation == "donor":
        row["permuted_from_state_id"] = "s1"
    elif mutation == "e_label":
        row["e_label"] = "INCORRECT"
    elif mutation == "e_score":
        row["e_score"] = 0.8
    elif mutation == "posthoc":
        row["r_ref"] = 0.9
    elif mutation == "attempt_seed":
        row["attempts"][0]["seed"] += 1
    elif mutation == "attempt_hash":
        row["attempts"][0]["prompt_hash"] = "changed"
    elif mutation == "attempt_parse":
        row["attempts"][0]["parsed"]["action"] = "rewrite"
    elif mutation == "attempt_failure":
        row["attempts"][0]["parse_fail_reason"] = "malformed_json"
    elif mutation == "raw_output":
        row["attempts"][0]["raw_output"] = "malformed"
    elif mutation == "retries":
        row["n_parse_retries"] = 1
    elif mutation == "prompt_hash":
        row["prompt_hash"] = "changed"
    else:
        row["invented"] = True
    write_rows(path, rows)
    before, n_calls = path.read_bytes(), len(provider.calls)
    with pytest.raises(Exp1ExtError):
        run()
    assert path.read_bytes() == before
    assert len(provider.calls) == n_calls


@pytest.mark.parametrize("arm", COHORTS[E_CORRECT].arms)
def test_each_arm_outcome_is_replayed_from_raw_attempts(tmp_path, arm):
    path, provider, run = harness(tmp_path)
    rows = list(read_jsonl(path))
    row = next(row for row in rows if row["arm"] == arm.value)
    row["action"], row["revised_text"] = "rewrite", "Invented replacement"
    write_rows(path, rows)
    n_calls = len(provider.calls)
    with pytest.raises(Exp1ExtError, match="outcome"):
        run()
    assert len(provider.calls) == n_calls


@pytest.mark.parametrize("mode", ["accept", "retry_then_rewrite", "parse_fail"])
def test_valid_interrupted_stream_resumes_exactly_missing_cells(tmp_path, mode):
    def respond(req):
        if mode == "parse_fail" or (mode == "retry_then_rewrite" and req.attempt == 0):
            return "malformed"
        if mode == "retry_then_rewrite":
            return '{"action":"rewrite","revised_text":"Updated solution"}'
        return '{"action":"accept","revised_text":""}'

    path, provider, run = harness(tmp_path, respond)
    original = list(read_jsonl(path))
    prefix = original[:10]
    write_rows(path, prefix)
    n_calls = len(provider.calls)
    result = run()
    rows = list(read_jsonl(path))
    assert result["n_rows"] == 63 and result["n_new_decisions"] == 53
    assert len(rows) == len({(r["state_id"], r["arm"], r["lam"], r["t"]) for r in rows}) == 63
    assert rows[:10] == prefix
    assert (
        len(provider.calls) - n_calls
        == 53 * {"accept": 1, "retry_then_rewrite": 2, "parse_fail": 3}[mode]
    )
    n_calls = len(provider.calls)
    assert run()["n_new_decisions"] == 0
    assert len(provider.calls) == n_calls


def test_malformed_partial_json_is_preserved_and_refused(tmp_path):
    path, provider, run = harness(tmp_path)
    path.write_text(path.read_text(encoding="utf-8") + '{"state_id":', encoding="utf-8")
    before, n_calls = path.read_bytes(), len(provider.calls)
    with pytest.raises(Exp1ExtError, match="valid Core LogRecord"):
        run()
    assert path.read_bytes() == before and len(provider.calls) == n_calls


def test_parse_failure_with_missing_attempt_is_not_a_completed_cell(tmp_path):
    path, provider, run = harness(tmp_path, lambda req: "malformed")
    rows = list(read_jsonl(path))
    rows[0]["attempts"] = rows[0]["attempts"][:1]
    rows[0]["n_parse_retries"] = 0
    rows[0]["prompt_hash"] = rows[0]["attempts"][0]["prompt_hash"]
    write_rows(path, rows)
    n_calls = len(provider.calls)
    with pytest.raises(Exp1ExtError, match="exhaust"):
        run()
    assert len(provider.calls) == n_calls
