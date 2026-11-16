"""analysis/ext_ceiling_summary.py on synthetic inputs (no real run is read)."""

from __future__ import annotations

import hashlib
import json

import pytest
from analysis import ext_ceiling_summary as C


def _round(t, status):
    return {"t": t, "grade": {"status": status}, "output_sha256": f"s{t}"}


def _result(x0, statuses):
    rounds = [_round(t, s) for t, s in enumerate(statuses, start=1)]
    return {
        "x0_id": x0,
        "arm": C.CEILING,
        "n_decisions": len(rounds),
        "rounds": rounds,
        "task_success": "correct" in statuses,
    }


@pytest.mark.parametrize(
    ("reason", "prediction", "expected"),
    [
        ("prediction_missing_final_box", "", "no final box"),
        ("e_timeout_after_retries:worker_wall_clock_deadline", r"\int_0^1 f", "E timeout"),
        ("prediction_not_parsed", r"\begin{cases}0,&x<0\\1\end{cases}", "piecewise"),
        ("prediction_not_parsed", r"\bigl(0,\ \text{not uniform}\bigr)", "contains words"),
        ("prediction_not_parsed", r"\{z\in\mathbb C:|z|=1\}", "set"),
        ("prediction_not_parsed", r"1+\frac12+\cdots+\frac1n", "unevaluated"),
        ("prediction_not_parsed", r"\operatorname{Uniform}(0,1)", "named object"),
        ("prediction_not_parsed", r"C_6\rtimes C_2", "named object"),
        ("prediction_not_parsed", r"xy=C(x+y)^3,\quad y=0", "equation"),
        ("prediction_not_parsed", "42", "other"),
    ],
)
def test_answer_type_rules_in_priority_order(reason, prediction, expected):
    assert C.answer_type(reason, prediction).startswith(expected)


def test_first_correct_and_cumulative_rates_count_unresolved_as_not_correct():
    ceiling = [
        _result("a", ["correct"]),
        _result("b", ["incorrect", "unresolved", "correct"]),
        _result("c", ["unresolved", "incorrect", "incorrect", "incorrect"]),
        _result("d", ["incorrect"] * 4),
    ]
    assert [C.first_correct(r["rounds"]) for r in ceiling] == [1, 3, None, None]
    finals = [
        {"x0_id": x, "arm": arm, "task_success": ok}
        for x, ok in zip("abcd", (False, True, False, False), strict=True)
        for arm in (C.NV, C.A1)
    ]
    cum, failed, flagged = C.outcomes(ceiling, finals)
    assert [sum(cum[f"t<={t}"].values()) for t in range(1, 5)] == [1, 1, 2, 2]
    assert failed[C.CEILING] == {"a": 0.0, "b": 0.0, "c": 1.0, "d": 1.0}
    assert flagged == {"b", "c"}
    main = C.blocks(cum, failed, sorted("abcd"), n_boot=50, seed=1)
    contrast = main["tfr"]["contrasts"]["no_verifier - oracle_feedback"]
    assert contrast["n_pairs"] == 4 and contrast["estimate"] == pytest.approx(0.25)
    rest = C.blocks(cum, failed, ["a", "d"], n_boot=50, seed=1)  # the sensitivity drop
    assert rest["n_starts"] == 2 and rest["tfr"]["arms"][C.A1]["n"] == 2


def test_outcomes_require_every_arm_on_the_ceiling_starts():
    ceiling = [_result("a", ["correct"]), _result("b", ["incorrect"])]
    finals = [{"x0_id": "a", "arm": arm, "task_success": True} for arm in (C.NV, C.A1)]
    with pytest.raises(C.CeilingSummaryError, match="does not cover"):
        C.outcomes(ceiling, finals)


def test_load_ceiling_checks_the_completion_marker(tmp_path):
    rows = [_result("a", ["correct"]), _result("b", ["incorrect"])]
    (tmp_path / "ceiling_results.json").write_text(json.dumps(rows), encoding="utf-8")
    meta = {"SMOKE": False, "experiment": "correction_ceiling"}
    (tmp_path / "ext_meta.json").write_text(json.dumps(meta), encoding="utf-8")
    files = {
        n: hashlib.sha256((tmp_path / n).read_bytes()).hexdigest()
        for n in ("ceiling_results.json", "ext_meta.json")
    }
    marker = tmp_path / "ceiling_complete.json"
    marker.write_text(json.dumps({"n": 2, "files": files}), encoding="utf-8")
    assert [r["x0_id"] for r in C.load_ceiling(tmp_path)] == ["a", "b"]
    (tmp_path / "ceiling_results.json").write_text(json.dumps(rows[:1]), encoding="utf-8")
    with pytest.raises(C.CeilingSummaryError, match="sha256"):
        C.load_ceiling(tmp_path)


def test_load_ceiling_refuses_a_smoke_run(tmp_path):
    (tmp_path / "ceiling_results.json").write_text("[]", encoding="utf-8")
    meta = {"SMOKE": True, "experiment": "correction_ceiling"}
    (tmp_path / "ext_meta.json").write_text(json.dumps(meta), encoding="utf-8")
    (tmp_path / "ceiling_complete.json").write_text('{"n": 0, "files": {}}', encoding="utf-8")
    with pytest.raises(C.CeilingSummaryError, match="formal"):
        C.load_ceiling(tmp_path)
