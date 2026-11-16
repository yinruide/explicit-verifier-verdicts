"""paper App. D.1 (2026-10-04): 110 E-INCORRECT per D_fit / D_gate is a design target, not a
stop."""

from pipelines import math_pipeline

from math_v2_helpers import make_harness
from rafr.contracts.schemas import Split
from rafr.data.labels import LabelPurpose
from rafr.domains.math.storage import read_json


def test_design_target_boundary_is_reported_without_raising(capsys):
    below = math_pipeline.incorrect_count_report(Split.D_GATE, [1] * 109 + [0] * 1391)
    assert below == {
        "split": "d_gate",
        "n_states": 1500,
        "n_incorrect": 109,
        "design_target": 110,
        "below_design_target": True,
        "rule": "M.2.1 design target; a shortfall is reported, splits are not adjusted",
    }
    err = capsys.readouterr().err
    assert "M.2.1 SCALE SHORTFALL: d_gate has 109 E-INCORRECT of 1500 states" in err
    met = math_pipeline.incorrect_count_report(Split.D_FIT, [1] * 110 + [0] * 1390)
    assert met["n_incorrect"] == 110 and met["below_design_target"] is False
    assert "SHORTFALL" not in capsys.readouterr().err


def test_live_fit_and_gate_continue_below_target_and_record_the_count(
    tmp_path, monkeypatch, capsys
):
    h = make_harness(tmp_path, monkeypatch)
    pipe = h.pipeline
    pipe.generate()
    pipe.split()
    for split in (Split.D_FIT, Split.UNLABELED_POOL, Split.D_GATE, Split.D_EXP1):
        pipe.score(split)
    # Before the revision only a live run stopped; exercise that branch.
    pipe.config.execution_mode = "live"
    accesses = []
    original = pipe._labels

    def track(split, purpose):
        accesses.append((split, purpose))
        return original(split, purpose)

    monkeypatch.setattr(pipe, "_labels", track)
    capsys.readouterr()

    fit = pipe.fit()
    count = fit["e_incorrect_count"]
    assert count["split"] == "d_fit" and count["n_states"] == 20
    assert 0 < count["n_incorrect"] < 110 and count["below_design_target"] is True
    assert read_json(h.root / "fit.json")["e_incorrect_count"] == count
    assert f"d_fit has {count['n_incorrect']} E-INCORRECT of 20 states" in capsys.readouterr().err
    # The count is reported from D_fit alone; D_gate labels still open only at step 9.
    assert accesses == [(Split.D_FIT, LabelPurpose.TAU_FIT)]

    gate = pipe.gate()
    count = gate["e_incorrect_count"]
    assert count["split"] == "d_gate" and count["n_states"] == 40
    assert count["n_incorrect"] == gate["n_states"] and count["below_design_target"] is True
    assert read_json(h.root / "gate_1.json")["e_incorrect_count"] == count
    assert f"d_gate has {count['n_incorrect']} E-INCORRECT of 40 states" in capsys.readouterr().err
    assert accesses[-1] == (Split.D_GATE, LabelPurpose.GATE)
