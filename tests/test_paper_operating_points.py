"""analysis/paper_operating_points.py: the axis ranges shared by the detox and math figures."""

from __future__ import annotations

from analysis import paper_operating_points as P


def _report(fa: float, frr: float = 0.1) -> dict:
    point = {"estimate": fa, "ci": [fa - 0.01, min(fa + 0.002, 1.0)]}
    arm = {"FA": point, "FRR": {"estimate": frr, "ci": [frr - 0.01, frr + 0.01]}}
    rule = {"FA": {"estimate": fa}, "FRR": {"estimate": frr}}
    return {"operating_points": {"high": {"points": {"arms": {"A1": arm}, "rules": {"AND": rule}}}}}


def test_false_accept_axis_runs_to_one_for_both_domains():
    assert P.YLIM == (0.0, 1.0)
    assert P.XLIM == (0.0, 0.55)
    # math's highest plotted FA: the AND rule at lambda high, 0.987
    assert P.check_range(_report(0.987), P.XLIM, P.YLIM) == []
    assert P.check_range(_report(0.987), P.XLIM, (0.0, 0.95))  # the old limit refused it


def test_values_outside_the_axes_are_reported_not_clipped():
    bad = P.check_range(_report(0.5, frr=0.6), P.XLIM, P.YLIM)
    assert bad and all("FRR" in b for b in bad)
