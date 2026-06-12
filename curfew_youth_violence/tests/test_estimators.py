"""Validation tests: estimators must recover the known simulated ATT.

These are the project's load-bearing tests. If Callaway-Sant'Anna can recover
the planted dynamic effect (and TWFE visibly cannot under heterogeneity), we
have evidence the estimation code is correct before pointing it at real NIBRS
data.
"""

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from curfew.estimators import (  # noqa: E402
    estimate_att_gt,
    estimate_sun_abraham,
    estimate_twfe_event_study,
)
from curfew.simulate import simulate_panel  # noqa: E402


def _merge_truth(est: pd.DataFrame, truth: pd.DataFrame) -> pd.DataFrame:
    return est.merge(truth, on="event_time", how="inner")


def test_callaway_santanna_recovers_truth():
    sim = simulate_panel(seed=7)
    res = estimate_att_gt(
        sim.panel, min_event_time=-12, max_event_time=24, n_boot=300, seed=1
    )

    cmp = _merge_truth(res.event_study, sim.true_event_study)
    post = cmp[cmp["event_time"] >= 0]
    # Mean absolute error on the post-treatment path should be small.
    mae = np.mean(np.abs(post["att"] - post["true_att"]))
    assert mae < 1.0, f"CS event-study MAE too large: {mae:.3f}"

    # Overall ATT close to truth (both strongly negative).
    assert abs(res.overall_att - sim.true_overall_att) < 1.0
    assert res.overall_att < 0

    # Pre-trends should NOT be rejected (data are generated with parallel trends).
    assert res.pretrend_pvalue > 0.01


def test_sun_abraham_recovers_truth():
    sim = simulate_panel(seed=11)
    sa = estimate_sun_abraham(sim.panel, max_lead=12, max_lag=24)
    cmp = _merge_truth(sa, sim.true_event_study)
    post = cmp[cmp["event_time"] >= 0]
    mae = np.mean(np.abs(post["att"] - post["true_att"]))
    assert mae < 1.0, f"Sun-Abraham event-study MAE too large: {mae:.3f}"


def test_twfe_is_biased_under_heterogeneity():
    # With strong cross-cohort heterogeneity, naive TWFE should be measurably
    # worse than CS on the post path -- documenting WHY we don't headline it.
    sim = simulate_panel(seed=3, theta=12.0)
    twfe = estimate_twfe_event_study(sim.panel, max_lead=12, max_lag=24)
    cs = estimate_att_gt(
        sim.panel, min_event_time=-12, max_event_time=24, n_boot=150, seed=2
    )

    twfe_cmp = _merge_truth(twfe, sim.true_event_study)
    cs_cmp = _merge_truth(cs.event_study, sim.true_event_study)
    twfe_post = twfe_cmp[twfe_cmp["event_time"] >= 0]
    cs_post = cs_cmp[cs_cmp["event_time"] >= 0]

    twfe_mae = np.mean(np.abs(twfe_post["att"] - twfe_post["true_att"]))
    cs_mae = np.mean(np.abs(cs_post["att"] - cs_post["true_att"]))
    assert cs_mae < twfe_mae, (
        f"Expected CS ({cs_mae:.3f}) to beat TWFE ({twfe_mae:.3f}) under heterogeneity"
    )


if __name__ == "__main__":
    test_callaway_santanna_recovers_truth()
    test_sun_abraham_recovers_truth()
    test_twfe_is_biased_under_heterogeneity()
    print("All validation tests passed.")
