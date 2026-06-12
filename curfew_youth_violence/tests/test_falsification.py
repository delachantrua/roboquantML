"""Validation tests for the victim-age join and curfew-hours falsification.

Confirms that:
  * the curfew-hour classifier handles overnight windows,
  * the victim-age join + aggregation produce correct stratum counts,
  * on simulated incident data where ONLY juvenile curfew-hour violence is
    treated, Callaway-Sant'Anna finds the effect in curfew hours and ~nothing
    in non-curfew hours (a clean falsification).
"""

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from curfew.estimators import estimate_att_gt  # noqa: E402
from curfew.nibrs_incidents import aggregate_incidents, is_curfew_hour  # noqa: E402
from curfew.panel import assemble_panel  # noqa: E402
from curfew.pipeline import _falsification_verdict  # noqa: E402
from curfew.simulate import simulate_incidents  # noqa: E402


def test_is_curfew_hour_overnight():
    # 22:00-06:00 overnight window.
    for h in [22, 23, 0, 3, 5]:
        assert is_curfew_hour(h, 22, 6)
    for h in [6, 9, 12, 17, 21]:
        assert not is_curfew_hour(h, 22, 6)
    # Daytime (non-wrapping) window 9-17.
    assert is_curfew_hour(10, 9, 17)
    assert not is_curfew_hour(20, 9, 17)


def test_victim_age_join_counts():
    # Hand-built incidents: known juvenile/curfew composition.
    rows = [
        # ori, period, hour, age
        ("A", 0, 23, 15),   # juvenile + curfew
        ("A", 0, 23, 40),   # adult + curfew
        ("A", 0, 12, 16),   # juvenile + non-curfew
        ("A", 0, 12, 30),   # adult + non-curfew
        ("A", 1, 2, 17),    # juvenile + curfew
    ]
    inc = pd.DataFrame(rows, columns=["ori", "period", "hour", "victim_age"])
    strata = aggregate_incidents(inc, curfew_start=22, curfew_end=6, juvenile_max_age=17)

    jc = strata["juvenile_curfew"].set_index(["unit", "period"])["count"]
    assert jc[("A", 0)] == 1 and jc[("A", 1)] == 1
    jn = strata["juvenile_noncurfew"].set_index(["unit", "period"])["count"]
    assert jn[("A", 0)] == 1 and jn[("A", 1)] == 0
    ja = strata["juvenile_all"].set_index(["unit", "period"])["count"]
    assert ja[("A", 0)] == 2  # two juvenile victims in period 0
    aa = strata["all_ages_all"].set_index(["unit", "period"])["count"]
    assert aa[("A", 0)] == 4


def test_clean_falsification_on_simulated_incidents():
    sim = simulate_incidents(seed=21)
    strata = aggregate_incidents(sim.incidents, sim.curfew_start, sim.curfew_end)

    panel_c = assemble_panel(strata["juvenile_curfew"], sim.cohorts)
    panel_n = assemble_panel(strata["juvenile_noncurfew"], sim.cohorts)

    cs_c = estimate_att_gt(panel_c, min_event_time=-12, max_event_time=24,
                           n_boot=300, seed=21)
    cs_n = estimate_att_gt(panel_n, min_event_time=-12, max_event_time=24,
                           n_boot=300, seed=21)

    # Curfew-hour effect is real and negative; non-curfew is near zero.
    assert cs_c.overall_att < 0 and abs(cs_c.overall_att / cs_c.overall_se) > 1.96
    assert abs(cs_n.overall_att) < 0.5 * abs(cs_c.overall_att)

    verdict = _falsification_verdict(cs_c, cs_n)
    assert verdict["clean_falsification"], verdict


if __name__ == "__main__":
    test_is_curfew_hour_overnight()
    test_victim_age_join_counts()
    test_clean_falsification_on_simulated_incidents()
    print("All falsification/victim-age tests passed.")
