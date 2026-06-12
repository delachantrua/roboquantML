"""Two-way fixed-effects event study -- the *biased* staggered-DiD baseline.

This estimator is included deliberately as a foil. Under staggered adoption
with heterogeneous and dynamic treatment effects, the dynamic TWFE event study
contaminates each relative-time coefficient with "forbidden comparisons"
(already-treated units acting as controls; Goodman-Bacon 2021, Sun-Abraham
2021). We report it ONLY to contrast with Callaway-Sant'Anna / Sun-Abraham --
never as the headline estimate.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

from .callaway_santanna import NEVER_TREATED


def estimate_twfe_event_study(
    panel: pd.DataFrame,
    unit: str = "unit",
    period: str = "period",
    outcome: str = "y",
    ref_event_time: int = -1,
    max_lead: int | None = None,
    max_lag: int | None = None,
) -> pd.DataFrame:
    """Dynamic TWFE: y_it = a_i + l_t + sum_{e != ref} b_e 1{t-g = e} + eps.

    Never-treated units contribute only to the fixed effects (their event time
    is undefined and excluded from the indicators). Standard errors are
    cluster-robust by unit.
    """
    df = panel.copy()
    df["event_time"] = np.where(
        df["cohort"] == NEVER_TREATED, np.nan, df[period] - df["cohort"]
    )
    if max_lead is not None:
        df.loc[df["event_time"] < -max_lead, "event_time"] = -max_lead
    if max_lag is not None:
        df.loc[df["event_time"] > max_lag, "event_time"] = max_lag

    event_times = sorted(int(e) for e in df["event_time"].dropna().unique())
    # Build one indicator per event time except the reference (omitted) period,
    # all at once to avoid DataFrame fragmentation.
    ind_cols = []
    new_cols = {}
    for e in event_times:
        if e == ref_event_time:
            continue
        col = f"e_{'m' if e < 0 else 'p'}{abs(e)}"
        new_cols[col] = (df["event_time"] == e).astype(float)
        ind_cols.append((e, col))
    df = pd.concat([df, pd.DataFrame(new_cols, index=df.index)], axis=1)

    formula = f"{outcome} ~ " + " + ".join(c for _, c in ind_cols) + f" + C({unit}) + C({period})"
    model = smf.ols(formula, data=df).fit(
        cov_type="cluster", cov_kwds={"groups": df[unit]}
    )

    rows = [{"event_time": ref_event_time, "att": 0.0, "se": 0.0}]
    for e, col in ind_cols:
        rows.append(
            {"event_time": e, "att": float(model.params[col]), "se": float(model.bse[col])}
        )
    out = pd.DataFrame(rows).sort_values("event_time").reset_index(drop=True)
    out["ci_low"] = out["att"] - 1.96 * out["se"]
    out["ci_high"] = out["att"] + 1.96 * out["se"]
    return out
