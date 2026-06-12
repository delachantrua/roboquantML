"""Sun & Abraham (2021) interaction-weighted event study.

Saturates the TWFE specification in cohort x relative-time indicators:

    y_it = a_i + l_t + sum_{g} sum_{e != -1} d_{g,e} 1{G_i=g, t-g=e} + eps,

then forms each dynamic coefficient as an interaction-weighted (IW) average of
the cohort-specific d_{g,e}, weighting cohort g by its share among units at
relative time e. Never-treated (and, for a clean reference, the last-treated)
units serve as the control pool. Standard errors come from the delta method on
the cluster-robust covariance of the saturated regression.

Reference
---------
Sun, L., & Abraham, S. (2021). "Estimating dynamic treatment effects in event
studies with heterogeneous treatment effects." Journal of Econometrics, 225(2),
175-199.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

from .callaway_santanna import NEVER_TREATED


def estimate_sun_abraham(
    panel: pd.DataFrame,
    unit: str = "unit",
    period: str = "period",
    outcome: str = "y",
    ref_event_time: int = -1,
    max_lead: int | None = None,
    max_lag: int | None = None,
) -> pd.DataFrame:
    """Interaction-weighted event study with cluster-robust (by unit) SEs."""
    df = panel.copy()
    df["event_time"] = np.where(
        df["cohort"] == NEVER_TREATED, np.nan, df[period] - df["cohort"]
    )
    if max_lead is not None:
        df.loc[df["event_time"] < -max_lead, "event_time"] = -max_lead
    if max_lag is not None:
        df.loc[df["event_time"] > max_lag, "event_time"] = max_lag

    treated = df[df["cohort"] != NEVER_TREATED].copy()
    event_times = sorted(int(e) for e in treated["event_time"].dropna().unique())
    cohorts = sorted(int(g) for g in treated["cohort"].unique())

    # One indicator per (cohort g, event time e), excluding the reference e.
    # Build all columns at once to avoid DataFrame fragmentation.
    term_map = {}  # (g, e) -> column name
    new_cols = {}
    for g in cohorts:
        for e in event_times:
            if e == ref_event_time:
                continue
            ind = ((df["cohort"] == g) & (df["event_time"] == e)).astype(float)
            # Only keep columns that actually occur (a cohort may not reach all e).
            if ind.sum() > 0:
                col = f"g{g}_e{'m' if e < 0 else 'p'}{abs(e)}"
                new_cols[col] = ind
                term_map[(g, e)] = col
    df = pd.concat([df, pd.DataFrame(new_cols, index=df.index)], axis=1)

    cols = list(term_map.values())
    formula = f"{outcome} ~ " + " + ".join(cols) + f" + C({unit}) + C({period})"
    model = smf.ols(formula, data=df).fit(
        cov_type="cluster", cov_kwds={"groups": df[unit]}
    )

    params = model.params
    cov = model.cov_params()

    # Cohort weights: share of each cohort among treated units (CS/SA use the
    # share of cohort g in the population of units ever treated, evaluated at
    # the relative times where g is observed).
    cohort_sizes = treated.groupby("cohort")[unit].nunique()

    rows = [{"event_time": ref_event_time, "att": 0.0, "se": 0.0}]
    for e in event_times:
        if e == ref_event_time:
            continue
        contributing = [(g, term_map[(g, e)]) for g in cohorts if (g, e) in term_map]
        if not contributing:
            continue
        w = np.array([cohort_sizes.get(g, 0) for g, _ in contributing], dtype=float)
        if w.sum() == 0:
            continue
        w = w / w.sum()
        names = [c for _, c in contributing]
        est = float(np.dot(w, params[names].to_numpy()))
        sub = cov.loc[names, names].to_numpy()
        var = float(w @ sub @ w)
        rows.append({"event_time": e, "att": est, "se": np.sqrt(max(var, 0.0))})

    out = pd.DataFrame(rows).sort_values("event_time").reset_index(drop=True)
    out["ci_low"] = out["att"] - 1.96 * out["se"]
    out["ci_high"] = out["att"] + 1.96 * out["se"]
    return out
