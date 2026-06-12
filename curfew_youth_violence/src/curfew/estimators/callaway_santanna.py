"""Callaway & Sant'Anna (2021) staggered difference-in-differences estimator.

Self-contained implementation (no external DiD packages) of the group-time
average treatment effects ATT(g, t) with:

  * never-treated OR not-yet-treated comparison groups,
  * a *universal* base period g-1 (so pre-period coefficients are placebo
    estimates that test the parallel-trends / no-anticipation assumption),
  * cohort-size weighted aggregations: an event-study (dynamic) path, a
    simple overall ATT, and per-cohort/per-calendar summaries,
  * a clustered (by unit) multiplier bootstrap that yields both pointwise
    standard errors and a sup-t *uniform* confidence band for the event study.

The estimator is deliberately written to be readable and auditable rather than
maximally fast: the panel sizes in this project (tens to a few hundred
agencies x ~10 years of months) are small enough that clarity wins.

Reference
---------
Callaway, B., & Sant'Anna, P. H. C. (2021). "Difference-in-Differences with
multiple time periods." Journal of Econometrics, 225(2), 200-230.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd


# Sentinel cohort value for never-treated units. Using a large finite integer
# keeps all the "G > t" comparisons working without special-casing infinity.
NEVER_TREATED = 0


@dataclass
class ATTGTResult:
    """Group-time ATT(g, t) estimates and their aggregations."""

    att_gt: pd.DataFrame          # columns: cohort, period, event_time, att, se, n_treated, n_comp
    event_study: pd.DataFrame     # columns: event_time, att, se, ci_low, ci_high, band_low, band_high
    overall_att: float            # simple weighted average of post-treatment ATT(g,t)
    overall_se: float
    pretrend_pvalue: float        # joint test that all pre-period ATT(g,t) == 0
    comparison: str
    n_boot: int
    meta: dict = field(default_factory=dict)


def _balanced_wide(panel: pd.DataFrame, unit: str, period: str, outcome: str):
    """Return (wide outcome matrix [n_units x n_periods], units, periods, G).

    Units must be observed in every period (balanced panel). The build step
    upstream guarantees this; we assert it here so silent imbalance can't bias
    the long-difference comparisons.
    """
    periods = np.sort(panel[period].unique())
    units = np.sort(panel[unit].unique())
    pivot = panel.pivot_table(index=unit, columns=period, values=outcome, aggfunc="first")
    pivot = pivot.reindex(index=units, columns=periods)
    if pivot.isna().any().any():
        missing = int(pivot.isna().sum().sum())
        raise ValueError(
            f"Unbalanced panel: {missing} unit-period cells are missing. "
            "Build a balanced panel (fill or drop) before estimation."
        )
    # Cohort per unit (first treated period; NEVER_TREATED if never).
    g_series = panel.groupby(unit)["cohort"].first().reindex(units)
    return pivot.to_numpy(dtype=float), units, periods, g_series.to_numpy()


def _att_gt_pointwise(Y, periods, G, comparison):
    """Compute ATT(g, t) for every estimable (cohort g, period t) pair.

    Uses the universal base period b = g-1 (the last period before cohort g is
    treated). Δ_i = Y[i, t] - Y[i, b]. ATT(g,t) is the difference in mean Δ
    between cohort g and the comparison group. Returns a list of dict rows plus
    the per-unit influence contributions needed for the cluster bootstrap is
    handled separately (we bootstrap by resampling units directly).
    """
    period_index = {p: j for j, p in enumerate(periods)}
    cohorts = sorted(g for g in np.unique(G) if g != NEVER_TREATED)
    rows = []
    for g in cohorts:
        if g not in period_index:
            continue
        b_period = g - 1
        # Base period must exist in the observed window.
        if b_period not in period_index:
            continue
        jb = period_index[b_period]
        treated_mask = G == g
        n_treated = int(treated_mask.sum())
        if n_treated == 0:
            continue
        for t in periods:
            if t == b_period:
                continue  # ATT(g, g-1) == 0 by construction (omitted reference)
            jt = period_index[t]
            # Comparison group: untreated through max(t, base) so both periods
            # of the long difference are pre-treatment for comparison units.
            horizon = max(t, b_period)
            if comparison == "nevertreated":
                comp_mask = G == NEVER_TREATED
            else:  # not-yet-treated (includes never-treated)
                comp_mask = (G == NEVER_TREATED) | (G > horizon)
            comp_mask = comp_mask & (G != g)
            n_comp = int(comp_mask.sum())
            if n_comp == 0:
                continue
            d_treated = Y[treated_mask, jt] - Y[treated_mask, jb]
            d_comp = Y[comp_mask, jt] - Y[comp_mask, jb]
            att = d_treated.mean() - d_comp.mean()
            rows.append(
                {
                    "cohort": int(g),
                    "period": int(t),
                    "event_time": int(t - g),
                    "att": float(att),
                    "n_treated": n_treated,
                    "n_comp": n_comp,
                }
            )
    return pd.DataFrame(rows)


def _aggregate_event_study(att_gt: pd.DataFrame, G):
    """Cohort-size weighted dynamic (event-study) aggregation.

    For each event time e, average ATT(g, g+e) across cohorts g, weighting by
    the number of treated units in cohort g (CS dynamic aggregation).
    """
    cohort_sizes = pd.Series(G).value_counts()
    rows = []
    for e, grp in att_gt.groupby("event_time"):
        w = grp["cohort"].map(cohort_sizes).to_numpy(dtype=float)
        w = w / w.sum()
        rows.append({"event_time": int(e), "att": float(np.average(grp["att"], weights=w))})
    return pd.DataFrame(rows).sort_values("event_time").reset_index(drop=True)


def _overall_att(att_gt: pd.DataFrame, G):
    """Simple overall ATT: weighted mean of post-treatment ATT(g,t) (e >= 0)."""
    post = att_gt[att_gt["event_time"] >= 0]
    if post.empty:
        return np.nan
    cohort_sizes = pd.Series(G).value_counts()
    w = post["cohort"].map(cohort_sizes).to_numpy(dtype=float)
    w = w / w.sum()
    return float(np.average(post["att"], weights=w))


def estimate_att_gt(
    panel: pd.DataFrame,
    unit: str = "unit",
    period: str = "period",
    outcome: str = "y",
    comparison: str = "notyettreated",
    min_event_time: int | None = None,
    max_event_time: int | None = None,
    n_boot: int = 1000,
    alpha: float = 0.05,
    seed: int = 12345,
) -> ATTGTResult:
    """Estimate ATT(g,t), aggregate, and run a clustered multiplier bootstrap.

    Parameters
    ----------
    panel : long DataFrame with columns [unit, period, outcome, 'cohort'].
        'cohort' is the first treated period (integer) or ``NEVER_TREATED`` (0).
    comparison : {'notyettreated', 'nevertreated'}.
    min_event_time, max_event_time : optional trim of the event-time window for
        all aggregations and the pre-trend test. Trimming is standard practice;
        without it, far-from-adoption leads/lags are noisy and inflate the joint
        pre-trend test. ATT(g,t) cells are still computed on the full window;
        only the *aggregations* are restricted.
    n_boot : number of cluster (unit) bootstrap replications.
    alpha : significance level for confidence intervals / uniform band.
    """
    if comparison not in {"notyettreated", "nevertreated"}:
        raise ValueError("comparison must be 'notyettreated' or 'nevertreated'")

    Y, units, periods, G = _balanced_wide(panel, unit, period, outcome)

    point = _att_gt_pointwise(Y, periods, G, comparison)
    if min_event_time is not None:
        point = point[point["event_time"] >= min_event_time]
    if max_event_time is not None:
        point = point[point["event_time"] <= max_event_time]
    point = point.reset_index(drop=True)
    if point.empty:
        raise ValueError("No estimable ATT(g,t) cells; check cohorts and window.")
    es_point = _aggregate_event_study(point, G)
    overall_point = _overall_att(point, G)

    # ---- Cluster (unit) bootstrap -----------------------------------------
    rng = np.random.default_rng(seed)
    n_units = Y.shape[0]
    boot_att_gt = []   # aligned to point rows
    boot_es = []       # aligned to es_point event_times
    boot_overall = []
    es_event_times = es_point["event_time"].to_numpy()
    # Pre-index point rows so each bootstrap recomputes the *same* cells.
    for _ in range(n_boot):
        idx = rng.integers(0, n_units, size=n_units)
        Yb, Gb = Y[idx], G[idx]
        pb = _att_gt_pointwise(Yb, periods, Gb, comparison)
        if pb.empty:
            # Resample lost every treated cohort (possible when cohorts have
            # very few units). Contributes NaN to all cells.
            boot_att_gt.append(np.full(len(point), np.nan))
            boot_es.append(np.full(len(es_event_times), np.nan))
            boot_overall.append(np.nan)
            continue
        # Align by (cohort, period); missing cells -> NaN (dropped in std).
        merged = point[["cohort", "period"]].merge(
            pb[["cohort", "period", "att"]], on=["cohort", "period"], how="left"
        )
        boot_att_gt.append(merged["att"].to_numpy())
        if not pb.empty:
            esb = _aggregate_event_study(pb, Gb).set_index("event_time")["att"]
            boot_es.append(esb.reindex(es_event_times).to_numpy())
            boot_overall.append(_overall_att(pb, Gb))
        else:
            boot_es.append(np.full(len(es_event_times), np.nan))
            boot_overall.append(np.nan)

    boot_att_gt = np.vstack(boot_att_gt)
    boot_es = np.vstack(boot_es)
    boot_overall = np.asarray(boot_overall, dtype=float)

    point["se"] = np.nanstd(boot_att_gt, axis=0, ddof=1)
    es_se = np.nanstd(boot_es, axis=0, ddof=1)
    overall_se = float(np.nanstd(boot_overall, ddof=1))

    # Pointwise normal CIs for the event study.
    from scipy.stats import norm

    z = norm.ppf(1 - alpha / 2)
    es = es_point.copy()
    es["se"] = es_se
    es["ci_low"] = es["att"] - z * es_se
    es["ci_high"] = es["att"] + z * es_se

    # Sup-t uniform band: critical value = (1-alpha) quantile of the bootstrap
    # max-t statistic across event times (Callaway-Sant'Anna simultaneous band).
    with np.errstate(invalid="ignore", divide="ignore"):
        tstats = np.abs((boot_es - es_point["att"].to_numpy()) / es_se)
    valid = ~np.isnan(tstats).all(axis=1)  # drop resamples with no treated units
    max_t = np.nanmax(tstats[valid], axis=1)
    crit = np.nanquantile(max_t, 1 - alpha)
    es["band_low"] = es["att"] - crit * es_se
    es["band_high"] = es["att"] + crit * es_se

    # Joint pre-trend test on the aggregated event-study pre-period path
    # (well-conditioned: few coefficients vs. many bootstrap draws).
    pretrend_p = _pretrend_test(es_point["att"].to_numpy(), es_event_times, boot_es)

    point = point[
        ["cohort", "period", "event_time", "att", "se", "n_treated", "n_comp"]
    ].sort_values(["cohort", "period"]).reset_index(drop=True)

    return ATTGTResult(
        att_gt=point,
        event_study=es.reset_index(drop=True),
        overall_att=overall_point,
        overall_se=overall_se,
        pretrend_pvalue=pretrend_p,
        comparison=comparison,
        n_boot=n_boot,
        meta={"n_units": int(n_units), "n_periods": int(len(periods)), "alpha": alpha},
    )


def _pretrend_test(es_att: np.ndarray, event_times: np.ndarray, boot_es: np.ndarray) -> float:
    """Sup-t test that the pre-period event-study path is flat at zero.

    Compares the largest pre-period |t| statistic against its bootstrap null
    distribution (the same construction as the sup-t uniform band). This is
    numerically robust where a joint Wald test is not: with highly correlated
    pre-period coefficients the bootstrap covariance is near-singular and the
    Wald statistic explodes, producing spurious p ~ 0 even when every pre
    coefficient is individually tiny.

    Returns a p-value; small values flag violated parallel trends or
    anticipation.
    """
    pre = np.asarray(event_times) < 0
    if pre.sum() == 0:
        return np.nan
    theta = np.asarray(es_att)[pre]
    boot_pre = boot_es[:, pre]
    se = np.nanstd(boot_pre, axis=0, ddof=1)
    se = np.where(se > 0, se, np.nan)
    obs = np.nanmax(np.abs(theta / se))
    # Null distribution: bootstrap deviations around the point estimates.
    with np.errstate(invalid="ignore", divide="ignore"):
        tstats = np.abs((boot_pre - theta) / se)
    ok = ~np.isnan(tstats).all(axis=1)
    if ok.sum() < 50:
        return np.nan
    boot_max = np.nanmax(tstats[ok], axis=1)
    return float(np.mean(boot_max >= obs))
