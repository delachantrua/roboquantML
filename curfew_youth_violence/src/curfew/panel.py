"""Build the balanced city x month analysis panel for staggered DiD.

Combines FBI offense counts with the curfew-policy panel into the long frame the
estimators expect: columns [unit, period, cohort, y], where

  * ``unit``   = agency ORI (one city/agency),
  * ``period`` = integer month index (0 = earliest observed month),
  * ``cohort`` = integer period of first curfew adoption/tightening, or
                 ``NEVER_TREATED`` (0) for cities that never treated,
  * ``y``      = violent-offense outcome (count, or per-100k rate if population
                 supplied; optionally scaled by an assumed juvenile share).
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from .estimators.callaway_santanna import NEVER_TREATED


def _month_to_index(periods: pd.Series) -> tuple[pd.Series, dict]:
    """Map YYYY-MM strings to a dense integer index starting at 0."""
    uniq = sorted(periods.unique())
    lut = {p: i for i, p in enumerate(uniq)}
    return periods.map(lut), lut


def build_panel_from_counts(
    counts: pd.DataFrame,
    policies: pd.DataFrame,
    population: pd.DataFrame | None = None,
    juvenile_share: float | None = None,
    require_balanced: bool = True,
    interpolate_gaps: int = 0,
    log_outcome: bool = False,
) -> pd.DataFrame:
    """Assemble the analysis panel.

    Parameters
    ----------
    counts : long frame with [ori, period (YYYY-MM), offense, count].
    policies : output of ``policies.load_policies`` (needs ori, policy_type,
        effective_month).
    population : optional [ori, year, population] to convert counts to a
        per-100k monthly rate (recommended for cross-city comparability).
    juvenile_share : optional constant in (0, 1]; multiplies the outcome to
        approximate juvenile victimization when victim-age data is unavailable.
        Documented assumption, not a substitute for NIBRS victim-age tables.
    require_balanced : drop units not observed in every period (estimators need
        a balanced panel for the long-difference comparisons).
    interpolate_gaps : maximum length (months) of consecutive missing/zero runs
        to fill by linear interpolation within an agency series. Real CDE data
        has short reporting gaps (e.g. the 2021 SRS->NIBRS transition); a "0"
        in those runs is non-reporting, not zero crime. Gaps longer than this
        are left missing (the agency is then dropped by balancing). 0 disables.
    log_outcome : use log(violent) as the outcome -> ATT in log points
        (~percent changes), making effects comparable across cities of very
        different sizes. Requires strictly positive counts after interpolation.
    """
    df = counts.copy()
    # Treat blank/NaN counts as missing, sum offenses into violent per ori-month.
    df["count"] = pd.to_numeric(df["count"], errors="coerce")
    grp = df.groupby(["ori", "period"], as_index=False)["count"].agg(
        lambda s: s.sum(min_count=len(s))  # NaN if ANY component is missing
    )
    grp = grp.rename(columns={"count": "violent"})

    if interpolate_gaps:
        grp = _interpolate_short_gaps(grp, max_gap=interpolate_gaps)

    # Optional population -> per-100k rate.
    if population is not None:
        grp["year"] = grp["period"].str.slice(0, 4).astype(int)
        pop = population.copy()
        pop["year"] = pop["year"].astype(int)
        grp = grp.merge(pop[["ori", "year", "population"]], on=["ori", "year"], how="left")
        grp["y"] = 1e5 * grp["violent"] / grp["population"]
        grp = grp.drop(columns=["year", "population"])
    else:
        grp["y"] = grp["violent"]

    if juvenile_share is not None:
        if not 0 < juvenile_share <= 1:
            raise ValueError("juvenile_share must be in (0, 1].")
        grp["y"] = grp["y"] * juvenile_share

    if log_outcome:
        if (grp["y"].dropna() <= 0).any():
            bad = grp.loc[grp["y"] <= 0, "ori"].unique().tolist()
            raise ValueError(
                f"log_outcome requires positive counts; non-positive values for "
                f"{bad}. Increase interpolate_gaps or drop those agencies."
            )
        grp["y"] = np.log(grp["y"])

    # Drop agencies with unresolved missing months (gaps longer than the
    # interpolation cap) BEFORE balancing so the warning names them.
    has_nan = grp.groupby("ori")["y"].apply(lambda s: s.isna().any())
    bad_oris = has_nan[has_nan].index.tolist()
    if bad_oris:
        warnings.warn(
            f"Dropping {len(bad_oris)} agencies with unresolved reporting gaps "
            f"(longer than interpolate_gaps): {bad_oris}"
        )
        grp = grp[~grp["ori"].isin(bad_oris)]

    # Integer period index.
    grp["period_idx"], lut = _month_to_index(grp["period"])

    # Assign treatment cohorts from policy adopt/tighten events.
    cohort_lut = _build_cohort_lut(policies, lut)
    grp["cohort"] = grp["ori"].map(cohort_lut).fillna(NEVER_TREATED).astype(int)

    panel = grp.drop(columns=["period"]).rename(
        columns={"ori": "unit", "period_idx": "period"}
    )[["unit", "period", "cohort", "y", "violent"]]

    if require_balanced:
        panel = _balance(panel)

    return panel.sort_values(["unit", "period"]).reset_index(drop=True)


def _interpolate_short_gaps(
    grp: pd.DataFrame, max_gap: int, low_count_frac: float = 0.2
) -> pd.DataFrame:
    """Linearly interpolate short runs of non-reporting months within an agency.

    Big-city monthly *violent* totals (assault+robbery+homicide) are never
    genuinely near zero, so a month below ``low_count_frac`` x the agency's
    median (computed over plausible months) is treated as non-reporting --
    this catches both hard zeros AND partial-reporting months (e.g. Chicago
    June 2021 = 1 offense during the SRS->NIBRS transition). Runs longer than
    ``max_gap`` are left missing. Interpolated cells are flagged in the
    'interpolated' column.
    """
    out = []
    for ori, sub in grp.groupby("ori"):
        sub = sub.sort_values("period").copy()
        y = sub["violent"].astype(float)
        med = y[y > 0].median()
        y[y < low_count_frac * med] = np.nan
        filled = y.interpolate(method="linear", limit=max_gap, limit_area="inside")
        # Identify runs longer than max_gap and revert them to NaN.
        isna = y.isna()
        run_id = (isna != isna.shift()).cumsum()
        run_len = isna.groupby(run_id).transform("sum")
        too_long = isna & (run_len > max_gap)
        filled[too_long] = np.nan
        sub["interpolated"] = isna & filled.notna()
        sub["violent"] = filled
        out.append(sub)
    res = pd.concat(out, ignore_index=True)
    n_interp = int(res["interpolated"].sum())
    if n_interp:
        per_agency = res[res["interpolated"]].groupby("ori").size().to_dict()
        warnings.warn(
            f"Interpolated {n_interp} non-reporting agency-months "
            f"(zero/missing runs <= {max_gap}): {per_agency}"
        )
    return res


def assemble_panel(
    counts: pd.DataFrame,
    cohorts: dict,
    require_balanced: bool = True,
) -> pd.DataFrame:
    """Attach cohorts to a single-outcome count frame -> estimator panel.

    ``counts`` is long [unit, period(int), count]; ``cohorts`` maps unit -> first
    treated period (int) or ``NEVER_TREATED``. Used by the incident-level
    (juvenile / falsification) path where periods are already integer indices.
    """
    df = counts.rename(columns={"count": "y"}).copy()
    df["cohort"] = df["unit"].map(cohorts).fillna(NEVER_TREATED).astype(int)
    panel = df[["unit", "period", "cohort", "y"]]
    if require_balanced:
        panel = _balance(panel)
    return panel.sort_values(["unit", "period"]).reset_index(drop=True)


def build_incident_panels(
    incidents: pd.DataFrame,
    policies: pd.DataFrame,
    curfew_start: int = 22,
    curfew_end: int = 6,
    juvenile_max_age: int = 17,
) -> dict[str, pd.DataFrame]:
    """From incident records + policy panel, build one estimator panel per stratum.

    Returns a dict {outcome_name: panel[unit, period, cohort, y]} for the four
    strata produced by ``aggregate_incidents`` (juvenile_curfew is the primary
    outcome; juvenile_noncurfew is the falsification outcome). Periods (YYYY-MM)
    are converted to a dense integer index so the estimators' g-1 arithmetic
    works, and cohorts are mapped from the curfew effective months.
    """
    from .nibrs_incidents import aggregate_incidents

    strata = aggregate_incidents(
        incidents, curfew_start=curfew_start, curfew_end=curfew_end,
        juvenile_max_age=juvenile_max_age,
    )
    all_periods = sorted(incidents["period"].unique())
    lut = {p: i for i, p in enumerate(all_periods)}
    cohort_lut = _build_cohort_lut(policies, lut)

    panels = {}
    for name, c in strata.items():
        c = c.copy()
        c["period"] = c["period"].map(lut)
        c["cohort"] = c["unit"].map(cohort_lut).fillna(NEVER_TREATED).astype(int)
        panel = c.rename(columns={"count": "y"})[["unit", "period", "cohort", "y"]]
        panels[name] = _balance(panel).sort_values(["unit", "period"]).reset_index(drop=True)
    return panels


def _build_cohort_lut(policies: pd.DataFrame, period_lut: dict) -> dict:
    """Map each treated ORI to the integer period of its first onset event."""
    onset = policies[policies["policy_type"].isin(["adopt", "tighten"])].copy()
    onset = onset[onset["ori"].str.strip() != ""]
    if onset.empty:
        warnings.warn(
            "No curfew-policy rows have an ORI -> no city can be matched to "
            "treatment. Fill the 'ori' column in curfew_policies.csv."
        )
        return {}
    lut = {}
    for ori, rows in onset.groupby("ori"):
        # Earliest onset month that exists in the observed window.
        months = sorted(rows["effective_month"])
        matched = next((period_lut[m] for m in months if m in period_lut), None)
        if matched is None:
            warnings.warn(
                f"ORI {ori}: curfew effective month(s) {months} fall outside the "
                "observed data window; treated as never-treated."
            )
            continue
        lut[ori] = matched
    return lut


def _balance(panel: pd.DataFrame) -> pd.DataFrame:
    """Keep only units observed in every period (drop incomplete agencies)."""
    n_periods = panel["period"].nunique()
    counts = panel.groupby("unit")["period"].nunique()
    keep = counts[counts == n_periods].index
    dropped = set(counts.index) - set(keep)
    if dropped:
        warnings.warn(
            f"Dropping {len(dropped)} unbalanced unit(s) with gaps in reporting: "
            f"{sorted(dropped)[:5]}{'...' if len(dropped) > 5 else ''}"
        )
    return panel[panel["unit"].isin(keep)].copy()
