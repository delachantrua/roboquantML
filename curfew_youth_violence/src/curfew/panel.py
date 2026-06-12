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
    """
    df = counts.copy()
    # Sum the component offenses into a single violent-crime outcome per ori-month.
    grp = df.groupby(["ori", "period"], as_index=False)["count"].sum()
    grp = grp.rename(columns={"count": "violent"})

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

    # Integer period index.
    grp["period_idx"], lut = _month_to_index(grp["period"])

    # Assign treatment cohorts from policy adopt/tighten events.
    cohort_lut = _build_cohort_lut(policies, lut)
    grp["cohort"] = grp["ori"].map(cohort_lut).fillna(NEVER_TREATED).astype(int)

    panel = grp.rename(columns={"ori": "unit", "period_idx": "period"})[
        ["unit", "period", "cohort", "y", "violent"]
    ]

    if require_balanced:
        panel = _balance(panel)

    return panel.sort_values(["unit", "period"]).reset_index(drop=True)


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
