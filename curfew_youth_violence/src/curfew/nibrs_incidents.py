"""Incident-level NIBRS adapter: victim-age join + curfew-hour stratification.

The summarized CDE endpoint (``nibrs.py``) only gives *total* monthly offense
counts. To build a genuinely **juvenile** outcome (victim age < 18) and a
**non-curfew-hours falsification** outcome, we need incident-level records that
carry victim age and the hour of the incident. Those come from NIBRS bulk
extracts (FBI CDE bulk downloads / NACJD ICPSR), not the summarized API.

This module:
  * defines the small incident schema we depend on,
  * classifies each incident as curfew-hour vs not (handles overnight windows),
  * aggregates incidents into agency x month count panels for four strata:
    juvenile-curfew (primary outcome), juvenile-noncurfew (falsification),
    juvenile-all, and all-ages-all (context).

The same aggregation is exercised on simulated incident data in the tests, so
the victim-age join and curfew-hour logic are validated offline.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# Columns the aggregation needs. ``period`` may be an int month index
# (simulation) or a 'YYYY-MM' string (real extracts) -- both group fine.
INCIDENT_COLUMNS = ("ori", "period", "hour", "victim_age")


def is_curfew_hour(hour: int, start: int = 22, end: int = 6) -> bool:
    """Is ``hour`` inside the nightly curfew window [start, end)?

    Handles the usual *overnight* curfew (e.g. 22:00-06:00) where start > end.
    """
    hour = int(hour)
    if start == end:
        return False
    if start < end:
        return start <= hour < end
    return hour >= start or hour < end  # overnight window wraps past midnight


def aggregate_incidents(
    incidents: pd.DataFrame,
    curfew_start: int = 22,
    curfew_end: int = 6,
    juvenile_max_age: int = 17,
    unit_col: str = "ori",
    period_col: str = "period",
    hour_col: str = "hour",
    age_col: str = "victim_age",
) -> dict[str, pd.DataFrame]:
    """Aggregate incident records into agency x month count panels by stratum.

    Returns a dict mapping outcome name -> long DataFrame [unit, period, count].
    Crucially, the count grid is the full unit x period cross-product with
    **zero-filled** empty cells, so months with no qualifying incident are 0
    rather than missing (missingness would bias the long-difference DiD).
    """
    missing = set(INCIDENT_COLUMNS) - set(incidents.columns) - {
        c for c in (unit_col, period_col, hour_col, age_col)
    }
    # Validate the actual columns we were pointed at exist.
    for c in (unit_col, period_col, hour_col, age_col):
        if c not in incidents.columns:
            raise ValueError(f"Incident data missing required column: {c!r}")

    df = incidents.copy()
    df["_juv"] = df[age_col].astype(float) <= juvenile_max_age
    df["_curfew"] = df[hour_col].map(lambda h: is_curfew_hour(h, curfew_start, curfew_end))

    units = sorted(df[unit_col].unique())
    periods = sorted(df[period_col].unique())
    grid = pd.MultiIndex.from_product([units, periods], names=["unit", "period"])

    def _counts(mask: pd.Series) -> pd.DataFrame:
        sub = df[mask]
        c = sub.groupby([unit_col, period_col]).size()
        c.index = c.index.set_names(["unit", "period"])
        return c.reindex(grid, fill_value=0).reset_index(name="count")

    all_true = pd.Series(True, index=df.index)
    return {
        "juvenile_curfew": _counts(df["_juv"] & df["_curfew"]),
        "juvenile_noncurfew": _counts(df["_juv"] & ~df["_curfew"]),
        "juvenile_all": _counts(df["_juv"]),
        "all_ages_all": _counts(all_true),
    }


def load_incident_extract(
    path: str,
    ori_col: str = "ori",
    datetime_col: str = "incident_datetime",
    age_col: str = "victim_age",
    offense_col: str | None = "offense_code",
    offenses: tuple[str, ...] | None = None,
) -> pd.DataFrame:
    """Load a NIBRS-style incident extract into the [ori, period, hour, age] schema.

    Expects one row per victim-incident with at least an agency ORI, an incident
    timestamp, and the victim's age. ``period`` is derived as 'YYYY-MM' and
    ``hour`` as the 0-23 hour of day. Optionally filter to a set of offense
    codes/slugs.

    Supports CSV and Parquet by extension.
    """
    if path.endswith(".parquet"):
        df = pd.read_parquet(path)
    else:
        df = pd.read_csv(path)

    if offenses and offense_col and offense_col in df.columns:
        df = df[df[offense_col].isin(offenses)]

    ts = pd.to_datetime(df[datetime_col], errors="coerce")
    keep = ts.notna() & df[age_col].notna()
    df = df[keep]
    ts = ts[keep]

    out = pd.DataFrame(
        {
            "ori": df[ori_col].astype(str).values,
            "period": ts.dt.strftime("%Y-%m").values,
            "hour": ts.dt.hour.values,
            "victim_age": pd.to_numeric(df[age_col], errors="coerce").values,
        }
    )
    return out.dropna(subset=["victim_age"]).reset_index(drop=True)
