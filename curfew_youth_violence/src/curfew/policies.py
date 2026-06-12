"""Load and validate the curated curfew-policy panel.

The policy panel is the hand-collected keystone of the design: city x effective
month of curfew adoption/tightening. Treat every row as a claim that needs a
source and verification -- ``load_policies`` surfaces unverified rows loudly.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import pandas as pd

REQUIRED_COLUMNS = {
    "city",
    "state",
    "ori",
    "policy_type",
    "effective_month",
    "source_url",
    "verified",
}


def load_policies(path: str | Path, require_verified: bool = False) -> pd.DataFrame:
    """Read the curfew-policy CSV, validate schema, and normalise dates.

    Parameters
    ----------
    require_verified : if True, drop rows whose ``verified`` flag is not truthy
        (use this once you have checked sources). Otherwise unverified rows are
        kept but a warning is emitted.
    """
    df = pd.read_csv(path, dtype=str).fillna("")
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"Policy file missing columns: {sorted(missing)}")

    df["verified"] = df["verified"].str.strip().str.lower().isin({"true", "1", "yes"})
    # effective_month -> period integer is assigned later against the panel grid;
    # here we just validate the YYYY-MM format.
    df["effective_month"] = df["effective_month"].str.strip()
    bad = df[~df["effective_month"].str.match(r"^\d{4}-\d{2}$")]
    if not bad.empty:
        raise ValueError(
            f"Bad effective_month values (need YYYY-MM): {bad['effective_month'].tolist()}"
        )

    n_unverified = int((~df["verified"]).sum())
    if n_unverified:
        msg = (
            f"{n_unverified}/{len(df)} curfew-policy rows are UNVERIFIED. "
            "Verify dates/sources before publishing results."
        )
        if require_verified:
            df = df[df["verified"]].reset_index(drop=True)
            warnings.warn(msg + " Dropping unverified rows (require_verified=True).")
        else:
            warnings.warn(msg)

    valid_types = {"adopt", "tighten", "loosen", "repeal"}
    bad_types = set(df["policy_type"].str.strip()) - valid_types
    if bad_types:
        raise ValueError(f"Unknown policy_type values: {sorted(bad_types)}")

    return df
