"""FBI Crime Data Explorer (CDE) API client.

Pulls agency-level monthly offense counts that become the outcome in the
city x month staggered-DiD panel. The CDE API is the modern, documented gateway
to NIBRS/SRS data (https://cde.ucr.cjis.gov/, API at api.usa.gov/crime/fbi/cde).

Authentication
--------------
You need a FREE api.data.gov key: https://api.data.gov/signup/ . Put it in the
``FBI_CDE_API_KEY`` environment variable (or pass ``api_key=...``). Without a
key the pipeline falls back to validated simulation mode (see ``simulate.py``).

IMPORTANT data caveat
---------------------
The CDE *summarized* endpoint returns TOTAL offense counts per agency-month, not
victim-age-restricted counts. True juvenile victimization requires the NIBRS
victim-demographic tables (separate endpoints / bulk extracts). For a first-pass
panel we use total violent offenses per agency-month as the outcome and treat
"juvenile share" as a documented modelling assumption -- see ``juvenile_share``.
Do not report these as juvenile-only counts without the victim-age join.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass

import pandas as pd
import requests

CDE_BASE = "https://api.usa.gov/crime/fbi/cde"

# CDE offense slugs that make up "violent crime" most relevant to youth.
DEFAULT_OFFENSES = ("aggravated-assault", "robbery", "homicide")


@dataclass
class CDEConfig:
    api_key: str
    base_url: str = CDE_BASE
    max_retries: int = 4
    timeout: int = 30
    backoff: float = 2.0


class FBICDEClient:
    """Thin, retrying client over the CDE REST API."""

    def __init__(self, api_key: str | None = None, **kwargs):
        key = api_key or os.environ.get("FBI_CDE_API_KEY")
        if not key:
            raise RuntimeError(
                "No FBI CDE API key. Set FBI_CDE_API_KEY (free key at "
                "https://api.data.gov/signup/) or run the pipeline in --simulate mode."
            )
        self.cfg = CDEConfig(api_key=key, **kwargs)

    def _get(self, path: str, params: dict | None = None) -> dict:
        params = dict(params or {})
        params["API_KEY"] = self.cfg.api_key
        url = f"{self.cfg.base_url}/{path.lstrip('/')}"
        last_err = None
        for attempt in range(self.cfg.max_retries):
            try:
                resp = requests.get(url, params=params, timeout=self.cfg.timeout)
                if resp.status_code == 200:
                    return resp.json()
                # 429/5xx are transient; 4xx (except 429) are not.
                if resp.status_code not in (429, 500, 502, 503, 504):
                    resp.raise_for_status()
                last_err = f"HTTP {resp.status_code}"
            except requests.RequestException as exc:  # network-level
                last_err = str(exc)
            time.sleep(self.cfg.backoff * (2 ** attempt))
        raise RuntimeError(f"CDE request failed after retries ({path}): {last_err}")

    def agencies_by_state(self, state_abbr: str) -> pd.DataFrame:
        """All reporting agencies (ORIs) in a state."""
        data = self._get(f"agency/byStateAbbr/{state_abbr}")
        records = data if isinstance(data, list) else data.get("results", data)
        return pd.DataFrame(records)

    def agency_offense_monthly(
        self, ori: str, offense: str, from_month: str, to_month: str
    ) -> pd.DataFrame:
        """Monthly actual counts for one agency & offense.

        Parameters use MM-YYYY (CDE convention). Returns columns
        [period (YYYY-MM, str), offense, count].
        """
        data = self._get(
            f"summarized/agency/{ori}/{offense}",
            params={"from": from_month, "to": to_month},
        )
        # CDE returns offense series under "offenses"/"actuals" keyed by MM-YYYY.
        actuals = _extract_actuals(data)
        rows = [
            {"period": _mmyyyy_to_iso(k), "offense": offense, "count": float(v)}
            for k, v in actuals.items()
        ]
        return pd.DataFrame(rows)


def _extract_actuals(data: dict) -> dict:
    """Best-effort extraction of {MM-YYYY: count} from a CDE summarized payload.

    The CDE schema has shifted over time; we probe the common shapes and fail
    loudly if none match so the caller knows to inspect the raw response.
    """
    if not isinstance(data, dict):
        raise ValueError("Unexpected CDE payload (not an object).")
    # Shape A (current, verified 2026-06): {"offenses": {"actuals":
    #   {"<Agency Name> Offenses": {"MM-YYYY": n},
    #    "<Agency Name> Clearances": {...}}}}
    # We must select the agency *Offenses* series explicitly -- grabbing the
    # first dict could silently return Clearances.
    off = data.get("offenses") or data.get("actuals")
    if isinstance(off, dict):
        actuals = off.get("actuals", off)
        if isinstance(actuals, dict):
            offense_series = {
                k: v for k, v in actuals.items()
                if isinstance(v, dict) and k.endswith(" Offenses")
            }
            if len(offense_series) == 1:
                return next(iter(offense_series.values()))
            if len(offense_series) > 1:
                raise ValueError(
                    f"Multiple 'Offenses' series in CDE response: "
                    f"{sorted(offense_series)} -- disambiguate in _extract_actuals."
                )
            # Fallback: a single unlabelled dict of months.
            dicts = [v for v in actuals.values() if isinstance(v, dict)]
            if len(dicts) == 1:
                return dicts[0]
            if not dicts and actuals:
                return actuals
    # Shape B: {"results": [{"data_year":..., "month":..., "value":...}]}
    if isinstance(data.get("results"), list):
        out = {}
        for r in data["results"]:
            m = r.get("month") or r.get("data_month")
            y = r.get("data_year") or r.get("year")
            val = r.get("value") or r.get("actual") or r.get("count")
            if m and y and val is not None:
                out[f"{int(m):02d}-{int(y)}"] = val
        if out:
            return out
    raise ValueError(
        "Could not locate monthly actuals in CDE response; inspect the raw JSON "
        "and extend _extract_actuals() for the current schema."
    )


def _mmyyyy_to_iso(mmyyyy: str) -> str:
    """'06-2018' -> '2018-06'."""
    mm, yyyy = mmyyyy.split("-")
    return f"{int(yyyy):04d}-{int(mm):02d}"


def fetch_city_panel(
    client: FBICDEClient,
    agencies: list[dict],
    offenses=DEFAULT_OFFENSES,
    from_month: str = "01-2010",
    to_month: str = "12-2022",
) -> pd.DataFrame:
    """Build a tidy [agency, period, offense, count] frame for given agencies.

    ``agencies`` is a list of dicts each with at least 'ori' and a display
    'name'. Offenses are summed later into the violent-crime outcome.
    """
    frames = []
    for ag in agencies:
        ori = ag["ori"]
        for offense in offenses:
            df = client.agency_offense_monthly(ori, offense, from_month, to_month)
            if df.empty:
                continue
            df["ori"] = ori
            df["agency"] = ag.get("name", ori)
            frames.append(df)
    if not frames:
        raise RuntimeError("No data returned for the requested agencies/offenses.")
    return pd.concat(frames, ignore_index=True)
