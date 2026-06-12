"""End-to-end pipeline: data -> panel -> staggered DiD -> figures & tables.

Two modes:

  * ``--simulate`` (default when no API key): draws a validated synthetic panel
    with a known effect, so the whole pipeline runs offline and you can see the
    estimators recover the truth.
  * live: fetches agency-month offense counts from the FBI CDE API for the
    cities in the curfew-policy panel, builds the real panel, and estimates.

The estimation layer is identical across modes -- only the data source differs.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import pandas as pd

from .estimators import (
    estimate_att_gt,
    estimate_sun_abraham,
    estimate_twfe_event_study,
)
from .nibrs import DEFAULT_OFFENSES, FBICDEClient, fetch_city_panel
from .panel import build_panel_from_counts
from .plots import plot_event_study
from .policies import load_policies
from .simulate import simulate_panel


def _estimate_all(panel, min_e, max_e, n_boot, comparison, seed):
    """Run CS + Sun-Abraham + TWFE on a prepared panel."""
    cs = estimate_att_gt(
        panel, comparison=comparison, min_event_time=min_e, max_event_time=max_e,
        n_boot=n_boot, seed=seed,
    )
    sa = estimate_sun_abraham(panel, max_lead=abs(min_e), max_lag=max_e)
    twfe = estimate_twfe_event_study(panel, max_lead=abs(min_e), max_lag=max_e)
    return cs, sa, twfe


def run_simulation(outdir: Path, n_boot: int = 1000, min_e: int = -12,
                   max_e: int = 24, comparison: str = "notyettreated", seed: int = 7):
    """Simulation mode: validate estimators against a known effect."""
    sim = simulate_panel(seed=seed)
    cs, sa, twfe = _estimate_all(sim.panel, min_e, max_e, n_boot, comparison, seed)

    truth = sim.true_event_study
    plot_event_study(
        cs.event_study, outdir / "event_study.png", sa=sa, twfe=twfe, truth=truth,
        title="Curfews & youth violence (SIMULATION: estimators vs. known truth)",
    )
    _write_outputs(outdir, cs, sa, twfe, mode="simulation", extra={
        "true_overall_att": sim.true_overall_att,
        "estimated_overall_att": cs.overall_att,
        "overall_att_abs_error": abs(cs.overall_att - sim.true_overall_att),
        "pretrend_pvalue": cs.pretrend_pvalue,
    })
    return cs, sa, twfe, sim


def run_live(config: dict, outdir: Path):
    """Live mode: fetch CDE data for cities in the policy panel and estimate."""
    policies = load_policies(config["policy_file"])
    client = FBICDEClient()

    # Resolve the set of agencies to pull: those with an ORI in the policy panel
    # (treated) plus any explicit donor/never-treated ORIs in the config.
    treated_oris = [o for o in policies["ori"].unique() if str(o).strip()]
    donor_oris = config.get("donor_oris", [])
    oris = list(dict.fromkeys(treated_oris + donor_oris))
    if not oris:
        raise RuntimeError(
            "No ORIs to fetch. Fill the 'ori' column in curfew_policies.csv and/or "
            "add 'donor_oris' (never-treated comparison cities) to config.yaml."
        )
    agencies = [{"ori": o, "name": o} for o in oris]

    counts = fetch_city_panel(
        client, agencies,
        offenses=config.get("offenses", DEFAULT_OFFENSES),
        from_month=config.get("from_month", "01-2010"),
        to_month=config.get("to_month", "12-2022"),
    )
    counts.to_csv(outdir / "raw_counts.csv", index=False)

    population = None
    if config.get("population_file"):
        population = pd.read_csv(config["population_file"])

    panel = build_panel_from_counts(
        counts, policies, population=population,
        juvenile_share=config.get("juvenile_share"),
    )
    panel.to_csv(outdir / "panel.csv", index=False)

    min_e = config.get("min_event_time", -12)
    max_e = config.get("max_event_time", 24)
    cs, sa, twfe = _estimate_all(
        panel, min_e, max_e, config.get("n_boot", 1000),
        config.get("comparison", "notyettreated"), config.get("seed", 7),
    )
    plot_event_study(cs.event_study, outdir / "event_study.png", sa=sa, twfe=twfe)
    _write_outputs(outdir, cs, sa, twfe, mode="live", extra={
        "n_agencies": int(panel["unit"].nunique()),
        "n_periods": int(panel["period"].nunique()),
        "pretrend_pvalue": cs.pretrend_pvalue,
        "overall_att": cs.overall_att,
        "overall_se": cs.overall_se,
    })
    return cs, sa, twfe, panel


def _write_outputs(outdir: Path, cs, sa, twfe, mode: str, extra: dict):
    outdir.mkdir(parents=True, exist_ok=True)
    cs.att_gt.to_csv(outdir / "att_gt.csv", index=False)
    cs.event_study.to_csv(outdir / "event_study_cs.csv", index=False)
    sa.to_csv(outdir / "event_study_sa.csv", index=False)
    twfe.to_csv(outdir / "event_study_twfe.csv", index=False)
    summary = {
        "mode": mode,
        "comparison": cs.comparison,
        "n_boot": cs.n_boot,
        "overall_att": cs.overall_att,
        "overall_se": cs.overall_se,
        "pretrend_pvalue": cs.pretrend_pvalue,
        **extra,
    }
    (outdir / "summary.json").write_text(json.dumps(summary, indent=2, default=float))
    return summary
