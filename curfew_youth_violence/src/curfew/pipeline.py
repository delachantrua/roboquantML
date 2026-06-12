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
from .nibrs_incidents import load_incident_extract
from .panel import assemble_panel, build_incident_panels, build_panel_from_counts
from .plots import plot_event_study, plot_falsification
from .policies import load_policies
from .simulate import simulate_incidents, simulate_panel


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


def _falsification_verdict(cs_curfew, cs_noncurfew) -> dict:
    """Summarise the curfew-hours vs. non-curfew-hours comparison.

    The design predicts a (negative) effect concentrated in curfew hours. We
    flag a clean falsification when the curfew-hour effect is significant and the
    non-curfew-hour effect is both small relative to it and not significant.
    """
    att_c, se_c = cs_curfew.overall_att, cs_curfew.overall_se
    att_n, se_n = cs_noncurfew.overall_att, cs_noncurfew.overall_se
    z_c = att_c / se_c if se_c else float("nan")
    z_n = att_n / se_n if se_n else float("nan")
    ratio = abs(att_n) / abs(att_c) if att_c else float("nan")
    clean = (
        att_c < 0 and abs(z_c) > 1.96            # real curfew-hour effect
        and abs(z_n) < 1.96                       # no significant non-curfew effect
        and ratio < 0.3                           # and it's small in magnitude
    )
    return {
        "curfew_overall_att": att_c, "curfew_overall_se": se_c, "curfew_z": z_c,
        "noncurfew_overall_att": att_n, "noncurfew_overall_se": se_n, "noncurfew_z": z_n,
        "noncurfew_to_curfew_ratio": ratio,
        "clean_falsification": bool(clean),
    }


def run_falsification_simulation(outdir: Path, n_boot: int = 600, min_e: int = -12,
                                 max_e: int = 24, comparison: str = "notyettreated",
                                 seed: int = 21):
    """Simulate incident-level data and run the curfew-hours falsification."""
    sim = simulate_incidents(seed=seed)
    from .nibrs_incidents import aggregate_incidents

    strata = aggregate_incidents(
        sim.incidents, curfew_start=sim.curfew_start, curfew_end=sim.curfew_end,
    )
    panel_c = assemble_panel(strata["juvenile_curfew"], sim.cohorts)
    panel_n = assemble_panel(strata["juvenile_noncurfew"], sim.cohorts)

    cs_c = estimate_att_gt(panel_c, comparison=comparison, min_event_time=min_e,
                           max_event_time=max_e, n_boot=n_boot, seed=seed)
    cs_n = estimate_att_gt(panel_n, comparison=comparison, min_event_time=min_e,
                           max_event_time=max_e, n_boot=n_boot, seed=seed)

    plot_falsification(cs_c.event_study, cs_n.event_study,
                       outdir / "falsification.png", truth=sim.true_event_study)
    verdict = _falsification_verdict(cs_c, cs_n)
    verdict.update({"mode": "simulation",
                    "true_curfew_overall_att": sim.true_overall_att})
    (outdir / "falsification.json").write_text(json.dumps(verdict, indent=2, default=float))
    cs_c.event_study.to_csv(outdir / "falsification_curfew_es.csv", index=False)
    cs_n.event_study.to_csv(outdir / "falsification_noncurfew_es.csv", index=False)
    return cs_c, cs_n, verdict, sim


def run_falsification_live(config: dict, outdir: Path):
    """Live incident-level path: juvenile victim-age join + curfew falsification."""
    policies = load_policies(config["policy_file"])
    incidents = load_incident_extract(
        config["incident_file"],
        offenses=tuple(config["offenses"]) if config.get("offenses") else None,
    )
    panels = build_incident_panels(
        incidents, policies,
        curfew_start=config.get("curfew_start", 22),
        curfew_end=config.get("curfew_end", 6),
        juvenile_max_age=config.get("juvenile_max_age", 17),
    )
    min_e = config.get("min_event_time", -12)
    max_e = config.get("max_event_time", 24)
    n_boot = config.get("n_boot", 1000)
    comparison = config.get("comparison", "notyettreated")
    seed = config.get("seed", 7)

    cs_c = estimate_att_gt(panels["juvenile_curfew"], comparison=comparison,
                           min_event_time=min_e, max_event_time=max_e,
                           n_boot=n_boot, seed=seed)
    cs_n = estimate_att_gt(panels["juvenile_noncurfew"], comparison=comparison,
                           min_event_time=min_e, max_event_time=max_e,
                           n_boot=n_boot, seed=seed)
    cs_all = estimate_att_gt(panels["juvenile_all"], comparison=comparison,
                             min_event_time=min_e, max_event_time=max_e,
                             n_boot=n_boot, seed=seed)

    plot_falsification(cs_c.event_study, cs_n.event_study, outdir / "falsification.png")
    plot_event_study(cs_all.event_study, outdir / "event_study_juvenile.png",
                     title="Effect of curfew on juvenile victimization (all hours)")
    verdict = _falsification_verdict(cs_c, cs_n)
    verdict.update({"mode": "live",
                    "juvenile_all_overall_att": cs_all.overall_att,
                    "juvenile_all_overall_se": cs_all.overall_se,
                    "n_agencies": int(panels["juvenile_curfew"]["unit"].nunique())})
    (outdir / "falsification.json").write_text(json.dumps(verdict, indent=2, default=float))
    for name, cs in [("curfew", cs_c), ("noncurfew", cs_n), ("juvenile_all", cs_all)]:
        cs.event_study.to_csv(outdir / f"falsification_{name}_es.csv", index=False)
    return cs_c, cs_n, cs_all, verdict


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
