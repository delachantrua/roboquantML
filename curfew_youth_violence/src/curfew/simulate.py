"""Staggered-adoption data-generating process with a KNOWN dynamic ATT.

Two uses:

  1. Validation. Because the true treatment-effect path is known exactly, we
     can confirm that Callaway-Sant'Anna and Sun-Abraham recover it (and that
     naive TWFE does not) -- the simulation test in ``tests/`` asserts this.
  2. Offline runs. The full pipeline can execute end-to-end without an FBI API
     key or network access by drawing a realistic placebo panel.

Outcome scale: juvenile violent incidents per 100,000 residents per month --
the panel's analysis outcome. Effects are additive on this scale so the true
ATT equals the simulated coefficients exactly. Cohort heterogeneity (later
adopters get larger effects) is built in on purpose, since that is precisely
the setting where TWFE breaks and CS/SA shine.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .estimators.callaway_santanna import NEVER_TREATED


@dataclass
class SimulatedPanel:
    panel: pd.DataFrame                 # long: unit, period, cohort, y, ...
    true_event_study: pd.DataFrame      # event_time, true_att
    true_overall_att: float


def _dynamic_effect(event_time: int, cohort_factor: float, theta: float, tau: float) -> float:
    """Saturating ramp: curfew bites gradually after adoption, then plateaus.

    Negative (violence-reducing) by convention. ``cohort_factor`` injects
    cross-cohort heterogeneity so TWFE is contaminated.
    """
    if event_time < 0:
        return 0.0
    return -theta * cohort_factor * (1.0 - np.exp(-event_time / tau))


def simulate_panel(
    n_units: int = 120,
    n_periods: int = 96,
    cohort_periods: tuple[int, ...] = (30, 48, 66),
    frac_never_treated: float = 0.4,
    theta: float = 8.0,
    tau: float = 4.0,
    unit_sd: float = 6.0,
    time_trend: float = -0.02,
    season_amp: float = 4.0,
    noise_sd: float = 3.0,
    base_rate: float = 40.0,
    seed: int = 7,
) -> SimulatedPanel:
    """Draw a balanced staggered-adoption panel with a known treatment path.

    Parameters mirror plausible juvenile-violence monthly rates: a base rate
    (~40 per 100k/month), agency heterogeneity, a mild secular decline, summer
    seasonality, and idiosyncratic noise.
    """
    rng = np.random.default_rng(seed)
    periods = np.arange(n_periods)

    # Assign units to cohorts (some never treated).
    units = np.arange(n_units)
    is_never = rng.random(n_units) < frac_never_treated
    cohorts = np.empty(n_units, dtype=int)
    cohorts[is_never] = NEVER_TREATED
    treated_units = units[~is_never]
    cohorts[~is_never] = rng.choice(cohort_periods, size=treated_units.size)

    # Later-adopting cohorts get larger effects -> heterogeneity across g.
    cohort_factor = {g: 0.8 + 0.4 * i for i, g in enumerate(sorted(cohort_periods))}

    alpha = rng.normal(0.0, unit_sd, size=n_units)            # agency FE
    season = season_amp * np.sin(2 * np.pi * (periods % 12) / 12)
    lam = time_trend * periods + season                       # time FE

    records = []
    for i in units:
        g = cohorts[i]
        cf = cohort_factor.get(g, 0.0) if g != NEVER_TREATED else 0.0
        for t in periods:
            e = (t - g) if g != NEVER_TREATED else None
            tau_e = _dynamic_effect(e, cf, theta, tau) if e is not None else 0.0
            y = base_rate + alpha[i] + lam[t] + tau_e + rng.normal(0.0, noise_sd)
            records.append((int(i), int(t), int(g), float(y)))

    panel = pd.DataFrame(records, columns=["unit", "period", "cohort", "y"])

    # True event-study path (cohort-size weighted, matching the estimand).
    treated_sizes = pd.Series(cohorts[~is_never]).value_counts()
    max_e = n_periods - min(cohort_periods)
    es_rows = []
    for e in range(-min(cohort_periods) + 1, max_e):
        contribs, weights = [], []
        for g in cohort_periods:
            t = g + e
            if 0 <= t < n_periods:
                contribs.append(_dynamic_effect(e, cohort_factor[g], theta, tau))
                weights.append(treated_sizes.get(g, 0))
        if contribs and sum(weights) > 0:
            es_rows.append({"event_time": e, "true_att": float(np.average(contribs, weights=weights))})
    true_es = pd.DataFrame(es_rows)

    post = true_es[true_es["event_time"] >= 0]
    true_overall = float(post["true_att"].mean()) if not post.empty else np.nan

    return SimulatedPanel(panel=panel, true_event_study=true_es, true_overall_att=true_overall)
