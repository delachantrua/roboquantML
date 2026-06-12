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


@dataclass
class SimulatedIncidents:
    incidents: pd.DataFrame          # ori, period(int), hour, victim_age
    cohorts: dict                    # ori -> cohort period (NEVER_TREATED if none)
    true_event_study: pd.DataFrame   # truth for the juvenile-curfew outcome
    true_overall_att: float
    curfew_start: int
    curfew_end: int


# Hour sets and age ranges used to expand stratum counts into incident rows.
_CURFEW_HOURS = np.array([22, 23, 0, 1, 2, 3, 4, 5])
_NONCURFEW_HOURS = np.array([h for h in range(24) if h not in set(_CURFEW_HOURS.tolist())])
_JUV_AGES = np.arange(10, 18)
_ADULT_AGES = np.arange(18, 41)


def simulate_incidents(
    n_units: int = 120,
    n_periods: int = 96,
    cohort_periods: tuple[int, ...] = (30, 48, 66),
    frac_never_treated: float = 0.4,
    theta: float = 4.0,
    tau: float = 4.0,
    unit_sd: float = 2.5,
    time_trend: float = -0.01,
    season_amp: float = 2.0,
    noise_sd: float = 1.5,
    base_counts: dict | None = None,
    displacement: float = 0.0,
    curfew_start: int = 22,
    curfew_end: int = 6,
    juvenile_max_age: int = 17,
    seed: int = 21,
) -> SimulatedIncidents:
    """Generate incident-level data where the curfew reduces ONLY juvenile,
    curfew-hour incidents -- the structure a valid falsification test should see.

    Four strata per agency-month are drawn from agency + time effects: juvenile
    x curfew (jc), juvenile x non-curfew (jn), adult x curfew (ac), adult x
    non-curfew (an). The dynamic treatment effect is applied only to ``jc``;
    ``displacement`` optionally pushes some suppressed violence into ``jn`` to
    demonstrate the displacement threat (default 0 -> clean falsification).

    The returned counts are expanded into incident rows (with sampled hours and
    ages) so the downstream victim-age join and curfew-hour aggregation are
    exercised end to end.
    """
    base_counts = base_counts or {"jc": 10, "jn": 12, "ac": 12, "an": 16}
    rng = np.random.default_rng(seed)
    periods = np.arange(n_periods)

    units = np.arange(n_units)
    is_never = rng.random(n_units) < frac_never_treated
    cohorts_arr = np.where(is_never, NEVER_TREATED, 0)
    treated_idx = units[~is_never]
    cohorts_arr[~is_never] = rng.choice(cohort_periods, size=treated_idx.size)
    cohorts = {int(i): int(cohorts_arr[i]) for i in units}

    cohort_factor = {g: 0.8 + 0.4 * i for i, g in enumerate(sorted(cohort_periods))}

    alpha = rng.normal(0.0, unit_sd, size=n_units)
    season = season_amp * np.sin(2 * np.pi * (periods % 12) / 12)
    lam = time_trend * periods + season

    chunks = []
    for i in units:
        g = cohorts_arr[i]
        if g != NEVER_TREATED:
            e = periods - g
            cf = cohort_factor[g]
            eff = np.where(e >= 0, -theta * cf * (1.0 - np.exp(-e / tau)), 0.0)
        else:
            eff = np.zeros(n_periods)

        for stratum, base in base_counts.items():
            mean = base + alpha[i] + lam
            if stratum == "jc":
                mean = mean + eff
            elif stratum == "jn":
                mean = mean + displacement * (-eff)  # displaced violence (>=0)
            cnt = np.clip(
                np.round(mean + rng.normal(0.0, noise_sd, n_periods)), 0, None
            ).astype(int)

            rep_periods = np.repeat(periods, cnt)
            m = rep_periods.size
            if m == 0:
                continue
            if stratum in ("jc", "ac"):
                hrs = rng.choice(_CURFEW_HOURS, size=m)
            else:
                hrs = rng.choice(_NONCURFEW_HOURS, size=m)
            ages = rng.choice(_JUV_AGES if stratum in ("jc", "jn") else _ADULT_AGES, size=m)
            chunks.append(
                pd.DataFrame(
                    {
                        "ori": int(i),
                        "period": rep_periods,
                        "hour": hrs,
                        "victim_age": ages,
                    }
                )
            )

    incidents = pd.concat(chunks, ignore_index=True)

    # Truth for the juvenile-curfew outcome (cohort-size weighted dynamic path).
    treated_sizes = pd.Series(cohorts_arr[~is_never]).value_counts()
    max_e = n_periods - min(cohort_periods)
    es_rows = []
    for ev in range(-min(cohort_periods) + 1, max_e):
        contribs, weights = [], []
        for g in cohort_periods:
            t = g + ev
            if 0 <= t < n_periods:
                contribs.append(_dynamic_effect(ev, cohort_factor[g], theta, tau))
                weights.append(treated_sizes.get(g, 0))
        if contribs and sum(weights) > 0:
            es_rows.append(
                {"event_time": ev, "true_att": float(np.average(contribs, weights=weights))}
            )
    true_es = pd.DataFrame(es_rows)
    post = true_es[true_es["event_time"] >= 0]
    true_overall = float(post["true_att"].mean()) if not post.empty else np.nan

    return SimulatedIncidents(
        incidents=incidents,
        cohorts=cohorts,
        true_event_study=true_es,
        true_overall_att=true_overall,
        curfew_start=curfew_start,
        curfew_end=curfew_end,
    )
