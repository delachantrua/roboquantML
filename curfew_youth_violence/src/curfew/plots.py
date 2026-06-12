"""Event-study plotting for the curfew study."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless / no display
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402


def plot_event_study(
    cs: pd.DataFrame,
    out_path: str | Path,
    sa: pd.DataFrame | None = None,
    twfe: pd.DataFrame | None = None,
    truth: pd.DataFrame | None = None,
    title: str = "Effect of curfew adoption on youth violence",
    ylabel: str = "ATT (violent incidents per 100k/month)",
):
    """Plot the Callaway-Sant'Anna event study with optional overlays.

    Shows the CS point estimates with pointwise CIs and the sup-t uniform band,
    overlaying Sun-Abraham and (biased) TWFE for comparison, and the true path
    when running on simulated data.
    """
    fig, ax = plt.subplots(figsize=(9, 5.5))

    # Uniform band (if present from CS bootstrap).
    if {"band_low", "band_high"}.issubset(cs.columns):
        ax.fill_between(
            cs["event_time"], cs["band_low"], cs["band_high"],
            color="C0", alpha=0.12, label="CS sup-t uniform band",
        )
    ax.errorbar(
        cs["event_time"], cs["att"],
        yerr=[cs["att"] - cs["ci_low"], cs["ci_high"] - cs["att"]],
        fmt="o-", color="C0", capsize=3, label="Callaway-Sant'Anna",
    )

    if sa is not None:
        ax.plot(sa["event_time"], sa["att"], "s--", color="C1", alpha=0.8,
                label="Sun-Abraham (IW)")
    if twfe is not None:
        ax.plot(twfe["event_time"], twfe["att"], "^:", color="C3", alpha=0.7,
                label="TWFE (biased baseline)")
    if truth is not None:
        # Clip the truth line to the estimated event-time window for readability.
        lo, hi = cs["event_time"].min(), cs["event_time"].max()
        t = truth[(truth["event_time"] >= lo) & (truth["event_time"] <= hi)]
        ax.plot(t["event_time"], t["true_att"], "-", color="k", lw=2,
                alpha=0.6, label="True effect (simulation)")

    ax.axhline(0, color="grey", lw=0.8)
    ax.axvline(-0.5, color="grey", ls="--", lw=0.8)
    ax.set_xlabel("Months relative to curfew adoption")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path
