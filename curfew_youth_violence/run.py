#!/usr/bin/env python3
"""CLI entry point for the curfews & youth-violence staggered-DiD pipeline.

Examples
--------
# Offline: validate estimators against a known synthetic effect (no API key).
python run.py --simulate

# Live: pull FBI CDE data for the cities in data/curfew_policies.csv.
FBI_CDE_API_KEY=xxxx python run.py --config config.yaml
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

import yaml  # noqa: E402

from curfew.pipeline import run_live, run_simulation  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--simulate", action="store_true",
                    help="Run in validated simulation mode (no network/API key).")
    ap.add_argument("--config", default="config.yaml", help="Path to config.yaml (live mode).")
    ap.add_argument("--outdir", default="outputs", help="Directory for figures & tables.")
    ap.add_argument("--n-boot", type=int, default=None, help="Override bootstrap reps.")
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # Choose mode: explicit --simulate, or auto-fallback when no API key is set.
    use_sim = args.simulate or not os.environ.get("FBI_CDE_API_KEY")
    if use_sim and not args.simulate:
        print("No FBI_CDE_API_KEY found -> running in simulation mode. "
              "Set the key and pass --config for live data.\n")

    if use_sim:
        n_boot = args.n_boot or 1000
        cs, sa, twfe, sim = run_simulation(outdir, n_boot=n_boot)
        print(f"SIMULATION | true overall ATT = {sim.true_overall_att:+.2f} | "
              f"estimated (CS) = {cs.overall_att:+.2f} "
              f"(abs err {abs(cs.overall_att - sim.true_overall_att):.2f}) | "
              f"pretrend p = {cs.pretrend_pvalue:.3f}")
    else:
        with open(args.config) as fh:
            config = yaml.safe_load(fh)
        if args.n_boot:
            config["n_boot"] = args.n_boot
        cs, sa, twfe, panel = run_live(config, outdir)
        sig = "" if cs.overall_se == 0 else f" (se {cs.overall_se:.2f})"
        print(f"LIVE | agencies={panel['unit'].nunique()} | "
              f"overall ATT (CS) = {cs.overall_att:+.2f}{sig} | "
              f"pretrend p = {cs.pretrend_pvalue:.3f}")

    print(f"\nWrote figures & tables to: {outdir.resolve()}")


if __name__ == "__main__":
    main()
