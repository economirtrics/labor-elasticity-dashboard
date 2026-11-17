"""Command-line refresh: pull the latest data and write elasticity snapshots.

    python -m labordash.refresh                       # industries, 12-month changes
    python -m labordash.refresh --dimension states --horizon 3 --source bls

Reads FRED_API_KEY / BLS_API_KEY from the environment when set. Useful for a
scheduled job (cron, GitHub Actions) that keeps a dated archive of estimates.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import pandas as pd

from . import elasticity as el
from .config import DEFAULT_STATES
from .pipeline import build_model_frame, load_data


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dimension", choices=["industries", "states"], default="industries")
    ap.add_argument("--source", choices=["fred", "bls"], default="fred")
    ap.add_argument("--horizon", type=int, choices=[1, 3, 12], default=12)
    ap.add_argument("--demand", choices=["pce", "indpro"], default="pce")
    ap.add_argument("--window", type=int, default=60)
    ap.add_argument("--include-pandemic", action="store_true")
    ap.add_argument("--out", type=Path, default=Path("data/snapshots"))
    args = ap.parse_args(argv)

    keys = DEFAULT_STATES if args.dimension == "states" else None
    b = load_data(args.dimension, keys, args.source, fred_key=os.environ.get("FRED_API_KEY"),
                  bls_key=os.environ.get("BLS_API_KEY"), force=True)
    bad = b.meta[b.meta["status"] != "live"]
    for _, r in bad.iterrows():
        print(f"warning: {r['series_id']} is {r['status']}: {r['error']}")
    if b.panel.empty or "cpi" not in b.macro or args.demand not in b.macro:
        print("error: core series failed to load")
        return 1

    frame = build_model_frame(b.macro, b.panel, args.horizon, args.demand)
    opts = dict(exclude_pandemic=not args.include_pandemic,
                month_dummies=el.needs_month_dummies(args.dimension == "industries", args.horizon))
    results = el.estimate_all(frame, args.horizon, **opts)
    rolling = []
    for unit, g in frame.groupby("unit", sort=False):
        r = el.rolling_unit(g, args.horizon, args.window, **opts)
        r.insert(0, "unit", unit)
        rolling.append(r)

    args.out.mkdir(parents=True, exist_ok=True)
    stamp = frame["date"].max().strftime("%Y-%m")
    tag = f"{args.dimension}_h{args.horizon}_{stamp}"
    results.to_csv(args.out / f"elasticities_{tag}.csv", index=False)
    pd.concat(rolling, ignore_index=True).to_csv(args.out / f"rolling_{tag}.csv", index=False)
    print(f"data through {stamp}: wrote {len(results)} unit estimates to {args.out}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
