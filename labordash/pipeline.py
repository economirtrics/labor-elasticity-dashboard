"""Extract -> transform: pull the catalog live and build the estimation panel."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .config import EARLIEST, MACRO, MACRO_LABELS, Unit, catalog
from .sources import DEFAULT_CACHE_DIR, fetch_many


@dataclass
class Bundle:
    macro: pd.DataFrame   # monthly index; columns unrate, cpi, payems, pce, indpro
    panel: pd.DataFrame   # long: date, unit, label, emp, wage
    meta: pd.DataFrame    # one row per series: freshness and fetch status
    units: list[Unit]     # units that loaded successfully


def _monthly(s: pd.Series) -> pd.Series:
    """Put a series on a gap-free month-start index so that diff(h) means h months."""
    s = s.copy()
    s.index = pd.to_datetime(s.index).to_period("M").to_timestamp()
    s = s[~s.index.duplicated(keep="last")].sort_index()
    return s.asfreq("MS")


def load_data(
    dimension: str = "industries",
    unit_keys: list[str] | None = None,
    source: str = "fred",
    *,
    fred_key: str | None = None,
    bls_key: str | None = None,
    force: bool = False,
    cache_dir=DEFAULT_CACHE_DIR,
) -> Bundle:
    """Pull every series the dashboard needs from the live APIs.

    ``source`` selects where the labor series (employment, earnings, CPI,
    unemployment) come from. The demand proxies are Fed / BEA products, not BLS
    ones, so they always come from FRED.
    """
    units = [u for u in catalog(dimension) if unit_keys is None or u.key in unit_keys]
    kw = dict(fred_key=fred_key, bls_key=bls_key, force=force, cache_dir=cache_dir)

    roles: dict[str, tuple[str, str]] = {}  # series_id -> (role label, source)
    labor_ids, fred_only_ids = [], []
    macro_ids: dict[str, str] = {}
    for role, (fred_id, bls_id) in MACRO.items():
        if source == "bls" and bls_id:
            labor_ids.append(bls_id)
            macro_ids[role] = bls_id
        else:
            (labor_ids if source == "fred" else fred_only_ids).append(fred_id)
            macro_ids[role] = fred_id
        roles[macro_ids[role]] = (MACRO_LABELS[role], "")
    for u in units:
        emp_id, wage_id = u.ids(source)
        labor_ids += [emp_id, wage_id]
        roles[emp_id] = (f"Employment: {u.label}", u.key)
        roles[wage_id] = (f"Hourly earnings: {u.label}", u.key)

    data, meta = fetch_many(labor_ids, source, **kw)
    if fred_only_ids:
        d2, m2 = fetch_many(fred_only_ids, "fred", **kw)
        data.update(d2)
        meta += m2

    macro = pd.DataFrame({role: _monthly(data[sid]) for role, sid in macro_ids.items() if sid in data})
    macro = macro.loc[EARLIEST:]
    macro.index.name = "date"

    frames, loaded = [], []
    for u in units:
        emp_id, wage_id = u.ids(source)
        if emp_id not in data or wage_id not in data:
            continue
        df = pd.DataFrame({"emp": _monthly(data[emp_id]), "wage": _monthly(data[wage_id])}).loc[EARLIEST:]
        df = df.dropna(how="all")
        if df.empty:
            continue
        df = df.rename_axis("date").reset_index()
        df.insert(1, "unit", u.key)
        df.insert(2, "label", u.label)
        frames.append(df)
        loaded.append(u)
    panel = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=["date", "unit", "label", "emp", "wage"])

    meta_df = pd.DataFrame(meta)
    meta_df.insert(0, "series", [roles.get(sid, (sid, ""))[0] for sid in meta_df["series_id"]])
    return Bundle(macro=macro, panel=panel, meta=meta_df, units=loaded)


def deflator(macro: pd.DataFrame) -> pd.Series:
    """CPI with short interior gaps bridged log-linearly.

    The October 2025 CPI was never collected (federal shutdown), which would
    otherwise knock out every real-wage change that touches that month.
    """
    cpi = np.log(macro["cpi"]).interpolate(limit=2, limit_area="inside")
    return np.exp(cpi)


def build_model_frame(macro: pd.DataFrame, panel: pd.DataFrame, horizon: int = 12, demand: str = "pce") -> pd.DataFrame:
    """Long panel of h-month log changes used by every regression.

    Columns: date, unit, label, dlog_emp, dlog_rwage, dlog_demand, real_wage, emp.
    Rows with any missing regressor are dropped, so the frame ends at the latest
    month for which employment, earnings, CPI and the demand proxy all exist.
    """
    if demand not in macro:
        raise KeyError(f"demand proxy {demand!r} did not load")
    cpi = deflator(macro)
    base = cpi.dropna().iloc[-1]
    d_demand = np.log(macro[demand]).diff(horizon)

    out = []
    for (unit, label), g in panel.groupby(["unit", "label"], sort=False):
        g = g.set_index("date").sort_index().asfreq("MS")
        real_wage = g["wage"] / cpi.reindex(g.index) * base  # dollars of the latest CPI month
        df = pd.DataFrame(
            {
                "unit": unit,
                "label": label,
                "dlog_emp": np.log(g["emp"]).diff(horizon),
                "dlog_rwage": np.log(real_wage).diff(horizon),
                "dlog_demand": d_demand.reindex(g.index),
                "real_wage": real_wage,
                "emp": g["emp"],
            }
        )
        out.append(df.dropna(subset=["dlog_emp", "dlog_rwage", "dlog_demand"]).reset_index())
    if not out:
        return pd.DataFrame(columns=["date", "unit", "label", "dlog_emp", "dlog_rwage", "dlog_demand", "real_wage", "emp"])
    return pd.concat(out, ignore_index=True)


def _yoy(s: pd.Series) -> pd.Series:
    return (s / s.shift(12) - 1) * 100


def macro_indicators(macro: pd.DataFrame) -> pd.DataFrame:
    """Headline national series for the overview tab (all in percent)."""
    cpi = deflator(macro)
    out = pd.DataFrame(index=macro.index)
    out["unrate"] = macro["unrate"]
    out["inflation"] = _yoy(cpi)
    out["payroll_growth"] = _yoy(macro["payems"])
    out["real_wage_growth"] = _yoy(macro["ahe"] / cpi)
    return out
