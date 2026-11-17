"""Simulated series with a planted elasticity, used ONLY by the test suite.

The dashboard never touches this module: it always pulls live data. These
helpers exist so the estimators can be checked against a known truth, and so the
API clients can be exercised without a network connection.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from labordash.config import INDUSTRIES, MACRO


def simulate_levels(units, eta=-0.4, gamma=0.8, start="2006-01-01", periods=240, seed=0, eta_after=None, break_at=None):
    """Return ``{series_id: Series}`` keyed by FRED ids for macro + unit series.

    Monthly log changes follow  dlog E = eta * dlog w + gamma * dlog Y + noise,
    with w the real wage. ``eta_after``/``break_at`` plant a structural break.
    """
    rng = np.random.default_rng(seed)
    dates = pd.date_range(start, periods=periods, freq="MS")
    d_cpi = 0.002 + 0.001 * rng.standard_normal(periods)
    d_y = 0.002 + 0.006 * rng.standard_normal(periods)
    cpi = 200 * np.exp(np.cumsum(d_cpi))
    out = {
        MACRO["cpi"][0]: pd.Series(cpi, dates),
        MACRO["pce"][0]: pd.Series(12000 * np.exp(np.cumsum(d_y)), dates),
        MACRO["indpro"][0]: pd.Series(100 * np.exp(np.cumsum(d_y)), dates),
        MACRO["unrate"][0]: pd.Series(5 + np.cumsum(0.05 * rng.standard_normal(periods)), dates),
        MACRO["payems"][0]: pd.Series(140000 * np.exp(np.cumsum(0.001 + 0.5 * d_y)), dates),
        MACRO["ahe"][0]: pd.Series(25 * np.exp(np.cumsum(d_cpi + 0.0005)), dates),
    }
    for u in units:
        d_w = 0.0005 + 0.004 * rng.standard_normal(periods)
        e = np.full(periods, eta)
        if eta_after is not None:
            e[dates >= pd.Timestamp(break_at)] = eta_after
        d_e = e * d_w + gamma * d_y + 0.001 * rng.standard_normal(periods)
        out[u.emp_fred] = pd.Series(1000 * np.exp(np.cumsum(d_e)), dates)
        out[u.wage_fred] = pd.Series(20 * np.exp(np.cumsum(d_w)) * cpi / cpi[0], dates)
    return out


class FakeResponse:
    def __init__(self, status_code=200, text="", payload=None):
        self.status_code, self.text, self._payload = status_code, text, payload

    def json(self):
        if self._payload is None:
            raise ValueError("no json")
        return self._payload


def fake_request_factory(levels, calls=None):
    """Build a stand-in for ``requests.request`` that serves ``levels`` in the
    exact wire formats of the FRED JSON API, the FRED CSV endpoint and the BLS API."""
    bls_alias = {u.emp_bls: u.emp_fred for u in INDUSTRIES}
    bls_alias.update({bls: fred for fred, bls in MACRO.values() if bls})

    def fake(method, url, **kw):
        if calls is not None:
            calls.append(url)
        if "api.stlouisfed.org" in url:
            sid = kw["params"]["series_id"]
            if sid not in levels:
                return FakeResponse(400, payload={"error_code": 400, "error_message": "Bad Request. The series does not exist."})
            obs = [{"date": d.strftime("%Y-%m-%d"), "value": "." if np.isnan(v) else f"{v:.6f}"} for d, v in levels[sid].items()]
            return FakeResponse(200, payload={"observations": obs})
        if "fredgraph.csv" in url:
            sid = kw["params"]["id"]
            if sid not in levels:
                return FakeResponse(404, text="<html>Not Found</html>")
            rows = "\n".join(f"{d:%Y-%m-%d},{'.' if np.isnan(v) else f'{v:.6f}'}" for d, v in levels[sid].items())
            return FakeResponse(200, text=f"observation_date,{sid}\n{rows}\n")
        if "api.bls.gov" in url:
            body = kw["json"]
            y0, y1 = int(body["startyear"]), int(body["endyear"])
            series = []
            for sid in body["seriesid"]:
                s = levels.get(bls_alias.get(sid, sid))
                data = []
                if s is not None:
                    for d, v in s.items():
                        if y0 <= d.year <= y1:
                            data.append({"year": str(d.year), "period": f"M{d.month:02d}", "value": "-" if np.isnan(v) else f"{v:.6f}", "footnotes": [{}]})
                    data.append({"year": str(y1), "period": "M13", "value": "999", "footnotes": [{}]})
                series.append({"seriesID": sid, "data": data[::-1]})
            return FakeResponse(200, payload={"status": "REQUEST_SUCCEEDED", "message": [], "Results": {"series": series}})
        raise AssertionError(f"unexpected URL {url}")

    return fake
