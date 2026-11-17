"""Live data clients for FRED and the BLS Public Data API, plus a small disk cache.

Every series is returned as a float ``pd.Series`` indexed by month-start
timestamps. Nothing here is specific to this dashboard.

FRED
    With an API key   -> https://api.stlouisfed.org/fred/series/observations (JSON)
    Without a key     -> https://fred.stlouisfed.org/graph/fredgraph.csv (public CSV)
BLS
    With a key        -> API v2 (50 series, 20 years per request, 500 requests/day)
    Without a key     -> API v1 (25 series, 10 years per request, 25 requests/day)
"""
from __future__ import annotations

import io
import os
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests

from .config import EARLIEST

FRED_API_URL = "https://api.stlouisfed.org/fred/series/observations"
FRED_CSV_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv"
BLS_V1_URL = "https://api.bls.gov/publicAPI/v1/timeseries/data/"
BLS_V2_URL = "https://api.bls.gov/publicAPI/v2/timeseries/data/"

USER_AGENT = "labor-elasticity-dashboard/1.0 (python-requests)"
DEFAULT_CACHE_DIR = Path(os.environ.get("LABORDASH_CACHE", Path(__file__).resolve().parent.parent / "data" / "cache"))
CACHE_TTL_HOURS = 6.0


class FetchError(RuntimeError):
    """A series could not be retrieved from its source."""


# --------------------------------------------------------------------------- #
# HTTP with retries
# --------------------------------------------------------------------------- #
def _request(method: str, url: str, *, retries: int = 3, timeout: int = 30, **kwargs) -> requests.Response:
    headers = {"User-Agent": USER_AGENT, **kwargs.pop("headers", {})}
    last: Exception | None = None
    for attempt in range(retries):
        try:
            resp = requests.request(method, url, headers=headers, timeout=timeout, **kwargs)
            if resp.status_code == 429 or resp.status_code >= 500:
                last = FetchError(f"HTTP {resp.status_code} from {url}")
            else:
                return resp
        except (requests.ConnectionError, requests.Timeout) as exc:
            last = exc
        time.sleep(1.5 ** attempt)
    raise FetchError(f"{url}: {last}")


def _to_series(dates, values, name: str) -> pd.Series:
    s = pd.Series(
        pd.to_numeric(pd.Series(list(values), dtype="object"), errors="coerce").to_numpy(dtype=float),
        index=pd.to_datetime(list(dates)),
        name=name,
    )
    s = s[~s.index.duplicated(keep="last")].sort_index()
    s.index.name = "date"
    return s


# --------------------------------------------------------------------------- #
# FRED
# --------------------------------------------------------------------------- #
def fetch_fred(series_id: str, start: str = EARLIEST, api_key: str | None = None) -> pd.Series:
    """Fetch one FRED series from ``start`` to the latest available observation."""
    if api_key:
        resp = _request(
            "GET",
            FRED_API_URL,
            params={
                "series_id": series_id,
                "api_key": api_key,
                "file_type": "json",
                "observation_start": start,
            },
        )
        if resp.status_code != 200:
            try:
                msg = resp.json().get("error_message", resp.text[:200])
            except ValueError:
                msg = resp.text[:200]
            raise FetchError(f"FRED {series_id}: {msg}")
        obs = resp.json().get("observations", [])
        if not obs:
            raise FetchError(f"FRED {series_id}: no observations returned")
        # FRED encodes missing values as "." -> coerced to NaN
        return _to_series([o["date"] for o in obs], [o["value"] for o in obs], series_id)

    resp = _request("GET", FRED_CSV_URL, params={"id": series_id, "cosd": start})
    text = resp.text
    first_line = text.splitlines()[0] if text.strip() else ""
    if resp.status_code != 200 or "," not in first_line:  # an HTML error page has no CSV header
        raise FetchError(f"FRED {series_id}: HTTP {resp.status_code} from the public CSV endpoint")
    df = pd.read_csv(io.StringIO(text), dtype=str)
    if df.shape[1] < 2 or df.empty:
        raise FetchError(f"FRED {series_id}: unexpected CSV layout")
    return _to_series(df.iloc[:, 0], df.iloc[:, 1], series_id)


# --------------------------------------------------------------------------- #
# BLS
# --------------------------------------------------------------------------- #
def _chunks(seq: list, n: int):
    for i in range(0, len(seq), n):
        yield seq[i : i + n]


def parse_bls_payload(payload: dict) -> dict[str, pd.Series]:
    """Turn one BLS API response into ``{series_id: Series}`` (monthly rows only)."""
    if payload.get("status") != "REQUEST_SUCCEEDED":
        raise FetchError("BLS: " + ("; ".join(payload.get("message", [])) or "request failed"))
    out: dict[str, pd.Series] = {}
    for item in payload.get("Results", {}).get("series", []):
        dates, values = [], []
        for row in item.get("data", []):
            period = row.get("period", "")
            if not (period.startswith("M") and period != "M13"):  # M13 = annual average
                continue
            dates.append(f"{row['year']}-{period[1:]}-01")
            values.append(row.get("value"))  # "-" marks unavailable months -> NaN
        if dates:
            out[item["seriesID"]] = _to_series(dates, values, item["seriesID"])
    return out


def fetch_bls(series_ids: list[str], start: str = EARLIEST, api_key: str | None = None) -> dict[str, pd.Series]:
    """Fetch many BLS series at once, chunking by the API's series and year limits."""
    url = BLS_V2_URL if api_key else BLS_V1_URL
    max_series, max_years = (50, 20) if api_key else (25, 10)
    first_year, last_year = int(start[:4]), datetime.now(timezone.utc).year

    pieces: dict[str, list[pd.Series]] = {}
    for ids in _chunks(list(dict.fromkeys(series_ids)), max_series):
        for y0 in range(first_year, last_year + 1, max_years):
            body = {"seriesid": ids, "startyear": str(y0), "endyear": str(min(y0 + max_years - 1, last_year))}
            if api_key:
                body["registrationkey"] = api_key
            resp = _request("POST", url, json=body)
            if resp.status_code != 200:
                raise FetchError(f"BLS: HTTP {resp.status_code}")
            for sid, s in parse_bls_payload(resp.json()).items():
                pieces.setdefault(sid, []).append(s)

    out = {}
    for sid, parts in pieces.items():
        s = pd.concat(parts).sort_index()
        out[sid] = s[~s.index.duplicated(keep="last")].rename(sid)
    return out


# --------------------------------------------------------------------------- #
# Disk cache
# --------------------------------------------------------------------------- #
def _cache_path(cache_dir: Path, source: str, series_id: str) -> Path:
    return Path(cache_dir) / f"{source}__{series_id}.csv"


def _read_cache(path: Path) -> tuple[pd.Series, datetime] | None:
    try:
        df = pd.read_csv(path)
        s = _to_series(df["date"], df["value"], path.stem.split("__", 1)[1])
        return s, datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc)
    except Exception:
        return None


def _write_cache(path: Path, s: pd.Series) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        s.rename("value").rename_axis("date").to_frame().to_csv(path)
    except OSError:
        pass  # a read-only filesystem only costs us the cache


def fetch_many(
    series_ids: list[str],
    source: str = "fred",
    *,
    fred_key: str | None = None,
    bls_key: str | None = None,
    force: bool = False,
    cache_dir: Path | str = DEFAULT_CACHE_DIR,
    ttl_hours: float = CACHE_TTL_HOURS,
    max_workers: int = 4,
) -> tuple[dict[str, pd.Series], list[dict]]:
    """Fetch a batch of series from one source, falling back to cache on failure.

    Returns ``(data, meta)``. ``meta`` has one record per requested series with a
    status of ``live`` (pulled just now), ``cached`` (pulled within the TTL),
    ``stale`` (the source failed, an older copy was used) or ``failed``.
    """
    cache_dir = Path(cache_dir)
    now = datetime.now(timezone.utc)
    ids = list(dict.fromkeys(series_ids))
    data: dict[str, pd.Series] = {}
    meta: dict[str, dict] = {}
    cached = {sid: _read_cache(_cache_path(cache_dir, source, sid)) for sid in ids}

    def record(sid, s, status, fetched_at, error=""):
        if s is not None:
            data[sid] = s
        valid = s.dropna() if s is not None else pd.Series(dtype=float)
        meta[sid] = {
            "series_id": sid,
            "source": source.upper(),
            "status": status,
            "last_observation": valid.index.max() if len(valid) else pd.NaT,
            "observations": int(len(valid)),
            "fetched_at": fetched_at,
            "error": error,
        }

    todo = []
    for sid in ids:
        hit = cached[sid]
        if hit and not force and (now - hit[1]).total_seconds() < ttl_hours * 3600:
            record(sid, hit[0], "cached", hit[1])
        else:
            todo.append(sid)

    fresh: dict[str, pd.Series] = {}
    errors: dict[str, str] = {}
    if todo and source == "fred":
        def one(sid):
            try:
                return sid, fetch_fred(sid, EARLIEST, fred_key), ""
            except Exception as exc:  # noqa: BLE001 - reported per series
                return sid, None, str(exc)

        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            for sid, s, err in pool.map(one, todo):
                if s is not None:
                    fresh[sid] = s
                else:
                    errors[sid] = err
    elif todo and source == "bls":
        try:
            fresh = fetch_bls(todo, EARLIEST, bls_key)
        except Exception as exc:  # noqa: BLE001
            errors = {sid: str(exc) for sid in todo}
        for sid in todo:
            if sid not in fresh:
                errors.setdefault(sid, "series not returned by the BLS API")
    elif todo:
        raise ValueError(f"unknown source: {source!r}")

    for sid in todo:
        if sid in fresh:
            _write_cache(_cache_path(cache_dir, source, sid), fresh[sid])
            record(sid, fresh[sid], "live", now)
        elif cached[sid]:
            record(sid, cached[sid][0], "stale", cached[sid][1], errors.get(sid, ""))
        else:
            record(sid, None, "failed", pd.NaT, errors.get(sid, ""))

    return data, [meta[sid] for sid in ids]
