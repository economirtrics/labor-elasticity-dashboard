import numpy as np
import pandas as pd
import pytest

from labordash import elasticity as el
from labordash import sources
from labordash.config import INDUSTRIES, MACRO, STATES, catalog
from labordash.pipeline import build_model_frame, deflator, load_data, macro_indicators
from tests.simulate import FakeResponse, fake_request_factory, simulate_levels

UNITS = [u for u in INDUSTRIES if not u.aggregate][:4]
KEYS = [u.key for u in UNITS]


@pytest.fixture
def levels():
    return simulate_levels(UNITS, eta=-0.4, gamma=0.8, seed=11)


@pytest.fixture(autouse=True)
def no_sleep(monkeypatch):
    monkeypatch.setattr(sources.time, "sleep", lambda s: None)


def test_catalog_ids_are_well_formed():
    assert len(INDUSTRIES) == 11 and len(STATES) == 51
    il = next(u for u in STATES if u.key == "IL")
    assert (il.emp_fred, il.wage_fred) == ("SMU17000000500000001", "SMU17000000500000003")
    assert not il.seasonally_adjusted
    lh = next(u for u in INDUSTRIES if u.key == "leisure")
    assert (lh.emp_fred, lh.wage_fred, lh.emp_bls) == ("USLAH", "CES7000000003", "CES7000000001")
    all_ids = [i for u in catalog("industries") + catalog("states") for i in u.ids("fred")]
    assert len(all_ids) == len(set(all_ids))


@pytest.mark.parametrize("source,key", [("fred", None), ("fred", "KEY"), ("bls", None), ("bls", "KEY")])
def test_end_to_end_recovers_elasticity_through_each_api(monkeypatch, tmp_path, levels, source, key):
    calls = []
    monkeypatch.setattr(sources.requests, "request", fake_request_factory(levels, calls))
    b = load_data("industries", KEYS, source, fred_key=key, bls_key=key, cache_dir=tmp_path)
    assert set(b.meta["status"]) == {"live"}
    assert [u.key for u in b.units] == KEYS
    assert list(b.macro.columns) == list(MACRO)
    if source == "fred":
        assert all(("api.stlouisfed.org" in c) == bool(key) for c in calls)
    else:  # BLS for labor series, FRED for the two demand proxies
        assert sum("api.bls.gov" in c for c in calls) >= 2 and sum("stlouisfed" in c for c in calls) == 2
    frame = build_model_frame(b.macro, b.panel, 12, "pce")
    res = el.estimate_all(frame, 12, exclude_pandemic=False)
    assert np.allclose(res["wage_elast"], -0.4, atol=0.15)
    ind = macro_indicators(b.macro)
    assert ind[["unrate", "inflation", "payroll_growth", "real_wage_growth"]].iloc[-1].notna().all()


def test_cache_serves_second_load_and_force_refetches(monkeypatch, tmp_path, levels):
    calls = []
    monkeypatch.setattr(sources.requests, "request", fake_request_factory(levels, calls))
    load_data("industries", KEYS, "fred", cache_dir=tmp_path)
    n = len(calls)
    b = load_data("industries", KEYS, "fred", cache_dir=tmp_path)
    assert len(calls) == n and set(b.meta["status"]) == {"cached"}
    b = load_data("industries", KEYS, "fred", cache_dir=tmp_path, force=True)
    assert len(calls) == 2 * n and set(b.meta["status"]) == {"live"}


def test_outage_falls_back_to_stale_cache(monkeypatch, tmp_path, levels):
    monkeypatch.setattr(sources.requests, "request", fake_request_factory(levels))
    first = load_data("industries", KEYS, "fred", cache_dir=tmp_path)
    monkeypatch.setattr(sources.requests, "request", lambda *a, **k: FakeResponse(503, text="down"))
    b = load_data("industries", KEYS, "fred", cache_dir=tmp_path, force=True)
    assert set(b.meta["status"]) == {"stale"} and b.meta["error"].str.contains("503").all()
    pd.testing.assert_frame_equal(b.panel, first.panel)


def test_bad_series_is_reported_and_unit_dropped(monkeypatch, tmp_path, levels):
    levels.pop(UNITS[0].wage_fred)
    for key in (None, "KEY"):
        monkeypatch.setattr(sources.requests, "request", fake_request_factory(levels))
        b = load_data("industries", KEYS, "fred", fred_key=key, cache_dir=tmp_path / str(key))
        failed = b.meta[b.meta["status"] == "failed"]
        assert list(failed["series_id"]) == [UNITS[0].wage_fred] and failed["error"].iloc[0]
        assert [u.key for u in b.units] == KEYS[1:]


def test_missing_values_and_gaps(monkeypatch, tmp_path, levels):
    # FRED "." / BLS "-" missing markers, and a hole in CPI like October 2025
    gap = pd.Timestamp("2018-10-01")
    levels["CPIAUCSL"].loc[gap] = np.nan
    levels[UNITS[0].emp_fred].loc[pd.Timestamp("2015-05-01")] = np.nan
    for i, source in enumerate(["fred", "bls"]):
        monkeypatch.setattr(sources.requests, "request", fake_request_factory(levels))
        b = load_data("industries", KEYS, source, cache_dir=tmp_path / source)
        assert np.isnan(b.macro.loc[gap, "cpi"])
        assert deflator(b.macro).notna().all()  # short CPI gap bridged
        frame = build_model_frame(b.macro, b.panel, 12, "pce")
        u0 = frame[frame["unit"] == UNITS[0].key]
        # the missing employment month drops itself and the month 12 later, nothing shifts
        assert not u0["date"].isin([pd.Timestamp("2015-05-01"), pd.Timestamp("2016-05-01")]).any()
        assert frame[frame["unit"] == UNITS[1].key]["date"].isin([gap]).any()


def test_bls_parser_skips_annual_rows_and_flags_errors():
    payload = {"status": "REQUEST_SUCCEEDED", "message": [], "Results": {"series": [{"seriesID": "X", "data": [
        {"year": "2024", "period": "M13", "value": "9"},
        {"year": "2024", "period": "M02", "value": "2.5"},
        {"year": "2024", "period": "M01", "value": "-"},
    ]}]}}
    s = sources.parse_bls_payload(payload)["X"]
    assert list(s.index) == [pd.Timestamp("2024-01-01"), pd.Timestamp("2024-02-01")]
    assert np.isnan(s.iloc[0]) and s.iloc[1] == 2.5
    with pytest.raises(sources.FetchError, match="daily threshold"):
        sources.parse_bls_payload({"status": "REQUEST_NOT_PROCESSED", "message": ["daily threshold reached"]})


def test_bls_requests_respect_api_limits(monkeypatch, levels):
    bodies = []
    fake = fake_request_factory(levels)

    def spy(method, url, **kw):
        bodies.append(kw["json"])
        return fake(method, url, **kw)

    monkeypatch.setattr(sources.requests, "request", spy)
    ids = [u.emp_bls for u in INDUSTRIES] + [u.wage_bls for u in INDUSTRIES] + [f"PAD{i}" for i in range(10)]
    sources.fetch_bls(ids, "2006-01-01", api_key=None)
    assert all(len(b["seriesid"]) <= 25 and int(b["endyear"]) - int(b["startyear"]) <= 9 for b in bodies)
    assert all("registrationkey" not in b for b in bodies)
    bodies.clear()
    sources.fetch_bls(ids, "2006-01-01", api_key="K")
    assert all(len(b["seriesid"]) <= 50 and int(b["endyear"]) - int(b["startyear"]) <= 19 for b in bodies)
    assert all(b["registrationkey"] == "K" for b in bodies)


def test_refresh_cli_writes_snapshots(monkeypatch, tmp_path, levels):
    from labordash import pipeline, refresh
    full = simulate_levels(INDUSTRIES, seed=12)
    monkeypatch.setattr(sources.requests, "request", fake_request_factory(full))
    monkeypatch.setattr(refresh, "load_data", lambda *a, **k: pipeline.load_data(*a, **{**k, "cache_dir": tmp_path / "c"}))
    assert refresh.main(["--out", str(tmp_path / "snap"), "--horizon", "3"]) == 0
    files = sorted(p.name for p in (tmp_path / "snap").iterdir())
    assert len(files) == 2 and files[0].startswith("elasticities_industries_h3_") and files[1].startswith("rolling_")
    est = pd.read_csv(tmp_path / "snap" / files[0])
    assert len(est) == len(INDUSTRIES) and est["wage_elast"].between(-0.6, -0.2).all()
