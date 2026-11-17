import numpy as np
import pandas as pd
import pytest

from labordash import elasticity as el
from labordash.config import INDUSTRIES
from labordash.pipeline import build_model_frame
from tests.simulate import simulate_levels

UNITS = [u for u in INDUSTRIES if not u.aggregate][:6]


def frame_from(levels, horizon, units=None):
    units = [u for u in UNITS if u.emp_fred in levels] if units is None else units
    macro = pd.DataFrame({"cpi": levels["CPIAUCSL"], "pce": levels["PCEC96"]})
    panel = pd.concat(
        [pd.DataFrame({"date": levels[u.emp_fred].index, "unit": u.key, "label": u.label,
                       "emp": levels[u.emp_fred].to_numpy(), "wage": levels[u.wage_fred].to_numpy()}) for u in units],
        ignore_index=True,
    )
    return build_model_frame(macro, panel, horizon, "pce")


@pytest.mark.parametrize("horizon", [1, 3, 12])
def test_recovers_planted_elasticities(horizon):
    frame = frame_from(simulate_levels(UNITS, eta=-0.4, gamma=0.8, seed=1), horizon)
    res = el.estimate_all(frame, horizon, exclude_pandemic=False)
    assert len(res) == len(UNITS)
    tol = {1: 0.05, 3: 0.08, 12: 0.15}[horizon]  # overlapping differences carry less information
    assert np.allclose(res["wage_elast"], -0.4, atol=tol)
    assert np.allclose(res["demand_elast"], 0.8, atol=tol)


def test_unbiased_and_reasonably_covered_across_draws():
    """Monte Carlo at the hardest setting (12-month overlapping differences)."""
    est, covered = [], []
    for seed in range(40):
        res = el.estimate_all(frame_from(simulate_levels(UNITS[:3], seed=100 + seed), 12), 12, exclude_pandemic=False)
        est += list(res["wage_elast"])
        covered += list((res["wage_lo"] < -0.4) & (res["wage_hi"] > -0.4))
    assert np.mean(est) == pytest.approx(-0.4, abs=0.015)
    # Newey-West intervals are known to under-cover with overlapping data; they
    # should still be in the neighbourhood of 95%, far above naive OLS intervals.
    assert np.mean(covered) > 0.75


def test_hac_with_zero_lags_is_white():
    rng = np.random.default_rng(0)
    X = np.column_stack([np.ones(200), rng.standard_normal((200, 2))])
    u = rng.standard_normal(200) * (1 + np.abs(X[:, 1]))
    bread = np.linalg.inv(X.T @ X)
    white = bread @ (X.T @ (X * (u**2)[:, None])) @ bread
    assert np.allclose(el.hac_cov(X, u, 0), white)


def test_hac_widens_intervals_for_overlapping_differences():
    frame = frame_from(simulate_levels(UNITS[:1], seed=2), 12)
    df = el.prepare(frame, 12, False)
    y, X = el._design(df, False)
    assert el.ols_hac(y, X, el.nw_lags(len(y), 12)).se[1] > el.ols_hac(y, X, 0).se[1]


def test_pandemic_mask_scales_with_horizon():
    dates = pd.date_range("2019-01-01", "2023-12-01", freq="MS")
    m1, m12 = el.pandemic_mask(dates, 1), el.pandemic_mask(dates, 12)
    assert dates[m1].min() == pd.Timestamp("2020-03-01") and dates[m1].max() == pd.Timestamp("2021-06-01")
    assert dates[m12].max() == pd.Timestamp("2022-05-01")
    frame = frame_from(simulate_levels(UNITS[:1], seed=3), 12)
    kept = el.prepare(frame, 12, True)
    assert not el.pandemic_mask(kept["date"], 12).any() and len(kept) < len(frame)


def test_rolling_tracks_a_structural_break():
    levels = simulate_levels(UNITS[:1], eta=-0.2, eta_after=-0.9, break_at="2016-01-01", seed=4)
    frame = frame_from(levels, 1, UNITS[:1])
    roll = el.rolling_unit(frame, 1, window=60, exclude_pandemic=False)
    early = roll.loc[roll["date"] <= "2015-12-01", "wage_elast"].mean()
    late = roll.loc[roll["date"] >= "2021-01-01", "wage_elast"].mean()
    assert abs(early + 0.2) < 0.08 and abs(late + 0.9) < 0.08
    assert roll["date"].is_monotonic_increasing and roll["date"].max() == frame["date"].max()
    # the first estimate waits for a full window, so longer windows start later
    assert roll["date"].min() == frame["date"].min() + pd.DateOffset(months=59)
    longer = el.rolling_unit(frame, 1, window=120, exclude_pandemic=False)
    assert longer["date"].min() == frame["date"].min() + pd.DateOffset(months=119)


def test_rolling_skips_thin_windows():
    frame = frame_from(simulate_levels(UNITS[:1], seed=5), 12)
    roll = el.rolling_unit(frame, 12, window=36, exclude_pandemic=True)
    # windows that are mostly pandemic months are dropped rather than estimated on scraps
    assert not roll["date"].between("2020-06-01", "2022-05-01").any()
    assert len(roll) > 50


def test_partial_regression_slope_equals_wage_elasticity():
    frame = frame_from(simulate_levels(UNITS[:1], seed=6), 3)
    res = el.estimate_unit(frame, 3, exclude_pandemic=False)
    pr = el.partial_regression(frame, 3, exclude_pandemic=False)
    slope = (pr["wage_resid"] @ pr["emp_resid"]) / (pr["wage_resid"] @ pr["wage_resid"])
    assert slope == pytest.approx(res["wage_elast"], rel=1e-9)


def test_month_dummies_remove_planted_seasonality():
    levels = simulate_levels(UNITS[:1], eta=-0.4, seed=7)
    u = UNITS[0]
    # seasonal swings in BOTH series, positively correlated -> biases eta upward
    season = 0.03 * np.sin(2 * np.pi * levels[u.emp_fred].index.month / 12)
    levels[u.emp_fred] = levels[u.emp_fred] * np.exp(season)
    levels[u.wage_fred] = levels[u.wage_fred] * np.exp(season)
    frame = frame_from(levels, 1, UNITS[:1])
    naive = el.estimate_unit(frame, 1, exclude_pandemic=False, month_dummies=False)
    fixed = el.estimate_unit(frame, 1, exclude_pandemic=False, month_dummies=True)
    assert naive["wage_elast"] > 0.2
    assert fixed["wage_elast"] == pytest.approx(-0.4, abs=0.06)


def test_pooled_twfe_recovers_common_elasticity():
    frame = frame_from(simulate_levels(UNITS, eta=-0.4, seed=8), 12)
    res = el.pooled_twfe(frame, 12, exclude_pandemic=False)
    assert res["units"] == len(UNITS)
    assert res["wage_elast"] == pytest.approx(-0.4, abs=0.06)
    assert res["wage_lo"] < -0.4 < res["wage_hi"]


def test_too_little_data_returns_none():
    frame = frame_from(simulate_levels(UNITS[:2], periods=30, seed=9), 12)
    assert el.estimate_unit(frame[frame["unit"] == UNITS[0].key], 12, exclude_pandemic=False) is None
    assert el.pooled_twfe(frame, 12) is None
